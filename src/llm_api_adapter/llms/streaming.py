"""Shared synchronous SSE transport for provider streaming clients.

This module deliberately knows only about the Server-Sent Events transport.
Provider clients remain responsible for interpreting the decoded payloads.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import time
from typing import Any, Callable, Iterator, Mapping, Optional

import requests

from ..errors.llm_api_error import (
    LLMAPIClientError,
    LLMAPIError,
    LLMAPIAuthorizationError,
    LLMAPIRateLimitError,
    LLMAPIServerError,
    LLMAPITimeoutError,
)
from ..models.responses.chat_response import Usage
from ..models.responses.stream_chunk import StreamChunk

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SSEEvent:
    """Decoded transport event produced by :func:`iter_sse_events`.

    ``event`` is the optional SSE event name and ``data`` is the decoded JSON
    value.  ``done`` marks the provider-independent ``data: [DONE]`` sentinel.
    This is an internal transport representation, not the public stream API.
    """

    event: Optional[str]
    data: Any = None
    done: bool = False


HTTPErrorHandler = Callable[[requests.exceptions.HTTPError], Any]
StreamErrorHandler = Callable[[SSEEvent], Any]
Clock = Callable[[], float]


class StreamChunkBuffer:
    """Buffer visible text and emit :class:`StreamChunk` values.

    With ``buffer_chars=None`` each non-empty input is emitted immediately.
    A positive ``buffer_chars`` coalesces text until full chunks can be
    emitted; :meth:`flush` emits the remaining text.  The class deliberately
    has no provider-specific parsing or background work.
    """

    def __init__(
        self,
        buffer_chars: Optional[int] = None,
        *,
        clock: Clock = time.perf_counter,
    ) -> None:
        if (
            buffer_chars is not None
            and (
                isinstance(buffer_chars, bool)
                or not isinstance(buffer_chars, int)
                or buffer_chars <= 0
            )
        ):
            raise ValueError("buffer_chars must be None or a positive integer")

        self._buffer_chars = buffer_chars
        self._clock = clock
        self._started_at = clock()
        self._last_emitted_at = self._started_at
        self._next_index = 0
        self._pending_text = ""
        self._usage: Optional[Usage] = None
        self._output_tokens_delta: Optional[int] = None

    def add(
        self,
        text: str,
        *,
        usage: Optional[Usage] = None,
        output_tokens_delta: Optional[int] = None,
    ) -> Iterator[StreamChunk]:
        """Accept provider-normalized text and yield every completed chunk."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        self._update_metadata(usage, output_tokens_delta)
        if not text:
            return

        if self._buffer_chars is None:
            yield self._build_chunk(text, self._clock())
            return

        self._pending_text += text
        if len(self._pending_text) < self._buffer_chars:
            return

        emitted_at = self._clock()
        while len(self._pending_text) >= self._buffer_chars:
            chunk_text = self._pending_text[: self._buffer_chars]
            self._pending_text = self._pending_text[self._buffer_chars :]
            yield self._build_chunk(chunk_text, emitted_at)

    def update_metadata(
        self,
        *,
        usage: Optional[Usage] = None,
        output_tokens_delta: Optional[int] = None,
    ) -> None:
        """Store metadata for the next emitted chunk without adding text."""
        self._update_metadata(usage, output_tokens_delta)

    def flush(self) -> Iterator[StreamChunk]:
        """Emit pending text, if any, at normal stream completion."""
        if not self._pending_text:
            return

        pending_text = self._pending_text
        self._pending_text = ""
        yield self._build_chunk(pending_text, self._clock())

    def _update_metadata(
        self,
        usage: Optional[Usage],
        output_tokens_delta: Optional[int],
    ) -> None:
        if usage is not None:
            self._usage = Usage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
            )
        if output_tokens_delta is not None:
            self._output_tokens_delta = output_tokens_delta

    def _build_chunk(self, text: str, emitted_at: float) -> StreamChunk:
        chunk = StreamChunk(
            text=text,
            index=self._next_index,
            elapsed_s=emitted_at - self._started_at,
            delta_s=emitted_at - self._last_emitted_at,
            usage=self._usage,
            output_tokens_delta=self._output_tokens_delta,
        )
        self._next_index += 1
        self._last_emitted_at = emitted_at
        self._output_tokens_delta = None
        return chunk


class StreamUsageTracker:
    """Attach cumulative provider usage snapshots to future stream chunks."""

    def __init__(self) -> None:
        self._last_output_tokens: Optional[int] = None

    def record(
        self,
        buffer: StreamChunkBuffer,
        usage: Optional[Usage],
    ) -> None:
        """Store ``usage`` and its output-token increment for the next chunk."""
        if usage is None:
            return

        previous_output_tokens = self._last_output_tokens
        output_tokens_delta = (
            usage.output_tokens
            if previous_output_tokens is None
            else max(0, usage.output_tokens - previous_output_tokens)
        )
        self._last_output_tokens = (
            usage.output_tokens
            if previous_output_tokens is None
            else max(previous_output_tokens, usage.output_tokens)
        )
        buffer.update_metadata(
            usage=usage,
            output_tokens_delta=output_tokens_delta,
        )


def _decode_line(line: bytes | str) -> str:
    if isinstance(line, bytes):
        return line.decode("utf-8")
    return str(line)


def _build_event(event_name: Optional[str], data_lines: list[str]) -> Optional[SSEEvent]:
    if not data_lines:
        return None

    if event_name and event_name.lower() in {"ping", "keep-alive", "keep_alive"}:
        return None

    raw_data = "\n".join(data_lines)
    if raw_data.strip() == "[DONE]":
        return SSEEvent(event=event_name, done=True)

    try:
        decoded_data = json.loads(raw_data)
    except json.JSONDecodeError as exc:
        raise LLMAPIClientError(
            detail=f"Malformed SSE JSON data: {exc}"
        ) from exc

    return SSEEvent(event=event_name, data=decoded_data)


def iter_sse_events(response: requests.Response) -> Iterator[SSEEvent]:
    """Yield decoded SSE events and close ``response`` on every exit path.

    The parser handles only SSE framing.  It ignores comments and transport
    keep-alives, joins multiple ``data:`` lines with a newline, decodes JSON,
    and recognizes ``[DONE]``.  Unknown fields and event names are preserved
    or ignored without provider-specific interpretation.
    """

    event_name: Optional[str] = None
    data_lines: list[str] = []

    try:
        try:
            lines = response.iter_lines(decode_unicode=True)
        except TypeError:
            # Small test doubles and compatible response implementations may
            # expose iter_lines() without the requests keyword argument.
            lines = response.iter_lines()

        for raw_line in lines:
            line = _decode_line(raw_line).rstrip("\r\n")

            if line == "":
                event = _build_event(event_name, data_lines)
                event_name = None
                data_lines = []
                if event is not None:
                    yield event
                    if event.done:
                        return
                continue

            if line.startswith(":"):
                continue

            field, separator, value = line.partition(":")
            if separator and value.startswith(" "):
                value = value[1:]

            if field == "event":
                event_name = value
            elif field == "data":
                data_lines.append(value)

        event = _build_event(event_name, data_lines)
        if event is not None:
            yield event
    finally:
        response.close()


def _default_http_error_handler(http_err: requests.exceptions.HTTPError) -> None:
    response = getattr(http_err, "response", None)
    status_code = getattr(response, "status_code", None)
    detail = str(http_err)

    if status_code in (401, 403):
        raise LLMAPIAuthorizationError(detail=detail)
    if status_code == 429:
        raise LLMAPIRateLimitError(detail=detail)
    if status_code in (408, 504):
        raise LLMAPITimeoutError(detail=detail)
    if status_code is not None and 500 <= status_code < 600:
        raise LLMAPIServerError(detail=detail)
    raise LLMAPIClientError(detail=detail)


def _stream_error_detail(event: SSEEvent) -> str:
    payload = event.data
    if isinstance(payload, Mapping):
        error = payload.get("error")
        if isinstance(error, Mapping):
            code = error.get("type") or error.get("code")
            message = error.get("message")
            if code and message:
                return f"{code}: {message}"
            if message:
                return str(message)
            if code:
                return str(code)
        message = payload.get("message")
        if message:
            return str(message)
        code = payload.get("code") or payload.get("type")
        if code:
            return str(code)
    return str(payload)


def _default_stream_error_handler(event: SSEEvent) -> None:
    raise LLMAPIClientError(detail=_stream_error_detail(event))


def _is_generic_stream_error(event: SSEEvent) -> bool:
    if event.event and event.event.lower() == "error":
        return True
    return isinstance(event.data, Mapping) and event.data.get("type") == "error"


def stream_request(
    url: str,
    *,
    headers: Optional[Mapping[str, str]] = None,
    payload: Any = None,
    timeout: Optional[float] = None,
    http_error_handler: Optional[HTTPErrorHandler] = None,
    stream_error_handler: Optional[StreamErrorHandler] = None,
) -> Iterator[SSEEvent]:
    """Perform a synchronous streaming POST and yield decoded SSE events.

    The request is made when the returned iterator is first consumed.  HTTP
    errors are mapped through the supplied provider handler when available;
    otherwise the shared status-code mapping is used.  Generic in-stream
    ``error`` events use the same unified error hierarchy.
    """

    response: Optional[requests.Response] = None
    parser_owns_response = False
    try:
        response = requests.post(
            url,
            headers=dict(headers or {}),
            json=payload,
            timeout=timeout,
            stream=True,
        )
        response.raise_for_status()

        parser_owns_response = True
        for event in iter_sse_events(response):
            if event.done:
                return
            if _is_generic_stream_error(event):
                handler = stream_error_handler or _default_stream_error_handler
                handler(event)
                _default_stream_error_handler(event)
            yield event
    except LLMAPIError:
        raise
    except requests.exceptions.Timeout as exc:
        logger.error("Streaming request timed out: %s", exc)
        raise LLMAPITimeoutError(detail=str(exc)) from exc
    except requests.exceptions.HTTPError as exc:
        logger.error("Streaming HTTP error: %s", exc)
        handler = http_error_handler or _default_http_error_handler
        handler(exc)
        _default_http_error_handler(exc)
    except requests.exceptions.RequestException as exc:
        logger.error("Streaming request exception: %s", exc)
        raise LLMAPIClientError(detail=str(exc)) from exc
    finally:
        if response is not None and not parser_owns_response:
            response.close()
