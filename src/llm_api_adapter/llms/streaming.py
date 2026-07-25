"""Shared synchronous SSE transport for provider streaming clients.

This module deliberately knows only about the Server-Sent Events transport.
Provider clients remain responsible for interpreting the decoded payloads.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
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
