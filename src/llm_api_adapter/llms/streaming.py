"""Shared synchronous SSE transport for provider streaming clients.

This module deliberately knows only about the Server-Sent Events transport.
Provider clients remain responsible for interpreting the decoded payloads.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Iterator, List, Mapping, Optional

import requests

from ..models.responses.chat_response import Usage
from ..models.responses.reasoning_event import ReasoningEvent, ReasoningEventKind
from ..models.responses.stream_chunk import StreamChunk
from .transports import (
    HTTPErrorHandler,
    SSEEvent,
    SSEFrameDecoder,
    StreamErrorHandler,
    TransportRequest,
)


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


class StreamReasoningCollector:
    """Collect normalized reasoning fragments with deterministic timing."""

    def __init__(self, *, clock: Clock = time.perf_counter) -> None:
        self._clock = clock
        self._started_at = clock()
        self._last_emitted_at = self._started_at
        self._next_index = 0
        self._events: List[ReasoningEvent] = []

    def add(
        self,
        text: str,
        *,
        kind: ReasoningEventKind = "summary",
    ) -> Optional[ReasoningEvent]:
        """Record a non-empty reasoning fragment and return its event."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text:
            return None

        emitted_at = self._clock()
        event = ReasoningEvent(
            text=text,
            kind=kind,
            index=self._next_index,
            elapsed_s=emitted_at - self._started_at,
            delta_s=emitted_at - self._last_emitted_at,
        )
        self._events.append(event)
        self._next_index += 1
        self._last_emitted_at = emitted_at
        return event

    def snapshot(self) -> List[ReasoningEvent]:
        """Return a shallow copy of the collected immutable events."""
        return list(self._events)


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


def iter_sse_events(response: requests.Response) -> Iterator[SSEEvent]:
    """Yield decoded SSE events and close ``response`` on every exit path.

    The parser handles only SSE framing.  It ignores comments and transport
    keep-alives, joins multiple ``data:`` lines with a newline, decodes JSON,
    and recognizes ``[DONE]``.  Unknown fields and event names are preserved
    or ignored without provider-specific interpretation.
    """

    decoder = SSEFrameDecoder()

    try:
        try:
            lines = response.iter_lines(decode_unicode=True)
        except TypeError:
            # Small test doubles and compatible response implementations may
            # expose iter_lines() without the requests keyword argument.
            lines = response.iter_lines()

        for raw_line in lines:
            event = decoder.feed(raw_line)
            if event is not None:
                yield event
                if event.done:
                    return

        event = decoder.finish()
        if event is not None:
            yield event
    finally:
        response.close()


def stream_request(
    url: str,
    *,
    headers: Optional[Mapping[str, str]] = None,
    payload: Any = None,
    timeout: Optional[float] = None,
    http_error_handler: Optional[HTTPErrorHandler] = None,
    stream_error_handler: Optional[StreamErrorHandler] = None,
) -> Iterator[SSEEvent]:
    """Perform a synchronous streaming POST through the default transport.

    Kept as a compatibility wrapper while provider clients migrate to the
    :class:`RequestsSyncTransport` implementation directly.
    """
    from .requests_transport import RequestsSyncTransport

    return RequestsSyncTransport().post_sse(
        TransportRequest(
            url=url,
            headers=headers or {},
            payload=payload,
            timeout=timeout,
        ),
        http_error_handler=http_error_handler,
        stream_error_handler=stream_error_handler,
    )
