"""Transport-neutral contracts and Server-Sent Events framing primitives.

Provider clients own URLs, headers, payload serialization, response parsing,
and provider-specific error mapping. Concrete transports own HTTP resource
lifecycle and translate their library's failures through the callback hooks
defined here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
from types import MappingProxyType
from typing import Any, AsyncIterator, Callable, Iterator, Mapping, Optional

from ..errors.llm_api_error import (
    LLMAPIClientError,
    LLMAPIAuthorizationError,
    LLMAPIRateLimitError,
    LLMAPIServerError,
    LLMAPITimeoutError,
)


@dataclass(frozen=True)
class SSEEvent:
    """A decoded Server-Sent Event independent of an HTTP implementation."""

    event: Optional[str]
    data: Any = None
    done: bool = False


@dataclass(frozen=True)
class TransportRequest:
    """Provider-owned request data passed to a concrete transport.

    ``headers`` are copied into an immutable mapping so a transport cannot
    mutate data retained by the provider client. The concrete transport must
    create its own mutable copy when handing headers to an HTTP library.
    """

    url: str
    headers: Mapping[str, str] = field(default_factory=dict)
    payload: Any = None
    timeout: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "headers",
            MappingProxyType(dict(self.headers)),
        )

    def headers_dict(self) -> dict[str, str]:
        """Return a mutable copy suitable for a concrete HTTP client."""
        return dict(self.headers)


@dataclass(frozen=True)
class JSONResponse:
    """Decoded JSON with the minimal response accessor used by sync clients.

    Concrete transports decode the body before closing their native response.
    Keeping the ``.json()`` accessor preserves the existing provider-client
    boundary while making the native HTTP response lifetime transport-owned.
    """

    payload: Any

    def json(self) -> Any:
        """Return the already decoded JSON payload."""
        return self.payload


HTTPErrorHandler = Callable[[Any], Any]
StreamErrorHandler = Callable[[SSEEvent], Any]
SYNC_TRANSPORTS = ("requests", "httpx")


def validate_sync_transport(transport: object) -> str:
    """Validate a public synchronous transport selection."""
    if not isinstance(transport, str) or transport not in SYNC_TRANSPORTS:
        raise ValueError("transport must be either 'requests' or 'httpx'")
    return transport


def create_sync_transport(transport: object) -> "SyncTransport":
    """Create the selected synchronous transport without importing HTTPX eagerly."""
    selected = validate_sync_transport(transport)
    if selected == "requests":
        from .requests_transport import RequestsSyncTransport

        return RequestsSyncTransport()

    from .httpx_transport import HttpxSyncTransport

    return HttpxSyncTransport()


class SyncTransport(ABC):
    """Internal contract for a synchronous JSON and SSE HTTP transport."""

    @abstractmethod
    def post_json(
        self,
        request: TransportRequest,
        *,
        http_error_handler: Optional[HTTPErrorHandler] = None,
    ) -> JSONResponse:
        """POST JSON and return an accessor for the decoded response body.

        The implementation closes any HTTP response on success and failure.
        Provider handlers receive their native HTTP status exception so they
        retain ownership of provider-specific error classification.
        """

    @abstractmethod
    def post_sse(
        self,
        request: TransportRequest,
        *,
        http_error_handler: Optional[HTTPErrorHandler] = None,
        stream_error_handler: Optional[StreamErrorHandler] = None,
    ) -> Iterator[SSEEvent]:
        """POST JSON and yield framed SSE events until completion or close."""


class AsyncTransport(ABC):
    """Internal contract for an asynchronous JSON and SSE HTTP transport."""

    @abstractmethod
    async def post_json(
        self,
        request: TransportRequest,
        *,
        http_error_handler: Optional[HTTPErrorHandler] = None,
    ) -> Any:
        """POST JSON asynchronously and close resources on every exit path."""

    @abstractmethod
    def post_sse(
        self,
        request: TransportRequest,
        *,
        http_error_handler: Optional[HTTPErrorHandler] = None,
        stream_error_handler: Optional[StreamErrorHandler] = None,
    ) -> AsyncIterator[SSEEvent]:
        """POST JSON and asynchronously yield framed SSE events."""


class SSEFrameDecoder:
    """Incrementally decode SSE framing without depending on an HTTP library."""

    def __init__(self) -> None:
        self._event_name: Optional[str] = None
        self._data_lines: list[str] = []

    def feed(self, raw_line: bytes | str) -> Optional[SSEEvent]:
        """Consume one SSE line and return a completed event, when available."""
        line = decode_sse_line(raw_line).rstrip("\r\n")

        if line == "":
            return self._take_event()
        if line.startswith(":"):
            return None

        field, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]

        if field == "event":
            self._event_name = value
        elif field == "data":
            self._data_lines.append(value)
        return None

    def finish(self) -> Optional[SSEEvent]:
        """Return a final event when a stream ends without a blank delimiter."""
        return self._take_event()

    def _take_event(self) -> Optional[SSEEvent]:
        event = build_sse_event(self._event_name, self._data_lines)
        self._event_name = None
        self._data_lines = []
        return event


def decode_sse_line(line: bytes | str) -> str:
    """Normalize a line supplied by either a sync or async HTTP response."""
    if isinstance(line, bytes):
        return line.decode("utf-8")
    return str(line)


def build_sse_event(
    event_name: Optional[str],
    data_lines: list[str],
) -> Optional[SSEEvent]:
    """Build one neutral event from accumulated SSE fields."""
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
        raise LLMAPIClientError(detail=f"Malformed SSE JSON data: {exc}") from exc
    return SSEEvent(event=event_name, data=decoded_data)


def is_generic_stream_error(event: SSEEvent) -> bool:
    """Return whether an event uses the provider-independent error shape."""
    if event.event and event.event.lower() == "error":
        return True
    return isinstance(event.data, Mapping) and event.data.get("type") == "error"


def raise_default_stream_error(event: SSEEvent) -> None:
    """Map a generic SSE error event to the public error hierarchy."""
    raise LLMAPIClientError(detail=stream_error_detail(event))


def stream_error_detail(event: SSEEvent) -> str:
    """Extract a useful error message from a neutral SSE event."""
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


def raise_default_http_error(*, status_code: Optional[int], detail: str) -> None:
    """Map an HTTP status failure when a provider has no custom handler."""
    if status_code in (401, 403):
        raise LLMAPIAuthorizationError(detail=detail)
    if status_code == 429:
        raise LLMAPIRateLimitError(detail=detail)
    if status_code in (408, 504):
        raise LLMAPITimeoutError(detail=detail)
    if status_code is not None and 500 <= status_code < 600:
        raise LLMAPIServerError(detail=detail)
    raise LLMAPIClientError(detail=detail)


__all__ = [
    "AsyncTransport",
    "HTTPErrorHandler",
    "JSONResponse",
    "SYNC_TRANSPORTS",
    "SSEEvent",
    "SSEFrameDecoder",
    "StreamErrorHandler",
    "SyncTransport",
    "TransportRequest",
    "build_sse_event",
    "create_sync_transport",
    "decode_sse_line",
    "is_generic_stream_error",
    "raise_default_http_error",
    "raise_default_stream_error",
    "stream_error_detail",
    "validate_sync_transport",
]
