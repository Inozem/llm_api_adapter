"""Shared asynchronous HTTPX transport for provider clients.

This module deliberately knows only about JSON POST requests and Server-Sent
Events framing. Provider clients remain responsible for interpreting decoded
payloads and mapping provider-specific error bodies.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Callable, Mapping, Optional

from ..errors.llm_api_error import (
    LLMAPIClientError,
    LLMAPIError,
    LLMAPIAuthorizationError,
    LLMAPIRateLimitError,
    LLMAPIServerError,
    LLMAPITimeoutError,
)
from .streaming import (
    SSEEvent,
    _build_event,
    _decode_line,
    _default_stream_error_handler,
    _is_generic_stream_error,
)

logger = logging.getLogger(__name__)

ASYNC_DEPENDENCY_MESSAGE = (
    "Async HTTP transport requires the optional 'httpx' dependency. "
    "Install it with: pip install 'llm-api-adapter[async]'."
)

HTTPErrorHandler = Callable[[Any], Any]
StreamErrorHandler = Callable[[SSEEvent], Any]


def _require_httpx() -> Any:
    """Import HTTPX only when an async transport operation is requested."""
    try:
        import httpx
    except ImportError as exc:
        raise ImportError(ASYNC_DEPENDENCY_MESSAGE) from exc
    return httpx


def _default_http_error_handler(http_error: Any) -> None:
    """Map HTTP status failures to the shared error hierarchy."""
    response = getattr(http_error, "response", None)
    status_code = getattr(response, "status_code", None)
    detail = str(http_error)

    if status_code in (401, 403):
        raise LLMAPIAuthorizationError(detail=detail)
    if status_code == 429:
        raise LLMAPIRateLimitError(detail=detail)
    if status_code in (408, 504):
        raise LLMAPITimeoutError(detail=detail)
    if status_code is not None and 500 <= status_code < 600:
        raise LLMAPIServerError(detail=detail)
    raise LLMAPIClientError(detail=detail)


async def _read_http_error_body(response: Optional[Any]) -> None:
    """Load a streamed error body before provider handlers inspect ``.json()``."""
    if response is None:
        return
    aread = getattr(response, "aread", None)
    if aread is not None:
        await aread()


async def async_request(
    url: str,
    *,
    headers: Optional[Mapping[str, str]] = None,
    payload: Any = None,
    timeout: Optional[float] = None,
    http_error_handler: Optional[HTTPErrorHandler] = None,
) -> Any:
    """Perform an asynchronous JSON POST and return its decoded JSON body."""
    httpx = _require_httpx()
    client = httpx.AsyncClient()
    response: Optional[Any] = None

    try:
        response = await client.post(
            url,
            headers=dict(headers or {}),
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()
    except LLMAPIError:
        raise
    except httpx.TimeoutException as exc:
        logger.error("Async request timed out: %s", exc)
        raise LLMAPITimeoutError(detail=str(exc)) from exc
    except httpx.HTTPStatusError as exc:
        logger.error("Async HTTP error: %s", exc)
        await _read_http_error_body(response)
        handler = http_error_handler or _default_http_error_handler
        handler(exc)
        _default_http_error_handler(exc)
    except httpx.RequestError as exc:
        logger.error("Async request exception: %s", exc)
        raise LLMAPIClientError(detail=str(exc)) from exc
    finally:
        try:
            if response is not None:
                await response.aclose()
        finally:
            await client.aclose()


async def aiter_sse_events(response: Any) -> AsyncIterator[SSEEvent]:
    """Decode SSE framing from an HTTPX response and close it on exit."""
    event_name: Optional[str] = None
    data_lines: list[str] = []

    try:
        async for raw_line in response.aiter_lines():
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
        await response.aclose()


async def async_stream_request(
    url: str,
    *,
    headers: Optional[Mapping[str, str]] = None,
    payload: Any = None,
    timeout: Optional[float] = None,
    http_error_handler: Optional[HTTPErrorHandler] = None,
    stream_error_handler: Optional[StreamErrorHandler] = None,
) -> AsyncIterator[SSEEvent]:
    """Perform an asynchronous streaming POST and yield decoded SSE events.

    The request is made when the returned async iterator is first consumed.
    HTTP errors are mapped through the supplied provider handler when
    available; otherwise the shared status-code mapping is used. Generic
    in-stream ``error`` events use the shared error hierarchy.
    """
    httpx = _require_httpx()
    client = httpx.AsyncClient()
    response: Optional[Any] = None
    parser: Optional[AsyncIterator[SSEEvent]] = None

    try:
        request = client.build_request(
            "POST",
            url,
            headers=dict(headers or {}),
            json=payload,
            timeout=timeout,
        )
        response = await client.send(request, stream=True)
        response.raise_for_status()
        parser = aiter_sse_events(response)

        async for event in parser:
            if event.done:
                return
            if _is_generic_stream_error(event):
                handler = stream_error_handler or _default_stream_error_handler
                handler(event)
                _default_stream_error_handler(event)
            yield event
    except LLMAPIError:
        raise
    except httpx.TimeoutException as exc:
        logger.error("Async streaming request timed out: %s", exc)
        raise LLMAPITimeoutError(detail=str(exc)) from exc
    except httpx.HTTPStatusError as exc:
        logger.error("Async streaming HTTP error: %s", exc)
        await _read_http_error_body(response)
        handler = http_error_handler or _default_http_error_handler
        handler(exc)
        _default_http_error_handler(exc)
    except httpx.RequestError as exc:
        logger.error("Async streaming request exception: %s", exc)
        raise LLMAPIClientError(detail=str(exc)) from exc
    finally:
        try:
            if parser is not None:
                await parser.aclose()
            elif response is not None:
                await response.aclose()
        finally:
            await client.aclose()


__all__ = [
    "ASYNC_DEPENDENCY_MESSAGE",
    "SSEEvent",
    "aiter_sse_events",
    "async_request",
    "async_stream_request",
]
