"""Shared asynchronous HTTPX transport for provider clients.

This module deliberately knows only about JSON POST requests and Server-Sent
Events framing. Provider clients remain responsible for interpreting decoded
payloads and mapping provider-specific error bodies.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Mapping, Optional

from ..errors.llm_api_error import LLMAPIError, LLMAPIClientError, LLMAPITimeoutError
from .transports import (
    HTTPErrorHandler,
    MultipartForm,
    SSEEvent,
    SSEFrameDecoder,
    StreamErrorHandler,
    is_generic_stream_error as _is_generic_stream_error,
    multipart_headers,
    raise_default_http_error,
    raise_default_stream_error as _default_stream_error_handler,
)

logger = logging.getLogger(__name__)

ASYNC_DEPENDENCY_MESSAGE = (
    "Async HTTP transport requires the optional 'httpx' dependency. "
    "Install it with: pip install 'llm-api-adapter[async]'."
)

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
    raise_default_http_error(status_code=status_code, detail=str(http_error))


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


async def async_multipart_request(
    url: str,
    *,
    headers: Optional[Mapping[str, str]] = None,
    form: MultipartForm,
    timeout: Optional[float] = None,
    http_error_handler: Optional[HTTPErrorHandler] = None,
) -> Any:
    """Perform an asynchronous multipart POST and return decoded JSON.

    The shared multipart form keeps field and file ownership with the provider
    client while HTTPX owns boundary construction and resource cleanup.

    HTTPX builds a synchronous multipart stream for ``files=``. Materialize
    that stream before handing it to :class:`httpx.AsyncClient`; otherwise
    HTTPX rejects it as a synchronous request stream. The adapter already owns
    file bytes in memory, so this does not change the transport contract.
    """
    httpx = _require_httpx()
    client = httpx.AsyncClient()
    response: Optional[Any] = None

    try:
        multipart_parts = [(name, (None, value)) for name, value in form.fields]
        multipart_parts.extend(form.files_list())
        multipart_request = httpx.Request(
            "POST",
            url,
            headers=multipart_headers(headers or {}),
            files=multipart_parts,
        )
        content = multipart_request.read()
        request_headers = {
            name: value
            for name, value in multipart_request.headers.items()
            if name.lower() != "host"
        }
        response = await client.post(
            url,
            headers=request_headers,
            content=content,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()
    except LLMAPIError:
        raise
    except httpx.TimeoutException as exc:
        logger.error("Async multipart request timed out: %s", exc)
        raise LLMAPITimeoutError(detail=str(exc)) from exc
    except httpx.HTTPStatusError as exc:
        logger.error("Async multipart HTTP error: %s", exc)
        await _read_http_error_body(response)
        handler = http_error_handler or _default_http_error_handler
        handler(exc)
        _default_http_error_handler(exc)
    except httpx.RequestError as exc:
        logger.error("Async multipart request exception: %s", exc)
        raise LLMAPIClientError(detail=str(exc)) from exc
    finally:
        try:
            if response is not None:
                await response.aclose()
        finally:
            await client.aclose()


async def aiter_sse_events(response: Any) -> AsyncIterator[SSEEvent]:
    """Decode SSE framing from an HTTPX response and close it on exit."""
    decoder = SSEFrameDecoder()

    try:
        async for raw_line in response.aiter_lines():
            event = decoder.feed(raw_line)
            if event is not None:
                yield event
                if event.done:
                    return

        event = decoder.finish()
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
    "async_multipart_request",
    "async_request",
    "async_stream_request",
]
