"""Opt-in synchronous HTTPX implementation of the internal transport API."""

from __future__ import annotations

import logging
from typing import Any, Iterator, Optional

from ..errors.llm_api_error import LLMAPIClientError, LLMAPIError, LLMAPITimeoutError
from .streaming import iter_sse_events
from .transports import (
    HTTPErrorHandler,
    JSONResponse,
    SSEEvent,
    StreamErrorHandler,
    SyncTransport,
    TransportRequest,
    is_generic_stream_error,
    raise_default_http_error,
    raise_default_stream_error,
)

logger = logging.getLogger(__name__)

HTTPX_SYNC_DEPENDENCY_MESSAGE = (
    "Synchronous HTTPX transport requires the optional 'httpx' dependency. "
    "Install it with: pip install 'llm-api-adapter[httpx]'."
)


def _require_httpx() -> Any:
    """Import HTTPX only when the opt-in synchronous transport is selected."""
    try:
        import httpx
    except ImportError as exc:
        raise ImportError(HTTPX_SYNC_DEPENDENCY_MESSAGE) from exc
    return httpx


class HttpxSyncTransport(SyncTransport):
    """Run synchronous JSON and SSE requests through HTTPX."""

    def __init__(self) -> None:
        self._httpx = _require_httpx()

    def post_json(
        self,
        request: TransportRequest,
        *,
        http_error_handler: Optional[HTTPErrorHandler] = None,
    ) -> JSONResponse:
        client = self._httpx.Client()
        response: Optional[Any] = None

        try:
            response = client.post(
                request.url,
                headers=request.headers_dict(),
                json=request.payload,
                timeout=request.timeout,
            )
            response.raise_for_status()
            return JSONResponse(response.json())
        except LLMAPIError:
            raise
        except self._httpx.TimeoutException as exc:
            logger.error("Synchronous HTTPX request timed out: %s", exc)
            raise LLMAPITimeoutError(detail=str(exc)) from exc
        except self._httpx.HTTPStatusError as exc:
            logger.error("Synchronous HTTPX HTTP error: %s", exc)
            self._handle_http_error(exc, http_error_handler)
        except self._httpx.RequestError as exc:
            logger.error("Synchronous HTTPX request exception: %s", exc)
            raise LLMAPIClientError(detail=str(exc)) from exc
        finally:
            try:
                if response is not None:
                    response.close()
            finally:
                client.close()

    def post_sse(
        self,
        request: TransportRequest,
        *,
        http_error_handler: Optional[HTTPErrorHandler] = None,
        stream_error_handler: Optional[StreamErrorHandler] = None,
    ) -> Iterator[SSEEvent]:
        client = self._httpx.Client()
        response: Optional[Any] = None
        parser_owns_response = False

        try:
            raw_request = client.build_request(
                "POST",
                request.url,
                headers=request.headers_dict(),
                json=request.payload,
                timeout=request.timeout,
            )
            response = client.send(raw_request, stream=True)
            response.raise_for_status()

            parser_owns_response = True
            for event in iter_sse_events(response):
                if event.done:
                    return
                if is_generic_stream_error(event):
                    handler = stream_error_handler or raise_default_stream_error
                    handler(event)
                    raise_default_stream_error(event)
                yield event
        except LLMAPIError:
            raise
        except self._httpx.TimeoutException as exc:
            logger.error("Synchronous HTTPX streaming request timed out: %s", exc)
            raise LLMAPITimeoutError(detail=str(exc)) from exc
        except self._httpx.HTTPStatusError as exc:
            logger.error("Synchronous HTTPX streaming HTTP error: %s", exc)
            self._handle_http_error(exc, http_error_handler)
        except self._httpx.RequestError as exc:
            logger.error("Synchronous HTTPX streaming request exception: %s", exc)
            raise LLMAPIClientError(detail=str(exc)) from exc
        finally:
            try:
                if response is not None and not parser_owns_response:
                    response.close()
            finally:
                client.close()

    @staticmethod
    def _handle_http_error(error: Any, handler: Optional[HTTPErrorHandler]) -> None:
        if handler is not None:
            handler(error)

        response = getattr(error, "response", None)
        raise_default_http_error(
            status_code=getattr(response, "status_code", None),
            detail=str(error),
        )


__all__ = ["HTTPX_SYNC_DEPENDENCY_MESSAGE", "HttpxSyncTransport"]
