"""Synchronous :mod:`requests` implementation of the internal transport API."""

from __future__ import annotations

import logging
from typing import Iterator, Optional

import requests

from ..errors.llm_api_error import LLMAPIClientError, LLMAPIError, LLMAPITimeoutError
from .streaming import iter_sse_events
from .transports import (
    HTTPErrorHandler,
    JSONResponse,
    MultipartForm,
    SSEEvent,
    StreamErrorHandler,
    SyncTransport,
    TransportRequest,
    is_generic_stream_error,
    multipart_headers,
    raise_default_http_error,
    raise_default_stream_error,
)

logger = logging.getLogger(__name__)


class RequestsSyncTransport(SyncTransport):
    """Run JSON and SSE requests through ``requests``.

    The class deliberately contains no provider-specific URL, payload, or
    response parsing knowledge. Provider clients pass their error mappers as
    callbacks; this transport owns resource cleanup and generic request error
    conversion.
    """

    def post_json(
        self,
        request: TransportRequest,
        *,
        http_error_handler: Optional[HTTPErrorHandler] = None,
    ) -> JSONResponse:
        response: Optional[requests.Response] = None

        try:
            response = requests.post(
                request.url,
                headers=request.headers_dict(),
                json=request.payload,
                timeout=request.timeout,
            )
            response.raise_for_status()
            return JSONResponse(response.json())
        except LLMAPIError:
            raise
        except requests.exceptions.Timeout as exc:
            logger.error("Synchronous request timed out: %s", exc)
            raise LLMAPITimeoutError(detail=str(exc)) from exc
        except requests.exceptions.HTTPError as exc:
            logger.error("Synchronous HTTP error: %s", exc)
            self._handle_http_error(exc, http_error_handler)
        except requests.exceptions.RequestException as exc:
            logger.error("Synchronous request exception: %s", exc)
            raise LLMAPIClientError(detail=str(exc)) from exc
        finally:
            if response is not None:
                response.close()

    def post_sse(
        self,
        request: TransportRequest,
        *,
        http_error_handler: Optional[HTTPErrorHandler] = None,
        stream_error_handler: Optional[StreamErrorHandler] = None,
    ) -> Iterator[SSEEvent]:
        response: Optional[requests.Response] = None
        parser_owns_response = False

        try:
            response = requests.post(
                request.url,
                headers=request.headers_dict(),
                json=request.payload,
                timeout=request.timeout,
                stream=True,
            )
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
        except requests.exceptions.Timeout as exc:
            logger.error("Synchronous streaming request timed out: %s", exc)
            raise LLMAPITimeoutError(detail=str(exc)) from exc
        except requests.exceptions.HTTPError as exc:
            logger.error("Synchronous streaming HTTP error: %s", exc)
            self._handle_http_error(exc, http_error_handler)
        except requests.exceptions.RequestException as exc:
            logger.error("Synchronous streaming request exception: %s", exc)
            raise LLMAPIClientError(detail=str(exc)) from exc
        finally:
            if response is not None and not parser_owns_response:
                response.close()

    def post_multipart(
        self,
        request: TransportRequest,
        form: MultipartForm,
        *,
        http_error_handler: Optional[HTTPErrorHandler] = None,
    ) -> JSONResponse:
        response: Optional[requests.Response] = None

        try:
            response = requests.post(
                request.url,
                headers=multipart_headers(request.headers),
                data=form.fields_list(),
                files=form.files_list(),
                timeout=request.timeout,
            )
            response.raise_for_status()
            return JSONResponse(response.json())
        except LLMAPIError:
            raise
        except requests.exceptions.Timeout as exc:
            logger.error("Synchronous multipart request timed out: %s", exc)
            raise LLMAPITimeoutError(detail=str(exc)) from exc
        except requests.exceptions.HTTPError as exc:
            logger.error("Synchronous multipart HTTP error: %s", exc)
            self._handle_http_error(exc, http_error_handler)
        except requests.exceptions.RequestException as exc:
            logger.error("Synchronous multipart request exception: %s", exc)
            raise LLMAPIClientError(detail=str(exc)) from exc
        finally:
            if response is not None:
                response.close()

    @staticmethod
    def _handle_http_error(
        error: requests.exceptions.HTTPError,
        handler: Optional[HTTPErrorHandler],
    ) -> None:
        if handler is not None:
            handler(error)

        response = getattr(error, "response", None)
        raise_default_http_error(
            status_code=getattr(response, "status_code", None),
            detail=str(error),
        )


__all__ = ["RequestsSyncTransport"]
