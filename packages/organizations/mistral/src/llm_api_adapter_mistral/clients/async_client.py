"""Asynchronous transport calls for Mistral's official API."""

from __future__ import annotations

from typing import Any, AsyncIterator, Awaitable, Callable, Mapping, Optional

from llm_api_adapter.errors.llm_api_error import LLMAPIClientError
from llm_api_adapter.llms.async_streaming import async_request, async_stream_request
from llm_api_adapter.llms.openai_compatible.async_client import (
    OpenAICompatibleAsyncClient,
)
from llm_api_adapter.llms.transports import (
    HTTPErrorHandler,
    SSEEvent,
    StreamErrorHandler,
)

from .sync_client import (
    MISTRAL_OCR_URL,
    MISTRAL_RESPONSES_URL,
    _NativeRequestView,
    MistralSyncClient,
)


class MistralAsyncClient(OpenAICompatibleAsyncClient, MistralSyncClient):
    """Use Core's async engine with Mistral's inherited native request policy."""

    def __init__(self) -> None:
        """Initialize native async settings without constructing sync transport."""
        self.api_key = ""
        self.endpoint = MISTRAL_RESPONSES_URL
        self._api_variant_override = "responses"
        self._diagnostic_formatter = None
        self._adapter_apost_payload: Optional[
            Callable[[dict[str, Any], Optional[float]], Awaitable[dict[str, Any]]]
        ] = None
        self._adapter_astream_payload: Optional[
            Callable[[dict[str, Any], Optional[float]], AsyncIterator[SSEEvent]]
        ] = None

    async def chat(
        self,
        *,
        endpoint: str,
        headers: Mapping[str, str],
        payload: dict[str, Any],
        timeout_s: Optional[float],
        http_error_handler: Callable[[Any], None],
    ) -> dict[str, Any]:
        """Submit a legacy raw JSON request through the shared async seam."""
        view = self._async_request_view(headers, http_error_handler)
        response_data = await view._send_request(endpoint, payload, timeout_s)
        return _validated_response(
            response_data, "Mistral returned a non-object response"
        )

    async def ocr(
        self,
        *,
        headers: Mapping[str, str],
        payload: dict[str, Any],
        timeout_s: Optional[float],
        http_error_handler: Callable[[Any], None],
    ) -> dict[str, Any]:
        """Keep OCR on its existing independent Mistral resource."""
        view = self._async_request_view(headers, http_error_handler)
        response_data = await view._send_request(MISTRAL_OCR_URL, payload, timeout_s)
        return _validated_response(
            response_data, "Mistral OCR returned a non-object response"
        )

    def stream_chat(
        self,
        *,
        endpoint: str,
        headers: Mapping[str, str],
        payload: dict[str, Any],
        timeout_s: Optional[float],
        http_error_handler: Callable[[Any], None],
        stream_error_handler: Callable[[SSEEvent], None],
    ) -> AsyncIterator[SSEEvent]:
        """Return lazy raw events through Core's runner and cleanup lifecycle."""
        view = self._async_request_view(
            headers, http_error_handler, stream_error_handler
        )
        return view._stream_request(endpoint, payload, timeout_s)

    async def responses(
        self, model: str, timeout: float | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Run inherited Responses preparation against a complete native URL."""
        endpoint = kwargs.pop("endpoint", self.endpoint)
        headers = kwargs.pop("headers", self._headers())
        http_error_handler = kwargs.pop(
            "http_error_handler", self._handle_http_error
        )
        payload = self._prepare_responses_payload_for_model(model, kwargs)
        if self._adapter_apost_payload is not None:
            return await self._adapter_apost_payload(payload, timeout)
        view = self._async_request_view(headers, http_error_handler)
        response = await view._send_request(endpoint, payload, timeout)
        return _validated_response(response, "Mistral returned a non-object response")

    def stream_responses(
        self, model: str, timeout: float | None = None, **kwargs: Any
    ) -> AsyncIterator[SSEEvent]:
        """Stream native Responses from the caller's complete resource URL."""
        endpoint = kwargs.pop("endpoint", self.endpoint)
        headers = kwargs.pop("headers", self._headers())
        http_error_handler = kwargs.pop(
            "http_error_handler", self._handle_http_error
        )
        stream_error_handler = kwargs.pop(
            "stream_error_handler", self._handle_stream_error
        )
        payload = self._prepare_responses_payload_for_model(model, kwargs)
        payload["stream"] = True
        if self._adapter_astream_payload is not None:
            return self._adapter_astream_payload(payload, timeout)
        view = self._async_request_view(
            headers, http_error_handler, stream_error_handler
        )
        return view._stream_request(endpoint, payload, timeout)

    def _async_request_view(
        self,
        headers: Mapping[str, str],
        http_error_handler: HTTPErrorHandler,
        stream_error_handler: Optional[StreamErrorHandler] = None,
    ) -> _NativeAsyncRequestView:
        """Bind per-call headers and callbacks without mutable client state."""
        return _NativeAsyncRequestView(
            headers=headers,
            http_error_handler=http_error_handler,
            stream_error_handler=stream_error_handler,
        )

    async def _get_async_request(self, *args: Any, **kwargs: Any) -> Any:
        """Resolve the package-level async JSON patch point at call time."""
        return await async_request(*args, **kwargs)

    def _get_async_stream_request(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[SSEEvent]:
        """Resolve the package-level async SSE patch point at stream-open time."""
        return async_stream_request(*args, **kwargs)


class _NativeAsyncRequestView(MistralAsyncClient, _NativeRequestView):
    """Snapshot one async request's headers and handlers for Core's runners."""

    def __init__(
        self,
        *,
        headers: Mapping[str, str],
        http_error_handler: HTTPErrorHandler,
        stream_error_handler: Optional[StreamErrorHandler],
    ) -> None:
        super().__init__()
        self._request_headers = dict(headers)
        self._request_http_error_handler = http_error_handler
        self._request_stream_error_handler = stream_error_handler

    async def _get_async_request(self, *args: Any, **kwargs: Any) -> Any:
        """Keep the caller's HTTP callback identity at the transport boundary."""
        handler = kwargs.get("http_error_handler")
        if getattr(handler, "__self__", None) is self:
            kwargs["http_error_handler"] = self._request_http_error_handler
        return await super()._get_async_request(*args, **kwargs)

    def _get_async_stream_request(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[SSEEvent]:
        """Keep native stream callback identity at the async alias boundary."""
        handler = kwargs.get("http_error_handler")
        if getattr(handler, "__self__", None) is self:
            kwargs["http_error_handler"] = self._request_http_error_handler
        handler = kwargs.get("stream_error_handler")
        if handler is not None and getattr(handler, "__self__", None) is self:
            kwargs["stream_error_handler"] = self._request_stream_error_handler
        events = super()._get_async_stream_request(*args, **kwargs)
        return _ensure_closeable_async_iterator(events)


class _CloseableAsyncIterator:
    """Provide Core's async close hook for legacy iterators that lack one."""

    _events: AsyncIterator[SSEEvent]

    def __init__(self, events: AsyncIterator[SSEEvent]) -> None:
        self._events = events

    def __aiter__(self) -> _CloseableAsyncIterator:
        return self

    async def __anext__(self) -> SSEEvent:
        return await self._events.__anext__()

    async def aclose(self) -> None:
        close = getattr(self._events, "aclose", None)
        if callable(close):
            await close()


def _ensure_closeable_async_iterator(
    events: AsyncIterator[SSEEvent],
) -> AsyncIterator[SSEEvent]:
    """Keep real async close behavior and adapt legacy iterators locally."""
    if callable(getattr(events, "aclose", None)):
        return events
    return _CloseableAsyncIterator(events)


def _validated_response(
    response_data: Any, malformed_detail: str
) -> dict[str, Any]:
    """Validate the open JSON result at the native low-level boundary."""
    if not isinstance(response_data, dict):
        raise LLMAPIClientError(detail=malformed_detail)
    return response_data


__all__ = ["MistralAsyncClient"]
