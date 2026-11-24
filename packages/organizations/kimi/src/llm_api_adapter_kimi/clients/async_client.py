"""Asynchronous HTTPX calls for Kimi's official Chat Completions API."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from typing import Any

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

from .sync_client import KIMI_CHAT_COMPLETIONS_URL, KimiSyncClient


class KimiAsyncClient(OpenAICompatibleAsyncClient, KimiSyncClient):
    """Use Core's async engine with Kimi's native request and error policy."""

    def __init__(self) -> None:
        """Initialize async policy without creating a synchronous transport."""
        self.api_key = ""
        self.endpoint = KIMI_CHAT_COMPLETIONS_URL
        self._api_variant_override = None
        self._diagnostic_formatter = None

    async def chat(
        self,
        *,
        api_key: str,
        payload: dict[str, Any],
        timeout_s: float | None,
    ) -> dict[str, Any]:
        """Post one unchanged raw payload through Core's async JSON runner."""
        view = self._async_request_view(api_key)
        response_data = await view._send_request(
            KIMI_CHAT_COMPLETIONS_URL,
            payload,
            timeout_s,
        )
        if not isinstance(response_data, dict):
            raise LLMAPIClientError(
                detail="Kimi Chat Completions returned a non-object response",
            )
        return response_data

    def stream(
        self,
        *,
        api_key: str,
        payload: dict[str, Any],
        timeout_s: float | None,
    ) -> AsyncIterator[SSEEvent]:
        """Return a lazy stream with Kimi's terminal-usage request option."""

        async def run() -> AsyncIterator[SSEEvent]:
            # Defer payload copying and request binding until first iteration.
            stream_payload = dict(payload)
            stream_payload["stream"] = True
            stream_payload["stream_options"] = {"include_usage": True}
            view = self._async_request_view(api_key)
            events = view._stream_request(
                KIMI_CHAT_COMPLETIONS_URL,
                stream_payload,
                timeout_s,
            )
            try:
                async for event in events:
                    yield event
            except BaseException:
                try:
                    await events.aclose()
                except BaseException:
                    # Preserve the active stream error, cancellation, or
                    # explicit close while still attempting resource cleanup.
                    pass
                raise
            else:
                await events.aclose()

        return run()

    def _async_request_view(self, api_key: str) -> _KimiAsyncRequestView:
        """Snapshot per-call credentials and dynamically dispatched handlers."""
        return _KimiAsyncRequestView(
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            http_error_handler=self._handle_http_error,
            stream_error_handler=self._handle_stream_error,
        )

    async def _get_async_request(self, *args: Any, **kwargs: Any) -> Any:
        """Resolve Kimi's module alias at the start of each JSON operation."""
        return await async_request(*args, **kwargs)

    def _get_async_stream_request(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> AsyncIterator[SSEEvent]:
        """Resolve the SSE alias when iteration opens and ensure close support."""
        events = async_stream_request(*args, **kwargs)
        return _ensure_closeable_async_iterator(events)

    @staticmethod
    def _is_failed_stream_event(event: SSEEvent) -> bool:
        """Kimi does not classify Responses-only failure frames as errors."""
        return False


class _KimiAsyncRequestView(KimiAsyncClient):
    """Compatible request view with immutable invocation-local bindings."""

    _request_headers: dict[str, str]
    _request_http_error_handler: HTTPErrorHandler
    _request_stream_error_handler: StreamErrorHandler

    def __init__(
        self,
        *,
        headers: Mapping[str, str],
        http_error_handler: HTTPErrorHandler,
        stream_error_handler: StreamErrorHandler,
    ) -> None:
        super().__init__()
        self._request_headers = dict(headers)
        self._request_http_error_handler = http_error_handler
        self._request_stream_error_handler = stream_error_handler

    def _headers(self) -> dict[str, str]:
        return dict(self._request_headers)

    def _handle_http_error(self, http_error: Any) -> None:
        self._request_http_error_handler(http_error)

    def _handle_stream_error(self, event: SSEEvent) -> None:
        self._request_stream_error_handler(event)


class _CloseableAsyncIterator:
    """Adapt only accepted async iterators lacking Core's ``aclose`` hook."""

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
    """Forward a real close hook or supply the minimal accepted adapter."""
    if callable(getattr(events, "aclose", None)):
        return events
    return _CloseableAsyncIterator(events)


__all__ = ["KimiAsyncClient"]
