"""Asynchronous HTTPX calls for Z.ai's official Chat Completions API."""

from __future__ import annotations

from typing import Any, AsyncIterator

from llm_api_adapter.llms.async_streaming import async_request, async_stream_request
from llm_api_adapter.llms.openai_compatible.async_client import _AsyncExecutionMixin
from llm_api_adapter.llms.openai_compatible.sync_client import (
    OpenAICompatibleSyncClient,
)
from llm_api_adapter.llms.transports import SSEEvent

from .sync_client import (
    ZAI_BASE_URL,
    ZAI_CHAT_COMPLETIONS_URL,
    ZaiSyncClient,
    _validated_response,
)


class _ZaiBoundAsyncClient(_AsyncExecutionMixin, OpenAICompatibleSyncClient):
    """Request-bound Z.ai policy over Core's async JSON and SSE engine."""

    def __init__(self, api_key: str) -> None:
        """Bind this call's key without constructing a synchronous transport."""
        self.api_key = api_key
        self.endpoint = ZAI_BASE_URL
        self.transport = "requests"
        self._sync_transport = None
        self._api_variant_override = None
        self._diagnostic_formatter = None

    async def chat_completion(
        self, model: str, timeout: float | None = None, **kwargs: Any
    ) -> Any:
        """Send native max_tokens and sampling fields through the shared engine."""
        payload = {"model": model, **dict(kwargs)}
        response_data = await self._send_request(
            ZAI_CHAT_COMPLETIONS_URL, payload, timeout
        )
        return _validated_response(response_data)

    def stream_chat_completion(
        self, model: str, timeout: float | None = None, **kwargs: Any
    ) -> AsyncIterator[SSEEvent]:
        """Open a native stream with the existing terminal-usage default."""
        payload = {"model": model, **dict(kwargs)}
        return self._stream_raw_payload(payload, timeout)

    async def _complete_raw(
        self, payload: dict[str, Any], timeout: float | None
    ) -> dict[str, Any]:
        """Send a legacy raw payload without requiring an explicit model field."""
        response_data = await self._send_request(
            ZAI_CHAT_COMPLETIONS_URL, dict(payload), timeout
        )
        return _validated_response(response_data)

    def _stream_raw_payload(
        self, payload: dict[str, Any], timeout: float | None
    ) -> AsyncIterator[SSEEvent]:
        """Copy and prepare the legacy payload before shared SSE execution."""
        stream_payload = dict(payload)
        stream_payload["stream"] = True
        stream_payload.setdefault("stream_options", {"include_usage": True})
        return self._stream_request(
            ZAI_CHAT_COMPLETIONS_URL, stream_payload, timeout
        )

    async def _get_async_request(self, *args: Any, **kwargs: Any) -> Any:
        """Resolve the package request seam at send time for live patching."""
        return await async_request(*args, **kwargs)

    def _get_async_stream_request(self, *args: Any, **kwargs: Any):
        """Resolve the package stream seam at stream-open time."""
        return async_stream_request(*args, **kwargs)

    def _headers(self) -> dict[str, str]:
        return ZaiAsyncClient._headers(self.api_key)

    def _handle_http_error(self, http_error: Any) -> None:
        ZaiSyncClient._handle_http_error(http_error)

    def _handle_stream_error(self, event: SSEEvent) -> None:
        ZaiSyncClient._handle_stream_error(event)


class ZaiAsyncClient:
    """Submit Z.ai Chat Completions requests through Core's async engine."""

    def _for_request(self, api_key: str) -> _ZaiBoundAsyncClient:
        """Create an invocation-bound client without shared mutable state."""
        return _ZaiBoundAsyncClient(api_key)

    async def chat(
        self,
        *,
        api_key: str,
        payload: dict[str, Any],
        timeout_s: float | None,
    ) -> dict[str, Any]:
        """Post one raw JSON payload and validate its top-level response."""
        return await self._for_request(api_key)._complete_raw(payload, timeout_s)

    def stream(
        self,
        *,
        api_key: str,
        payload: dict[str, Any],
        timeout_s: float | None,
    ) -> AsyncIterator[SSEEvent]:
        """Open a Z.ai SSE response and request terminal usage when available."""
        return self._for_request(api_key)._stream_raw_payload(payload, timeout_s)

    @staticmethod
    def _headers(api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }


# Keep the explicit protocol name available to package-local callers.
ZaiChatCompletionsAsyncClient = ZaiAsyncClient


__all__ = ["ZaiAsyncClient", "ZaiChatCompletionsAsyncClient"]
