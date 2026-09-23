"""Asynchronous HTTPX calls for Z.ai's official Chat Completions API."""

from __future__ import annotations

from typing import Any, AsyncIterator

from llm_api_adapter.errors.llm_api_error import LLMAPIClientError
from llm_api_adapter.llms.async_streaming import async_request, async_stream_request
from llm_api_adapter.llms.transports import SSEEvent

from .sync_client import ZAI_CHAT_COMPLETIONS_URL, ZaiSyncClient


class ZaiAsyncClient:
    """Submit Z.ai JSON and SSE requests without blocking the event loop."""

    async def chat(
        self,
        *,
        api_key: str,
        payload: dict[str, Any],
        timeout_s: float | None,
    ) -> dict[str, Any]:
        """Post one Z.ai Chat Completions request through HTTPX."""
        response_data = await async_request(
            ZAI_CHAT_COMPLETIONS_URL,
            headers=self._headers(api_key),
            payload=payload,
            timeout=timeout_s,
            http_error_handler=ZaiSyncClient._handle_http_error,
        )
        if not isinstance(response_data, dict):
            raise LLMAPIClientError(
                detail="Z.ai Chat Completions returned a non-object response",
            )
        return response_data

    def stream(
        self,
        *,
        api_key: str,
        payload: dict[str, Any],
        timeout_s: float | None,
    ) -> AsyncIterator[SSEEvent]:
        """Open a Z.ai SSE response and request terminal usage when available."""
        stream_payload = dict(payload)
        stream_payload["stream"] = True
        stream_payload.setdefault("stream_options", {"include_usage": True})
        return async_stream_request(
            ZAI_CHAT_COMPLETIONS_URL,
            headers=self._headers(api_key),
            payload=stream_payload,
            timeout=timeout_s,
            http_error_handler=ZaiSyncClient._handle_http_error,
            stream_error_handler=ZaiSyncClient._handle_stream_error,
        )

    @staticmethod
    def _headers(api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }


# Keep the explicit protocol name available to package-local callers.
ZaiChatCompletionsAsyncClient = ZaiAsyncClient


__all__ = ["ZaiAsyncClient", "ZaiChatCompletionsAsyncClient"]
