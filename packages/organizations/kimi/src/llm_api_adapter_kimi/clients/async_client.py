"""Asynchronous HTTPX calls for Kimi's official Chat Completions API."""

from __future__ import annotations

from typing import Any, AsyncIterator

from llm_api_adapter.errors.llm_api_error import LLMAPIClientError
from llm_api_adapter.llms.async_streaming import async_request, async_stream_request
from llm_api_adapter.llms.transports import SSEEvent

from .sync_client import KIMI_CHAT_COMPLETIONS_URL, KimiSyncClient


class KimiAsyncClient:
    """Submit Kimi JSON and SSE requests without blocking the event loop."""

    async def chat(
        self,
        *,
        api_key: str,
        payload: dict[str, Any],
        timeout_s: float | None,
    ) -> dict[str, Any]:
        """Post one Kimi Chat Completions request through HTTPX."""
        response_data = await async_request(
            KIMI_CHAT_COMPLETIONS_URL,
            headers=self._headers(api_key),
            payload=payload,
            timeout=timeout_s,
            http_error_handler=KimiSyncClient._handle_http_error,
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
        """Open a Kimi SSE response and request final usage in its last chunk."""
        stream_payload = dict(payload)
        stream_payload["stream"] = True
        stream_payload["stream_options"] = {"include_usage": True}
        return async_stream_request(
            KIMI_CHAT_COMPLETIONS_URL,
            headers=self._headers(api_key),
            payload=stream_payload,
            timeout=timeout_s,
            http_error_handler=KimiSyncClient._handle_http_error,
            stream_error_handler=KimiSyncClient._handle_stream_error,
        )

    @staticmethod
    def _headers(api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }


__all__ = ["KimiAsyncClient"]
