"""Asynchronous HTTP client for Model Studio's Messages-compatible API."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from llm_api_adapter.errors.llm_api_error import LLMAPIClientError
from llm_api_adapter.llms.async_streaming import async_request, async_stream_request
from llm_api_adapter.llms.transports import SSEEvent

from .sync_client import QwenMessagesSyncClient, messages_url


class QwenMessagesAsyncClient:
    """Submit asynchronous Qwen Messages JSON and SSE requests."""

    async def chat(
        self,
        *,
        api_key: str,
        workspace_id: object,
        payload: dict[str, Any],
        timeout_s: float | None,
    ) -> dict[str, Any]:
        """Create one non-streaming Qwen Messages response."""
        response_data = await async_request(
            messages_url(workspace_id),
            headers=self._headers(api_key),
            payload=payload,
            timeout=timeout_s,
            http_error_handler=QwenMessagesSyncClient._handle_http_error,
        )
        if not isinstance(response_data, dict):
            raise LLMAPIClientError(
                detail="Qwen Messages returned a non-object response",
            )
        return response_data

    def stream(
        self,
        *,
        api_key: str,
        workspace_id: object,
        payload: dict[str, Any],
        timeout_s: float | None,
    ) -> AsyncIterator[SSEEvent]:
        """Open one Qwen Messages SSE response through the Core async helper."""
        stream_payload = dict(payload)
        stream_payload["stream"] = True
        return async_stream_request(
            messages_url(workspace_id),
            headers=self._headers(api_key),
            payload=stream_payload,
            timeout=timeout_s,
            http_error_handler=QwenMessagesSyncClient._handle_http_error,
            stream_error_handler=QwenMessagesSyncClient._handle_stream_error,
        )

    @staticmethod
    def _headers(api_key: str) -> dict[str, str]:
        return {
            "x-api-key": api_key,
            "Content-Type": "application/json",
        }


__all__ = ["QwenMessagesAsyncClient"]
