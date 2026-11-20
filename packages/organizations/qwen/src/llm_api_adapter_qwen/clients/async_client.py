"""Asynchronous HTTP client for Model Studio's Messages-compatible API."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from llm_api_adapter.errors.llm_api_error import LLMAPIClientError
from llm_api_adapter.llms.async_streaming import async_request, async_stream_request
from llm_api_adapter.llms.anthropic_compatible.async_client import (
    AnthropicCompatibleAsyncClient,
)
from llm_api_adapter.llms.transports import SSEEvent, TransportRequest

from .sync_client import QwenMessagesSyncClient, messages_url


class QwenMessagesAsyncClient(AnthropicCompatibleAsyncClient):
    """Submit asynchronous Qwen Messages JSON and SSE requests."""

    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        return object.__repr__(self)

    async def _get_async_request(self, *args: Any, **kwargs: Any) -> Any:
        # Keep Qwen's historical module-level monkeypatch surface.
        return await async_request(*args, **kwargs)

    def _get_async_stream_request(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[SSEEvent]:
        return async_stream_request(*args, **kwargs)

    _handle_http_error = QwenMessagesSyncClient._handle_http_error
    _handle_stream_error = QwenMessagesSyncClient._handle_stream_error

    def _raise_execution_error(self, error: BaseException) -> None:
        raise error

    async def chat(
        self,
        *,
        api_key: str,
        workspace_id: object,
        payload: dict[str, Any],
        timeout_s: float | None,
    ) -> dict[str, Any]:
        """Create one non-streaming Qwen Messages response."""
        response_data = await self._execute_prepared_request(
            TransportRequest(
                url=messages_url(workspace_id),
                headers=self._headers(api_key),
                payload=payload,
                timeout=timeout_s,
            )
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
        """Prepare one native SSE request for shared lazy async execution."""
        stream_payload = dict(payload)
        stream_payload["stream"] = True
        request = TransportRequest(
            url=messages_url(workspace_id),
            headers=self._headers(api_key),
            payload=stream_payload,
            timeout=timeout_s,
        )
        return self._execute_prepared_stream(request)

    @staticmethod
    def _headers(api_key: str) -> dict[str, str]:
        return {
            "x-api-key": api_key,
            "Content-Type": "application/json",
        }


__all__ = ["QwenMessagesAsyncClient"]
