"""Asynchronous client for DeepSeek's official Responses API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator

from llm_api_adapter.errors.llm_api_error import LLMAPIClientError
from llm_api_adapter.llms.async_streaming import (
    async_request,
    async_stream_request,
)
from llm_api_adapter.llms.transports import SSEEvent

from .sync_client import (
    DEEPSEEK_RESPONSES_URL,
    DeepSeekResponsesSyncClient,
)


@dataclass(repr=False)
class DeepSeekResponsesAsyncClient:
    """Send asynchronous Responses requests through Core's async helpers."""

    api_key: str
    endpoint: str = DEEPSEEK_RESPONSES_URL

    async def create(
        self,
        *,
        model: str,
        timeout: float | None = None,
        **parameters: Any,
    ) -> dict[str, Any]:
        """Create one Responses result and validate its JSON envelope."""
        payload = await async_request(
            self.endpoint,
            headers=self._headers(),
            payload={"model": model, **parameters},
            timeout=timeout,
            http_error_handler=self._handle_http_error,
        )
        if not isinstance(payload, dict):
            raise LLMAPIClientError(
                detail="DeepSeek Responses API returned a non-object response",
            )
        return payload

    def stream(
        self,
        *,
        model: str,
        timeout: float | None = None,
        **parameters: Any,
    ) -> AsyncIterator[SSEEvent]:
        """Stream semantic Responses events through Core's async transport."""
        return async_stream_request(
            self.endpoint,
            headers=self._headers(),
            payload={
                "model": model,
                **parameters,
                "stream": True,
            },
            timeout=timeout,
            http_error_handler=self._handle_http_error,
            stream_error_handler=self._handle_stream_error,
        )

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _handle_http_error(error: Any) -> None:
        DeepSeekResponsesSyncClient._handle_http_error(error)

    @staticmethod
    def _handle_stream_error(event: SSEEvent) -> None:
        DeepSeekResponsesSyncClient._handle_stream_error(event)


DeepSeekAsyncClient = DeepSeekResponsesAsyncClient


__all__ = ["DeepSeekResponsesAsyncClient", "DeepSeekAsyncClient"]
