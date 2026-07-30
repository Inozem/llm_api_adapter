"""Asynchronous HTTPX client for the Anthropic Messages API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator

from ..async_streaming import async_request, async_stream_request
from ..streaming import SSEEvent
from .sync_client import ClaudeSyncClient


@dataclass
class ClaudeAsyncClient(ClaudeSyncClient):
    """Call Anthropic's Messages API without blocking the event loop."""

    def __repr__(self) -> str:
        masked = (
            f"{self.api_key[:8]}...{self.api_key[-4:]}"
            if len(self.api_key) > 12
            else "***"
        )
        return (
            f"ClaudeAsyncClient(api_key='{masked}', endpoint='{self.endpoint}', "
            f"api_version='{self.api_version}')"
        )

    async def chat_completion(
        self,
        model: str,
        timeout_s: float | None = None,
        **kwargs,
    ):
        url = f"{self.endpoint}/messages"
        payload = self._prepare_chat_payload_for_model(model, kwargs)
        return await self._send_request(url, payload, timeout_s)

    def stream(
        self,
        model: str,
        timeout_s: float | None = None,
        **kwargs,
    ) -> AsyncIterator[SSEEvent]:
        """Return raw Anthropic Messages SSE events."""
        url = f"{self.endpoint}/messages"
        payload = self._prepare_chat_payload_for_model(model, kwargs)
        payload["stream"] = True
        return self._stream_request(url, payload, timeout_s)

    async def _send_request(
        self,
        url: str,
        payload: dict,
        timeout_s: float | None = None,
    ):
        return await async_request(
            url,
            headers=self._headers(),
            payload=payload,
            timeout=timeout_s,
            http_error_handler=self._handle_http_error,
        )

    async def _stream_request(
        self,
        url: str,
        payload: dict,
        timeout_s: float | None = None,
    ) -> AsyncIterator[SSEEvent]:
        events = async_stream_request(
            url,
            headers=self._headers(),
            payload=payload,
            timeout=timeout_s,
            http_error_handler=self._handle_http_error,
            stream_error_handler=self._handle_stream_error,
        )
        try:
            async for event in events:
                yield event
        finally:
            await events.aclose()


__all__ = ["ClaudeAsyncClient"]
