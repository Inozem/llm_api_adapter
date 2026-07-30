"""Asynchronous HTTPX client for the Google Generative Language API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator

from ..async_streaming import async_request, async_stream_request
from ..streaming import SSEEvent
from .sync_client import GeminiSyncClient


@dataclass
class GeminiAsyncClient(GeminiSyncClient):
    """Call Google's Generate Content API without blocking the event loop."""

    def __repr__(self) -> str:
        masked = (
            f"{self.api_key[:8]}...{self.api_key[-4:]}"
            if len(self.api_key) > 12
            else "***"
        )
        return f"GeminiAsyncClient(api_key='{masked}', endpoint='{self.endpoint}')"

    async def chat_completion(
        self,
        model: str,
        timeout_s: float | None = None,
        **kwargs,
    ):
        url = f"{self.endpoint}/models/{model}:generateContent"
        payload = self._prepare_chat_payload_for_model(model, kwargs)
        response = await self._send_request(url, payload, timeout_s)
        return response

    def stream(
        self,
        model: str,
        timeout_s: float | None = None,
        **kwargs,
    ) -> AsyncIterator[SSEEvent]:
        """Stream raw GenerateContentResponse chunks from Gemini."""
        url = f"{self.endpoint}/models/{model}:streamGenerateContent?alt=sse"
        payload = self._prepare_chat_payload_for_model(model, kwargs)
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
                if self._is_failed_stream_event(event):
                    self._handle_stream_error(event)
                yield event
        finally:
            await events.aclose()


__all__ = ["GeminiAsyncClient"]
