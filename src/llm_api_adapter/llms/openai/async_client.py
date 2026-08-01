"""Asynchronous HTTPX client for the OpenAI APIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator

from ..async_streaming import async_request, async_stream_request
from ..streaming import SSEEvent
from .sync_client import OpenAISyncClient


@dataclass
class OpenAIAsyncClient(OpenAISyncClient):
    """Call OpenAI's JSON and SSE endpoints without blocking the event loop."""

    def __repr__(self) -> str:
        masked = (
            f"{self.api_key[:8]}...{self.api_key[-4:]}"
            if len(self.api_key) > 12
            else "***"
        )
        return f"OpenAIAsyncClient(api_key='{masked}', endpoint='{self.endpoint}')"

    async def complete(self, model: str, timeout: float | None = None, **kwargs):
        if self._should_use_responses_api(model):
            return await self.responses(model=model, timeout=timeout, **kwargs)
        return await self.chat_completion(model=model, timeout=timeout, **kwargs)

    def stream(
        self, model: str, timeout: float | None = None, **kwargs
    ) -> AsyncIterator[SSEEvent]:
        """Return raw OpenAI events from the API appropriate for ``model``."""
        if self._should_use_responses_api(model):
            return self.stream_responses(model=model, timeout=timeout, **kwargs)
        return self.stream_chat_completion(model=model, timeout=timeout, **kwargs)

    async def chat_completion(self, model: str, timeout: float | None = None, **kwargs):
        url = f"{self.endpoint}/chat/completions"
        payload = self._prepare_chat_payload_for_model(model, kwargs)
        return await self._send_request(url, payload, timeout)

    async def responses(self, model: str, timeout: float | None = None, **kwargs):
        url = f"{self.endpoint}/responses"
        payload = self._prepare_responses_payload_for_model(model, kwargs)
        return await self._send_request(url, payload, timeout)

    def stream_chat_completion(
        self, model: str, timeout: float | None = None, **kwargs
    ) -> AsyncIterator[SSEEvent]:
        """Return raw Chat Completions SSE events without interpreting them."""
        url = f"{self.endpoint}/chat/completions"
        payload = self._prepare_chat_payload_for_model(model, kwargs)
        payload["stream"] = True
        return self._stream_request(url, payload, timeout)

    def stream_responses(
        self, model: str, timeout: float | None = None, **kwargs
    ) -> AsyncIterator[SSEEvent]:
        """Return raw Responses API SSE events without interpreting them."""
        url = f"{self.endpoint}/responses"
        payload = self._prepare_responses_payload_for_model(model, kwargs)
        payload["stream"] = True
        return self._stream_request(url, payload, timeout)

    async def _send_request(
        self,
        url: str,
        payload: dict,
        timeout: float | None = None,
    ):
        return await async_request(
            url,
            headers=self._headers(),
            payload=payload,
            timeout=timeout,
            http_error_handler=self._handle_http_error,
        )

    async def _stream_request(
        self,
        url: str,
        payload: dict,
        timeout: float | None = None,
    ) -> AsyncIterator[SSEEvent]:
        events = async_stream_request(
            url,
            headers=self._headers(),
            payload=payload,
            timeout=timeout,
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


__all__ = ["OpenAIAsyncClient"]
