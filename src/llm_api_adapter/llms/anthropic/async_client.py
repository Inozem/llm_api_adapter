"""Asynchronous HTTPX client for the Anthropic Messages API."""

from __future__ import annotations

from contextlib import aclosing
from dataclasses import dataclass
from typing import AsyncIterator

from ..async_streaming import async_request, async_stream_request
from ..anthropic_compatible.async_client import _AsyncMessagesExecutionMixin
from ..transports import TransportRequest
from ..streaming import SSEEvent
from .sync_client import ClaudeSyncClient


@dataclass
class ClaudeAsyncClient(_AsyncMessagesExecutionMixin, ClaudeSyncClient):
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

    async def _get_async_request(self, *args, **kwargs):
        return await async_request(*args, **kwargs)

    def _get_async_stream_request(self, *args, **kwargs):
        return async_stream_request(*args, **kwargs)

    async def chat_completion(
        self,
        model: str,
        timeout_s: float | None = None,
        **kwargs,
    ):
        url, payload = self._request_data(model, kwargs, stream=False)
        return await self._send_request(url, payload, timeout_s)

    def stream(
        self,
        model: str,
        timeout_s: float | None = None,
        **kwargs,
    ) -> AsyncIterator[SSEEvent]:
        """Return raw Anthropic Messages SSE events."""
        url, payload = self._request_data(model, kwargs, stream=True)
        return self._stream_request(url, payload, timeout_s)

    async def _send_request(
        self,
        url: str,
        payload: dict,
        timeout_s: float | None = None,
    ):
        return await self._execute_prepared_request(
            TransportRequest(
                url=url,
                headers=self._headers(),
                payload=payload,
                timeout=timeout_s,
            )
        )

    async def _stream_request(
        self,
        url: str,
        payload: dict,
        timeout_s: float | None = None,
    ) -> AsyncIterator[SSEEvent]:
        """Stream raw events, resolving native headers and helper on iteration."""
        async with aclosing(
            self._execute_prepared_stream(
                TransportRequest(
                    url=url,
                    headers=self._headers(),
                    payload=payload,
                    timeout=timeout_s,
                )
            )
        ) as events:
            async for event in events:
                yield event


__all__ = ["ClaudeAsyncClient"]
