"""Asynchronous HTTPX client for the OpenAI APIs."""

from __future__ import annotations

from dataclasses import dataclass

from ..async_streaming import async_request, async_stream_request
from ..openai_compatible.async_client import _AsyncExecutionMixin
from .sync_client import OpenAISyncClient


@dataclass
class OpenAIAsyncClient(_AsyncExecutionMixin, OpenAISyncClient):
    """Call OpenAI's JSON and SSE endpoints without blocking the event loop."""

    def __repr__(self) -> str:
        masked = (
            f"{self.api_key[:8]}...{self.api_key[-4:]}"
            if len(self.api_key) > 12
            else "***"
        )
        result = f"OpenAIAsyncClient(api_key='{masked}', endpoint='{self.endpoint}')"
        return self._format_diagnostic(result)

    async def _get_async_request(self, *args, **kwargs):
        return await async_request(*args, **kwargs)

    def _get_async_stream_request(self, *args, **kwargs):
        return async_stream_request(*args, **kwargs)


__all__ = ["OpenAIAsyncClient"]
