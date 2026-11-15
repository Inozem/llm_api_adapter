"""Asynchronous compatible client with Z.ai's native request policy."""

from __future__ import annotations

from typing import Any

from llm_api_adapter.llms.async_streaming import async_request, async_stream_request
from llm_api_adapter.llms.openai_compatible.async_client import OpenAICompatibleAsyncClient

from .sync_client import (
    ZaiSyncClient,
    _format_zai_diagnostic,
    _validated_response,
)


class ZaiAsyncClient(OpenAICompatibleAsyncClient, ZaiSyncClient):
    """Use Core's async engine and inherit Z.ai's native payload/error policy."""

    def __init__(self, api_key: str, transport: str = "requests") -> None:
        """Initialize through the inherited dataclass without a sync transport."""
        super().__init__(api_key=api_key, transport=transport, sync_transport=None)

    def __post_init__(self, sync_transport=None) -> None:
        """Keep the async client transport-free and install safe diagnostics."""
        self._sync_transport = None
        self._diagnostic_formatter = _format_zai_diagnostic

    async def _get_async_request(self, *args: Any, **kwargs: Any) -> Any:
        """Resolve the package request seam and validate native completion JSON."""
        return _validated_response(await async_request(*args, **kwargs))

    def _get_async_stream_request(self, *args: Any, **kwargs: Any):
        """Resolve the package stream seam at stream-open time."""
        return async_stream_request(*args, **kwargs)


__all__ = ["ZaiAsyncClient"]
