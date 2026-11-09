from __future__ import annotations

from dataclasses import dataclass

from ..base_adapter import LLMAdapterBase
from ...llms.openai.async_client import OpenAIAsyncClient
from ...llms.openai.sync_client import OpenAISyncClient
from ..openai_compatible.common.adapter import _OpenAIExecutionMixin
from .payloads import _OpenAIPayloadMixin
from .streaming import _OpenAIStreamingMixin


@dataclass(repr=False)
class OpenAIAdapter(
    _OpenAIExecutionMixin, _OpenAIPayloadMixin, _OpenAIStreamingMixin, LLMAdapterBase
):
    """OpenAI adapter facade for both sync and async APIs."""

    company: str = "openai"

    def _create_sync_client(self) -> OpenAISyncClient:
        """Construct the default synchronous OpenAI client."""
        return OpenAISyncClient(
            api_key=self.api_key,
            transport=self.transport,
        )

    def _create_async_client(self) -> OpenAIAsyncClient:
        """Construct the default asynchronous OpenAI client."""
        return OpenAIAsyncClient(api_key=self.api_key)
