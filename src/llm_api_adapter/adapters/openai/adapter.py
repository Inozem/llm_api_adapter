from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from ..base_adapter import LLMAdapterBase
from ...llms.openai.async_client import OpenAIAsyncClient
from ...llms.openai.sync_client import OpenAISyncClient
from ..openai_compatible.common.adapter import _OpenAIExecutionMixin
from .payloads import _OpenAIPayloadMixin
from .streaming import _OpenAIStreamingMixin, _ResponsesStreamState


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

    def _build_responses_stream_response(
        self,
        state: _ResponsesStreamState,
    ) -> Dict[str, Any]:
        output: List[Dict[str, Any]] = []
        if state.text_parts:
            output.append(
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "".join(state.text_parts)}
                    ],
                }
            )
        output.extend(state.function_calls.values())
        return {
            **state.response_metadata,
            "model": state.response_metadata.get("model", self.model),
            "output": output,
            "status": state.response_metadata.get("status", "completed"),
        }
