from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List

from ..base_adapter import LLMAdapterBase
from ...llms.openai.async_client import OpenAIAsyncClient
from ...llms.openai.sync_client import OpenAISyncClient
from ..openai_compatible.common.adapter import _OpenAIExecutionMixin
from ..openai_compatible.adapter import OpenAICompatibleAdapter
from ..openai_compatible.chat_completions.payloads import (
    _ChatCompletionsPayloadMixin,
)
from ..openai_compatible.responses.payloads import _ResponsesPayloadMixin
from ..openai_compatible.responses.streaming import _ResponsesStreamingMixin
from .payloads import _OpenAIPayloadMixin
from .streaming import _OpenAIStreamingMixin, _ResponsesStreamState


@dataclass(repr=False)
class OpenAIAdapter(OpenAICompatibleAdapter):
    """OpenAI adapter facade for both sync and async APIs.

    The inherited compatible endpoint fields remain class-level defaults here,
    so they do not enter the native constructor or dataclass field set.
    """

    company: str = "openai"
    base_url: ClassVar[str] = ""
    api_variant: ClassVar[str | None] = None

    def __post_init__(self) -> None:
        LLMAdapterBase.__post_init__(self)

    def __repr__(self) -> str:
        return LLMAdapterBase.__repr__(self)

    def handle_error(self, error: Exception, error_message: str | None = None):
        return LLMAdapterBase.handle_error(self, error, error_message)

    def _apply_response_pricing(self, chat_response) -> None:
        LLMAdapterBase._apply_response_pricing(self, chat_response)

    _normalize_chat_completions_response = (
        _ChatCompletionsPayloadMixin._normalize_chat_completions_response
    )
    _normalize_responses_response = _ResponsesPayloadMixin._normalize_responses_response
    _validate_chat_completions_stream_event = (
        _ChatCompletionsPayloadMixin._validate_chat_completions_stream_event
    )
    _validate_responses_stream_event = (
        _ResponsesStreamingMixin._validate_responses_stream_event
    )
    _finalize_responses_stream = _ResponsesStreamingMixin._finalize_responses_stream
    _normalize_stream_usage = staticmethod(
        _OpenAIStreamingMixin._normalize_stream_usage
    )
    _token_count = staticmethod(_OpenAIStreamingMixin._token_count)
    _to_openai_structured_output_schema = staticmethod(
        _OpenAIPayloadMixin._to_openai_structured_output_schema
    )
    _build_responses_chat_params = _OpenAIPayloadMixin._build_responses_chat_params
    _responses_input_for_request = staticmethod(
        _OpenAIPayloadMixin._responses_input_for_request
    )
    _resolve_openai_reasoning_effort = (
        _OpenAIPayloadMixin._resolve_openai_reasoning_effort
    )
    _resolve_protocol_reasoning_effort = (
        _OpenAIPayloadMixin._resolve_protocol_reasoning_effort
    )

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
