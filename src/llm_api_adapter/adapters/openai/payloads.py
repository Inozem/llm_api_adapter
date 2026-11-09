"""OpenAI request construction and response normalization helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from ...models.messages.chat_message import Messages
from ...models.responses.chat_response import ChatResponse
from ...models.tools import ToolSpec
from ..structured_output import validate_core_portable_schema
from ..openai_compatible.chat_completions.payloads import _ChatCompletionsPayloadMixin
from ..openai_compatible.common.usage import (
    _apply_cache_usage,
    _cache_token_counts,
    _token_count,
)
from ..openai_compatible.responses.payloads import _ResponsesPayloadMixin


class _OpenAIPayloadMixin(_ChatCompletionsPayloadMixin, _ResponsesPayloadMixin):
    """Build provider payloads while keeping adapter-level options normalized."""

    def _build_chat_params(
        self,
        *,
        normalized_messages: Messages,
        use_responses_api: bool,
        max_tokens: Optional[int],
        temperature: float,
        top_p: float,
        reasoning_level: Optional[str | int],
        tools: Optional[List[ToolSpec]],
        normalized_tool_choice: Optional[str],
        parallel_tool_calls: Optional[bool],
        previous_response: Optional[ChatResponse],
        effective_schema: Optional[dict],
        capture_reasoning: bool,
    ) -> Dict[str, Any]:
        normalized_reasoning_level = self._resolve_openai_reasoning_effort(
            reasoning_level
        )
        params: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "reasoning_effort": normalized_reasoning_level,
        }

        if use_responses_api:
            params.update(
                self._build_responses_chat_params(
                    normalized_messages=normalized_messages,
                    tools=tools,
                    normalized_tool_choice=normalized_tool_choice,
                    previous_response=previous_response,
                    effective_schema=effective_schema,
                    capture_reasoning=capture_reasoning,
                )
            )
        else:
            params.update(
                self._build_chat_completions_params(
                    normalized_messages=normalized_messages,
                    tools=tools,
                    normalized_tool_choice=normalized_tool_choice,
                    parallel_tool_calls=parallel_tool_calls,
                    effective_schema=effective_schema,
                )
            )

        return {key: value for key, value in params.items() if value is not None}

    def _build_responses_chat_params(
        self,
        *,
        normalized_messages: Messages,
        tools: Optional[List[ToolSpec]],
        normalized_tool_choice: Optional[str],
        previous_response: Optional[ChatResponse],
        effective_schema: Optional[dict],
        capture_reasoning: bool,
    ) -> Dict[str, Any]:
        params = self._build_responses_wire_params(
            input_items=self._responses_input_for_request(
                normalized_messages,
                previous_response,
            ),
            instructions=normalized_messages.to_openai_responses_instructions(),
            tools=tools,
            normalized_tool_choice=normalized_tool_choice,
            effective_schema=effective_schema,
        )
        if capture_reasoning:
            params["capture_reasoning"] = True
        if previous_response is not None and previous_response.response_id is not None:
            params["previous_response_id"] = previous_response.response_id
        return params

    def _parse_chat_response(
        self,
        response: dict,
        *,
        use_responses_api: bool,
        capture_reasoning: bool,
    ) -> ChatResponse:
        if use_responses_api:
            return self._normalize_responses_response(
                response,
                capture_reasoning=capture_reasoning,
            )
        return self._normalize_chat_completions_response(response)

    def _build_stream_params(
        self,
        *,
        normalized_messages: Messages,
        use_responses_api: bool,
        max_tokens: Optional[int],
        temperature: float,
        top_p: float,
        reasoning_level: Optional[str | int],
        tools: Optional[List[ToolSpec]],
        normalized_tool_choice: Optional[str],
        parallel_tool_calls: Optional[bool],
        previous_response: Optional[ChatResponse],
        effective_schema: Optional[dict],
        capture_reasoning: bool,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "reasoning_effort": self._resolve_openai_reasoning_effort(
                reasoning_level
            ),
        }
        if use_responses_api:
            params.update(
                self._build_responses_chat_params(
                    normalized_messages=normalized_messages,
                    tools=tools,
                    normalized_tool_choice=normalized_tool_choice,
                    previous_response=previous_response,
                    effective_schema=effective_schema,
                    capture_reasoning=capture_reasoning,
                )
            )
        else:
            params.update(
                self._build_chat_completions_params(
                    normalized_messages=normalized_messages,
                    tools=tools,
                    normalized_tool_choice=normalized_tool_choice,
                    parallel_tool_calls=parallel_tool_calls,
                    effective_schema=effective_schema,
                )
            )
        return {key: value for key, value in params.items() if value is not None}

    @staticmethod
    def _responses_input_for_request(
        normalized_messages: Messages,
        previous_response: Optional[ChatResponse],
    ) -> List[Dict[str, Any]]:
        """Use server context for calls covered by a Responses continuation ID."""
        input_items = normalized_messages.to_openai_responses_input()
        if previous_response is None or previous_response.response_id is None:
            return input_items
        return [
            item
            for item in input_items
            if not (isinstance(item, Mapping) and item.get("type") == "function_call")
        ]

    @staticmethod
    def _to_openai_structured_output_schema(schema: dict) -> dict:
        """Validate the Core profile without changing schema semantics."""
        return validate_core_portable_schema(schema, provider="openai")

    def _resolve_openai_reasoning_effort(
        self,
        reasoning_level: str | int | None,
    ) -> str | None:
        # A missing value retains the Responses API default used before the
        # model-aware resolver. OpenAI's client normalizes the older GPT-5
        # family from ``none`` to ``minimal`` where necessary.
        if reasoning_level is None:
            return "none" if self.is_reasoning else None

        provider_value = self._resolve_reasoning_level(reasoning_level).provider_value
        if provider_value is None:
            return None
        if not isinstance(provider_value, str):
            raise TypeError(
                "OpenAI reasoning resolution must produce a categorical value"
            )
        return provider_value
