"""OpenAI request construction and response normalization helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from ...models.messages.chat_message import Messages
from ...models.responses.chat_response import ChatResponse
from ...models.tools import ToolSpec
from ..structured_output import validate_core_portable_schema
from ..openai_compatible.common.payloads import _OpenAICompatiblePayloadMixin
from ..openai_compatible.common.usage import (
    _apply_cache_usage,
    _cache_token_counts,
    _token_count,
)


class _OpenAIPayloadMixin(_OpenAICompatiblePayloadMixin):
    """Add native policy hooks to shared OpenAI-style request handling."""

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

    def _resolve_protocol_reasoning_effort(
        self,
        reasoning_level: str | int | None,
    ) -> str | None:
        """Keep native reasoning policy behind the shared assembly hook."""
        return self._resolve_openai_reasoning_effort(reasoning_level)

    @staticmethod
    def _to_openai_structured_output_schema(schema: dict) -> dict:
        """Validate the Core profile without changing schema semantics."""
        return validate_core_portable_schema(schema, provider="openai")

    def _resolve_openai_reasoning_effort(
        self,
        reasoning_level: str | int | None,
    ) -> str | None:
        """Resolve reasoning effort for OpenAI requests.

        An omitted ``reasoning_level`` retains the historical Responses API
        default. OpenAI's client maps older GPT-5 ``none`` to ``minimal`` where
        necessary.
        """
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
