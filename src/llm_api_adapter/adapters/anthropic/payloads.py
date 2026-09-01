"""Anthropic request construction and response normalization helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ...errors.config_errors import LLMReasoningLevelError
from ...errors.llm_api_error import ToolChoiceError
from ...models.messages.chat_message import Messages
from ...models.responses.chat_response import ChatResponse
from ...models.tools.tool_spec import ToolSpec
from ..structured_output import validate_core_portable_schema


class _AnthropicPayloadMixin:
    """Build Anthropic payloads while keeping adapter options normalized."""

    def _build_chat_params(
        self,
        *,
        normalized_messages: Messages,
        max_tokens: int,
        temperature: float,
        top_p: float,
        reasoning_level: Optional[str | int],
        timeout_s: Optional[float],
        tools: Optional[List[ToolSpec]],
        normalized_tool_choice: Optional[str],
        parallel_tool_calls: Optional[bool],
        effective_schema: Optional[dict],
        capture_reasoning: bool,
    ) -> Dict[str, Any]:
        return self._build_request_params(
            normalized_messages=normalized_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            reasoning_level=reasoning_level,
            timeout_s=timeout_s,
            tools=tools,
            normalized_tool_choice=normalized_tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            effective_schema=effective_schema,
            capture_reasoning=capture_reasoning,
        )

    def _build_stream_params(
        self,
        *,
        normalized_messages: Messages,
        max_tokens: int,
        temperature: float,
        top_p: float,
        reasoning_level: Optional[str | int],
        timeout_s: Optional[float],
        tools: Optional[List[ToolSpec]],
        normalized_tool_choice: Optional[str],
        parallel_tool_calls: Optional[bool],
        effective_schema: Optional[dict],
        capture_reasoning: bool,
    ) -> Dict[str, Any]:
        return self._build_request_params(
            normalized_messages=normalized_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            reasoning_level=reasoning_level,
            timeout_s=timeout_s,
            tools=tools,
            normalized_tool_choice=normalized_tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            effective_schema=effective_schema,
            capture_reasoning=capture_reasoning,
        )

    def _build_request_params(
        self,
        *,
        normalized_messages: Messages,
        max_tokens: int,
        temperature: float,
        top_p: float,
        reasoning_level: Optional[str | int],
        timeout_s: Optional[float],
        tools: Optional[List[ToolSpec]],
        normalized_tool_choice: Optional[str],
        parallel_tool_calls: Optional[bool],
        effective_schema: Optional[dict],
        capture_reasoning: bool,
    ) -> Dict[str, Any]:
        system_prompt, transformed_messages = normalized_messages.to_anthropic()
        params: Dict[str, Any] = {
            "model": self.model,
            "messages": transformed_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "system": system_prompt,
            "timeout_s": timeout_s,
            "is_adaptive_thinking": self.is_adaptive_thinking,
        }
        if tools:
            params["tools"] = [self._to_anthropic_tool(tool) for tool in tools]
        if normalized_tool_choice is not None:
            params["tool_choice"] = self._to_anthropic_tool_choice(
                normalized_tool_choice
            )
        if parallel_tool_calls is False:
            params["disable_parallel_tool_use"] = True
        elif parallel_tool_calls is True:
            params["disable_parallel_tool_use"] = False
        if effective_schema is not None:
            params["output_config"] = {
                "format": {
                    "type": "json_schema",
                    "schema": self._to_anthropic_structured_output_schema(effective_schema),
                }
            }
        if reasoning_level is not None:
            provider_value = self._resolve_reasoning_level(
                reasoning_level
            ).provider_value
            if isinstance(provider_value, int):
                self.validate_reasoning_and_tokens(
                    max_tokens=max_tokens,
                    reasoning_level=reasoning_level,
                    normalized_reasoning_level=provider_value,
                )
                params["budget_tokens"] = provider_value
            elif isinstance(provider_value, str):
                params["effort"] = provider_value
        if capture_reasoning:
            params["capture_reasoning"] = True
        return {key: value for key, value in params.items() if value is not None}

    @staticmethod
    def _parse_chat_response(
        response: dict,
        *,
        capture_reasoning: bool,
    ) -> ChatResponse:
        parser_kwargs = {"capture_reasoning": True} if capture_reasoning else {}
        return ChatResponse.from_anthropic_response(
            response,
            **parser_kwargs,
        )

    def _to_anthropic_tool(self, tool: ToolSpec) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "name": tool.name,
            "input_schema": tool.json_schema,
        }
        if tool.description:
            payload["description"] = tool.description
        return payload

    @staticmethod
    def _to_anthropic_structured_output_schema(schema: dict) -> dict:
        """Validate the Core profile without changing schema semantics."""
        return validate_core_portable_schema(schema, provider="anthropic")

    def _to_anthropic_tool_choice(
        self,
        tool_choice: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        if tool_choice is None:
            return None
        allowed_modes = (
            self.model_spec.request_rules.allowed_tool_choice_modes
            if self.model_spec is not None
            else None
        )
        choice_mode = (
            tool_choice if tool_choice in {"auto", "none", "any"} else "tool"
        )
        if allowed_modes is not None and choice_mode not in allowed_modes:
            raise ToolChoiceError(
                detail=(
                    f"Model {self.model!r} does not support forced tool use; "
                    "use tool_choice='auto' or 'none'."
                )
            )
        if tool_choice == "none":
            return None
        if tool_choice in ("auto", "any"):
            return {"type": tool_choice}
        return {"type": "tool", "name": tool_choice}

    def validate_reasoning_and_tokens(
        self,
        max_tokens: int,
        reasoning_level: int | str,
        normalized_reasoning_level: int,
    ) -> None:
        if max_tokens <= normalized_reasoning_level:
            raise LLMReasoningLevelError(
                detail=(
                    f"Provided max_tokens={max_tokens}, "
                    f"reasoning_level={normalized_reasoning_level} "
                    f"(requested '{reasoning_level}'). "
                    f"Increase max_tokens above {normalized_reasoning_level} "
                    "or reduce reasoning_level."
                )
            )


__all__ = ["_AnthropicPayloadMixin"]
