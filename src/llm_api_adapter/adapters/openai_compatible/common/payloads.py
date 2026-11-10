"""Shared request assembly and protocol dispatch for OpenAI-style APIs."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ....models.messages.chat_message import Messages
from ....models.responses.chat_response import ChatResponse
from ....models.tools import ToolSpec
from ..chat_completions.payloads import _ChatCompletionsPayloadMixin
from ..responses.payloads import _ResponsesPayloadMixin


class _OpenAICompatiblePayloadMixin(
    _ChatCompletionsPayloadMixin,
    _ResponsesPayloadMixin,
):
    """Assemble shared adapter parameters and dispatch normalized responses."""

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
        normalized_reasoning_level = self._resolve_protocol_reasoning_effort(
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
            "reasoning_effort": self._resolve_protocol_reasoning_effort(
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
