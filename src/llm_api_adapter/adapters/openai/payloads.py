"""OpenAI request construction and response normalization helpers."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import warnings

from ...models.messages.chat_message import Messages
from ...models.responses.chat_response import ChatResponse
from ...models.tools import ToolSpec

logger = logging.getLogger(__name__)


class _OpenAIPayloadMixin:
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
        normalized_reasoning_level = self._normalize_reasoning_level(reasoning_level)
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
        params: Dict[str, Any] = {
            "input": normalized_messages.to_openai_responses_input(),
            "tools": self._map_tools_to_openai_responses(tools),
            "tool_choice": self._map_tool_choice_to_openai_responses(
                normalized_tool_choice
            ),
        }
        if capture_reasoning:
            params["capture_reasoning"] = True
        instructions = normalized_messages.to_openai_responses_instructions()
        if instructions is not None:
            params["instructions"] = instructions
        if previous_response is not None and previous_response.response_id is not None:
            params["previous_response_id"] = previous_response.response_id
        if effective_schema is not None:
            params["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "response",
                    "strict": True,
                    "schema": self._enforce_strict_schema(effective_schema),
                }
            }
        return params

    def _build_chat_completions_params(
        self,
        *,
        normalized_messages: Messages,
        tools: Optional[List[ToolSpec]],
        normalized_tool_choice: Optional[str],
        parallel_tool_calls: Optional[bool],
        effective_schema: Optional[dict],
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "messages": normalized_messages.to_openai(),
            "tools": self._map_tools_to_openai(tools),
            "tool_choice": self._map_tool_choice_to_openai(
                normalized_tool_choice
            ),
            "parallel_tool_calls": parallel_tool_calls,
        }
        if effective_schema is not None:
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "strict": True,
                    "schema": self._enforce_strict_schema(effective_schema),
                },
            }
        return params

    @staticmethod
    def _parse_chat_response(
        response: dict,
        *,
        use_responses_api: bool,
        capture_reasoning: bool,
    ) -> ChatResponse:
        if use_responses_api:
            parser_kwargs = {"capture_reasoning": True} if capture_reasoning else {}
            return ChatResponse.from_openai_responses_response(
                response,
                **parser_kwargs,
            )
        return ChatResponse.from_openai_response(response)

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
            "reasoning_effort": self._normalize_reasoning_level(reasoning_level),
        }
        if use_responses_api:
            params["input"] = normalized_messages.to_openai_responses_input()
            params["tools"] = self._map_tools_to_openai_responses(tools)
            params["tool_choice"] = self._map_tool_choice_to_openai_responses(
                normalized_tool_choice
            )
            instructions = normalized_messages.to_openai_responses_instructions()
            if instructions is not None:
                params["instructions"] = instructions
            if previous_response is not None and previous_response.response_id is not None:
                params["previous_response_id"] = previous_response.response_id
            if effective_schema is not None:
                params["text"] = {
                    "format": {
                        "type": "json_schema",
                        "name": "response",
                        "strict": True,
                        "schema": self._enforce_strict_schema(effective_schema),
                    }
                }
            if capture_reasoning:
                params["capture_reasoning"] = True
        else:
            params["messages"] = normalized_messages.to_openai()
            params["tools"] = self._map_tools_to_openai(tools)
            params["tool_choice"] = self._map_tool_choice_to_openai(
                normalized_tool_choice
            )
            params["parallel_tool_calls"] = parallel_tool_calls
            if effective_schema is not None:
                params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "strict": True,
                        "schema": self._enforce_strict_schema(effective_schema),
                    },
                }
        return {key: value for key, value in params.items() if value is not None}

    def _map_tools_to_openai(
        self,
        tools: Optional[List[ToolSpec]],
    ) -> Optional[List[Dict[str, Any]]]:
        if not tools:
            return None
        mapped: List[Dict[str, Any]] = []
        for tool in tools:
            function_payload: Dict[str, Any] = {
                "name": tool.name,
                "parameters": tool.json_schema,
            }
            if tool.description:
                function_payload["description"] = tool.description
            mapped.append(
                {
                    "type": "function",
                    "function": function_payload,
                }
            )
        return mapped

    def _map_tool_choice_to_openai(self, tool_choice: Optional[str]) -> Any:
        if tool_choice is None:
            return None
        if tool_choice in ("auto", "none"):
            return tool_choice
        if tool_choice == "any":
            return "required"
        return {"type": "function", "function": {"name": tool_choice}}

    def _normalize_reasoning_level(self, level: str | int | None) -> str | None:
        if level is None:
            return "none" if self.is_reasoning else None
        if not self.is_reasoning and level not in ("none", 0):
            warning_message = (
                f"Model '{self.model}' does not support reasoning вЂ” reasoning disabled."
            )
            warnings.warn(warning_message, UserWarning)
            logger.info(warning_message)
            return None
        if isinstance(level, bool):
            raise ValueError("Invalid type for level: bool is not accepted")
        if isinstance(level, str):
            if level in self.reasoning_levels:
                return level
            raise ValueError(
                f"Unknown reasoning level key: {level!r}. "
                f"Valid keys: {list(self.reasoning_levels.keys())}"
            )
        if isinstance(level, int):
            for key, val in self.reasoning_levels.items():
                if level <= val:
                    return key
            return list(self.reasoning_levels.keys())[-1]
        raise ValueError(
            "Invalid type for level: expected int or str, "
            f"got {type(level).__name__!r}"
        )

    def _map_tools_to_openai_responses(
        self,
        tools: Optional[List[ToolSpec]],
    ) -> Optional[List[Dict[str, Any]]]:
        if not tools:
            return None

        result: List[Dict[str, Any]] = []
        for tool in tools:
            result.append(
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.json_schema,
                }
            )
        return result

    def _map_tool_choice_to_openai_responses(self, tool_choice: Any) -> Any:
        if tool_choice is None:
            return None
        if tool_choice == "auto":
            return "auto"
        if tool_choice == "none":
            return "none"
        if tool_choice == "any":
            return "required"
        if isinstance(tool_choice, str):
            return {
                "type": "function",
                "name": tool_choice,
            }
        return tool_choice


