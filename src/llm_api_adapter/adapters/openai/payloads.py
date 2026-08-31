"""OpenAI request construction and response normalization helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ...models.messages.chat_message import Messages
from ...models.responses.chat_response import ChatResponse
from ...models.tools import ToolSpec
from ..structured_output import validate_core_portable_schema

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
                    "schema": self._to_openai_structured_output_schema(effective_schema),
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
                    "schema": self._to_openai_structured_output_schema(effective_schema),
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
            "reasoning_effort": self._resolve_openai_reasoning_effort(
                reasoning_level
            ),
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
                        "schema": self._to_openai_structured_output_schema(effective_schema),
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
                        "schema": self._to_openai_structured_output_schema(effective_schema),
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

    @staticmethod
    def _to_openai_structured_output_schema(schema: dict) -> dict:
        """Validate the Core profile without changing schema semantics."""
        return validate_core_portable_schema(schema, provider="openai")

    def _map_tool_choice_to_openai(self, tool_choice: Optional[str]) -> Any:
        if tool_choice is None:
            return None
        if tool_choice in ("auto", "none"):
            return tool_choice
        if tool_choice == "any":
            return "required"
        return {"type": "function", "function": {"name": tool_choice}}

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


