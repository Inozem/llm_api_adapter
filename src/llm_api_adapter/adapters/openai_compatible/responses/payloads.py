"""Pure Responses request and response mapping helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ....models.responses.chat_response import ChatResponse
from ....models.tools import ToolSpec
from ..common.usage import _apply_cache_usage


class _ResponsesPayloadMixin:
    """Map already-selected common data to the Responses wire format."""

    def _build_responses_wire_params(
        self,
        *,
        input_items: List[Dict[str, Any]],
        instructions: Optional[str],
        tools: Optional[List[ToolSpec]],
        normalized_tool_choice: Optional[str],
        effective_schema: Optional[dict],
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "input": input_items,
            "tools": self._map_tools_to_openai_responses(tools),
            "tool_choice": self._map_tool_choice_to_openai_responses(
                normalized_tool_choice
            ),
        }
        if instructions is not None:
            params["instructions"] = instructions
        if effective_schema is not None:
            params["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "response",
                    "strict": True,
                    "schema": self._to_openai_structured_output_schema(
                        effective_schema
                    ),
                }
            }
        return params

    def _normalize_responses_response(
        self,
        response: dict,
        *,
        capture_reasoning: bool,
    ) -> ChatResponse:
        parser_kwargs = {"capture_reasoning": True} if capture_reasoning else {}
        chat_response = ChatResponse.from_openai_responses_response(
            response,
            **parser_kwargs,
        )
        return _apply_cache_usage(
            chat_response,
            response,
            details_field="input_tokens_details",
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
