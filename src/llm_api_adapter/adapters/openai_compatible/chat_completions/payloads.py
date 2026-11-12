"""Shared Chat Completions request and response mapping."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ....models.messages.chat_message import Messages
from ....models.responses.chat_response import ChatResponse
from ....models.tools import ToolSpec
from ..common.usage import _apply_cache_usage


class _ChatCompletionsPayloadMixin:
    """Build and normalize the shared Chat Completions wire contract."""

    def _normalize_chat_completions_response(
        self,
        response: dict,
    ) -> ChatResponse:
        """Normalize a complete Chat Completions response before finalization."""
        chat_response = ChatResponse.from_openai_response(response)
        return _apply_cache_usage(
            chat_response,
            response,
            details_field="prompt_tokens_details",
        )

    def _validate_chat_completions_stream_event(self, event: Any) -> None:
        """Hook for organization-specific validation of Chat Completions events."""

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
