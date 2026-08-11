"""Google request construction and response normalization helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ...models.messages.chat_message import Messages
from ...models.responses.chat_response import ChatResponse
from ...models.tools import ToolSpec

class _GooglePayloadMixin:
    """Build Gemini payloads while keeping adapter options normalized."""

    def _build_chat_params(
        self,
        *,
        normalized_messages: Messages,
        max_tokens: Optional[int],
        temperature: float,
        top_p: float,
        reasoning_level: Optional[str | int],
        timeout_s: Optional[float],
        tools: Optional[List[ToolSpec]],
        normalized_tool_choice: Optional[str],
        effective_schema: Optional[dict],
        capture_reasoning: bool,
    ) -> Dict[str, Any]:
        system_prompt, transformed_messages = normalized_messages.to_google()
        params: Dict[str, Any] = {
            "model": self.model,
            "timeout_s": timeout_s,
            "contents": transformed_messages,
            "generationConfig": self._build_generation_config(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                reasoning_level=reasoning_level,
                effective_schema=effective_schema,
                capture_reasoning=capture_reasoning,
            ),
        }
        if system_prompt:
            params["system_instruction"] = {"parts": [{"text": system_prompt}]}
        if tools:
            params["tools"] = [{
                "functionDeclarations": [
                    self._to_google_function_declaration(tool) for tool in tools
                ]
            }]
        tool_config = self._to_google_tool_config(normalized_tool_choice)
        if tool_config is not None:
            params["toolConfig"] = tool_config
        return {key: value for key, value in params.items() if value is not None}

    def _build_stream_payload(
        self,
        *,
        normalized_messages: Messages,
        max_tokens: Optional[int],
        temperature: float,
        top_p: float,
        reasoning_level: Optional[str | int],
        tools: Optional[List[ToolSpec]],
        normalized_tool_choice: Optional[str],
        effective_schema: Optional[dict],
        capture_reasoning: bool,
    ) -> Dict[str, Any]:
        system_prompt, transformed_messages = normalized_messages.to_google()
        payload: Dict[str, Any] = {
            "contents": transformed_messages,
            "generationConfig": self._build_generation_config(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                reasoning_level=reasoning_level,
                effective_schema=effective_schema,
                capture_reasoning=capture_reasoning,
            ),
        }
        if system_prompt:
            payload["system_instruction"] = {"parts": [{"text": system_prompt}]}
        if tools:
            payload["tools"] = [{
                "functionDeclarations": [
                    self._to_google_function_declaration(tool) for tool in tools
                ]
            }]
        tool_config = self._to_google_tool_config(normalized_tool_choice)
        if tool_config is not None:
            payload["toolConfig"] = tool_config
        return payload

    def _build_generation_config(
        self,
        *,
        max_tokens: Optional[int],
        temperature: float,
        top_p: float,
        reasoning_level: Optional[str | int],
        effective_schema: Optional[dict],
        capture_reasoning: bool,
    ) -> Dict[str, Any]:
        generation_config: Dict[str, Any] = {
            "maxOutputTokens": max_tokens,
            "temperature": temperature,
            "topP": top_p,
        }
        if effective_schema is not None:
            generation_config["responseMimeType"] = "application/json"
            generation_config["responseSchema"] = self._to_google_schema(effective_schema)
        thinking_config = self._build_thinking_config(
            reasoning_level=reasoning_level,
            capture_reasoning=capture_reasoning,
        )
        if thinking_config is not None:
            generation_config["thinkingConfig"] = thinking_config
        return generation_config

    def _build_thinking_config(
        self,
        *,
        reasoning_level: Optional[str | int],
        capture_reasoning: bool,
    ) -> Optional[Dict[str, Any]]:
        thinking_config: Dict[str, Any] = {}
        if reasoning_level is not None:
            provider_value = self._resolve_reasoning_level(
                reasoning_level
            ).provider_value
            if isinstance(provider_value, int):
                thinking_config["thinkingBudget"] = provider_value
                thinking_config["includeThoughts"] = False
            elif isinstance(provider_value, str):
                thinking_config["thinkingLevel"] = provider_value
                thinking_config["includeThoughts"] = False
        if capture_reasoning:
            thinking_config["includeThoughts"] = True
        return thinking_config or None

    @staticmethod
    def _parse_chat_response(
        response: dict,
        *,
        capture_reasoning: bool,
    ) -> ChatResponse:
        parser_kwargs = {"capture_reasoning": True} if capture_reasoning else {}
        return ChatResponse.from_google_response(response, **parser_kwargs)

    # Fields not supported by Google's responseSchema subset of JSON Schema.
    _GOOGLE_SCHEMA_UNSUPPORTED = frozenset(
        {"additionalProperties", "$schema", "$id", "$ref"}
    )

    def _to_google_schema(self, schema: dict) -> dict:
        """Convert JSON Schema to Google's format and strip unsupported fields."""
        schema = {
            key: value
            for key, value in schema.items()
            if key not in self._GOOGLE_SCHEMA_UNSUPPORTED
        }
        if "type" in schema and isinstance(schema["type"], str):
            schema["type"] = schema["type"].upper()
        if "properties" in schema:
            schema["properties"] = {
                key: self._to_google_schema(value) if isinstance(value, dict) else value
                for key, value in schema["properties"].items()
            }
        if "items" in schema and isinstance(schema["items"], dict):
            schema["items"] = self._to_google_schema(schema["items"])
        return schema

    def _to_google_function_declaration(self, tool: ToolSpec) -> Dict[str, Any]:
        declaration: Dict[str, Any] = {
            "name": tool.name,
            "parametersJsonSchema": tool.json_schema,
        }
        if tool.description:
            declaration["description"] = tool.description
        return declaration

    def _to_google_tool_config(
        self,
        tool_choice: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        if tool_choice is None:
            return None
        if tool_choice == "none":
            mode = "NONE"
            allowed_function_names = None
        elif tool_choice == "auto":
            mode = "AUTO"
            allowed_function_names = None
        elif tool_choice == "any":
            mode = "ANY"
            allowed_function_names = None
        else:
            mode = "ANY"
            allowed_function_names = [tool_choice]
        function_calling_config: Dict[str, Any] = {"mode": mode}
        if allowed_function_names:
            function_calling_config["allowedFunctionNames"] = allowed_function_names
        return {"functionCallingConfig": function_calling_config}

__all__ = ["_GooglePayloadMixin"]
