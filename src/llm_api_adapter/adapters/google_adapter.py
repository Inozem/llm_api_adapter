from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, Iterator, List, Mapping, Optional
import warnings

from ..adapters.base_adapter import LLMAdapterBase, OnChunk, OnDelta, OnDone, OnToolCall
from ..errors.llm_api_error import LLMAPIError
from ..llms.google.sync_client import GeminiSyncClient
from ..llms.streaming import StreamChunkBuffer
from ..models.messages.chat_message import Message, Messages
from ..models.responses.chat_response import ChatResponse
from ..models.tools import ToolSpec

logger = logging.getLogger(__name__)


@dataclass(repr=False)
class GoogleAdapter(LLMAdapterBase):
    company: str = "google"

    def chat(
        self,
        messages: List[Message] | Messages,
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        reasoning_level: Optional[str | int] = None,
        timeout_s: Optional[float] = None,
        *,
        tools: Optional[List[ToolSpec]] = None,
        tool_choice: Optional[str | dict] = None,
        parallel_tool_calls: Optional[bool] = None,
        previous_response: Optional[ChatResponse] = None,
        json_schema: Optional[dict] = None,
        response_model: Optional[Any] = None,
    ) -> ChatResponse:
        temperature = self._validate_parameter(
            name="temperature",
            value=temperature,
            min_value=0,
            max_value=2,
        )
        top_p = self._validate_parameter(
            name="top_p",
            value=top_p,
            min_value=0,
            max_value=1,
        )
        try:
            self._validate_tools(tools)
            effective_schema = self._resolve_json_schema(json_schema, response_model, tools)
            validated_tools = tools
            normalized_tool_choice = self._normalize_tool_choice(
                tool_choice,
                validated_tools,
            )
            normalized_messages = self._normalize_messages(messages)
            _ = previous_response
            system_prompt, transformed_messages = normalized_messages.to_google()
            generation_config: Dict[str, Any] = {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
                "topP": top_p,
            }
            if effective_schema is not None:
                generation_config["responseMimeType"] = "application/json"
                generation_config["responseSchema"] = self._to_google_schema(effective_schema)
            if reasoning_level:
                normalized_reasoning_level = self._normalize_reasoning_level(
                    reasoning_level
                )
                if normalized_reasoning_level is not None:
                    generation_config["thinkingConfig"] = {
                        "thinkingBudget": normalized_reasoning_level,
                        "includeThoughts": False,
                    }
            payload: Dict[str, Any] = {
                "contents": transformed_messages,
                "generationConfig": generation_config,
            }
            if system_prompt:
                payload["system_instruction"] = {
                    "parts": [{"text": system_prompt}]
                }
            if validated_tools:
                payload["tools"] = [
                    {
                        "functionDeclarations": [
                            self._to_google_function_declaration(tool)
                            for tool in validated_tools
                        ]
                    }
                ]
            tool_config = self._to_google_tool_config(normalized_tool_choice)
            if tool_config is not None:
                payload["toolConfig"] = tool_config
            _ = parallel_tool_calls
            client = GeminiSyncClient(self.api_key)
            response_json = client.chat_completion(
                model=self.model,
                timeout_s=timeout_s,
                **payload,
            )
            chat_response = ChatResponse.from_google_response(response_json)
            chat_response.parsed_json = self._parse_json_response(chat_response.content, effective_schema)
            chat_response.parsed_model = self._parse_response_model(chat_response.parsed_json, response_model)
            if self.pricing:
                chat_response.apply_pricing(
                    price_input_per_token=self.pricing.in_per_token,
                    price_output_per_token=self.pricing.out_per_token,
                    currency=self.pricing.currency,
                )
            return chat_response
        except LLMAPIError as e:
            self.handle_error(e)
        except Exception as e:
            error_message = getattr(e, "text", None) or str(e)
            self.handle_error(error=e, error_message=error_message)

    def stream_chat(
        self,
        messages: List[Message] | Messages,
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        reasoning_level: Optional[str | int] = None,
        timeout_s: Optional[float] = None,
        *,
        tools: Optional[List[ToolSpec]] = None,
        tool_choice: Optional[str | dict] = None,
        parallel_tool_calls: Optional[bool] = None,
        previous_response: Optional[ChatResponse] = None,
        json_schema: Optional[dict] = None,
        response_model: Optional[Any] = None,
        on_delta: Optional[OnDelta] = None,
        on_tool_call: Optional[OnToolCall] = None,
        on_done: Optional[OnDone] = None,
        buffer_chars: Optional[int] = None,
        on_chunk: Optional[OnChunk] = None,
    ) -> Iterator[str]:
        temperature = self._validate_parameter("temperature", temperature, 0, 2)
        top_p = self._validate_parameter("top_p", top_p, 0, 1)
        self._validate_tools(tools)
        effective_schema = self._resolve_json_schema(json_schema, response_model, tools)
        normalized_tool_choice = self._normalize_tool_choice(tool_choice, tools)
        normalized_messages = self._normalize_messages(messages)
        _ = previous_response
        payload = self._build_stream_payload(
            normalized_messages=normalized_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            reasoning_level=reasoning_level,
            tools=tools,
            normalized_tool_choice=normalized_tool_choice,
            effective_schema=effective_schema,
        )
        _ = parallel_tool_calls
        client = GeminiSyncClient(self.api_key)
        events = client.stream(model=self.model, timeout_s=timeout_s, **payload)
        yield from self._consume_stream(
            events,
            effective_schema,
            response_model,
            on_delta,
            on_tool_call,
            on_done,
            buffer_chars,
            on_chunk,
        )

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
    ) -> Dict[str, Any]:
        system_prompt, transformed_messages = normalized_messages.to_google()
        generation_config: Dict[str, Any] = {
            "maxOutputTokens": max_tokens,
            "temperature": temperature,
            "topP": top_p,
        }
        if effective_schema is not None:
            generation_config["responseMimeType"] = "application/json"
            generation_config["responseSchema"] = self._to_google_schema(effective_schema)
        if reasoning_level:
            normalized_reasoning_level = self._normalize_reasoning_level(reasoning_level)
            if normalized_reasoning_level is not None:
                generation_config["thinkingConfig"] = {
                    "thinkingBudget": normalized_reasoning_level,
                    "includeThoughts": False,
                }
        payload: Dict[str, Any] = {
            "contents": transformed_messages,
            "generationConfig": generation_config,
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

    def _consume_stream(
        self,
        events: Iterator[Any],
        effective_schema: Optional[dict],
        response_model: Optional[Any],
        on_delta: Optional[OnDelta],
        on_tool_call: Optional[OnToolCall],
        on_done: Optional[OnDone],
        buffer_chars: Optional[int],
        on_chunk: Optional[OnChunk],
    ) -> Iterator[str]:
        text_parts: List[str] = []
        function_parts: List[Dict[str, Any]] = []
        finish_reason: Optional[str] = None
        usage_metadata: Dict[str, Any] = {}
        response_metadata: Dict[str, Any] = {}
        chunk_buffer = StreamChunkBuffer(buffer_chars)

        for event in self._iter_provider_stream_events(events):
            chunk = event.data if isinstance(event.data, Mapping) else {}
            for field in ("modelVersion", "responseId", "promptFeedback"):
                if field in chunk:
                    response_metadata[field] = chunk[field]
            chunk_usage = chunk.get("usageMetadata")
            if isinstance(chunk_usage, Mapping):
                usage_metadata.update(chunk_usage)
            candidates = chunk.get("candidates")
            if not isinstance(candidates, list) or not candidates:
                continue
            candidate = candidates[0]
            if not isinstance(candidate, Mapping):
                continue
            if candidate.get("finishReason") is not None:
                finish_reason = str(candidate["finishReason"])
            content = candidate.get("content")
            parts = content.get("parts") if isinstance(content, Mapping) else []
            if not isinstance(parts, list):
                continue
            for part in parts:
                if not isinstance(part, Mapping):
                    continue
                text = part.get("text")
                if (
                    isinstance(text, str)
                    and text
                    and not part.get("thought", False)
                ):
                    text_parts.append(text)
                    yield from self._emit_stream_chunks(
                        chunk_buffer.add(text),
                        on_chunk,
                        on_delta,
                    )
                if isinstance(part.get("functionCall"), Mapping):
                    function_parts.append(dict(part))

        final_parts: List[Dict[str, Any]] = []
        if text_parts:
            final_parts.append({"text": "".join(text_parts)})
        final_parts.extend(function_parts)
        final_candidate: Dict[str, Any] = {"content": {"parts": final_parts}}
        if finish_reason is not None:
            final_candidate["finishReason"] = finish_reason
        final_response: Dict[str, Any] = {
            **response_metadata,
            "candidates": [final_candidate],
        }
        if usage_metadata:
            final_response["usageMetadata"] = usage_metadata
        chat_response = ChatResponse.from_google_response(final_response)
        self._prepare_stream_response(
            chat_response,
            effective_schema,
            response_model,
        )
        yield from self._emit_stream_chunks(
            chunk_buffer.flush(),
            on_chunk,
            on_delta,
        )
        self._invoke_stream_completion_callbacks(
            chat_response,
            on_tool_call,
            on_done,
        )

    # Fields not supported by Google's responseSchema subset of JSON Schema.
    _GOOGLE_SCHEMA_UNSUPPORTED = frozenset({"additionalProperties", "$schema", "$id", "$ref"})

    def _to_google_schema(self, schema: dict) -> dict:
        """Convert standard JSON Schema to Google's format (type uppercase, unsupported fields stripped)."""
        schema = {k: v for k, v in schema.items() if k not in self._GOOGLE_SCHEMA_UNSUPPORTED}
        if "type" in schema and isinstance(schema["type"], str):
            schema["type"] = schema["type"].upper()
        if "properties" in schema:
            schema["properties"] = {
                k: self._to_google_schema(v) if isinstance(v, dict) else v
                for k, v in schema["properties"].items()
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

    def _normalize_reasoning_level(self, level: str | int | None) -> int | None:
        minimum_level = 0
        normalized_level: Optional[int] = None
        if level is not None and not self.is_reasoning:
            warning_message = (
                f"Model '{self.model}' does not support reasoning — reasoning disabled."
            )
            warnings.warn(warning_message, UserWarning)
            logger.info(warning_message)
            return None
        if isinstance(level, bool):
            raise ValueError("Invalid type for level: bool is not accepted")
        if isinstance(level, str):
            if level in self.reasoning_levels:
                normalized_level = self.reasoning_levels[level]
            else:
                raise ValueError(
                    f"Unknown reasoning level key: {level!r}. "
                    f"Valid keys: {list(self.reasoning_levels.keys())}"
                )
        if isinstance(level, int):
            normalized_level = level
        if normalized_level is not None:
            if normalized_level >= minimum_level:
                return normalized_level
            return minimum_level
        raise ValueError(
            "Invalid type for level: expected int or str, "
            f"got {type(level).__name__!r}"
        )
