from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, Iterator, List, Mapping, Optional
import warnings

from ..adapters.base_adapter import (
    LLMAdapterBase,
    OnChunk,
    OnDelta,
    OnDone,
    OnReasoning,
    OnToolCall,
)
from ..errors.llm_api_error import LLMAPIError
from ..llms.google.sync_client import GeminiSyncClient
from ..llms.streaming import (
    StreamChunkBuffer,
    StreamReasoningCollector,
    StreamUsageTracker,
)
from ..models.messages.chat_message import Message, Messages
from ..models.responses.chat_response import ChatResponse, Usage
from ..models.tools import ToolSpec

logger = logging.getLogger(__name__)


@dataclass
class _GoogleStreamState:
    chunk_buffer: StreamChunkBuffer
    usage_tracker: StreamUsageTracker
    text_parts: List[str] = field(default_factory=list)
    function_parts: List[Dict[str, Any]] = field(default_factory=list)
    finish_reason: Optional[str] = None
    usage_metadata: Dict[str, Any] = field(default_factory=dict)
    response_metadata: Dict[str, Any] = field(default_factory=dict)
    reasoning_collector: Optional[StreamReasoningCollector] = None
    reasoning_response: Optional[ChatResponse] = None


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
        capture_reasoning: bool = False,
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
            normalized_tool_choice = self._normalize_tool_choice(
                tool_choice,
                tools,
            )
            normalized_messages = self._normalize_messages(messages)
            params = self._build_chat_params(
                normalized_messages=normalized_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                reasoning_level=reasoning_level,
                timeout_s=timeout_s,
                tools=tools,
                normalized_tool_choice=normalized_tool_choice,
                effective_schema=effective_schema,
                capture_reasoning=capture_reasoning,
            )
            _ = previous_response
            _ = parallel_tool_calls
            client = GeminiSyncClient(self.api_key)
            response = client.chat_completion(**params)
            chat_response = self._parse_chat_response(
                response,
                capture_reasoning=capture_reasoning,
            )
            return self._finalize_chat_response(
                chat_response,
                effective_schema=effective_schema,
                response_model=response_model,
            )
        except LLMAPIError as e:
            self.handle_error(e)
        except Exception as e:
            error_message = getattr(e, "text", None) or str(e)
            self.handle_error(error=e, error_message=error_message)

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

    @staticmethod
    def _parse_chat_response(
        response: dict,
        *,
        capture_reasoning: bool,
    ) -> ChatResponse:
        parser_kwargs = {"capture_reasoning": True} if capture_reasoning else {}
        return ChatResponse.from_google_response(response, **parser_kwargs)

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
        capture_reasoning: bool = False,
        on_reasoning: Optional[OnReasoning] = None,
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
            capture_reasoning=capture_reasoning,
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
            capture_reasoning,
            on_reasoning,
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
        capture_reasoning: bool,
    ) -> Dict[str, Any]:
        system_prompt, transformed_messages = normalized_messages.to_google()
        generation_config = self._build_generation_config(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            reasoning_level=reasoning_level,
            effective_schema=effective_schema,
            capture_reasoning=capture_reasoning,
        )
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
        capture_reasoning: bool,
        on_reasoning: Optional[OnReasoning],
    ) -> Iterator[str]:
        state = _GoogleStreamState(
            chunk_buffer=StreamChunkBuffer(buffer_chars),
            usage_tracker=StreamUsageTracker(),
            reasoning_collector=(
                StreamReasoningCollector() if capture_reasoning else None
            ),
            reasoning_response=ChatResponse() if capture_reasoning else None,
        )

        for event in self._iter_provider_stream_events(events):
            yield from self._consume_stream_event(
                event,
                state,
                on_chunk=on_chunk,
                on_delta=on_delta,
                on_reasoning=on_reasoning,
            )

        chat_response = self._finalize_stream_response(
            state,
            capture_reasoning=capture_reasoning,
            effective_schema=effective_schema,
            response_model=response_model,
        )
        yield from self._emit_stream_chunks(
            state.chunk_buffer.flush(),
            on_chunk,
            on_delta,
        )
        self._invoke_stream_completion_callbacks(
            chat_response,
            on_tool_call,
            on_done,
        )

    def _consume_stream_event(
        self,
        event: Any,
        state: _GoogleStreamState,
        *,
        on_chunk: Optional[OnChunk],
        on_delta: Optional[OnDelta],
        on_reasoning: Optional[OnReasoning],
    ) -> Iterator[str]:
        chunk = event.data if isinstance(event.data, Mapping) else {}
        self._update_stream_state_from_chunk(chunk, state)

        candidates = chunk.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            return
        candidate = candidates[0]
        if not isinstance(candidate, Mapping):
            return
        if candidate.get("finishReason") is not None:
            state.finish_reason = str(candidate["finishReason"])
        content = candidate.get("content")
        parts = content.get("parts") if isinstance(content, Mapping) else []
        if not isinstance(parts, list):
            return
        for part in parts:
            yield from self._consume_stream_part(
                part,
                state,
                on_chunk=on_chunk,
                on_delta=on_delta,
                on_reasoning=on_reasoning,
            )

    def _update_stream_state_from_chunk(
        self,
        chunk: Mapping[str, Any],
        state: _GoogleStreamState,
    ) -> None:
        for field in ("modelVersion", "responseId", "promptFeedback"):
            if field in chunk:
                state.response_metadata[field] = chunk[field]
        chunk_usage = chunk.get("usageMetadata")
        if isinstance(chunk_usage, Mapping):
            state.usage_metadata.update(chunk_usage)
            state.usage_tracker.record(
                state.chunk_buffer,
                self._normalize_stream_usage(state.usage_metadata),
            )

    def _consume_stream_part(
        self,
        part: Any,
        state: _GoogleStreamState,
        *,
        on_chunk: Optional[OnChunk],
        on_delta: Optional[OnDelta],
        on_reasoning: Optional[OnReasoning],
    ) -> Iterator[str]:
        if not isinstance(part, Mapping):
            return
        text = part.get("text")
        if part.get("thought"):
            if (
                state.reasoning_collector is not None
                and state.reasoning_response is not None
                and isinstance(text, str)
                and text
            ):
                self._record_reasoning_event(
                    state.reasoning_response,
                    state.reasoning_collector,
                    text,
                    capture_reasoning=True,
                    kind="summary",
                    on_reasoning=on_reasoning,
                )
        elif isinstance(text, str) and text:
            state.text_parts.append(text)
            yield from self._emit_stream_chunks(
                state.chunk_buffer.add(text),
                on_chunk,
                on_delta,
            )
        if isinstance(part.get("functionCall"), Mapping):
            state.function_parts.append(dict(part))

    def _finalize_stream_response(
        self,
        state: _GoogleStreamState,
        *,
        capture_reasoning: bool,
        effective_schema: Optional[dict],
        response_model: Optional[Any],
    ) -> ChatResponse:
        final_response = self._build_stream_response(state)
        parser_kwargs = {"capture_reasoning": True} if capture_reasoning else {}
        chat_response = ChatResponse.from_google_response(
            final_response,
            **parser_kwargs,
        )
        if state.reasoning_collector is not None:
            streamed_reasoning_events = state.reasoning_collector.snapshot()
            if streamed_reasoning_events:
                chat_response.reasoning_events = streamed_reasoning_events
        self._prepare_stream_response(
            chat_response,
            effective_schema,
            response_model,
        )
        return chat_response

    @staticmethod
    def _build_stream_response(state: _GoogleStreamState) -> Dict[str, Any]:
        final_parts: List[Dict[str, Any]] = []
        if state.text_parts:
            final_parts.append({"text": "".join(state.text_parts)})
        final_parts.extend(state.function_parts)
        final_candidate: Dict[str, Any] = {"content": {"parts": final_parts}}
        if state.finish_reason is not None:
            final_candidate["finishReason"] = state.finish_reason
        final_response: Dict[str, Any] = {
            **state.response_metadata,
            "candidates": [final_candidate],
        }
        if state.usage_metadata:
            final_response["usageMetadata"] = state.usage_metadata
        return final_response

    def _build_thinking_config(
        self,
        *,
        reasoning_level: Optional[str | int],
        capture_reasoning: bool,
    ) -> Optional[Dict[str, Any]]:
        thinking_config: Dict[str, Any] = {}
        if reasoning_level:
            normalized_reasoning_level = self._normalize_reasoning_level(
                reasoning_level
            )
            if normalized_reasoning_level is not None:
                thinking_config["thinkingBudget"] = normalized_reasoning_level
                thinking_config["includeThoughts"] = False
        if capture_reasoning:
            thinking_config["includeThoughts"] = True
        return thinking_config or None

    @staticmethod
    def _normalize_stream_usage(raw_usage: Mapping[str, Any]) -> Optional[Usage]:
        input_tokens = GoogleAdapter._token_count(raw_usage.get("promptTokenCount"))
        candidate_tokens = GoogleAdapter._token_count(
            raw_usage.get("candidatesTokenCount")
        )
        thoughts_tokens = GoogleAdapter._token_count(raw_usage.get("thoughtsTokenCount"))
        total_tokens = GoogleAdapter._token_count(raw_usage.get("totalTokenCount"))
        if (
            input_tokens is None
            and candidate_tokens is None
            and thoughts_tokens is None
            and total_tokens is None
        ):
            return None
        return Usage(
            input_tokens=input_tokens or 0,
            output_tokens=(candidate_tokens or 0) + (thoughts_tokens or 0),
            total_tokens=total_tokens or 0,
        )

    @staticmethod
    def _token_count(value: Any) -> Optional[int]:
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
        return None

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
