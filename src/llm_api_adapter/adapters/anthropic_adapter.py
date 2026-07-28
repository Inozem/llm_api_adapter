from __future__ import annotations

from dataclasses import dataclass
import json
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
from ..errors.llm_api_error import InvalidToolArgumentsError, LLMAPIError
from ..errors.config_errors import LLMReasoningLevelError
from ..llms.anthropic.sync_client import ClaudeSyncClient
from ..llms.streaming import (
    StreamChunkBuffer,
    StreamReasoningCollector,
    StreamUsageTracker,
)
from ..models.messages.chat_message import Message, Messages
from ..models.responses.chat_response import ChatResponse, Usage
from ..models.tools.tool_spec import ToolSpec

logger = logging.getLogger(__name__)


@dataclass
class _AnthropicStreamState:
    message_data: Dict[str, Any]
    content_blocks: Dict[int, Dict[str, Any]]
    input_json_fragments: Dict[int, List[str]]
    usage: Dict[str, Any]
    message_delta: Dict[str, Any]
    chunk_buffer: StreamChunkBuffer
    usage_tracker: StreamUsageTracker
    reasoning_collector: Optional[StreamReasoningCollector]
    reasoning_response: Optional[ChatResponse]


@dataclass(repr=False)
class AnthropicAdapter(LLMAdapterBase):
    company: str = "anthropic"

    def chat(
        self,
        messages: List[Message] | Messages,
        max_tokens: int,
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
        temperature, top_p = self._validate_sampling_parameters(
            temperature,
            top_p,
        )
        try:
            request_context = self._prepare_chat_request(
                messages,
                tools,
                tool_choice,
                json_schema,
                response_model,
            )
            effective_schema = request_context.effective_schema
            normalized_tool_choice = request_context.normalized_tool_choice
            normalized_messages = request_context.normalized_messages
            params = self._build_chat_params(
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
            _ = previous_response
            client = ClaudeSyncClient(api_key=self.api_key)
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
                    "schema": self._enforce_strict_schema(effective_schema),
                }
            }
        if reasoning_level:
            normalized_reasoning_level = self._normalize_reasoning_level(reasoning_level)
            if normalized_reasoning_level:
                if not self.is_adaptive_thinking:
                    self.validate_reasoning_and_tokens(
                        max_tokens=max_tokens,
                        reasoning_level=reasoning_level,
                        normalized_reasoning_level=normalized_reasoning_level,
                    )
                params["budget_tokens"] = normalized_reasoning_level
            if self.is_reasoning:
                effort = self._reasoning_level_to_effort(reasoning_level)
                if effort:
                    params["effort"] = effort
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

    def stream_chat(
        self,
        messages: List[Message] | Messages,
        max_tokens: int,
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
        temperature, top_p = self._validate_sampling_parameters(
            temperature,
            top_p,
        )
        request_context = self._prepare_chat_request(
            messages,
            tools,
            tool_choice,
            json_schema,
            response_model,
        )
        effective_schema = request_context.effective_schema
        normalized_tool_choice = request_context.normalized_tool_choice
        normalized_messages = request_context.normalized_messages
        params = self._build_stream_params(
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
        _ = previous_response
        client = ClaudeSyncClient(api_key=self.api_key)
        events = client.stream(**params)
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
                    "schema": self._enforce_strict_schema(effective_schema),
                }
            }
        if reasoning_level:
            normalized_reasoning_level = self._normalize_reasoning_level(reasoning_level)
            if normalized_reasoning_level:
                if not self.is_adaptive_thinking:
                    self.validate_reasoning_and_tokens(
                        max_tokens=max_tokens,
                        reasoning_level=reasoning_level,
                        normalized_reasoning_level=normalized_reasoning_level,
                    )
                params["budget_tokens"] = normalized_reasoning_level
                if self.is_reasoning:
                    effort = self._reasoning_level_to_effort(reasoning_level)
                    if effort:
                        params["effort"] = effort
        if capture_reasoning:
            params["capture_reasoning"] = True
        return {key: value for key, value in params.items() if value is not None}

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
        state = _AnthropicStreamState(
            message_data={"model": self.model, "content": []},
            content_blocks={},
            input_json_fragments={},
            usage={},
            message_delta={},
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
        state: _AnthropicStreamState,
        *,
        on_chunk: Optional[OnChunk],
        on_delta: Optional[OnDelta],
        on_reasoning: Optional[OnReasoning],
    ) -> Iterator[str]:
        payload = event.data if isinstance(event.data, Mapping) else {}
        event_type = event.event or payload.get("type")
        if event_type == "message_start":
            state.message_data = self._handle_message_start(
                payload,
                state.usage,
                state.message_data,
            )
            state.usage_tracker.record(
                state.chunk_buffer,
                self._normalize_stream_usage(state.usage),
            )
        elif event_type == "content_block_start":
            self._start_content_block(payload, state.content_blocks)
        elif event_type == "content_block_delta":
            yield from self._consume_content_block_delta(
                payload,
                state.content_blocks,
                state.input_json_fragments,
                state.chunk_buffer,
                on_chunk,
                on_delta,
                state.reasoning_collector,
                state.reasoning_response,
                on_reasoning,
            )
        elif event_type == "content_block_stop":
            self._finalize_content_block(
                payload,
                state.content_blocks,
                state.input_json_fragments,
            )
        elif event_type == "message_delta":
            self._handle_message_delta(
                payload,
                state.message_delta,
                state.usage,
            )
            state.usage_tracker.record(
                state.chunk_buffer,
                self._normalize_stream_usage(state.usage),
            )

    def _finalize_stream_response(
        self,
        state: _AnthropicStreamState,
        *,
        capture_reasoning: bool,
        effective_schema: Optional[dict],
        response_model: Optional[Any],
    ) -> ChatResponse:
        parser_kwargs = {"capture_reasoning": True} if capture_reasoning else {}
        chat_response = ChatResponse.from_anthropic_response(
            self._build_stream_response(
                state.message_data,
                state.content_blocks,
                state.message_delta,
                state.usage,
            ),
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

    def _handle_message_start(
        self,
        payload: Mapping[str, Any],
        usage: Dict[str, Any],
        message_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        message = payload.get("message")
        if not isinstance(message, Mapping):
            return message_data
        initial_usage = message.get("usage")
        if isinstance(initial_usage, Mapping):
            usage.update(initial_usage)
        return dict(message)

    def _start_content_block(
        self,
        payload: Mapping[str, Any],
        content_blocks: Dict[int, Dict[str, Any]],
    ) -> None:
        index = payload.get("index")
        block = payload.get("content_block")
        if not isinstance(index, int) or not isinstance(block, Mapping):
            return
        content_blocks[index] = dict(block)
        if block.get("type") == "text":
            content_blocks[index]["text"] = str(block.get("text") or "")
        elif block.get("type") == "thinking":
            content_blocks[index]["thinking"] = str(block.get("thinking") or "")
        elif block.get("type") == "tool_use":
            current_input = block.get("input")
            content_blocks[index]["input"] = (
                dict(current_input) if isinstance(current_input, Mapping) else {}
            )

    def _consume_content_block_delta(
        self,
        payload: Mapping[str, Any],
        content_blocks: Dict[int, Dict[str, Any]],
        input_json_fragments: Dict[int, List[str]],
        chunk_buffer: StreamChunkBuffer,
        on_chunk: Optional[OnChunk],
        on_delta: Optional[OnDelta],
        reasoning_collector: Optional[StreamReasoningCollector],
        reasoning_response: Optional[ChatResponse],
        on_reasoning: Optional[OnReasoning],
    ) -> Iterator[str]:
        index = payload.get("index")
        delta = payload.get("delta")
        if not isinstance(index, int) or not isinstance(delta, Mapping):
            return
        block = content_blocks.get(index)
        if block is None:
            return
        if delta.get("type") == "text_delta":
            text = delta.get("text")
            if isinstance(text, str) and text:
                block["text"] = f"{block.get('text', '')}{text}"
                yield from self._emit_stream_chunks(
                    chunk_buffer.add(text),
                    on_chunk,
                    on_delta,
                )
        elif delta.get("type") == "thinking_delta":
            thinking_text = delta.get("thinking")
            if isinstance(thinking_text, str) and thinking_text:
                block["thinking"] = f"{block.get('thinking', '')}{thinking_text}"
                if reasoning_collector is not None and reasoning_response is not None:
                    self._record_reasoning_event(
                        reasoning_response,
                        reasoning_collector,
                        thinking_text,
                        capture_reasoning=True,
                        kind="summary",
                        on_reasoning=on_reasoning,
                    )
        elif delta.get("type") == "input_json_delta":
            partial_json = delta.get("partial_json")
            if isinstance(partial_json, str):
                input_json_fragments.setdefault(index, []).append(partial_json)

    def _finalize_content_block(
        self,
        payload: Mapping[str, Any],
        content_blocks: Dict[int, Dict[str, Any]],
        input_json_fragments: Dict[int, List[str]],
    ) -> None:
        index = payload.get("index")
        block = content_blocks.get(index) if isinstance(index, int) else None
        if not block or block.get("type") != "tool_use":
            return
        raw_input = "".join(input_json_fragments.get(index, []))
        if not raw_input:
            return
        try:
            parsed_input = json.loads(raw_input)
        except json.JSONDecodeError as error:
            raise InvalidToolArgumentsError(
                detail=(
                    "Anthropic tool input JSON parse failed "
                    f"for tool={block.get('name')!r}: {error}"
                )
            ) from error
        if not isinstance(parsed_input, dict):
            raise InvalidToolArgumentsError(
                detail=(
                    "Anthropic tool input must decode to dict "
                    f"for tool={block.get('name')!r}"
                )
            )
        block["input"] = parsed_input

    def _handle_message_delta(
        self,
        payload: Mapping[str, Any],
        message_delta: Dict[str, Any],
        usage: Dict[str, Any],
    ) -> None:
        delta = payload.get("delta")
        if isinstance(delta, Mapping):
            message_delta.update(delta)
        delta_usage = payload.get("usage")
        if isinstance(delta_usage, Mapping):
            usage.update(delta_usage)

    @staticmethod
    def _normalize_stream_usage(raw_usage: Mapping[str, Any]) -> Optional[Usage]:
        input_tokens = AnthropicAdapter._token_count(raw_usage.get("input_tokens"))
        output_tokens = AnthropicAdapter._token_count(raw_usage.get("output_tokens"))
        if input_tokens is None and output_tokens is None:
            return None
        return Usage(
            input_tokens=input_tokens or 0,
            output_tokens=output_tokens or 0,
            total_tokens=(input_tokens or 0) + (output_tokens or 0),
        )

    @staticmethod
    def _token_count(value: Any) -> Optional[int]:
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
        return None

    def _build_stream_response(
        self,
        message_data: Dict[str, Any],
        content_blocks: Dict[int, Dict[str, Any]],
        message_delta: Dict[str, Any],
        usage: Dict[str, Any],
    ) -> Dict[str, Any]:
        final_response = dict(message_data)
        final_response["content"] = [
            content_blocks[index] for index in sorted(content_blocks)
        ]
        final_response.update(message_delta)
        if usage:
            final_response["usage"] = usage
        return final_response

    def _to_anthropic_tool(self, tool: ToolSpec) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "name": tool.name,
            "input_schema": tool.json_schema,
        }
        if tool.description:
            payload["description"] = tool.description
        return payload

    def _to_anthropic_tool_choice(
        self,
        tool_choice: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        if tool_choice is None:
            return None
        if tool_choice == "none":
            return None
        if tool_choice in ("auto", "any"):
            return {"type": tool_choice}
        return {"type": "tool", "name": tool_choice}

    def _validate_not_bool(self, level) -> None:
        if isinstance(level, bool):
            raise ValueError("Invalid type for level: bool is not accepted")

    def _validate_reasoning_str(self, level: str) -> None:
        if level not in self.reasoning_levels:
            raise ValueError(
                f"Unknown reasoning level key: {level!r}. "
                f"Valid keys: {list(self.reasoning_levels.keys())}"
            )

    def _resolve_reasoning_int(self, level: int) -> int:
        if isinstance(level, int):
            return level
        raise ValueError(
            "Invalid type for level: expected int or str, "
            f"got {type(level).__name__!r}"
        )

    def _reasoning_level_to_effort(self, level: str | int) -> str | None:
        self._validate_not_bool(level)
        if isinstance(level, str):
            self._validate_reasoning_str(level)
            return None if level == "none" else level
        numeric = self._resolve_reasoning_int(level)
        for key, threshold in self.reasoning_levels.items():
            if threshold > 0 and numeric <= threshold:
                return key
        return list(self.reasoning_levels)[-1]

    def _normalize_reasoning_level(self, level: str | int) -> int | None:
        minimum_level = 1024
        if not self.is_reasoning:
            warning_message = (
                f"Model '{self.model}' does not support reasoning — reasoning disabled."
            )
            warnings.warn(warning_message, UserWarning)
            logger.info(warning_message)
            return None
        self._validate_not_bool(level)
        if isinstance(level, str):
            self._validate_reasoning_str(level)
            numeric = self.reasoning_levels[level]
        else:
            numeric = self._resolve_reasoning_int(level)
        if numeric >= minimum_level:
            return numeric
        warning_message = (
            f"Reasoning level '{level}' is below the minimum supported value "
            f"{minimum_level}; using {minimum_level} instead."
        )
        warnings.warn(warning_message, UserWarning)
        logger.info(warning_message)
        return minimum_level

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
