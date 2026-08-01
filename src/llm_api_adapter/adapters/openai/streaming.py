"""OpenAI Chat Completions and Responses streaming consumers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Iterator, List, Mapping, Optional

from ..base_adapter import (
    AsyncOnChunk,
    AsyncOnDelta,
    AsyncOnDone,
    AsyncOnReasoning,
    AsyncOnToolCall,
    OnChunk,
    OnDelta,
    OnDone,
    OnReasoning,
    OnToolCall,
    _StreamState,
)
from ...llms.openai.sync_client import OpenAISyncClient
from ...llms.streaming import (
    StreamChunkBuffer,
    StreamReasoningCollector,
    StreamUsageTracker,
)
from ...models.messages.chat_message import Message, Messages
from ...models.responses.chat_response import ChatResponse, Usage
from ...models.tools import ToolSpec


@dataclass
class _ResponsesStreamState(_StreamState):
    final_response: Optional[dict] = None
    response_metadata: Dict[str, Any] = field(default_factory=dict)
    text_parts: List[str] = field(default_factory=list)
    function_calls: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class _OpenAIStreamingMixin:
    """Implement sync and async provider-event consumption for OpenAI."""

    def stream_chat(
        self,
        messages: List[Message] | Messages,
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        reasoning_level: Optional[str | int] = None,
        timeout_s: Optional[float] = None,
        tools: Optional[List[ToolSpec]] = None,
        tool_choice: Any = None,
        parallel_tool_calls: Optional[bool] = None,
        previous_response: Optional[ChatResponse] = None,
        json_schema: Optional[dict] = None,
        response_model: Optional[Any] = None,
        on_delta: Optional[OnDelta] = None,
        on_tool_call: Optional[OnToolCall] = None,
        on_done: Optional[OnDone] = None,
        buffer_chars: Optional[int] = None,
        on_chunk: Optional[OnChunk] = None,
        *,
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
        client = OpenAISyncClient(api_key=self.api_key)
        normalized_messages = request_context.normalized_messages
        use_responses_api = client._should_use_responses_api(self.model)
        params = self._build_stream_params(
            normalized_messages=normalized_messages,
            use_responses_api=use_responses_api,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            reasoning_level=reasoning_level,
            tools=tools,
            normalized_tool_choice=normalized_tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            previous_response=previous_response,
            effective_schema=effective_schema,
            capture_reasoning=capture_reasoning,
        )
        events = client.stream(timeout=timeout_s, **params)
        if use_responses_api:
            yield from self._consume_responses_stream(
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
            return
        yield from self._consume_chat_completions_stream(
            events,
            effective_schema,
            response_model,
            on_delta,
            on_tool_call,
            on_done,
            buffer_chars,
            on_chunk,
        )

    async def _consume_responses_stream_async(
        self,
        events: AsyncIterator[Any],
        effective_schema: Optional[dict],
        response_model: Optional[Any],
        on_delta: Optional[AsyncOnDelta],
        on_tool_call: Optional[AsyncOnToolCall],
        on_done: Optional[AsyncOnDone],
        buffer_chars: Optional[int],
        on_chunk: Optional[AsyncOnChunk],
        capture_reasoning: bool,
        on_reasoning: Optional[AsyncOnReasoning],
    ) -> AsyncIterator[str]:
        state = _ResponsesStreamState(
            chunk_buffer=StreamChunkBuffer(buffer_chars),
            usage_tracker=StreamUsageTracker(),
            reasoning_collector=(
                StreamReasoningCollector() if capture_reasoning else None
            ),
            reasoning_response=ChatResponse() if capture_reasoning else None,
        )
        async for text in self._run_async_stream(
            events,
            state,
            consume_event=self._consume_responses_stream_event_async,
            finalize_response=self._finalize_responses_stream,
            effective_schema=effective_schema,
            response_model=response_model,
            on_delta=on_delta,
            on_tool_call=on_tool_call,
            on_done=on_done,
            on_chunk=on_chunk,
            capture_reasoning=capture_reasoning,
            on_reasoning=on_reasoning,
        ):
            yield text

    async def _consume_responses_stream_event_async(
        self,
        event: Any,
        state: _ResponsesStreamState,
        *,
        on_chunk: Optional[AsyncOnChunk],
        on_delta: Optional[AsyncOnDelta],
        on_reasoning: Optional[AsyncOnReasoning],
    ) -> AsyncIterator[str]:
        payload = event.data if isinstance(event.data, Mapping) else {}
        event_type = event.event or payload.get("type")
        response_data = payload.get("response")
        if isinstance(response_data, Mapping):
            state.response_metadata.update(response_data)
        state.usage_tracker.record(
            state.chunk_buffer,
            self._normalize_stream_usage(
                self._response_event_usage(payload, response_data),
                input_field="input_tokens",
                output_field="output_tokens",
            ),
        )

        if event_type in (
            "response.reasoning_summary_text.delta",
            "response.reasoning_text.delta",
        ):
            await self._handle_responses_reasoning_delta_async(
                payload,
                event_type,
                state,
                on_reasoning,
            )
        elif event_type == "response.output_text.delta":
            delta = payload.get("delta")
            if isinstance(delta, str) and delta:
                state.text_parts.append(delta)
                async for text in self._emit_async_stream_chunks(
                    state.chunk_buffer.add(delta),
                    on_chunk,
                    on_delta,
                ):
                    yield text
        elif event_type in (
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
        ):
            self._handle_responses_function_call_event(
                payload,
                event_type,
                state,
            )
        elif event_type == "response.completed" and isinstance(response_data, Mapping):
            state.final_response = dict(response_data)

    async def _handle_responses_reasoning_delta_async(
        self,
        payload: Mapping[str, Any],
        event_type: str,
        state: _ResponsesStreamState,
        on_reasoning: Optional[AsyncOnReasoning],
    ) -> None:
        delta = payload.get("delta")
        if (
            state.reasoning_collector is None
            or state.reasoning_response is None
            or not isinstance(delta, str)
            or not delta
        ):
            return
        await self._record_async_reasoning_event(
            state.reasoning_response,
            state.reasoning_collector,
            delta,
            capture_reasoning=True,
            kind=(
                "summary"
                if event_type == "response.reasoning_summary_text.delta"
                else "content"
            ),
            on_reasoning=on_reasoning,
        )

    def _consume_responses_stream(
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
        state = _ResponsesStreamState(
            chunk_buffer=StreamChunkBuffer(buffer_chars),
            usage_tracker=StreamUsageTracker(),
            reasoning_collector=(
                StreamReasoningCollector() if capture_reasoning else None
            ),
            reasoning_response=ChatResponse() if capture_reasoning else None,
        )
        yield from self._run_sync_stream(
            events,
            state,
            consume_event=self._consume_responses_stream_event,
            finalize_response=self._finalize_responses_stream,
            effective_schema=effective_schema,
            response_model=response_model,
            on_delta=on_delta,
            on_tool_call=on_tool_call,
            on_done=on_done,
            on_chunk=on_chunk,
            capture_reasoning=capture_reasoning,
            on_reasoning=on_reasoning,
        )

    def _consume_responses_stream_event(
        self,
        event: Any,
        state: _ResponsesStreamState,
        *,
        on_chunk: Optional[OnChunk],
        on_delta: Optional[OnDelta],
        on_reasoning: Optional[OnReasoning],
    ) -> Iterator[str]:
        payload = event.data if isinstance(event.data, Mapping) else {}
        event_type = event.event or payload.get("type")
        response_data = payload.get("response")
        if isinstance(response_data, Mapping):
            state.response_metadata.update(response_data)
        state.usage_tracker.record(
            state.chunk_buffer,
            self._normalize_stream_usage(
                self._response_event_usage(payload, response_data),
                input_field="input_tokens",
                output_field="output_tokens",
            ),
        )

        if event_type in (
            "response.reasoning_summary_text.delta",
            "response.reasoning_text.delta",
        ):
            self._handle_responses_reasoning_delta(
                payload,
                event_type,
                state,
                on_reasoning,
            )
        elif event_type == "response.output_text.delta":
            yield from self._handle_responses_output_text_delta(
                payload,
                state,
                on_chunk,
                on_delta,
            )
        elif event_type in (
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
        ):
            self._handle_responses_function_call_event(
                payload,
                event_type,
                state,
            )
        elif event_type == "response.completed" and isinstance(response_data, Mapping):
            state.final_response = dict(response_data)

    def _handle_responses_reasoning_delta(
        self,
        payload: Mapping[str, Any],
        event_type: str,
        state: _ResponsesStreamState,
        on_reasoning: Optional[OnReasoning],
    ) -> None:
        delta = payload.get("delta")
        if (
            state.reasoning_collector is None
            or state.reasoning_response is None
            or not isinstance(delta, str)
            or not delta
        ):
            return
        self._record_reasoning_event(
            state.reasoning_response,
            state.reasoning_collector,
            delta,
            capture_reasoning=True,
            kind=(
                "summary"
                if event_type == "response.reasoning_summary_text.delta"
                else "content"
            ),
            on_reasoning=on_reasoning,
        )

    def _handle_responses_output_text_delta(
        self,
        payload: Mapping[str, Any],
        state: _ResponsesStreamState,
        on_chunk: Optional[OnChunk],
        on_delta: Optional[OnDelta],
    ) -> Iterator[str]:
        delta = payload.get("delta")
        if not isinstance(delta, str) or not delta:
            return
        state.text_parts.append(delta)
        yield from self._emit_stream_chunks(
            state.chunk_buffer.add(delta),
            on_chunk,
            on_delta,
        )

    @staticmethod
    def _handle_responses_function_call_event(
        payload: Mapping[str, Any],
        event_type: str,
        state: _ResponsesStreamState,
    ) -> None:
        call_key = str(
            payload.get("call_id")
            or payload.get("item_id")
            or payload.get("output_index")
            or len(state.function_calls)
        )
        call = state.function_calls.setdefault(
            call_key,
            {"type": "function_call", "arguments": ""},
        )
        call["call_id"] = payload.get("call_id") or call.get("call_id")
        call["id"] = payload.get("item_id") or call.get("id")
        call["name"] = payload.get("name") or call.get("name")
        if event_type.endswith(".delta") and isinstance(payload.get("delta"), str):
            call["arguments"] = f"{call.get('arguments', '')}{payload['delta']}"
        elif "arguments" in payload:
            call["arguments"] = payload["arguments"]

    def _finalize_responses_stream(
        self,
        state: _ResponsesStreamState,
        *,
        capture_reasoning: bool,
        effective_schema: Optional[dict],
        response_model: Optional[Any],
    ) -> ChatResponse:
        if state.final_response is None:
            final_response = self._build_responses_stream_response(state)
        else:
            final_response = state.final_response
        parser_kwargs = {"capture_reasoning": True} if capture_reasoning else {}
        chat_response = ChatResponse.from_openai_responses_response(
            final_response,
            **parser_kwargs,
        )
        return self._finalize_stream_response(
            chat_response,
            reasoning_collector=state.reasoning_collector,
            effective_schema=effective_schema,
            response_model=response_model,
        )

    def _build_responses_stream_response(
        self,
        state: _ResponsesStreamState,
    ) -> Dict[str, Any]:
        output: List[Dict[str, Any]] = []
        if state.text_parts:
            output.append(
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "".join(state.text_parts)}
                    ],
                }
            )
        output.extend(state.function_calls.values())
        return {
            **state.response_metadata,
            "model": state.response_metadata.get("model", self.model),
            "output": output,
            "status": state.response_metadata.get("status", "completed"),
        }

    async def _consume_chat_completions_stream_async(
        self,
        events: AsyncIterator[Any],
        effective_schema: Optional[dict],
        response_model: Optional[Any],
        on_delta: Optional[AsyncOnDelta],
        on_tool_call: Optional[AsyncOnToolCall],
        on_done: Optional[AsyncOnDone],
        buffer_chars: Optional[int],
        on_chunk: Optional[AsyncOnChunk],
    ) -> AsyncIterator[str]:
        text_parts: List[str] = []
        tool_calls: Dict[int, Dict[str, Any]] = {}
        legacy_response: Dict[str, Any] = {"model": self.model, "choices": []}
        finish_reason: Optional[str] = None
        chunk_buffer = StreamChunkBuffer(buffer_chars)
        usage_tracker = StreamUsageTracker()
        async for event in self._aiter_provider_stream_events(events):
            payload = event.data if isinstance(event.data, Mapping) else {}
            self._update_legacy_stream_metadata(payload, legacy_response)
            usage_tracker.record(
                chunk_buffer,
                self._normalize_stream_usage(
                    payload.get("usage"),
                    input_field="prompt_tokens",
                    output_field="completion_tokens",
                ),
            )
            choices = payload.get("choices")
            if not isinstance(choices, list):
                continue
            for choice in choices:
                text, choice_finish_reason = self._consume_legacy_stream_choice(
                    choice,
                    text_parts,
                    tool_calls,
                )
                if text is not None:
                    async for emitted_text in self._emit_async_stream_chunks(
                        chunk_buffer.add(text),
                        on_chunk,
                        on_delta,
                    ):
                        yield emitted_text
                if choice_finish_reason is not None:
                    finish_reason = choice_finish_reason
        chat_response = self._build_legacy_stream_response(
            legacy_response,
            text_parts,
            tool_calls,
            finish_reason,
        )
        chat_response = self._finalize_stream_response(
            chat_response,
            effective_schema=effective_schema,
            response_model=response_model,
        )
        async for text in self._complete_async_stream(
            chat_response,
            chunk_buffer,
            on_chunk,
            on_delta,
            on_tool_call,
            on_done,
        ):
            yield text

    def _consume_chat_completions_stream(
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
        tool_calls: Dict[int, Dict[str, Any]] = {}
        legacy_response: Dict[str, Any] = {"model": self.model, "choices": []}
        finish_reason: Optional[str] = None
        chunk_buffer = StreamChunkBuffer(buffer_chars)
        usage_tracker = StreamUsageTracker()
        for event in self._iter_provider_stream_events(events):
            payload = event.data if isinstance(event.data, Mapping) else {}
            self._update_legacy_stream_metadata(payload, legacy_response)
            usage_tracker.record(
                chunk_buffer,
                self._normalize_stream_usage(
                    payload.get("usage"),
                    input_field="prompt_tokens",
                    output_field="completion_tokens",
                ),
            )
            choices = payload.get("choices")
            if not isinstance(choices, list):
                continue
            for choice in choices:
                text, choice_finish_reason = self._consume_legacy_stream_choice(
                    choice,
                    text_parts,
                    tool_calls,
                )
                if text is not None:
                    yield from self._emit_stream_chunks(
                        chunk_buffer.add(text),
                        on_chunk,
                        on_delta,
                    )
                if choice_finish_reason is not None:
                    finish_reason = choice_finish_reason
        chat_response = self._build_legacy_stream_response(
            legacy_response,
            text_parts,
            tool_calls,
            finish_reason,
        )
        chat_response = self._finalize_stream_response(
            chat_response,
            effective_schema=effective_schema,
            response_model=response_model,
        )
        yield from self._complete_stream(
            chat_response,
            chunk_buffer,
            on_chunk,
            on_delta,
            on_tool_call,
            on_done,
        )

    @staticmethod
    def _update_legacy_stream_metadata(
        payload: Mapping[str, Any],
        legacy_response: Dict[str, Any],
    ) -> None:
        for field in ("id", "model", "created", "usage"):
            if field in payload:
                legacy_response[field] = payload[field]

    @staticmethod
    def _response_event_usage(
        payload: Mapping[str, Any],
        response_data: Any,
    ) -> Any:
        if isinstance(payload.get("usage"), Mapping):
            return payload["usage"]
        if isinstance(response_data, Mapping):
            return response_data.get("usage")
        return None

    @staticmethod
    def _normalize_stream_usage(
        raw_usage: Any,
        *,
        input_field: str,
        output_field: str,
    ) -> Optional[Usage]:
        if not isinstance(raw_usage, Mapping):
            return None

        input_tokens = _OpenAIStreamingMixin._token_count(raw_usage.get(input_field))
        output_tokens = _OpenAIStreamingMixin._token_count(raw_usage.get(output_field))
        total_tokens = _OpenAIStreamingMixin._token_count(raw_usage.get("total_tokens"))
        if input_tokens is None and output_tokens is None and total_tokens is None:
            return None
        return Usage(
            input_tokens=input_tokens or 0,
            output_tokens=output_tokens or 0,
            total_tokens=total_tokens or 0,
        )

    @staticmethod
    def _token_count(value: Any) -> Optional[int]:
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
        return None

    def _consume_legacy_stream_choice(
        self,
        choice: Any,
        text_parts: List[str],
        tool_calls: Dict[int, Dict[str, Any]],
    ) -> tuple[Optional[str], Optional[str]]:
        if not isinstance(choice, Mapping) or choice.get("index", 0) != 0:
            return None, None

        delta = choice.get("delta") or {}
        if not isinstance(delta, Mapping):
            delta = {}
        content = delta.get("content")
        text = content if isinstance(content, str) and content else None
        if text is not None:
            text_parts.append(text)

        raw_tool_calls = delta.get("tool_calls")
        if isinstance(raw_tool_calls, list):
            self._accumulate_legacy_tool_calls(raw_tool_calls, tool_calls)

        finish_reason = choice.get("finish_reason")
        return text, finish_reason

    @staticmethod
    def _accumulate_legacy_tool_calls(
        raw_tool_calls: List[Any],
        tool_calls: Dict[int, Dict[str, Any]],
    ) -> None:
        for raw_tool_call in raw_tool_calls:
            if not isinstance(raw_tool_call, Mapping):
                continue
            index = raw_tool_call.get("index")
            if not isinstance(index, int):
                index = len(tool_calls)
            tool_call = tool_calls.setdefault(index, {"function": {"arguments": ""}})
            for field in ("id", "type"):
                if raw_tool_call.get(field) is not None:
                    tool_call[field] = raw_tool_call[field]
            function = raw_tool_call.get("function") or {}
            if not isinstance(function, Mapping):
                continue
            target = tool_call["function"]
            if function.get("name") is not None:
                target["name"] = function["name"]
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                target["arguments"] = f"{target.get('arguments', '')}{arguments}"
            elif isinstance(arguments, dict):
                target["arguments"] = arguments

    @staticmethod
    def _build_legacy_stream_response(
        legacy_response: Dict[str, Any],
        text_parts: List[str],
        tool_calls: Dict[int, Dict[str, Any]],
        finish_reason: Optional[str],
    ) -> ChatResponse:
        message: Dict[str, Any] = {"content": "".join(text_parts) or None}
        if tool_calls:
            message["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]
        legacy_response["choices"] = [{"message": message, "finish_reason": finish_reason}]
        return ChatResponse.from_openai_response(legacy_response)
