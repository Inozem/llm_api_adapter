"""Core facade adapter for Z.ai's official Chat Completions API."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Optional

from llm_api_adapter.adapters.base_adapter import (
    AsyncOnChunk,
    AsyncOnDelta,
    AsyncOnDone,
    AsyncOnReasoning,
    AsyncOnToolCall,
    LLMAdapterBase,
    OnChunk,
    OnDelta,
    OnDone,
    OnReasoning,
    OnToolCall,
    _StreamState,
)
from llm_api_adapter.errors.config_errors import LLMReasoningLevelError
from llm_api_adapter.errors.llm_api_error import (
    LLMAPIClientError,
    LLMAPIError,
    ToolChoiceError,
)
from llm_api_adapter.llms.streaming import (
    StreamChunkBuffer,
    StreamReasoningCollector,
    StreamUsageTracker,
)
from llm_api_adapter.llms.transports import (
    SSEEvent,
    SyncTransport,
    create_sync_transport,
)
from llm_api_adapter.models.messages.chat_message import Message, Messages, UserMessage
from llm_api_adapter.models.messages.file_parts import DocumentPart
from llm_api_adapter.models.responses.chat_response import ChatResponse, Usage
from llm_api_adapter.models.responses.reasoning_event import ReasoningEvent
from llm_api_adapter.models.tools.tool_spec import ToolSpec

from .clients import ZaiAsyncClient, ZaiSyncClient
from .registry import CACHE_PRICING
from .streaming import ZaiStreamAssembler, assemble_zai_response


_SUPPORTED_REASONING_LEVELS = frozenset({"low", "high", "max"})


@dataclass
class ZaiUsage(Usage):
    """Normalized usage retaining Z.ai's optional cached-token split."""

    cached_tokens: int | None = None


@dataclass
class _ZaiStreamState(_StreamState):
    """Core stream lifecycle state plus the Z.ai protocol assembler."""

    assembler: ZaiStreamAssembler = field(default_factory=ZaiStreamAssembler)


@dataclass(repr=False)
class ZaiAdapter(LLMAdapterBase):
    """Map the shared adapter contract to Z.ai Chat Completions."""

    company: str = "zai"
    _client: ZaiSyncClient = field(init=False, repr=False, compare=False)
    _async_client: ZaiAsyncClient = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._client = ZaiSyncClient(create_sync_transport(self.transport))
        self._async_client = ZaiAsyncClient()

    @property
    def _sync_transport(self) -> SyncTransport:
        """Compatibility hook used by deterministic transport tests."""

        return self._client.transport

    @_sync_transport.setter
    def _sync_transport(self, transport: SyncTransport) -> None:
        self._client.transport = transport

    def chat(
        self,
        messages: list[Message] | Messages,
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        reasoning_level: Optional[str | int] = None,
        timeout_s: Optional[float] = None,
        tools: Optional[list[ToolSpec]] = None,
        tool_choice: Any = None,
        parallel_tool_calls: Optional[bool] = None,
        previous_response: Optional[ChatResponse] = None,
        json_schema: Optional[dict] = None,
        response_model: Optional[Any] = None,
        *,
        capture_reasoning: bool = False,
    ) -> ChatResponse:
        """Create one normalized response through Z.ai Chat Completions."""

        _ = previous_response
        request_context, payload = self._prepare_request_payload(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            reasoning_level=reasoning_level,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            json_schema=json_schema,
            response_model=response_model,
        )
        try:
            response = self._client.chat(
                api_key=self.api_key,
                payload=payload,
                timeout_s=timeout_s,
            )
            return self._finalize_zai_chat_response(
                response,
                capture_reasoning=capture_reasoning,
                effective_schema=request_context.effective_schema,
                response_model=request_context.response_model,
            )
        except LLMAPIError as error:
            self.handle_error(error)

    async def achat(
        self,
        messages: list[Message] | Messages,
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        reasoning_level: Optional[str | int] = None,
        timeout_s: Optional[float] = None,
        tools: Optional[list[ToolSpec]] = None,
        tool_choice: Any = None,
        parallel_tool_calls: Optional[bool] = None,
        previous_response: Optional[ChatResponse] = None,
        json_schema: Optional[dict] = None,
        response_model: Optional[Any] = None,
        *,
        capture_reasoning: bool = False,
    ) -> ChatResponse:
        """Create one normalized response through HTTPX asynchronously."""

        _ = previous_response
        request_context, payload = self._prepare_request_payload(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            reasoning_level=reasoning_level,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            json_schema=json_schema,
            response_model=response_model,
        )
        try:
            response = await self._async_client.chat(
                api_key=self.api_key,
                payload=payload,
                timeout_s=timeout_s,
            )
            return self._finalize_zai_chat_response(
                response,
                capture_reasoning=capture_reasoning,
                effective_schema=request_context.effective_schema,
                response_model=request_context.response_model,
            )
        except LLMAPIError as error:
            self.handle_error(error)

    def stream_chat(
        self,
        messages: list[Message] | Messages,
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        reasoning_level: Optional[str | int] = None,
        timeout_s: Optional[float] = None,
        tools: Optional[list[ToolSpec]] = None,
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
        """Stream visible Z.ai deltas through the shared Core lifecycle."""

        _ = previous_response
        request_context, payload = self._prepare_request_payload(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            reasoning_level=reasoning_level,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            json_schema=json_schema,
            response_model=response_model,
        )
        state = self._new_stream_state(
            buffer_chars=buffer_chars,
            capture_reasoning=capture_reasoning,
        )
        events = self._client.stream(
            api_key=self.api_key,
            payload=payload,
            timeout_s=timeout_s,
        )
        return self._run_sync_stream(
            events,
            state,
            consume_event=self._consume_stream_event,
            finalize_response=self._finalize_stream,
            effective_schema=request_context.effective_schema,
            response_model=request_context.response_model,
            on_delta=on_delta,
            on_tool_call=on_tool_call,
            on_done=on_done,
            on_chunk=on_chunk,
            capture_reasoning=capture_reasoning,
            on_reasoning=on_reasoning,
        )

    def astream_chat(
        self,
        messages: list[Message] | Messages,
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        reasoning_level: Optional[str | int] = None,
        timeout_s: Optional[float] = None,
        tools: Optional[list[ToolSpec]] = None,
        tool_choice: Any = None,
        parallel_tool_calls: Optional[bool] = None,
        previous_response: Optional[ChatResponse] = None,
        json_schema: Optional[dict] = None,
        response_model: Optional[Any] = None,
        on_delta: Optional[AsyncOnDelta] = None,
        on_tool_call: Optional[AsyncOnToolCall] = None,
        on_done: Optional[AsyncOnDone] = None,
        buffer_chars: Optional[int] = None,
        on_chunk: Optional[AsyncOnChunk] = None,
        *,
        capture_reasoning: bool = False,
        on_reasoning: Optional[AsyncOnReasoning] = None,
    ) -> AsyncIterator[str]:
        """Return an asynchronous Z.ai SSE iterator."""

        return self._astream_chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            reasoning_level=reasoning_level,
            timeout_s=timeout_s,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            previous_response=previous_response,
            json_schema=json_schema,
            response_model=response_model,
            on_delta=on_delta,
            on_tool_call=on_tool_call,
            on_done=on_done,
            buffer_chars=buffer_chars,
            on_chunk=on_chunk,
            capture_reasoning=capture_reasoning,
            on_reasoning=on_reasoning,
        )

    async def _astream_chat(
        self,
        *,
        messages: list[Message] | Messages,
        max_tokens: Optional[int],
        temperature: float,
        top_p: float,
        reasoning_level: Optional[str | int],
        timeout_s: Optional[float],
        tools: Optional[list[ToolSpec]],
        tool_choice: Any,
        parallel_tool_calls: Optional[bool],
        previous_response: Optional[ChatResponse],
        json_schema: Optional[dict],
        response_model: Optional[Any],
        on_delta: Optional[AsyncOnDelta],
        on_tool_call: Optional[AsyncOnToolCall],
        on_done: Optional[AsyncOnDone],
        buffer_chars: Optional[int],
        on_chunk: Optional[AsyncOnChunk],
        capture_reasoning: bool,
        on_reasoning: Optional[AsyncOnReasoning],
    ) -> AsyncIterator[str]:
        """Run one asynchronous Z.ai stream without provider continuation."""

        _ = previous_response
        request_context, payload = self._prepare_request_payload(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            reasoning_level=reasoning_level,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            json_schema=json_schema,
            response_model=response_model,
        )
        state = self._new_stream_state(
            buffer_chars=buffer_chars,
            capture_reasoning=capture_reasoning,
        )
        events = self._async_client.stream(
            api_key=self.api_key,
            payload=payload,
            timeout_s=timeout_s,
        )
        async for text in self._run_async_stream(
            events,
            state,
            consume_event=self._consume_stream_event_async,
            finalize_response=self._finalize_stream,
            effective_schema=request_context.effective_schema,
            response_model=request_context.response_model,
            on_delta=on_delta,
            on_tool_call=on_tool_call,
            on_done=on_done,
            on_chunk=on_chunk,
            capture_reasoning=capture_reasoning,
            on_reasoning=on_reasoning,
        ):
            yield text

    @staticmethod
    def _new_stream_state(
        *,
        buffer_chars: Optional[int],
        capture_reasoning: bool,
    ) -> _ZaiStreamState:
        return _ZaiStreamState(
            chunk_buffer=StreamChunkBuffer(buffer_chars),
            usage_tracker=StreamUsageTracker(),
            reasoning_collector=(
                StreamReasoningCollector() if capture_reasoning else None
            ),
            reasoning_response=ChatResponse() if capture_reasoning else None,
            assembler=ZaiStreamAssembler(),
        )

    def _consume_stream_event(
        self,
        event: SSEEvent,
        state: _ZaiStreamState,
        *,
        on_chunk: Optional[OnChunk],
        on_delta: Optional[OnDelta],
        on_reasoning: Optional[OnReasoning],
    ) -> Iterator[str]:
        delta = state.assembler.consume(event)
        state.usage_tracker.record(
            state.chunk_buffer,
            self._stream_usage(event),
        )
        if delta.reasoning_text:
            self._record_stream_reasoning(
                state,
                delta.reasoning_text,
                on_reasoning=on_reasoning,
            )
        if delta.visible_text:
            yield from self._emit_stream_chunks(
                state.chunk_buffer.add(delta.visible_text),
                on_chunk,
                on_delta,
            )

    async def _consume_stream_event_async(
        self,
        event: SSEEvent,
        state: _ZaiStreamState,
        *,
        on_chunk: Optional[AsyncOnChunk],
        on_delta: Optional[AsyncOnDelta],
        on_reasoning: Optional[AsyncOnReasoning],
    ) -> AsyncIterator[str]:
        delta = state.assembler.consume(event)
        state.usage_tracker.record(
            state.chunk_buffer,
            self._stream_usage(event),
        )
        if delta.reasoning_text:
            await self._record_stream_reasoning_async(
                state,
                delta.reasoning_text,
                on_reasoning=on_reasoning,
            )
        if delta.visible_text:
            async for text in self._emit_async_stream_chunks(
                state.chunk_buffer.add(delta.visible_text),
                on_chunk,
                on_delta,
            ):
                yield text

    def _record_stream_reasoning(
        self,
        state: _ZaiStreamState,
        text: str,
        *,
        on_reasoning: Optional[OnReasoning],
    ) -> None:
        if state.reasoning_collector is None or state.reasoning_response is None:
            return
        self._record_reasoning_event(
            state.reasoning_response,
            state.reasoning_collector,
            text,
            capture_reasoning=True,
            on_reasoning=on_reasoning,
        )

    async def _record_stream_reasoning_async(
        self,
        state: _ZaiStreamState,
        text: str,
        *,
        on_reasoning: Optional[AsyncOnReasoning],
    ) -> None:
        if state.reasoning_collector is None or state.reasoning_response is None:
            return
        await self._record_async_reasoning_event(
            state.reasoning_response,
            state.reasoning_collector,
            text,
            capture_reasoning=True,
            on_reasoning=on_reasoning,
        )

    @staticmethod
    def _stream_usage(event: SSEEvent) -> Optional[Usage]:
        payload = event.data if isinstance(event.data, Mapping) else None
        raw_usage = payload.get("usage") if payload is not None else None
        if not isinstance(raw_usage, Mapping):
            return None
        values = tuple(
            raw_usage.get(field_name)
            for field_name in ("prompt_tokens", "completion_tokens", "total_tokens")
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values
        ):
            return None
        if values[2] != values[0] + values[1]:
            return None
        return Usage(
            input_tokens=values[0],
            output_tokens=values[1],
            total_tokens=values[2],
        )

    def _finalize_stream(
        self,
        state: _ZaiStreamState,
        *,
        capture_reasoning: bool,
        effective_schema: Optional[dict],
        response_model: Optional[Any],
    ) -> ChatResponse:
        """Normalize the assembled terminal stream response."""

        _ = capture_reasoning
        response = self._parse_response(
            assemble_zai_response(state.assembler.state, model=self.model),
            capture_reasoning=False,
        )
        chat_response = self._finalize_stream_response(
            response,
            reasoning_collector=state.reasoning_collector,
            effective_schema=effective_schema,
            response_model=response_model,
        )
        self._apply_cache_aware_pricing(chat_response)
        return chat_response

    def _prepare_request_payload(
        self,
        *,
        messages: list[Message] | Messages,
        max_tokens: Optional[int],
        temperature: float,
        top_p: float,
        reasoning_level: Optional[str | int],
        tools: Optional[list[ToolSpec]],
        tool_choice: Any,
        parallel_tool_calls: Optional[bool],
        json_schema: Optional[dict],
        response_model: Optional[Any],
    ) -> tuple[Any, dict[str, Any]]:
        """Validate Core inputs and serialize the official Z.ai wire shape."""

        self._validate_capability_preflight(
            parallel_tool_calls=parallel_tool_calls,
        )
        if json_schema is not None or response_model is not None:
            raise NotImplementedError(
                "Z.ai glm-5.3-flash does not support portable structured output",
            )

        request_context = self._prepare_chat_request(
            messages,
            tools,
            tool_choice,
            None,
            None,
        )
        self._reject_unsupported_file_parts(request_context.normalized_messages)
        validated_max_tokens = self._validate_max_tokens(max_tokens)
        temperature, top_p = self._validate_sampling_parameters(temperature, top_p)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": request_context.normalized_messages.to_openai(),
        }
        if validated_max_tokens is not None:
            payload["max_tokens"] = validated_max_tokens
        if temperature != 1.0:
            payload["temperature"] = temperature
        if top_p != 1.0:
            payload["top_p"] = top_p

        mapped_tools = self._map_tools(tools)
        if mapped_tools is not None:
            payload["tools"] = mapped_tools
        if request_context.normalized_tool_choice is not None:
            payload["tool_choice"] = self._map_tool_choice(
                request_context.normalized_tool_choice,
            )
        self._apply_reasoning_options(payload, reasoning_level)
        return request_context, payload

    def _validate_capability_preflight(
        self,
        *,
        parallel_tool_calls: Optional[bool],
    ) -> None:
        """Reject requests outside the closed Z.ai capability profile."""

        if self.model_spec is None:
            raise NotImplementedError(
                f"Z.ai model {self.model!r} is not verified for supported "
                "capabilities",
            )
        if parallel_tool_calls is not None:
            raise NotImplementedError(
                "Z.ai parallel_tool_calls control is not implemented",
            )

    @staticmethod
    def _map_tools(tools: Optional[list[ToolSpec]]) -> Optional[list[dict[str, Any]]]:
        if not tools:
            return None
        if len(tools) > 128:
            raise ValueError("Z.ai supports at most 128 function tools per request")
        mapped_tools: list[dict[str, Any]] = []
        for tool in tools:
            function: dict[str, Any] = {
                "name": tool.name,
                "parameters": tool.json_schema,
            }
            if tool.description:
                function["description"] = tool.description
            mapped_tools.append({"type": "function", "function": function})
        return mapped_tools

    @staticmethod
    def _map_tool_choice(tool_choice: str) -> str:
        if tool_choice != "auto":
            raise ToolChoiceError(
                detail=(
                    "Z.ai glm-5.3-flash supports only tool_choice='auto'; "
                    f"received {tool_choice!r}"
                ),
            )
        return "auto"

    @staticmethod
    def _reject_unsupported_file_parts(messages: Messages) -> None:
        """Reject documents while preserving URL/data-URL image serialization."""

        for message in messages.items:
            if not isinstance(message, UserMessage) or not message.files:
                continue
            for file_part in message.files:
                if isinstance(file_part, DocumentPart):
                    raise ValueError(
                        "Z.ai glm-5.3-flash does not support DocumentPart yet",
                    )

    def _validate_max_tokens(self, max_tokens: Optional[int]) -> Optional[int]:
        if max_tokens is None:
            return None
        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens <= 0
        ):
            raise ValueError("max_tokens must be a positive integer for Z.ai")
        if (
            self.model_spec is not None
            and max_tokens > self.model_spec.limits.max_output_tokens
        ):
            raise ValueError(
                "max_tokens must not exceed "
                f"{self.model_spec.limits.max_output_tokens} "
                f"for Z.ai model {self.model!r}",
            )
        return max_tokens

    @staticmethod
    def _apply_reasoning_options(
        payload: dict[str, Any],
        reasoning_level: Optional[str | int],
    ) -> None:
        if reasoning_level is None:
            return
        if (
            not isinstance(reasoning_level, str)
            or reasoning_level not in _SUPPORTED_REASONING_LEVELS
        ):
            allowed = ", ".join(sorted(_SUPPORTED_REASONING_LEVELS))
            raise LLMReasoningLevelError(
                detail=f"Z.ai reasoning_level must be one of: {allowed}",
            )
        payload["thinking"] = {"type": "enabled"}
        payload["reasoning_effort"] = reasoning_level

    @staticmethod
    def _parse_response(
        response: Mapping[str, Any],
        *,
        capture_reasoning: bool = False,
    ) -> ChatResponse:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMAPIClientError(
                detail=(
                    "Z.ai Chat Completions response.choices must be a non-empty array"
                ),
            )
        choice = choices[0]
        if not isinstance(choice, Mapping) or not isinstance(
            choice.get("message"), Mapping
        ):
            raise LLMAPIClientError(
                detail=(
                    "Z.ai Chat Completions response.choices[0].message must be "
                    "an object"
                ),
            )
        message = choice["message"]
        content = message.get("content")
        if content is not None and not isinstance(content, str):
            raise LLMAPIClientError(
                detail=(
                    "Z.ai Chat Completions response content must be a string or null"
                ),
            )

        usage = ZaiAdapter._parse_usage(response.get("usage"))
        parse_payload = dict(response)
        if usage is None:
            parse_payload.pop("usage", None)
        chat_response = ChatResponse.from_openai_response(parse_payload)
        if usage is not None:
            chat_response.usage = usage

        reasoning = message.get("reasoning_content")
        if capture_reasoning and isinstance(reasoning, str) and reasoning:
            chat_response.reasoning_events = [
                ReasoningEvent(
                    text=reasoning,
                    kind="summary",
                    index=0,
                    elapsed_s=0.0,
                    delta_s=0.0,
                ),
            ]
        return chat_response

    @staticmethod
    def _parse_usage(raw_usage: Any) -> Optional[ZaiUsage]:
        if not isinstance(raw_usage, Mapping):
            return None
        values = tuple(
            raw_usage.get(field_name)
            for field_name in ("prompt_tokens", "completion_tokens", "total_tokens")
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values
        ):
            return None
        if values[2] != values[0] + values[1]:
            return None

        cached_tokens: int | None = None
        details = raw_usage.get("prompt_tokens_details")
        if isinstance(details, Mapping):
            raw_cached = details.get("cached_tokens")
            if (
                isinstance(raw_cached, int)
                and not isinstance(raw_cached, bool)
                and 0 <= raw_cached <= values[0]
            ):
                cached_tokens = raw_cached

        return ZaiUsage(
            input_tokens=values[0],
            output_tokens=values[1],
            total_tokens=values[2],
            cached_tokens=cached_tokens,
        )

    def _finalize_zai_chat_response(
        self,
        response: Mapping[str, Any],
        *,
        capture_reasoning: bool,
        effective_schema: Optional[dict],
        response_model: Optional[Any],
    ) -> ChatResponse:
        chat_response = self._finalize_chat_response(
            self._parse_response(response, capture_reasoning=capture_reasoning),
            effective_schema=effective_schema,
            response_model=response_model,
        )
        self._apply_cache_aware_pricing(chat_response)
        return chat_response

    def _apply_cache_aware_pricing(self, chat_response: ChatResponse) -> None:
        pricing = CACHE_PRICING.get(self.model)
        usage = chat_response.usage
        if pricing is None or not isinstance(usage, ZaiUsage):
            return
        if usage.cached_tokens is None:
            return
        estimate = pricing.calculate(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cached_tokens=usage.cached_tokens,
        )
        if estimate is None:
            return
        chat_response.currency = "USD"
        chat_response.cost_input = estimate.input_cost
        chat_response.cost_output = estimate.output_cost
        chat_response.cost_total = estimate.total_cost


__all__ = ["ZaiAdapter", "ZaiUsage"]
