"""Core facade adapter for Z.ai's official Chat Completions API."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Optional

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
from llm_api_adapter.adapters.openai_compatible.adapter import OpenAICompatibleAdapter
from llm_api_adapter.errors.llm_api_error import (
    LLMAPIClientError,
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

from .clients import ZAI_BASE_URL, ZaiAsyncClient, ZaiSyncClient
from .streaming import ZaiStreamAssembler, assemble_zai_response


@dataclass
class ZaiUsage(Usage):
    """Normalized usage retaining Z.ai's optional cached-token split."""

    cached_tokens: int | None = None


@dataclass
class _ZaiStreamState(_StreamState):
    """Core stream lifecycle state plus the Z.ai protocol assembler."""

    assembler: ZaiStreamAssembler = field(default_factory=ZaiStreamAssembler)


class _NonClosingIterator:
    """Let the shared lifecycle iterate without taking ownership of a source."""

    def __init__(self, source: Iterator[Any]) -> None:
        self._source = source

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._source)


@dataclass(repr=False)
class ZaiAdapter(OpenAICompatibleAdapter):
    """Map the shared adapter contract to Z.ai Chat Completions."""

    company: str = "zai"
    base_url: ClassVar[str] = ZAI_BASE_URL
    api_variant: ClassVar[str | None] = "chat_completions"
    _client: ZaiSyncClient = field(init=False, repr=False, compare=False)
    _async_client: ZaiAsyncClient = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        LLMAdapterBase.__post_init__(self)
        self._client = ZaiSyncClient(create_sync_transport(self.transport))
        self._async_client = ZaiAsyncClient()

    def __repr__(self) -> str:
        """Keep the native adapter's established credential-masked repr."""

        return LLMAdapterBase.__repr__(self)

    def _create_sync_client(self):
        """Bind the native key and current injected transport for this call."""

        return self._client._for_request(self.api_key)

    def _create_async_client(self):
        """Bind the native key to a fresh request-scoped async client."""

        return self._async_client._for_request(self.api_key)

    @property
    def _sync_transport(self) -> SyncTransport:
        """Compatibility hook used by deterministic transport tests."""

        return self._client.transport

    @_sync_transport.setter
    def _sync_transport(self, transport: SyncTransport) -> None:
        self._client.transport = transport

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
        """Lazily stream visible Z.ai deltas through the shared lifecycle."""

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
        stream_payload = dict(payload)
        stream_payload.pop("model", None)
        events = self._create_sync_client().stream(
            model=self.model,
            timeout=timeout_s,
            **stream_payload,
        )
        lifecycle = self._run_sync_stream(
            _NonClosingIterator(events),
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
        try:
            yield from lifecycle
        except BaseException:
            self._close_sync_iterators(lifecycle, events, suppress_errors=True)
            raise
        else:
            self._close_sync_iterators(lifecycle, events, suppress_errors=False)

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
        stream_payload = dict(payload)
        stream_payload.pop("model", None)
        events = self._create_async_client().stream(
            model=self.model,
            timeout=timeout_s,
            **stream_payload,
        )
        lifecycle = self._run_async_stream(
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
        )
        async with self._closing_async_source(events):
            async with self._closing_async_source(lifecycle):
                async for text in lifecycle:
                    yield text

    @staticmethod
    def _close_sync_iterators(
        *iterators: Any,
        suppress_errors: bool,
    ) -> None:
        first_error: BaseException | None = None
        for iterator in iterators:
            close = getattr(iterator, "close", None)
            if not callable(close):
                continue
            try:
                close()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None and not suppress_errors:
            raise first_error

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
        return ZaiAdapter._parse_usage(raw_usage)

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
        request_context = self._prepare_chat_request(
            messages,
            tools,
            tool_choice,
            json_schema,
            response_model,
        )
        temperature, top_p = self._validate_sampling_parameters(temperature, top_p)
        payload = self._build_zai_payload(
            normalized_messages=request_context.normalized_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            reasoning_level=reasoning_level,
            tools=tools,
            normalized_tool_choice=request_context.normalized_tool_choice,
        )
        return request_context, payload

    def _prepare_chat_request(
        self,
        messages: list[Message] | Messages,
        tools: Optional[list[ToolSpec]],
        tool_choice: Any,
        json_schema: Optional[dict],
        response_model: Optional[Any],
    ):
        """Prepare common message/tool inputs under the native capability policy."""

        if json_schema is not None or response_model is not None:
            raise NotImplementedError(
                f"Z.ai model {self.model!r} does not support portable structured output",
            )
        if self.model_spec is None:
            raise NotImplementedError(
                f"Z.ai model {self.model!r} is not verified for supported capabilities",
            )
        context = LLMAdapterBase._prepare_chat_request(
            self,
            messages,
            tools,
            tool_choice,
            None,
            None,
        )
        self._reject_unsupported_file_parts(context.normalized_messages)
        return context

    def _build_chat_params(
        self,
        *,
        normalized_messages: Messages,
        use_responses_api: bool,
        max_tokens: Optional[int],
        temperature: float,
        top_p: float,
        reasoning_level: Optional[str | int],
        tools: Optional[list[ToolSpec]],
        normalized_tool_choice: Optional[str],
        parallel_tool_calls: Optional[bool],
        previous_response: Optional[ChatResponse],
        effective_schema: Optional[dict],
        capture_reasoning: bool,
    ) -> dict[str, Any]:
        """Build native Chat Completions parameters for inherited execution."""

        _ = previous_response, effective_schema, capture_reasoning
        self._validate_capability_preflight(
            parallel_tool_calls=parallel_tool_calls,
        )
        if use_responses_api:
            raise NotImplementedError("Z.ai uses Chat Completions only")
        return self._build_zai_payload(
            normalized_messages=normalized_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            reasoning_level=reasoning_level,
            tools=tools,
            normalized_tool_choice=normalized_tool_choice,
        )

    def _build_zai_payload(
        self,
        *,
        normalized_messages: Messages,
        max_tokens: Optional[int],
        temperature: float,
        top_p: float,
        reasoning_level: Optional[str | int],
        tools: Optional[list[ToolSpec]],
        normalized_tool_choice: Optional[str],
    ) -> dict[str, Any]:
        """Apply one native payload policy for completions and both streams."""

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": normalized_messages.to_openai(),
        }
        validated_max_tokens = self._validate_max_tokens(max_tokens)
        if validated_max_tokens is not None:
            payload["max_tokens"] = validated_max_tokens
        if temperature != 1.0:
            payload["temperature"] = temperature
        if top_p != 1.0:
            payload["top_p"] = top_p

        mapped_tools = self._map_tools(tools)
        if mapped_tools is not None:
            payload["tools"] = mapped_tools
        if normalized_tool_choice is not None:
            payload["tool_choice"] = self._map_tool_choice(
                normalized_tool_choice,
            )
        self._apply_reasoning_options(payload, reasoning_level)
        return payload

    def _parse_chat_response(
        self,
        response: dict,
        *,
        use_responses_api: bool,
        capture_reasoning: bool,
    ) -> ChatResponse:
        """Keep native envelope, usage, and reasoning normalization in control."""

        if use_responses_api:
            raise NotImplementedError("Z.ai uses Chat Completions only")
        return self._parse_response(response, capture_reasoning=capture_reasoning)

    def _apply_response_pricing(self, chat_response: ChatResponse) -> None:
        """Keep native usage/pricing semantics when sharing response finalization."""

        LLMAdapterBase._apply_response_pricing(self, chat_response)

    def handle_error(self, error: Exception, error_message: Optional[str] = None):
        """Preserve native errors without logging caller or provider payload text."""

        _ = error_message
        LLMAdapterBase.handle_error(self, error)

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

    def _map_tools(
        self,
        tools: Optional[list[ToolSpec]],
    ) -> Optional[list[dict[str, Any]]]:
        """Enforce Z.ai's native limit and share Core function-tool mapping."""
        if tools and len(tools) > 128:
            raise ValueError("Z.ai supports at most 128 function tools per request")
        return super()._map_tools_to_openai(tools)

    def _map_tool_choice(self, tool_choice: str) -> str:
        if tool_choice != "auto":
            raise ToolChoiceError(
                detail=(
                    f"Z.ai model {self.model!r} supports only tool_choice='auto'; "
                    f"received {tool_choice!r}"
                ),
            )
        return "auto"

    def _reject_unsupported_file_parts(self, messages: Messages) -> None:
        """Reject documents while preserving URL/data-URL image serialization."""

        for message in messages.items:
            if not isinstance(message, UserMessage) or not message.files:
                continue
            for file_part in message.files:
                if isinstance(file_part, DocumentPart):
                    raise ValueError(
                        f"Z.ai model {self.model!r} does not support DocumentPart yet",
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

    def _apply_reasoning_options(
        self,
        payload: dict[str, Any],
        reasoning_level: Optional[str | int],
    ) -> None:
        if reasoning_level is None:
            return

        provider_value = self._resolve_reasoning_level(reasoning_level).provider_value
        if provider_value is None:
            return
        if not isinstance(provider_value, str):
            raise TypeError("Z.ai reasoning resolution must produce a string")
        payload["thinking"] = {"type": "enabled"}
        payload["reasoning_effort"] = provider_value

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

        def read_count(field_name: str) -> int | None:
            if field_name not in raw_usage or raw_usage[field_name] is None:
                return None
            value = raw_usage[field_name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"invalid Z.ai usage count: {field_name}")
            return value

        try:
            input_tokens = read_count("prompt_tokens")
            output_tokens = read_count("completion_tokens")
            total_tokens = read_count("total_tokens")
        except ValueError:
            return None

        if input_tokens is None and output_tokens is None and total_tokens is None:
            return None
        if input_tokens is not None and output_tokens is not None:
            exact_total = input_tokens + output_tokens
            if total_tokens is not None and total_tokens != exact_total:
                return None
            total_tokens = exact_total

        cached_tokens: int | None = None
        details = raw_usage.get("prompt_tokens_details")
        if input_tokens is not None and isinstance(details, Mapping):
            raw_cached = details.get("cached_tokens")
            if (
                isinstance(raw_cached, int)
                and not isinstance(raw_cached, bool)
                and 0 <= raw_cached <= input_tokens
            ):
                cached_tokens = raw_cached

        return ZaiUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cached_tokens=cached_tokens,
        )

__all__ = ["ZaiAdapter", "ZaiUsage"]
