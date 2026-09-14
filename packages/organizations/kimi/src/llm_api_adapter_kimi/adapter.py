"""Synchronous adapter for Kimi's official Chat Completions API."""

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
from llm_api_adapter.adapters.structured_output import validate_core_portable_schema
from llm_api_adapter.errors.llm_api_error import LLMAPIClientError, LLMAPIError
from llm_api_adapter.llm_registry.llm_registry import CategoricalReasoningCapability
from llm_api_adapter.llm_registry.request_rules import apply_request_rules
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
from llm_api_adapter.models.responses.chat_response import ChatResponse, Usage
from llm_api_adapter.models.responses.reasoning_event import ReasoningEvent
from llm_api_adapter.models.tools.tool_spec import ToolSpec

from .clients import KimiAsyncClient, KimiSyncClient
from .registry import CACHE_PRICING, KimiCachePricing


@dataclass
class KimiUsage(Usage):
    """Normalized Kimi usage retaining the cache-hit split used for pricing."""

    cached_tokens: int | None = None


@dataclass
class _KimiStreamState(_StreamState):
    """Provider state retained while reconstructing a Kimi SSE response."""

    response_metadata: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)
    text_parts: list[str] = field(default_factory=list)
    tool_calls: dict[int, dict[str, Any]] = field(default_factory=dict)
    finish_reason: str | None = None


@dataclass(repr=False)
class KimiAdapter(LLMAdapterBase):
    """Map the shared text-chat contract to Kimi Chat Completions."""

    company: str = "kimi"
    _client: KimiSyncClient = field(init=False, repr=False, compare=False)
    _async_client: KimiAsyncClient = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._client = KimiSyncClient(create_sync_transport(self.transport))
        self._async_client = KimiAsyncClient()

    @property
    def _sync_transport(self) -> SyncTransport:
        """Compatibility hook for deterministic transport tests."""
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
        """Create one Kimi response through ``POST /v1/chat/completions``."""
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
            return self._finalize_kimi_chat_response(
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
        """Create one Kimi response without blocking the event loop."""
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
            return self._finalize_kimi_chat_response(
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
        """Stream Kimi visible deltas through the shared lifecycle."""
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
        """Return Kimi's asynchronous SSE iterator."""
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
        """Run one asynchronous Kimi stream without server-side continuation."""
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
    ) -> _KimiStreamState:
        return _KimiStreamState(
            chunk_buffer=StreamChunkBuffer(buffer_chars),
            usage_tracker=StreamUsageTracker(),
            reasoning_collector=(
                StreamReasoningCollector() if capture_reasoning else None
            ),
            reasoning_response=ChatResponse() if capture_reasoning else None,
        )

    def _consume_stream_event(
        self,
        event: SSEEvent,
        state: _KimiStreamState,
        *,
        on_chunk: Optional[OnChunk],
        on_delta: Optional[OnDelta],
        on_reasoning: Optional[OnReasoning],
    ) -> Iterator[str]:
        visible_text, reasoning_text = self._consume_stream_payload(event, state)
        self._record_stream_reasoning(
            state,
            reasoning_text,
            on_reasoning=on_reasoning,
        )
        for text in visible_text:
            yield from self._emit_stream_chunks(
                state.chunk_buffer.add(text),
                on_chunk,
                on_delta,
            )

    async def _consume_stream_event_async(
        self,
        event: SSEEvent,
        state: _KimiStreamState,
        *,
        on_chunk: Optional[AsyncOnChunk],
        on_delta: Optional[AsyncOnDelta],
        on_reasoning: Optional[AsyncOnReasoning],
    ) -> AsyncIterator[str]:
        visible_text, reasoning_text = self._consume_stream_payload(event, state)
        await self._record_stream_reasoning_async(
            state,
            reasoning_text,
            on_reasoning=on_reasoning,
        )
        for text in visible_text:
            async for emitted_text in self._emit_async_stream_chunks(
                state.chunk_buffer.add(text),
                on_chunk,
                on_delta,
            ):
                yield emitted_text

    def _consume_stream_payload(
        self,
        event: SSEEvent,
        state: _KimiStreamState,
    ) -> tuple[list[str], list[str]]:
        """Accumulate one OpenAI-compatible Kimi chunk without exposing reasoning."""
        payload = event.data if isinstance(event.data, Mapping) else {}
        for field in ("id", "model", "created"):
            if field in payload:
                state.response_metadata[field] = payload[field]
        usage = payload.get("usage")
        if isinstance(usage, Mapping):
            state.usage.update(usage)
        state.usage_tracker.record(
            state.chunk_buffer,
            self._stream_usage(usage),
        )

        visible_text: list[str] = []
        reasoning_text: list[str] = []
        choices = payload.get("choices")
        if not isinstance(choices, list):
            return visible_text, reasoning_text
        for choice in choices:
            if not isinstance(choice, Mapping) or choice.get("index", 0) != 0:
                continue
            delta = choice.get("delta")
            if isinstance(delta, Mapping):
                content = delta.get("content")
                if isinstance(content, str) and content:
                    visible_text.append(content)
                reasoning_content = delta.get("reasoning_content")
                if isinstance(reasoning_content, str) and reasoning_content:
                    reasoning_text.append(reasoning_content)
                raw_tool_calls = delta.get("tool_calls")
                if isinstance(raw_tool_calls, list):
                    self._accumulate_stream_tool_calls(raw_tool_calls, state.tool_calls)
            finish_reason = choice.get("finish_reason")
            if isinstance(finish_reason, str):
                state.finish_reason = finish_reason

        state.text_parts.extend(visible_text)
        return visible_text, reasoning_text

    @staticmethod
    def _stream_usage(raw_usage: Any) -> Optional[Usage]:
        if not isinstance(raw_usage, Mapping):
            return None
        input_tokens = raw_usage.get("prompt_tokens")
        output_tokens = raw_usage.get("completion_tokens")
        total_tokens = raw_usage.get("total_tokens")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (input_tokens, output_tokens, total_tokens)
        ):
            return None
        return Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

    def _record_stream_reasoning(
        self,
        state: _KimiStreamState,
        reasoning_text: list[str],
        *,
        on_reasoning: Optional[OnReasoning],
    ) -> None:
        if state.reasoning_collector is None or state.reasoning_response is None:
            return
        for text in reasoning_text:
            self._record_reasoning_event(
                state.reasoning_response,
                state.reasoning_collector,
                text,
                capture_reasoning=True,
                on_reasoning=on_reasoning,
            )

    async def _record_stream_reasoning_async(
        self,
        state: _KimiStreamState,
        reasoning_text: list[str],
        *,
        on_reasoning: Optional[AsyncOnReasoning],
    ) -> None:
        if state.reasoning_collector is None or state.reasoning_response is None:
            return
        for text in reasoning_text:
            await self._record_async_reasoning_event(
                state.reasoning_response,
                state.reasoning_collector,
                text,
                capture_reasoning=True,
                on_reasoning=on_reasoning,
            )

    @staticmethod
    def _accumulate_stream_tool_calls(
        raw_tool_calls: list[Any],
        tool_calls: dict[int, dict[str, Any]],
    ) -> None:
        """Retain OpenAI-style function fragments for stream finalization."""
        for raw_tool_call in raw_tool_calls:
            if not isinstance(raw_tool_call, Mapping):
                continue
            index = raw_tool_call.get("index")
            if isinstance(index, bool) or not isinstance(index, int):
                continue
            target = tool_calls.setdefault(index, {"function": {"arguments": ""}})
            for field in ("id", "type"):
                if raw_tool_call.get(field) is not None:
                    target[field] = raw_tool_call[field]
            function = raw_tool_call.get("function")
            if not isinstance(function, Mapping):
                continue
            target_function = target["function"]
            if function.get("name") is not None:
                target_function["name"] = function["name"]
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                target_function["arguments"] = (
                    f"{target_function.get('arguments', '')}{arguments}"
                )
            elif isinstance(arguments, Mapping):
                target_function["arguments"] = dict(arguments)

    def _finalize_stream(
        self,
        state: _KimiStreamState,
        *,
        capture_reasoning: bool,
        effective_schema: Optional[dict],
        response_model: Optional[Any],
    ) -> ChatResponse:
        """Reconstruct and normalize the terminal Kimi Chat Completions response."""
        _ = capture_reasoning
        response_data = dict(state.response_metadata)
        response_data["model"] = response_data.get("model") or self.model
        if state.usage:
            response_data["usage"] = dict(state.usage)
        message: dict[str, Any] = {"content": "".join(state.text_parts) or None}
        if state.tool_calls:
            message["tool_calls"] = [
                state.tool_calls[index] for index in sorted(state.tool_calls)
            ]
        response_data["choices"] = [
            {"message": message, "finish_reason": state.finish_reason},
        ]
        chat_response = self._finalize_stream_response(
            self._parse_response(response_data),
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
        """Validate the Chat Completions request and apply metadata rules."""
        self._reject_deferred_features(
            parallel_tool_calls=parallel_tool_calls,
        )
        request_context = self._prepare_chat_request(
            messages,
            tools,
            tool_choice,
            json_schema,
            response_model,
        )
        self._reject_file_parts(request_context.normalized_messages)
        validated_max_tokens = self._validate_max_tokens(max_tokens)
        temperature, top_p = self._validate_sampling_parameters(temperature, top_p)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": request_context.normalized_messages.to_openai(),
            "temperature": temperature,
            "top_p": top_p,
        }
        if validated_max_tokens is not None:
            payload["max_tokens"] = validated_max_tokens
        mapped_tools = self._map_tools(tools)
        if mapped_tools is not None:
            payload["tools"] = mapped_tools
        mapped_tool_choice = self._map_tool_choice(
            request_context.normalized_tool_choice,
        )
        if mapped_tool_choice is not None:
            payload["tool_choice"] = mapped_tool_choice
        if request_context.effective_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": self._to_kimi_structured_output_schema(
                    request_context.effective_schema,
                ),
            }
        self._apply_reasoning_options(payload, reasoning_level)
        if self.model_spec is None:
            return request_context, payload
        transformed_payload, _ = apply_request_rules(
            payload,
            self.model_spec.request_rules,
            model=self.model,
        )
        return request_context, transformed_payload

    @staticmethod
    def _reject_deferred_features(
        *,
        parallel_tool_calls: Optional[bool],
    ) -> None:
        if parallel_tool_calls is not None:
            raise NotImplementedError(
                "Kimi parallel_tool_calls control is not implemented because "
                "the Chat Completions API has no documented parameter for it",
            )

    def _apply_reasoning_options(
        self,
        payload: dict[str, Any],
        reasoning_level: Optional[str | int],
    ) -> None:
        """Serialize only registry-resolved Kimi reasoning controls.

        Kimi keeps no adapter-managed server state.  The common registry resolves
        the native value; this adapter only selects Kimi's documented wire shape
        and never synthesizes or logs readable reasoning history.
        """
        if reasoning_level is None:
            return

        provider_value = self._resolve_reasoning_level(reasoning_level).provider_value
        if provider_value is None:
            return
        if not isinstance(provider_value, str):
            raise TypeError("Kimi reasoning resolution must produce a string")
        reasoning_capability = (
            self.model_spec.reasoning_capability if self.model_spec else None
        )
        if not isinstance(reasoning_capability, CategoricalReasoningCapability):
            return
        if provider_value == "none":
            payload["thinking"] = {"type": "disabled"}
        elif "enabled" not in reasoning_capability.allowed_values:
            payload["reasoning_effort"] = provider_value

    @staticmethod
    def _to_kimi_structured_output_schema(schema: dict) -> dict:
        """Validate the shared portable profile without changing its meaning."""
        return validate_core_portable_schema(schema, provider="kimi")

    @staticmethod
    def _map_tools(tools: Optional[list[ToolSpec]]) -> Optional[list[dict[str, Any]]]:
        """Serialize provider-neutral tools as Kimi's OpenAI-compatible format."""
        if not tools:
            return None
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
    def _map_tool_choice(tool_choice: Optional[str]) -> Any:
        """Translate the canonical selection modes to Kimi's wire values."""
        if tool_choice is None or tool_choice in {"auto", "none"}:
            return tool_choice
        if tool_choice == "any":
            return "required"
        return {
            "type": "function",
            "function": {"name": tool_choice},
        }

    @staticmethod
    def _reject_file_parts(messages: Messages) -> None:
        if any(
            isinstance(message, UserMessage) and message.files
            for message in messages.items
        ):
            raise NotImplementedError("Kimi image and document inputs are not implemented yet")

    def _validate_max_tokens(self, max_tokens: Optional[int]) -> Optional[int]:
        if max_tokens is None:
            return None
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer for Kimi")
        if (
            self.model_spec is not None
            and max_tokens > self.model_spec.limits.max_output_tokens
        ):
            raise ValueError(
                f"max_tokens must not exceed {self.model_spec.limits.max_output_tokens} "
                f"for Kimi model {self.model!r}",
            )
        return max_tokens

    @staticmethod
    def _parse_response(
        response: Mapping[str, Any],
        *,
        capture_reasoning: bool = False,
    ) -> ChatResponse:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMAPIClientError(
                detail="Kimi Chat Completions response.choices must be a non-empty array",
            )
        choice = choices[0]
        if not isinstance(choice, Mapping) or not isinstance(choice.get("message"), Mapping):
            raise LLMAPIClientError(
                detail="Kimi Chat Completions response.choices[0].message must be an object",
            )
        content = choice["message"].get("content")
        if content is not None and not isinstance(content, str):
            raise LLMAPIClientError(
                detail="Kimi Chat Completions response content must be a string or null",
            )

        chat_response = ChatResponse.from_openai_response(dict(response))
        reasoning_content = choice["message"].get("reasoning_content")
        if capture_reasoning and isinstance(reasoning_content, str) and reasoning_content:
            chat_response.reasoning_events = [
                ReasoningEvent(
                    text=reasoning_content,
                    kind="summary",
                    index=0,
                    elapsed_s=0.0,
                    delta_s=0.0,
                ),
            ]
        usage = response.get("usage")
        if usage is None:
            return chat_response
        if not isinstance(usage, Mapping):
            raise LLMAPIClientError(
                detail="Kimi Chat Completions response.usage must be an object",
            )
        required_tokens = ("prompt_tokens", "completion_tokens", "total_tokens")
        if any(
            isinstance(usage.get(field), bool) or not isinstance(usage.get(field), int)
            for field in required_tokens
        ):
            raise LLMAPIClientError(
                detail="Kimi Chat Completions usage token counts must be integers",
            )
        cached_tokens = usage.get("cached_tokens")
        if cached_tokens is not None and (
            isinstance(cached_tokens, bool) or not isinstance(cached_tokens, int)
        ):
            raise LLMAPIClientError(
                detail="Kimi Chat Completions usage.cached_tokens must be an integer",
            )
        if cached_tokens is not None and not 0 <= cached_tokens <= usage["prompt_tokens"]:
            raise LLMAPIClientError(
                detail="Kimi Chat Completions usage.cached_tokens must not exceed prompt_tokens",
            )
        chat_response.usage = KimiUsage(
            input_tokens=usage["prompt_tokens"],
            output_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"],
            cached_tokens=cached_tokens,
        )
        return chat_response

    def _finalize_kimi_chat_response(
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
        """Replace the generic cache-miss estimate only when the split is reported."""
        cache_pricing: KimiCachePricing | None = CACHE_PRICING.get(self.model)
        usage = chat_response.usage
        if cache_pricing is None or not isinstance(usage, KimiUsage):
            return
        if usage.cached_tokens is None:
            chat_response.cost_input = None
            chat_response.cost_total = None
            return
        uncached_tokens = usage.input_tokens - usage.cached_tokens
        chat_response.currency = "USD"
        chat_response.cost_input = (
            usage.cached_tokens * cache_pricing.cache_hit_input_per_token
            + uncached_tokens * cache_pricing.cache_miss_input_per_token
        )
        chat_response.cost_output = usage.output_tokens * cache_pricing.output_per_token
        chat_response.cost_total = chat_response.cost_input + chat_response.cost_output


__all__ = ["KimiAdapter", "KimiUsage"]
