"""Anthropic's verified native policy over the shared Messages runners."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, AsyncIterator, ClassVar, Iterator, List, Mapping, Optional

from ..base_adapter import (
    AsyncOnChunk, AsyncOnDelta, AsyncOnDone, AsyncOnReasoning, AsyncOnToolCall,
    LLMAdapterBase, OnChunk, OnDelta, OnDone, OnReasoning, OnToolCall, _ChatRequestContext,
)
from ...errors.llm_api_error import LLMAPIError
from ...llms.anthropic.async_client import ClaudeAsyncClient
from ...llms.anthropic.sync_client import ClaudeSyncClient
from ...models.messages.chat_message import Message, Messages
from ...models.responses.chat_response import ChatResponse
from ...models.tools.tool_spec import ToolSpec
from ..anthropic_compatible.adapter import AnthropicCompatibleAdapter
from .payloads import _AnthropicPayloadMixin
from .streaming import _AnthropicStreamingMixin


@dataclass(frozen=True)
class _NativePreparedOperation:
    """Native client binding captured privately for one prepared operation."""

    chat_context: _ChatRequestContext = field(repr=False)
    client: Any = field(repr=False)
    call_kwargs: Mapping[str, Any] = field(repr=False)
    capture_reasoning: bool = False


@dataclass(repr=False)
class AnthropicAdapter(_AnthropicPayloadMixin, _AnthropicStreamingMixin, AnthropicCompatibleAdapter):
    """Native Claude adapter; native policy is retained at its public boundary."""

    company: str = "anthropic"
    base_url: ClassVar[None] = None

    def __post_init__(self) -> None:
        # Keep registry, pricing, reasoning and initialization policy native.
        LLMAdapterBase.__post_init__(self)

    def __repr__(self) -> str:
        return LLMAdapterBase.__repr__(self)

    def _apply_response_pricing(self, chat_response: ChatResponse) -> None:
        """Retain native registry pricing semantics over generic caller tariffs."""
        LLMAdapterBase._apply_response_pricing(self, chat_response)

    def handle_error(self, error: Exception, error_message: Optional[str] = None):
        """Retain the native base adapter's error identity and diagnostics."""
        if error_message is None and not isinstance(error, LLMAPIError):
            error_message = getattr(error, "text", None) or str(error)
        return LLMAdapterBase.handle_error(self, error, error_message)

    def _create_sync_client(self) -> ClaudeSyncClient:
        return ClaudeSyncClient(api_key=self.api_key, transport=self.transport)

    def _create_async_client(self) -> ClaudeAsyncClient:
        return ClaudeAsyncClient(api_key=self.api_key)

    def _prepare_native_operation(
        self, messages: List[Message] | Messages, *, max_tokens: int,
        temperature: float, top_p: float, reasoning_level: Optional[str | int],
        timeout_s: Optional[float], tools: Optional[List[ToolSpec]], tool_choice: Any,
        parallel_tool_calls: Optional[bool], json_schema: Optional[dict],
        response_model: Optional[Any], capture_reasoning: bool, stream: bool,
        async_mode: bool = False,
    ) -> _NativePreparedOperation:
        try:
            temperature, top_p = self._validate_sampling_parameters(temperature, top_p)
            context = self._prepare_chat_request(messages, tools, tool_choice, json_schema, response_model)
            builder = self._build_stream_params if stream else self._build_chat_params
            params = builder(
                normalized_messages=context.normalized_messages, max_tokens=max_tokens,
                temperature=temperature, top_p=top_p, reasoning_level=reasoning_level,
                timeout_s=timeout_s, tools=tools,
                normalized_tool_choice=context.normalized_tool_choice,
                parallel_tool_calls=parallel_tool_calls, effective_schema=context.effective_schema,
                capture_reasoning=capture_reasoning,
            )
            kwargs = dict(params)
            kwargs.pop("model", None)
            client = self._create_async_client() if async_mode else self._create_sync_client()
            return _NativePreparedOperation(
                chat_context=context, client=client, capture_reasoning=capture_reasoning,
                call_kwargs=MappingProxyType(deepcopy({"model": self.model, **kwargs})),
            )
        except LLMAPIError as error:
            self.handle_error(error)
        except Exception as error:
            self.handle_error(error=error, error_message=getattr(error, "text", None) or str(error))

    def _dispatch_prepared_chat(self, operation: _NativePreparedOperation) -> Any:
        return operation.client.chat_completion(**operation.call_kwargs)

    async def _dispatch_prepared_achat(self, operation: _NativePreparedOperation) -> Any:
        return await operation.client.chat_completion(**operation.call_kwargs)

    def _dispatch_prepared_stream(self, operation: _NativePreparedOperation) -> Any:
        return operation.client.stream(**operation.call_kwargs)

    def _consume_dispatched_stream(
        self, operation: _NativePreparedOperation, events: Iterator[Any], **kwargs: Any
    ) -> Iterator[str]:
        """Enter the compatibility consumer after common dispatch and ownership begin."""
        yield from self._consume_stream(
            events, operation.chat_context.effective_schema,
            operation.chat_context.response_model, kwargs["on_delta"],
            kwargs["on_tool_call"], kwargs["on_done"], kwargs["buffer_chars"],
            kwargs["on_chunk"], operation.capture_reasoning, kwargs["on_reasoning"],
        )

    def _consume_stream(
        self, events: Iterator[Any], effective_schema: Optional[dict],
        response_model: Optional[Any], on_delta: Optional[OnDelta],
        on_tool_call: Optional[OnToolCall], on_done: Optional[OnDone],
        buffer_chars: Optional[int], on_chunk: Optional[OnChunk],
        capture_reasoning: bool, on_reasoning: Optional[OnReasoning],
    ) -> Iterator[str]:
        """Compatibility seam delegating consumption and callbacks to the common runner."""
        state = self._new_messages_stream_state(buffer_chars, capture_reasoning)
        yield from self._consume_prepared_source(
            events, state, capture_reasoning=capture_reasoning,
            effective_schema=effective_schema, response_model=response_model,
            on_delta=on_delta, on_tool_call=on_tool_call, on_done=on_done,
            on_chunk=on_chunk, on_reasoning=on_reasoning,
        )

    def chat(
        self, messages: List[Message] | Messages, max_tokens: int, temperature: float = 1.0,
        top_p: float = 1.0, reasoning_level: Optional[str | int] = None,
        timeout_s: Optional[float] = None, *, tools: Optional[List[ToolSpec]] = None,
        tool_choice: Optional[str | dict] = None, parallel_tool_calls: Optional[bool] = None,
        previous_response: Optional[ChatResponse] = None, json_schema: Optional[dict] = None,
        response_model: Optional[Any] = None, capture_reasoning: bool = False,
    ) -> ChatResponse:
        _ = previous_response
        operation = self._prepare_native_operation(
            messages, max_tokens=max_tokens, temperature=temperature, top_p=top_p,
            reasoning_level=reasoning_level, timeout_s=timeout_s, tools=tools,
            tool_choice=tool_choice, parallel_tool_calls=parallel_tool_calls,
            json_schema=json_schema, response_model=response_model,
            capture_reasoning=capture_reasoning, stream=False,
        )
        return self._run_prepared_chat(operation)

    async def achat(
        self, messages: List[Message] | Messages, max_tokens: int, temperature: float = 1.0,
        top_p: float = 1.0, reasoning_level: Optional[str | int] = None,
        timeout_s: Optional[float] = None, *, tools: Optional[List[ToolSpec]] = None,
        tool_choice: Optional[str | dict] = None, parallel_tool_calls: Optional[bool] = None,
        previous_response: Optional[ChatResponse] = None, json_schema: Optional[dict] = None,
        response_model: Optional[Any] = None, capture_reasoning: bool = False,
    ) -> ChatResponse:
        _ = previous_response
        operation = self._prepare_native_operation(
            messages, max_tokens=max_tokens, temperature=temperature, top_p=top_p,
            reasoning_level=reasoning_level, timeout_s=timeout_s, tools=tools,
            tool_choice=tool_choice, parallel_tool_calls=parallel_tool_calls,
            json_schema=json_schema, response_model=response_model,
            capture_reasoning=capture_reasoning, stream=False, async_mode=True,
        )
        return await self._run_prepared_achat(operation)

    def stream_chat(
        self, messages: List[Message] | Messages, max_tokens: int, temperature: float = 1.0,
        top_p: float = 1.0, reasoning_level: Optional[str | int] = None,
        timeout_s: Optional[float] = None, *, tools: Optional[List[ToolSpec]] = None,
        tool_choice: Optional[str | dict] = None, parallel_tool_calls: Optional[bool] = None,
        previous_response: Optional[ChatResponse] = None, json_schema: Optional[dict] = None,
        response_model: Optional[Any] = None, on_delta: Optional[OnDelta] = None,
        on_tool_call: Optional[OnToolCall] = None, on_done: Optional[OnDone] = None,
        buffer_chars: Optional[int] = None, on_chunk: Optional[OnChunk] = None,
        capture_reasoning: bool = False, on_reasoning: Optional[OnReasoning] = None,
    ) -> Iterator[str]:
        _ = previous_response
        operation = self._prepare_native_operation(
            messages, max_tokens=max_tokens, temperature=temperature, top_p=top_p,
            reasoning_level=reasoning_level, timeout_s=timeout_s, tools=tools,
            tool_choice=tool_choice, parallel_tool_calls=parallel_tool_calls,
            json_schema=json_schema, response_model=response_model,
            capture_reasoning=capture_reasoning, stream=True,
        )
        yield from self._run_prepared_stream(
            operation, buffer_chars=buffer_chars, on_delta=on_delta,
            on_tool_call=on_tool_call, on_done=on_done, on_chunk=on_chunk,
            on_reasoning=on_reasoning,
        )

    def astream_chat(
        self, messages: List[Message] | Messages, max_tokens: int, temperature: float = 1.0,
        top_p: float = 1.0, reasoning_level: Optional[str | int] = None,
        timeout_s: Optional[float] = None, *, tools: Optional[List[ToolSpec]] = None,
        tool_choice: Optional[str | dict] = None, parallel_tool_calls: Optional[bool] = None,
        previous_response: Optional[ChatResponse] = None, json_schema: Optional[dict] = None,
        response_model: Optional[Any] = None, on_delta: Optional[AsyncOnDelta] = None,
        on_tool_call: Optional[AsyncOnToolCall] = None, on_done: Optional[AsyncOnDone] = None,
        buffer_chars: Optional[int] = None, on_chunk: Optional[AsyncOnChunk] = None,
        capture_reasoning: bool = False, on_reasoning: Optional[AsyncOnReasoning] = None,
    ) -> AsyncIterator[str]:
        _ = previous_response

        async def run() -> AsyncIterator[str]:
            operation = self._prepare_native_operation(
                messages, max_tokens=max_tokens, temperature=temperature, top_p=top_p,
                reasoning_level=reasoning_level, timeout_s=timeout_s, tools=tools,
                tool_choice=tool_choice, parallel_tool_calls=parallel_tool_calls,
                json_schema=json_schema, response_model=response_model,
                capture_reasoning=capture_reasoning, stream=True, async_mode=True,
            )
            inner = self._run_prepared_astream(
                operation, buffer_chars=buffer_chars, on_delta=on_delta,
                on_tool_call=on_tool_call, on_done=on_done, on_chunk=on_chunk,
                on_reasoning=on_reasoning,
            )
            async with self._closing_async_source(inner):
                async for text in inner:
                    yield text

        return run()


__all__ = ["AnthropicAdapter"]
