"""Anthropic adapter facade for synchronous and asynchronous APIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Iterator, List, Optional

from ..base_adapter import (
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
)
from ...errors.llm_api_error import LLMAPIError
from ...llms.anthropic.async_client import ClaudeAsyncClient
from ...llms.anthropic.sync_client import ClaudeSyncClient
from ...models.messages.chat_message import Message, Messages
from ...models.responses.chat_response import ChatResponse
from ...models.tools.tool_spec import ToolSpec
from .payloads import _AnthropicPayloadMixin
from .streaming import _AnthropicStreamingMixin


@dataclass(repr=False)
class AnthropicAdapter(
    _AnthropicPayloadMixin,
    _AnthropicStreamingMixin,
    LLMAdapterBase,
):
    """Anthropic adapter facade for both sync and async APIs."""

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

    async def achat(
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
            client = ClaudeAsyncClient(api_key=self.api_key)
            response = await client.chat_completion(**params)
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

    def astream_chat(
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
        on_delta: Optional[AsyncOnDelta] = None,
        on_tool_call: Optional[AsyncOnToolCall] = None,
        on_done: Optional[AsyncOnDone] = None,
        buffer_chars: Optional[int] = None,
        on_chunk: Optional[AsyncOnChunk] = None,
        capture_reasoning: bool = False,
        on_reasoning: Optional[AsyncOnReasoning] = None,
    ) -> AsyncIterator[str]:
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
        messages: List[Message] | Messages,
        max_tokens: int,
        temperature: float,
        top_p: float,
        reasoning_level: Optional[str | int],
        timeout_s: Optional[float],
        tools: Optional[List[ToolSpec]],
        tool_choice: Optional[str | dict],
        parallel_tool_calls: Optional[bool],
        previous_response: Optional[ChatResponse],
        json_schema: Optional[dict],
        response_model: Optional[Any],
        on_delta: Optional[AsyncOnDelta],
        on_tool_call: Optional[AsyncOnToolCall],
        on_done: Optional[AsyncOnDone],
        buffer_chars: Optional[int],
        on_chunk: Optional[AsyncOnChunk],
        *,
        capture_reasoning: bool,
        on_reasoning: Optional[AsyncOnReasoning],
    ) -> AsyncIterator[str]:
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
        client = ClaudeAsyncClient(api_key=self.api_key)
        events = client.stream(**params)
        async for text in self._consume_stream_async(
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
        ):
            yield text

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


__all__ = ["AnthropicAdapter"]
