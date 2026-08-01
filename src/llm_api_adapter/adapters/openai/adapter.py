from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, List, Optional

from ..base_adapter import (
    AsyncOnChunk,
    AsyncOnDelta,
    AsyncOnDone,
    AsyncOnReasoning,
    AsyncOnToolCall,
    LLMAdapterBase,
)
from ...errors.llm_api_error import LLMAPIError
from ...llms.openai.async_client import OpenAIAsyncClient
from ...llms.openai.sync_client import OpenAISyncClient
from ...models.messages.chat_message import Message, Messages
from ...models.responses.chat_response import ChatResponse
from ...models.tools import ToolSpec
from .payloads import _OpenAIPayloadMixin
from .streaming import _OpenAIStreamingMixin


@dataclass(repr=False)
class OpenAIAdapter(_OpenAIPayloadMixin, _OpenAIStreamingMixin, LLMAdapterBase):
    """OpenAI adapter facade for both sync and async APIs."""

    company: str = "openai"

    def chat(
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
        *,
        capture_reasoning: bool = False,
    ) -> ChatResponse:
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

        try:
            client = OpenAISyncClient(api_key=self.api_key)
            normalized_messages = request_context.normalized_messages
            use_responses_api = client._should_use_responses_api(self.model)
            params = self._build_chat_params(
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
            response = client.complete(timeout=timeout_s, **params)
            chat_response = self._parse_chat_response(
                response,
                use_responses_api=use_responses_api,
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
        *,
        capture_reasoning: bool = False,
    ) -> ChatResponse:
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

        try:
            client = OpenAIAsyncClient(api_key=self.api_key)
            normalized_messages = request_context.normalized_messages
            use_responses_api = client._should_use_responses_api(self.model)
            params = self._build_chat_params(
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
            response = await client.complete(timeout=timeout_s, **params)
            chat_response = self._parse_chat_response(
                response,
                use_responses_api=use_responses_api,
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
        on_delta: Optional[AsyncOnDelta] = None,
        on_tool_call: Optional[AsyncOnToolCall] = None,
        on_done: Optional[AsyncOnDone] = None,
        buffer_chars: Optional[int] = None,
        on_chunk: Optional[AsyncOnChunk] = None,
        *,
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
        max_tokens: Optional[int],
        temperature: float,
        top_p: float,
        reasoning_level: Optional[str | int],
        timeout_s: Optional[float],
        tools: Optional[List[ToolSpec]],
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
        client = OpenAIAsyncClient(api_key=self.api_key)
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
            async for text in self._consume_responses_stream_async(
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
            return
        async for text in self._consume_chat_completions_stream_async(
            events,
            effective_schema,
            response_model,
            on_delta,
            on_tool_call,
            on_done,
            buffer_chars,
            on_chunk,
        ):
            yield text


