"""Adapter for xAI's official Responses API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterator, List, Optional

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
)
from llm_api_adapter.errors.llm_api_error import LLMAPIClientError, LLMAPIError
from llm_api_adapter.llms.transports import SSEEvent
from llm_api_adapter.models.messages.chat_message import (
    AIMessage,
    Message,
    Messages,
    ToolMessage,
    UserMessage,
)
from llm_api_adapter.models.responses.chat_response import ChatResponse
from llm_api_adapter.models.tools import ToolSpec

from .clients import XAIResponsesAsyncClient, XAIResponsesSyncClient
from .streaming import XAIResponsesStreamParser, XAIResponsesStreamState


@dataclass(repr=False)
class XAIAdapter(LLMAdapterBase):
    """Map the shared text-chat contract to xAI's Responses API."""

    company: str = "xai"
    endpoint: str = "https://api.x.ai/v1/responses"
    _client: XAIResponsesSyncClient = field(init=False, repr=False, compare=False)
    _async_client: XAIResponsesAsyncClient = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self._client = XAIResponsesSyncClient(
            api_key=self.api_key,
            transport=self.transport,
            endpoint=self.endpoint,
        )
        self._async_client = XAIResponsesAsyncClient(
            api_key=self.api_key,
            endpoint=self.endpoint,
        )

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
        """Generate one text response through ``POST /v1/responses``."""
        parameters = self._prepare_responses_parameters(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            reasoning_level=reasoning_level,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            previous_response=previous_response,
            json_schema=json_schema,
            response_model=response_model,
            capture_reasoning=capture_reasoning,
        )
        try:
            response = self._client.create(
                model=self.model,
                timeout=timeout_s,
                **parameters,
            )
            return self._finalize_chat_response(
                self._parse_response(response),
                effective_schema=None,
                response_model=None,
            )
        except LLMAPIError as error:
            self.handle_error(error)
        except Exception as error:
            error_message = getattr(error, "text", None) or str(error)
            self.handle_error(error=error, error_message=error_message)

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
        """Generate one text response through the core async transport helper."""
        parameters = self._prepare_responses_parameters(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            reasoning_level=reasoning_level,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            previous_response=previous_response,
            json_schema=json_schema,
            response_model=response_model,
            capture_reasoning=capture_reasoning,
        )
        try:
            response = await self._async_client.create(
                model=self.model,
                timeout=timeout_s,
                **parameters,
            )
            return self._finalize_chat_response(
                self._parse_response(response),
                effective_schema=None,
                response_model=None,
            )
        except LLMAPIError as error:
            self.handle_error(error)
        except Exception as error:
            error_message = getattr(error, "text", None) or str(error)
            self.handle_error(error=error, error_message=error_message)

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
        """Stream visible Responses text through the shared sync lifecycle."""
        parameters = self._prepare_responses_parameters(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            reasoning_level=reasoning_level,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            previous_response=previous_response,
            json_schema=json_schema,
            response_model=response_model,
            capture_reasoning=capture_reasoning,
        )
        state = XAIResponsesStreamParser.new_state(buffer_chars=buffer_chars)
        events = self._client.stream(
            model=self.model,
            timeout=timeout_s,
            **parameters,
        )
        yield from self._run_sync_stream(
            events,
            state,
            consume_event=self._consume_stream_event,
            finalize_response=self._finalize_stream,
            effective_schema=None,
            response_model=None,
            on_delta=on_delta,
            on_tool_call=on_tool_call,
            on_done=on_done,
            on_chunk=on_chunk,
            capture_reasoning=False,
            on_reasoning=on_reasoning,
        )

    async def astream_chat(
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
        """Stream visible Responses text through the shared async lifecycle."""
        parameters = self._prepare_responses_parameters(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            reasoning_level=reasoning_level,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            previous_response=previous_response,
            json_schema=json_schema,
            response_model=response_model,
            capture_reasoning=capture_reasoning,
        )
        state = XAIResponsesStreamParser.new_state(buffer_chars=buffer_chars)
        events = self._async_client.stream(
            model=self.model,
            timeout=timeout_s,
            **parameters,
        )
        async for text in self._run_async_stream(
            events,
            state,
            consume_event=self._consume_stream_event_async,
            finalize_response=self._finalize_stream,
            effective_schema=None,
            response_model=None,
            on_delta=on_delta,
            on_tool_call=on_tool_call,
            on_done=on_done,
            on_chunk=on_chunk,
            capture_reasoning=False,
            on_reasoning=on_reasoning,
        ):
            yield text

    def _prepare_responses_parameters(
        self,
        *,
        messages: List[Message] | Messages,
        max_tokens: Optional[int],
        temperature: float,
        top_p: float,
        reasoning_level: Optional[str | int],
        tools: Optional[List[ToolSpec]],
        tool_choice: Any,
        parallel_tool_calls: Optional[bool],
        previous_response: Optional[ChatResponse],
        json_schema: Optional[dict],
        response_model: Optional[Any],
        capture_reasoning: bool,
    ) -> dict[str, Any]:
        self._reject_unsupported_options(
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            previous_response=previous_response,
            json_schema=json_schema,
            response_model=response_model,
            reasoning_level=reasoning_level,
            capture_reasoning=capture_reasoning,
        )
        temperature, top_p = self._validate_sampling_parameters(temperature, top_p)
        request_context = self._prepare_chat_request(
            messages,
            None,
            None,
            None,
            None,
        )
        normalized_messages = request_context.normalized_messages
        self._reject_file_and_tool_messages(normalized_messages)

        parameters: dict[str, Any] = {
            "input": normalized_messages.to_openai_responses_input(),
            "max_output_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        instructions = normalized_messages.to_openai_responses_instructions()
        if instructions is not None:
            parameters["instructions"] = instructions
        return {key: value for key, value in parameters.items() if value is not None}

    def _consume_stream_event(
        self,
        event: SSEEvent,
        state: XAIResponsesStreamState,
        *,
        on_chunk: Optional[OnChunk],
        on_delta: Optional[OnDelta],
        on_reasoning: Optional[OnReasoning],
    ) -> Iterator[str]:
        del on_reasoning
        delta = XAIResponsesStreamParser.consume_event(event, state)
        if delta is not None:
            yield from self._emit_stream_chunks(
                state.chunk_buffer.add(delta),
                on_chunk,
                on_delta,
            )

    async def _consume_stream_event_async(
        self,
        event: SSEEvent,
        state: XAIResponsesStreamState,
        *,
        on_chunk: Optional[AsyncOnChunk],
        on_delta: Optional[AsyncOnDelta],
        on_reasoning: Optional[AsyncOnReasoning],
    ) -> AsyncIterator[str]:
        del on_reasoning
        delta = XAIResponsesStreamParser.consume_event(event, state)
        if delta is not None:
            async for text in self._emit_async_stream_chunks(
                state.chunk_buffer.add(delta),
                on_chunk,
                on_delta,
            ):
                yield text

    def _finalize_stream(
        self,
        state: XAIResponsesStreamState,
        *,
        capture_reasoning: bool,
        effective_schema: Optional[dict],
        response_model: Optional[Any],
    ) -> ChatResponse:
        del capture_reasoning
        return self._finalize_stream_response(
            XAIResponsesStreamParser.finalize(state, model=self.model),
            effective_schema=effective_schema,
            response_model=response_model,
        )

    @staticmethod
    def _reject_unsupported_options(
        *,
        tools: Optional[List[ToolSpec]],
        tool_choice: Any,
        parallel_tool_calls: Optional[bool],
        previous_response: Optional[ChatResponse],
        json_schema: Optional[dict],
        response_model: Optional[Any],
        reasoning_level: Optional[str | int],
        capture_reasoning: bool,
    ) -> None:
        unsupported = {
            "tools": tools is not None,
            "tool_choice": tool_choice is not None,
            "parallel_tool_calls": parallel_tool_calls is not None,
            "previous_response": previous_response is not None,
            "json_schema": json_schema is not None,
            "response_model": response_model is not None,
            "reasoning_level": reasoning_level is not None,
            "capture_reasoning": capture_reasoning,
        }
        selected = [name for name, present in unsupported.items() if present]
        if selected:
            raise ValueError(
                "xAI text chat does not support these options yet: "
                + ", ".join(selected),
            )

    @staticmethod
    def _reject_file_and_tool_messages(messages: Messages) -> None:
        for message in messages.items:
            if isinstance(message, UserMessage) and message.files:
                raise ValueError("xAI file input is not available yet")
            if isinstance(message, ToolMessage) or (
                isinstance(message, AIMessage) and message.tool_calls
            ):
                raise ValueError("xAI tool-result messages are not available yet")

    @staticmethod
    def _parse_response(response: dict[str, Any]) -> ChatResponse:
        if response.get("object") != "response":
            raise LLMAPIClientError(
                detail="xAI Responses API returned an invalid response object",
            )
        if not isinstance(response.get("output"), list):
            raise LLMAPIClientError(
                detail="xAI Responses API response.output must be an array",
            )
        return ChatResponse.from_openai_responses_response(response)


__all__ = ["XAIAdapter"]
