"""Qwen Model Studio adapter for the Frankfurt Messages-compatible API."""

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
)
from llm_api_adapter.errors.llm_api_error import LLMAPIClientError, LLMAPIError
from llm_api_adapter.llms.transports import SyncTransport, create_sync_transport
from llm_api_adapter.models.messages.chat_message import Message, Messages
from llm_api_adapter.models.responses.chat_response import ChatResponse
from llm_api_adapter.models.tools.tool_spec import ToolSpec

from .clients.async_client import QwenMessagesAsyncClient
from .clients.sync_client import QwenMessagesSyncClient, validate_workspace_id
from .streaming import QwenMessagesStreamParser, QwenMessagesStreamState


@dataclass(repr=False)
class QwenAdapter(LLMAdapterBase):
    """Map the shared text-chat contract to Qwen Messages."""

    company: str = "qwen"
    _client: QwenMessagesSyncClient = field(init=False, repr=False, compare=False)
    _async_client: QwenMessagesAsyncClient = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self._client = QwenMessagesSyncClient(create_sync_transport(self.transport))
        self._async_client = QwenMessagesAsyncClient()

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
        workspace_id: str | None = None,
        capture_reasoning: bool = False,
    ) -> ChatResponse:
        """Create one Qwen Messages response in an explicit Frankfurt workspace."""
        workspace_id, payload = self._prepare_request_payload(
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
            workspace_id=workspace_id,
        )
        _ = previous_response

        try:
            response = self._client.chat(
                api_key=self.api_key,
                workspace_id=workspace_id,
                payload=payload,
                timeout_s=timeout_s,
            )
            chat_response = self._parse_response(
                response,
                capture_reasoning=capture_reasoning,
            )
            self._apply_response_pricing(chat_response)
            return chat_response
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
        workspace_id: str | None = None,
        capture_reasoning: bool = False,
    ) -> ChatResponse:
        """Create one Qwen Messages response without blocking the event loop."""
        workspace_id, payload = self._prepare_request_payload(
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
            workspace_id=workspace_id,
        )
        _ = previous_response

        try:
            response = await self._async_client.chat(
                api_key=self.api_key,
                workspace_id=workspace_id,
                payload=payload,
                timeout_s=timeout_s,
            )
            chat_response = self._parse_response(
                response,
                capture_reasoning=capture_reasoning,
            )
            self._apply_response_pricing(chat_response)
            return chat_response
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
        workspace_id: str | None = None,
        capture_reasoning: bool = False,
        on_reasoning: Optional[OnReasoning] = None,
    ) -> Iterator[str]:
        """Stream Qwen Messages text through the shared synchronous lifecycle."""
        workspace_id, payload = self._prepare_request_payload(
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
            workspace_id=workspace_id,
        )
        _ = previous_response
        state = QwenMessagesStreamParser.new_state(buffer_chars=buffer_chars)
        events = self._client.stream(
            api_key=self.api_key,
            workspace_id=workspace_id,
            payload=payload,
            timeout_s=timeout_s,
        )
        return self._run_sync_stream(
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
        workspace_id: str | None = None,
        capture_reasoning: bool = False,
        on_reasoning: Optional[AsyncOnReasoning] = None,
    ) -> AsyncIterator[str]:
        """Stream Qwen Messages text through the shared async lifecycle."""
        workspace_id, payload = self._prepare_request_payload(
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
            workspace_id=workspace_id,
        )
        _ = previous_response
        state = QwenMessagesStreamParser.new_state(buffer_chars=buffer_chars)
        events = self._async_client.stream(
            api_key=self.api_key,
            workspace_id=workspace_id,
            payload=payload,
            timeout_s=timeout_s,
        )
        return self._run_async_stream(
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
            capture_reasoning=capture_reasoning,
            on_reasoning=on_reasoning,
        )

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
        workspace_id: str | None,
    ) -> tuple[str, dict[str, Any]]:
        """Validate one shared request and serialize its Qwen Messages payload."""
        validated_workspace_id = validate_workspace_id(workspace_id)
        self._reject_deferred_features(
            reasoning_level=reasoning_level,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            json_schema=json_schema,
            response_model=response_model,
        )
        validated_max_tokens = self._validate_max_tokens(max_tokens)
        temperature, top_p = self._validate_sampling_parameters(temperature, top_p)
        normalized_messages = self._normalize_messages(messages)
        system_prompt, message_payload = normalized_messages.to_anthropic()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": message_payload,
            "max_tokens": validated_max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if system_prompt is not None:
            payload["system"] = system_prompt
        return validated_workspace_id, payload

    def _consume_stream_event(
        self,
        event: Any,
        state: QwenMessagesStreamState,
        *,
        on_chunk: Optional[OnChunk],
        on_delta: Optional[OnDelta],
        on_reasoning: Optional[OnReasoning],
    ) -> Iterator[str]:
        """Normalize one Qwen SSE event and emit its text delta, if any."""
        _ = on_reasoning
        text = QwenMessagesStreamParser.consume_event(event, state)
        if text is not None:
            yield from self._emit_stream_chunks(
                state.chunk_buffer.add(text),
                on_chunk,
                on_delta,
            )

    async def _consume_stream_event_async(
        self,
        event: Any,
        state: QwenMessagesStreamState,
        *,
        on_chunk: Optional[AsyncOnChunk],
        on_delta: Optional[AsyncOnDelta],
        on_reasoning: Optional[AsyncOnReasoning],
    ) -> AsyncIterator[str]:
        """Normalize one Qwen SSE event with async callback ordering."""
        _ = on_reasoning
        text = QwenMessagesStreamParser.consume_event(event, state)
        if text is None:
            return
        async for emitted_text in self._emit_async_stream_chunks(
            state.chunk_buffer.add(text),
            on_chunk,
            on_delta,
        ):
            yield emitted_text

    def _finalize_stream(
        self,
        state: QwenMessagesStreamState,
        *,
        capture_reasoning: bool,
        effective_schema: Optional[dict],
        response_model: Optional[Any],
    ) -> ChatResponse:
        """Finalize reconstructed Qwen stream usage, pricing, and callbacks."""
        chat_response = self._parse_response(
            QwenMessagesStreamParser.finalize(state, model=self.model),
            capture_reasoning=capture_reasoning,
        )
        return self._finalize_stream_response(
            chat_response,
            effective_schema=effective_schema,
            response_model=response_model,
        )

    @staticmethod
    def _validate_max_tokens(max_tokens: Optional[int]) -> int:
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer for Qwen Messages")
        return max_tokens

    @staticmethod
    def _reject_deferred_features(
        *,
        reasoning_level: Optional[str | int],
        tools: Optional[list[ToolSpec]],
        tool_choice: Any,
        parallel_tool_calls: Optional[bool],
        json_schema: Optional[dict],
        response_model: Optional[Any],
    ) -> None:
        if reasoning_level is not None:
            raise NotImplementedError("Qwen reasoning controls are not implemented yet")
        if tools:
            raise NotImplementedError("Qwen application tools are not implemented yet")
        if tool_choice is not None or parallel_tool_calls is not None:
            raise NotImplementedError("Qwen application tools are not implemented yet")
        if json_schema is not None or response_model is not None:
            raise NotImplementedError("Qwen structured output is not implemented yet")

    @staticmethod
    def _parse_response(
        response: Mapping[str, Any],
        *,
        capture_reasoning: bool,
    ) -> ChatResponse:
        content = response.get("content")
        if not isinstance(content, list) or not all(
            isinstance(block, Mapping) for block in content
        ):
            raise LLMAPIClientError(
                detail="Qwen Messages response.content must be an array of objects",
            )
        for block in content:
            if block.get("type") == "text" and not isinstance(block.get("text"), str):
                raise LLMAPIClientError(
                    detail="Qwen Messages text content blocks must contain a string",
                )
        usage = response.get("usage")
        if usage is not None and not isinstance(usage, Mapping):
            raise LLMAPIClientError(
                detail="Qwen Messages response.usage must be an object when present",
            )
        if isinstance(usage, Mapping) and any(
            isinstance(usage.get(field), bool)
            or not isinstance(usage.get(field), int)
            for field in ("input_tokens", "output_tokens")
            if field in usage
        ):
            raise LLMAPIClientError(
                detail="Qwen Messages usage token counts must be integers",
            )
        return ChatResponse.from_anthropic_response(
            dict(response),
            **({"capture_reasoning": True} if capture_reasoning else {}),
        )


__all__ = ["QwenAdapter"]
