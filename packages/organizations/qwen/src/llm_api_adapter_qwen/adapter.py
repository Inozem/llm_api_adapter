"""Qwen Model Studio adapter for the Frankfurt Messages-compatible API."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Mapping
from dataclasses import dataclass, field
import logging
from typing import Any, Optional
import warnings

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
from llm_api_adapter.adapters.structured_output import validate_core_portable_schema
from llm_api_adapter.errors.llm_api_error import LLMAPIClientError, LLMAPIError
from llm_api_adapter.llms.transports import SyncTransport, create_sync_transport
from llm_api_adapter.models.messages.chat_message import Message, Messages, UserMessage
from llm_api_adapter.models.messages.file_parts import DocumentPart
from llm_api_adapter.models.responses.chat_response import ChatResponse
from llm_api_adapter.models.tools.tool_spec import ToolSpec

from .clients.async_client import QwenMessagesAsyncClient
from .clients.sync_client import QwenMessagesSyncClient, validate_workspace_id
from .streaming import QwenMessagesStreamParser, QwenMessagesStreamState


logger = logging.getLogger(__name__)


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
        workspace_id, request_context, payload = self._prepare_request_payload(
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
            return self._finalize_chat_response(
                chat_response,
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
        workspace_id: str | None = None,
        capture_reasoning: bool = False,
    ) -> ChatResponse:
        """Create one Qwen Messages response without blocking the event loop."""
        workspace_id, request_context, payload = self._prepare_request_payload(
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
            return self._finalize_chat_response(
                chat_response,
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
        workspace_id: str | None = None,
        capture_reasoning: bool = False,
        on_reasoning: Optional[OnReasoning] = None,
    ) -> Iterator[str]:
        """Stream Qwen Messages text through the shared synchronous lifecycle."""
        workspace_id, request_context, payload = self._prepare_request_payload(
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
        state = QwenMessagesStreamParser.new_state(
            buffer_chars=buffer_chars,
            capture_reasoning=capture_reasoning,
        )
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
        workspace_id: str | None = None,
        capture_reasoning: bool = False,
        on_reasoning: Optional[AsyncOnReasoning] = None,
    ) -> AsyncIterator[str]:
        """Stream Qwen Messages text through the shared async lifecycle."""
        workspace_id, request_context, payload = self._prepare_request_payload(
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
        state = QwenMessagesStreamParser.new_state(
            buffer_chars=buffer_chars,
            capture_reasoning=capture_reasoning,
        )
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
            effective_schema=request_context.effective_schema,
            response_model=request_context.response_model,
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
    ) -> tuple[str, Any, dict[str, Any]]:
        """Validate one shared request and serialize its Qwen Messages payload."""
        validated_workspace_id = validate_workspace_id(workspace_id)
        self._reject_unsupported_features(parallel_tool_calls=parallel_tool_calls)
        request_context = self._prepare_chat_request(
            messages,
            tools,
            tool_choice,
            json_schema,
            response_model,
        )
        self._reject_document_parts(request_context.normalized_messages)
        validated_max_tokens = self._validate_max_tokens(max_tokens)
        temperature, top_p = self._validate_sampling_parameters(temperature, top_p)
        system_prompt, message_payload = request_context.normalized_messages.to_anthropic()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": message_payload,
            "max_tokens": validated_max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if system_prompt is not None:
            payload["system"] = system_prompt
        if tools:
            payload["tools"] = [self._to_qwen_tool(tool) for tool in tools]
        if request_context.normalized_tool_choice is not None:
            payload["tool_choice"] = self._to_qwen_tool_choice(
                request_context.normalized_tool_choice,
            )
        if request_context.effective_schema is not None:
            payload.setdefault("output_config", {})["format"] = {
                "type": "json_schema",
                "schema": self._to_qwen_structured_output_schema(
                    request_context.effective_schema,
                ),
            }
        self._apply_reasoning_options(payload, reasoning_level)
        self._disable_thinking_for_forced_tool_choice(
            payload,
            normalized_tool_choice=request_context.normalized_tool_choice,
        )
        return validated_workspace_id, request_context, payload

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
        text, thinking = QwenMessagesStreamParser.consume_event(event, state)
        if (
            thinking is not None
            and state.reasoning_collector is not None
            and state.reasoning_response is not None
        ):
            self._record_reasoning_event(
                state.reasoning_response,
                state.reasoning_collector,
                thinking,
                capture_reasoning=True,
                kind="summary",
                on_reasoning=on_reasoning,
            )
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
        text, thinking = QwenMessagesStreamParser.consume_event(event, state)
        if (
            thinking is not None
            and state.reasoning_collector is not None
            and state.reasoning_response is not None
        ):
            await self._record_async_reasoning_event(
                state.reasoning_response,
                state.reasoning_collector,
                thinking,
                capture_reasoning=True,
                kind="summary",
                on_reasoning=on_reasoning,
            )
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
            reasoning_collector=state.reasoning_collector,
            effective_schema=effective_schema,
            response_model=response_model,
        )

    @staticmethod
    def _validate_max_tokens(max_tokens: Optional[int]) -> int:
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer for Qwen Messages")
        return max_tokens

    @staticmethod
    def _reject_unsupported_features(
        *,
        parallel_tool_calls: Optional[bool],
    ) -> None:
        if parallel_tool_calls is not None:
            raise NotImplementedError(
                "Qwen parallel tool-call controls are not implemented yet",
            )

    @staticmethod
    def _reject_document_parts(messages: Messages) -> None:
        """Reject PDFs before Qwen serializes or sends a Messages request."""
        if any(
            isinstance(file, DocumentPart)
            for message in messages.items
            if isinstance(message, UserMessage) and message.files
            for file in message.files
        ):
            raise ValueError(
                "Qwen does not support DocumentPart; PDF and OCR are unavailable "
                "in Qwen 0.1.0.",
            )

    def _apply_reasoning_options(
        self,
        payload: dict[str, Any],
        reasoning_level: Optional[str | int],
    ) -> None:
        """Serialize registry-resolved Qwen thinking controls."""
        if reasoning_level is None:
            return

        provider_value = self._resolve_reasoning_level(reasoning_level).provider_value
        if isinstance(provider_value, int):
            payload["thinking"] = (
                {"type": "disabled"}
                if provider_value == 0
                else {"type": "enabled", "budget_tokens": provider_value}
            )
            return
        if isinstance(provider_value, str):
            if provider_value == "none":
                payload["thinking"] = {"type": "disabled"}
            else:
                payload.setdefault("output_config", {})["effort"] = provider_value

    @staticmethod
    def _disable_thinking_for_forced_tool_choice(
        payload: dict[str, Any],
        *,
        normalized_tool_choice: str | None,
    ) -> None:
        """Apply Model Studio's forced-tool restriction before transport."""
        if normalized_tool_choice in {None, "auto", "none"}:
            return

        thinking = payload.get("thinking")
        if isinstance(thinking, Mapping) and thinking.get("type") == "disabled":
            return

        output_config = payload.get("output_config")
        if isinstance(output_config, dict):
            output_config.pop("effort", None)
            if not output_config:
                payload.pop("output_config")
        payload["thinking"] = {"type": "disabled"}

        message = (
            "Qwen disabled thinking because forced tool_choice ('any' or a named "
            "tool) is unsupported in thinking mode. Pass reasoning_level='none' "
            "to make this choice explicit."
        )
        warnings.warn(message, UserWarning, stacklevel=4)
        logger.warning(message)

    @staticmethod
    def _to_qwen_structured_output_schema(schema: dict) -> dict:
        """Validate the shared portable profile without changing its meaning."""
        return validate_core_portable_schema(schema, provider="qwen")

    @staticmethod
    def _to_qwen_tool(tool: ToolSpec) -> dict[str, Any]:
        """Map one validated Core tool to the Messages wire format."""
        payload: dict[str, Any] = {
            "name": tool.name,
            "input_schema": tool.json_schema,
        }
        if tool.description:
            payload["description"] = tool.description
        return payload

    @staticmethod
    def _to_qwen_tool_choice(normalized_tool_choice: str) -> dict[str, str]:
        """Map normalized Core tool selection to Model Studio Messages."""
        if normalized_tool_choice in {"auto", "none", "any"}:
            return {"type": normalized_tool_choice}
        return {"type": "tool", "name": normalized_tool_choice}

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
