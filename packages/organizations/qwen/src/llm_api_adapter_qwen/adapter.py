"""Qwen Model Studio adapter for the Frankfurt Messages-compatible API."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Mapping
from dataclasses import dataclass, field
import logging
from typing import Any, ClassVar, Optional
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
from llm_api_adapter.adapters.anthropic_compatible.adapter import (
    AnthropicCompatibleAdapter,
)
from llm_api_adapter.adapters.anthropic_compatible.payloads import (
    build_messages_payload,
)
from llm_api_adapter.errors.llm_api_error import LLMAPIClientError
from llm_api_adapter.llms.transports import SyncTransport, create_sync_transport
from llm_api_adapter.models.messages.chat_message import Message, Messages, UserMessage
from llm_api_adapter.models.messages.file_parts import DocumentPart
from llm_api_adapter.models.responses.chat_response import ChatResponse, Usage
from llm_api_adapter.models.tools.tool_spec import ToolSpec

from .clients.async_client import QwenMessagesAsyncClient
from .clients.sync_client import QwenMessagesSyncClient, validate_workspace_id
from .request_rules import forced_tool_reasoning_modes
from .streaming import QwenMessagesStreamParser, QwenMessagesStreamState


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _PreparedQwenOperation:
    """Capture routing and native preparation for one Qwen operation."""

    chat_context: Any = field(repr=False)
    client: Any = field(repr=False)
    api_key: str = field(repr=False)
    workspace_id: str
    payload: dict[str, Any] = field(repr=False)
    timeout_s: Optional[float]
    capture_reasoning: bool = False


@dataclass
class _QwenPreparedStreamState:
    """Adapt Qwen's native parser state to the shared Messages runner seam."""

    common: QwenMessagesStreamState = field(repr=False)

    @property
    def blocks(self) -> Any:
        return self.common


@dataclass(repr=False)
class QwenAdapter(AnthropicCompatibleAdapter):
    """Map the shared text-chat contract to Qwen Messages."""

    company: str = "qwen"
    base_url: ClassVar[None] = None
    _client: QwenMessagesSyncClient = field(init=False, repr=False, compare=False)
    _async_client: QwenMessagesAsyncClient = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        LLMAdapterBase.__post_init__(self)
        self._client = QwenMessagesSyncClient(create_sync_transport(self.transport))
        self._async_client = QwenMessagesAsyncClient()

    def __repr__(self) -> str:
        return LLMAdapterBase.__repr__(self)

    def handle_error(
        self,
        error: Exception,
        error_message: Optional[str] = None,
    ) -> None:
        """Keep Qwen's native public error identity and details."""
        LLMAdapterBase.handle_error(self, error, error_message)

    def _apply_response_pricing(self, chat_response: ChatResponse) -> None:
        """Keep the native registry pricing behavior for Qwen responses."""
        LLMAdapterBase._apply_response_pricing(self, chat_response)

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
        operation = self._prepare_operation(
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
            timeout_s=timeout_s,
            capture_reasoning=capture_reasoning,
        )
        _ = previous_response
        return self._run_prepared_chat(operation)

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
        operation = self._prepare_operation(
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
            timeout_s=timeout_s,
            capture_reasoning=capture_reasoning,
            async_mode=True,
        )
        _ = previous_response
        return await self._run_prepared_achat(operation)

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
        operation = self._prepare_operation(
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
            timeout_s=timeout_s,
            capture_reasoning=capture_reasoning,
        )
        _ = previous_response
        state = self._new_messages_stream_state(buffer_chars, capture_reasoning)
        runner = self._run_prepared_stream(
            operation,
            state=state,
            buffer_chars=buffer_chars,
            on_delta=on_delta,
            on_tool_call=on_tool_call,
            on_done=on_done,
            on_chunk=on_chunk,
            on_reasoning=on_reasoning,
        )
        return runner

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
        async def run() -> AsyncIterator[str]:
            operation = self._prepare_operation(
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
                timeout_s=timeout_s,
                capture_reasoning=capture_reasoning,
                async_mode=True,
            )
            state = self._new_messages_stream_state(
                buffer_chars, capture_reasoning
            )
            _ = previous_response
            runner = self._run_prepared_astream(
                operation,
                state=state,
                buffer_chars=buffer_chars,
                on_delta=on_delta,
                on_tool_call=on_tool_call,
                on_done=on_done,
                on_chunk=on_chunk,
                on_reasoning=on_reasoning,
            )
            async with self._closing_async_source(runner):
                async for text in runner:
                    yield text

        return run()

    def _prepare_operation(
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
        timeout_s: Optional[float],
        capture_reasoning: bool,
        async_mode: bool = False,
    ) -> _PreparedQwenOperation:
        """Validate, serialize, and capture one request before execution."""
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
        payload = build_messages_payload(
            model=self.model,
            messages=request_context.normalized_messages,
            max_tokens=validated_max_tokens,
            tools=tools,
            tool_choice=request_context.normalized_tool_choice,
            response_schema=request_context.effective_schema,
            temperature=temperature,
            top_p=top_p,
        )
        self._apply_reasoning_options(payload, reasoning_level)
        if self.model_spec is not None:
            declared_modes = forced_tool_reasoning_modes(self.model)
            normalized_mode = (
                "any"
                if request_context.normalized_tool_choice == "any"
                else "tool"
                if request_context.normalized_tool_choice
                not in {None, "auto", "none"}
                else request_context.normalized_tool_choice
            )
            if declared_modes and normalized_mode in declared_modes:
                self._disable_thinking_for_forced_tool_choice(
                    payload,
                    normalized_tool_choice=normalized_mode,
                )
        return _PreparedQwenOperation(
            chat_context=request_context,
            client=self._async_client if async_mode else self._client,
            api_key=self.api_key,
            workspace_id=validated_workspace_id,
            payload=payload,
            timeout_s=timeout_s,
            capture_reasoning=capture_reasoning,
        )

    def _dispatch_prepared_chat(self, operation: _PreparedQwenOperation) -> Any:
        return operation.client.chat(
            api_key=operation.api_key,
            workspace_id=operation.workspace_id,
            payload=operation.payload,
            timeout_s=operation.timeout_s,
        )

    async def _dispatch_prepared_achat(
        self,
        operation: _PreparedQwenOperation,
    ) -> Any:
        return await operation.client.chat(
            api_key=operation.api_key,
            workspace_id=operation.workspace_id,
            payload=operation.payload,
            timeout_s=operation.timeout_s,
        )

    def _dispatch_prepared_stream(self, operation: _PreparedQwenOperation) -> Any:
        return operation.client.stream(
            api_key=operation.api_key,
            workspace_id=operation.workspace_id,
            payload=operation.payload,
            timeout_s=operation.timeout_s,
        )

    def _normalize_prepared_response(
        self,
        raw: Any,
        *,
        capture_reasoning: bool,
    ) -> ChatResponse:
        return self._parse_response(raw, capture_reasoning=capture_reasoning)

    def _consume_messages_event(
        self,
        state: QwenMessagesStreamState,
        event: Any,
    ) -> tuple[Optional[str], Optional[str]]:
        """Keep Qwen's tolerant event policy in its existing parser."""
        return QwenMessagesStreamParser.consume_event(event, state)

    def _record_prepared_usage(
        self,
        event: Any,
        state: Any,
    ) -> None:
        """Usage is recorded by QwenMessagesStreamParser.consume_event."""
        _ = event, state

    def _finalize_messages_blocks(
        self,
        state: QwenMessagesStreamState,
    ) -> dict[str, Any]:
        return QwenMessagesStreamParser.finalize(state, model=self.model)

    def _new_messages_stream_state(
        self,
        buffer_chars: Optional[int],
        capture_reasoning: bool,
    ) -> _QwenPreparedStreamState:
        return _QwenPreparedStreamState(
            common=QwenMessagesStreamParser.new_state(
                buffer_chars=buffer_chars,
                capture_reasoning=capture_reasoning,
            )
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
        if isinstance(usage, Mapping):
            for field in ("input_tokens", "output_tokens"):
                if field not in usage:
                    continue
                token_count = usage[field]
                if (
                    isinstance(token_count, bool)
                    or not isinstance(token_count, int)
                    or token_count < 0
                ):
                    raise LLMAPIClientError(
                        detail=(
                            "Qwen Messages usage token counts must be "
                            "non-negative integers"
                        ),
                    )
            cache_read_tokens = None
            if "cache_read_input_tokens" in usage:
                cache_read_tokens = usage["cache_read_input_tokens"]
                if (
                    isinstance(cache_read_tokens, bool)
                    or not isinstance(cache_read_tokens, int)
                    or cache_read_tokens < 0
                ):
                    raise LLMAPIClientError(
                        detail=(
                            "Qwen cache_read_input_tokens must be a "
                            "non-negative integer"
                        ),
                    )
        else:
            cache_read_tokens = None

        chat_response = ChatResponse.from_anthropic_response(
            dict(response),
            **({"capture_reasoning": True} if capture_reasoning else {}),
        )
        if isinstance(usage, Mapping):
            input_tokens = usage.get("input_tokens")
            output_tokens = usage.get("output_tokens")
            if input_tokens is not None and cache_read_tokens is not None:
                input_tokens += cache_read_tokens
            total_tokens = (
                input_tokens + output_tokens
                if input_tokens is not None and output_tokens is not None
                else None
            )
            chat_response.usage = Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cached_tokens=cache_read_tokens,
            )
        return chat_response


__all__ = ["QwenAdapter"]
