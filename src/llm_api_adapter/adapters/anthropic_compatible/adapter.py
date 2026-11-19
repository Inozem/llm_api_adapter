"""Generic Anthropic Messages connection adapter."""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import math
from numbers import Real
from typing import Any, AsyncIterator, Iterator, List, Mapping, Optional
import warnings

from ...errors.config_errors import LLMConfigError, LLMReasoningLevelError
from ...errors.llm_api_error import (
    InvalidToolArgumentsError,
    LLMAPIClientError,
    LLMAPIError,
)
from ...llm_registry.reasoning import CANONICAL_REASONING_LEVELS
from ...llms.transports import SSEEvent, TransportRequest, validate_sync_transport
from ...models.messages.chat_message import Message, Messages
from ...models.responses.chat_response import ChatResponse, Usage
from ...models.tools.tool_spec import ToolSpec
from ..base_adapter import (
    AsyncOnChunk, AsyncOnDelta, AsyncOnDone, AsyncOnReasoning, AsyncOnToolCall,
    LLMAdapterBase, OnChunk, OnDelta, OnDone, OnReasoning, OnToolCall,
)
from ..compatible_connection import (
    _redact_compatible_diagnostic,
    _validate_api_root,
    _validate_compatible_pricing,
)
from ...llms.anthropic_compatible.sync_client import AnthropicCompatibleSyncClient
from ...llms.anthropic_compatible.async_client import AnthropicCompatibleAsyncClient
from .execution import (
    _MessagesPreparedRunners,
    _MessagesStreamState,
    _PreparedMessagesOperation,
)
from .payloads import build_generic_messages_payload, normalize_messages_response
from .streaming import MessagesBlockState
from .usage import (
    normalize_messages_response_usage,
    normalize_messages_usage,
    merge_messages_usage_snapshots,
)


@dataclass(repr=False)
class AnthropicCompatibleAdapter(_MessagesPreparedRunners, LLMAdapterBase):
    """Use an explicitly selected Messages API root and exact model name."""

    company: str = "anthropic_compatible"
    base_url: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model:
            raise ValueError("model must be a non-empty string")
        if not isinstance(self.api_key, str) or not self.api_key:
            raise ValueError("api_key must be a non-empty string")
        if self.company != "anthropic_compatible":
            raise LLMConfigError(message="company must be 'anthropic_compatible'.")
        if self.service_provider is None:
            self.service_provider = self.company
        elif self.service_provider != self.company:
            raise LLMConfigError(
                message="service_provider must match the selected compatible connection."
            )
        if not isinstance(self.service_provider, str) or not self.service_provider:
            raise LLMConfigError(message="service_provider must be a non-empty string.")
        self.base_url = _validate_api_root(
            self.base_url, resource_paths=("/messages",)
        )
        self.transport = validate_sync_transport(self.transport)
        self.model_spec = None
        self.is_reasoning = False
        self.is_adaptive_thinking = False
        if self.pricing is not None:
            self.pricing = deepcopy(_validate_compatible_pricing(self.pricing))

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(company={self.company!r}, model='<redacted>', "
            f"transport={self.transport!r}, api_key='***', base_url='<redacted>')"
        )

    def handle_error(self, error: Exception, error_message: Optional[str] = None):
        """Clone errors with redacted diagnostics while retaining public type."""
        _ = error_message
        if isinstance(error, InvalidToolArgumentsError):
            safe = InvalidToolArgumentsError(
                message="Invalid tool arguments.",
                detail="Messages tool arguments are invalid.",
            )
        elif isinstance(error, LLMAPIError):
            safe = type(error)(
                message=self._redact_diagnostic(error.message),
                detail=(
                    None if error.detail is None
                    else self._redact_diagnostic(error.detail)
                ),
            )
        elif isinstance(error, LLMConfigError):
            safe = type(error)(
                message=self._redact_diagnostic(error.message),
                detail=(
                    None if error.detail is None
                    else self._redact_diagnostic(error.detail)
                ),
            )
        elif isinstance(error, ValueError):
            safe = type(error)(self._redact_diagnostic(error))
        elif isinstance(error, (ImportError, TypeError)):
            safe = type(error)(self._redact_diagnostic(error))
        else:
            safe = LLMAPIClientError(detail="Messages request failed.")
        raise safe from None

    def _redact_diagnostic(self, value: object) -> str:
        return _redact_compatible_diagnostic(value, self.api_key)

    def _create_sync_client(self) -> AnthropicCompatibleSyncClient:
        return AnthropicCompatibleSyncClient(self.api_key, self.base_url, self.transport)

    def _create_async_client(self) -> AnthropicCompatibleAsyncClient:
        return AnthropicCompatibleAsyncClient(self.api_key, self.base_url, self.transport)

    def _prepare_transport_request(
        self, payload: dict[str, Any], timeout_s: Optional[float], *, stream: bool,
        client: Any,
    ) -> TransportRequest:
        wire_payload = dict(payload)
        if stream:
            wire_payload["stream"] = True
        return TransportRequest(
            url=f"{self.base_url}/messages",
            headers=client._headers(),
            payload=wire_payload,
            timeout=timeout_s,
        )

    def _normalize_prepared_response(
        self, raw: Any, *, capture_reasoning: bool
    ) -> ChatResponse:
        return normalize_messages_response(
            raw,
            capture_reasoning=capture_reasoning,
            usage_normalizer=normalize_messages_response_usage,
        )

    def _normalize_prepared_stream_response(
        self, raw: Any, *, capture_reasoning: bool
    ) -> ChatResponse:
        return self._normalize_prepared_response(raw, capture_reasoning=capture_reasoning)

    def _apply_response_pricing(self, chat_response: ChatResponse) -> None:
        """Apply only covered caller tiers and preserve usage completeness."""
        if not self.pricing or chat_response.usage is None:
            return
        input_tokens = chat_response.usage.input_tokens
        if input_tokens is None:
            return
        try:
            self.pricing.tier_for_prompt_tokens(input_tokens)
        except ValueError:
            # A finite caller schedule does not authorize extrapolation.
            return
        super()._apply_response_pricing(chat_response)

    def _record_prepared_usage(
        self, event: SSEEvent, state: _MessagesStreamState
    ) -> None:
        """Attach the current reported-only usage snapshot to subsequent chunks."""
        if not state.blocks.usage_snapshots:
            return
        event_type = event.event
        if event_type is None and isinstance(event.data, Mapping):
            event_type = event.data.get("type")
        if event_type not in {"message_start", "message_delta"}:
            return
        usage = normalize_messages_usage(
            merge_messages_usage_snapshots(state.blocks.usage_snapshots)
        )
        if usage is None:
            # A reported null/malformed final update must clear pending chunk
            # metadata; StreamUsageTracker intentionally ignores None for its
            # native callers, so this generic policy writes an explicit unknown.
            state.common.chunk_buffer.update_metadata(
                usage=Usage(None, None, None, None, None)
            )
            return
        state.common.usage_tracker.record(state.common.chunk_buffer, usage)

    def _finalize_messages_blocks(
        self, state: MessagesBlockState
    ) -> dict[str, Any]:
        response = super()._finalize_messages_blocks(state)
        if state.usage_snapshots:
            response["usage"] = merge_messages_usage_snapshots(
                state.usage_snapshots
            )
        return response

    def _validate_generic_reasoning(self, reasoning_level: Optional[str | int]) -> None:
        if reasoning_level is None:
            return
        if isinstance(reasoning_level, bool) or not isinstance(reasoning_level, (str, int)):
            raise LLMReasoningLevelError(
                message="Invalid reasoning level.",
                detail="reasoning_level must be a canonical string or non-negative integer.",
            )
        if isinstance(reasoning_level, int) and reasoning_level < 0:
            raise LLMReasoningLevelError(
                message="Invalid reasoning level.",
                detail="reasoning_level must be a non-negative integer.",
            )
        if isinstance(reasoning_level, str) and reasoning_level not in {
            *CANONICAL_REASONING_LEVELS
        }:
            raise LLMReasoningLevelError(message="Invalid reasoning level.")
        warnings.warn(
            "Reasoning controls are ignored for unverified compatible models.",
            UserWarning,
            stacklevel=3,
        )

    def _prepare_operation(
        self,
        messages: List[Message] | Messages,
        *,
        max_tokens: Optional[int],
        temperature: float = 1.0,
        top_p: float = 1.0,
        reasoning_level: Optional[str | int] = None,
        timeout_s: Optional[float] = None,
        tools: Optional[List[ToolSpec]] = None,
        tool_choice: Any = None,
        parallel_tool_calls: Optional[bool] = None,
        json_schema: Optional[dict] = None,
        response_model: Optional[Any] = None,
        capture_reasoning: bool = False,
        async_mode: bool = False,
        stream: bool = False,
    ) -> _PreparedMessagesOperation:
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        try:
            if self.pricing is not None:
                _validate_compatible_pricing(self.pricing)
            self._validate_sampling_scalar("temperature", temperature)
            self._validate_sampling_scalar("top_p", top_p)
            if parallel_tool_calls is not None and not isinstance(parallel_tool_calls, bool):
                raise ValueError("parallel_tool_calls must be a boolean or None")
            temperature, top_p = self._validate_sampling_parameters(temperature, top_p)
            self._validate_generic_reasoning(reasoning_level)
            context = self._prepare_chat_request(
                messages, tools, tool_choice, json_schema, response_model
            )
            payload = build_generic_messages_payload(
                model=self.model,
                messages=context.normalized_messages,
                max_tokens=max_tokens,
                tools=tools,
                tool_choice=context.normalized_tool_choice,
                parallel_tool_calls=parallel_tool_calls,
                response_schema=context.effective_schema,
                temperature=temperature,
                top_p=top_p,
            )
            client = self._create_async_client() if async_mode else self._create_sync_client()
            request = self._prepare_transport_request(
                payload, timeout_s, stream=stream, client=client
            )
            return _PreparedMessagesOperation(
                chat_context=context,
                transport_request=request,
                client=client,
                capture_reasoning=capture_reasoning,
            )
        except Exception as error:
            self.handle_error(error)

    @staticmethod
    def _validate_sampling_scalar(name: str, value: object) -> None:
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
        ):
            raise ValueError(f"{name} must be a finite number")

    def chat(self, messages: List[Message] | Messages, max_tokens: Optional[int] = None,
             temperature: float = 1.0, top_p: float = 1.0,
             reasoning_level: Optional[str | int] = None, timeout_s: Optional[float] = None,
             tools: Optional[List[ToolSpec]] = None, tool_choice: Any = None,
             parallel_tool_calls: Optional[bool] = None,
             previous_response: Optional[ChatResponse] = None,
             json_schema: Optional[dict] = None, response_model: Optional[Any] = None,
             *, capture_reasoning: bool = False) -> ChatResponse:
        _ = previous_response
        operation = self._prepare_operation(
            messages, max_tokens=max_tokens, temperature=temperature, top_p=top_p,
            reasoning_level=reasoning_level, timeout_s=timeout_s, tools=tools,
            tool_choice=tool_choice, parallel_tool_calls=parallel_tool_calls,
            json_schema=json_schema, response_model=response_model,
            capture_reasoning=capture_reasoning,
        )
        return self._run_prepared_chat(operation)

    async def achat(self, messages: List[Message] | Messages, max_tokens: Optional[int] = None,
                    temperature: float = 1.0, top_p: float = 1.0,
                    reasoning_level: Optional[str | int] = None,
                    timeout_s: Optional[float] = None, tools: Optional[List[ToolSpec]] = None,
                    tool_choice: Any = None, parallel_tool_calls: Optional[bool] = None,
                    previous_response: Optional[ChatResponse] = None,
                    json_schema: Optional[dict] = None, response_model: Optional[Any] = None,
                    *, capture_reasoning: bool = False) -> ChatResponse:
        _ = previous_response
        operation = self._prepare_operation(
            messages, max_tokens=max_tokens, temperature=temperature, top_p=top_p,
            reasoning_level=reasoning_level, timeout_s=timeout_s, tools=tools,
            tool_choice=tool_choice, parallel_tool_calls=parallel_tool_calls,
            json_schema=json_schema, response_model=response_model,
            capture_reasoning=capture_reasoning, async_mode=True,
        )
        return await self._run_prepared_achat(operation)

    def stream_chat(self, messages: List[Message] | Messages,
                    max_tokens: Optional[int] = None, temperature: float = 1.0,
                    top_p: float = 1.0, reasoning_level: Optional[str | int] = None,
                    timeout_s: Optional[float] = None, tools: Optional[List[ToolSpec]] = None,
                    tool_choice: Any = None, parallel_tool_calls: Optional[bool] = None,
                    previous_response: Optional[ChatResponse] = None,
                    json_schema: Optional[dict] = None, response_model: Optional[Any] = None,
                    on_delta: Optional[OnDelta] = None,
                    on_tool_call: Optional[OnToolCall] = None, on_done: Optional[OnDone] = None,
                    buffer_chars: Optional[int] = None, on_chunk: Optional[OnChunk] = None,
                    *, capture_reasoning: bool = False,
                    on_reasoning: Optional[OnReasoning] = None) -> Iterator[str]:
        _ = previous_response
        state = self._new_messages_stream_state(buffer_chars, capture_reasoning)
        operation = self._prepare_operation(
            messages, max_tokens=max_tokens, temperature=temperature, top_p=top_p,
            reasoning_level=reasoning_level, timeout_s=timeout_s, tools=tools,
            tool_choice=tool_choice, parallel_tool_calls=parallel_tool_calls,
            json_schema=json_schema, response_model=response_model,
            capture_reasoning=capture_reasoning, stream=True,
        )
        return self._run_prepared_stream(
            operation, state=state, buffer_chars=buffer_chars, on_delta=on_delta,
            on_tool_call=on_tool_call, on_done=on_done, on_chunk=on_chunk,
            on_reasoning=on_reasoning,
        )

    def astream_chat(self, messages: List[Message] | Messages,
                     max_tokens: Optional[int] = None, temperature: float = 1.0,
                     top_p: float = 1.0, reasoning_level: Optional[str | int] = None,
                     timeout_s: Optional[float] = None, tools: Optional[List[ToolSpec]] = None,
                     tool_choice: Any = None, parallel_tool_calls: Optional[bool] = None,
                     previous_response: Optional[ChatResponse] = None,
                     json_schema: Optional[dict] = None, response_model: Optional[Any] = None,
                     on_delta: Optional[AsyncOnDelta] = None,
                     on_tool_call: Optional[AsyncOnToolCall] = None,
                     on_done: Optional[AsyncOnDone] = None, buffer_chars: Optional[int] = None,
                     on_chunk: Optional[AsyncOnChunk] = None,
                     *, capture_reasoning: bool = False,
                     on_reasoning: Optional[AsyncOnReasoning] = None) -> AsyncIterator[str]:
        _ = previous_response
        async def run() -> AsyncIterator[str]:
            state = self._new_messages_stream_state(buffer_chars, capture_reasoning)
            operation = self._prepare_operation(
                messages, max_tokens=max_tokens, temperature=temperature, top_p=top_p,
                reasoning_level=reasoning_level, timeout_s=timeout_s, tools=tools,
                tool_choice=tool_choice, parallel_tool_calls=parallel_tool_calls,
                json_schema=json_schema, response_model=response_model,
                capture_reasoning=capture_reasoning, async_mode=True, stream=True,
            )
            inner = self._run_prepared_astream(
                operation, state=state, buffer_chars=buffer_chars, on_delta=on_delta,
                on_tool_call=on_tool_call, on_done=on_done, on_chunk=on_chunk,
                on_reasoning=on_reasoning,
            )
            async with self._closing_async_source(inner):
                async for text in inner:
                    yield text
        return run()


__all__ = ["AnthropicCompatibleAdapter"]
