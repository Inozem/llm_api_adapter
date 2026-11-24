"""Native Kimi policy over Core's OpenAI-compatible Chat Completions flow."""

from __future__ import annotations

import copy
import logging
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
    _ChatRequestContext,
)
from llm_api_adapter.adapters.openai_compatible.adapter import OpenAICompatibleAdapter
from llm_api_adapter.adapters.structured_output import validate_core_portable_schema
from llm_api_adapter.errors.llm_api_error import LLMAPIClientError, ToolChoiceError
from llm_api_adapter.llm_registry.llm_registry import CategoricalReasoningCapability
from llm_api_adapter.llm_registry.request_rules import apply_request_rules
from llm_api_adapter.llms.streaming import StreamReasoningCollector
from llm_api_adapter.llms.transports import SSEEvent, SyncTransport, create_sync_transport
from llm_api_adapter.models.messages.chat_message import Message, Messages, UserMessage
from llm_api_adapter.models.messages.file_parts import DocumentPart, ImagePart
from llm_api_adapter.models.responses.chat_response import ChatResponse, Usage
from llm_api_adapter.models.tools import ToolSpec

from .clients import KimiAsyncClient, KimiSyncClient
from .clients.async_client import _KimiAsyncAdapterClientBinding
from .clients.sync_client import _KimiAdapterClientBinding
from .streaming import (
    _KimiStreamInvocation,
    close_async_layers,
    close_sync_consumer,
    observe_async_events,
    observe_sync_events,
)
from .usage import KimiUsage, parse_kimi_usage

logger = logging.getLogger(__name__)


@dataclass(repr=False)
class KimiAdapter(OpenAICompatibleAdapter):
    """Keep Kimi's registered policy while inheriting shared execution."""

    company: str = "kimi"
    base_url: ClassVar[str] = ""
    api_variant: ClassVar[str | None] = None
    _client: KimiSyncClient = field(init=False, repr=False, compare=False)
    _async_client: KimiAsyncClient = field(init=False, repr=False, compare=False)
    _kimi_stream_invocation: _KimiStreamInvocation | None = field(
        init=False,
        repr=False,
        compare=False,
        default=None,
    )

    def __post_init__(self) -> None:
        """Initialize native registry facts before creating native clients."""
        LLMAdapterBase.__post_init__(self)
        self._client = KimiSyncClient(create_sync_transport(self.transport))
        self._async_client = KimiAsyncClient()

    @property
    def _sync_transport(self) -> SyncTransport:
        """Retain the native deterministic transport injection seam."""
        return self._client.transport

    @_sync_transport.setter
    def _sync_transport(self, transport: SyncTransport) -> None:
        self._client.transport = transport

    def _create_sync_client(self) -> _KimiAdapterClientBinding:
        return self._client._adapter_binding(self.api_key)

    def _create_async_client(self) -> _KimiAsyncAdapterClientBinding:
        return self._async_client._adapter_binding(self.api_key)

    def _prepare_chat_request(
        self,
        messages: list[Message] | Messages,
        tools: Optional[list[ToolSpec]],
        tool_choice: Any,
        json_schema: Optional[dict],
        response_model: Optional[Any],
    ) -> _ChatRequestContext:
        """Reuse common normalization, then enforce native file/schema policy."""
        context = super()._prepare_chat_request(
            messages,
            tools,
            tool_choice,
            json_schema,
            response_model,
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
        """Build common Chat Completions fields, then apply native rules once."""
        self._validate_native_dispatch(use_responses_api, parallel_tool_calls)
        payload = super()._build_chat_params(
            normalized_messages=normalized_messages,
            use_responses_api=False,
            max_tokens=self._validate_max_tokens(max_tokens),
            temperature=temperature,
            top_p=top_p,
            reasoning_level=None,
            tools=tools,
            normalized_tool_choice=normalized_tool_choice,
            parallel_tool_calls=None,
            previous_response=None,
            effective_schema=effective_schema,
            capture_reasoning=capture_reasoning,
        )
        self._apply_reasoning_options(payload, reasoning_level)
        if self.model_spec is None:
            return payload
        transformed, _ = apply_request_rules(
            payload,
            self.model_spec.request_rules,
            model=self.model,
        )
        return transformed

    def _build_stream_params(
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
        """Use the same native request policy for shared stream execution."""
        return self._build_chat_params(
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

    @staticmethod
    def _validate_native_dispatch(
        use_responses_api: bool,
        parallel_tool_calls: Optional[bool],
    ) -> None:
        if use_responses_api:
            raise NotImplementedError("Kimi uses Chat Completions only")
        if parallel_tool_calls is not None:
            raise NotImplementedError(
                "Kimi parallel_tool_calls control is not implemented because "
                "the Chat Completions API has no documented parameter for it",
            )

    def _resolve_protocol_reasoning_effort(
        self,
        reasoning_level: str | int | None,
    ) -> str | None:
        """Leave native reasoning resolution to the native payload policy."""
        _ = reasoning_level
        return None

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
            raise TypeError("Kimi reasoning resolution must produce a string")
        capability = self.model_spec.reasoning_capability if self.model_spec else None
        if not isinstance(capability, CategoricalReasoningCapability):
            return
        if provider_value == "none":
            payload["thinking"] = {"type": "disabled"}
        elif "enabled" not in capability.allowed_values:
            payload["reasoning_effort"] = provider_value

    def _map_tool_choice_to_openai(self, tool_choice: Optional[str]) -> Any:
        if tool_choice is None:
            return None
        mode = tool_choice if tool_choice in {"auto", "none", "any"} else "tool"
        allowed_modes = (
            self.model_spec.request_rules.allowed_tool_choice_modes
            if self.model_spec is not None
            else None
        )
        if allowed_modes is not None and mode not in allowed_modes:
            allowed = ", ".join(sorted(allowed_modes))
            raise ToolChoiceError(
                detail=(
                    f"Model {self.model!r} does not support tool_choice mode "
                    f"{mode!r}; allowed modes: {allowed}."
                ),
            )
        if tool_choice in {"auto", "none"}:
            return tool_choice
        if tool_choice == "any":
            return "required"
        return {"type": "function", "function": {"name": tool_choice}}

    @staticmethod
    def _to_openai_structured_output_schema(schema: dict) -> dict:
        return validate_core_portable_schema(schema, provider="kimi")

    @staticmethod
    def _reject_unsupported_file_parts(messages: Messages) -> None:
        for message in messages.items:
            if not isinstance(message, UserMessage) or not message.files:
                continue
            for file_part in message.files:
                if isinstance(file_part, ImagePart) and file_part._is_url():
                    raise ValueError(
                        "Kimi does not support public ImagePart URLs; pass image "
                        "bytes or a data URI instead",
                    )
                if isinstance(file_part, DocumentPart):
                    raise ValueError(
                        "Kimi does not support DocumentPart because its Files API "
                        "cannot attach PDF bytes and public URLs through the same "
                        "Chat Completions contract",
                    )

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

    def _parse_chat_response(
        self,
        response: dict,
        *,
        use_responses_api: bool,
        capture_reasoning: bool,
    ) -> ChatResponse:
        if use_responses_api:
            raise NotImplementedError("Kimi uses Chat Completions only")
        return self._parse_response(response, capture_reasoning=capture_reasoning)

    def _normalize_chat_completions_response(self, response: dict) -> ChatResponse:
        return self._parse_response(response)

    def _normalize_stream_usage(
        self,
        raw_usage: object,
        *,
        input_field: str,
        output_field: str,
    ) -> Usage | None:
        _ = input_field, output_field
        try:
            return parse_kimi_usage(raw_usage)
        except LLMAPIClientError:
            return None

    def _validate_chat_completions_stream_event(self, event: object) -> None:
        """Retain Kimi's tolerant treatment of malformed/interim SSE frames."""
        _ = event

    def _update_legacy_stream_metadata(
        self,
        payload: Mapping[str, Any],
        legacy_response: dict[str, Any],
    ) -> None:
        for field_name in ("id", "model", "created"):
            if field_name in payload:
                legacy_response[field_name] = payload[field_name]
        usage = payload.get("usage")
        if isinstance(usage, Mapping):
            existing = legacy_response.get("usage")
            merged = dict(existing) if isinstance(existing, Mapping) else {}
            merged.update(usage)
            if merged:
                legacy_response["usage"] = merged

    def _consume_legacy_stream_choice(
        self,
        choice: Any,
        text_parts: list[str],
        refusal_parts: list[str],
        tool_calls: dict[int, dict[str, Any]],
    ) -> tuple[Optional[str], Optional[str]]:
        """Filter Kimi's native ignored refusal and malformed finish fields."""
        if not isinstance(choice, Mapping):
            return super()._consume_legacy_stream_choice(
                choice, text_parts, refusal_parts, tool_calls
            )
        native_choice = dict(choice)
        finish_reason = native_choice.get("finish_reason")
        if finish_reason is not None and not isinstance(finish_reason, str):
            native_choice["finish_reason"] = None
        delta = native_choice.get("delta")
        if isinstance(delta, Mapping) and "refusal" in delta:
            native_delta = dict(delta)
            native_delta.pop("refusal", None)
            native_choice["delta"] = native_delta
        return super()._consume_legacy_stream_choice(
            native_choice,
            text_parts,
            refusal_parts,
            tool_calls,
        )

    @staticmethod
    def _accumulate_legacy_tool_calls(
        raw_tool_calls: list[Any],
        tool_calls: dict[int, dict[str, Any]],
    ) -> None:
        """Ignore invalid indices and retain native Mapping arguments."""
        normalized_calls: list[dict[str, Any]] = []
        for raw_call in raw_tool_calls:
            if not isinstance(raw_call, Mapping):
                continue
            index = raw_call.get("index")
            if isinstance(index, bool) or not isinstance(index, int):
                continue
            call = dict(raw_call)
            function = call.get("function")
            if isinstance(function, Mapping):
                normalized_function = dict(function)
                arguments = normalized_function.get("arguments")
                if isinstance(arguments, Mapping):
                    normalized_function["arguments"] = dict(arguments)
                call["function"] = normalized_function
            normalized_calls.append(call)
        super(KimiAdapter, KimiAdapter)._accumulate_legacy_tool_calls(
            normalized_calls,
            tool_calls,
        )

    def _build_legacy_stream_response(
        self,
        legacy_response: dict[str, Any],
        text_parts: list[str],
        refusal_parts: list[str],
        tool_calls: dict[int, dict[str, Any]],
        finish_reason: Optional[str],
    ) -> ChatResponse:
        if not legacy_response.get("model"):
            legacy_response["model"] = self.model
        if not legacy_response.get("usage"):
            legacy_response.pop("usage", None)
        return super()._build_legacy_stream_response(
            legacy_response,
            text_parts,
            refusal_parts,
            tool_calls,
            finish_reason,
        )

    def _finalize_stream_response(
        self,
        chat_response: ChatResponse,
        *,
        reasoning_collector: StreamReasoningCollector | None = None,
        effective_schema: Optional[dict],
        response_model: Optional[Any],
    ) -> ChatResponse:
        invocation = self._kimi_stream_invocation
        if invocation is not None:
            reasoning_collector = invocation.collector
        return super()._finalize_stream_response(
            chat_response,
            reasoning_collector=reasoning_collector,
            effective_schema=effective_schema,
            response_model=response_model,
        )

    def _iter_provider_stream_events(
        self,
        events: Iterator[SSEEvent],
    ) -> Iterator[SSEEvent]:
        invocation = self._kimi_stream_invocation
        if invocation is None:
            return super()._iter_provider_stream_events(events)
        inherited = super()._iter_provider_stream_events(events)
        observed = observe_sync_events(self, inherited, invocation)
        invocation.provider_events = observed
        return observed

    def _aiter_provider_stream_events(
        self,
        events: AsyncIterator[SSEEvent],
    ) -> AsyncIterator[SSEEvent]:
        invocation = self._kimi_stream_invocation
        if invocation is None:
            return super()._aiter_provider_stream_events(events)
        inherited = super()._aiter_provider_stream_events(events)
        observed = observe_async_events(self, inherited, invocation)
        invocation.async_provider_events = observed
        return observed

    def _consume_chat_completions_stream(
        self,
        events: Iterator[Any],
        effective_schema: Optional[dict],
        response_model: Optional[Any],
        on_delta: Optional[OnDelta],
        on_tool_call: Optional[OnToolCall],
        on_done: Optional[OnDone],
        buffer_chars: Optional[int],
        on_chunk: Optional[OnChunk],
    ) -> Iterator[str]:
        consumer = super()._consume_chat_completions_stream(
            events,
            effective_schema,
            response_model,
            on_delta,
            on_tool_call,
            on_done,
            buffer_chars,
            on_chunk,
        )
        invocation = self._kimi_stream_invocation
        yield from close_sync_consumer(
            events,
            consumer,
            invocation,
        )

    def _consume_chat_completions_stream_async(
        self,
        events: AsyncIterator[SSEEvent],
        effective_schema: Optional[dict],
        response_model: Optional[Any],
        on_delta: Optional[AsyncOnDelta],
        on_tool_call: Optional[AsyncOnToolCall],
        on_done: Optional[AsyncOnDone],
        buffer_chars: Optional[int],
        on_chunk: Optional[AsyncOnChunk],
    ) -> AsyncIterator[str]:
        consumer = super()._consume_chat_completions_stream_async(
            events,
            effective_schema,
            response_model,
            on_delta,
            on_tool_call,
            on_done,
            buffer_chars,
            on_chunk,
        )
        invocation = self._kimi_stream_invocation
        if invocation is not None:
            invocation.async_consumer = consumer
        return consumer

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
        invocation = _new_stream_invocation(
            capture_reasoning=capture_reasoning,
            on_reasoning=on_reasoning,
        )
        view = copy.copy(self)
        view._kimi_stream_invocation = invocation
        yield from OpenAICompatibleAdapter.stream_chat(
            view,
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
        invocation = _new_stream_invocation(
            capture_reasoning=capture_reasoning,
            on_async_reasoning=on_reasoning,
        )
        view = copy.copy(self)
        view._kimi_stream_invocation = invocation
        delegated = OpenAICompatibleAdapter._astream_chat(
            view,
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
        try:
            async with view._closing_async_source(delegated):
                async for text in delegated:
                    yield text
        except BaseException:
            await close_async_layers(
                view,
                invocation.async_consumer,
                invocation.async_provider_events,
                suppress_errors=True,
            )
            raise
        else:
            await close_async_layers(
                view,
                invocation.async_consumer,
                invocation.async_provider_events,
                suppress_errors=False,
            )

    def _apply_response_pricing(self, chat_response: ChatResponse) -> None:
        LLMAdapterBase._apply_response_pricing(self, chat_response)

    def handle_error(
        self,
        error: Exception,
        error_message: Optional[str] = None,
    ) -> None:
        """Preserve native error identity while keeping diagnostics content-free."""
        _ = error_message
        logger.error("Kimi operation failed (%s)", type(error).__name__)
        raise error from None

    @staticmethod
    def _parse_usage(raw_usage: Any) -> KimiUsage | None:
        return parse_kimi_usage(raw_usage)

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
        message = choice["message"]
        content = message.get("content")
        if content is not None and not isinstance(content, str):
            raise LLMAPIClientError(
                detail="Kimi Chat Completions response content must be a string or null",
            )
        chat_response = ChatResponse.from_openai_response(dict(response))
        reasoning = message.get("reasoning_content")
        if capture_reasoning and isinstance(reasoning, str) and reasoning:
            from llm_api_adapter.models.responses.reasoning_event import ReasoningEvent

            chat_response.reasoning_events = [
                ReasoningEvent(
                    text=reasoning,
                    kind="summary",
                    index=0,
                    elapsed_s=0.0,
                    delta_s=0.0,
                ),
            ]
        raw_usage = response.get("usage")
        if raw_usage is not None:
            if not isinstance(raw_usage, Mapping):
                raise LLMAPIClientError(
                    detail="Kimi Chat Completions response.usage must be an object",
                )
            chat_response.usage = parse_kimi_usage(raw_usage)
        return chat_response


def _new_stream_invocation(
    *,
    capture_reasoning: bool,
    on_reasoning: Optional[OnReasoning] = None,
    on_async_reasoning: Optional[AsyncOnReasoning] = None,
) -> _KimiStreamInvocation:
    enabled = capture_reasoning
    return _KimiStreamInvocation(
        capture_reasoning=enabled,
        collector=StreamReasoningCollector() if enabled else None,
        response=ChatResponse() if enabled else None,
        on_reasoning=on_reasoning,
        on_async_reasoning=on_async_reasoning,
    )


__all__ = ["KimiAdapter", "KimiUsage"]
