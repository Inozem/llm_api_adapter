"""Mistral's native policy over the shared OpenAI Responses foundation."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, AsyncIterator, ClassVar, Iterator, List, Mapping, Optional

import llm_api_adapter.adapters.base_adapter as base_adapter_module
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
from llm_api_adapter.adapters.openai_compatible.responses.streaming import _ResponsesStreamState
from llm_api_adapter.adapters.structured_output import validate_core_portable_schema
from llm_api_adapter.errors.llm_api_error import LLMAPIClientError
from llm_api_adapter.llm_registry.llm_registry import (
    MeteredOperationSpec,
    Pricing,
    resolve_metered_operation_spec,
)
from llm_api_adapter.llms.streaming import StreamChunkBuffer, StreamReasoningCollector, StreamUsageTracker
from llm_api_adapter.llms.transports import SSEEvent, SyncTransport, create_sync_transport
from llm_api_adapter.models.messages.chat_message import Message, Messages
from llm_api_adapter.models.responses.chat_response import ChatResponse, CostLineItem, Usage
from llm_api_adapter.models.tools import ToolSpec

from . import documents
from .clients import MistralAsyncClient, MistralSyncClient
from .clients.sync_client import MISTRAL_RESPONSES_URL


@dataclass(repr=False)
class MistralAdapter(OpenAICompatibleAdapter):
    """Native Mistral adapter using inherited Responses execution and lifecycle."""

    company: str = "mistral"
    base_url: ClassVar[str] = ""
    api_variant: ClassVar[str | None] = None
    endpoint: str = MISTRAL_RESPONSES_URL
    _sync_client: MistralSyncClient = field(init=False, repr=False, compare=False)
    _async_client: MistralAsyncClient = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Initialize the established native registry contract and clients."""
        LLMAdapterBase.__post_init__(self)
        self._sync_client = MistralSyncClient(create_sync_transport(self.transport))
        self._async_client = MistralAsyncClient()

    @property
    def _sync_transport(self) -> SyncTransport:
        """Retain the historical transport injection hook."""
        return self._sync_client.transport

    @_sync_transport.setter
    def _sync_transport(self, transport: SyncTransport) -> None:
        self._sync_client.transport = transport

    def _create_sync_client(self) -> MistralSyncClient:
        self._sync_client.api_key = self.api_key
        self._sync_client.endpoint = self.endpoint
        self._sync_client._set_api_variant("responses")
        self._sync_client._adapter_post_payload = self._post_payload
        self._sync_client._adapter_stream_payload = self._stream_payload
        return self._sync_client

    def _create_async_client(self) -> MistralAsyncClient:
        self._async_client.api_key = self.api_key
        self._async_client.endpoint = self.endpoint
        self._async_client._set_api_variant("responses")
        self._async_client._adapter_apost_payload = self._apost_payload
        self._async_client._adapter_astream_payload = self._astream_payload
        return self._async_client

    def _validate_sampling_parameters(
        self, temperature: float, top_p: float
    ) -> tuple[float, float]:
        """Keep Mistral's established 1.5 temperature ceiling and order."""
        return (
            self._validate_parameter("temperature", temperature, 0, 1.5),
            self._validate_parameter("top_p", top_p, 0, 1),
        )

    def _prepare_chat_request(
        self,
        messages: Any,
        tools: Optional[List[ToolSpec]],
        tool_choice: Any,
        json_schema: Optional[dict],
        response_model: Optional[Any],
    ) -> _ChatRequestContext:
        """Validate Mistral's portable schema before any OCR dispatch."""
        context = super()._prepare_chat_request(
            messages, tools, tool_choice, json_schema, response_model
        )
        if context.effective_schema is None:
            return context
        return replace(
            context,
            effective_schema=validate_core_portable_schema(
                context.effective_schema, provider="mistral"
            ),
        )

    def _build_chat_params(
        self, *, normalized_messages: Messages, use_responses_api: bool,
        max_tokens: Optional[int], temperature: float, top_p: float,
        reasoning_level: Optional[str | int], tools: Optional[List[ToolSpec]],
        normalized_tool_choice: Optional[str], parallel_tool_calls: Optional[bool],
        previous_response: Optional[ChatResponse], effective_schema: Optional[dict],
        capture_reasoning: bool,
    ) -> dict[str, Any]:
        params = super()._build_chat_params(
            normalized_messages=normalized_messages, use_responses_api=use_responses_api,
            max_tokens=max_tokens, temperature=temperature, top_p=top_p,
            reasoning_level=reasoning_level, tools=tools,
            normalized_tool_choice=normalized_tool_choice,
            parallel_tool_calls=parallel_tool_calls, previous_response=previous_response,
            effective_schema=effective_schema, capture_reasoning=capture_reasoning,
        )
        if use_responses_api and parallel_tool_calls is not None:
            params["parallel_tool_calls"] = parallel_tool_calls
        return params

    def _build_stream_params(
        self, *, normalized_messages: Messages, use_responses_api: bool,
        max_tokens: Optional[int], temperature: float, top_p: float,
        reasoning_level: Optional[str | int], tools: Optional[List[ToolSpec]],
        normalized_tool_choice: Optional[str], parallel_tool_calls: Optional[bool],
        previous_response: Optional[ChatResponse], effective_schema: Optional[dict],
        capture_reasoning: bool,
    ) -> dict[str, Any]:
        params = super()._build_stream_params(
            normalized_messages=normalized_messages, use_responses_api=use_responses_api,
            max_tokens=max_tokens, temperature=temperature, top_p=top_p,
            reasoning_level=reasoning_level, tools=tools,
            normalized_tool_choice=normalized_tool_choice,
            parallel_tool_calls=parallel_tool_calls, previous_response=previous_response,
            effective_schema=effective_schema, capture_reasoning=capture_reasoning,
        )
        if use_responses_api and parallel_tool_calls is not None:
            params["parallel_tool_calls"] = parallel_tool_calls
        return params

    def _normalize_responses_response(
        self, response: dict[str, Any], *, capture_reasoning: bool
    ) -> ChatResponse:
        """Retain Mistral's established reported-total and cache validation."""
        chat_response = super()._normalize_responses_response(
            response, capture_reasoning=capture_reasoning
        )
        compatible_usage = chat_response.usage
        native_usage = self._parse_responses_usage(response.get("usage"))
        if native_usage is not None and compatible_usage is not None:
            if not getattr(compatible_usage, "_compatible_cache_usage_valid", True):
                native_usage._compatible_cache_usage_valid = False
        chat_response.usage = native_usage
        status = response.get("status")
        if status == "incomplete":
            details = response.get("incomplete_details")
            reason = details.get("reason") if isinstance(details, Mapping) else None
            chat_response.finish_reason = "length" if reason == "max_output_tokens" else "incomplete"
            chat_response.incomplete_reason = chat_response.finish_reason
        else:
            output = response.get("output")
            has_tool_call = isinstance(output, list) and any(
                isinstance(item, Mapping) and item.get("type") in {"function_call", "tool_call"}
                for item in output
            )
            chat_response.finish_reason = (
                "content_filter" if chat_response.refusal else
                "tool_calls" if has_tool_call else "stop"
            )
        return chat_response

    @staticmethod
    def _parse_responses_usage(value: object) -> Optional[Usage]:
        if not isinstance(value, Mapping):
            return None
        input_tokens = _nonnegative_int(value.get("input_tokens"))
        output_tokens = _nonnegative_int(value.get("output_tokens"))
        total_tokens = _nonnegative_int(value.get("total_tokens"))
        details = value.get("input_tokens_details")
        if details is not None and not isinstance(details, Mapping):
            raise LLMAPIClientError(detail="Mistral input_tokens_details must be an object")
        cached_tokens = None
        if isinstance(details, Mapping) and "cached_tokens" in details:
            cached_tokens = _nonnegative_int(details.get("cached_tokens"))
            if cached_tokens is None:
                raise LLMAPIClientError(detail="Mistral cached_tokens must be a non-negative integer")
        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens
        return Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cached_tokens=cached_tokens,
        )

    def _prepare_document_messages(
        self, messages: List[Message] | Messages, timeout_s: Optional[float]
    ) -> documents.PreparedDocuments:
        return documents.prepare_document_messages(self, messages, timeout_s)

    async def _prepare_document_messages_async(
        self, messages: List[Message] | Messages, timeout_s: Optional[float]
    ) -> documents.PreparedDocuments:
        return await documents.prepare_document_messages_async(self, messages, timeout_s)

    def _post_ocr_payload(
        self, payload: dict[str, Any], timeout_s: Optional[float]
    ) -> dict[str, Any]:
        return self._create_sync_client().ocr(
            headers=self._sync_client._headers(),
            payload=payload,
            timeout_s=timeout_s,
            http_error_handler=self._handle_http_error,
        )

    async def _apost_ocr_payload(
        self, payload: dict[str, Any], timeout_s: Optional[float]
    ) -> dict[str, Any]:
        return await self._create_async_client().ocr(
            headers=self._async_client._headers(),
            payload=payload,
            timeout_s=timeout_s,
            http_error_handler=self._handle_http_error,
        )

    def _post_payload(
        self, payload: dict[str, Any], timeout_s: Optional[float]
    ) -> dict[str, Any]:
        """Retain the native full-resource JSON submission compatibility hook."""
        client = self._create_sync_client()
        return client.chat(
            endpoint=self.endpoint, headers=client._headers(), payload=payload,
            timeout_s=timeout_s, http_error_handler=self._handle_http_error,
        )

    async def _apost_payload(
        self, payload: dict[str, Any], timeout_s: Optional[float]
    ) -> dict[str, Any]:
        """Retain the native async full-resource JSON submission compatibility hook."""
        client = self._create_async_client()
        return await client.chat(
            endpoint=self.endpoint, headers=client._headers(), payload=payload,
            timeout_s=timeout_s, http_error_handler=self._handle_http_error,
        )

    def _stream_payload(
        self, payload: dict[str, Any], timeout_s: Optional[float]
    ) -> Iterator[SSEEvent]:
        """Retain the native full-resource synchronous event submission hook."""
        client = self._create_sync_client()
        stream_payload = dict(payload)
        stream_payload["stream"] = True
        return client.stream_chat(
            endpoint=self.endpoint, headers=client._headers(), payload=stream_payload,
            timeout_s=timeout_s, http_error_handler=self._handle_http_error,
            stream_error_handler=self._handle_stream_error,
        )

    def _astream_payload(
        self, payload: dict[str, Any], timeout_s: Optional[float]
    ) -> AsyncIterator[SSEEvent]:
        """Retain the native full-resource asynchronous event submission hook."""
        client = self._create_async_client()
        stream_payload = dict(payload)
        stream_payload["stream"] = True
        return client.stream_chat(
            endpoint=self.endpoint, headers=client._headers(), payload=stream_payload,
            timeout_s=timeout_s, http_error_handler=self._handle_http_error,
            stream_error_handler=self._handle_stream_error,
        )

    def _ocr_meter(self) -> Optional[MeteredOperationSpec]:
        meter = resolve_metered_operation_spec(
            base_adapter_module.LLM_REGISTRY,
            self.service_provider,
            "ocr",
        )
        if meter is None or meter.model != documents.MISTRAL_OCR_MODEL:
            return None
        return meter

    @staticmethod
    def _apply_ocr_costs(
        response: ChatResponse,
        operations: tuple[documents.PreparedOCROperation, ...],
    ) -> None:
        if not operations:
            return
        complete = True
        costs: list[CostLineItem] = []
        for operation in operations:
            if operation.pages_processed is None or operation.meter is None:
                complete = False
                continue
            costs.append(CostLineItem(
                operation=operation.meter.name,
                model=operation.model,
                unit=operation.meter.unit,
                quantity=operation.pages_processed,
                rate=operation.meter.rate,
                currency=operation.meter.currency,
            ))
        response.apply_cost_breakdown(costs, accounting_complete=complete)

    def _apply_response_pricing(self, chat_response: ChatResponse) -> None:
        """Retain native Base pricing errors and tier-boundary behavior."""
        LLMAdapterBase._apply_response_pricing(self, chat_response)

    def chat(self, messages: List[Message] | Messages, max_tokens: Optional[int] = None,
             temperature: float = 1.0, top_p: float = 1.0,
             reasoning_level: Optional[str | int] = None, timeout_s: Optional[float] = None,
             tools: Optional[List[ToolSpec]] = None, tool_choice: Any = None,
             parallel_tool_calls: Optional[bool] = None,
             previous_response: Optional[ChatResponse] = None,
             json_schema: Optional[dict] = None, response_model: Optional[Any] = None,
             *, capture_reasoning: bool = False) -> ChatResponse:
        temperature, top_p = self._validate_sampling_parameters(temperature, top_p)
        context = self._prepare_chat_request(messages, tools, tool_choice, json_schema, response_model)
        reasoning_level = self._resolve_protocol_reasoning_effort(reasoning_level)
        prepared = self._prepare_document_messages(messages, timeout_s)
        response = super().chat(
            prepared.messages, max_tokens, temperature, top_p, reasoning_level, timeout_s,
            tools, tool_choice, parallel_tool_calls, previous_response, json_schema,
            response_model, capture_reasoning=capture_reasoning,
        )
        _ = context
        self._apply_ocr_costs(response, prepared.ocr_operations)
        return response

    async def achat(self, messages: List[Message] | Messages, max_tokens: Optional[int] = None,
                    temperature: float = 1.0, top_p: float = 1.0,
                    reasoning_level: Optional[str | int] = None,
                    timeout_s: Optional[float] = None, tools: Optional[List[ToolSpec]] = None,
                    tool_choice: Any = None, parallel_tool_calls: Optional[bool] = None,
                    previous_response: Optional[ChatResponse] = None,
                    json_schema: Optional[dict] = None, response_model: Optional[Any] = None,
                    *, capture_reasoning: bool = False) -> ChatResponse:
        temperature, top_p = self._validate_sampling_parameters(temperature, top_p)
        context = self._prepare_chat_request(messages, tools, tool_choice, json_schema, response_model)
        reasoning_level = self._resolve_protocol_reasoning_effort(reasoning_level)
        prepared = await self._prepare_document_messages_async(messages, timeout_s)
        response = await super().achat(
            prepared.messages, max_tokens, temperature, top_p, reasoning_level, timeout_s,
            tools, tool_choice, parallel_tool_calls, previous_response, json_schema,
            response_model, capture_reasoning=capture_reasoning,
        )
        _ = context
        self._apply_ocr_costs(response, prepared.ocr_operations)
        return response

    def stream_chat(
        self, messages: List[Message] | Messages, max_tokens: Optional[int] = None,
        temperature: float = 1.0, top_p: float = 1.0,
        reasoning_level: Optional[str | int] = None, timeout_s: Optional[float] = None,
        tools: Optional[List[ToolSpec]] = None, tool_choice: Any = None,
        parallel_tool_calls: Optional[bool] = None,
        previous_response: Optional[ChatResponse] = None, json_schema: Optional[dict] = None,
        response_model: Optional[Any] = None, on_delta: Optional[OnDelta] = None,
        on_tool_call: Optional[OnToolCall] = None, on_done: Optional[OnDone] = None,
        buffer_chars: Optional[int] = None, on_chunk: Optional[OnChunk] = None, *,
        capture_reasoning: bool = False, on_reasoning: Optional[OnReasoning] = None,
    ) -> Iterator[str]:
        temperature, top_p = self._validate_sampling_parameters(temperature, top_p)
        context = self._prepare_chat_request(messages, tools, tool_choice, json_schema, response_model)
        reasoning_level = self._resolve_protocol_reasoning_effort(reasoning_level)
        state = _ResponsesStreamState(
            chunk_buffer=StreamChunkBuffer(buffer_chars), usage_tracker=StreamUsageTracker(),
            reasoning_collector=StreamReasoningCollector() if capture_reasoning else None,
            reasoning_response=ChatResponse() if capture_reasoning else None,
        )
        yield from self._stream_chat_prepared(
            context, messages, max_tokens, temperature, top_p, reasoning_level, timeout_s,
            tools, parallel_tool_calls, previous_response, response_model, buffer_chars,
            on_delta, on_tool_call, on_done, on_chunk, capture_reasoning, on_reasoning,
            state,
        )

    def _stream_chat_prepared(
        self, context: _ChatRequestContext, messages: List[Message] | Messages,
        max_tokens: Optional[int], temperature: float, top_p: float,
        reasoning_level: Optional[str | int], timeout_s: Optional[float],
        tools: Optional[List[ToolSpec]], parallel_tool_calls: Optional[bool],
        previous_response: Optional[ChatResponse], response_model: Optional[Any],
        buffer_chars: Optional[int], on_delta: Optional[OnDelta],
        on_tool_call: Optional[OnToolCall], on_done: Optional[OnDone],
        on_chunk: Optional[OnChunk], capture_reasoning: bool,
        on_reasoning: Optional[OnReasoning],
        state: _ResponsesStreamState,
    ) -> Iterator[str]:
        reasoning_level = self._resolve_protocol_reasoning_effort(reasoning_level)
        prepared = self._prepare_document_messages(messages, timeout_s)
        context = replace(context, normalized_messages=prepared.messages)
        client = self._create_sync_client()
        params = self._build_stream_params(
            normalized_messages=context.normalized_messages, use_responses_api=True,
            max_tokens=max_tokens, temperature=temperature, top_p=top_p,
            reasoning_level=reasoning_level, tools=tools,
            normalized_tool_choice=context.normalized_tool_choice,
            parallel_tool_calls=parallel_tool_calls, previous_response=previous_response,
            effective_schema=context.effective_schema, capture_reasoning=capture_reasoning,
        )
        params["endpoint"] = self.endpoint
        events = client.stream(timeout=timeout_s, **params)
        def finalize_response(
            current: _ResponsesStreamState, *, capture_reasoning: bool,
            effective_schema: Optional[dict], response_model: Optional[Any],
        ) -> ChatResponse:
            response = super(MistralAdapter, self)._finalize_responses_stream(
                current, capture_reasoning=capture_reasoning,
                effective_schema=effective_schema, response_model=response_model,
            )
            self._apply_ocr_costs(response, prepared.ocr_operations)
            return response

        yield from self._run_sync_stream(
            events, state, consume_event=self._consume_responses_stream_event,
            finalize_response=finalize_response, effective_schema=context.effective_schema,
            response_model=response_model, on_delta=on_delta, on_tool_call=on_tool_call,
            on_done=on_done, on_chunk=on_chunk, capture_reasoning=capture_reasoning,
            on_reasoning=on_reasoning,
        )

    def astream_chat(
        self, messages: List[Message] | Messages, max_tokens: Optional[int] = None,
        temperature: float = 1.0, top_p: float = 1.0,
        reasoning_level: Optional[str | int] = None, timeout_s: Optional[float] = None,
        tools: Optional[List[ToolSpec]] = None, tool_choice: Any = None,
        parallel_tool_calls: Optional[bool] = None,
        previous_response: Optional[ChatResponse] = None, json_schema: Optional[dict] = None,
        response_model: Optional[Any] = None, on_delta: Optional[AsyncOnDelta] = None,
        on_tool_call: Optional[AsyncOnToolCall] = None, on_done: Optional[AsyncOnDone] = None,
        buffer_chars: Optional[int] = None, on_chunk: Optional[AsyncOnChunk] = None, *,
        capture_reasoning: bool = False, on_reasoning: Optional[AsyncOnReasoning] = None,
    ) -> AsyncIterator[str]:
        return self._astream_chat_prepared(
            messages, max_tokens, temperature, top_p, reasoning_level, timeout_s,
            tools, tool_choice, parallel_tool_calls, previous_response, json_schema,
            response_model, on_delta, on_tool_call, on_done, buffer_chars, on_chunk,
            capture_reasoning, on_reasoning,
        )

    async def _astream_chat_prepared(
        self, messages: List[Message] | Messages, max_tokens: Optional[int],
        temperature: float, top_p: float, reasoning_level: Optional[str | int],
        timeout_s: Optional[float], tools: Optional[List[ToolSpec]], tool_choice: Any,
        parallel_tool_calls: Optional[bool], previous_response: Optional[ChatResponse],
        json_schema: Optional[dict], response_model: Optional[Any],
        on_delta: Optional[AsyncOnDelta], on_tool_call: Optional[AsyncOnToolCall],
        on_done: Optional[AsyncOnDone], buffer_chars: Optional[int],
        on_chunk: Optional[AsyncOnChunk], capture_reasoning: bool,
        on_reasoning: Optional[AsyncOnReasoning],
    ) -> AsyncIterator[str]:
        temperature, top_p = self._validate_sampling_parameters(temperature, top_p)
        context = self._prepare_chat_request(messages, tools, tool_choice, json_schema, response_model)
        reasoning_level = self._resolve_protocol_reasoning_effort(reasoning_level)
        state = _ResponsesStreamState(
            chunk_buffer=StreamChunkBuffer(buffer_chars), usage_tracker=StreamUsageTracker(),
            reasoning_collector=StreamReasoningCollector() if capture_reasoning else None,
            reasoning_response=ChatResponse() if capture_reasoning else None,
        )
        prepared = await self._prepare_document_messages_async(messages, timeout_s)
        context = replace(context, normalized_messages=prepared.messages)
        client = self._create_async_client()
        params = self._build_stream_params(
            normalized_messages=context.normalized_messages, use_responses_api=True,
            max_tokens=max_tokens, temperature=temperature, top_p=top_p,
            reasoning_level=reasoning_level, tools=tools,
            normalized_tool_choice=context.normalized_tool_choice,
            parallel_tool_calls=parallel_tool_calls, previous_response=previous_response,
            effective_schema=context.effective_schema, capture_reasoning=capture_reasoning,
        )
        params["endpoint"] = self.endpoint
        events = client.stream(timeout=timeout_s, **params)
        def finalize_response(
            current: _ResponsesStreamState, *, capture_reasoning: bool,
            effective_schema: Optional[dict], response_model: Optional[Any],
        ) -> ChatResponse:
            response = super(MistralAdapter, self)._finalize_responses_stream(
                current, capture_reasoning=capture_reasoning,
                effective_schema=effective_schema, response_model=response_model,
            )
            self._apply_ocr_costs(response, prepared.ocr_operations)
            return response

        async with self._closing_async_source(events):
            async for text in self._run_async_stream(
                events, state, consume_event=self._consume_responses_stream_event_async,
                finalize_response=finalize_response, effective_schema=context.effective_schema,
                response_model=response_model, on_delta=on_delta, on_tool_call=on_tool_call,
                on_done=on_done, on_chunk=on_chunk, capture_reasoning=capture_reasoning,
                on_reasoning=on_reasoning,
            ):
                yield text

    def _handle_http_error(self, error: Any) -> None:
        self._sync_client._handle_http_error(error)

    def _handle_stream_error(self, event: SSEEvent) -> None:
        self._sync_client._handle_stream_error(event)


__all__ = ["MistralAdapter"]


def _nonnegative_int(value: object) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None
