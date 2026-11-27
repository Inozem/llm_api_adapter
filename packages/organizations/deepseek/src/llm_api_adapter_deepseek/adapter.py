"""Core facade adapter for DeepSeek's official Responses API."""

from __future__ import annotations

import base64
import binascii
from copy import deepcopy
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from datetime import datetime, time, timezone
import logging
from math import isfinite
from typing import Any, AsyncIterator, ClassVar, Iterator, List, Mapping, Optional
from urllib.parse import urlparse
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
    _ChatRequestContext,
)
from llm_api_adapter.adapters.openai_compatible.adapter import OpenAICompatibleAdapter
from llm_api_adapter.adapters.structured_output import (
    validate_core_portable_schema,
)
from llm_api_adapter.errors.llm_api_error import (
    InvalidToolSchemaError,
    LLMAPIError,
    LLMAPIClientError,
)
from llm_api_adapter.errors.config_errors import LLMConfigError
from llm_api_adapter.llms.transports import SSEEvent
from llm_api_adapter.models.messages.chat_message import (
    AIMessage,
    Message,
    Messages,
    Prompt,
    ToolMessage,
    UserMessage,
)
from llm_api_adapter.models.messages.file_parts import DocumentPart, ImagePart
from llm_api_adapter.models.responses.chat_response import ChatResponse, Usage
from llm_api_adapter.models.tools import ToolSpec

from .clients.sync_client import (
    DEEPSEEK_RESPONSES_URL,
    _BoundEventSource,
    DeepSeekResponsesSyncClient,
)
from .clients.async_client import DeepSeekResponsesAsyncClient
from .documents import (
    PDFSlot,
    pdf_slots,
    preserve_raw_pdf_parts,
    validate_raw_document_forms,
)
from .registry import ORGANIZATION_DATA
from .streaming import (
    DeepSeekResponsesStreamParser,
    DeepSeekResponsesStreamState,
    normalize_deepseek_usage,
)


_REASONING_REPLAY_KEY = "deepseek.reasoning_replay"
_SUPPORTED_IMAGE_MEDIA_TYPES = frozenset(
    {"image/jpeg", "image/png", "image/gif", "image/webp"}
)
_MAX_IMAGE_URL_LENGTH = 8192
_MAX_INLINE_IMAGE_BYTES = 32 * 1024 * 1024
_MAX_IMAGES_PER_REQUEST = 600
_DEEPSEEK_DISPATCH_TIME: ContextVar[Optional[datetime]] = ContextVar(
    "deepseek_dispatch_time",
    default=None,
)
_DEEPSEEK_COMPLETED_INVOCATION: ContextVar[Optional["_CompletedResponsesInvocation"]] = ContextVar(
    "deepseek_completed_invocation",
    default=None,
)
_DEEPSEEK_PDF_ACTIVE: ContextVar[bool] = ContextVar("deepseek_pdf_active", default=False)
logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    """Return an aware UTC instant for time-of-use pricing."""
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class _PreparedResponsesRequest:
    """Normalized request data shared by the adapter's request paths."""

    parameters: dict[str, Any]
    effective_schema: Optional[dict]
    response_model: Optional[Any]
    capture_reasoning: bool
    request_context: _ChatRequestContext
    pdf_slots: tuple[PDFSlot, ...] = ()


@dataclass
class DeepSeekUsage(Usage):
    """Provider usage with optional native reasoning token details."""

    cached_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None


@dataclass
class _CompletedResponsesInvocation:
    prepared: _PreparedResponsesRequest
    dispatch_time: Optional[datetime] = None
    upload_ids: tuple[str, ...] = ()


class _DeepSeekSyncRequestView:
    def __init__(self, client: DeepSeekResponsesSyncClient):
        self._client = client

    @staticmethod
    def _should_use_responses_api(model: str) -> bool:
        del model
        return True

    def complete(self, model: str, timeout: float | None = None, **parameters: Any):
        invocation = _DEEPSEEK_COMPLETED_INVOCATION.get()
        if invocation is None:
            raise RuntimeError("DeepSeek completed execution requires prepared state")
        invocation.dispatch_time = _utc_now()
        try:
            return self._client.create(model=model, timeout=timeout, **parameters)
        except LLMAPIError as error:
            if invocation.prepared.pdf_slots:
                raise DeepSeekAdapter._redact_pdf_error(error) from None
            raise


class _DeepSeekAsyncRequestView:
    def __init__(self, client: DeepSeekResponsesAsyncClient):
        self._client = client

    @staticmethod
    def _should_use_responses_api(model: str) -> bool:
        del model
        return True

    async def complete(self, model: str, timeout: float | None = None, **parameters: Any):
        invocation = _DEEPSEEK_COMPLETED_INVOCATION.get()
        if invocation is None:
            raise RuntimeError("DeepSeek completed execution requires prepared state")
        invocation.dispatch_time = _utc_now()
        try:
            return await self._client.create(model=model, timeout=timeout, **parameters)
        except LLMAPIError as error:
            if invocation.prepared.pdf_slots:
                raise DeepSeekAdapter._redact_pdf_error(error) from None
            raise


@dataclass(repr=False)
class DeepSeekAdapter(OpenAICompatibleAdapter):
    """Map Core's text-chat contract to DeepSeek Responses requests."""

    company: str = "deepseek"
    base_url: ClassVar[str] = ""
    api_variant: ClassVar[str | None] = None
    endpoint: str = DEEPSEEK_RESPONSES_URL
    _client: DeepSeekResponsesSyncClient = field(
        init=False,
        repr=False,
        compare=False,
    )
    _async_client: DeepSeekResponsesAsyncClient = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        LLMAdapterBase.__post_init__(self)
        self._client = DeepSeekResponsesSyncClient(
            api_key=self.api_key,
            transport=self.transport,
            endpoint=self.endpoint,
        )
        self._async_client = DeepSeekResponsesAsyncClient(
            api_key=self.api_key,
            endpoint=self.endpoint,
        )

    def _create_sync_client(self) -> "_DeepSeekSyncRequestView":
        return _DeepSeekSyncRequestView(self._client)

    def _create_async_client(self) -> "_DeepSeekAsyncRequestView":
        return _DeepSeekAsyncRequestView(self._async_client)

    def _validate_sampling_parameters(
        self,
        temperature: float,
        top_p: float,
    ) -> tuple[float, float]:
        if _DEEPSEEK_COMPLETED_INVOCATION.get() is not None:
            return temperature, top_p
        return LLMAdapterBase._validate_sampling_parameters(
            self,
            temperature,
            top_p,
        )

    def _prepare_chat_request(
        self,
        messages: Any,
        tools: Optional[List[ToolSpec]],
        tool_choice: Any,
        json_schema: Optional[dict],
        response_model: Optional[Any],
    ) -> _ChatRequestContext:
        invocation = _DEEPSEEK_COMPLETED_INVOCATION.get()
        if invocation is not None:
            return invocation.prepared.request_context
        return LLMAdapterBase._prepare_chat_request(
            self,
            messages,
            tools,
            tool_choice,
            json_schema,
            response_model,
        )

    def _build_chat_params(
        self,
        *,
        normalized_messages: Messages,
        use_responses_api: bool,
        max_tokens: Optional[int],
        temperature: float,
        top_p: float,
        reasoning_level: Optional[str | int],
        tools: Optional[List[ToolSpec]],
        normalized_tool_choice: Optional[str],
        parallel_tool_calls: Optional[bool],
        previous_response: Optional[ChatResponse],
        effective_schema: Optional[dict],
        capture_reasoning: bool,
    ) -> dict[str, Any]:
        if not use_responses_api:
            raise AssertionError("DeepSeek uses the Responses API")
        invocation = _DEEPSEEK_COMPLETED_INVOCATION.get()
        if invocation is None:
            raise RuntimeError("DeepSeek completed execution requires prepared state")
        return {"model": self.model, **invocation.prepared.parameters}

    def _normalize_responses_response(
        self,
        response: dict[str, Any],
        *,
        capture_reasoning: bool,
    ) -> ChatResponse:
        invocation = _DEEPSEEK_COMPLETED_INVOCATION.get()
        if invocation is not None and invocation.prepared.pdf_slots:
            response = self._scrub_pdf_response_metadata(response, invocation.upload_ids)
        chat_response = self._parse_response(
            response,
            capture_reasoning=capture_reasoning,
        )
        self._store_reasoning_replay(
            chat_response,
            response,
            upload_ids=invocation.upload_ids if invocation is not None else (),
            pdf_contents=(
                tuple(slot.content for slot in invocation.prepared.pdf_slots)
                if invocation is not None and invocation.prepared.pdf_slots
                else ()
            ),
        )
        return chat_response

    def handle_error(
        self,
        error: Exception,
        error_message: str | None = None,
    ) -> None:
        if _DEEPSEEK_PDF_ACTIVE.get():
            logger.error("DeepSeek Responses request failed while processing a PDF")
            raise
        LLMAdapterBase.handle_error(self, error, error_message)

    def _finalize_chat_response(
        self,
        chat_response: ChatResponse,
        *,
        effective_schema: Optional[dict],
        response_model: Optional[Any],
    ) -> ChatResponse:
        invocation = _DEEPSEEK_COMPLETED_INVOCATION.get()
        if invocation is None or invocation.dispatch_time is None:
            return super()._finalize_chat_response(
                chat_response,
                effective_schema=effective_schema,
                response_model=response_model,
            )

        token = _DEEPSEEK_DISPATCH_TIME.set(invocation.dispatch_time)
        try:
            response = super()._finalize_chat_response(
                chat_response,
                effective_schema=effective_schema,
                response_model=response_model,
            )
            if invocation.prepared.pdf_slots:
                response.apply_cost_breakdown([], accounting_complete=False)
            return response
        finally:
            _DEEPSEEK_DISPATCH_TIME.reset(token)

    def __repr__(self) -> str:
        return LLMAdapterBase.__repr__(self)

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
        prepared = self._prepare_responses_parameters(
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
        invocation = _CompletedResponsesInvocation(prepared)
        token_pdf = _DEEPSEEK_PDF_ACTIVE.set(bool(prepared.pdf_slots))
        token = _DEEPSEEK_COMPLETED_INVOCATION.set(invocation)
        try:
            if prepared.pdf_slots:
                invocation.upload_ids = self._upload_pdfs_sync(prepared, timeout_s)
                self._bind_pdf_upload_ids(prepared, invocation.upload_ids)
            return super().chat(
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
                capture_reasoning=capture_reasoning,
            )
        finally:
            _DEEPSEEK_COMPLETED_INVOCATION.reset(token)
            _DEEPSEEK_PDF_ACTIVE.reset(token_pdf)

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
        prepared = self._prepare_responses_parameters(
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
        invocation = _CompletedResponsesInvocation(prepared)
        token_pdf = _DEEPSEEK_PDF_ACTIVE.set(bool(prepared.pdf_slots))
        token = _DEEPSEEK_COMPLETED_INVOCATION.set(invocation)
        try:
            if prepared.pdf_slots:
                invocation.upload_ids = await self._upload_pdfs_async(prepared, timeout_s)
                self._bind_pdf_upload_ids(prepared, invocation.upload_ids)
            return await super().achat(
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
                capture_reasoning=capture_reasoning,
            )
        finally:
            _DEEPSEEK_COMPLETED_INVOCATION.reset(token)
            _DEEPSEEK_PDF_ACTIVE.reset(token_pdf)

    def stream_chat(
        self,
        messages: Any,
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
        """Stream visible Responses text through Core's sync lifecycle."""
        prepared = self._prepare_responses_parameters(
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
        state = DeepSeekResponsesStreamParser.new_state(
            buffer_chars=buffer_chars,
            capture_reasoning=prepared.capture_reasoning,
        )
        events = None
        runner = None
        primary_error = None
        try:
            upload_ids = self._upload_pdfs_sync(prepared, timeout_s) if prepared.pdf_slots else ()
            if prepared.pdf_slots:
                self._bind_pdf_upload_ids(prepared, upload_ids)
            dispatch_time = _utc_now()
            pdf_token = _DEEPSEEK_PDF_ACTIVE.set(bool(prepared.pdf_slots))
            try:
                raw_events = self._client.stream(
                    model=self.model,
                    timeout=timeout_s,
                    **prepared.parameters,
                )
                if prepared.pdf_slots:
                    raw_events = self._safe_pdf_events(raw_events, upload_ids)
                events = _BoundEventSource(raw_events)
            except LLMAPIError as error:
                if prepared.pdf_slots:
                    raise self._redact_pdf_error(error) from None
                raise
            finally:
                _DEEPSEEK_PDF_ACTIVE.reset(pdf_token)
            runner = self._run_sync_stream(
                events,
                state,
                consume_event=self._consume_stream_event,
                effective_schema=prepared.effective_schema,
                response_model=prepared.response_model,
                on_delta=on_delta,
                on_tool_call=on_tool_call,
                on_done=on_done,
                on_chunk=on_chunk,
                capture_reasoning=prepared.capture_reasoning,
                on_reasoning=on_reasoning,
                finalize_response=lambda state, **kwargs: self._finalize_stream(
                    state,
                    upload_ids=upload_ids,
                    pdf_contents=tuple(slot.content for slot in prepared.pdf_slots),
                    **kwargs,
                ),
            )
            while True:
                token = _DEEPSEEK_DISPATCH_TIME.set(dispatch_time)
                pdf_token = _DEEPSEEK_PDF_ACTIVE.set(bool(prepared.pdf_slots))
                try:
                    text = next(runner)
                except StopIteration:
                    break
                finally:
                    _DEEPSEEK_DISPATCH_TIME.reset(token)
                    _DEEPSEEK_PDF_ACTIVE.reset(pdf_token)
                yield text
        except BaseException as error:
            primary_error = error
            raise
        finally:
            close_errors = []
            for stream in (runner, events):
                close = getattr(stream, "close", None)
                if close is None:
                    continue
                try:
                    close()
                except BaseException as error:
                    close_errors.append(error)
            if primary_error is None and close_errors:
                raise close_errors[0]

    async def astream_chat(
        self,
        messages: Any,
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
        """Stream visible Responses text through Core's async lifecycle."""
        prepared = self._prepare_responses_parameters(
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
        state = DeepSeekResponsesStreamParser.new_state(
            buffer_chars=buffer_chars,
            capture_reasoning=prepared.capture_reasoning,
        )
        events = None
        runner = None
        primary_error = None
        try:
            upload_ids = await self._upload_pdfs_async(prepared, timeout_s) if prepared.pdf_slots else ()
            if prepared.pdf_slots:
                self._bind_pdf_upload_ids(prepared, upload_ids)
            dispatch_time = _utc_now()
            pdf_token = _DEEPSEEK_PDF_ACTIVE.set(bool(prepared.pdf_slots))
            try:
                raw_events = self._async_client.stream(
                    model=self.model,
                    timeout=timeout_s,
                    **prepared.parameters,
                )
                if prepared.pdf_slots:
                    raw_events = self._safe_pdf_events_async(raw_events, upload_ids)
                events = raw_events
            except LLMAPIError as error:
                if prepared.pdf_slots:
                    raise self._redact_pdf_error(error) from None
                raise
            finally:
                _DEEPSEEK_PDF_ACTIVE.reset(pdf_token)
            runner = self._run_async_stream(
                events,
                state,
                consume_event=self._consume_stream_event_async,
                effective_schema=prepared.effective_schema,
                response_model=prepared.response_model,
                on_delta=on_delta,
                on_tool_call=on_tool_call,
                on_done=on_done,
                on_chunk=on_chunk,
                capture_reasoning=prepared.capture_reasoning,
                on_reasoning=on_reasoning,
                finalize_response=lambda state, **kwargs: self._finalize_stream(
                    state,
                    upload_ids=upload_ids,
                    pdf_contents=tuple(slot.content for slot in prepared.pdf_slots),
                    **kwargs,
                ),
            )
            while True:
                token = _DEEPSEEK_DISPATCH_TIME.set(dispatch_time)
                pdf_token = _DEEPSEEK_PDF_ACTIVE.set(bool(prepared.pdf_slots))
                try:
                    text = await runner.__anext__()
                except StopAsyncIteration:
                    break
                finally:
                    _DEEPSEEK_DISPATCH_TIME.reset(token)
                    _DEEPSEEK_PDF_ACTIVE.reset(pdf_token)
                yield text
        except BaseException as error:
            primary_error = error
            raise
        finally:
            close_errors = []
            for stream in (runner, events):
                close = getattr(stream, "aclose", None)
                if close is None:
                    continue
                try:
                    await close()
                except BaseException as error:
                    close_errors.append(error)
            if primary_error is None and close_errors:
                raise close_errors[0]

    def _consume_stream_event(
        self,
        event: Any,
        state: DeepSeekResponsesStreamState,
        *,
        on_chunk: Optional[OnChunk],
        on_delta: Optional[OnDelta],
        on_reasoning: Optional[OnReasoning],
    ) -> Iterator[str]:
        event_type = self._stream_event_type(event)
        if state.terminal_event is not None or not isinstance(event_type, str):
            return
        self._record_deepseek_reasoning_event(
            event,
            state,
            on_reasoning=on_reasoning,
        )
        delta = DeepSeekResponsesStreamParser.consume_event(
            event,
            state,
            record_text_delta=event_type != "response.output_text.delta",
        )
        if delta is not None and event_type == "response.output_text.delta":
            yield from super()._consume_responses_stream_event(
                SSEEvent(
                    event=event_type,
                    data={"type": event_type, "delta": delta},
                ),
                state,
                on_chunk=on_chunk,
                on_delta=on_delta,
                on_reasoning=None,
            )
        elif delta is not None:
            yield from self._emit_stream_chunks(
                state.chunk_buffer.add(delta), on_chunk, on_delta,
            )

    async def _consume_stream_event_async(
        self,
        event: Any,
        state: DeepSeekResponsesStreamState,
        *,
        on_chunk: Optional[AsyncOnChunk],
        on_delta: Optional[AsyncOnDelta],
        on_reasoning: Optional[AsyncOnReasoning],
    ) -> AsyncIterator[str]:
        event_type = self._stream_event_type(event)
        if state.terminal_event is not None or not isinstance(event_type, str):
            return
        await self._record_deepseek_reasoning_event_async(
            event,
            state,
            on_reasoning=on_reasoning,
        )
        delta = DeepSeekResponsesStreamParser.consume_event(
            event,
            state,
            record_text_delta=event_type != "response.output_text.delta",
        )
        if delta is not None and event_type == "response.output_text.delta":
            async for text in super()._consume_responses_stream_event_async(
                SSEEvent(
                    event=event_type,
                    data={"type": event_type, "delta": delta},
                ),
                state,
                on_chunk=on_chunk,
                on_delta=on_delta,
                on_reasoning=None,
            ):
                yield text
        elif delta is not None:
            async for text in self._emit_async_stream_chunks(
                state.chunk_buffer.add(delta),
                on_chunk,
                on_delta,
            ):
                yield text

    @staticmethod
    def _stream_event_type(event: Any) -> Any:
        payload = getattr(event, "data", None)
        return getattr(event, "event", None) or (
            payload.get("type") if isinstance(payload, Mapping) else None
        )

    def _finalize_stream(
        self,
        state: DeepSeekResponsesStreamState,
        *,
        capture_reasoning: bool,
        effective_schema: Optional[dict],
        response_model: Optional[Any],
        upload_ids: tuple[str, ...] = (),
        pdf_contents: tuple[bytes, ...] = (),
    ) -> ChatResponse:
        chat_response = self._finalize_stream_response(
            DeepSeekResponsesStreamParser.finalize(
                state,
                model=self.model,
                capture_reasoning=capture_reasoning,
            ),
            reasoning_collector=state.reasoning_collector,
            effective_schema=effective_schema,
            response_model=response_model,
        )
        if _DEEPSEEK_PDF_ACTIVE.get():
            chat_response.apply_cost_breakdown([], accounting_complete=False)
        self._store_reasoning_replay(
            chat_response,
            state.final_response or {},
            upload_ids=upload_ids,
            pdf_contents=pdf_contents,
        )
        return chat_response

    def _record_deepseek_reasoning_event(
        self,
        event: Any,
        state: DeepSeekResponsesStreamState,
        *,
        on_reasoning: Optional[OnReasoning],
    ) -> None:
        if state.terminal_event is not None:
            return
        if state.reasoning_collector is None or state.reasoning_response is None:
            return
        reasoning = DeepSeekResponsesStreamParser.reasoning_delta(event)
        if reasoning is None:
            return
        text, kind = reasoning
        self._record_reasoning_event(
            state.reasoning_response,
            state.reasoning_collector,
            text,
            capture_reasoning=True,
            kind=kind,
            on_reasoning=on_reasoning,
        )

    async def _record_deepseek_reasoning_event_async(
        self,
        event: Any,
        state: DeepSeekResponsesStreamState,
        *,
        on_reasoning: Optional[AsyncOnReasoning],
    ) -> None:
        if state.terminal_event is not None:
            return
        if state.reasoning_collector is None or state.reasoning_response is None:
            return
        reasoning = DeepSeekResponsesStreamParser.reasoning_delta(event)
        if reasoning is None:
            return
        text, kind = reasoning
        await self._record_async_reasoning_event(
            state.reasoning_response,
            state.reasoning_collector,
            text,
            capture_reasoning=True,
            kind=kind,
            on_reasoning=on_reasoning,
        )

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
    ) -> _PreparedResponsesRequest:
        """Normalize Core messages and supported text-request parameters."""
        self._validate_capability_preflight(parallel_tool_calls)
        reasoning_replay = self._replay_from_previous_response(previous_response)
        temperature, top_p = LLMAdapterBase._validate_sampling_parameters(
            self,
            temperature,
            top_p,
        )
        prepared_messages = validate_raw_document_forms(messages)
        request_context = LLMAdapterBase._prepare_chat_request(
            self,
            preserve_raw_pdf_parts(prepared_messages),
            tools,
            tool_choice,
            json_schema,
            response_model,
        )
        normalized_messages = request_context.normalized_messages
        reasoning_level = self._reasoning_level_for_function_tool_request(
            request_context.normalized_tool_choice,
            normalized_messages,
            reasoning_level,
        )
        pdf_attachments = pdf_slots(normalized_messages)
        self._validate_deepseek_file_inputs(normalized_messages)
        if pdf_attachments and not self._pdf_bytes_profile_is_enabled():
            raise NotImplementedError(
                f"DeepSeek model {self.model!r} is not registered for PDF byte inputs",
            )
        if pdf_attachments and self.endpoint != DEEPSEEK_RESPONSES_URL:
            raise ValueError("DeepSeek PDF uploads require the official Responses endpoint")
        effective_schema = request_context.effective_schema
        if effective_schema is not None:
            effective_schema = validate_core_portable_schema(
                effective_schema,
                provider="deepseek",
            )
            request_context = replace(
                request_context,
                effective_schema=effective_schema,
            )
        parameters: dict[str, Any] = {
            "input": self._to_deepseek_responses_input(
                normalized_messages,
                reasoning_replay=reasoning_replay,
                pdf_slots=pdf_attachments,
            ),
            "max_output_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "tools": self._map_tools_for_deepseek(tools),
            "tool_choice": self._map_tool_choice_to_openai_responses(
                request_context.normalized_tool_choice,
            ),
        }
        instructions = normalized_messages.to_openai_responses_instructions()
        if instructions is not None:
            parameters["instructions"] = instructions
        if reasoning_level is not None:
            provider_value = self._resolve_reasoning_level(
                reasoning_level,
            ).provider_value
            if isinstance(provider_value, str):
                parameters["reasoning"] = {"effort": provider_value}
        if effective_schema is not None:
            parameters["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "response",
                    "schema": effective_schema,
                },
            }
        return _PreparedResponsesRequest(
            parameters={
                key: value for key, value in parameters.items() if value is not None
            },
            effective_schema=effective_schema,
            response_model=request_context.response_model,
            capture_reasoning=capture_reasoning,
            request_context=request_context,
            pdf_slots=pdf_attachments,
        )

    def _pdf_bytes_profile_is_enabled(self) -> bool:
        """Require this exact model's registry profile to opt into PDF bytes."""
        if self.model_spec is None:
            return False
        try:
            profile = self.model_spec.require_capability_profile()
        except ValueError:
            return False
        matches = [item for item in profile if item.capability_id == "pdf_bytes"]
        return len(matches) == 1 and matches[0].behavior_id == "pass"

    def _reasoning_level_for_function_tool_request(
        self,
        normalized_tool_choice: Optional[str],
        messages: Messages,
        reasoning_level: Optional[str | int],
    ) -> Optional[str | int]:
        """Disable DeepSeek thinking throughout a named-function tool loop.

        DeepSeek rejects a named function choice while its default thinking mode
        is active, and requires reasoning text when a later tool-result request
        re-enters thinking mode. This is a provider protocol restriction, so
        preserve the portable tool loop and make the required mode change here.
        """
        named_tool_choice = normalized_tool_choice not in {
            None,
            "auto",
            "none",
            "any",
        }
        has_tool_result = any(
            isinstance(message, ToolMessage)
            for message in messages.items
        )
        if not named_tool_choice and not has_tool_result:
            return reasoning_level

        resolved_level = (
            self._resolve_reasoning_level(reasoning_level).provider_value
            if reasoning_level is not None
            else None
        )
        if resolved_level != "none":
            message = (
                "DeepSeek disables reasoning (reasoning_level='none') for a "
                "named tool_choice or tool-result continuation because its "
                "thinking mode rejects that function-tool combination."
            )
            warnings.warn(message, UserWarning, stacklevel=3)
            logger.warning(message)
        return "none"

    @staticmethod
    def _validate_deepseek_file_inputs(messages: Messages) -> None:
        """Validate image-specific limits after PDF and attachment preflight."""
        image_count = 0
        for message in messages.items:
            files = getattr(message, "files", None)
            if files is None:
                continue
            for part in files:
                if not isinstance(part, ImagePart):
                    continue
                image_count += 1
                if image_count > _MAX_IMAGES_PER_REQUEST:
                    raise ValueError(
                        "DeepSeek supports at most 600 images per request",
                    )
                DeepSeekAdapter._validate_image_part(part)

    def _upload_pdfs_sync(
        self,
        prepared: _PreparedResponsesRequest,
        timeout: Optional[float],
    ) -> tuple[str, ...]:
        """Upload this invocation's PDFs in order, without retaining receipts."""
        receipts = []
        try:
            for slot in prepared.pdf_slots:
                receipts.append(self._client._upload_pdf(slot.content, timeout))
            return tuple(receipts)
        except LLMAPIError as error:
            raise self._redact_pdf_error(error) from None

    async def _upload_pdfs_async(
        self,
        prepared: _PreparedResponsesRequest,
        timeout: Optional[float],
    ) -> tuple[str, ...]:
        """Upload this invocation's PDFs serially through the async client."""
        receipts = []
        try:
            for slot in prepared.pdf_slots:
                receipts.append(await self._async_client._upload_pdf(slot.content, timeout))
            return tuple(receipts)
        except LLMAPIError as error:
            raise self._redact_pdf_error(error) from None

    @staticmethod
    def _bind_pdf_upload_ids(
        prepared: _PreparedResponsesRequest,
        upload_ids: tuple[str, ...],
    ) -> None:
        """Replace only the private slot markers created during preflight."""
        if len(upload_ids) != len(prepared.pdf_slots):
            raise RuntimeError("DeepSeek PDF upload receipts do not match prepared slots")
        for slot_index, upload_id in enumerate(upload_ids):
            marker = f"__deepseek_pdf_slot_{slot_index}__"
            found = False
            for item in prepared.parameters["input"]:
                content = item.get("content") if isinstance(item, Mapping) else None
                if not isinstance(content, list):
                    continue
                for part in content:
                    if (
                        isinstance(part, dict)
                        and part.get("type") == "input_file"
                        and part.get("file_id") == marker
                    ):
                        part["file_id"] = upload_id
                        found = True
                        break
            if not found:
                raise RuntimeError("DeepSeek PDF slot marker is missing from prepared input")

    @staticmethod
    def _safe_pdf_events(events: Iterator[SSEEvent], upload_ids: tuple[str, ...]):
        primary_error = None
        try:
            for event in events:
                yield DeepSeekAdapter._redact_pdf_event(event, upload_ids)
        except LLMAPIError as error:
            primary_error = error
            raise DeepSeekAdapter._redact_pdf_error(error) from None
        except BaseException as error:
            primary_error = error
            raise
        finally:
            close = getattr(events, "close", None)
            if callable(close):
                try:
                    close()
                except BaseException:
                    if primary_error is None:
                        raise

    async def _safe_pdf_events_async(self, events, upload_ids: tuple[str, ...]):
        async with self._closing_async_source(events):
            try:
                async for event in events:
                    yield self._redact_pdf_event(event, upload_ids)
            except LLMAPIError as error:
                raise DeepSeekAdapter._redact_pdf_error(error) from None

    @staticmethod
    def _redact_pdf_event(event: SSEEvent, upload_ids: tuple[str, ...]) -> SSEEvent:
        if not isinstance(event.data, Mapping):
            return event
        event_type = event.event or event.data.get("type")
        failure_event = event_type in {"error", "response.failed"}
        response = event.data.get("response")
        if not failure_event and not isinstance(response, Mapping):
            return event
        payload = dict(event.data)
        if isinstance(response, Mapping):
            response = dict(response)
            metadata = response.get("metadata")
            if isinstance(metadata, Mapping):
                response["metadata"] = DeepSeekAdapter._scrub_pdf_metadata(
                    metadata,
                    upload_ids,
                )
            if failure_event and "error" in response:
                response["error"] = DeepSeekAdapter._scrub_provider_error(
                    response["error"],
                )
            payload["response"] = response
        if failure_event and "error" in payload:
            payload["error"] = DeepSeekAdapter._scrub_provider_error(payload["error"])
        if failure_event and isinstance(payload.get("message"), str):
            payload["message"] = "DeepSeek Responses request failed"
        return SSEEvent(
            event=event.event,
            data=payload,
            done=event.done,
        )

    @staticmethod
    def _scrub_pdf_response_metadata(
        response: dict[str, Any],
        upload_ids: tuple[str, ...],
    ) -> dict[str, Any]:
        result = dict(response)
        metadata = result.get("metadata")
        if isinstance(metadata, Mapping):
            result["metadata"] = DeepSeekAdapter._scrub_pdf_metadata(
                metadata,
                upload_ids,
            )
        return result

    @staticmethod
    def _scrub_pdf_metadata(
        metadata: Mapping[str, Any],
        upload_ids: tuple[str, ...],
    ) -> dict[str, Any]:
        private_fields = {
            "file_id",
            "file_data",
            "pdf_data",
            "pdf_bytes",
            "pdf_base64",
        }

        def scrub(value: Any) -> Any:
            if isinstance(value, Mapping):
                return {
                    key: scrub(item)
                    for key, item in value.items()
                    if str(key).lower() not in private_fields
                }
            if isinstance(value, list):
                return [scrub(item) for item in value]
            if isinstance(value, str):
                for upload_id in upload_ids:
                    value = value.replace(upload_id, "[redacted]")
            return value

        return scrub(metadata)

    @staticmethod
    def _scrub_provider_error(error: Any) -> Any:
        if isinstance(error, Mapping):
            cleaned = {}
            for key, value in error.items():
                if str(key).lower() in {"message", "detail", "error"}:
                    cleaned[key] = "DeepSeek Responses request failed"
                else:
                    cleaned[key] = value
            return cleaned
        return "DeepSeek Responses request failed"

    @staticmethod
    def _redact_pdf_error(error: LLMAPIError) -> LLMAPIError:
        """Replace provider details with a generic message for a PDF request."""
        return type(error)(detail="DeepSeek Responses request failed")

    @staticmethod
    def _validate_image_part(part: ImagePart) -> None:
        """Validate one image's media type, source form, and documented limits."""
        media_type = (part._get_media_type() or "").lower()
        if media_type not in _SUPPORTED_IMAGE_MEDIA_TYPES:
            raise ValueError(
                "DeepSeek supports only JPEG, PNG, GIF, and WebP images",
            )

        if part.url is not None:
            if not isinstance(part.url, str):
                raise ValueError("DeepSeek image URL must be a string")
            if part.url.startswith("data:"):
                prefix, separator, encoded = part.url.partition(",")
                prefix_parts = (
                    prefix[5:].split(";")
                    if prefix.lower().startswith("data:")
                    else []
                )
                declared_media_type = (
                    prefix_parts[0].lower() if prefix_parts else None
                )
                parameters = {value.lower() for value in prefix_parts[1:]}
                if (
                    not separator
                    or "base64" not in parameters
                    or declared_media_type != media_type
                ):
                    raise ValueError(
                        "DeepSeek image data URI must be base64 and match its media type",
                    )
                try:
                    decoded_size = len(base64.b64decode(encoded, validate=True))
                except (binascii.Error, ValueError):
                    raise ValueError(
                        "DeepSeek image data URI contains invalid base64",
                    ) from None
                if decoded_size == 0:
                    raise ValueError("DeepSeek image data URI must not be empty")
                if decoded_size > _MAX_INLINE_IMAGE_BYTES:
                    raise ValueError("DeepSeek inline images must be at most 32 MiB")
                return
            if len(part.url) > _MAX_IMAGE_URL_LENGTH:
                raise ValueError("DeepSeek image URLs must be at most 8192 characters")
            parsed = urlparse(part.url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("DeepSeek image URLs must use http(s)")
            return

        if part.data is None or not isinstance(part.data, bytes) or not part.data:
            raise ValueError("DeepSeek image data must be non-empty bytes")
        if len(part.data) > _MAX_INLINE_IMAGE_BYTES:
            raise ValueError("DeepSeek inline images must be at most 32 MiB")

    def _validate_capability_preflight(
        self,
        parallel_tool_calls: Optional[bool],
    ) -> None:
        """Reject DeepSeek modes outside the verified package contract."""
        if self.model_spec is None:
            raise NotImplementedError(
                f"DeepSeek model {self.model!r} is not verified for supported "
                "capabilities",
            )
        if parallel_tool_calls is not None:
            raise NotImplementedError(
                "DeepSeek Responses does not support explicit parallel_tool_calls",
            )

    def _replay_from_previous_response(
        self,
        previous_response: Optional[ChatResponse],
    ) -> tuple[dict[str, Any], ...]:
        """Return validated opaque reasoning items from one prior DeepSeek reply."""
        if previous_response is None:
            return ()
        provider_data = previous_response.provider_data
        if not isinstance(provider_data, Mapping):
            return ()
        replay = provider_data.get(_REASONING_REPLAY_KEY)
        if replay is None:
            return ()
        if not isinstance(replay, Mapping):
            raise LLMConfigError(
                detail="DeepSeek previous_response contains invalid reasoning replay data",
            )
        if previous_response.model != self.model:
            raise LLMConfigError(
                detail="DeepSeek previous_response model does not match this request",
            )
        if replay.get("model") != self.model:
            raise LLMConfigError(
                detail="DeepSeek reasoning replay model does not match this request",
            )
        if replay.get("response_id") != previous_response.response_id:
            raise LLMConfigError(
                detail="DeepSeek reasoning replay does not match previous_response",
            )
        return self._validate_reasoning_replay_items(replay.get("items"))

    @staticmethod
    def _validate_reasoning_replay_items(value: Any) -> tuple[dict[str, Any], ...]:
        """Permit only normalized Responses reasoning input items to be replayed."""
        if not isinstance(value, list) or not value:
            raise LLMConfigError(
                detail="DeepSeek reasoning replay must contain reasoning items",
            )
        replay_items: list[dict[str, Any]] = []
        for item in value:
            if not isinstance(item, Mapping) or item.get("type") != "reasoning":
                raise LLMConfigError(
                    detail="DeepSeek reasoning replay contains an invalid item",
                )
            raw_content = item.get("content")
            if not isinstance(raw_content, list) or not raw_content:
                raise LLMConfigError(
                    detail="DeepSeek reasoning replay item has invalid content",
                )
            content: list[dict[str, str]] = []
            for part in raw_content:
                if (
                    not isinstance(part, Mapping)
                    or part.get("type") != "reasoning_text"
                    or not isinstance(part.get("text"), str)
                    or not part["text"]
                ):
                    raise LLMConfigError(
                        detail="DeepSeek reasoning replay contains invalid reasoning text",
                    )
                content.append({"type": "reasoning_text", "text": part["text"]})
            replay_items.append({"type": "reasoning", "content": content})
        return tuple(replay_items)

    def _map_tools_for_deepseek(
        self,
        tools: Optional[List[ToolSpec]],
    ) -> Optional[list[dict[str, Any]]]:
        if tools:
            for tool in tools:
                if not isinstance(tool.description, str) or not tool.description.strip():
                    raise InvalidToolSchemaError(
                        detail=f"DeepSeek function {tool.name!r} requires a description",
                    )
        return self._map_tools_to_openai_responses(tools)

    @staticmethod
    def _to_deepseek_responses_input(
        messages: Messages,
        *,
        reasoning_replay: tuple[dict[str, Any], ...] = (),
        pdf_slots: tuple[PDFSlot, ...] = (),
    ) -> list[dict[str, Any]]:
        """Serialize an application-controlled Responses tool round-trip."""
        input_items: list[dict[str, Any]] = []
        last_assistant_start: Optional[int] = None
        slot_index_by_position = {
            (slot.message_index, slot.file_index): slot_index
            for slot_index, slot in enumerate(pdf_slots)
        }
        for message_index, message in enumerate(messages.items):
            if isinstance(message, Prompt):
                continue
            if isinstance(message, AIMessage):
                last_assistant_start = len(input_items)
                for tool_call in message.tool_calls or []:
                    if not tool_call.call_id:
                        raise ValueError(
                            "DeepSeek function_call history requires a non-empty "
                            "call_id",
                        )
                input_items.extend(message.to_openai_responses_input())
                continue
            if (
                isinstance(message, UserMessage)
                and message.files
                and any(isinstance(part, DocumentPart) for part in message.files)
            ):
                content: list[dict[str, Any]] = [{"type": "input_text", "text": message.content}]
                for file_index, part in enumerate(message.files):
                    if isinstance(part, DocumentPart):
                        slot_index = slot_index_by_position[(message_index, file_index)]
                        marker = f"__deepseek_pdf_slot_{slot_index}__"
                        content.append({"type": "input_file", "file_id": marker})
                    else:
                        content.append(message._part_to_openai_responses(part))
                input_items.append({"role": "user", "content": content})
            else:
                input_items.extend(message.to_openai_responses_input())
        if reasoning_replay:
            if last_assistant_start is None:
                raise LLMConfigError(
                    detail=(
                        "DeepSeek reasoning replay requires the matching prior "
                        "assistant message in the supplied history"
                    ),
                )
            input_items[last_assistant_start:last_assistant_start] = deepcopy(
                list(reasoning_replay),
            )
        return input_items

    @staticmethod
    def _parse_response(
        response: dict[str, Any],
        *,
        capture_reasoning: bool = False,
    ) -> ChatResponse:
        if response.get("object") != "response":
            raise LLMAPIClientError(
                detail="DeepSeek Responses API returned an invalid response object",
            )
        if not isinstance(response.get("output"), list):
            raise LLMAPIClientError(
                detail="DeepSeek Responses API response.output must be an array",
            )
        chat_response = ChatResponse.from_openai_responses_response(
            response,
            capture_reasoning=capture_reasoning,
        )
        chat_response.usage = DeepSeekAdapter._normalize_deepseek_usage(
            response.get("usage"),
        )
        return chat_response

    @staticmethod
    def _normalize_deepseek_usage(raw_usage: Any) -> Optional[DeepSeekUsage]:
        """Normalize confirmed usage counts while retaining omitted values."""
        usage = normalize_deepseek_usage(raw_usage)
        if usage is None:
            return None
        reasoning_tokens = None
        output_details = raw_usage.get("output_tokens_details")
        if output_details is not None:
            if not isinstance(output_details, Mapping):
                return None
            if "reasoning_tokens" in output_details:
                reasoning_tokens = output_details.get("reasoning_tokens")
                if (
                    reasoning_tokens is not None
                    and (
                        not isinstance(reasoning_tokens, int)
                        or isinstance(reasoning_tokens, bool)
                        or reasoning_tokens < 0
                        or (
                            usage.output_tokens is not None
                            and reasoning_tokens > usage.output_tokens
                        )
                    )
                ):
                    return None
        return DeepSeekUsage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            cached_tokens=usage.cached_tokens,
            reasoning_tokens=reasoning_tokens,
        )

    def _apply_response_pricing(self, chat_response: ChatResponse) -> None:
        """Pass verified dispatch-time rates to shared token accounting."""
        chat_response.currency = None
        chat_response.cost_input = None
        chat_response.cost_output = None
        chat_response.cost_total = None

        usage = chat_response.usage
        if usage is None or not isinstance(usage, Usage):
            return

        pricing = self.pricing
        if pricing is None or self.model_spec is None:
            return

        dispatch_time = _DEEPSEEK_DISPATCH_TIME.get() or _utc_now()
        multiplier = self._dispatch_pricing_multiplier(dispatch_time)
        if multiplier is None:
            return

        if usage.input_tokens is not None:
            tier = pricing.tier_for_prompt_tokens(usage.input_tokens)
        elif (
            len(pricing.tiers) == 1
            and pricing.tiers[0].up_to_prompt_tokens is None
        ):
            tier = pricing.tiers[0]
        else:
            return

        chat_response.apply_pricing(
            price_input_per_token=tier.in_per_token * multiplier,
            price_output_per_token=tier.out_per_token * multiplier,
            currency=pricing.currency,
            price_cache_read_per_token=(
                tier.cache_read_in_per_token * multiplier
                if tier.cache_read_in_per_token is not None
                else None
            ),
        )

    def _dispatch_pricing_multiplier(
        self,
        dispatch_time: datetime,
    ) -> Optional[float]:
        """Resolve the model's UTC time window and multiplier from its registry."""
        models = ORGANIZATION_DATA.get("models")
        model = models.get(self.model) if isinstance(models, Mapping) else None
        schedule = model.get("pricing_schedule") if isinstance(model, Mapping) else None
        if not isinstance(schedule, Mapping) or schedule.get("timezone") != "UTC":
            return None
        if (
            not isinstance(dispatch_time, datetime)
            or dispatch_time.tzinfo is None
            or dispatch_time.utcoffset() is None
        ):
            return None
        try:
            pricing_time = dispatch_time.astimezone(timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None

        weekdays = schedule.get("peak_weekdays")
        windows = schedule.get("peak_windows")
        if (
            not isinstance(weekdays, list)
            or any(
                isinstance(day, bool)
                or not isinstance(day, int)
                or day not in range(7)
                for day in weekdays
            )
            or not isinstance(windows, list)
        ):
            return None

        peak = pricing_time.weekday() in weekdays
        if peak:
            current_time = pricing_time.time()
            for window in windows:
                if not isinstance(window, list) or len(window) != 2:
                    return None
                try:
                    start, end = (time.fromisoformat(value) for value in window)
                except (TypeError, ValueError):
                    return None
                if start.tzinfo is not None or end.tzinfo is not None or start >= end:
                    return None
                if start <= current_time < end:
                    break
            else:
                peak = False

        multiplier = schedule.get(
            "peak_multiplier" if peak else "off_peak_multiplier",
        )
        if (
            isinstance(multiplier, bool)
            or not isinstance(multiplier, (int, float))
            or not isfinite(multiplier)
            or multiplier < 0
        ):
            return None
        return float(multiplier)

    def _store_reasoning_replay(
        self,
        chat_response: ChatResponse,
        response: Mapping[str, Any],
        *,
        upload_ids: tuple[str, ...] = (),
        pdf_contents: tuple[bytes, ...] = (),
    ) -> None:
        """Keep continuation material opaque and outside visible response fields."""
        items = self._reasoning_replay_items(response.get("output"))
        if upload_ids or pdf_contents:
            items = self._scrub_reasoning_replay_items(
                items,
                upload_ids=upload_ids,
                pdf_contents=pdf_contents,
            )
        if not items:
            return
        chat_response.provider_data = {
            _REASONING_REPLAY_KEY: {
                "model": chat_response.model or self.model,
                "response_id": chat_response.response_id,
                "items": items,
            },
        }

    @staticmethod
    def _scrub_reasoning_replay_items(
        items: list[dict[str, Any]],
        *,
        upload_ids: tuple[str, ...],
        pdf_contents: tuple[bytes, ...],
    ) -> list[dict[str, Any]]:
        """Remove request-local PDF echoes while retaining unrelated replay text."""
        scrubbed_items: list[dict[str, Any]] = []
        for item in items:
            cleaned = {**item}
            content = []
            for part in item.get("content", []):
                text = part.get("text", "")
                for upload_id in upload_ids:
                    text = text.replace(upload_id, "[redacted]")
                if DeepSeekAdapter._contains_pdf_echo(text, pdf_contents):
                    text = "[DeepSeek PDF content redacted]"
                content.append({**part, "text": text})
            cleaned["content"] = content
            scrubbed_items.append(cleaned)
        return scrubbed_items

    @staticmethod
    def _contains_pdf_echo(text: str, pdf_contents: tuple[bytes, ...]) -> bool:
        """Detect literal or base64 PDF echoes from short byte signatures."""
        if not text or not pdf_contents:
            return False
        for pdf in pdf_contents:
            for sample in (pdf[:48], pdf[-48:]):
                if len(sample) < 8:
                    continue
                try:
                    literal = sample.decode("utf-8")
                except UnicodeDecodeError:
                    literal = ""
                if literal and literal in text:
                    return True
                encoded = base64.b64encode(sample).decode("ascii")
                if encoded in text:
                    return True
        return False

    @staticmethod
    def _reasoning_replay_items(raw_output: Any) -> list[dict[str, Any]]:
        """Extract only replayable reasoning text from a Responses output array."""
        if not isinstance(raw_output, list):
            return []
        replay_items: list[dict[str, Any]] = []
        for item in raw_output:
            if not isinstance(item, Mapping) or item.get("type") != "reasoning":
                continue
            content: list[dict[str, str]] = []
            raw_content = item.get("content")
            if isinstance(raw_content, list):
                for part in raw_content:
                    if (
                        isinstance(part, Mapping)
                        and part.get("type") == "reasoning_text"
                        and isinstance(part.get("text"), str)
                        and part["text"]
                    ):
                        content.append(
                            {"type": "reasoning_text", "text": part["text"]},
                        )
            legacy_reasoning_content = item.get("reasoning_content")
            if isinstance(legacy_reasoning_content, str) and legacy_reasoning_content:
                content.append(
                    {"type": "reasoning_text", "text": legacy_reasoning_content},
                )
            if content:
                replay_items.append({"type": "reasoning", "content": content})
        return replay_items


__all__ = ["DeepSeekAdapter", "DeepSeekUsage"]
