"""Synchronous adapter for Mistral's official Chat Completions API."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, AsyncIterator, Iterator, List, Mapping, Optional
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
    _StreamState,
)
from llm_api_adapter.errors.llm_api_error import (
    InvalidToolArgumentsError,
    LLMAPIAuthorizationError,
    LLMAPIClientError,
    LLMAPIError,
    LLMAPIRateLimitError,
    LLMAPIServerError,
    LLMAPITimeoutError,
    LLMAPITokenLimitError,
    LLMAPIUsageLimitError,
)
from llm_api_adapter.llms.transports import (
    JSONResponse,
    SSEEvent,
    SyncTransport,
    TransportRequest,
    create_sync_transport,
)
from llm_api_adapter.llms.async_streaming import (
    async_request,
    async_stream_request,
)
from llm_api_adapter.llms.streaming import (
    StreamChunkBuffer,
    StreamReasoningCollector,
    StreamUsageTracker,
)
from llm_api_adapter.models.messages.chat_message import (
    Message,
    Messages,
    UserMessage,
)
from llm_api_adapter.models.messages.file_parts import DocumentPart
from llm_api_adapter.models.responses.chat_response import ChatResponse, Usage
from llm_api_adapter.models.responses.reasoning_event import ReasoningEvent
from llm_api_adapter.models.tools import ToolCall, ToolSpec


_MISTRAL_CHAT_COMPLETIONS_URL = "https://api.mistral.ai/v1/chat/completions"
_MISTRAL_OCR_URL = "https://api.mistral.ai/v1/ocr"
_MISTRAL_OCR_MODEL = "mistral-ocr-latest"


@dataclass
class _MistralStreamState(_StreamState):
    """Mistral-specific state retained while consuming one SSE response."""

    response_metadata: dict[str, Any] = field(default_factory=dict)
    text_parts: list[str] = field(default_factory=list)
    tool_calls: dict[int, dict[str, Any]] = field(default_factory=dict)
    finish_reason: Optional[str] = None


@dataclass(repr=False)
class MistralAdapter(LLMAdapterBase):
    """Call Mistral Chat Completions and expand PDF inputs through OCR."""

    company: str = "mistral"
    endpoint: str = _MISTRAL_CHAT_COMPLETIONS_URL
    _sync_transport: SyncTransport = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._sync_transport = create_sync_transport(self.transport)

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
        """Generate one response through ``POST /v1/chat/completions``."""
        prepared_messages = self._prepare_document_messages(messages, timeout_s)
        request_context, payload = self._prepare_request_payload(
            prepared_messages,
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
        )

        try:
            response = self._post_payload(payload, timeout_s)
            chat_response = self._parse_response(
                response,
                capture_reasoning=capture_reasoning,
            )
            return self._finalize_chat_response(
                chat_response,
                effective_schema=request_context.effective_schema,
                response_model=response_model,
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
        """Generate one response without blocking the event loop."""
        prepared_messages = await self._prepare_document_messages_async(
            messages,
            timeout_s,
        )
        request_context, payload = self._prepare_request_payload(
            prepared_messages,
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
        )

        try:
            response = await self._apost_payload(payload, timeout_s)
            chat_response = self._parse_response(
                response,
                capture_reasoning=capture_reasoning,
            )
            return self._finalize_chat_response(
                chat_response,
                effective_schema=request_context.effective_schema,
                response_model=response_model,
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
        """Stream normalized visible deltas from Mistral's SSE endpoint."""
        prepared_messages = self._prepare_document_messages(messages, timeout_s)
        request_context, payload = self._prepare_request_payload(
            prepared_messages,
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
        )
        state = self._new_stream_state(
            buffer_chars=buffer_chars,
            capture_reasoning=capture_reasoning,
        )
        events = self._stream_payload(payload, timeout_s)
        yield from self._run_sync_stream(
            events,
            state,
            consume_event=self._consume_stream_event,
            finalize_response=self._finalize_stream,
            effective_schema=request_context.effective_schema,
            response_model=response_model,
            on_delta=on_delta,
            on_tool_call=on_tool_call,
            on_done=on_done,
            on_chunk=on_chunk,
            capture_reasoning=capture_reasoning,
            on_reasoning=on_reasoning,
        )

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
        """Return the asynchronous streaming iterator for Mistral."""
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
        *,
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
        capture_reasoning: bool,
        on_reasoning: Optional[AsyncOnReasoning],
    ) -> AsyncIterator[str]:
        prepared_messages = await self._prepare_document_messages_async(
            messages,
            timeout_s,
        )
        request_context, payload = self._prepare_request_payload(
            prepared_messages,
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
        )
        state = self._new_stream_state(
            buffer_chars=buffer_chars,
            capture_reasoning=capture_reasoning,
        )
        events = self._astream_payload(payload, timeout_s)
        async for text in self._run_async_stream(
            events,
            state,
            consume_event=self._consume_stream_event_async,
            finalize_response=self._finalize_stream,
            effective_schema=request_context.effective_schema,
            response_model=response_model,
            on_delta=on_delta,
            on_tool_call=on_tool_call,
            on_done=on_done,
            on_chunk=on_chunk,
            capture_reasoning=capture_reasoning,
            on_reasoning=on_reasoning,
        ):
            yield text

    def _prepare_request_payload(
        self,
        messages: List[Message] | Messages,
        *,
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
    ) -> tuple[Any, dict[str, Any]]:
        # Mistral Chat Completions is stateless.  Conversation history is
        # represented by ``messages``; match the Anthropic and Google adapters
        # by accepting the common argument without serializing it.
        _ = previous_response
        temperature = self._validate_parameter("temperature", temperature, 0, 1.5)
        top_p = self._validate_parameter("top_p", top_p, 0, 1)
        request_context = self._prepare_chat_request(
            messages,
            tools,
            tool_choice,
            json_schema,
            response_model,
        )
        return request_context, self._build_payload(
            messages=request_context.normalized_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            reasoning_level=reasoning_level,
            tools=tools,
            tool_choice=request_context.normalized_tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            json_schema=request_context.effective_schema,
        )

    def _prepare_document_messages(
        self,
        messages: List[Message] | Messages,
        timeout_s: Optional[float],
    ) -> Messages:
        """Replace PDF inputs with their Mistral OCR Markdown output."""
        normalized_messages = self._normalize_messages(messages)
        documents = self._document_parts(normalized_messages)
        if not documents:
            return normalized_messages
        markdowns = [
            self._extract_ocr_markdown(
                self._post_ocr_payload(
                    self._build_ocr_payload(document),
                    timeout_s,
                )
            )
            for document in documents
        ]
        return self._replace_documents(normalized_messages, markdowns)

    async def _prepare_document_messages_async(
        self,
        messages: List[Message] | Messages,
        timeout_s: Optional[float],
    ) -> Messages:
        """Asynchronously replace PDF inputs with OCR Markdown output."""
        normalized_messages = self._normalize_messages(messages)
        documents = self._document_parts(normalized_messages)
        if not documents:
            return normalized_messages
        markdowns = [
            self._extract_ocr_markdown(
                await self._apost_ocr_payload(
                    self._build_ocr_payload(document),
                    timeout_s,
                )
            )
            for document in documents
        ]
        return self._replace_documents(normalized_messages, markdowns)

    @staticmethod
    def _document_parts(messages: Messages) -> list[DocumentPart]:
        return [
            file
            for message in messages.items
            if isinstance(message, UserMessage) and message.files
            for file in message.files
            if isinstance(file, DocumentPart)
        ]

    @staticmethod
    def _replace_documents(
        messages: Messages,
        markdowns: list[str],
    ) -> Messages:
        markdown_iterator = iter(markdowns)
        processed_messages: list[Message] = []
        for message in messages.items:
            if not isinstance(message, UserMessage) or not message.files:
                processed_messages.append(message)
                continue
            document_markdowns = [
                next(markdown_iterator)
                for file in message.files
                if isinstance(file, DocumentPart)
            ]
            if not document_markdowns:
                processed_messages.append(message)
                continue
            files = [
                file
                for file in message.files
                if not isinstance(file, DocumentPart)
            ]
            processed_messages.append(
                UserMessage(
                    content=MistralAdapter._append_document_markdown(
                        message.content,
                        document_markdowns,
                    ),
                    files=files or None,
                )
            )
        return Messages(processed_messages)

    @staticmethod
    def _append_document_markdown(
        content: str,
        markdowns: list[str],
    ) -> str:
        documents = [
            f"<document index=\"{index}\">\n{markdown}\n</document>"
            for index, markdown in enumerate(markdowns, start=1)
        ]
        return "\n\n".join([content, *documents])

    @staticmethod
    def _build_ocr_payload(document: DocumentPart) -> dict[str, Any]:
        document_url = (
            document.url if document._is_url() else document._to_data_uri()
        )
        return {
            "model": _MISTRAL_OCR_MODEL,
            "document": {
                "type": "document_url",
                "document_url": document_url,
            },
        }

    @staticmethod
    def _extract_ocr_markdown(response: Mapping[str, Any]) -> str:
        pages = response.get("pages")
        if not isinstance(pages, list):
            raise LLMAPIClientError(
                detail="Mistral OCR response does not contain document pages"
            )
        markdowns = [
            page["markdown"]
            for page in pages
            if isinstance(page, Mapping)
            and isinstance(page.get("markdown"), str)
            and page["markdown"].strip()
        ]
        if not markdowns:
            raise LLMAPIClientError(
                detail="Mistral OCR response contains no document text"
            )
        return "\n\n---\n\n".join(markdowns)

    def _new_stream_state(
        self,
        *,
        buffer_chars: Optional[int],
        capture_reasoning: bool,
    ) -> _MistralStreamState:
        return _MistralStreamState(
            chunk_buffer=StreamChunkBuffer(buffer_chars),
            usage_tracker=StreamUsageTracker(),
            reasoning_collector=(
                StreamReasoningCollector() if capture_reasoning else None
            ),
            reasoning_response=ChatResponse() if capture_reasoning else None,
        )

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def _apost_payload(
        self,
        payload: dict[str, Any],
        timeout_s: Optional[float],
    ) -> dict[str, Any]:
        response_data = await async_request(
            self.endpoint,
            headers=self._headers(),
            payload=payload,
            timeout=timeout_s,
            http_error_handler=self._handle_http_error,
        )
        if not isinstance(response_data, dict):
            raise LLMAPIClientError(detail="Mistral returned a non-object response")
        return response_data

    async def _apost_ocr_payload(
        self,
        payload: dict[str, Any],
        timeout_s: Optional[float],
    ) -> dict[str, Any]:
        response_data = await async_request(
            _MISTRAL_OCR_URL,
            headers=self._headers(),
            payload=payload,
            timeout=timeout_s,
            http_error_handler=self._handle_http_error,
        )
        if not isinstance(response_data, dict):
            raise LLMAPIClientError(
                detail="Mistral OCR returned a non-object response"
            )
        return response_data

    def _stream_payload(
        self,
        payload: dict[str, Any],
        timeout_s: Optional[float],
    ) -> Iterator[SSEEvent]:
        return self._sync_transport.post_sse(
            TransportRequest(
                url=self.endpoint,
                headers=self._headers(),
                payload=self._with_stream_options(payload),
                timeout=timeout_s,
            ),
            http_error_handler=self._handle_http_error,
            stream_error_handler=self._handle_stream_error,
        )

    def _astream_payload(
        self,
        payload: dict[str, Any],
        timeout_s: Optional[float],
    ) -> AsyncIterator[SSEEvent]:
        return async_stream_request(
            self.endpoint,
            headers=self._headers(),
            payload=self._with_stream_options(payload),
            timeout=timeout_s,
            http_error_handler=self._handle_http_error,
            stream_error_handler=self._handle_stream_error,
        )

    @staticmethod
    def _with_stream_options(payload: Mapping[str, Any]) -> dict[str, Any]:
        stream_payload = dict(payload)
        stream_payload["stream"] = True
        stream_payload["stream_options"] = {"include_usage": True}
        return stream_payload

    def _consume_stream_event(
        self,
        event: SSEEvent,
        state: _MistralStreamState,
        *,
        on_chunk: Optional[OnChunk],
        on_delta: Optional[OnDelta],
        on_reasoning: Optional[OnReasoning],
    ) -> Iterator[str]:
        payload = event.data if isinstance(event.data, Mapping) else {}
        visible_text, reasoning_text = self._consume_stream_payload(payload, state)
        self._record_stream_reasoning(
            state,
            reasoning_text,
            on_reasoning=on_reasoning,
        )
        for text in visible_text:
            yield from self._emit_stream_chunks(
                state.chunk_buffer.add(text),
                on_chunk,
                on_delta,
            )

    async def _consume_stream_event_async(
        self,
        event: SSEEvent,
        state: _MistralStreamState,
        *,
        on_chunk: Optional[AsyncOnChunk],
        on_delta: Optional[AsyncOnDelta],
        on_reasoning: Optional[AsyncOnReasoning],
    ) -> AsyncIterator[str]:
        payload = event.data if isinstance(event.data, Mapping) else {}
        visible_text, reasoning_text = self._consume_stream_payload(payload, state)
        await self._record_stream_reasoning_async(
            state,
            reasoning_text,
            on_reasoning=on_reasoning,
        )
        for text in visible_text:
            async for emitted_text in self._emit_async_stream_chunks(
                state.chunk_buffer.add(text),
                on_chunk,
                on_delta,
            ):
                yield emitted_text

    def _consume_stream_payload(
        self,
        payload: Mapping[str, Any],
        state: _MistralStreamState,
    ) -> tuple[list[str], list[str]]:
        self._raise_stream_payload_error(payload)
        for field in ("id", "model", "created", "usage"):
            if field in payload:
                state.response_metadata[field] = payload[field]
        state.usage_tracker.record(
            state.chunk_buffer,
            self._parse_usage(payload.get("usage")),
        )

        visible_text: list[str] = []
        reasoning_text: list[str] = []
        choices = payload.get("choices")
        if not isinstance(choices, list):
            return visible_text, reasoning_text

        for choice in choices:
            if not isinstance(choice, Mapping) or choice.get("index", 0) != 0:
                continue
            delta = choice.get("delta")
            if isinstance(delta, Mapping):
                visible, reasoning = self._parse_stream_content(delta.get("content"))
                visible_text.extend(visible)
                reasoning_text.extend(reasoning)
                raw_tool_calls = delta.get("tool_calls")
                if isinstance(raw_tool_calls, list):
                    self._accumulate_stream_tool_calls(raw_tool_calls, state.tool_calls)
            finish_reason = choice.get("finish_reason")
            if isinstance(finish_reason, str):
                state.finish_reason = finish_reason

        state.text_parts.extend(visible_text)
        return visible_text, reasoning_text

    @staticmethod
    def _parse_stream_content(value: Any) -> tuple[list[str], list[str]]:
        if isinstance(value, str):
            return ([value] if value else []), []
        if isinstance(value, Mapping):
            chunks = [value]
        elif isinstance(value, list):
            chunks = value
        else:
            return [], []

        visible_text: list[str] = []
        reasoning_text: list[str] = []
        for chunk in chunks:
            if not isinstance(chunk, Mapping):
                continue
            if chunk.get("type") == "text":
                text = chunk.get("text")
                if isinstance(text, str) and text:
                    visible_text.append(text)
                continue
            if chunk.get("type") != "thinking":
                continue
            thinking = chunk.get("thinking")
            if isinstance(thinking, str):
                if thinking:
                    reasoning_text.append(thinking)
                continue
            if not isinstance(thinking, list):
                continue
            for thinking_chunk in thinking:
                if not isinstance(thinking_chunk, Mapping):
                    continue
                text = thinking_chunk.get("text")
                if isinstance(text, str) and text:
                    reasoning_text.append(text)
        return visible_text, reasoning_text

    def _record_stream_reasoning(
        self,
        state: _MistralStreamState,
        reasoning_text: list[str],
        *,
        on_reasoning: Optional[OnReasoning],
    ) -> None:
        if state.reasoning_collector is None or state.reasoning_response is None:
            return
        for text in reasoning_text:
            self._record_reasoning_event(
                state.reasoning_response,
                state.reasoning_collector,
                text,
                capture_reasoning=True,
                kind="content",
                on_reasoning=on_reasoning,
            )

    async def _record_stream_reasoning_async(
        self,
        state: _MistralStreamState,
        reasoning_text: list[str],
        *,
        on_reasoning: Optional[AsyncOnReasoning],
    ) -> None:
        if state.reasoning_collector is None or state.reasoning_response is None:
            return
        for text in reasoning_text:
            await self._record_async_reasoning_event(
                state.reasoning_response,
                state.reasoning_collector,
                text,
                capture_reasoning=True,
                kind="content",
                on_reasoning=on_reasoning,
            )

    @staticmethod
    def _accumulate_stream_tool_calls(
        raw_tool_calls: list[Any],
        tool_calls: dict[int, dict[str, Any]],
    ) -> None:
        for raw_tool_call in raw_tool_calls:
            if not isinstance(raw_tool_call, Mapping):
                continue
            index = raw_tool_call.get("index")
            if not isinstance(index, int):
                index = len(tool_calls)
            tool_call = tool_calls.setdefault(index, {"function": {"arguments": ""}})
            for field in ("id", "type"):
                if raw_tool_call.get(field) is not None:
                    tool_call[field] = raw_tool_call[field]
            function = raw_tool_call.get("function")
            if not isinstance(function, Mapping):
                continue
            target = tool_call["function"]
            if function.get("name") is not None:
                target["name"] = function["name"]
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                target["arguments"] = f"{target.get('arguments', '')}{arguments}"
            elif isinstance(arguments, Mapping):
                target["arguments"] = dict(arguments)

    def _finalize_stream(
        self,
        state: _MistralStreamState,
        *,
        capture_reasoning: bool,
        effective_schema: Optional[dict],
        response_model: Optional[Any],
    ) -> ChatResponse:
        response_data = dict(state.response_metadata)
        response_data["model"] = response_data.get("model") or self.model
        message: dict[str, Any] = {"content": "".join(state.text_parts) or None}
        if state.tool_calls:
            message["tool_calls"] = [
                state.tool_calls[index] for index in sorted(state.tool_calls)
            ]
        response_data["choices"] = [
            {"message": message, "finish_reason": state.finish_reason}
        ]
        chat_response = self._parse_response(
            response_data,
            capture_reasoning=False,
        )
        return self._finalize_stream_response(
            chat_response,
            reasoning_collector=state.reasoning_collector,
            effective_schema=effective_schema,
            response_model=response_model,
        )

    def _build_payload(
        self,
        *,
        messages: Messages,
        max_tokens: Optional[int],
        temperature: float,
        top_p: float,
        reasoning_level: Optional[str | int],
        tools: Optional[List[ToolSpec]],
        tool_choice: Optional[str],
        parallel_tool_calls: Optional[bool],
        json_schema: Optional[dict],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._to_mistral_messages(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "reasoning_effort": self._resolve_reasoning_effort(reasoning_level),
            "tools": self._map_tools(tools),
            "tool_choice": self._map_tool_choice(tool_choice),
            "parallel_tool_calls": parallel_tool_calls,
        }
        if json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "strict": True,
                    "schema": self._enforce_strict_schema(json_schema),
                },
            }
        return {key: value for key, value in payload.items() if value is not None}

    @staticmethod
    def _to_mistral_messages(messages: Messages) -> list[dict[str, Any]]:
        """Convert the shared OpenAI-like message shape to Mistral's variant."""
        serialized_messages = messages.to_openai()
        for message in serialized_messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "image_url":
                    continue
                image_url = part.get("image_url")
                if isinstance(image_url, Mapping):
                    url = image_url.get("url")
                    if not isinstance(url, str):
                        raise ValueError(
                            "Mistral image_url must contain a string URL"
                        )
                    part["image_url"] = url
        return serialized_messages

    def _resolve_reasoning_effort(
        self,
        reasoning_level: Optional[str | int],
    ) -> Optional[str]:
        if reasoning_level is None:
            return "none" if self.is_reasoning else None
        provider_value = self._resolve_reasoning_level(reasoning_level).provider_value
        if provider_value is None:
            return None
        if not isinstance(provider_value, str):
            raise TypeError("Mistral reasoning resolution must produce a string")
        return provider_value

    @staticmethod
    def _map_tools(tools: Optional[List[ToolSpec]]) -> Optional[list[dict[str, Any]]]:
        if not tools:
            return None
        result: list[dict[str, Any]] = []
        for tool in tools:
            function: dict[str, Any] = {
                "name": tool.name,
                "parameters": tool.json_schema,
            }
            if tool.description:
                function["description"] = tool.description
            result.append({"type": "function", "function": function})
        return result

    @staticmethod
    def _map_tool_choice(tool_choice: Optional[str]) -> Any:
        if tool_choice is None or tool_choice in {"auto", "none"}:
            return tool_choice
        if tool_choice == "any":
            return "any"
        return {"type": "function", "function": {"name": tool_choice}}

    def _post_payload(
        self,
        payload: dict[str, Any],
        timeout_s: Optional[float],
    ) -> dict[str, Any]:
        response: JSONResponse = self._sync_transport.post_json(
            TransportRequest(
                url=self.endpoint,
                headers=self._headers(),
                payload=payload,
                timeout=timeout_s,
            ),
            http_error_handler=self._handle_http_error,
        )
        response_data = response.json()
        if not isinstance(response_data, dict):
            raise LLMAPIClientError(detail="Mistral returned a non-object response")
        return response_data

    def _post_ocr_payload(
        self,
        payload: dict[str, Any],
        timeout_s: Optional[float],
    ) -> dict[str, Any]:
        response: JSONResponse = self._sync_transport.post_json(
            TransportRequest(
                url=_MISTRAL_OCR_URL,
                headers=self._headers(),
                payload=payload,
                timeout=timeout_s,
            ),
            http_error_handler=self._handle_http_error,
        )
        response_data = response.json()
        if not isinstance(response_data, dict):
            raise LLMAPIClientError(
                detail="Mistral OCR returned a non-object response"
            )
        return response_data

    def _handle_stream_error(self, event: SSEEvent) -> None:
        payload = event.data if isinstance(event.data, Mapping) else {}
        self._raise_stream_payload_error(payload)
        raise LLMAPIClientError(detail="Mistral returned an invalid SSE error event")

    def _raise_stream_payload_error(self, payload: Mapping[str, Any]) -> None:
        is_error = payload.get("type") == "error" or payload.get("object") == "error"
        if not is_error:
            return
        error_data = payload.get("error", payload)
        if not isinstance(error_data, Mapping):
            error_data = {}
        error_type = error_data.get("type") or error_data.get("code")
        detail = error_data.get("message") or "Mistral returned an SSE error"
        self._raise_mapped_error(
            status_code=None,
            error_type=str(error_type) if error_type else None,
            detail=str(detail),
        )

    @classmethod
    def _parse_response(
        cls,
        response: Mapping[str, Any],
        *,
        capture_reasoning: bool,
    ) -> ChatResponse:
        choice = ((response.get("choices") or [None])[0]) or {}
        if not isinstance(choice, Mapping):
            raise LLMAPIClientError(detail="Mistral response choice is malformed")
        message = choice.get("message") or {}
        if not isinstance(message, Mapping):
            raise LLMAPIClientError(detail="Mistral response message is malformed")

        content, reasoning_events = cls._parse_content(
            message.get("content"),
            capture_reasoning=capture_reasoning,
        )
        tool_calls = cls._parse_tool_calls(message.get("tool_calls"))
        if content is None and not tool_calls:
            warnings.warn("Mistral returned empty content and no tool calls.", UserWarning)

        return ChatResponse(
            model=_as_optional_str(response.get("model")),
            response_id=_as_optional_str(response.get("id")),
            timestamp=_as_optional_int(response.get("created")),
            usage=cls._parse_usage(response.get("usage")),
            content=content,
            tool_calls=tool_calls,
            finish_reason=_as_optional_str(choice.get("finish_reason")),
            reasoning_events=reasoning_events,
        )

    @staticmethod
    def _parse_usage(value: Any) -> Optional[Usage]:
        if not isinstance(value, Mapping):
            return None
        input_tokens = _as_non_negative_int(value.get("prompt_tokens"))
        output_tokens = _as_non_negative_int(value.get("completion_tokens"))
        total_tokens = _as_non_negative_int(value.get("total_tokens"))
        return Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=(total_tokens or input_tokens + output_tokens),
        )

    @classmethod
    def _parse_content(
        cls,
        value: Any,
        *,
        capture_reasoning: bool,
    ) -> tuple[Optional[str], list[ReasoningEvent]]:
        if value is None:
            return None, []
        if isinstance(value, str):
            return value, []
        if not isinstance(value, list):
            raise LLMAPIClientError(detail="Mistral response content is malformed")

        text_parts: list[str] = []
        reasoning_events: list[ReasoningEvent] = []
        for chunk in value:
            if not isinstance(chunk, Mapping):
                continue
            if chunk.get("type") == "text":
                text = chunk.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
                continue
            if chunk.get("type") == "thinking" and capture_reasoning:
                for thinking_chunk in chunk.get("thinking") or []:
                    if not isinstance(thinking_chunk, Mapping):
                        continue
                    text = thinking_chunk.get("text")
                    if isinstance(text, str) and text:
                        reasoning_events.append(
                            ReasoningEvent(
                                text=text,
                                kind="content",
                                index=len(reasoning_events),
                                elapsed_s=0.0,
                                delta_s=0.0,
                            )
                        )
        return ("".join(text_parts) if text_parts else None), reasoning_events

    @staticmethod
    def _parse_tool_calls(value: Any) -> Optional[list[ToolCall]]:
        if value is None:
            return None
        if not isinstance(value, list):
            raise LLMAPIClientError(detail="Mistral response tool_calls is malformed")

        tool_calls: list[ToolCall] = []
        for raw_tool_call in value:
            if not isinstance(raw_tool_call, Mapping):
                raise LLMAPIClientError(detail="Mistral tool call is malformed")
            function = raw_tool_call.get("function") or {}
            if not isinstance(function, Mapping):
                raise LLMAPIClientError(detail="Mistral tool function is malformed")
            name = function.get("name")
            if not isinstance(name, str) or not name:
                raise InvalidToolArgumentsError(
                    detail="Mistral tool function.name must be a non-empty string"
                )
            raw_arguments = function.get("arguments", "{}")
            try:
                if isinstance(raw_arguments, str):
                    arguments = json.loads(raw_arguments) if raw_arguments.strip() else {}
                elif isinstance(raw_arguments, Mapping):
                    arguments = dict(raw_arguments)
                else:
                    arguments = {}
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise InvalidToolArgumentsError(
                    detail=(
                        "Mistral tool arguments JSON parse failed for "
                        f"tool={name!r}: {error}"
                    )
                ) from error
            if not isinstance(arguments, dict):
                raise InvalidToolArgumentsError(
                    detail=f"Mistral tool arguments must decode to an object for tool={name!r}"
                )
            tool_calls.append(
                ToolCall(
                    name=name,
                    arguments=arguments,
                    call_id=_as_optional_str(raw_tool_call.get("id")),
                )
            )
        return tool_calls or None

    def _handle_http_error(self, error: Any) -> None:
        response = getattr(error, "response", None)
        status_code = getattr(response, "status_code", None)
        payload: Mapping[str, Any] = {}
        if response is not None:
            try:
                candidate = response.json()
                if isinstance(candidate, Mapping):
                    payload = candidate
            except Exception:
                pass
        error_data = payload.get("error", payload)
        if not isinstance(error_data, Mapping):
            error_data = {}
        error_type = error_data.get("type") or error_data.get("code")
        detail = error_data.get("message") or str(error)
        self._raise_mapped_error(
            status_code=status_code,
            error_type=str(error_type) if error_type else None,
            detail=str(detail),
        )

    @staticmethod
    def _raise_mapped_error(
        *,
        status_code: Optional[int],
        error_type: Optional[str],
        detail: str,
    ) -> None:
        normalized_type = (error_type or "").lower()
        if status_code in {401, 403} or normalized_type in {
            "authentication_error",
            "authorization_error",
            "invalid_api_key",
        }:
            raise LLMAPIAuthorizationError(detail=detail)
        if status_code == 429 or normalized_type in {
            "rate_limit_error",
            "rate_limit_exceeded",
        }:
            raise LLMAPIRateLimitError(detail=detail)
        if normalized_type in {
            "context_length_exceeded",
            "input_too_long",
            "max_tokens_exceeded",
        }:
            raise LLMAPITokenLimitError(detail=detail)
        if normalized_type in {"insufficient_quota", "usage_limit_exceeded"}:
            raise LLMAPIUsageLimitError(detail=detail)
        if status_code in {408, 504} or normalized_type in {"timeout", "timeout_error"}:
            raise LLMAPITimeoutError(detail=detail)
        if status_code is not None and 500 <= status_code < 600:
            raise LLMAPIServerError(detail=detail)
        raise LLMAPIClientError(detail=detail)


def _as_optional_str(value: Any) -> Optional[str]:
    return value if isinstance(value, str) else None


def _as_optional_int(value: Any) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _as_non_negative_int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


__all__ = ["MistralAdapter"]
