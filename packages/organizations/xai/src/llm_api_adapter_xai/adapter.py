"""Adapter for xAI's official Responses API."""

from __future__ import annotations

import base64
import binascii
import json
import re
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
from llm_api_adapter.errors.llm_api_error import (
    InvalidToolSchemaError,
    JSONSchemaError,
    LLMAPIClientError,
    LLMAPIError,
)
from llm_api_adapter.llms.transports import SSEEvent
from llm_api_adapter.models.messages.chat_message import (
    AIMessage,
    Message,
    Messages,
    Prompt,
    UserMessage,
)
from llm_api_adapter.models.messages.file_parts import DocumentPart, ImagePart
from llm_api_adapter.models.responses.chat_response import ChatResponse
from llm_api_adapter.models.tools import ToolSpec

from .clients import XAIResponsesAsyncClient, XAIResponsesSyncClient
from .streaming import XAIResponsesStreamParser, XAIResponsesStreamState


_XAI_ADAPTER_FILE_TTL_SECONDS = 86_400


@dataclass(frozen=True)
class _PreparedResponsesRequest:
    """Provider parameters plus common finalization context for one request."""

    parameters: dict[str, Any]
    normalized_messages: Messages
    document_uploads: tuple[DocumentPart, ...]
    effective_schema: Optional[dict]
    response_model: Optional[Any]
    capture_reasoning: bool


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
        try:
            parameters = self._materialize_responses_parameters(
                prepared,
                self._upload_document_parts(
                    prepared.document_uploads,
                    timeout_s,
                ),
            )
            response = self._client.create(
                model=self.model,
                timeout=timeout_s,
                **parameters,
            )
            return self._finalize_xai_chat_response(
                response,
                effective_schema=prepared.effective_schema,
                response_model=prepared.response_model,
                capture_reasoning=prepared.capture_reasoning,
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
        try:
            parameters = self._materialize_responses_parameters(
                prepared,
                await self._aupload_document_parts(
                    prepared.document_uploads,
                    timeout_s,
                ),
            )
            response = await self._async_client.create(
                model=self.model,
                timeout=timeout_s,
                **parameters,
            )
            return self._finalize_xai_chat_response(
                response,
                effective_schema=prepared.effective_schema,
                response_model=prepared.response_model,
                capture_reasoning=prepared.capture_reasoning,
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
        state = XAIResponsesStreamParser.new_state(
            buffer_chars=buffer_chars,
            capture_reasoning=prepared.capture_reasoning,
        )
        parameters = self._materialize_responses_parameters(
            prepared,
            self._upload_document_parts(prepared.document_uploads, timeout_s),
        )
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
            effective_schema=prepared.effective_schema,
            response_model=prepared.response_model,
            on_delta=on_delta,
            on_tool_call=on_tool_call,
            on_done=on_done,
            on_chunk=on_chunk,
            capture_reasoning=prepared.capture_reasoning,
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
        state = XAIResponsesStreamParser.new_state(
            buffer_chars=buffer_chars,
            capture_reasoning=prepared.capture_reasoning,
        )
        parameters = self._materialize_responses_parameters(
            prepared,
            await self._aupload_document_parts(
                prepared.document_uploads,
                timeout_s,
            ),
        )
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
            effective_schema=prepared.effective_schema,
            response_model=prepared.response_model,
            on_delta=on_delta,
            on_tool_call=on_tool_call,
            on_done=on_done,
            on_chunk=on_chunk,
            capture_reasoning=prepared.capture_reasoning,
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
    ) -> _PreparedResponsesRequest:
        # ``previous_response`` is an optional provider-specific optimization.
        # xAI server-side continuation is intentionally not exposed by this
        # package, so retain the shared API shape and use the supplied messages.
        del previous_response
        temperature, top_p = self._validate_sampling_parameters(temperature, top_p)
        request_context = self._prepare_chat_request(
            messages,
            tools,
            tool_choice,
            json_schema,
            response_model,
        )
        normalized_messages = request_context.normalized_messages

        parameters: dict[str, Any] = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "tools": self._map_tools(tools),
            "tool_choice": self._map_tool_choice(
                request_context.normalized_tool_choice,
            ),
            "parallel_tool_calls": parallel_tool_calls,
        }
        if request_context.effective_schema is not None:
            schema = self._validate_xai_structured_schema(
                request_context.effective_schema,
            )
            parameters["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "response",
                    "schema": schema,
                    "strict": True,
                }
            }
        if reasoning_level is not None:
            provider_value = self._resolve_reasoning_level(
                reasoning_level,
            ).provider_value
            if isinstance(provider_value, str):
                parameters["reasoning"] = {"effort": provider_value}
        instructions = normalized_messages.to_openai_responses_instructions()
        if instructions is not None:
            parameters["instructions"] = instructions
        return _PreparedResponsesRequest(
            parameters={
                key: value for key, value in parameters.items() if value is not None
            },
            normalized_messages=normalized_messages,
            document_uploads=self._document_uploads(normalized_messages),
            effective_schema=request_context.effective_schema,
            response_model=response_model,
            capture_reasoning=capture_reasoning,
        )

    def _consume_stream_event(
        self,
        event: SSEEvent,
        state: XAIResponsesStreamState,
        *,
        on_chunk: Optional[OnChunk],
        on_delta: Optional[OnDelta],
        on_reasoning: Optional[OnReasoning],
    ) -> Iterator[str]:
        self._record_xai_reasoning_event(
            event,
            state,
            on_reasoning=on_reasoning,
        )
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
        await self._record_xai_reasoning_event_async(
            event,
            state,
            on_reasoning=on_reasoning,
        )
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
        chat_response = self._finalize_stream_response(
            XAIResponsesStreamParser.finalize(
                state,
                model=self.model,
                capture_reasoning=capture_reasoning,
            ),
            reasoning_collector=state.reasoning_collector,
            effective_schema=effective_schema,
            response_model=response_model,
        )
        self._apply_xai_reported_cost(
            chat_response,
            state.reported_cost_in_usd_ticks,
        )
        return chat_response

    def _finalize_xai_chat_response(
        self,
        response: dict[str, Any],
        *,
        effective_schema: Optional[dict],
        response_model: Optional[Any],
        capture_reasoning: bool,
    ) -> ChatResponse:
        """Parse, normalize and price one non-streaming xAI response."""
        chat_response = self._finalize_chat_response(
            self._parse_response(response, capture_reasoning=capture_reasoning),
            effective_schema=effective_schema,
            response_model=response_model,
        )
        self._apply_xai_reported_cost(
            chat_response,
            self._reported_cost_in_usd_ticks(response),
        )
        return chat_response

    def _record_xai_reasoning_event(
        self,
        event: SSEEvent,
        state: XAIResponsesStreamState,
        *,
        on_reasoning: Optional[OnReasoning],
    ) -> None:
        if state.reasoning_collector is None or state.reasoning_response is None:
            return
        reasoning = self._xai_reasoning_delta(event)
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

    async def _record_xai_reasoning_event_async(
        self,
        event: SSEEvent,
        state: XAIResponsesStreamState,
        *,
        on_reasoning: Optional[AsyncOnReasoning],
    ) -> None:
        if state.reasoning_collector is None or state.reasoning_response is None:
            return
        reasoning = self._xai_reasoning_delta(event)
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

    @staticmethod
    def _xai_reasoning_delta(event: SSEEvent) -> Optional[tuple[str, str]]:
        payload = event.data if isinstance(event.data, dict) else {}
        event_type = event.event or payload.get("type")
        if event_type not in {
            "response.reasoning_text.delta",
            "response.reasoning_summary_text.delta",
        }:
            return None
        delta = payload.get("delta")
        if not isinstance(delta, str) or not delta:
            return None
        kind = (
            "summary"
            if event_type == "response.reasoning_summary_text.delta"
            else "content"
        )
        return delta, kind

    @classmethod
    def _validate_xai_structured_schema(cls, schema: dict) -> dict:
        """Reject schema forms which xAI documents as a 400 response."""
        cls._validate_xai_schema_node(schema, path="$")
        return schema

    @classmethod
    def _validate_xai_schema_node(cls, node: Any, *, path: str) -> None:
        if isinstance(node, bool):
            raise JSONSchemaError(
                detail=(
                    f"xAI structured output rejects boolean schemas at {path}; "
                    "use an object schema instead."
                ),
            )
        if not isinstance(node, dict):
            return

        if "minContains" in node or "maxContains" in node:
            raise JSONSchemaError(
                detail=(
                    f"xAI structured output rejects minContains/maxContains at "
                    f"{path}."
                ),
            )
        if isinstance(node.get("items"), list):
            raise JSONSchemaError(
                detail=(
                    f"xAI structured output rejects an items array at {path}; "
                    "use prefixItems for tuple validation."
                ),
            )
        for keyword in ("enum", "anyOf"):
            value = node.get(keyword)
            if isinstance(value, list) and not value:
                raise JSONSchemaError(
                    detail=(
                        f"xAI structured output rejects an empty {keyword} at "
                        f"{path}."
                    ),
                )
        pattern = node.get("pattern")
        if isinstance(pattern, str) and re.search(
            r"(?<!\\)\\(?:[1-9]|[bBpP]|k<)",
            pattern,
        ):
            raise JSONSchemaError(
                detail=(
                    f"xAI structured output does not support this regex construct "
                    f"at {path}.pattern."
                ),
            )
        cls._validate_xai_schema_mapping(
            node.get("properties"),
            path=f"{path}.properties",
        )
        cls._validate_xai_schema_mapping(node.get("$defs"), path=f"{path}.$defs")
        cls._validate_xai_schema_mapping(
            node.get("dependentSchemas"),
            path=f"{path}.dependentSchemas",
        )
        for keyword in (
            "items",
            "contains",
            "not",
            "if",
            "then",
            "else",
            "propertyNames",
        ):
            value = node.get(keyword)
            if value is not None:
                cls._validate_xai_schema_node(value, path=f"{path}.{keyword}")
        additional_properties = node.get("additionalProperties")
        if isinstance(additional_properties, dict):
            cls._validate_xai_schema_node(
                additional_properties,
                path=f"{path}.additionalProperties",
            )
        for keyword in ("allOf", "anyOf", "oneOf", "prefixItems"):
            value = node.get(keyword)
            if isinstance(value, list):
                for index, item in enumerate(value):
                    cls._validate_xai_schema_node(
                        item,
                        path=f"{path}.{keyword}[{index}]",
                    )

    @classmethod
    def _validate_xai_schema_mapping(cls, value: Any, *, path: str) -> None:
        if not isinstance(value, dict):
            return
        for name, schema in value.items():
            cls._validate_xai_schema_node(schema, path=f"{path}.{name}")

    @staticmethod
    def _reported_cost_in_usd_ticks(response: dict[str, Any]) -> Optional[int]:
        usage = response.get("usage")
        if not isinstance(usage, dict):
            return None
        value = usage.get("cost_in_usd_ticks")
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
        return None

    @staticmethod
    def _apply_xai_reported_cost(
        chat_response: ChatResponse,
        cost_in_usd_ticks: Optional[int],
    ) -> None:
        """Prefer xAI's exact per-request cost over token-rate estimation."""
        if cost_in_usd_ticks is None:
            return
        chat_response.currency = "USD"
        chat_response.cost_input = None
        chat_response.cost_output = None
        chat_response.cost_total = cost_in_usd_ticks / 10_000_000_000

    def _materialize_responses_parameters(
        self,
        prepared: _PreparedResponsesRequest,
        uploaded_document_ids: tuple[str, ...],
    ) -> dict[str, Any]:
        """Add request-specific file IDs after uploading private PDF bytes."""
        parameters = dict(prepared.parameters)
        parameters["input"] = self._to_xai_responses_input(
            prepared.normalized_messages,
            uploaded_document_ids=uploaded_document_ids,
        )
        return parameters

    def _upload_document_parts(
        self,
        documents: tuple[DocumentPart, ...],
        timeout_s: Optional[float],
    ) -> tuple[str, ...]:
        """Upload byte-backed PDFs with a bounded xAI-owned lifecycle."""
        return tuple(
            self._client.upload_file(
                content=self._document_upload_content(document),
                filename="document.pdf",
                content_type="application/pdf",
                expires_after=_XAI_ADAPTER_FILE_TTL_SECONDS,
                timeout=timeout_s,
            )
            for document in documents
        )

    async def _aupload_document_parts(
        self,
        documents: tuple[DocumentPart, ...],
        timeout_s: Optional[float],
    ) -> tuple[str, ...]:
        """Async counterpart of :meth:`_upload_document_parts`."""
        uploaded_ids: list[str] = []
        for document in documents:
            uploaded_ids.append(
                await self._async_client.upload_file(
                    content=self._document_upload_content(document),
                    filename="document.pdf",
                    content_type="application/pdf",
                    expires_after=_XAI_ADAPTER_FILE_TTL_SECONDS,
                    timeout=timeout_s,
                )
            )
        return tuple(uploaded_ids)

    @staticmethod
    def _document_uploads(messages: Messages) -> tuple[DocumentPart, ...]:
        """Return PDFs that need an xAI upload rather than a public URL."""
        return tuple(
            part
            for message in messages.items
            if isinstance(message, UserMessage) and message.files
            for part in message.files
            if isinstance(part, DocumentPart) and not part._is_url()
        )

    @staticmethod
    def _document_upload_content(document: DocumentPart) -> bytes:
        if document.data is not None:
            return document.data
        try:
            return base64.b64decode(document._get_b64_data(), validate=True)
        except (ValueError, IndexError, binascii.Error) as error:
            raise ValueError("xAI PDF data must be valid base64") from error

    @staticmethod
    def _map_tools(tools: Optional[List[ToolSpec]]) -> Optional[list[dict[str, Any]]]:
        if not tools:
            return None

        mapped_tools: list[dict[str, Any]] = []
        for tool in tools:
            if not isinstance(tool.description, str) or not tool.description.strip():
                raise InvalidToolSchemaError(
                    detail=f"xAI function {tool.name!r} requires a description",
                )
            mapped_tools.append(
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.json_schema,
                }
            )
        return mapped_tools

    @staticmethod
    def _map_tool_choice(tool_choice: Optional[str]) -> Any:
        if tool_choice is None:
            return None
        if tool_choice in {"auto", "none"}:
            return tool_choice
        if tool_choice == "any":
            return "required"
        return {"type": "function", "name": tool_choice}

    @staticmethod
    def _to_xai_responses_input(
        messages: Messages,
        *,
        uploaded_document_ids: tuple[str, ...] = (),
    ) -> list[dict[str, Any]]:
        """Serialize a full tool round-trip without Responses server state."""
        input_items: list[dict[str, Any]] = []
        document_ids = iter(uploaded_document_ids)
        for message in messages.items:
            if isinstance(message, Prompt):
                continue
            if isinstance(message, AIMessage):
                if message.content:
                    input_items.append(
                        {"role": "assistant", "content": message.content},
                    )
                for tool_call in message.tool_calls or []:
                    if not tool_call.call_id:
                        raise ValueError(
                            "xAI function_call history requires a non-empty call_id",
                        )
                    input_items.append(
                        {
                            "type": "function_call",
                            "call_id": tool_call.call_id,
                            "name": tool_call.name,
                            "arguments": json.dumps(
                                tool_call.arguments,
                                ensure_ascii=False,
                            ),
                        }
                )
                continue
            if isinstance(message, UserMessage) and message.files:
                content: list[dict[str, Any]] = [
                    {"type": "input_text", "text": message.content},
                ]
                for part in message.files:
                    if isinstance(part, ImagePart):
                        image_url = part.url if part._is_url() else part._to_data_uri()
                        content.append(
                            {"type": "input_image", "image_url": image_url},
                        )
                    elif isinstance(part, DocumentPart):
                        if part._is_url():
                            content.append(
                                {"type": "input_file", "file_url": part.url},
                            )
                        else:
                            try:
                                file_id = next(document_ids)
                            except StopIteration as error:
                                raise ValueError(
                                    "xAI document upload did not return a file id",
                                ) from error
                            content.append(
                                {"type": "input_file", "file_id": file_id},
                            )
                    else:
                        raise ValueError(
                            f"xAI does not support {type(part).__name__} file input",
                        )
                input_items.append({"role": "user", "content": content})
                continue
            input_items.extend(message.to_openai_responses_input())
        try:
            next(document_ids)
        except StopIteration:
            return input_items
        raise ValueError("xAI received unused uploaded document file ids")

    @staticmethod
    def _parse_response(
        response: dict[str, Any],
        *,
        capture_reasoning: bool = False,
    ) -> ChatResponse:
        if response.get("object") != "response":
            raise LLMAPIClientError(
                detail="xAI Responses API returned an invalid response object",
            )
        if not isinstance(response.get("output"), list):
            raise LLMAPIClientError(
                detail="xAI Responses API response.output must be an array",
            )
        return ChatResponse.from_openai_responses_response(
            response,
            capture_reasoning=capture_reasoning,
        )


__all__ = ["XAIAdapter"]
