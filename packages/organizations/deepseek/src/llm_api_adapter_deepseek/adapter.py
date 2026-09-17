"""Core facade adapter for DeepSeek's official Responses API."""

from __future__ import annotations

import base64
import binascii
from copy import deepcopy
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
from typing import Any, AsyncIterator, Iterator, List, Mapping, Optional
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
)
from llm_api_adapter.adapters.structured_output import (
    validate_core_portable_schema,
)
from llm_api_adapter.errors.llm_api_error import (
    InvalidToolSchemaError,
    LLMAPIClientError,
    LLMAPIError,
)
from llm_api_adapter.errors.config_errors import LLMConfigError
from llm_api_adapter.models.messages.chat_message import (
    AIMessage,
    Message,
    Messages,
    Prompt,
    ToolMessage,
    UserMessage,
)
from llm_api_adapter.models.messages.file_parts import ImagePart
from llm_api_adapter.models.responses.chat_response import ChatResponse, Usage
from llm_api_adapter.models.tools import ToolSpec

from .clients.sync_client import (
    DEEPSEEK_RESPONSES_URL,
    DeepSeekResponsesSyncClient,
)
from .clients.async_client import DeepSeekResponsesAsyncClient
from .registry.cache_pricing import (
    DeepSeekFlashPricing,
    pricing_for_dispatch,
)
from .streaming import (
    DeepSeekResponsesStreamParser,
    DeepSeekResponsesStreamState,
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
logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    """Return an aware UTC instant for time-of-use pricing."""
    return datetime.now(timezone.utc)


def _reset_dispatch_time(token: Any) -> None:
    """Reset a request context token when the async generator context permits it."""
    try:
        _DEEPSEEK_DISPATCH_TIME.reset(token)
    except ValueError:
        # Async-generator cancellation may finalize in a different context.
        pass


@dataclass
class DeepSeekUsage(Usage):
    """Provider usage with optional cache and reasoning token details."""

    cached_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None


@dataclass(frozen=True)
class _PreparedResponsesRequest:
    """Normalized request data shared by the adapter's request paths."""

    parameters: dict[str, Any]
    normalized_messages: Messages
    effective_schema: Optional[dict]
    response_model: Optional[Any]
    capture_reasoning: bool


@dataclass(repr=False)
class DeepSeekAdapter(LLMAdapterBase):
    """Map Core's text-chat contract to DeepSeek Responses requests."""

    company: str = "deepseek"
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
        super().__post_init__()
        self._client = DeepSeekResponsesSyncClient(
            api_key=self.api_key,
            transport=self.transport,
            endpoint=self.endpoint,
        )
        self._async_client = DeepSeekResponsesAsyncClient(
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
        """Create one normalized response through ``POST /responses``."""
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
        dispatch_token = None
        try:
            dispatch_token = _DEEPSEEK_DISPATCH_TIME.set(_utc_now())
            response = self._client.create(
                model=self.model,
                timeout=timeout_s,
                **prepared.parameters,
            )
            return self._finalize_deepseek_chat_response(
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
        finally:
            if dispatch_token is not None:
                _reset_dispatch_time(dispatch_token)

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
        """Create one normalized response through Core's async transport."""
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
        dispatch_token = None
        try:
            dispatch_token = _DEEPSEEK_DISPATCH_TIME.set(_utc_now())
            response = await self._async_client.create(
                model=self.model,
                timeout=timeout_s,
                **prepared.parameters,
            )
            return self._finalize_deepseek_chat_response(
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
        finally:
            if dispatch_token is not None:
                _reset_dispatch_time(dispatch_token)

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
        dispatch_token = _DEEPSEEK_DISPATCH_TIME.set(_utc_now())
        try:
            events = self._client.stream(
                model=self.model,
                timeout=timeout_s,
                **prepared.parameters,
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
        finally:
            _reset_dispatch_time(dispatch_token)

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
        dispatch_token = _DEEPSEEK_DISPATCH_TIME.set(_utc_now())
        try:
            events = self._async_client.stream(
                model=self.model,
                timeout=timeout_s,
                **prepared.parameters,
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
        finally:
            _reset_dispatch_time(dispatch_token)

    def _consume_stream_event(
        self,
        event: Any,
        state: DeepSeekResponsesStreamState,
        *,
        on_chunk: Optional[OnChunk],
        on_delta: Optional[OnDelta],
        on_reasoning: Optional[OnReasoning],
    ) -> Iterator[str]:
        self._record_deepseek_reasoning_event(
            event,
            state,
            on_reasoning=on_reasoning,
        )
        delta = DeepSeekResponsesStreamParser.consume_event(event, state)
        if delta is not None:
            yield from self._emit_stream_chunks(
                state.chunk_buffer.add(delta),
                on_chunk,
                on_delta,
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
        await self._record_deepseek_reasoning_event_async(
            event,
            state,
            on_reasoning=on_reasoning,
        )
        delta = DeepSeekResponsesStreamParser.consume_event(event, state)
        if delta is not None:
            async for text in self._emit_async_stream_chunks(
                state.chunk_buffer.add(delta),
                on_chunk,
                on_delta,
            ):
                yield text

    def _finalize_stream(
        self,
        state: DeepSeekResponsesStreamState,
        *,
        capture_reasoning: bool,
        effective_schema: Optional[dict],
        response_model: Optional[Any],
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
        self._store_reasoning_replay(chat_response, state.final_response or {})
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
        temperature, top_p = self._validate_sampling_parameters(temperature, top_p)
        request_context = self._prepare_chat_request(
            messages,
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
        self._validate_deepseek_file_inputs(normalized_messages)
        effective_schema = request_context.effective_schema
        if effective_schema is not None:
            effective_schema = validate_core_portable_schema(
                effective_schema,
                provider="deepseek",
            )
        parameters: dict[str, Any] = {
            "input": self._to_deepseek_responses_input(
                normalized_messages,
                reasoning_replay=reasoning_replay,
            ),
            "max_output_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "tools": self._map_tools(tools),
            "tool_choice": self._map_tool_choice(
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
            normalized_messages=normalized_messages,
            effective_schema=effective_schema,
            response_model=request_context.response_model,
            capture_reasoning=capture_reasoning,
        )

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
        """Reject files outside DeepSeek's verified Responses image boundary."""
        image_count = 0
        for message in messages.items:
            files = getattr(message, "files", None)
            if files is None:
                continue
            if not isinstance(message, UserMessage):
                raise ValueError(
                    "DeepSeek input_image parts are supported only in user messages",
                )
            if not isinstance(files, list):
                raise ValueError("DeepSeek user message files must be a list")

            for part in files:
                if not isinstance(part, ImagePart):
                    raise ValueError(
                        "DeepSeek Responses supports only image inputs; "
                        "DocumentPart and non-image file inputs are unsupported",
                    )
                image_count += 1
                if image_count > _MAX_IMAGES_PER_REQUEST:
                    raise ValueError(
                        "DeepSeek supports at most 600 images per request",
                    )
                DeepSeekAdapter._validate_image_part(part)

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

    @staticmethod
    def _to_deepseek_responses_input(
        messages: Messages,
        *,
        reasoning_replay: tuple[dict[str, Any], ...] = (),
    ) -> list[dict[str, Any]]:
        """Serialize an application-controlled Responses tool round-trip."""
        input_items: list[dict[str, Any]] = []
        last_assistant_start: Optional[int] = None
        for message in messages.items:
            if isinstance(message, Prompt):
                continue
            if isinstance(message, AIMessage):
                last_assistant_start = len(input_items)
                if message.content:
                    input_items.append(
                        {"role": "assistant", "content": message.content},
                    )
                for tool_call in message.tool_calls or []:
                    if not tool_call.call_id:
                        raise ValueError(
                            "DeepSeek function_call history requires a non-empty "
                            "call_id",
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
                        },
                    )
                continue
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
    def _map_tools(tools: Optional[List[ToolSpec]]) -> Optional[list[dict[str, Any]]]:
        if not tools:
            return None
        mapped_tools: list[dict[str, Any]] = []
        for tool in tools:
            if not isinstance(tool.description, str) or not tool.description.strip():
                raise InvalidToolSchemaError(
                    detail=f"DeepSeek function {tool.name!r} requires a description",
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
        """Normalize only complete, internally consistent provider usage."""
        if not isinstance(raw_usage, Mapping):
            return None

        input_tokens = raw_usage.get("input_tokens")
        output_tokens = raw_usage.get("output_tokens")
        total_tokens = raw_usage.get("total_tokens")
        required = (input_tokens, output_tokens, total_tokens)
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in required
        ):
            return None
        if total_tokens != input_tokens + output_tokens:
            return None

        cached_tokens: Optional[int] = None
        input_details = raw_usage.get("input_tokens_details")
        if input_details is not None:
            if not isinstance(input_details, Mapping):
                return None
            if "cached_tokens" in input_details:
                cached_tokens = input_details.get("cached_tokens")
                if cached_tokens is not None and (
                    isinstance(cached_tokens, bool)
                    or not isinstance(cached_tokens, int)
                    or cached_tokens < 0
                    or cached_tokens > input_tokens
                ):
                    return None

        reasoning_tokens: Optional[int] = None
        output_details = raw_usage.get("output_tokens_details")
        if output_details is not None:
            if not isinstance(output_details, Mapping):
                return None
            if "reasoning_tokens" in output_details:
                reasoning_tokens = output_details.get("reasoning_tokens")
                if reasoning_tokens is not None and (
                    isinstance(reasoning_tokens, bool)
                    or not isinstance(reasoning_tokens, int)
                    or reasoning_tokens < 0
                    or reasoning_tokens > output_tokens
                ):
                    return None

        return DeepSeekUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cached_tokens=cached_tokens,
            reasoning_tokens=reasoning_tokens,
        )

    def _finalize_deepseek_chat_response(
        self,
        response: dict[str, Any],
        *,
        effective_schema: Optional[dict],
        response_model: Optional[Any],
        capture_reasoning: bool,
    ) -> ChatResponse:
        """Parse and run Core structured-output/pricing finalization."""
        chat_response = self._parse_response(
            response,
            capture_reasoning=capture_reasoning,
        )
        self._store_reasoning_replay(chat_response, response)
        self._prepare_structured_output_response(
            chat_response,
            effective_schema,
            response_model,
        )
        self._apply_response_pricing(chat_response)
        return chat_response

    def _apply_response_pricing(self, chat_response: ChatResponse) -> None:
        """Apply only verifiable DeepSeek time-of-use standard estimates."""
        chat_response.currency = None
        chat_response.cost_input = None
        chat_response.cost_output = None
        chat_response.cost_total = None

        usage = chat_response.usage
        if usage is None or not isinstance(usage, Usage):
            return

        dispatch_time = _DEEPSEEK_DISPATCH_TIME.get() or _utc_now()
        pricing = pricing_for_dispatch(dispatch_time)
        if (
            not isinstance(pricing, DeepSeekFlashPricing)
            or not pricing.is_valid()
            or self.model != "deepseek-flash"
            or self.model_spec is None
        ):
            return

        if isinstance(usage, DeepSeekUsage) and usage.cached_tokens is not None:
            uncached_tokens = usage.input_tokens - usage.cached_tokens
            chat_response.cost_input = (
                usage.cached_tokens * pricing.cache_hit_input_per_token
                + uncached_tokens * pricing.cache_miss_input_per_token
            )
        chat_response.cost_output = usage.output_tokens * pricing.output_per_token
        chat_response.currency = "USD"
        if (
            chat_response.cost_input is not None
            and chat_response.cost_output is not None
        ):
            chat_response.cost_total = (
                chat_response.cost_input + chat_response.cost_output
            )

    def _store_reasoning_replay(
        self,
        chat_response: ChatResponse,
        response: Mapping[str, Any],
    ) -> None:
        """Keep continuation material opaque and outside visible response fields."""
        items = self._reasoning_replay_items(response.get("output"))
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
