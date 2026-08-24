from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
import json
import logging
import inspect
import re
from typing import (
    Any,
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
)
import warnings

from ..errors.llm_api_error import (
    InvalidToolSchemaError,
    JSONSchemaError,
    LLMAPIError,
    ToolChoiceError,
)
from ..llm_registry.llm_registry import (
    LLM_REGISTRY,
    ModelSpec,
    Pricing,
    resolve_model_spec,
)
from ..llm_registry.reasoning import ReasoningResolution, resolve_reasoning_level
from ..models.messages.chat_message import Messages
from ..models.responses.chat_response import ChatResponse
from ..models.responses.reasoning_event import (
    ReasoningEvent,
    ReasoningEventKind,
)
from ..models.responses.stream_chunk import StreamChunk
from ..llms.streaming import (
    StreamChunkBuffer,
    StreamReasoningCollector,
    StreamUsageTracker,
)
from ..llms.transports import validate_sync_transport
from ..models.tools import ToolCall, ToolSpec

logger = logging.getLogger(__name__)

TOOL_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


@dataclass(frozen=True)
class _ChatRequestContext:
    normalized_messages: Messages
    effective_schema: Optional[dict]
    normalized_tool_choice: Optional[str]


OnDelta = Callable[[str], None]
OnChunk = Callable[[StreamChunk], None]
OnToolCall = Callable[[ToolCall], None]
OnDone = Callable[[ChatResponse], None]
OnReasoning = Callable[[ReasoningEvent], None]
AsyncCallbackResult = Optional[Awaitable[None]]
AsyncOnDelta = Callable[[str], AsyncCallbackResult]
AsyncOnChunk = Callable[[StreamChunk], AsyncCallbackResult]
AsyncOnToolCall = Callable[[ToolCall], AsyncCallbackResult]
AsyncOnDone = Callable[[ChatResponse], AsyncCallbackResult]
AsyncOnReasoning = Callable[[ReasoningEvent], AsyncCallbackResult]


@dataclass
class _StreamState:
    chunk_buffer: StreamChunkBuffer
    usage_tracker: StreamUsageTracker
    reasoning_collector: Optional[StreamReasoningCollector]
    reasoning_response: Optional[ChatResponse]


@dataclass
class LLMAdapterBase(ABC):
    api_key: str
    model: str
    company: str
    transport: str = "requests"
    pricing: Optional[Pricing] = None
    is_reasoning: bool = False
    is_adaptive_thinking: bool = False
    service_provider: Optional[str] = None
    model_spec: Optional[ModelSpec] = field(default=None, init=False, repr=False)

    def __repr__(self) -> str:
        masked = f"{self.api_key[:8]}...{self.api_key[-4:]}" if len(self.api_key) > 12 else "***"
        return (
            f"{self.__class__.__name__}(company='{self.company}', "
            f"model='{self.model}', transport='{self.transport}', api_key='{masked}')"
        )

    def __post_init__(self):
        if not self.api_key:
            error_message = "api_key must be a non-empty string"
            logger.error(error_message)
            raise ValueError(error_message)
        if self.service_provider is None:
            self.service_provider = self.company
        elif (
            not isinstance(self.service_provider, str)
            or not self.service_provider
        ):
            error_message = "service_provider must be a non-empty string"
            logger.error(error_message)
            raise ValueError(error_message)
        self.transport = validate_sync_transport(self.transport)
        model_spec = resolve_model_spec(LLM_REGISTRY, self.company, self.model)
        self.model_spec = model_spec
        if not model_spec:
            warnings.warn(
                (
                    f"Model '{self.model}' is not verified for the {self.company} adapter. "
                    f"Continuing with the selected adapter."
                ),
                UserWarning,
            )
            logger.warning(f"Unverified model used: {self.model}")
            self.pricing = None
        else:
            base_pricing = getattr(model_spec, "pricing_tiers", None)
            self.pricing = deepcopy(base_pricing) if base_pricing else None
            self.is_reasoning = getattr(model_spec, "is_reasoning", False)
            self.is_adaptive_thinking = getattr(model_spec, "is_adaptive_thinking", False)

    @abstractmethod
    def chat(
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
        *,
        capture_reasoning: bool = False,
    ) -> ChatResponse:
        """
        Generates a response based on the provided conversation.

        ``capture_reasoning`` is an opt-in request flag implemented by
        provider adapters; the default preserves the existing request.
        """
        raise NotImplementedError

    @abstractmethod
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
        """Stream normalized text deltas synchronously.

        ``buffer_chars`` optionally coalesces visible text into bounded
        chunks.  ``on_chunk`` receives each emitted :class:`StreamChunk`
        before the corresponding ``on_delta`` callback and yielded text.
        ``on_reasoning`` receives provider-normalized reasoning events only
        when ``capture_reasoning`` is enabled; reasoning is never yielded as
        visible text.
        """
        raise NotImplementedError

    async def achat(
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
        *,
        capture_reasoning: bool = False,
    ) -> ChatResponse:
        """Generate one response asynchronously in provider adapters."""
        raise NotImplementedError

    def astream_chat(
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
        """Stream normalized text deltas asynchronously in provider adapters."""
        raise NotImplementedError

    def _resolve_reasoning_level(
        self,
        reasoning_level: str | int,
    ) -> ReasoningResolution:
        """Resolve a public reasoning level for this adapter's verified model."""
        if self.model_spec is None:
            resolution = ReasoningResolution(
                provider_value=None,
                reason="unverified_model",
                warning=(
                    f"Model '{self.model}' is not verified for the {self.company} "
                    "adapter; reasoning is disabled."
                ),
            )
        else:
            resolution = resolve_reasoning_level(self.model_spec, reasoning_level)

        if resolution.warning:
            warnings.warn(resolution.warning, UserWarning, stacklevel=2)
            logger.info(resolution.warning)
        return resolution

    def _validate_parameter(
        self, name: str, value: float, min_value: float, max_value: float
    ) -> float:
        if not (min_value <= value <= max_value):
            error_message = (
                f"{name} must be between {min_value} and {max_value}, got {value}"
            )
            logger.error(error_message)
            raise ValueError(error_message)
        return value

    def _validate_sampling_parameters(
        self,
        temperature: float,
        top_p: float,
    ) -> tuple[float, float]:
        """Validate the sampling parameters shared by all providers."""
        return (
            self._validate_parameter("temperature", temperature, 0, 2),
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
        """Validate and normalize provider-neutral chat inputs."""
        self._validate_tools(tools)
        effective_schema = self._resolve_json_schema(
            json_schema,
            response_model,
            tools,
        )
        normalized_tool_choice = self._normalize_tool_choice(
            tool_choice,
            tools,
        )
        normalized_messages = self._normalize_messages(messages)
        return _ChatRequestContext(
            normalized_messages=normalized_messages,
            effective_schema=effective_schema,
            normalized_tool_choice=normalized_tool_choice,
        )

    def _normalize_messages(self, messages: Any) -> Messages:
        if isinstance(messages, Messages):
            return messages
        if isinstance(messages, list):
            return Messages(messages)
        raise TypeError("messages must be a list or Messages instance")

    def _validate_tools(self, tools: Optional[List[ToolSpec]]) -> None:
        """
        Contract-level validation only.
        Provider-specific constraints are handled in provider adapters.
        """
        if tools is None:
            return
        if not isinstance(tools, list):
            raise InvalidToolSchemaError(detail="tools must be a list[ToolSpec] or None")
        seen: set[str] = set()
        for t in tools:
            if not isinstance(t, ToolSpec):
                raise InvalidToolSchemaError(
                    detail="tools must contain ToolSpec items only"
                )
            if not t.name or not isinstance(t.name, str):
                raise InvalidToolSchemaError(
                    detail="ToolSpec.name must be a non-empty string"
                )
            if not TOOL_NAME_RE.match(t.name):
                raise InvalidToolSchemaError(
                    detail=(
                        f"Invalid tool name {t.name!r}. "
                        "Must match ^[a-zA-Z0-9_-]{1,64}$"
                    )
                )
            if t.name in seen:
                raise InvalidToolSchemaError(detail=f"Duplicate tool name: {t.name!r}")
            seen.add(t.name)
            if not isinstance(t.json_schema, dict):
                raise InvalidToolSchemaError(
                    detail=f"Tool {t.name!r}: json_schema must be a dict"
                )

    def _normalize_tool_choice(
        self,
        tool_choice: Any,
        tools: Optional[List[ToolSpec]],
    ) -> Optional[str]:
        if tool_choice is None:
            return None
        tool_names = self._get_tool_names(tools)
        if isinstance(tool_choice, str):
            return self._normalize_tool_choice_from_str(tool_choice, tools, tool_names)
        if isinstance(tool_choice, dict):
            return self._normalize_tool_choice_from_dict(tool_choice, tools, tool_names)
        raise ToolChoiceError(
            detail=f"Invalid tool_choice type: {type(tool_choice).__name__}"
        )

    def _get_tool_names(self, tools: Optional[List[ToolSpec]]) -> set[str]:
        return {tool.name for tool in tools or []}

    def _normalize_tool_choice_from_str(
        self,
        tool_choice: str,
        tools: Optional[List[ToolSpec]],
        tool_names: set[str],
    ) -> str:
        if tool_choice == "required":
            self._raise_required_tool_choice_error()
        if tool_choice in ("auto", "none"):
            return tool_choice
        if tool_choice == "any":
            self._ensure_tools_provided(tools, detail="tool_choice='any' requires tools to be provided")
            return "any"
        self._ensure_tools_provided(
            tools,
            detail="tool_choice references a tool but tools=None",
        )
        if tool_choice in tool_names:
            return tool_choice
        raise ToolChoiceError(detail=f"Unknown tool_choice string: {tool_choice!r}")

    def _normalize_tool_choice_from_dict(
        self,
        tool_choice: Dict[str, Any],
        tools: Optional[List[ToolSpec]],
        tool_names: set[str],
    ) -> str:
        tc_type = tool_choice.get("type")
        tc_name = tool_choice.get("name")
        if tc_type == "required":
            self._raise_required_tool_choice_error()
        if tc_type in ("auto", "none"):
            return tc_type
        if tc_type == "any":
            self._ensure_tools_provided(
                tools,
                detail=f"tool_choice.type={tc_type!r} requires tools to be provided",
            )
            return "any"
        if tc_type == "tool":
            return self._normalize_named_tool_choice(tc_name, tools, tool_names)
        if isinstance(tc_name, str):
            return self._normalize_named_tool_choice(tc_name, tools, tool_names)
        raise ToolChoiceError(detail=f"Invalid tool_choice dict: {tool_choice!r}")

    def _normalize_named_tool_choice(
        self,
        tool_name: Any,
        tools: Optional[List[ToolSpec]],
        tool_names: set[str],
    ) -> str:
        if not isinstance(tool_name, str) or not tool_name:
            raise ToolChoiceError(
                detail="tool_choice.type='tool' requires non-empty name"
            )
        self._ensure_tools_provided(
            tools,
            detail="tool_choice references a tool but tools=None",
        )
        if tool_name not in tool_names:
            raise ToolChoiceError(
                detail=f"tool_choice references unknown tool: {tool_name!r}"
            )
        return tool_name

    def _ensure_tools_provided(
        self,
        tools: Optional[List[ToolSpec]],
        *,
        detail: str,
    ) -> None:
        if not tools:
            raise ToolChoiceError(detail=detail)

    def _raise_required_tool_choice_error(self) -> None:
        raise ToolChoiceError(
            detail="tool_choice='required' is not supported; use 'any'"
        )

    def _enforce_strict_schema(self, schema: dict) -> dict:
        """Recursively add additionalProperties: false to all object types — required by OpenAI and Anthropic strict mode."""
        schema = dict(schema)
        if schema.get("type") == "object":
            schema["additionalProperties"] = False
            if "properties" in schema:
                schema["properties"] = {
                    k: self._enforce_strict_schema(v) if isinstance(v, dict) else v
                    for k, v in schema["properties"].items()
                }
        if "items" in schema and isinstance(schema["items"], dict):
            schema["items"] = self._enforce_strict_schema(schema["items"])
        return schema


    def _resolve_json_schema(
        self,
        json_schema: Optional[dict],
        response_model: Optional[Any],
        tools: Optional[List[ToolSpec]],
    ) -> Optional[dict]:
        if json_schema is not None and response_model is not None:
            raise JSONSchemaError(detail="json_schema and response_model cannot be used together")
        if response_model is not None and tools is not None:
            raise JSONSchemaError(detail="response_model and tools cannot be used together")
        if json_schema is not None and tools is not None:
            raise JSONSchemaError(detail="json_schema and tools cannot be used together")
        if response_model is not None:
            try:
                return response_model.model_json_schema()
            except AttributeError:
                try:
                    import pydantic  # noqa
                except ImportError:
                    raise JSONSchemaError(
                        detail="pydantic is required for response_model; install it with: pip install pydantic"
                    )
                raise JSONSchemaError(detail="response_model must be a Pydantic BaseModel subclass")
        if json_schema is not None and not isinstance(json_schema, dict):
            raise JSONSchemaError(detail="json_schema must be a dict")
        return json_schema

    def _strip_json_fences(self, content: str) -> str:
        text = content.strip()
        fence_match = re.match(r'^```(?:json)?\s*([\s\S]*?)\s*```$', text, re.IGNORECASE)
        if fence_match:
            return fence_match.group(1).strip()
        block_match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', text)
        if block_match:
            return block_match.group(1)
        return text

    def _parse_json_response(
        self,
        content: Optional[str],
        json_schema: Optional[dict],
    ) -> Optional[dict]:
        if json_schema is None or content is None:
            return None
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            try:
                return json.loads(self._strip_json_fences(content))
            except json.JSONDecodeError as e:
                raise JSONSchemaError(detail=f"Model response is not valid JSON: {e}")

    def _parse_response_model(
        self,
        parsed_json: Optional[dict],
        response_model: Optional[Any],
    ) -> Optional[Any]:
        if response_model is None or parsed_json is None:
            return None
        try:
            return response_model.model_validate(parsed_json)
        except Exception:
            try:
                return response_model.model_validate(parsed_json, strict=False)
            except Exception as e:
                raise JSONSchemaError(detail=f"Response failed Pydantic validation: {e}")

    def _iter_provider_stream_events(self, events: Iterator[Any]) -> Iterator[Any]:
        """Yield provider events while preserving the adapter error contract.

        Callback invocation deliberately happens in the provider adapter,
        outside this generator, so user callback exceptions are not reported
        as provider failures.
        """
        try:
            yield from events
        except LLMAPIError as error:
            self.handle_error(error)
        except Exception as error:
            error_message = getattr(error, "text", None) or str(error)
            self.handle_error(error=error, error_message=error_message)

    async def _aiter_provider_stream_events(
        self,
        events: AsyncIterable[Any],
    ) -> AsyncIterator[Any]:
        """Yield async provider events while preserving the adapter error contract."""
        try:
            async for event in events:
                yield event
        except LLMAPIError as error:
            self.handle_error(error)
        except Exception as error:
            error_message = getattr(error, "text", None) or str(error)
            self.handle_error(error=error, error_message=error_message)

    def _run_sync_stream(
        self,
        events: Iterator[Any],
        state: Any,
        *,
        consume_event: Callable[..., Iterator[str]],
        finalize_response: Optional[Callable[..., ChatResponse]] = None,
        effective_schema: Optional[dict],
        response_model: Optional[Any],
        on_delta: Optional[OnDelta],
        on_tool_call: Optional[OnToolCall],
        on_done: Optional[OnDone],
        on_chunk: Optional[OnChunk],
        capture_reasoning: bool,
        on_reasoning: Optional[OnReasoning],
    ) -> Iterator[str]:
        """Run the shared synchronous stream lifecycle for a provider."""
        for event in self._iter_provider_stream_events(events):
            yield from consume_event(
                event,
                state,
                on_chunk=on_chunk,
                on_delta=on_delta,
                on_reasoning=on_reasoning,
            )

        finalizer = finalize_response or self._finalize_stream_response
        chat_response = finalizer(
            state,
            capture_reasoning=capture_reasoning,
            effective_schema=effective_schema,
            response_model=response_model,
        )
        yield from self._complete_stream(
            chat_response,
            state.chunk_buffer,
            on_chunk,
            on_delta,
            on_tool_call,
            on_done,
        )

    async def _run_async_stream(
        self,
        events: AsyncIterable[Any],
        state: Any,
        *,
        consume_event: Callable[..., AsyncIterator[str]],
        finalize_response: Optional[Callable[..., ChatResponse]] = None,
        effective_schema: Optional[dict],
        response_model: Optional[Any],
        on_delta: Optional[AsyncOnDelta],
        on_tool_call: Optional[AsyncOnToolCall],
        on_done: Optional[AsyncOnDone],
        on_chunk: Optional[AsyncOnChunk],
        capture_reasoning: bool,
        on_reasoning: Optional[AsyncOnReasoning],
    ) -> AsyncIterator[str]:
        """Run the shared asynchronous stream lifecycle for a provider."""
        async for event in self._aiter_provider_stream_events(events):
            async for text in consume_event(
                event,
                state,
                on_chunk=on_chunk,
                on_delta=on_delta,
                on_reasoning=on_reasoning,
            ):
                yield text

        finalizer = finalize_response or self._finalize_stream_response
        chat_response = finalizer(
            state,
            capture_reasoning=capture_reasoning,
            effective_schema=effective_schema,
            response_model=response_model,
        )
        async for text in self._complete_async_stream(
            chat_response,
            state.chunk_buffer,
            on_chunk,
            on_delta,
            on_tool_call,
            on_done,
        ):
            yield text

    async def _invoke_async_callback(
        self,
        callback: Optional[Callable[[Any], Any]],
        value: Any,
    ) -> None:
        """Invoke a callback and await its result when it is awaitable."""
        if callback is None:
            return
        result = callback(value)
        if inspect.isawaitable(result):
            await result

    def _apply_response_pricing(self, chat_response: ChatResponse) -> None:
        """Price a response from the provider-reported input-token count."""
        if not self.pricing:
            return

        if chat_response.usage is None:
            if len(self.pricing.tiers) != 1:
                return
            tier = self.pricing.tiers[0]
        else:
            tier = self.pricing.tier_for_prompt_tokens(
                chat_response.usage.input_tokens
            )
        chat_response.apply_pricing(
            price_input_per_token=tier.in_per_token,
            price_output_per_token=tier.out_per_token,
            currency=self.pricing.currency,
        )

    def _finalize_chat_response(
        self,
        chat_response: ChatResponse,
        *,
        effective_schema: Optional[dict],
        response_model: Optional[Any],
    ) -> ChatResponse:
        """Apply common structured-output parsing and pricing to a response."""
        chat_response.parsed_json = self._parse_json_response(
            chat_response.content,
            effective_schema,
        )
        chat_response.parsed_model = self._parse_response_model(
            chat_response.parsed_json,
            response_model,
        )
        self._apply_response_pricing(chat_response)
        return chat_response

    def _prepare_stream_response(
        self,
        chat_response: ChatResponse,
        effective_schema: Optional[dict],
        response_model: Optional[Any],
    ) -> None:
        """Apply final response processing before delivering stream callbacks."""
        try:
            chat_response.parsed_json = self._parse_json_response(
                chat_response.content,
                effective_schema,
            )
            chat_response.parsed_model = self._parse_response_model(
                chat_response.parsed_json,
                response_model,
            )
            self._apply_response_pricing(chat_response)
        except LLMAPIError as error:
            self.handle_error(error)
        except Exception as error:
            error_message = getattr(error, "text", None) or str(error)
            self.handle_error(error=error, error_message=error_message)

    def _invoke_stream_completion_callbacks(
        self,
        chat_response: ChatResponse,
        on_tool_call: Optional[OnToolCall],
        on_done: Optional[OnDone],
    ) -> None:
        """Deliver finalized tool calls and the completed response."""
        for tool_call in chat_response.tool_calls or []:
            if on_tool_call is not None:
                on_tool_call(tool_call)
        if on_done is not None:
            on_done(chat_response)

    def _record_reasoning_event(
        self,
        chat_response: ChatResponse,
        collector: StreamReasoningCollector,
        text: str,
        *,
        capture_reasoning: bool,
        kind: ReasoningEventKind = "summary",
        on_reasoning: Optional[OnReasoning] = None,
    ) -> Optional[ReasoningEvent]:
        """Append an opt-in reasoning event and then notify its callback.

        The response is updated before ``on_reasoning`` runs, so a later
        ``on_done`` callback observes the same sequence.  This helper is
        intentionally separate from visible-text emission and therefore does
        not call ``on_delta`` or yield anything.
        """
        if not capture_reasoning:
            return None

        event = collector.add(text, kind=kind)
        if event is None:
            return None

        chat_response.reasoning_events.append(event)
        if on_reasoning is not None:
            on_reasoning(event)
        return event

    def _emit_stream_chunks(
        self,
        chunks: Iterable[StreamChunk],
        on_chunk: Optional[OnChunk],
        on_delta: Optional[OnDelta],
    ) -> Iterator[str]:
        """Invoke streaming callbacks in contract order and yield visible text."""
        for chunk in chunks:
            if on_chunk is not None:
                on_chunk(chunk)
            if on_delta is not None:
                on_delta(chunk.text)
            yield chunk.text

    async def _emit_async_stream_chunks(
        self,
        chunks: Iterable[StreamChunk],
        on_chunk: Optional[AsyncOnChunk],
        on_delta: Optional[AsyncOnDelta],
    ) -> AsyncIterator[str]:
        """Invoke async-capable callbacks before yielding each visible chunk."""
        for chunk in chunks:
            await self._invoke_async_callback(on_chunk, chunk)
            await self._invoke_async_callback(on_delta, chunk.text)
            yield chunk.text

    def _finalize_stream_response(
        self,
        chat_response: ChatResponse,
        *,
        reasoning_collector: Optional[StreamReasoningCollector] = None,
        effective_schema: Optional[dict],
        response_model: Optional[Any],
    ) -> ChatResponse:
        """Apply common reasoning, structured-output and pricing finalization."""
        if reasoning_collector is not None:
            streamed_reasoning_events = reasoning_collector.snapshot()
            if streamed_reasoning_events:
                chat_response.reasoning_events = streamed_reasoning_events
        self._prepare_stream_response(
            chat_response,
            effective_schema,
            response_model,
        )
        return chat_response

    def _complete_stream(
        self,
        chat_response: ChatResponse,
        chunk_buffer: StreamChunkBuffer,
        on_chunk: Optional[OnChunk],
        on_delta: Optional[OnDelta],
        on_tool_call: Optional[OnToolCall],
        on_done: Optional[OnDone],
    ) -> Iterator[str]:
        """Flush visible text and invoke the common stream callbacks."""
        yield from self._emit_stream_chunks(
            chunk_buffer.flush(),
            on_chunk,
            on_delta,
        )
        self._invoke_stream_completion_callbacks(
            chat_response,
            on_tool_call,
            on_done,
        )

    async def _invoke_async_stream_completion_callbacks(
        self,
        chat_response: ChatResponse,
        on_tool_call: Optional[AsyncOnToolCall],
        on_done: Optional[AsyncOnDone],
    ) -> None:
        """Deliver async-capable tool and completion callbacks in order."""
        for tool_call in chat_response.tool_calls or []:
            await self._invoke_async_callback(on_tool_call, tool_call)
        await self._invoke_async_callback(on_done, chat_response)

    async def _complete_async_stream(
        self,
        chat_response: ChatResponse,
        chunk_buffer: StreamChunkBuffer,
        on_chunk: Optional[AsyncOnChunk],
        on_delta: Optional[AsyncOnDelta],
        on_tool_call: Optional[AsyncOnToolCall],
        on_done: Optional[AsyncOnDone],
    ) -> AsyncIterator[str]:
        """Flush visible text, then invoke async tool and completion callbacks."""
        async for text in self._emit_async_stream_chunks(
            chunk_buffer.flush(),
            on_chunk,
            on_delta,
        ):
            yield text
        await self._invoke_async_stream_completion_callbacks(
            chat_response,
            on_tool_call,
            on_done,
        )

    async def _record_async_reasoning_event(
        self,
        chat_response: ChatResponse,
        collector: StreamReasoningCollector,
        text: str,
        *,
        capture_reasoning: bool,
        kind: ReasoningEventKind = "summary",
        on_reasoning: Optional[AsyncOnReasoning] = None,
    ) -> Optional[ReasoningEvent]:
        """Append a reasoning event and await its optional callback."""
        if not capture_reasoning:
            return None

        event = collector.add(text, kind=kind)
        if event is None:
            return None

        chat_response.reasoning_events.append(event)
        await self._invoke_async_callback(on_reasoning, event)
        return event

    def handle_error(self, error: Exception, error_message: Optional[str] = None):
        err_msg = (
            f"Error with the provider '{self.company}' "
            f"the model '{self.model}': {error_message}. "
        )
        logger.error(err_msg)
        raise

    # ---------------- LEGACY ---------------- #

    def generate_chat_answer(self, **kwargs) -> ChatResponse:
        """Deprecated: use .chat() instead."""
        warnings.warn(
            "'generate_chat_answer' is deprecated, use 'chat' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.chat(**kwargs)
