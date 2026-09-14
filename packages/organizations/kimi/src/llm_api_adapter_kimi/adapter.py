"""Synchronous adapter for Kimi's official Chat Completions API."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Optional

from llm_api_adapter.adapters.base_adapter import LLMAdapterBase
from llm_api_adapter.errors.llm_api_error import LLMAPIClientError, LLMAPIError
from llm_api_adapter.llm_registry.request_rules import apply_request_rules
from llm_api_adapter.llms.transports import SyncTransport, create_sync_transport
from llm_api_adapter.models.messages.chat_message import Message, Messages, UserMessage
from llm_api_adapter.models.responses.chat_response import ChatResponse, Usage
from llm_api_adapter.models.tools.tool_spec import ToolSpec

from .clients import KimiSyncClient
from .registry import CACHE_PRICING, KimiCachePricing


@dataclass
class KimiUsage(Usage):
    """Normalized Kimi usage retaining the cache-hit split used for pricing."""

    cached_tokens: int | None = None


@dataclass(repr=False)
class KimiAdapter(LLMAdapterBase):
    """Map the shared text-chat contract to Kimi Chat Completions."""

    company: str = "kimi"
    _client: KimiSyncClient = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._client = KimiSyncClient(create_sync_transport(self.transport))

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
        capture_reasoning: bool = False,
    ) -> ChatResponse:
        """Create one Kimi response through ``POST /v1/chat/completions``."""
        _ = previous_response
        _ = capture_reasoning
        payload = self._prepare_request_payload(
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
        )
        try:
            response = self._client.chat(
                api_key=self.api_key,
                payload=payload,
                timeout_s=timeout_s,
            )
            return self._finalize_kimi_chat_response(response)
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
        on_delta: Any = None,
        on_tool_call: Any = None,
        on_done: Any = None,
        buffer_chars: Optional[int] = None,
        on_chunk: Any = None,
        *,
        capture_reasoning: bool = False,
        on_reasoning: Any = None,
    ) -> Iterator[str]:
        """Streaming is added with Kimi's SSE implementation in the next commit."""
        _ = (
            messages,
            max_tokens,
            temperature,
            top_p,
            reasoning_level,
            timeout_s,
            tools,
            tool_choice,
            parallel_tool_calls,
            previous_response,
            json_schema,
            response_model,
            on_delta,
            on_tool_call,
            on_done,
            buffer_chars,
            on_chunk,
            capture_reasoning,
            on_reasoning,
        )
        raise NotImplementedError("Kimi streaming is not implemented yet")
        yield ""

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
    ) -> dict[str, Any]:
        """Validate the direct-chat slice and apply metadata request rules."""
        self._reject_deferred_features(
            reasoning_level=reasoning_level,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            json_schema=json_schema,
            response_model=response_model,
        )
        request_context = self._prepare_chat_request(
            messages,
            None,
            None,
            None,
            None,
        )
        self._reject_file_parts(request_context.normalized_messages)
        validated_max_tokens = self._validate_max_tokens(max_tokens)
        temperature, top_p = self._validate_sampling_parameters(temperature, top_p)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": request_context.normalized_messages.to_openai(),
            "temperature": temperature,
            "top_p": top_p,
        }
        if validated_max_tokens is not None:
            payload["max_tokens"] = validated_max_tokens
        if self.model_spec is None:
            return payload
        transformed_payload, _ = apply_request_rules(
            payload,
            self.model_spec.request_rules,
            model=self.model,
        )
        return transformed_payload

    @staticmethod
    def _reject_deferred_features(
        *,
        reasoning_level: Optional[str | int],
        tools: Optional[list[ToolSpec]],
        tool_choice: Any,
        parallel_tool_calls: Optional[bool],
        json_schema: Optional[dict],
        response_model: Optional[Any],
    ) -> None:
        if reasoning_level is not None:
            raise NotImplementedError("Kimi reasoning controls are not implemented yet")
        if tools is not None or tool_choice is not None or parallel_tool_calls is not None:
            raise NotImplementedError("Kimi application tools are not implemented yet")
        if json_schema is not None or response_model is not None:
            raise NotImplementedError("Kimi structured output is not implemented yet")

    @staticmethod
    def _reject_file_parts(messages: Messages) -> None:
        if any(
            isinstance(message, UserMessage) and message.files
            for message in messages.items
        ):
            raise NotImplementedError("Kimi image and document inputs are not implemented yet")

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

    @staticmethod
    def _parse_response(response: Mapping[str, Any]) -> ChatResponse:
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
        content = choice["message"].get("content")
        if content is not None and not isinstance(content, str):
            raise LLMAPIClientError(
                detail="Kimi Chat Completions response content must be a string or null",
            )

        chat_response = ChatResponse.from_openai_response(dict(response))
        usage = response.get("usage")
        if usage is None:
            return chat_response
        if not isinstance(usage, Mapping):
            raise LLMAPIClientError(
                detail="Kimi Chat Completions response.usage must be an object",
            )
        required_tokens = ("prompt_tokens", "completion_tokens", "total_tokens")
        if any(
            isinstance(usage.get(field), bool) or not isinstance(usage.get(field), int)
            for field in required_tokens
        ):
            raise LLMAPIClientError(
                detail="Kimi Chat Completions usage token counts must be integers",
            )
        cached_tokens = usage.get("cached_tokens")
        if cached_tokens is not None and (
            isinstance(cached_tokens, bool) or not isinstance(cached_tokens, int)
        ):
            raise LLMAPIClientError(
                detail="Kimi Chat Completions usage.cached_tokens must be an integer",
            )
        if cached_tokens is not None and not 0 <= cached_tokens <= usage["prompt_tokens"]:
            raise LLMAPIClientError(
                detail="Kimi Chat Completions usage.cached_tokens must not exceed prompt_tokens",
            )
        chat_response.usage = KimiUsage(
            input_tokens=usage["prompt_tokens"],
            output_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"],
            cached_tokens=cached_tokens,
        )
        return chat_response

    def _finalize_kimi_chat_response(self, response: Mapping[str, Any]) -> ChatResponse:
        chat_response = self._finalize_chat_response(
            self._parse_response(response),
            effective_schema=None,
            response_model=None,
        )
        self._apply_cache_aware_pricing(chat_response)
        return chat_response

    def _apply_cache_aware_pricing(self, chat_response: ChatResponse) -> None:
        """Replace the generic cache-miss estimate only when the split is reported."""
        cache_pricing: KimiCachePricing | None = CACHE_PRICING.get(self.model)
        usage = chat_response.usage
        if cache_pricing is None or not isinstance(usage, KimiUsage):
            return
        if usage.cached_tokens is None:
            chat_response.cost_input = None
            chat_response.cost_total = None
            return
        uncached_tokens = usage.input_tokens - usage.cached_tokens
        chat_response.currency = "USD"
        chat_response.cost_input = (
            usage.cached_tokens * cache_pricing.cache_hit_input_per_token
            + uncached_tokens * cache_pricing.cache_miss_input_per_token
        )
        chat_response.cost_output = usage.output_tokens * cache_pricing.output_per_token
        chat_response.cost_total = chat_response.cost_input + chat_response.cost_output


__all__ = ["KimiAdapter", "KimiUsage"]
