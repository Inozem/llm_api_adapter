"""Synchronous adapter for Mistral's official Chat Completions API."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, List, Mapping, Optional
import warnings

from llm_api_adapter.adapters.base_adapter import LLMAdapterBase
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
    SyncTransport,
    TransportRequest,
    create_sync_transport,
)
from llm_api_adapter.models.messages.chat_message import Message, Messages
from llm_api_adapter.models.responses.chat_response import ChatResponse, Usage
from llm_api_adapter.models.responses.reasoning_event import ReasoningEvent
from llm_api_adapter.models.tools import ToolCall, ToolSpec


_MISTRAL_CHAT_COMPLETIONS_URL = "https://api.mistral.ai/v1/chat/completions"


@dataclass(repr=False)
class MistralAdapter(LLMAdapterBase):
    """Call Mistral's direct Chat Completions endpoint through a core transport."""

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
        if previous_response is not None:
            raise NotImplementedError(
                "Mistral Chat Completions has no previous_response parameter; "
                "include the conversation history in messages instead."
            )

        temperature = self._validate_parameter("temperature", temperature, 0, 1.5)
        top_p = self._validate_parameter("top_p", top_p, 0, 1)
        request_context = self._prepare_chat_request(
            messages,
            tools,
            tool_choice,
            json_schema,
            response_model,
        )
        payload = self._build_payload(
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

    def stream_chat(self, *args: Any, **kwargs: Any):
        """Streaming is delivered in the next provider-package change."""
        raise NotImplementedError("Mistral streaming is not available yet")

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
            "messages": messages.to_openai(),
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
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                payload=payload,
                timeout=timeout_s,
            ),
            http_error_handler=self._handle_http_error,
        )
        response_data = response.json()
        if not isinstance(response_data, dict):
            raise LLMAPIClientError(detail="Mistral returned a non-object response")
        return response_data

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
