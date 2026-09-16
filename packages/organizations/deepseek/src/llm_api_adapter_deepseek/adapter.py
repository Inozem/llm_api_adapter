"""Core facade adapter for DeepSeek's official Responses API."""

from __future__ import annotations

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
    LLMAPIClientError,
    LLMAPIError,
    ToolChoiceError,
)
from llm_api_adapter.models.messages.chat_message import Message, Messages, Prompt
from llm_api_adapter.models.responses.chat_response import ChatResponse
from llm_api_adapter.models.tools import ToolSpec

from .clients.sync_client import (
    DEEPSEEK_RESPONSES_URL,
    DeepSeekResponsesSyncClient,
)
from .clients.async_client import DeepSeekResponsesAsyncClient


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
        try:
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
        try:
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
        """Streaming support is completed in T013."""
        del (
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
        raise NotImplementedError("DeepSeek streaming support is implemented in T013")

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
        """Asynchronous streaming support is completed in T013."""
        del (
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
        raise NotImplementedError("DeepSeek streaming support is implemented in T013")
        yield ""

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
        del previous_response, parallel_tool_calls
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
            "input": normalized_messages.to_openai_responses_input(),
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
        if request_context.effective_schema is not None:
            parameters["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "response",
                    "schema": request_context.effective_schema,
                },
            }
        return _PreparedResponsesRequest(
            parameters={
                key: value for key, value in parameters.items() if value is not None
            },
            normalized_messages=normalized_messages,
            effective_schema=request_context.effective_schema,
            response_model=request_context.response_model,
            capture_reasoning=capture_reasoning,
        )

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
        return ChatResponse.from_openai_responses_response(
            response,
            capture_reasoning=capture_reasoning,
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
        return self._finalize_chat_response(
            self._parse_response(
                response,
                capture_reasoning=capture_reasoning,
            ),
            effective_schema=effective_schema,
            response_model=response_model,
        )


__all__ = ["DeepSeekAdapter"]
