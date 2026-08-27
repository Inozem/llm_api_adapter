"""Adapter for xAI's official synchronous Responses API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, List, Optional

from llm_api_adapter.adapters.base_adapter import LLMAdapterBase
from llm_api_adapter.errors.llm_api_error import LLMAPIClientError, LLMAPIError
from llm_api_adapter.models.messages.chat_message import (
    AIMessage,
    Message,
    Messages,
    ToolMessage,
    UserMessage,
)
from llm_api_adapter.models.responses.chat_response import ChatResponse
from llm_api_adapter.models.tools import ToolSpec

from .clients import XAIResponsesSyncClient


@dataclass(repr=False)
class XAIAdapter(LLMAdapterBase):
    """Map the shared text-chat contract to xAI's Responses API."""

    company: str = "xai"
    endpoint: str = "https://api.x.ai/v1/responses"
    _client: XAIResponsesSyncClient = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._client = XAIResponsesSyncClient(
            api_key=self.api_key,
            transport=self.transport,
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
        self._reject_unsupported_options(
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            previous_response=previous_response,
            json_schema=json_schema,
            response_model=response_model,
            reasoning_level=reasoning_level,
            capture_reasoning=capture_reasoning,
        )
        temperature, top_p = self._validate_sampling_parameters(temperature, top_p)
        request_context = self._prepare_chat_request(
            messages,
            None,
            None,
            None,
            None,
        )
        self._reject_file_and_tool_messages(request_context.normalized_messages)

        parameters: dict[str, Any] = {
            "input": request_context.normalized_messages.to_openai_responses_input(),
            "max_output_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        instructions = request_context.normalized_messages.to_openai_responses_instructions()
        if instructions is not None:
            parameters["instructions"] = instructions

        try:
            response = self._client.create(
                model=self.model,
                timeout=timeout_s,
                **{key: value for key, value in parameters.items() if value is not None},
            )
            chat_response = self._parse_response(response)
            return self._finalize_chat_response(
                chat_response,
                effective_schema=None,
                response_model=None,
            )
        except LLMAPIError as error:
            self.handle_error(error)
        except Exception as error:
            error_message = getattr(error, "text", None) or str(error)
            self.handle_error(error=error, error_message=error_message)

    def stream_chat(self, *args: Any, **kwargs: Any) -> Iterator[str]:
        """Reserve Responses SSE handling for the streaming implementation step."""
        raise NotImplementedError("xAI streaming is not available yet")

    @staticmethod
    def _reject_unsupported_options(
        *,
        tools: Optional[List[ToolSpec]],
        tool_choice: Any,
        parallel_tool_calls: Optional[bool],
        previous_response: Optional[ChatResponse],
        json_schema: Optional[dict],
        response_model: Optional[Any],
        reasoning_level: Optional[str | int],
        capture_reasoning: bool,
    ) -> None:
        unsupported = {
            "tools": tools is not None,
            "tool_choice": tool_choice is not None,
            "parallel_tool_calls": parallel_tool_calls is not None,
            "previous_response": previous_response is not None,
            "json_schema": json_schema is not None,
            "response_model": response_model is not None,
            "reasoning_level": reasoning_level is not None,
            "capture_reasoning": capture_reasoning,
        }
        selected = [name for name, present in unsupported.items() if present]
        if selected:
            raise ValueError(
                "xAI text chat does not support these options yet: "
                + ", ".join(selected),
            )

    @staticmethod
    def _reject_file_and_tool_messages(messages: Messages) -> None:
        for message in messages.items:
            if isinstance(message, UserMessage) and message.files:
                raise ValueError("xAI file input is not available yet")
            if isinstance(message, ToolMessage) or (
                isinstance(message, AIMessage) and message.tool_calls
            ):
                raise ValueError("xAI tool-result messages are not available yet")

    @staticmethod
    def _parse_response(response: dict[str, Any]) -> ChatResponse:
        if response.get("object") != "response":
            raise LLMAPIClientError(
                detail="xAI Responses API returned an invalid response object",
            )
        if not isinstance(response.get("output"), list):
            raise LLMAPIClientError(
                detail="xAI Responses API response.output must be an array",
            )
        return ChatResponse.from_openai_responses_response(response)


__all__ = ["XAIAdapter"]
