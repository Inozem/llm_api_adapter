from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from typing import Any, Dict, Iterator, List, Mapping, Optional
import warnings

from ..adapters.base_adapter import LLMAdapterBase, OnDelta, OnDone, OnToolCall
from ..errors.llm_api_error import LLMAPIError
from ..llms.openai.sync_client import OpenAISyncClient
from ..models.messages.chat_message import Message, Messages
from ..models.responses.chat_response import ChatResponse
from ..models.tools import ToolSpec

logger = logging.getLogger(__name__)


@dataclass(repr=False)
class OpenAIAdapter(LLMAdapterBase):
    company: str = "openai"

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
    ) -> ChatResponse:
        temperature = self._validate_parameter(
            name="temperature",
            value=temperature,
            min_value=0,
            max_value=2,
        )
        top_p = self._validate_parameter(
            name="top_p",
            value=top_p,
            min_value=0,
            max_value=1,
        )
        self._validate_tools(tools)
        effective_schema = self._resolve_json_schema(json_schema, response_model, tools)
        normalized_tool_choice = self._normalize_tool_choice(tool_choice, tools)

        try:
            client = OpenAISyncClient(api_key=self.api_key)
            normalized_messages = self._normalize_messages(messages)
            use_responses_api = client._should_use_responses_api(self.model)
            normalized_reasoning_level = self._normalize_reasoning_level(
                reasoning_level
            )

            previous_response_id: Optional[str] = None
            if previous_response is not None:
                previous_response_id = previous_response.response_id

            if use_responses_api:
                transformed_messages = normalized_messages.to_openai_responses_input()
                instructions = normalized_messages.to_openai_responses_instructions()
                openai_tools = self._map_tools_to_openai_responses(tools)
                openai_tool_choice = self._map_tool_choice_to_openai_responses(
                    normalized_tool_choice
                )
            else:
                transformed_messages = normalized_messages.to_openai()
                instructions = None
                openai_tools = self._map_tools_to_openai(tools)
                openai_tool_choice = self._map_tool_choice_to_openai(
                    normalized_tool_choice
                )

            params: Dict[str, Any] = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "reasoning_effort": normalized_reasoning_level,
                "tools": openai_tools,
                "tool_choice": openai_tool_choice,
            }

            if use_responses_api:
                params["input"] = transformed_messages
                if instructions is not None:
                    params["instructions"] = instructions
                if previous_response_id is not None:
                    params["previous_response_id"] = previous_response_id
                if effective_schema is not None:
                    params["text"] = {
                        "format": {
                            "type": "json_schema",
                            "name": "response",
                            "strict": True,
                            "schema": self._enforce_strict_schema(effective_schema),
                        }
                    }
            else:
                params["messages"] = transformed_messages
                params["parallel_tool_calls"] = parallel_tool_calls
                if effective_schema is not None:
                    params["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "response",
                            "strict": True,
                            "schema": self._enforce_strict_schema(effective_schema),
                        },
                    }

            params = {k: v for k, v in params.items() if v is not None}
            response = client.complete(timeout=timeout_s, **params)

            if use_responses_api:
                chat_response = ChatResponse.from_openai_responses_response(response)
            else:
                chat_response = ChatResponse.from_openai_response(response)

            chat_response.parsed_json = self._parse_json_response(chat_response.content, effective_schema)
            chat_response.parsed_model = self._parse_response_model(chat_response.parsed_json, response_model)

            if self.pricing:
                chat_response.apply_pricing(
                    price_input_per_token=self.pricing.in_per_token,
                    price_output_per_token=self.pricing.out_per_token,
                    currency=self.pricing.currency,
                )

            return chat_response

        except LLMAPIError as e:
            self.handle_error(e)
        except Exception as e:
            error_message = getattr(e, "text", None) or str(e)
            self.handle_error(error=e, error_message=error_message)

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
    ) -> Iterator[str]:
        temperature = self._validate_parameter("temperature", temperature, 0, 2)
        top_p = self._validate_parameter("top_p", top_p, 0, 1)
        self._validate_tools(tools)
        effective_schema = self._resolve_json_schema(json_schema, response_model, tools)
        normalized_tool_choice = self._normalize_tool_choice(tool_choice, tools)
        client = OpenAISyncClient(api_key=self.api_key)
        normalized_messages = self._normalize_messages(messages)
        use_responses_api = client._should_use_responses_api(self.model)
        params = self._build_stream_params(
            normalized_messages=normalized_messages,
            use_responses_api=use_responses_api,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            reasoning_level=reasoning_level,
            tools=tools,
            normalized_tool_choice=normalized_tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            previous_response=previous_response,
            effective_schema=effective_schema,
        )
        events = client.stream(timeout=timeout_s, **params)
        if use_responses_api:
            yield from self._consume_responses_stream(
                events,
                effective_schema,
                response_model,
                on_delta,
                on_tool_call,
                on_done,
            )
            return
        yield from self._consume_chat_completions_stream(
            events,
            effective_schema,
            response_model,
            on_delta,
            on_tool_call,
            on_done,
        )

    def _build_stream_params(
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
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "reasoning_effort": self._normalize_reasoning_level(reasoning_level),
        }
        if use_responses_api:
            params["input"] = normalized_messages.to_openai_responses_input()
            params["tools"] = self._map_tools_to_openai_responses(tools)
            params["tool_choice"] = self._map_tool_choice_to_openai_responses(
                normalized_tool_choice
            )
            instructions = normalized_messages.to_openai_responses_instructions()
            if instructions is not None:
                params["instructions"] = instructions
            if previous_response is not None and previous_response.response_id is not None:
                params["previous_response_id"] = previous_response.response_id
            if effective_schema is not None:
                params["text"] = {
                    "format": {
                        "type": "json_schema",
                        "name": "response",
                        "strict": True,
                        "schema": self._enforce_strict_schema(effective_schema),
                    }
                }
        else:
            params["messages"] = normalized_messages.to_openai()
            params["tools"] = self._map_tools_to_openai(tools)
            params["tool_choice"] = self._map_tool_choice_to_openai(
                normalized_tool_choice
            )
            params["parallel_tool_calls"] = parallel_tool_calls
            if effective_schema is not None:
                params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "strict": True,
                        "schema": self._enforce_strict_schema(effective_schema),
                    },
                }
        return {key: value for key, value in params.items() if value is not None}

    def _consume_responses_stream(
        self,
        events: Iterator[Any],
        effective_schema: Optional[dict],
        response_model: Optional[Any],
        on_delta: Optional[OnDelta],
        on_tool_call: Optional[OnToolCall],
        on_done: Optional[OnDone],
    ) -> Iterator[str]:
        final_response: Optional[dict] = None
        response_metadata: Dict[str, Any] = {}
        text_parts: List[str] = []
        function_calls: Dict[str, Dict[str, Any]] = {}
        for event in self._iter_provider_stream_events(events):
            payload = event.data if isinstance(event.data, Mapping) else {}
            event_type = event.event or payload.get("type")
            response_data = payload.get("response")
            if isinstance(response_data, Mapping):
                response_metadata.update(response_data)
            if event_type == "response.output_text.delta":
                delta = payload.get("delta")
                if isinstance(delta, str) and delta:
                    text_parts.append(delta)
                    if on_delta is not None:
                        on_delta(delta)
                    yield delta
            elif event_type in (
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
            ):
                call_key = str(
                    payload.get("call_id")
                    or payload.get("item_id")
                    or payload.get("output_index")
                    or len(function_calls)
                )
                call = function_calls.setdefault(
                    call_key,
                    {"type": "function_call", "arguments": ""},
                )
                call["call_id"] = payload.get("call_id") or call.get("call_id")
                call["id"] = payload.get("item_id") or call.get("id")
                call["name"] = payload.get("name") or call.get("name")
                if event_type.endswith(".delta") and isinstance(payload.get("delta"), str):
                    call["arguments"] = f"{call.get('arguments', '')}{payload['delta']}"
                elif "arguments" in payload:
                    call["arguments"] = payload["arguments"]
            elif event_type == "response.completed" and isinstance(response_data, Mapping):
                final_response = dict(response_data)
        if final_response is None:
            output: List[Dict[str, Any]] = []
            if text_parts:
                output.append({
                    "type": "message",
                    "content": [{"type": "output_text", "text": "".join(text_parts)}],
                })
            output.extend(function_calls.values())
            final_response = {
                **response_metadata,
                "model": response_metadata.get("model", self.model),
                "output": output,
                "status": response_metadata.get("status", "completed"),
            }
        self._finalize_stream_response(
            ChatResponse.from_openai_responses_response(final_response),
            effective_schema,
            response_model,
            on_tool_call,
            on_done,
        )

    def _consume_chat_completions_stream(
        self,
        events: Iterator[Any],
        effective_schema: Optional[dict],
        response_model: Optional[Any],
        on_delta: Optional[OnDelta],
        on_tool_call: Optional[OnToolCall],
        on_done: Optional[OnDone],
    ) -> Iterator[str]:
        text_parts: List[str] = []
        tool_calls: Dict[int, Dict[str, Any]] = {}
        legacy_response: Dict[str, Any] = {"model": self.model, "choices": []}
        finish_reason: Optional[str] = None
        for event in self._iter_provider_stream_events(events):
            payload = event.data if isinstance(event.data, Mapping) else {}
            for field in ("id", "model", "created", "usage"):
                if field in payload:
                    legacy_response[field] = payload[field]
            choices = payload.get("choices")
            if not isinstance(choices, list):
                continue
            for choice in choices:
                if not isinstance(choice, Mapping) or choice.get("index", 0) != 0:
                    continue
                delta = choice.get("delta") or {}
                if not isinstance(delta, Mapping):
                    delta = {}
                content = delta.get("content")
                if isinstance(content, str) and content:
                    text_parts.append(content)
                    if on_delta is not None:
                        on_delta(content)
                    yield content
                raw_tool_calls = delta.get("tool_calls")
                if isinstance(raw_tool_calls, list):
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
                        function = raw_tool_call.get("function") or {}
                        if isinstance(function, Mapping):
                            target = tool_call["function"]
                            if function.get("name") is not None:
                                target["name"] = function["name"]
                            arguments = function.get("arguments")
                            if isinstance(arguments, str):
                                target["arguments"] = f"{target.get('arguments', '')}{arguments}"
                            elif isinstance(arguments, dict):
                                target["arguments"] = arguments
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
        message: Dict[str, Any] = {"content": "".join(text_parts) or None}
        if tool_calls:
            message["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]
        legacy_response["choices"] = [{"message": message, "finish_reason": finish_reason}]
        self._finalize_stream_response(
            ChatResponse.from_openai_response(legacy_response),
            effective_schema,
            response_model,
            on_tool_call,
            on_done,
        )

    def _map_tools_to_openai(
        self,
        tools: Optional[List[ToolSpec]],
    ) -> Optional[List[Dict[str, Any]]]:
        if not tools:
            return None
        mapped: List[Dict[str, Any]] = []
        for tool in tools:
            function_payload: Dict[str, Any] = {
                "name": tool.name,
                "parameters": tool.json_schema,
            }
            if tool.description:
                function_payload["description"] = tool.description
            mapped.append(
                {
                    "type": "function",
                    "function": function_payload,
                }
            )
        return mapped

    def _map_tool_choice_to_openai(self, tool_choice: Optional[str]) -> Any:
        if tool_choice is None:
            return None
        if tool_choice in ("auto", "none"):
            return tool_choice
        if tool_choice == "any":
            return "required"
        return {"type": "function", "function": {"name": tool_choice}}

    def _normalize_reasoning_level(self, level: str | int | None) -> str | None:
        if level is None:
            return "none" if self.is_reasoning else None
        if not self.is_reasoning and level not in ("none", 0):
            warning_message = (
                f"Model '{self.model}' does not support reasoning — reasoning disabled."
            )
            warnings.warn(warning_message, UserWarning)
            logger.info(warning_message)
            return None
        if isinstance(level, bool):
            raise ValueError("Invalid type for level: bool is not accepted")
        if isinstance(level, str):
            if level in self.reasoning_levels:
                return level
            raise ValueError(
                f"Unknown reasoning level key: {level!r}. "
                f"Valid keys: {list(self.reasoning_levels.keys())}"
            )
        if isinstance(level, int):
            for key, val in self.reasoning_levels.items():
                if level <= val:
                    return key
            return list(self.reasoning_levels.keys())[-1]
        raise ValueError(
            "Invalid type for level: expected int or str, "
            f"got {type(level).__name__!r}"
        )

    def _map_tools_to_openai_responses(
        self,
        tools: Optional[List[ToolSpec]],
    ) -> Optional[List[Dict[str, Any]]]:
        if not tools:
            return None

        result: List[Dict[str, Any]] = []
        for tool in tools:
            result.append(
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.json_schema,
                }
            )
        return result

    def _map_tool_choice_to_openai_responses(self, tool_choice: Any) -> Any:
        if tool_choice is None:
            return None
        if tool_choice == "auto":
            return "auto"
        if tool_choice == "none":
            return "none"
        if tool_choice == "any":
            return "required"
        if isinstance(tool_choice, str):
            return {
                "type": "function",
                "name": tool_choice,
            }
        return tool_choice
