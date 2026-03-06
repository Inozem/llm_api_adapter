from dataclasses import dataclass
import json
from typing import List, Optional
import warnings

from ...errors.llm_api_error import InvalidToolArgumentsError, LLMAPIError
from ...models.tools import ToolCall


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ChatResponse:
    model: Optional[str] = None
    response_id: Optional[str] = None
    timestamp: Optional[int] = None
    usage: Optional[Usage] = None
    currency: Optional[str] = None
    cost_input: Optional[float] = None
    cost_output: Optional[float] = None
    cost_total: Optional[float] = None
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    finish_reason: Optional[str] = None

    @classmethod
    def from_openai_response(cls, api_response: dict) -> "ChatResponse":
        u = api_response.get("usage", {}) or {}
        usage = Usage(
            input_tokens=u.get("prompt_tokens", 0),
            output_tokens=u.get("completion_tokens", 0),
            total_tokens=u.get("total_tokens", 0),
        )
        choice0 = (api_response.get("choices") or [None])[0] or {}
        message = choice0.get("message") or {}
        text = message.get("content")
        parsed_tool_calls: Optional[List[ToolCall]] = None
        raw_tool_calls = message.get("tool_calls")
        if isinstance(raw_tool_calls, list) and raw_tool_calls:
            parsed_tool_calls = []
            for tc in raw_tool_calls:
                tc = tc or {}
                fn = tc.get("function") or {}
                name = fn.get("name")
                raw_args = fn.get("arguments", "{}")
                try:
                    if isinstance(raw_args, str):
                        arguments = json.loads(raw_args) if raw_args.strip() else {}
                    elif isinstance(raw_args, dict):
                        arguments = raw_args
                    else:
                        arguments = {}
                except Exception as e:
                    raise InvalidToolArgumentsError(
                        detail=f"OpenAI tool arguments JSON parse failed for tool={name!r}: {e}"
                    )
                parsed_tool_calls.append(
                    ToolCall(
                        name=name,
                        arguments=arguments,
                        call_id=tc.get("id"),
                    )
                )

        # Legacy function_call (older format)
        legacy_fc = message.get("function_call")
        if (not parsed_tool_calls) and isinstance(legacy_fc, dict) and legacy_fc:
            name = legacy_fc.get("name")
            raw_args = legacy_fc.get("arguments", "{}")
            try:
                if isinstance(raw_args, str):
                    arguments = json.loads(raw_args) if raw_args.strip() else {}
                elif isinstance(raw_args, dict):
                    arguments = raw_args
                else:
                    arguments = {}
            except Exception as e:
                raise InvalidToolArgumentsError(
                    detail=f"OpenAI function_call arguments JSON parse failed for tool={name!r}: {e}"
                )
            parsed_tool_calls = [ToolCall(name=name, arguments=arguments, call_id=None)]

        if not parsed_tool_calls:
            if not text or not str(text).strip():
                warnings.warn(
                    "OpenAI returned empty content. "
                    "The model may have stopped early due to a low max_tokens value.",
                    UserWarning,
                )

        return cls(
            model=api_response.get("model"),
            response_id=api_response.get("id"),
            timestamp=api_response.get("created"),
            usage=usage,
            content=text,
            tool_calls=parsed_tool_calls,
            finish_reason=choice0.get("finish_reason"),
        )

    @classmethod
    def from_anthropic_response(cls, api_response: dict) -> "ChatResponse":
        u = api_response.get("usage", {}) or {}
        usage = Usage(
            input_tokens=u.get("input_tokens", 0),
            output_tokens=u.get("output_tokens", 0),
            total_tokens=u.get("input_tokens", 0) + u.get("output_tokens", 0),
        )
        blocks = api_response.get("content", []) or []
        parsed_tool_calls: Optional[List[ToolCall]] = None
        text_content: Optional[str] = None
        for block in blocks:
            block_type = block.get("type")
            if block_type == "text" and text_content is None:
                text_content = block.get("text")
            elif block_type == "tool_use":
                if parsed_tool_calls is None:
                    parsed_tool_calls = []
                name = block.get("name")
                arguments = block.get("input")
                if not isinstance(arguments, dict):
                    raise InvalidToolArgumentsError(
                        detail=f"Anthropic tool input must be dict for tool={name!r}"
                    )
                parsed_tool_calls.append(
                    ToolCall(
                        name=name,
                        arguments=arguments,
                        call_id=block.get("id"),
                    )
                )
        return cls(
            model=api_response.get("model"),
            response_id=api_response.get("id"),
            usage=usage,
            content=text_content,
            tool_calls=parsed_tool_calls,
            finish_reason=api_response.get("stop_reason"),
        )

    @classmethod
    def from_google_response(cls, api_response: dict) -> "ChatResponse":
        u = api_response.get("usageMetadata", {}) or {}
        thoughts_tokens = u.get("thoughtsTokenCount", 0)
        usage = Usage(
            input_tokens=u.get("promptTokenCount", 0),
            output_tokens=u.get("candidatesTokenCount", 0) + thoughts_tokens,
            total_tokens=u.get("totalTokenCount", 0),
        )
        first_candidate = (api_response.get("candidates") or [None])[0] or {}
        finish_reason = first_candidate.get("finishReason")
        finish_reason_str = str(finish_reason) if finish_reason is not None else None
        content_obj = first_candidate.get("content") or {}
        parts = content_obj.get("parts") or []
        if not isinstance(parts, list):
            raise LLMAPIError(
                "Google API returned malformed response",
                detail="content.parts is not a list",
            )
        parsed_tool_calls: Optional[List[ToolCall]] = None
        text_content: Optional[str] = None
        for part in parts:
            if not isinstance(part, dict):
                continue
            if text_content is None and "text" in part:
                text_content = part.get("text")
            fc = part.get("functionCall") or part.get("function_call")
            if isinstance(fc, dict) and fc:
                if parsed_tool_calls is None:
                    parsed_tool_calls = []
                name = fc.get("name")
                if not isinstance(name, str) or not name:
                    raise InvalidToolArgumentsError(
                        detail="Google functionCall.name must be non-empty str"
                    )
                args = fc.get("args") if "args" in fc else fc.get("arguments", {})
                if args is None:
                    args = {}
                if not isinstance(args, dict):
                    raise InvalidToolArgumentsError(
                        detail=f"Google functionCall args must be dict for tool={name!r}"
                    )
                parsed_tool_calls.append(
                    ToolCall(
                        name=name,
                        arguments=args,
                        call_id=name,
                    )
                )
        return cls(
            usage=usage,
            content=text_content,
            tool_calls=parsed_tool_calls,
            finish_reason=finish_reason_str,
        )

    def apply_pricing(
        self,
        price_input_per_token: float,
        price_output_per_token: float,
        currency: str = "USD"
    ):
        if not self.usage:
            return
        self.currency = currency
        self.cost_input = self.usage.input_tokens * price_input_per_token
        self.cost_output = self.usage.output_tokens * price_output_per_token
        self.cost_total = self.cost_input + self.cost_output
