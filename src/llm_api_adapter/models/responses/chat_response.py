from dataclasses import dataclass, field
import json
from math import isfinite
from typing import Any, List, Optional, Sequence
import warnings

from ...errors.llm_api_error import InvalidToolArgumentsError, LLMAPIError
from ...models.tools import ToolCall
from .reasoning_event import ReasoningEvent


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class CostLineItem:
    """One provider-neutral, non-token cost component."""

    operation: str
    model: str
    unit: str
    quantity: float
    rate: float
    currency: Optional[str]
    cost: float = field(init=False)

    def __post_init__(self) -> None:
        for field_name, value in (
            ("operation", self.operation),
            ("model", self.model),
            ("unit", self.unit),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"cost line item {field_name} must be a non-empty string")
        for field_name, value in (
            ("quantity", self.quantity),
            ("rate", self.rate),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or value < 0
            ):
                raise ValueError(
                    f"cost line item {field_name} must be a non-negative finite number"
                )
        if self.currency is not None and (
            not isinstance(self.currency, str) or not self.currency
        ):
            raise ValueError("cost line item currency must be a non-empty string or None")
        object.__setattr__(self, "quantity", float(self.quantity))
        object.__setattr__(self, "rate", float(self.rate))
        object.__setattr__(self, "cost", self.quantity * self.rate)


@dataclass
class _ParsedResponsesOutput:
    text_parts: List[str] = field(default_factory=list)
    tool_calls: Optional[List[ToolCall]] = None
    refusal: Optional[str] = None
    reasoning_events: List[ReasoningEvent] = field(default_factory=list)


@dataclass
class _ParsedAnthropicContent:
    text: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    reasoning_events: List[ReasoningEvent] = field(default_factory=list)


@dataclass
class _ParsedGoogleContent:
    text: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    finish_reason: Optional[str] = None
    refusal: Optional[str] = None
    incomplete_reason: Optional[str] = None
    reasoning_events: List[ReasoningEvent] = field(default_factory=list)


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
    cost_breakdown: Optional[List[CostLineItem]] = None
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    finish_reason: Optional[str] = None
    refusal: Optional[str] = None
    incomplete_reason: Optional[str] = None
    parsed_json: Optional[dict] = None
    parsed_model: Optional[Any] = None
    reasoning_events: List[ReasoningEvent] = field(default_factory=list)

    @classmethod
    def from_openai_response(cls, api_response: dict) -> "ChatResponse":
        usage_data = api_response.get("usage")
        usage = (
            Usage(
                input_tokens=usage_data.get("prompt_tokens", 0),
                output_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            )
            if isinstance(usage_data, dict)
            else None
        )
        choice0 = (api_response.get("choices") or [None])[0] or {}
        message = choice0.get("message") or {}
        text = message.get("content")
        finish_reason = cls._optional_string(choice0.get("finish_reason"))
        refusal = cls._optional_string(message.get("refusal"))
        if refusal is None and finish_reason == "content_filter":
            refusal = finish_reason
        incomplete_reason = finish_reason if finish_reason == "length" else None
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

        if not parsed_tool_calls and refusal is None:
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
            finish_reason=finish_reason,
            refusal=refusal,
            incomplete_reason=incomplete_reason,
        )

    @classmethod
    def from_openai_responses_response(
        cls,
        api_response: dict,
        *,
        capture_reasoning: bool = False,
    ) -> "ChatResponse":
        usage_data = api_response.get("usage")
        usage = (
            Usage(
                input_tokens=usage_data.get("input_tokens", 0),
                output_tokens=usage_data.get("output_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            )
            if isinstance(usage_data, dict)
            else None
        )
        parsed_output = cls._parse_responses_output_items(
            api_response.get("output") or [],
            capture_reasoning=capture_reasoning,
        )
        text = "\n".join(parsed_output.text_parts) if parsed_output.text_parts else None
        incomplete_reason = cls._responses_incomplete_reason(api_response)
        if (
            not parsed_output.tool_calls
            and not parsed_output.reasoning_events
            and parsed_output.refusal is None
            and incomplete_reason is None
            and (not text or not text.strip())
        ):
            warnings.warn(
                "OpenAI Responses API returned empty content and no tool calls.",
                UserWarning,
            )
        return cls(
            model=api_response.get("model"),
            response_id=api_response.get("id"),
            timestamp=api_response.get("created_at"),
            usage=usage,
            content=text,
            tool_calls=parsed_output.tool_calls,
            finish_reason=api_response.get("status"),
            refusal=parsed_output.refusal,
            incomplete_reason=incomplete_reason,
            reasoning_events=parsed_output.reasoning_events,
        )

    @classmethod
    def _parse_responses_output_items(
        cls,
        output_items: Any,
        *,
        capture_reasoning: bool,
    ) -> _ParsedResponsesOutput:
        parsed_output = _ParsedResponsesOutput()
        if not isinstance(output_items, list):
            return parsed_output

        for item in output_items:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "message":
                cls._append_responses_message_content(item, parsed_output)
            elif item_type == "reasoning" and capture_reasoning:
                cls._append_responses_reasoning_events(
                    item,
                    parsed_output.reasoning_events,
                )
            elif item_type in ("function_call", "tool_call"):
                if parsed_output.tool_calls is None:
                    parsed_output.tool_calls = []
                parsed_output.tool_calls.append(cls._parse_responses_tool_call(item))
        return parsed_output

    @staticmethod
    def _append_responses_message_content(
        item: dict,
        parsed_output: _ParsedResponsesOutput,
    ) -> None:
        content_items = item.get("content") or []
        if not isinstance(content_items, list):
            return
        for content_item in content_items:
            if not isinstance(content_item, dict):
                continue
            content_type = content_item.get("type")
            if content_type in ("output_text", "text"):
                text_value = content_item.get("text")
                if isinstance(text_value, str) and text_value.strip():
                    parsed_output.text_parts.append(text_value)
            elif content_type == "refusal" and parsed_output.refusal is None:
                parsed_output.refusal = ChatResponse._optional_string(
                    content_item.get("refusal"),
                )

    @staticmethod
    def _optional_string(value: Any) -> Optional[str]:
        return value if isinstance(value, str) and value else None

    @classmethod
    def _responses_incomplete_reason(cls, api_response: dict) -> Optional[str]:
        if api_response.get("status") != "incomplete":
            return None
        details = api_response.get("incomplete_details")
        if isinstance(details, dict):
            reason = cls._optional_string(details.get("reason"))
            if reason is not None:
                return reason
        return "incomplete"

    @staticmethod
    def _append_responses_reasoning_events(
        item: dict,
        reasoning_events: List[ReasoningEvent],
    ) -> None:
        summary_items = item.get("summary") or []
        if isinstance(summary_items, list):
            for summary_item in summary_items:
                if not isinstance(summary_item, dict):
                    continue
                if summary_item.get("type") != "summary_text":
                    continue
                text_value = summary_item.get("text")
                if isinstance(text_value, str) and text_value:
                    reasoning_events.append(
                        ReasoningEvent(
                            text=text_value,
                            kind="summary",
                            index=len(reasoning_events),
                            elapsed_s=0.0,
                            delta_s=0.0,
                        )
                    )

        content_items = item.get("content") or []
        if isinstance(content_items, list):
            for content_item in content_items:
                if not isinstance(content_item, dict):
                    continue
                if content_item.get("type") != "reasoning_text":
                    continue
                text_value = content_item.get("text")
                if isinstance(text_value, str) and text_value:
                    reasoning_events.append(
                        ReasoningEvent(
                            text=text_value,
                            kind="content",
                            index=len(reasoning_events),
                            elapsed_s=0.0,
                            delta_s=0.0,
                        )
                    )

    @staticmethod
    def _parse_responses_tool_call(item: dict) -> ToolCall:
        name = item.get("name")
        raw_args = item.get("arguments", "{}")
        try:
            if isinstance(raw_args, str):
                arguments = json.loads(raw_args) if raw_args.strip() else {}
            elif isinstance(raw_args, dict):
                arguments = raw_args
            else:
                arguments = {}
        except Exception as e:
            raise InvalidToolArgumentsError(
                detail=f"OpenAI responses tool arguments JSON parse failed for tool={name!r}: {e}"
            )
        return ToolCall(
            name=name,
            arguments=arguments,
            call_id=item.get("call_id") or item.get("id"),
        )

    @classmethod
    def from_anthropic_response(
        cls,
        api_response: dict,
        *,
        capture_reasoning: bool = False,
    ) -> "ChatResponse":
        usage_data = api_response.get("usage")
        usage = (
            Usage(
                input_tokens=usage_data.get("input_tokens", 0),
                output_tokens=usage_data.get("output_tokens", 0),
                total_tokens=(
                    usage_data.get("input_tokens", 0)
                    + usage_data.get("output_tokens", 0)
                ),
            )
            if isinstance(usage_data, dict)
            else None
        )
        parsed_content = cls._parse_anthropic_content_blocks(
            api_response.get("content", []) or [],
            capture_reasoning=capture_reasoning,
        )
        finish_reason = cls._optional_string(api_response.get("stop_reason"))
        return cls(
            model=api_response.get("model"),
            response_id=api_response.get("id"),
            usage=usage,
            content=parsed_content.text,
            tool_calls=parsed_content.tool_calls,
            finish_reason=finish_reason,
            refusal=cls._anthropic_refusal(api_response, finish_reason),
            incomplete_reason=(
                finish_reason
                if finish_reason in {
                    "max_tokens",
                    "model_context_window_exceeded",
                }
                else None
            ),
            reasoning_events=parsed_content.reasoning_events,
        )

    @classmethod
    def _anthropic_refusal(
        cls,
        api_response: dict,
        finish_reason: Optional[str],
    ) -> Optional[str]:
        if finish_reason != "refusal":
            return None
        stop_details = api_response.get("stop_details")
        if isinstance(stop_details, dict):
            for field in ("reason", "category", "explanation"):
                value = cls._optional_string(stop_details.get(field))
                if value is not None:
                    return value
        return finish_reason

    @classmethod
    def _parse_anthropic_content_blocks(
        cls,
        blocks: Any,
        *,
        capture_reasoning: bool,
    ) -> _ParsedAnthropicContent:
        parsed_content = _ParsedAnthropicContent()
        for block in blocks:
            block_type = block.get("type")
            if block_type == "text" and parsed_content.text is None:
                parsed_content.text = cls._parse_anthropic_text_block(block)
            elif block_type == "thinking" and capture_reasoning:
                reasoning_event = cls._parse_anthropic_thinking_block(
                    block,
                    index=len(parsed_content.reasoning_events),
                )
                if reasoning_event is not None:
                    parsed_content.reasoning_events.append(reasoning_event)
            elif block_type == "tool_use":
                if parsed_content.tool_calls is None:
                    parsed_content.tool_calls = []
                parsed_content.tool_calls.append(
                    cls._parse_anthropic_tool_use_block(block)
                )
        return parsed_content

    @staticmethod
    def _parse_anthropic_text_block(block: dict) -> Optional[str]:
        return block.get("text")

    @staticmethod
    def _parse_anthropic_thinking_block(
        block: dict,
        *,
        index: int,
    ) -> Optional[ReasoningEvent]:
        thinking_text = block.get("thinking")
        if not isinstance(thinking_text, str) or not thinking_text:
            return None
        return ReasoningEvent(
            text=thinking_text,
            kind="summary",
            index=index,
            elapsed_s=0.0,
            delta_s=0.0,
        )

    @staticmethod
    def _parse_anthropic_tool_use_block(block: dict) -> ToolCall:
        name = block.get("name")
        arguments = block.get("input")
        if not isinstance(arguments, dict):
            raise InvalidToolArgumentsError(
                detail=f"Anthropic tool input must be dict for tool={name!r}"
            )
        return ToolCall(
            name=name,
            arguments=arguments,
            call_id=block.get("id"),
        )

    @classmethod
    def from_google_response(
        cls,
        api_response: dict,
        *,
        capture_reasoning: bool = False,
    ) -> "ChatResponse":
        usage = cls._parse_google_usage(api_response)
        first_candidate = (api_response.get("candidates") or [None])[0] or {}
        parsed_content = cls._parse_google_content(
            api_response,
            first_candidate,
            capture_reasoning=capture_reasoning,
        )
        return cls(
            model=api_response.get("modelVersion"),
            usage=usage,
            content=parsed_content.text,
            tool_calls=parsed_content.tool_calls,
            finish_reason=parsed_content.finish_reason,
            refusal=parsed_content.refusal,
            incomplete_reason=parsed_content.incomplete_reason,
            reasoning_events=parsed_content.reasoning_events,
        )

    @staticmethod
    def _parse_google_usage(api_response: dict) -> Optional[Usage]:
        usage_data = api_response.get("usageMetadata")
        if not isinstance(usage_data, dict):
            return None
        thoughts_tokens = usage_data.get("thoughtsTokenCount", 0)
        return Usage(
            input_tokens=usage_data.get("promptTokenCount", 0),
            output_tokens=usage_data.get("candidatesTokenCount", 0) + thoughts_tokens,
            total_tokens=usage_data.get("totalTokenCount", 0),
        )

    @classmethod
    def _parse_google_content(
        cls,
        api_response: dict,
        candidate: dict,
        *,
        capture_reasoning: bool,
    ) -> _ParsedGoogleContent:
        finish_reason = candidate.get("finishReason")
        normalized_finish_reason = (
            str(finish_reason) if finish_reason is not None else None
        )
        parsed_content = _ParsedGoogleContent(
            finish_reason=normalized_finish_reason,
            refusal=cls._google_refusal(api_response, candidate, normalized_finish_reason),
            incomplete_reason=(
                normalized_finish_reason
                if normalized_finish_reason in {"MAX_TOKENS", "length"}
                else None
            ),
        )
        content_obj = candidate.get("content") or {}
        parts = content_obj.get("parts") or []
        if not isinstance(parts, list):
            raise LLMAPIError(
                "Google API returned malformed response",
                detail="content.parts is not a list",
            )
        for part in parts:
            cls._parse_google_part(
                part,
                parsed_content,
                capture_reasoning=capture_reasoning,
            )
        return parsed_content

    @classmethod
    def _google_refusal(
        cls,
        api_response: dict,
        candidate: dict,
        finish_reason: Optional[str],
    ) -> Optional[str]:
        if finish_reason in {
            "SAFETY",
            "RECITATION",
            "BLOCKLIST",
            "PROHIBITED_CONTENT",
            "SPII",
            "IMAGE_SAFETY",
            "IMAGE_PROHIBITED_CONTENT",
            "IMAGE_RECITATION",
            "ESCALATION",
        }:
            return cls._optional_string(candidate.get("finishMessage")) or finish_reason
        prompt_feedback = api_response.get("promptFeedback")
        if isinstance(prompt_feedback, dict):
            return cls._optional_string(prompt_feedback.get("blockReason"))
        return None

    @classmethod
    def _parse_google_part(
        cls,
        part: Any,
        parsed_content: _ParsedGoogleContent,
        *,
        capture_reasoning: bool,
    ) -> None:
        if not isinstance(part, dict):
            return
        if part.get("thought"):
            if capture_reasoning:
                thought_text = part.get("text")
                if isinstance(thought_text, str) and thought_text:
                    parsed_content.reasoning_events.append(
                        ReasoningEvent(
                            text=thought_text,
                            kind="summary",
                            index=len(parsed_content.reasoning_events),
                            elapsed_s=0.0,
                            delta_s=0.0,
                        )
                    )
        elif parsed_content.text is None and "text" in part:
            parsed_content.text = part.get("text")
        fc = part.get("functionCall") or part.get("function_call")
        if isinstance(fc, dict) and fc:
            if parsed_content.tool_calls is None:
                parsed_content.tool_calls = []
            parsed_content.tool_calls.append(cls._parse_google_tool_call(part, fc))

    @staticmethod
    def _parse_google_tool_call(part: dict, function_call: dict) -> ToolCall:
        name = function_call.get("name")
        if not isinstance(name, str) or not name:
            raise InvalidToolArgumentsError(
                detail="Google functionCall.name must be non-empty str"
            )
        args = (
            function_call.get("args")
            if "args" in function_call
            else function_call.get("arguments", {})
        )
        if args is None:
            args = {}
        if not isinstance(args, dict):
            raise InvalidToolArgumentsError(
                detail=f"Google functionCall args must be dict for tool={name!r}"
            )
        thought_signature = part.get("thoughtSignature")
        provider_data = None
        if thought_signature is not None:
            provider_data = {"thoughtSignature": thought_signature}
        return ToolCall(
            name=name,
            arguments=args,
            call_id=name,
            provider_data=provider_data,
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

    def apply_cost_breakdown(
        self,
        cost_breakdown: Sequence[CostLineItem],
        *,
        accounting_complete: bool = True,
    ) -> None:
        """Attach non-token costs and retain a total only when it is complete."""
        line_items = list(cost_breakdown)
        if not all(isinstance(item, CostLineItem) for item in line_items):
            raise TypeError("cost_breakdown must contain CostLineItem values")
        if not isinstance(accounting_complete, bool):
            raise ValueError("accounting_complete must be a boolean")

        self.cost_breakdown = line_items
        if (
            not accounting_complete
            or self.cost_total is None
            or self.currency is None
            or any(item.currency != self.currency for item in self.cost_breakdown)
        ):
            self.cost_total = None
            return

        token_total = (
            self.cost_input + self.cost_output
            if self.cost_input is not None and self.cost_output is not None
            else self.cost_total
        )
        self.cost_total = token_total + sum(item.cost for item in self.cost_breakdown)
