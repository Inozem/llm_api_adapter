from dataclasses import dataclass
from typing import List, Optional
import warnings

from ...errors.llm_api_error import LLMAPIError
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
        u = api_response.get("usage", {})
        usage = Usage(
            input_tokens=u.get("prompt_tokens", 0),
            output_tokens=u.get("completion_tokens", 0),
            total_tokens=u.get("total_tokens", 0),
        )
        text = api_response["choices"][0]["message"]["content"]
        if not text or not text.strip():
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
            finish_reason=api_response["choices"][0].get("finish_reason"),
        )

    @classmethod
    def from_anthropic_response(cls, api_response: dict) -> "ChatResponse":
        u = api_response.get("usage", {})
        usage = Usage(
            input_tokens=u.get("input_tokens", 0),
            output_tokens=u.get("output_tokens", 0),
            total_tokens=u.get("input_tokens", 0) + u.get("output_tokens", 0),
        )
        content = api_response.get("content", [])
        first_text = next(
            (block.get("text") for block in content if block.get("type") == "text"),
            None
        )
        return cls(
            model=api_response.get("model"),
            response_id=api_response.get("id"),
            usage=usage,
            content=first_text,
            finish_reason=api_response.get("stop_reason"),
        )

    @classmethod
    def from_google_response(cls, api_response: dict) -> "ChatResponse":
        u = api_response.get("usageMetadata", {})
        thoughts_tokens = u.get("thoughtsTokenCount", 0)
        usage = Usage(
            input_tokens=u.get("promptTokenCount", 0),
            output_tokens=u.get("candidatesTokenCount", 0) + thoughts_tokens,
            total_tokens=u.get("totalTokenCount", 0),
        )
        first_candidate = api_response["candidates"][0]
        content = first_candidate.get("content", {})
        parts = content.get("parts")
        if not parts or not isinstance(parts, list):
            raise LLMAPIError(
                "Google API returned malformed response",
                detail="content.parts missing — likely max_output_tokens too small"
            )
        text = parts[0].get("text")
        return cls(
            usage=usage,
            content=text,
            finish_reason=str(first_candidate.get("finishReason")),
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
