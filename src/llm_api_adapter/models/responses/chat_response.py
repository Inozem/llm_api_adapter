from dataclasses import dataclass, field
from typing import Optional


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
    content: str = field(default_factory=str)
    finish_reason: Optional[str] = None

    @classmethod
    def from_openai_response(cls, api_response: dict) -> "ChatResponse":
        u = api_response.get("usage", {})
        usage = Usage(
            input_tokens=u.get("prompt_tokens", 0),
            output_tokens=u.get("completion_tokens", 0),
            total_tokens=u.get("total_tokens", 0),
        )
        return cls(
            model=api_response.get("model"),
            response_id=api_response.get("id"),
            timestamp=api_response.get("created"),
            usage=usage,
            content=api_response["choices"][0]["message"]["content"],
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
        return cls(
            model=api_response.get("model"),
            response_id=api_response.get("id"),
            usage=usage,
            content=api_response.get("content")[0].get("text"),
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
        return cls(
            usage=usage,
            content=first_candidate["content"]["parts"][0]["text"],
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
