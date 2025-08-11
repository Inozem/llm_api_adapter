from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ChatResponse:
    model: Optional[str] = None
    response_id: Optional[str] = None
    timestamp: Optional[int] = None
    tokens_used: Optional[int] = None
    content: str = field(default_factory=str)
    finish_reason: Optional[str] = None

    @classmethod
    def from_openai_response(cls, api_response: dict) -> "ChatResponse":
        return cls(
            model=api_response.get("model"),
            response_id=api_response.get("id"),
            timestamp=api_response.get("created"),
            tokens_used=api_response.get("usage", {}).get("total_tokens"),
            content=api_response["choices"][0]["message"]["content"],
            finish_reason=api_response["choices"][0].get("finish_reason")
        )

    @classmethod
    def from_anthropic_response(cls, api_response: dict) -> "ChatResponse":
        print(f"API response {api_response}")
        usage = api_response.get("usage")
        tokens_used = usage.get("input_tokens") + usage.get("output_tokens")
        return cls(
            model=api_response.get("model"),
            response_id=api_response.get("id"),
            tokens_used=tokens_used,
            content=api_response.get("content")[0].get("text"),
            finish_reason=api_response.get("stop_reason")
        )

    @classmethod
    def from_google_response(cls, api_response: dict) -> "ChatResponse":
        first_candidate = api_response["candidates"][0]
        return cls(
            tokens_used=api_response["usageMetadata"]["totalTokenCount"],
            content=first_candidate["content"]["parts"][0]["text"],
            finish_reason=str(first_candidate["finishReason"])
        )
