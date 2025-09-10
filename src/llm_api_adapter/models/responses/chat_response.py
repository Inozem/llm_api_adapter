from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ChatResponse:
    model: Optional[str] = None
    response_id: Optional[str] = None
    timestamp: Optional[int] = None
    usage: Optional[Usage] = None
    content: str = field(default_factory=str)
    finish_reason: Optional[str] = None

    @classmethod
    def from_openai_response(cls, api_response: dict) -> "ChatResponse":
        u = api_response.get("usage", {})
        usage = Usage(
            prompt_tokens=u.get("prompt_tokens", 0),
            completion_tokens=u.get("completion_tokens", 0),
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
            prompt_tokens=u.get("input_tokens", 0),
            completion_tokens=u.get("output_tokens", 0),
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
        usage = Usage(
            prompt_tokens=u.get("promptTokenCount", 0),
            completion_tokens=u.get("candidatesTokenCount", 0),
            total_tokens=u.get("totalTokenCount", 0),
        )
        first_candidate = api_response["candidates"][0]
        return cls(
            usage=usage,
            content=first_candidate["content"]["parts"][0]["text"],
            finish_reason=str(first_candidate.get("finishReason")),
        )
