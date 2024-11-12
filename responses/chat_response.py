from typing import Optional

from pydantic import BaseModel


class ChatResponse(BaseModel):
    model: Optional[str] = None
    response_id: Optional[str] = None
    timestamp: Optional[int] = None
    tokens_used: Optional[int] = None
    content: str
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
        tokens_used = api_response.usage.input_tokens + api_response.usage.output_tokens
        return cls(
            model=api_response.model,
            response_id=api_response.id,
            tokens_used=tokens_used,
            content=api_response.content[0].text,
            finish_reason=api_response.stop_reason
        )

    @classmethod
    def from_google_response(cls, api_response: dict) -> "ChatResponse":
        return cls(
            tokens_used=api_response.usage_metadata.total_token_count,
            content=api_response.candidates[0].content.parts[0].text,
            finish_reason=str(api_response.candidates[0].finish_reason.name)
        )
