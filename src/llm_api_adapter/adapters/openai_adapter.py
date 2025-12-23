from dataclasses import dataclass
import logging
from typing import List, Optional
import warnings

from ..adapters.base_adapter import LLMAdapterBase
from ..errors.llm_api_error import LLMAPIError
from ..llms.openai.sync_client import OpenAISyncClient
from ..models.messages.chat_message import Message, Messages
from ..models.responses.chat_response import ChatResponse

logger = logging.getLogger(__name__)


@dataclass
class OpenAIAdapter(LLMAdapterBase):
    company: str = "openai"

    def chat(
        self,
        messages: List[Message] | Messages,
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        reasoning_level: Optional[str | int] = None,
        timeout_s: Optional[float] = None
    ) -> ChatResponse:
        temperature = self._validate_parameter(
            name="temperature", value=temperature, min_value=0, max_value=2
        )
        top_p = self._validate_parameter(
            name="top_p", value=top_p, min_value=0, max_value=1
        )
        try:
            normalized_messages = self._normalize_messages(messages)
            transformed_messages = normalized_messages.to_openai()
            normalized_reasoning_level = self._normalize_reasoning_level(reasoning_level)
            client = OpenAISyncClient(api_key=self.api_key)
            params = {
                "model": self.model,
                "messages": transformed_messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "reasoning_effort": normalized_reasoning_level,
            }
            params = {k: v for k, v in params.items() if v is not None}
            response = client.chat_completion(timeout=timeout_s, **params)
            chat_response = ChatResponse.from_openai_response(response)
            if self.pricing:
                chat_response.apply_pricing(
                    price_input_per_token=self.pricing.in_per_token,
                    price_output_per_token=self.pricing.out_per_token,
                    currency=self.pricing.currency
                )
            return chat_response
        except LLMAPIError as e:
            self.handle_error(e)
        except Exception as e:
            error_message = getattr(e, "text", None) or str(e)
            self.handle_error(error=e, error_message=error_message)

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
