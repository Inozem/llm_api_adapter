from dataclasses import dataclass
import logging
from typing import List, Optional
import warnings

from ..adapters.base_adapter import LLMAdapterBase
from ..errors.llm_api_error import LLMAPIError
from ..errors.config_errors import LLMConfigError
from ..llms.anthropic.sync_client import ClaudeSyncClient
from ..models.messages.chat_message import Message, Messages
from ..models.responses.chat_response import ChatResponse

logger = logging.getLogger(__name__)


@dataclass
class AnthropicAdapter(LLMAdapterBase):
    company: str = "anthropic"

    def chat(
        self,
        messages: List[Message] | Messages,
        max_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        reasoning_level: Optional[str | int] = None,
        timeout_s: Optional[float] = None,
    ) -> ChatResponse:
        temperature = self._validate_parameter(
            name="temperature", value=temperature, min_value=0, max_value=2
        )
        top_p = self._validate_parameter(
            name="top_p", value=top_p, min_value=0, max_value=1
        )
        try:
            normalized_messages = self._normalize_messages(messages)
            system_prompt, transformed_messages = normalized_messages.to_anthropic()
            client = ClaudeSyncClient(api_key=self.api_key)
            params = {
                "model": self.model,
                "messages": transformed_messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "system": system_prompt,
                "timeout_s": timeout_s,
            }
            if reasoning_level:
                normalized_reasoning_level = self._normalize_reasoning_level(reasoning_level)
                if normalized_reasoning_level:
                    self.validate_reasoning_and_tokens(
                        max_tokens=max_tokens,
                        reasoning_level=reasoning_level,
                        normalized_reasoning_level=normalized_reasoning_level
                    )
                    params["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": normalized_reasoning_level
                    }
            params = {k: v for k, v in params.items() if v is not None}
            response = client.chat_completion(**params)
            chat_response = ChatResponse.from_anthropic_response(response)
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

    def _normalize_reasoning_level(self, level: str | int) -> int | None:
        minimum_level = 1024
        normalized_level = None
        if level and not self.is_reasoning:
            warning_message = (f"Model '{self.model}' does not support reasoning "
                               "— reasoning disabled.")
            warnings.warn(warning_message, UserWarning)
            logger.info(warning_message)
            return None
        if isinstance(level, bool):
            raise ValueError("Invalid type for level: bool is not accepted")
        if isinstance(level, str):
            if level in self.reasoning_levels:
                normalized_level = self.reasoning_levels[level]
            else:
                raise ValueError(f"Unknown reasoning level key: {level!r}. "
                                 f"Valid keys: {list(self.reasoning_levels.keys())}")
        if isinstance(level, int):
            normalized_level = level
        if normalized_level:
            if normalized_level >= minimum_level:
                return normalized_level
            warning_message = (
                f"Reasoning level '{level}' is below the minimum supported value {minimum_level}; "
                f"using {minimum_level} instead.")
            warnings.warn(warning_message, UserWarning)
            logger.info(warning_message)
            return minimum_level
        raise ValueError("Invalid type for level: expected int or str, "
                         f"got {type(level).__name__!r}")
    
    def validate_reasoning_and_tokens(
        self,
        max_tokens: int,
        reasoning_level: int | str,
        normalized_reasoning_level: int
    ) -> None:
        if max_tokens <= normalized_reasoning_level:
            raise LLMConfigError(
            detail=(
                f"Provided max_tokens={max_tokens}, "
                f"reasoning_level={normalized_reasoning_level} "
                f"(requested '{reasoning_level}'). "
                f"Increase max_tokens above {normalized_reasoning_level} "
                "or reduce reasoning_level."
            )
        )
