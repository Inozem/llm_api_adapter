from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
import logging
from typing import Any, Dict, Optional
import warnings

from ..llm_registry.llm_registry import Pricing, LLM_REGISTRY
from ..models.messages.chat_message import Messages
from ..models.responses.chat_response import ChatResponse

logger = logging.getLogger(__name__)

REASONING_LEVELS_DEFAULT = {
    "none": 0,
    "low": 100,
    "medium": 1000,
    "high": 10000,
}


@dataclass
class LLMAdapterBase(ABC):
    api_key: str
    model: str
    company: str
    pricing: Optional[Pricing] = None
    is_reasoning: bool = False
    reasoning_levels: Dict[str, int] = field(
        default_factory=lambda: REASONING_LEVELS_DEFAULT.copy()
    )

    def __post_init__(self):
        if not self.api_key:
            error_message = "api_key must be a non-empty string"
            logger.error(error_message)
            raise ValueError(error_message)
        provider = LLM_REGISTRY.providers.get(self.company)
        model_spec = provider.models.get(self.model) if provider else None
        if not model_spec:
            warnings.warn(
                (f"Model '{self.model}' is not verified for the {self.company} adapter. "
                 f"Continuing with the selected adapter."),
                 UserWarning
            )
            logger.warning(f"Unverified model used: {self.model}")
            self.pricing = None
        else:
            base_pricing = getattr(model_spec, "pricing", None)
            self.pricing = deepcopy(base_pricing) if base_pricing else None
            self.is_reasoning = getattr(model_spec, "is_reasoning", False)

    @abstractmethod
    def chat(self, **kwargs) -> ChatResponse:
        """
        Generates a response based on the provided conversation.
        """
        pass

    @abstractmethod
    def _normalize_reasoning_level(self, level: str | int) -> int | str:
        """
        Convert a user-provided reasoning level ('none', 'low', 'medium', 'high' or numeric)
        into the provider-specific format.
        Models without reasoning support should ignore
        """
        pass

    def _validate_parameter(
        self, name: str, value: float, min_value: float, max_value: float
    ) -> float:
        if not (min_value <= value <= max_value):
            error_message = (f"{name} must be between {min_value} and "
                             f"{max_value}, got {value}")
            logger.error(error_message)
            raise ValueError(error_message)
        return value

    def _normalize_messages(self, messages: Any) -> Messages:
        if isinstance(messages, Messages):
            return messages
        if isinstance(messages, list):
            return Messages(messages)
        raise TypeError("messages must be a list or Messages instance")

    def handle_error(self, error: Exception, error_message: Optional[str] = None):
        err_msg = (f"Error with the provider '{self.company}' "
               f"the model '{self.model}': {error_message}. ")
        if error_message:
            err_msg += f"{error_message}"
        logger.error(err_msg)
        raise

    # ---------------- LEGACY ---------------- #
    def generate_chat_answer(self, **kwargs) -> ChatResponse:
        """Deprecated: use .chat() instead."""
        warnings.warn(
            "'generate_chat_answer' is deprecated, use 'chat' instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.chat(**kwargs)
