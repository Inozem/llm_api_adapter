from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
import logging
from typing import Optional
import warnings

from ..llm_registry.llm_registry import Pricing, LLM_REGISTRY
from ..models.responses.chat_response import ChatResponse

logger = logging.getLogger(__name__)

@dataclass
class LLMAdapterBase(ABC):
    api_key: str
    model: str
    company: str
    pricing: Optional[Pricing] = None

    def __post_init__(self):
        if not self.api_key:
            error_message = "api_key must be a non-empty string"
            logger.error(error_message)
            raise ValueError(error_message)
        if self.pricing is None:
            provider = LLM_REGISTRY.providers.get(self.company)
            model_spec = provider.models.get(self.model) if provider else None
            if not model_spec:
                warnings.warn(
                    (f"Model '{self.model}' is not verified for this adapter. "
                     "Continuing with the selected adapter."),
                    UserWarning
                )
                logger.warning(f"Unverified model used: {self.model}")
                self.pricing = None
            else:
                base_pricing = getattr(model_spec, "pricing", None)
                self.pricing = deepcopy(base_pricing) if base_pricing else None

    @abstractmethod
    def generate_chat_answer(self, **kwargs) -> ChatResponse:
        """
        Generates a response based on the provided conversation.
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

    @classmethod
    def handle_error(cls, error: Exception, company: str):
        logger.error(f"Error in company {company}: {error}")
        raise error
