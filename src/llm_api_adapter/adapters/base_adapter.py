import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import FrozenSet
import warnings

from ..models.responses.chat_response import ChatResponse

logger = logging.getLogger(__name__)

@dataclass
class LLMAdapterBase(ABC):
    model: str
    api_key: str
    verified_models: FrozenSet[str] = field(default_factory=frozenset)

    def __post_init__(self):
        if len(self.api_key) < 1:
            error_message = "api_key must be a non-empty string"
            logger.error(error_message)
            raise ValueError(error_message)
        if self.model not in self.verified_models:
            warnings.warn(
                (f"Model '{self.model}' is not verified for this adapter. "
                 "Continuing with the selected adapter."),
                UserWarning
            )
            logger.warning(f"Unverified model used: {self.model}")

    @abstractmethod
    def chat(self, **kwargs) -> ChatResponse:
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
