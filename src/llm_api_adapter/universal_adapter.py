from dataclasses import dataclass, fields
import logging
from typing import Any

from .adapters.base_adapter import LLMAdapterBase
from .adapters.anthropic_adapter import AnthropicAdapter
from .adapters.openai_adapter import OpenAIAdapter
from .adapters.google_adapter import GoogleAdapter

logger = logging.getLogger(__name__)


@dataclass
class UniversalLLMAPIAdapter:
    organization: str
    model: str
    api_key: str

    def __post_init__(self) -> None:
        if not self.organization or not isinstance(self.organization, str):
            raise ValueError("Invalid organization")
        if not self.model or not isinstance(self.model, str):
            raise ValueError("Invalid model")
        if not self.api_key or not isinstance(self.api_key, str):
            raise ValueError("Invalid API key")
        self.adapter = self._select_adapter(self.organization, self.model,
                                            self.api_key)

    def _select_adapter(
        self, organization: str, model: str, api_key: str
    ) -> LLMAdapterBase:
        """
        Selects the adapter based on the company.
        """
        for adapter_class in LLMAdapterBase.__subclasses__():
            company_field = None
            for field in fields(adapter_class):
                if field.name == 'company':
                    company_field = field
                    break
            if company_field and company_field.default == organization:
                return adapter_class(model=model, api_key=api_key)
        error_message = f"Unsupported organization: {organization}"
        logger.error(error_message)
        raise ValueError(error_message)

    def __getattr__(self, name: str) -> Any:
        """
        Redirects method calls to the selected adapter.
        """
        if hasattr(self.adapter, name):
            return getattr(self.adapter, name)
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )
