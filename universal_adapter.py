from typing import Any

from .adapters.base_adapter import LLMAdapterBase
from .adapters.anthropic_adapter import AnthropicAdapter
from .adapters.openai_adapter import OpenAIAdapter
from .adapters.google_adapter import GoogleAdapter


class UniversalLLMAPIAdapter:
    def __init__(self, organization: str, model: str, api_key: str) -> None:
        self.adapter = self._select_adapter(organization, model, api_key)

    def _select_adapter(
            self,
            organization: str,
            model: str,
            api_key: str
        ) -> LLMAdapterBase:
        """
        Selects the adapter based on the company.
        """
        for adapter_class in LLMAdapterBase.__subclasses__():
            if adapter_class.model_fields['company'].default == organization:
                adapter = adapter_class(model=model, api_key=api_key)
                return adapter
        raise ValueError(f"Unsupported organization: {organization}")
    
    def __getattr__(self, name: str) -> Any:
        """
        Redirects method calls to the selected adapter.
        """
        if hasattr(self.adapter, name):
            return getattr(self.adapter, name)
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )
