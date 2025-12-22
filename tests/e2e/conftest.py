import os

from dotenv import load_dotenv
import pytest

from llm_api_adapter.llm_registry.llm_registry import LLM_REGISTRY

load_dotenv()

API_KEY_ENV = {
    "openai": os.getenv("OPENAI_API_KEY"),
    "anthropic": os.getenv("ANTHROPIC_API_KEY"),
    "google": os.getenv("GOOGLE_API_KEY"),
}

@pytest.fixture(scope="session")
def providers():
    providers_with_models = []
    for provider_name, provider_spec in LLM_REGISTRY.providers.items():
        api_key = API_KEY_ENV.get(provider_name)
        registry_models = list(provider_spec.models.keys())
        providers_with_models.append(
            {
                "name": provider_name,
                "api_key": api_key,
                "models": registry_models,
            }
        )
    return providers_with_models
