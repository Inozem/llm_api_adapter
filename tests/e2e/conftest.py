import os
import time
from itertools import zip_longest
from pathlib import Path

from dotenv import load_dotenv
import pytest

from llm_api_adapter.errors import LLMAPIRateLimitError, LLMAPIServerError
from llm_api_adapter.llm_registry.llm_registry import LLM_REGISTRY

_FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"

load_dotenv()

API_KEY_ENV = {
    "openai": os.getenv("OPENAI_API_KEY"),
    "anthropic": os.getenv("ANTHROPIC_API_KEY"),
    "google": os.getenv("GOOGLE_API_KEY"),
}

@pytest.fixture
def iter_provider_models(providers):
    """Returns a generator of (provider, model) pairs grouped round-robin across providers."""
    def _iter():
        groups = list(zip_longest(*[p["models"] for p in providers]))
        for group in groups:
            for p, model in zip(providers, group):
                if model is not None:
                    yield p, model
    return _iter


@pytest.fixture(scope="session")
def chat_with_retry():
    """Returns a helper that retries adapter.chat() on 5xx errors, rate limits, or model refusals.

    Delays follow exponential backoff: 2s, 4s, 8s.
    """
    _delays = [2, 4, 8]
    _max_attempts = len(_delays)

    def _call(adapter, **kwargs):
        for attempt in range(_max_attempts):
            try:
                resp = adapter.chat(**kwargs)
            except (LLMAPIServerError, LLMAPIRateLimitError):
                if attempt == _max_attempts - 1:
                    raise
                time.sleep(_delays[attempt])
                continue
            if resp.finish_reason != "refusal" or attempt == _max_attempts - 1:
                return resp
            time.sleep(_delays[attempt])
        return resp
    return _call


@pytest.fixture(scope="session")
def vision_image_bytes() -> bytes:
    return (_FIXTURES_DIR / "test_image.png").read_bytes()


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
