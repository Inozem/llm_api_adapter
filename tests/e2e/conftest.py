import asyncio
import os
import time
from itertools import zip_longest
from pathlib import Path

from dotenv import load_dotenv
import pytest

from llm_api_adapter.errors import LLMAPIRateLimitError, LLMAPIServerError
from llm_api_adapter.llm_registry.llm_registry import LLM_REGISTRY
from src.llm_api_adapter.errors import (
    LLMAPIRateLimitError as SourceLLMAPIRateLimitError,
    LLMAPIServerError as SourceLLMAPIServerError,
)

_FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
_RETRY_DELAYS = [2, 4, 8]
_MAX_ATTEMPTS = len(_RETRY_DELAYS) + 1
_TRANSIENT_ERRORS = (
    LLMAPIServerError,
    LLMAPIRateLimitError,
    SourceLLMAPIServerError,
    SourceLLMAPIRateLimitError,
)
_LATEST_MODEL_BY_PROVIDER = {
    provider_name: next(iter(provider_spec.models), None)
    for provider_name, provider_spec in LLM_REGISTRY.providers.items()
}

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
    def _call(adapter, **kwargs):
        for attempt in range(_MAX_ATTEMPTS):
            try:
                resp = adapter.chat(**kwargs)
            except _TRANSIENT_ERRORS:
                if attempt == _MAX_ATTEMPTS - 1:
                    raise
                time.sleep(_RETRY_DELAYS[attempt])
                continue
            if resp.finish_reason != "refusal" or attempt == _MAX_ATTEMPTS - 1:
                return resp
            time.sleep(_RETRY_DELAYS[attempt])
        return resp
    return _call


@pytest.fixture(scope="session")
def stream_with_retry():
    """Returns a helper that retries a complete stream on transient provider errors.

    ``on_retry`` may reset callback observers after a partial failed attempt.
    """
    def _call(adapter, **kwargs):
        on_retry = kwargs.pop("on_retry", None)
        for attempt in range(_MAX_ATTEMPTS):
            try:
                return list(adapter.stream_chat(**kwargs))
            except _TRANSIENT_ERRORS:
                if attempt == _MAX_ATTEMPTS - 1:
                    raise
                if on_retry is not None:
                    on_retry()
                time.sleep(_RETRY_DELAYS[attempt])
        return []

    return _call


@pytest.fixture(scope="session")
def async_chat_with_retry():
    """Return an async helper that retries transient achat() failures."""
    async def _call(adapter, **kwargs):
        for attempt in range(_MAX_ATTEMPTS):
            try:
                response = await adapter.achat(**kwargs)
            except _TRANSIENT_ERRORS:
                if attempt == _MAX_ATTEMPTS - 1:
                    raise
                await asyncio.sleep(_RETRY_DELAYS[attempt])
                continue
            if response.finish_reason != "refusal" or attempt == _MAX_ATTEMPTS - 1:
                return response
            await asyncio.sleep(_RETRY_DELAYS[attempt])
        return response

    return _call


@pytest.fixture(scope="session")
def async_stream_with_retry():
    """Return an async helper that retries a complete astream_chat() run."""
    async def _call(adapter, **kwargs):
        on_retry = kwargs.pop("on_retry", None)
        for attempt in range(_MAX_ATTEMPTS):
            try:
                chunks = []
                async for chunk in adapter.astream_chat(**kwargs):
                    chunks.append(chunk)
                return chunks
            except _TRANSIENT_ERRORS:
                if attempt == _MAX_ATTEMPTS - 1:
                    raise
                if on_retry is not None:
                    on_retry()
                await asyncio.sleep(_RETRY_DELAYS[attempt])
        return []

    return _call


@pytest.fixture(scope="session")
def vision_image_bytes() -> bytes:
    return (_FIXTURES_DIR / "test_image.png").read_bytes()


@pytest.fixture(scope="session")
def pdf_bytes() -> bytes:
    return (_FIXTURES_DIR / "test_document.pdf").read_bytes()


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


@pytest.fixture(scope="session")
def async_e2e_models(providers):
    """Select the latest registered model per provider for async E2E coverage."""
    selected = []
    for provider in providers:
        env_name = f"ASYNC_E2E_{provider['name'].upper()}_MODEL"
        override = os.getenv(env_name)

        if override:
            if override not in provider["models"]:
                raise pytest.UsageError(
                    f"{env_name}={override!r} is not registered for "
                    f"{provider['name']}"
                )
            model = override
        else:
            model = _LATEST_MODEL_BY_PROVIDER.get(provider["name"])
            if model is None or model not in provider["models"]:
                raise pytest.UsageError(
                    f"No latest model is registered for {provider['name']}"
                )

        selected.append((provider, model))
    return selected


@pytest.fixture(scope="session")
def configured_async_e2e_models(async_e2e_models):
    """Return the selected async E2E models whose API keys are configured."""
    return [
        (provider, model)
        for provider, model in async_e2e_models
        if provider["api_key"]
    ]
