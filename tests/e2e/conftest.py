import asyncio
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
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


@dataclass(frozen=True)
class E2EProviderProfile:
    """The providers selected for one independently runnable E2E lane."""

    name: str
    provider_names: tuple[str, ...]


class E2EProvider(dict):
    """Provider test data that never renders an API key in pytest output."""

    def __repr__(self) -> str:
        safe_data = dict(self)
        if safe_data.get("api_key"):
            safe_data["api_key"] = "***"
        return dict.__repr__(safe_data)


_BUILTIN_E2E_PROFILE = E2EProviderProfile(
    name="builtin",
    provider_names=("openai", "anthropic", "google"),
)
_MISTRAL_E2E_PROFILE = E2EProviderProfile(
    name="mistral",
    provider_names=("mistral",),
)
_E2E_PROFILE_PARAMS = (
    pytest.param(
        _BUILTIN_E2E_PROFILE,
        id="builtin",
        marks=pytest.mark.e2e_builtin,
    ),
    pytest.param(
        _MISTRAL_E2E_PROFILE,
        id="mistral",
        marks=pytest.mark.e2e_mistral,
    ),
)

load_dotenv()

API_KEY_ENV = {
    "openai": os.getenv("OPENAI_API_KEY"),
    "anthropic": os.getenv("ANTHROPIC_API_KEY"),
    "google": os.getenv("GOOGLE_API_KEY"),
    "mistral": os.getenv("MISTRAL_API_KEY"),
}


def _select_latest_e2e_models(providers, override_prefix: str):
    """Select one registered model per provider for a bounded E2E profile."""
    selected = []
    for provider in providers:
        env_name = f"{override_prefix}_{provider['name'].upper()}_MODEL"
        override = os.getenv(env_name)

        if override:
            if override not in provider["models"]:
                raise pytest.UsageError(
                    f"{env_name}={override!r} is not registered for "
                    f"{provider['name']}"
                )
            model = override
        else:
            model = provider["latest_model"]
            if model is None or model not in provider["models"]:
                raise pytest.UsageError(
                    f"No latest model is registered for {provider['name']}"
                )

        selected.append((provider, model))
    return selected


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


@pytest.fixture(scope="session", params=_E2E_PROFILE_PARAMS)
def e2e_provider_profile(request) -> E2EProviderProfile:
    """Select one independently runnable provider E2E lane."""
    return request.param


@pytest.fixture(scope="session")
def providers(e2e_provider_profile: E2EProviderProfile):
    """Return the providers selected for the current E2E lane."""
    if e2e_provider_profile.name == "mistral":
        try:
            version("llm-api-adapter-mistral")
        except PackageNotFoundError:
            pytest.skip("llm-api-adapter-mistral is not installed")
        if not API_KEY_ENV["mistral"]:
            pytest.skip("MISTRAL_API_KEY is not configured")

    providers_with_models = []
    for provider_name in e2e_provider_profile.provider_names:
        if provider_name == "mistral":
            registry_models = [
                os.getenv("MISTRAL_E2E_MODEL", "mistral-large-2512")
            ]
        else:
            provider_spec = LLM_REGISTRY.providers[provider_name]
            registry_models = list(provider_spec.models.keys())

        api_key = API_KEY_ENV.get(provider_name)
        providers_with_models.append(
            E2EProvider(
                {
                    "name": provider_name,
                    "api_key": api_key,
                    "models": registry_models,
                    "latest_model": registry_models[0] if registry_models else None,
                }
            )
        )
    return providers_with_models


@pytest.fixture(scope="session")
def async_e2e_models(providers):
    """Select the latest registered model per provider for async E2E coverage."""
    return _select_latest_e2e_models(providers, "ASYNC_E2E")


@pytest.fixture(scope="session")
def configured_async_e2e_models(async_e2e_models):
    """Return the selected async E2E models whose API keys are configured."""
    return [
        (provider, model)
        for provider, model in async_e2e_models
        if provider["api_key"]
    ]


@pytest.fixture(scope="session")
def sync_httpx_e2e_models(providers):
    """Select one latest model per provider for the sync HTTPX pilot."""
    return _select_latest_e2e_models(providers, "SYNC_HTTPX_E2E")


@pytest.fixture(scope="session")
def configured_sync_httpx_e2e_models(sync_httpx_e2e_models):
    """Return sync HTTPX pilot models whose provider keys are configured."""
    return [
        (provider, model)
        for provider, model in sync_httpx_e2e_models
        if provider["api_key"]
    ]
