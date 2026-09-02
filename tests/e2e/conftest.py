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
from llm_api_adapter.universal_adapter import (
    ORGANIZATION_PLUGIN_DISCOVERY,
    SERVICE_PROVIDER_REGISTRY,
)

_FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
_RETRY_DELAYS = [2, 4, 8]
_MAX_ATTEMPTS = len(_RETRY_DELAYS) + 1
_TRANSIENT_ERRORS = (
    LLMAPIServerError,
    LLMAPIRateLimitError,
)


@dataclass(frozen=True)
class E2EOrganizationProfile:
    """The organizations selected for one independently runnable E2E lane."""

    name: str
    organization_names: tuple[str, ...]
    default_models: tuple[tuple[str, str], ...] = ()


class E2EOrganization(dict):
    """Organization test data that never renders an API key in pytest output."""

    def __repr__(self) -> str:
        safe_data = dict(self)
        if safe_data.get("api_key"):
            safe_data["api_key"] = "***"
        return dict.__repr__(safe_data)


_OPENAI_E2E_PROFILE = E2EOrganizationProfile(
    name="openai",
    organization_names=("openai",),
)
_ANTHROPIC_E2E_PROFILE = E2EOrganizationProfile(
    name="anthropic",
    organization_names=("anthropic",),
)
_GOOGLE_E2E_PROFILE = E2EOrganizationProfile(
    name="google",
    organization_names=("google",),
)
_MISTRAL_E2E_PROFILE = E2EOrganizationProfile(
    name="mistral",
    organization_names=("mistral",),
)
_XAI_E2E_PROFILE = E2EOrganizationProfile(
    name="xai",
    organization_names=("xai",),
    default_models=(("xai", "grok-4.6"),),
)
_E2E_PROFILE_PARAMS = (
    pytest.param(
        _OPENAI_E2E_PROFILE,
        id="openai",
        marks=(pytest.mark.e2e_builtin, pytest.mark.e2e_openai),
    ),
    pytest.param(
        _ANTHROPIC_E2E_PROFILE,
        id="anthropic",
        marks=(pytest.mark.e2e_builtin, pytest.mark.e2e_anthropic),
    ),
    pytest.param(
        _GOOGLE_E2E_PROFILE,
        id="google",
        marks=(pytest.mark.e2e_builtin, pytest.mark.e2e_google),
    ),
    pytest.param(
        _MISTRAL_E2E_PROFILE,
        id="mistral",
        marks=pytest.mark.e2e_mistral,
    ),
    pytest.param(
        _XAI_E2E_PROFILE,
        id="xai",
        marks=pytest.mark.e2e_xai,
    ),
)

load_dotenv()

API_KEY_ENV = {
    "openai": os.getenv("OPENAI_API_KEY"),
    "anthropic": os.getenv("ANTHROPIC_API_KEY"),
    "google": os.getenv("GOOGLE_API_KEY"),
    "mistral": os.getenv("MISTRAL_API_KEY"),
    "xai": os.getenv("XAI_API_KEY"),
}


def _profile_default_model(
    profile: E2EOrganizationProfile,
    organization_name: str,
    registry_models: list[str],
) -> str | None:
    """Return the profile's bounded-transport model for one organization."""
    configured_models = dict(profile.default_models)
    return configured_models.get(
        organization_name,
        registry_models[0] if registry_models else None,
    )


def _select_latest_e2e_models(organizations, override_prefix: str):
    """Select one registered model per organization for a bounded E2E profile."""
    selected = []
    for organization in organizations:
        env_name = f"{override_prefix}_{organization['name'].upper()}_MODEL"
        override = os.getenv(env_name)

        if override:
            if override not in organization["models"]:
                raise pytest.UsageError(
                    f"{env_name}={override!r} is not registered for "
                    f"{organization['name']}"
                )
            model = override
        else:
            model = organization["latest_model"]
            if model is None or model not in organization["models"]:
                raise pytest.UsageError(
                    f"No latest model is registered for {organization['name']}"
                )

        selected.append((organization, model))
    return selected


@pytest.fixture
def iter_organization_models(organizations):
    """Return (organization, model) pairs grouped round-robin by organization."""
    def _iter():
        groups = list(zip_longest(*[o["models"] for o in organizations]))
        for group in groups:
            for organization, model in zip(organizations, group):
                if model is not None:
                    yield organization, model
    return _iter


@pytest.fixture(scope="session")
def tool_choice_for_model():
    """Select the strongest registered tool-choice mode for one E2E model."""

    def _select(organization_name: str, model_name: str, tool_name: str) -> str:
        model_spec = LLM_REGISTRY.organizations[organization_name].models[model_name]
        allowed_modes = model_spec.request_rules.allowed_tool_choice_modes
        if allowed_modes is None or "tool" in allowed_modes:
            return tool_name
        if "any" in allowed_modes:
            return "any"
        if "auto" in allowed_modes:
            return "auto"
        raise pytest.UsageError(
            f"{organization_name}/{model_name} has no tool-call mode enabled "
            "in its registered request rules"
        )

    return _select


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
def e2e_organization_profile(request) -> E2EOrganizationProfile:
    """Select one independently runnable organization E2E lane."""
    return request.param


@pytest.fixture(scope="session")
def organizations(e2e_organization_profile: E2EOrganizationProfile):
    """Return the organizations selected for the current E2E lane."""
    if e2e_organization_profile.name in {"mistral", "xai"}:
        distribution = f"llm-api-adapter-{e2e_organization_profile.name}"
        try:
            version(distribution)
        except PackageNotFoundError:
            pytest.skip(f"{distribution} is not installed")

        api_key_env_name = f"{e2e_organization_profile.name.upper()}_API_KEY"
        if not API_KEY_ENV[e2e_organization_profile.name]:
            if e2e_organization_profile.name == "xai":
                raise pytest.UsageError(
                    f"{api_key_env_name} is not configured for the xAI E2E profile"
                )
            pytest.skip(f"{api_key_env_name} is not configured")

        ORGANIZATION_PLUGIN_DISCOVERY.discover(
            SERVICE_PROVIDER_REGISTRY,
            model_registry=LLM_REGISTRY,
        )

    organizations_with_models = []
    for organization_name in e2e_organization_profile.organization_names:
        organization_spec = LLM_REGISTRY.organizations.get(organization_name)
        if organization_spec is None:
            raise pytest.UsageError(
                f"No models are registered for {organization_name}"
            )
        registry_models = list(organization_spec.models.keys())

        api_key = API_KEY_ENV.get(organization_name)
        organizations_with_models.append(
            E2EOrganization(
                {
                    "name": organization_name,
                    "api_key": api_key,
                    "models": registry_models,
                    "latest_model": _profile_default_model(
                        e2e_organization_profile,
                        organization_name,
                        registry_models,
                    ),
                }
            )
        )
    return organizations_with_models


@pytest.fixture(scope="session")
def async_e2e_models(organizations):
    """Select the latest registered model per organization for async E2E coverage."""
    return _select_latest_e2e_models(organizations, "ASYNC_E2E")


@pytest.fixture(scope="session")
def configured_async_e2e_models(async_e2e_models):
    """Return the selected async E2E models whose API keys are configured."""
    return [
        (organization, model)
        for organization, model in async_e2e_models
        if organization["api_key"]
    ]


@pytest.fixture(scope="session")
def sync_httpx_e2e_models(organizations):
    """Select one latest model per organization for the sync HTTPX pilot."""
    return _select_latest_e2e_models(organizations, "SYNC_HTTPX_E2E")


@pytest.fixture(scope="session")
def configured_sync_httpx_e2e_models(sync_httpx_e2e_models):
    """Return sync HTTPX pilot models whose organization keys are configured."""
    return [
        (organization, model)
        for organization, model in sync_httpx_e2e_models
        if organization["api_key"]
    ]
