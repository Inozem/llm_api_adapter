from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
import os
from itertools import zip_longest
from pathlib import Path

from dotenv import load_dotenv
import pytest

from llm_api_adapter.llm_registry.llm_registry import LLM_REGISTRY
from llm_api_adapter.universal_adapter import (
    ORGANIZATION_PLUGIN_DISCOVERY,
    SERVICE_PROVIDER_REGISTRY,
)
from tests.e2e import harness

_FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@dataclass(frozen=True)
class E2EOrganizationProfile:
    """The organizations selected for one independently runnable E2E lane."""

    name: str
    organization_names: tuple[str, ...]
    supported_features: frozenset[str]
    distribution: str | None = None
    api_key_is_required: bool = False
    missing_api_key_is_usage_error: bool = False
    operation_kwargs_env: tuple[tuple[str, str], ...] = ()


_PORTABLE_E2E_FEATURES = frozenset(
    {
        "text",
        "sync_chat",
        "async_chat",
        "streaming",
        "tools",
        "structured_output",
        "reasoning",
        "image_input",
        "document_input",
        "error_normalization",
    }
)
_QWEN_PORTABLE_E2E_FEATURES = frozenset(
    {
        "text",
        "sync_chat",
        "async_chat",
        "streaming",
        "tools",
        "structured_output",
        "reasoning",
        "image_input",
    }
)
_KIMI_PORTABLE_E2E_FEATURES = _QWEN_PORTABLE_E2E_FEATURES
_DEEPSEEK_PORTABLE_E2E_FEATURES = _PORTABLE_E2E_FEATURES - {"document_input"}


class E2EOrganization(dict):
    """Organization test data that never renders credentials in pytest output."""

    def __repr__(self) -> str:
        safe_data = dict(self)
        if safe_data.get("api_key"):
            safe_data["api_key"] = "***"
        if safe_data.get("operation_kwargs"):
            safe_data["operation_kwargs"] = {
                key: "***" for key in safe_data["operation_kwargs"]
            }
        return dict.__repr__(safe_data)


_OPENAI_E2E_PROFILE = E2EOrganizationProfile(
    name="openai",
    organization_names=("openai",),
    supported_features=_PORTABLE_E2E_FEATURES,
)
_ANTHROPIC_E2E_PROFILE = E2EOrganizationProfile(
    name="anthropic",
    organization_names=("anthropic",),
    supported_features=_PORTABLE_E2E_FEATURES,
)
_GOOGLE_E2E_PROFILE = E2EOrganizationProfile(
    name="google",
    organization_names=("google",),
    supported_features=_PORTABLE_E2E_FEATURES,
)
_MISTRAL_E2E_PROFILE = E2EOrganizationProfile(
    name="mistral",
    organization_names=("mistral",),
    supported_features=_PORTABLE_E2E_FEATURES | {"ocr"},
    distribution="llm-api-adapter-mistral",
    api_key_is_required=True,
)
_XAI_E2E_PROFILE = E2EOrganizationProfile(
    name="xai",
    organization_names=("xai",),
    supported_features=_PORTABLE_E2E_FEATURES,
    distribution="llm-api-adapter-xai",
    api_key_is_required=True,
    missing_api_key_is_usage_error=True,
)
_KIMI_E2E_PROFILE = E2EOrganizationProfile(
    name="kimi",
    organization_names=("kimi",),
    supported_features=_KIMI_PORTABLE_E2E_FEATURES,
    distribution="llm-api-adapter-kimi",
    api_key_is_required=True,
    missing_api_key_is_usage_error=True,
)
_QWEN_E2E_PROFILE = E2EOrganizationProfile(
    name="qwen",
    organization_names=("qwen",),
    supported_features=_QWEN_PORTABLE_E2E_FEATURES,
    distribution="llm-api-adapter-qwen",
    api_key_is_required=True,
    missing_api_key_is_usage_error=True,
    operation_kwargs_env=(("workspace_id", "QWEN_WORKSPACE_ID"),),
)
_DEEPSEEK_E2E_PROFILE = E2EOrganizationProfile(
    name="deepseek",
    organization_names=("deepseek",),
    supported_features=_DEEPSEEK_PORTABLE_E2E_FEATURES,
    distribution="llm-api-adapter-deepseek",
    api_key_is_required=True,
    missing_api_key_is_usage_error=True,
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
    pytest.param(
        _KIMI_E2E_PROFILE,
        id="kimi",
        marks=pytest.mark.e2e_kimi,
    ),
    pytest.param(
        _QWEN_E2E_PROFILE,
        id="qwen",
        marks=pytest.mark.e2e_qwen,
    ),
    pytest.param(
        _DEEPSEEK_E2E_PROFILE,
        id="deepseek",
        marks=pytest.mark.e2e_deepseek,
    ),
)

load_dotenv()

API_KEY_ENV = {
    "openai": os.getenv("OPENAI_API_KEY"),
    "anthropic": os.getenv("ANTHROPIC_API_KEY"),
    "google": os.getenv("GOOGLE_API_KEY"),
    "mistral": os.getenv("MISTRAL_API_KEY"),
    "xai": os.getenv("XAI_API_KEY"),
    "kimi": os.getenv("KIMI_API_KEY"),
    "qwen": os.getenv("QWEN_API_KEY"),
    "deepseek": os.getenv("DEEPSEEK_API_KEY"),
}


def _profile_operation_kwargs(profile: E2EOrganizationProfile) -> dict[str, str]:
    """Read declared test-only operation kwargs after ``.env`` is loaded."""
    operation_kwargs = {
        keyword: os.getenv(environment_name, "")
        for keyword, environment_name in profile.operation_kwargs_env
    }
    missing = [
        environment_name
        for keyword, environment_name in profile.operation_kwargs_env
        if not operation_kwargs[keyword]
    ]
    if missing:
        raise pytest.UsageError(
            f"{', '.join(missing)} is not configured for the {profile.name} "
            "E2E profile"
        )
    return operation_kwargs


def _profile_supports_features(
    profile: E2EOrganizationProfile,
    required_features: frozenset[str],
) -> bool:
    return required_features <= profile.supported_features


def get_e2e_organization_profile(name: str) -> E2EOrganizationProfile:
    """Return one named E2E profile for a package-local specialized check."""
    profiles = {
        profile.name: profile
        for profile in (
            _OPENAI_E2E_PROFILE,
            _ANTHROPIC_E2E_PROFILE,
            _GOOGLE_E2E_PROFILE,
            _MISTRAL_E2E_PROFILE,
            _XAI_E2E_PROFILE,
            _KIMI_E2E_PROFILE,
            _QWEN_E2E_PROFILE,
            _DEEPSEEK_E2E_PROFILE,
        )
    }
    try:
        return profiles[name]
    except KeyError as exc:
        raise pytest.UsageError(f"Unknown E2E organization profile: {name}") from exc


def pytest_collection_modifyitems(config, items) -> None:
    """Deselect profile/test combinations whose declared feature is unavailable."""
    selected = []
    deselected = []
    for item in items:
        callspec = getattr(item, "callspec", None)
        profile = (
            callspec.params.get("e2e_organization_profile")
            if callspec is not None
            else None
        )
        feature_markers = tuple(item.iter_markers("e2e_feature"))
        if not isinstance(profile, E2EOrganizationProfile) or not feature_markers:
            selected.append(item)
            continue

        required_features = frozenset(
            feature
            for marker in feature_markers
            for feature in marker.args
            if isinstance(feature, str)
        )
        if not required_features:
            raise pytest.UsageError(
                f"{item.nodeid} must declare at least one string e2e_feature"
            )
        if _profile_supports_features(profile, required_features):
            selected.append(item)
        else:
            deselected.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected


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


@pytest.fixture
def e2e_adapter():
    return harness.create_e2e_adapter


@pytest.fixture(scope="session")
def tool_choice_for_model():
    """Select the strongest registered tool-choice mode for one E2E model."""
    return harness.select_tool_choice_for_model


@pytest.fixture(scope="session")
def chat_with_retry():
    """Return the reusable synchronous transient-retry helper."""
    return harness.chat_with_transient_retry


@pytest.fixture(scope="session")
def stream_with_retry():
    """Return the reusable synchronous stream-retry helper."""
    return harness.stream_with_transient_retry


@pytest.fixture(scope="session")
def async_chat_with_retry():
    """Return the reusable asynchronous chat-retry helper."""
    return harness.async_chat_with_transient_retry


@pytest.fixture(scope="session")
def async_stream_with_retry():
    """Return the reusable asynchronous stream-retry helper."""
    return harness.async_stream_with_transient_retry


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


def resolve_e2e_organizations(e2e_organization_profile: E2EOrganizationProfile):
    """Return the organizations selected for the current E2E lane."""
    if e2e_organization_profile.distribution is not None:
        try:
            version(e2e_organization_profile.distribution)
        except PackageNotFoundError:
            pytest.skip(f"{e2e_organization_profile.distribution} is not installed")

        api_key_env_name = f"{e2e_organization_profile.name.upper()}_API_KEY"
        if (
            e2e_organization_profile.api_key_is_required
            and not API_KEY_ENV[e2e_organization_profile.name]
        ):
            if e2e_organization_profile.missing_api_key_is_usage_error:
                raise pytest.UsageError(
                    f"{api_key_env_name} is not configured for the "
                    f"{e2e_organization_profile.name} E2E profile"
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
        operation_kwargs = _profile_operation_kwargs(e2e_organization_profile)
        organizations_with_models.append(
            E2EOrganization(
                {
                    "name": organization_name,
                    "api_key": api_key,
                    "models": registry_models,
                    "latest_model": registry_models[0] if registry_models else None,
                    "operation_kwargs": operation_kwargs,
                }
            )
        )
    return organizations_with_models


@pytest.fixture(scope="session")
def organizations(e2e_organization_profile: E2EOrganizationProfile):
    return resolve_e2e_organizations(e2e_organization_profile)


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
