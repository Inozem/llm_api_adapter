from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
import os
from itertools import zip_longest
from pathlib import Path

from dotenv import load_dotenv
import pytest

from llm_api_adapter.llm_registry.llm_registry import LLM_REGISTRY, ModelSpec
from llm_api_adapter.universal_adapter import (
    ORGANIZATION_PLUGIN_DISCOVERY,
    SERVICE_PROVIDER_REGISTRY,
)
from tests.capability_selection import select_model_scenarios
from tests.capability_scenarios import E2E_SCENARIO_CAPABILITIES
from tests.e2e import harness

_FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@dataclass(frozen=True)
class E2EOrganizationProfile:
    """The organizations selected for one independently runnable E2E lane."""

    name: str
    organization_names: tuple[str, ...]
    distribution: str | None = None
    api_key_is_required: bool = False
    missing_api_key_is_usage_error: bool = False
    operation_kwargs_env: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class E2EModelCase:
    """One organization and its exact registered model for scenario selection."""

    organization: str
    model_spec: ModelSpec | None
    profile: E2EOrganizationProfile


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
    distribution="llm-api-adapter-mistral",
    api_key_is_required=True,
)
_XAI_E2E_PROFILE = E2EOrganizationProfile(
    name="xai",
    organization_names=("xai",),
    distribution="llm-api-adapter-xai",
    api_key_is_required=True,
    missing_api_key_is_usage_error=True,
)
_KIMI_E2E_PROFILE = E2EOrganizationProfile(
    name="kimi",
    organization_names=("kimi",),
    distribution="llm-api-adapter-kimi",
    api_key_is_required=True,
    missing_api_key_is_usage_error=True,
)
_QWEN_E2E_PROFILE = E2EOrganizationProfile(
    name="qwen",
    organization_names=("qwen",),
    distribution="llm-api-adapter-qwen",
    api_key_is_required=True,
    missing_api_key_is_usage_error=True,
    operation_kwargs_env=(("workspace_id", "QWEN_WORKSPACE_ID"),),
)
_DEEPSEEK_E2E_PROFILE = E2EOrganizationProfile(
    name="deepseek",
    organization_names=("deepseek",),
    distribution="llm-api-adapter-deepseek",
    api_key_is_required=True,
    missing_api_key_is_usage_error=True,
)
_ZAI_E2E_PROFILE = E2EOrganizationProfile(
    name="zai",
    organization_names=("zai",),
    distribution="llm-api-adapter-zai",
    api_key_is_required=True,
    missing_api_key_is_usage_error=True,
)
_E2E_PROFILES = (
    _OPENAI_E2E_PROFILE,
    _ANTHROPIC_E2E_PROFILE,
    _GOOGLE_E2E_PROFILE,
    _MISTRAL_E2E_PROFILE,
    _XAI_E2E_PROFILE,
    _KIMI_E2E_PROFILE,
    _QWEN_E2E_PROFILE,
    _DEEPSEEK_E2E_PROFILE,
    _ZAI_E2E_PROFILE,
)
_E2E_PROFILE_MARKS = {
    "openai": (pytest.mark.e2e_builtin, pytest.mark.e2e_openai),
    "anthropic": (pytest.mark.e2e_builtin, pytest.mark.e2e_anthropic),
    "google": (pytest.mark.e2e_builtin, pytest.mark.e2e_google),
    "mistral": (pytest.mark.e2e_mistral,),
    "xai": (pytest.mark.e2e_xai,),
    "kimi": (pytest.mark.e2e_kimi,),
    "qwen": (pytest.mark.e2e_qwen,),
    "deepseek": (pytest.mark.e2e_deepseek,),
    "zai": (pytest.mark.e2e_zai,),
}
_E2E_PROFILE_PARAMS = tuple(
    pytest.param(
        profile,
        id=profile.name,
        marks=_E2E_PROFILE_MARKS[profile.name],
    )
    for profile in _E2E_PROFILES
)
_CAPABILITY_BY_ID = {
    capability.id: capability for capability in E2E_SCENARIO_CAPABILITIES
}

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
    "zai": os.getenv("ZAI_API_KEY"),
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
            _ZAI_E2E_PROFILE,
        )
    }
    try:
        return profiles[name]
    except KeyError as exc:
        raise pytest.UsageError(f"Unknown E2E organization profile: {name}") from exc


@lru_cache(maxsize=None)
def e2e_model_case_parameters(organization_name: str | None = None):
    """Build exact-model pytest parameters, optionally for one organization."""
    ORGANIZATION_PLUGIN_DISCOVERY.discover(
        SERVICE_PROVIDER_REGISTRY,
        model_registry=LLM_REGISTRY,
    )

    parameters = []
    for profile in _E2E_PROFILES:
        if organization_name is not None and organization_name not in profile.organization_names:
            continue
        package_missing = False
        if profile.distribution is not None:
            try:
                version(profile.distribution)
            except PackageNotFoundError:
                package_missing = True

        for organization in profile.organization_names:
            if organization_name is not None and organization != organization_name:
                continue
            if package_missing:
                case = E2EModelCase(
                    organization=organization,
                    model_spec=None,
                    profile=profile,
                )
                parameters.append(
                    pytest.param(
                        case,
                        id=f"{organization}-package-not-installed",
                        marks=(
                            *_E2E_PROFILE_MARKS[profile.name],
                            pytest.mark.skip(
                                reason=f"{profile.distribution} is not installed"
                            ),
                        ),
                    )
                )
                continue

            organization_spec = LLM_REGISTRY.organizations.get(organization)
            if organization_spec is None:
                raise pytest.UsageError(
                    f"No valid model catalogue was registered for {organization}"
                )
            for model_spec in organization_spec.models.values():
                case = E2EModelCase(
                    organization=organization,
                    model_spec=model_spec,
                    profile=profile,
                )
                parameters.append(
                    pytest.param(
                        case,
                        id=f"{organization}-{model_spec.name}",
                        marks=_E2E_PROFILE_MARKS[profile.name],
                    )
                )

    if not parameters:
        raise pytest.UsageError(
            f"No E2E model cases were available for {organization_name or 'collection'}"
        )
    return tuple(parameters)


def pytest_collection_modifyitems(config, items) -> None:
    """Keep only scenario routes selected by each exact model profile."""
    selected = []
    deselected = []
    routes_by_model: dict[tuple[object, ...], frozenset[str]] = {}

    for item in items:
        callspec = getattr(item, "callspec", None)
        params = callspec.params if callspec is not None else {}
        model_case = params.get("e2e_model_case")
        profile = params.get("e2e_organization_profile")
        model_spec = params.get("e2e_model_spec")

        if "e2e_model_case" in params:
            if not isinstance(model_case, E2EModelCase):
                raise pytest.UsageError(
                    f"{item.nodeid} has a missing or invalid exact-model case"
                )
            if model_case.profile not in _E2E_PROFILES:
                raise pytest.UsageError(
                    f"{item.nodeid} has an unknown E2E organization profile"
                )
            if model_case.organization not in model_case.profile.organization_names:
                raise pytest.UsageError(
                    f"{item.nodeid} has an organization outside its E2E lane"
                )
            if "skip" in item.keywords and model_case.model_spec is None:
                selected.append(item)
                continue
            if not isinstance(model_case.model_spec, ModelSpec):
                raise pytest.UsageError(
                    f"{item.nodeid} has a missing or invalid exact-model profile"
                )

            capability_markers = tuple(item.iter_markers("e2e_capability"))
            capability_ids = tuple(
                capability_id
                for marker in capability_markers
                for capability_id in marker.args
            )
            if not capability_ids or any(
                not isinstance(capability_id, str)
                or capability_id not in _CAPABILITY_BY_ID
                for capability_id in capability_ids
            ):
                raise pytest.UsageError(
                    f"{item.nodeid} must declare known e2e_capability IDs"
                )
            if len(set(capability_ids)) != len(capability_ids):
                raise pytest.UsageError(
                    f"{item.nodeid} declares duplicate e2e_capability IDs"
                )

            route_key = (
                model_case.organization,
                model_case.model_spec.name,
                capability_ids,
            )
            if route_key not in routes_by_model:
                try:
                    routes_by_model[route_key] = frozenset(
                        select_model_scenarios(
                            organization=model_case.organization,
                            model=model_case.model_spec,
                            capabilities=tuple(
                                _CAPABILITY_BY_ID[capability_id]
                                for capability_id in capability_ids
                            ),
                        )
                    )
                except (TypeError, ValueError) as exc:
                    raise pytest.UsageError(
                        "Invalid capability profile for "
                        f"{model_case.organization}/{model_case.model_spec.name}: {exc}"
                    ) from exc

            if _base_pytest_node_id(item.nodeid) in routes_by_model[route_key]:
                selected.append(item)
            else:
                deselected.append(item)
            continue

        if "e2e_organization_profile" in params and not isinstance(
            profile, E2EOrganizationProfile
        ):
            raise pytest.UsageError(
                f"{item.nodeid} has a missing or invalid E2E organization profile"
            )

        if "e2e_model_spec" not in params:
            selected.append(item)
            continue

        if not isinstance(profile, E2EOrganizationProfile):
            raise pytest.UsageError(
                f"{item.nodeid} has an exact model but no valid E2E organization profile"
            )

        if not isinstance(model_spec, ModelSpec):
            raise pytest.UsageError(
                f"{item.nodeid} has a missing or invalid exact-model profile"
            )

        route_key = (profile.name, model_spec.name)
        if route_key not in routes_by_model:
            try:
                routes_by_model[route_key] = frozenset(
                    select_model_scenarios(
                        organization=profile.name,
                        model=model_spec,
                    )
                )
            except (TypeError, ValueError) as exc:
                raise pytest.UsageError(
                    f"Invalid capability profile for {profile.name}/{model_spec.name}: "
                    f"{exc}"
                ) from exc

        if _base_pytest_node_id(item.nodeid) in routes_by_model[route_key]:
            selected.append(item)
        else:
            deselected.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected


def _base_pytest_node_id(nodeid: str) -> str:
    """Drop parametrization IDs so static scenario routes match model variants."""
    parent, separator, test_name = nodeid.rpartition("::")
    if not separator:
        return nodeid
    test_name = test_name.partition("[")[0]
    return f"{parent}::{test_name}"


@pytest.fixture
def e2e_model_organization(e2e_model_case: E2EModelCase) -> E2EOrganization:
    """Resolve the provider lane for one exact-model E2E case."""
    if not isinstance(e2e_model_case, E2EModelCase) or not isinstance(
        e2e_model_case.model_spec,
        ModelSpec,
    ):
        raise pytest.UsageError("The E2E case has no valid exact-model profile")
    organizations = resolve_e2e_organizations(e2e_model_case.profile)
    for organization in organizations:
        if organization["name"] == e2e_model_case.organization:
            return organization
    raise pytest.UsageError(
        f"No E2E organization data was prepared for {e2e_model_case.organization}"
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
