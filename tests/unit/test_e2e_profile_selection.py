"""Credential-free checks for model-specific E2E collection."""

from __future__ import annotations

from dataclasses import dataclass, replace
import importlib
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from llm_api_adapter.llm_registry.llm_registry import ModelSpec
from tests.capability_scenarios import (
    ALL_SCENARIO_NODE_IDS,
    E2E_SCENARIO_CAPABILITIES,
    _routes,
    first_party_model_profiles,
)
from tests.capability_selection import select_model_scenarios


SYNC_CHAT = "tests/e2e/test_examples.py::test_sync_chat"
PDF_SUCCESS = "tests/e2e/test_examples.py::test_pdf_succeeds"
PDF_REJECTION = "tests/e2e/test_examples.py::test_pdf_rejected"
ERROR_NORMALIZATION = "tests/e2e/test_examples.py::test_error_normalization"
ASYNC_STRUCTURED_CHAT = (
    "tests/e2e/test_async.py::"
    "test_async_chat_returns_structured_response_and_pricing"
)
SYNTHETIC_NODES = (SYNC_CHAT, PDF_SUCCESS, PDF_REJECTION, ERROR_NORMALIZATION)


@dataclass(frozen=True)
class _Item:
    nodeid: str
    profile: object
    model: ModelSpec | None
    markers: tuple[object, ...] = ()
    model_case: object | None = None

    @property
    def callspec(self):
        return SimpleNamespace(
            params={
                "e2e_organization_profile": self.profile,
                **({"e2e_model_spec": self.model} if self.model is not None else {}),
                **(
                    {"e2e_model_case": self.model_case}
                    if self.model_case is not None
                    else {}
                ),
            }
        )

    def iter_markers(self, name=None):
        return iter(
            marker
            for marker in self.markers
            if name is None or marker.name == name
        )

    @property
    def keywords(self):
        return {}


@pytest.fixture
def e2e_collection(monkeypatch):
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "true")
    module = importlib.import_module("tests.e2e.conftest")
    monkeypatch.setattr(module, "API_KEY_ENV", dict.fromkeys(module.API_KEY_ENV))
    package_lookup = Mock(side_effect=AssertionError("collection queried packages"))
    plugin_discovery = Mock(side_effect=AssertionError("collection discovered plugins"))
    monkeypatch.setattr(module, "version", package_lookup)
    monkeypatch.setattr(
        module.ORGANIZATION_PLUGIN_DISCOVERY,
        "discover",
        plugin_discovery,
    )

    def fail_network(*args, **kwargs):
        raise AssertionError("collection attempted a network request")

    import requests

    monkeypatch.setattr(requests.sessions.Session, "request", fail_network)
    yield module, package_lookup, plugin_discovery
    module.e2e_model_case_parameters.cache_clear()


def _model(name: str) -> ModelSpec:
    return ModelSpec.from_dict(
        name,
        {
            "limits": {"context_window_tokens": 1, "max_output_tokens": 1},
            "pricing_tiers": [
                {"up_to_prompt_tokens": None, "input_per_1m": 1, "output_per_1m": 1}
            ],
            "capability_exceptions": [],
        },
    )


def _collect(module, items):
    config = SimpleNamespace(hook=SimpleNamespace(pytest_deselected=Mock()))
    module.pytest_collection_modifyitems(config, items)
    return items


@pytest.mark.unit
def test_collection_keeps_only_each_models_selected_routes(
    e2e_collection,
    monkeypatch,
):
    module, package_lookup, plugin_discovery = e2e_collection
    profile = module.get_e2e_organization_profile("mistral")
    baseline = _model("baseline")
    limited = _model("limited")
    routes = {
        baseline.name: {SYNC_CHAT, PDF_SUCCESS, ERROR_NORMALIZATION},
        limited.name: {SYNC_CHAT, PDF_REJECTION, ERROR_NORMALIZATION},
    }
    monkeypatch.setattr(
        module,
        "select_model_scenarios",
        lambda *, model, **kwargs: routes[model.name],
    )
    items = [
        _Item(node_id, profile, model)
        for model in (baseline, limited)
        for node_id in SYNTHETIC_NODES
    ]

    selected = _collect(module, items)

    assert {
        model.name: {item.nodeid for item in selected if item.model == model}
        for model in (baseline, limited)
    } == routes
    package_lookup.assert_not_called()
    plugin_discovery.assert_not_called()


@pytest.mark.unit
def test_collection_matches_parameterized_items_to_static_node_ids(
    e2e_collection,
    monkeypatch,
):
    module, _, _ = e2e_collection
    profile = module.get_e2e_organization_profile("mistral")
    model = _model("model")
    item = _Item(f"{SYNC_CHAT}[{model.name}]", profile, model)
    monkeypatch.setattr(
        module,
        "select_model_scenarios",
        lambda **kwargs: (SYNC_CHAT,),
    )

    assert _collect(module, [item]) == [item]


@pytest.mark.unit
def test_collection_reports_selector_errors(e2e_collection, monkeypatch):
    module, _, _ = e2e_collection
    profile = module.get_e2e_organization_profile("mistral")

    def fail_selection(**kwargs):
        raise ValueError("capability_id=pdf_url, behavior_id=unsupported")

    monkeypatch.setattr(module, "select_model_scenarios", fail_selection)

    with pytest.raises(pytest.UsageError, match="behavior_id=unsupported"):
        _collect(module, [_Item(PDF_SUCCESS, profile, _model("model"))])


@pytest.mark.unit
def test_duplicate_static_route_evidence_fails_deterministically():
    with pytest.raises(ValueError, match="duplicate scenario route"):
        _routes(("sync_chat", SYNC_CHAT), ("sync_chat", PDF_SUCCESS))


@pytest.mark.unit
def test_missing_static_route_evidence_fails_deterministically():
    capability = next(
        capability
        for capability in E2E_SCENARIO_CAPABILITIES
        if capability.scope == "model-dependent"
    )

    with pytest.raises(ValueError, match=f"capability_id={capability.id}"):
        select_model_scenarios(
            organization="mistral",
            model=_model("mistral-small"),
            capabilities=(capability,),
            baseline_scenarios={},
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "markers",
    [
        (),
        (
            pytest.mark.e2e_capability("sync_chat"),
            pytest.mark.e2e_capability("sync_chat"),
        ),
    ],
    ids=["missing", "duplicate"],
)
def test_model_case_collection_rejects_missing_or_duplicate_capability_markers(
    e2e_collection,
    markers,
):
    module, _, _ = e2e_collection
    profile = module.get_e2e_organization_profile("mistral")
    case = module.E2EModelCase(
        organization="mistral",
        model_spec=_model("mistral-small"),
        profile=profile,
    )
    item = _Item(SYNC_CHAT, profile, case.model_spec, markers=markers, model_case=case)

    expected = (
        "must declare known e2e_capability IDs"
        if not markers
        else "duplicate e2e_capability IDs"
    )
    with pytest.raises(pytest.UsageError, match=expected):
        _collect(module, [item])


@pytest.mark.unit
def test_hosted_flash_uses_mistral_lane_and_developer_mapping(e2e_collection, monkeypatch):
    module, package_lookup, _ = e2e_collection
    profile = module.get_e2e_organization_profile("mistral")
    flash = _model("glm-5.3-flash")
    package_lookup.side_effect = None
    package_lookup.return_value = "1.0"
    monkeypatch.setattr(module, "_E2E_PROFILES", (profile,))
    monkeypatch.setattr(module.ORGANIZATION_PLUGIN_DISCOVERY, "discover", Mock())
    module.e2e_model_case_parameters.cache_clear()
    monkeypatch.setattr(
        module,
        "LLM_REGISTRY",
        SimpleNamespace(
            organizations={"mistral": SimpleNamespace(models={flash.name: flash})}
        ),
    )

    parameters = module.e2e_model_case_parameters(service_provider="mistral")

    hosted = next(
        parameter.values[0]
        for parameter in parameters
        if "glm-5.3-flash" in parameter.id
    )
    assert hosted.organization == "zai"
    assert hosted.service_provider == "mistral"
    assert hosted.profile is profile
    assert profile.organization_names == ("mistral",)
    assert (
        profile.distribution
        == module.KNOWN_ORGANIZATION_PACKAGES["mistral"].distribution
    )
    assert profile.api_key_is_required is True
    assert ("glm-5.3-flash", "zai") in profile.model_organizations
    assert "mistral-glm-5.3-flash" in {parameter.id for parameter in parameters}
    assert not any(parameter.id.startswith("zai-") for parameter in parameters)
    package_lookup.assert_called_once_with(profile.distribution)

    package_lookup.side_effect = None
    package_lookup.return_value = "1.0"
    monkeypatch.setattr(
        module,
        "API_KEY_ENV",
        {"mistral": "mistral-test-key", "zai": None},
    )
    monkeypatch.setattr(
        module,
        "LLM_REGISTRY",
        SimpleNamespace(
            organizations={"mistral": SimpleNamespace(models={flash.name: flash})}
        ),
    )
    organizations = module.resolve_e2e_organizations(profile)
    assert [organization["name"] for organization in organizations] == ["mistral"]
    assert organizations[0]["api_key"] == "mistral-test-key"
    assert package_lookup.call_args_list[-1].args == (profile.distribution,)
    monkeypatch.setattr(module, "resolve_e2e_organizations", lambda _: organizations)
    assert module.e2e_model_organization.__wrapped__(hosted) is organizations[0]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("mappings", "expected_error"),
    [
        ((("not-registered", "zai"),), "unregistered provider model"),
        ((("registered", ""),), "non-empty string"),
        (
            (("registered", "zai"), ("registered", "other")),
            "Duplicate model organization mapping",
        ),
    ],
    ids=("unregistered-model", "empty-developer", "duplicate-model"),
)
def test_provider_model_mappings_are_validated_offline(
    e2e_collection,
    monkeypatch,
    mappings,
    expected_error,
):
    module, package_lookup, _ = e2e_collection
    package_lookup.side_effect = None
    package_lookup.return_value = "1.0"
    profile = replace(
        module.get_e2e_organization_profile("mistral"),
        model_organizations=mappings,
    )
    registered = _model("registered")
    flash = _model("glm-5.3-flash")
    monkeypatch.setattr(module.ORGANIZATION_PLUGIN_DISCOVERY, "discover", Mock())
    monkeypatch.setattr(module, "_E2E_PROFILES", (profile,))
    monkeypatch.setattr(
        module,
        "LLM_REGISTRY",
        SimpleNamespace(
            organizations={
                "mistral": SimpleNamespace(
                    models={
                        flash.name: flash,
                        registered.name: registered,
                    }
                )
            }
        ),
    )
    module.e2e_model_case_parameters.cache_clear()

    with pytest.raises(pytest.UsageError, match=expected_error):
        module.e2e_model_case_parameters(service_provider="mistral")


@pytest.mark.unit
def test_missing_provider_package_keeps_mapped_model_skip_placeholder(
    e2e_collection,
    monkeypatch,
):
    module, package_lookup, _ = e2e_collection
    profile = module.get_e2e_organization_profile("mistral")
    monkeypatch.setattr(module.ORGANIZATION_PLUGIN_DISCOVERY, "discover", Mock())
    monkeypatch.setattr(module, "_E2E_PROFILES", (profile,))
    monkeypatch.setattr(
        module,
        "LLM_REGISTRY",
        SimpleNamespace(organizations={}),
    )
    package_lookup.side_effect = module.PackageNotFoundError(profile.distribution)
    module.e2e_model_case_parameters.cache_clear()

    parameters = module.e2e_model_case_parameters(service_provider="mistral")

    assert len(parameters) == 1
    assert parameters[0].id == "mistral-package-not-installed"
    case = parameters[0].values[0]
    assert case.model_spec is None
    assert case.service_provider == "mistral"
    assert any(mark.name == "skip" for mark in parameters[0].marks)


@pytest.mark.unit
def test_direct_zai_enumeration_and_tool_choice_keep_zai_provider_by_default(
    e2e_collection,
    monkeypatch,
):
    module, package_lookup, _ = e2e_collection
    package_lookup.side_effect = None
    package_lookup.return_value = "1.0"
    zai_profile = module.get_e2e_organization_profile("zai")
    model = _model("glm-5.3-flash")
    monkeypatch.setattr(module, "_E2E_PROFILES", (zai_profile,))
    monkeypatch.setattr(module.ORGANIZATION_PLUGIN_DISCOVERY, "discover", Mock())
    module.e2e_model_case_parameters.cache_clear()
    monkeypatch.setattr(
        module,
        "LLM_REGISTRY",
        SimpleNamespace(
            organizations={"zai": SimpleNamespace(models={model.name: model})}
        ),
    )

    direct = module.e2e_model_case_parameters("zai")

    case = direct[0].values[0]
    assert case.organization == "zai"
    assert case.service_provider is None
    assert case.profile is zai_profile
    assert direct[0].id == "zai-glm-5.3-flash"

    harness = importlib.import_module("tests.e2e.harness")
    zai_model = SimpleNamespace(
        name=model.name,
        request_rules=SimpleNamespace(
            allowed_tool_choice_modes=frozenset({"auto"})
        ),
    )
    mistral_model = SimpleNamespace(
        name=model.name,
        request_rules=SimpleNamespace(allowed_tool_choice_modes=None),
    )
    monkeypatch.setattr(
        harness,
        "LLM_REGISTRY",
        SimpleNamespace(
            organizations={
                "zai": SimpleNamespace(models={model.name: zai_model}),
                "mistral": SimpleNamespace(models={model.name: mistral_model}),
            }
        ),
    )
    assert (
        harness.select_tool_choice_for_model("zai", model.name, "lookup")
        == "auto"
    )
    assert (
        harness.select_tool_choice_for_model(
            "zai", model.name, "lookup", service_provider="mistral"
        )
        == "lookup"
    )


@pytest.mark.unit
def test_hosted_adapter_uses_developer_identity_and_provider_profile(monkeypatch):
    harness = importlib.import_module("tests.e2e.harness")
    constructor = Mock()
    monkeypatch.setattr(harness, "UniversalLLMAPIAdapter", constructor)
    profile = {
        "name": "mistral",
        "api_key": "mistral-test-key",
        "operation_kwargs": {"max_retries": 3},
    }

    wrapped = harness.create_e2e_adapter(
        profile,
        "glm-5.3-flash",
        organization_name="zai",
        service_provider="mistral",
    )

    constructor.assert_called_once_with(
        organization="zai",
        service_provider="mistral",
        model="glm-5.3-flash",
        api_key="mistral-test-key",
        transport="requests",
    )
    assert wrapped._operation_kwargs == {"max_retries": 3}

    constructor.reset_mock()
    harness.create_e2e_adapter(
        organization=profile,
        model="glm-5.3-flash",
    )
    constructor.assert_called_once_with(
        organization="mistral",
        model="glm-5.3-flash",
        api_key="mistral-test-key",
        transport="requests",
    )


@pytest.mark.unit
def test_same_developer_model_selects_routes_separately_by_provider(
    e2e_collection,
    monkeypatch,
):
    module, _, _ = e2e_collection
    mistral_profile = module.get_e2e_organization_profile("mistral")
    zai_profile = module.get_e2e_organization_profile("zai")
    model = _model("glm-5.3-flash")
    capability = E2E_SCENARIO_CAPABILITIES[0]
    marker = pytest.mark.e2e_capability(capability.id)
    hosted = module.E2EModelCase(
        organization="zai",
        model_spec=model,
        profile=mistral_profile,
        service_provider="mistral",
    )
    direct = module.E2EModelCase(
        organization="zai",
        model_spec=model,
        profile=zai_profile,
        service_provider=None,
    )
    items = [
        _Item(
            f"{SYNC_CHAT}[mistral-glm-5.3-flash]",
            mistral_profile,
            model,
            (marker,),
            hosted,
        ),
        _Item(
            f"{SYNC_CHAT}[zai-glm-5.3-flash]",
            zai_profile,
            model,
            (marker,),
            direct,
        ),
    ]
    selections = []

    def select(*, organization, model, capabilities=None, service_provider=None):
        selections.append((organization, model.name, service_provider))
        return (SYNC_CHAT,)

    monkeypatch.setattr(module, "select_model_scenarios", select)

    selected = _collect(module, items)

    assert selected == items
    assert selections == [
        ("zai", "glm-5.3-flash", "mistral"),
        ("zai", "glm-5.3-flash", None),
    ]


@pytest.mark.unit
def test_native_model_case_calls_legacy_selector_without_provider_keyword(
    e2e_collection,
    monkeypatch,
):
    module, _, _ = e2e_collection
    profile = module.get_e2e_organization_profile("zai")
    model = _model("glm-5.3-flash")
    case = module.E2EModelCase(
        organization="zai",
        model_spec=model,
        profile=profile,
    )
    item = _Item(
        f"{SYNC_CHAT}[zai-{model.name}]",
        profile,
        model,
        markers=(pytest.mark.e2e_capability(E2E_SCENARIO_CAPABILITIES[0].id),),
        model_case=case,
    )
    calls = []

    def select(*, organization, model, capabilities):
        calls.append((organization, model.name, capabilities))
        return (SYNC_CHAT,)

    monkeypatch.setattr(module, "select_model_scenarios", select)

    assert _collect(module, [item]) == [item]
    assert calls == [("zai", model.name, (E2E_SCENARIO_CAPABILITIES[0],))]


@pytest.mark.unit
def test_collection_rejects_an_invalid_organization_profile(e2e_collection):
    module, _, _ = e2e_collection

    with pytest.raises(pytest.UsageError, match="invalid E2E organization profile"):
        _collect(module, [_Item(SYNC_CHAT, object(), _model("model"))])


@pytest.mark.unit
def test_every_first_party_route_names_a_collected_pytest_node():
    repository_root = Path(__file__).resolve().parents[2]
    route_files = sorted({node.partition("::")[0] for node in ALL_SCENARIO_NODE_IDS})
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "--import-mode=importlib",
            "-q",
            *route_files,
        ],
        cwd=repository_root,
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    collection_lines = result.stdout.splitlines()
    assert any(line.startswith(f"{ASYNC_STRUCTURED_CHAT}[") for line in collection_lines)
    assert not any(
        line.startswith(f"{ASYNC_STRUCTURED_CHAT}[zai-")
        and not line.startswith(
            f"{ASYNC_STRUCTURED_CHAT}[zai-package-not-installed]"
        )
        for line in collection_lines
    )
    collected = {
        f"{path}::{test_id.partition('[')[0]}"
        for line in collection_lines
        for path, separator, test_id in (line.partition("::"),)
        if separator
    }

    missing = []
    for organization, model in first_party_model_profiles():
        exceptions = {
            item.capability_id: item for item in model.require_capability_profile()
        }
        for capability in E2E_SCENARIO_CAPABILITIES:
            behavior_id = (
                exceptions[capability.id].behavior_id
                if capability.id in exceptions
                else "always-on"
                if capability.scope == "always-on"
                else "baseline"
            )
            for node_id in select_model_scenarios(
                organization=organization,
                model=model,
                capabilities=(capability,),
            ):
                if node_id not in collected:
                    missing.append(
                        f"{organization}/{model.name}: capability_id={capability.id}, "
                        f"behavior_id={behavior_id}, node_id={node_id}"
                    )

    assert not missing, "\n".join(missing)
