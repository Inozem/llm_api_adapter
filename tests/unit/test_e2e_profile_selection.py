"""Credential-free tests for model-specific E2E collection selection."""

from __future__ import annotations

from dataclasses import dataclass, replace
import importlib
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from llm_api_adapter.llm_registry.llm_registry import (
    CapabilityException,
    ModelSpec,
)
from tests.capability_scenarios import (
    E2E_SCENARIO_CAPABILITIES,
    SCENARIO_CATALOGUE,
    first_party_model_profiles,
)
from tests.capability_selection import select_model_scenarios


SYNC_CHAT = "tests/e2e/test_examples.py::test_sync_chat"
PDF_SUCCESS = "tests/e2e/test_examples.py::test_pdf_url_succeeds"
PDF_REJECTION = "tests/e2e/test_examples.py::test_pdf_url_rejected"
ERROR_NORMALIZATION = "tests/e2e/test_examples.py::test_error_normalization"
ALL_SCENARIOS = (
    SYNC_CHAT,
    PDF_SUCCESS,
    PDF_REJECTION,
    ERROR_NORMALIZATION,
)


@dataclass(frozen=True)
class _CollectedItem:
    """Minimum pytest item state consumed by the E2E collection hook."""

    nodeid: str
    organization_profile: object
    model_spec: ModelSpec

    @property
    def callspec(self):
        return SimpleNamespace(
            params={
                "e2e_organization_profile": self.organization_profile,
                "e2e_model_spec": self.model_spec,
            }
        )

    def iter_markers(self, name: str | None = None):
        return iter(())


@pytest.fixture
def e2e_collection(monkeypatch):
    """Load the real collection hook with packages, credentials, and network guarded."""
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "true")
    module = importlib.import_module("tests.e2e.conftest")
    monkeypatch.setattr(
        module,
        "API_KEY_ENV",
        {name: None for name in module.API_KEY_ENV},
    )
    package_lookup = Mock(side_effect=AssertionError("collection queried package installation"))
    plugin_discovery = Mock(side_effect=AssertionError("collection discovered provider plugins"))
    monkeypatch.setattr(module, "version", package_lookup)
    monkeypatch.setattr(module.ORGANIZATION_PLUGIN_DISCOVERY, "discover", plugin_discovery)

    def fail_network(*args, **kwargs):
        raise AssertionError("collection attempted a network request")

    import requests

    monkeypatch.setattr(requests.sessions.Session, "request", fail_network)
    return module, package_lookup, plugin_discovery


def _model(name: str, exceptions: list[dict[str, str]] | None = ()) -> ModelSpec:
    data = {
        "limits": {"context_window_tokens": 128_000, "max_output_tokens": 4_096},
        "pricing_tiers": [
            {
                "up_to_prompt_tokens": None,
                "input_per_1m": 1_000,
                "output_per_1m": 2_000,
            }
        ],
    }
    if exceptions is not None:
        data["capability_exceptions"] = exceptions
    return ModelSpec.from_dict(name, data)


def _pdf_exception(behavior_id: str) -> dict[str, str]:
    return {
        "capability_id": "pdf_url",
        "behavior_id": behavior_id,
        "behavior": "Synthetic profile for selection tests",
    }


def _items(profile, models: tuple[ModelSpec, ...]) -> list[_CollectedItem]:
    return [
        _CollectedItem(nodeid, profile, model)
        for model in models
        for nodeid in ALL_SCENARIOS
    ]


def _config():
    return SimpleNamespace(hook=SimpleNamespace(pytest_deselected=Mock()))


def _install_synthetic_selector(monkeypatch, conftest, routes_by_model):
    def select_model_scenarios(*, organization, model, **kwargs):
        if model.capability_exceptions is None:
            raise ValueError(f"{model.name} capability profile is missing")
        for exception in model.capability_exceptions:
            if not exception.behavior_id or not exception.behavior_id.islower():
                raise ValueError(f"{model.name} has invalid behavior_id")
            if exception.behavior_id == "unmapped_outcome":
                raise ValueError(
                    f"{model.name} has no evidence for {exception.capability_id}/"
                    f"{exception.behavior_id}"
                )
        try:
            return routes_by_model[(organization, model.name)]
        except KeyError as exc:
            raise ValueError(f"{model.name} has no synthetic route") from exc

    monkeypatch.setattr(
        conftest,
        "select_model_scenarios",
        select_model_scenarios,
        raising=False,
    )


def _collect(conftest, items: list[_CollectedItem]) -> list[_CollectedItem]:
    conftest.pytest_collection_modifyitems(_config(), items)
    return items


def _scenario_route_node_ids() -> tuple[str, ...]:
    routes = (
        *SCENARIO_CATALOGUE.positive,
        *SCENARIO_CATALOGUE.exceptions,
        *SCENARIO_CATALOGUE.always_on,
    )
    return tuple(dict.fromkeys(route.node_id for route in routes))


def _collected_scenario_node_ids() -> frozenset[str]:
    """Collect the static route files through pytest without running their tests."""
    repository_root = Path(__file__).resolve().parents[2]
    route_files = sorted(
        {
            node_id.partition("::")[0]
            for node_id in _scenario_route_node_ids()
            if "::" in node_id
        }
    )
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
    assert result.returncode == 0, (
        "pytest could not collect the scenario route files:\n"
        f"{result.stdout}\n{result.stderr}"
    )

    collected = set()
    for line in result.stdout.splitlines():
        path, separator, test_id = line.partition("::")
        if separator:
            collected.add(f"{path}::{test_id.partition('[')[0]}")
    return frozenset(collected)


def _route_contexts_for_profile(organization: str, model: ModelSpec, node_id: str):
    exceptions = {
        exception.capability_id: exception
        for exception in model.require_capability_profile()
    }
    contexts = []
    for capability in E2E_SCENARIO_CAPABILITIES:
        if capability.scope == "always-on":
            route = next(
                route
                for route in SCENARIO_CATALOGUE.always_on
                if route.capability_id == capability.id
            )
            if route.node_id == node_id:
                contexts.append((capability.id, "always-on", "unconditional route"))
            continue

        exception = exceptions.get(capability.id)
        if exception is not None and exception.behavior_id != "pass":
            candidates = [
                route
                for route in SCENARIO_CATALOGUE.exceptions
                if route.capability_id == capability.id
                and route.behavior_id == exception.behavior_id
                and route.organization in (organization, None)
            ]
            route = next(
                (candidate for candidate in candidates if candidate.organization == organization),
                candidates[0] if candidates else None,
            )
            if route is not None and route.node_id == node_id:
                contexts.append(
                    (capability.id, exception.behavior_id, "replacement route")
                )
            continue

        positive = next(
            route
            for route in SCENARIO_CATALOGUE.positive
            if route.capability_id == capability.id
        )
        behavior_id = exception.behavior_id if exception is not None else "baseline"
        if positive.node_id == node_id:
            contexts.append((capability.id, behavior_id, "positive route"))
    return tuple(contexts)


@pytest.mark.unit
def test_collection_selects_scenarios_per_model_within_one_organization(
    e2e_collection,
    monkeypatch,
):
    conftest, package_lookup, plugin_discovery = e2e_collection
    profile = conftest.get_e2e_organization_profile("mistral")
    baseline_model = _model("mistral-baseline-fixture", [])
    pass_model = _model(
        "mistral-small-2603",
        [_pdf_exception("pass")],
    )
    routes = {
        ("mistral", baseline_model.name): {
            SYNC_CHAT,
            PDF_SUCCESS,
            ERROR_NORMALIZATION,
        },
        ("mistral", pass_model.name): {
            SYNC_CHAT,
            PDF_SUCCESS,
            ERROR_NORMALIZATION,
        },
    }
    _install_synthetic_selector(monkeypatch, conftest, routes)
    items = _collect(conftest, _items(profile, (baseline_model, pass_model)))

    selected_by_model = {
        item.model_spec.name: {
            candidate.nodeid
            for candidate in items
            if candidate.model_spec.name == item.model_spec.name
        }
        for item in items
    }
    assert selected_by_model == {
        model.name: routes[("mistral", model.name)]
        for model in (baseline_model, pass_model)
    }
    assert all(ERROR_NORMALIZATION in selected for selected in selected_by_model.values())
    package_lookup.assert_not_called()
    plugin_discovery.assert_not_called()


@pytest.mark.unit
def test_collection_routes_a_model_deviation_to_its_declared_scenario(
    e2e_collection,
    monkeypatch,
):
    conftest, _, _ = e2e_collection
    profile = conftest.get_e2e_organization_profile("mistral")
    model = _model(
        "mistral-pdf-limited-fixture",
        [_pdf_exception("rejected_before_transport")],
    )
    routes = {
        ("mistral", model.name): {SYNC_CHAT, PDF_REJECTION, ERROR_NORMALIZATION},
    }
    _install_synthetic_selector(monkeypatch, conftest, routes)

    selected = _collect(conftest, _items(profile, (model,)))

    assert {item.nodeid for item in selected} == routes[("mistral", model.name)]
    assert PDF_SUCCESS not in {item.nodeid for item in selected}


@pytest.mark.unit
def test_collection_matches_static_routes_to_parameterized_model_node_ids(
    e2e_collection,
    monkeypatch,
):
    conftest, _, _ = e2e_collection
    profile = conftest.get_e2e_organization_profile("mistral")
    model = _model("mistral-small-2603", [])
    parameterized_item = _CollectedItem(
        f"{SYNC_CHAT}[{model.name}]",
        profile,
        model,
    )
    _install_synthetic_selector(
        monkeypatch,
        conftest,
        {("mistral", model.name): {SYNC_CHAT}},
    )

    selected = _collect(conftest, [parameterized_item])

    assert selected == [parameterized_item]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("model", "expected_detail"),
    [
        pytest.param(_model("legacy-model", None), "profile is missing", id="missing-profile"),
        pytest.param(
            replace(
                _model("malformed-model", []),
                capability_exceptions=(
                    CapabilityException(
                        capability_id="pdf_url",
                        behavior="malformed ID",
                        behavior_id="Bad-ID",
                    ),
                ),
            ),
            "invalid behavior_id",
            id="invalid-behavior-id",
        ),
        pytest.param(
            _model("unmapped-model", [_pdf_exception("unmapped_outcome")]),
            "no evidence",
            id="unmapped-behavior-id",
        ),
    ],
)
def test_collection_reports_missing_or_unmapped_model_evidence(
    e2e_collection,
    monkeypatch,
    model,
    expected_detail,
):
    conftest, _, _ = e2e_collection
    _install_synthetic_selector(monkeypatch, conftest, {})
    items = _items(conftest.get_e2e_organization_profile("mistral"), (model,))

    with pytest.raises(pytest.UsageError, match=expected_detail):
        _collect(conftest, items)


@pytest.mark.unit
def test_collection_rejects_an_invalid_organization_profile(
    e2e_collection,
):
    conftest, _, _ = e2e_collection
    item = _CollectedItem(
        SYNC_CHAT,
        object(),
        _model("mistral-small-2603", []),
    )

    with pytest.raises(pytest.UsageError, match="invalid E2E organization profile"):
        _collect(conftest, [item])


@pytest.mark.unit
def test_every_first_party_profile_selects_only_collected_scenario_nodes(
    e2e_collection,
):
    conftest, _, _ = e2e_collection
    model_profiles = first_party_model_profiles()
    profiles = {
        organization: conftest.get_e2e_organization_profile(organization)
        for organization, _ in model_profiles
    }
    collected_node_ids = _collected_scenario_node_ids()
    items = [
        _CollectedItem(node_id, profiles[organization], model)
        for organization, model in model_profiles
        for node_id in collected_node_ids
    ]

    selected_items = _collect(conftest, items)
    selected_by_model = {}
    for item in selected_items:
        key = (item.organization_profile.name, item.model_spec.name)
        selected_by_model.setdefault(key, set()).add(item.nodeid)

    gaps = []
    selected_route_owners = {}
    for organization, model in model_profiles:
        key = (organization, model.name)
        selected_routes = set(
            select_model_scenarios(
                organization=organization,
                model=model,
                capabilities=E2E_SCENARIO_CAPABILITIES,
            )
        )
        for node_id in selected_routes:
            selected_route_owners.setdefault(node_id, []).append((organization, model))
        collected_routes = selected_by_model.get(key, set())
        missing_nodes = selected_routes - collected_routes
        unexpected_nodes = collected_routes - selected_routes
        for node_id in sorted(missing_nodes):
            contexts = _route_contexts_for_profile(organization, model, node_id)
            if not contexts:
                contexts = (("unknown", "unknown", "scenario route"),)
            for capability_id, behavior_id, route_type in contexts:
                gaps.append(
                    f"{organization}/{model.name}: capability_id={capability_id}, "
                    f"behavior_id={behavior_id}, missing {route_type} collected node "
                    f"{node_id}"
                )
        for node_id in sorted(unexpected_nodes):
            gaps.append(
                f"{organization}/{model.name}: capability_id=unknown, "
                f"behavior_id=unknown, unexpected collected node {node_id}"
            )

    for node_id in sorted(selected_route_owners.keys() - collected_node_ids):
        owners = selected_route_owners[node_id]
        for organization, model in owners:
            contexts = _route_contexts_for_profile(organization, model, node_id)
            if not contexts:
                contexts = (("unknown", "unknown", "scenario route"),)
            for capability_id, behavior_id, route_type in contexts:
                gaps.append(
                    f"{organization}/{model.name}: capability_id={capability_id}, "
                    f"behavior_id={behavior_id}, selected {route_type} has no "
                    f"collected pytest node {node_id}"
                )

    assert not gaps, "\n".join(gaps)
