"""Credential-free checks for model-specific E2E collection."""

from __future__ import annotations

from dataclasses import dataclass
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
    model: ModelSpec

    @property
    def callspec(self):
        return SimpleNamespace(
            params={
                "e2e_organization_profile": self.profile,
                "e2e_model_spec": self.model,
            }
        )

    def iter_markers(self, name=None):
        return iter(())


@pytest.fixture
def e2e_collection(monkeypatch):
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "true")
    module = importlib.import_module("tests.e2e.conftest")
    monkeypatch.setattr(module, "API_KEY_ENV", dict.fromkeys(module.API_KEY_ENV))
    package_lookup = Mock(side_effect=AssertionError("collection queried packages"))
    plugin_discovery = Mock(side_effect=AssertionError("collection discovered plugins"))
    monkeypatch.setattr(module, "version", package_lookup)
    monkeypatch.setattr(module.ORGANIZATION_PLUGIN_DISCOVERY, "discover", plugin_discovery)

    def fail_network(*args, **kwargs):
        raise AssertionError("collection attempted a network request")

    import requests

    monkeypatch.setattr(requests.sessions.Session, "request", fail_network)
    return module, package_lookup, plugin_discovery


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
