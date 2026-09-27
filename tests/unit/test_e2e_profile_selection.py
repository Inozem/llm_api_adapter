"""Credential-free tests for model-specific E2E collection selection."""

from __future__ import annotations

from dataclasses import dataclass, replace
import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from llm_api_adapter.llm_registry.llm_registry import (
    CapabilityException,
    ModelSpec,
)


SYNC_CHAT = "tests/e2e/test_examples.py::test_sync_chat"
PDF_SUCCESS = "tests/e2e/test_examples.py::test_pdf_url_succeeds"
PDF_REJECTION = "tests/e2e/test_examples.py::test_pdf_url_rejected"
MISTRAL_OCR = "tests/e2e/test_examples.py::test_mistral_ocr_evidence"
ERROR_NORMALIZATION = "tests/e2e/test_examples.py::test_error_normalization"
ALL_SCENARIOS = (
    SYNC_CHAT,
    PDF_SUCCESS,
    PDF_REJECTION,
    MISTRAL_OCR,
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


@pytest.mark.unit
def test_collection_selects_scenarios_per_model_within_one_organization(
    e2e_collection,
    monkeypatch,
):
    conftest, package_lookup, plugin_discovery = e2e_collection
    profile = conftest.get_e2e_organization_profile("mistral")
    baseline_model = _model("mistral-baseline-fixture", [])
    ocr_model = _model(
        "mistral-small-2603",
        [_pdf_exception("pass")],
    )
    routes = {
        ("mistral", baseline_model.name): {
            SYNC_CHAT,
            PDF_SUCCESS,
            ERROR_NORMALIZATION,
        },
        ("mistral", ocr_model.name): {
            SYNC_CHAT,
            PDF_SUCCESS,
            MISTRAL_OCR,
            ERROR_NORMALIZATION,
        },
    }
    _install_synthetic_selector(monkeypatch, conftest, routes)
    items = _collect(conftest, _items(profile, (baseline_model, ocr_model)))

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
        for model in (baseline_model, ocr_model)
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
    assert MISTRAL_OCR not in {item.nodeid for item in selected}


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
