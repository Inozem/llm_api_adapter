"""Contract tests for exact-model E2E scenario selection."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from llm_api_adapter.llm_registry.llm_registry import ModelSpec
from llm_api_adapter.llm_registry.model_capabilities import ModelCapability
from tests.capability_scenarios import (
    BASELINE_SCENARIOS,
    EXCEPTION_SCENARIOS,
    first_party_model_profiles,
)
from tests.capability_selection import select_model_scenarios


SYNC_CHAT = "tests/e2e/test_chat.py::test_sync_chat"
PDF_SUCCESS = "tests/e2e/test_files.py::test_pdf_succeeds"
PDF_REJECTION = "tests/e2e/test_files.py::test_pdf_rejected"
ERROR_NORMALIZATION = "tests/e2e/test_errors.py::test_error_normalization"
CAPABILITIES = (
    ModelCapability("sync_chat", "model-dependent"),
    ModelCapability("pdf_url", "model-dependent"),
    ModelCapability("error_normalization", "always-on"),
)
BASELINES = {"sync_chat": SYNC_CHAT, "pdf_url": PDF_SUCCESS}
REPLACEMENTS = {
    ("pdf_url", "rejected_before_transport", None): PDF_REJECTION,
}
ALWAYS_ON = {"error_normalization": ERROR_NORMALIZATION}
FIRST_PARTY_PROFILES = first_party_model_profiles()


def _model(name: str, exceptions: Sequence[dict[str, str]] | None = ()) -> ModelSpec:
    data = {
        "limits": {"context_window_tokens": 1, "max_output_tokens": 1},
        "pricing_tiers": [
            {"up_to_prompt_tokens": None, "input_per_1m": 1, "output_per_1m": 1}
        ],
    }
    if exceptions is not None:
        data["capability_exceptions"] = list(exceptions)
    return ModelSpec.from_dict(name, data)


def _exception(behavior_id: str, behavior: str = "Provider limitation"):
    return {
        "capability_id": "pdf_url",
        "behavior_id": behavior_id,
        "behavior": behavior,
    }


def _select(
    model: ModelSpec,
    *,
    organization: str = "example",
    baselines=BASELINES,
    replacements=REPLACEMENTS,
):
    return select_model_scenarios(
        organization=organization,
        model=model,
        capabilities=CAPABILITIES,
        baseline_scenarios=baselines,
        exception_scenarios=replacements,
        always_on_scenarios=ALWAYS_ON,
    )


@pytest.mark.unit
@pytest.mark.parametrize("behavior_id", (None, "pass"))
def test_baseline_and_pass_use_the_same_shared_scenarios(behavior_id):
    exceptions = [] if behavior_id is None else [_exception(behavior_id, "PDF via OCR")]

    assert set(_select(_model("model", exceptions))) == {
        SYNC_CHAT,
        PDF_SUCCESS,
        ERROR_NORMALIZATION,
    }


@pytest.mark.unit
def test_non_pass_behavior_replaces_only_its_baseline_and_ignores_prose():
    selected = _select(
        _model(
            "limited-model",
            [_exception("rejected_before_transport", "PDF succeeds through OCR")],
        )
    )

    assert set(selected) == {SYNC_CHAT, PDF_REJECTION, ERROR_NORMALIZATION}


@pytest.mark.unit
def test_organization_route_takes_precedence_over_a_shared_replacement():
    scoped = "packages/organizations/example/tests/e2e/test_pdf.py::test_rejected"
    replacements = {
        **REPLACEMENTS,
        ("pdf_url", "rejected_before_transport", "example"): scoped,
    }
    model = _model("limited-model", [_exception("rejected_before_transport")])

    assert scoped in _select(model, replacements=replacements)
    assert PDF_REJECTION in _select(
        model,
        organization="another",
        replacements=replacements,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("model", "baselines", "replacements", "message"),
    [
        (_model("legacy", None), BASELINES, REPLACEMENTS, "profile is missing"),
        (_model("baseline-gap"), {"sync_chat": SYNC_CHAT}, REPLACEMENTS, "pdf_url"),
        (
            _model("replacement-gap", [_exception("unexpected_outcome")]),
            BASELINES,
            REPLACEMENTS,
            "unexpected_outcome",
        ),
    ],
)
def test_selection_reports_profile_and_route_gaps(
    model,
    baselines,
    replacements,
    message,
):
    with pytest.raises(ValueError, match=message):
        _select(model, baselines=baselines, replacements=replacements)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("organization", "model"),
    [
        pytest.param(organization, model, id=f"{organization}-{model.name}")
        for organization, model in FIRST_PARTY_PROFILES
    ],
)
def test_every_first_party_profile_has_complete_shared_routes(
    organization: str,
    model: ModelSpec,
):
    selected = select_model_scenarios(organization=organization, model=model)

    assert selected
    assert BASELINE_SCENARIOS["sync_chat"] in selected
    assert BASELINE_SCENARIOS["application_tools"] in selected


@pytest.mark.unit
def test_every_replacement_route_matches_a_declared_first_party_behavior():
    declared = {
        (exception.capability_id, exception.behavior_id, organization)
        for organization, model in FIRST_PARTY_PROFILES
        for exception in model.require_capability_profile()
    }

    for capability_id, behavior_id, organization in EXCEPTION_SCENARIOS:
        assert any(
            candidate_capability == capability_id
            and candidate_behavior == behavior_id
            and (organization is None or candidate_organization == organization)
            for candidate_capability, candidate_behavior, candidate_organization in declared
        )
