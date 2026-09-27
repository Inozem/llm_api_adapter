"""Contract tests for exact-model capability scenario selection.

The synthetic catalogue keeps routing rules separate from live E2E collection.
T021 supplies the real scenario inventory; T022 implements the selector.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import pytest

from llm_api_adapter.llm_registry.llm_registry import ModelSpec
from llm_api_adapter.llm_registry.model_capabilities import ModelCapability
from tests.capability_scenarios import (
    CapabilityScenario,
    E2E_SCENARIO_CAPABILITIES,
    ExceptionScenario,
    SCENARIO_CATALOGUE,
    ScenarioCatalogue,
    first_party_model_profiles,
)


SYNC_CHAT = "tests/e2e/test_llm_adapter_chat.py::test_sync_chat"
PDF_SUCCESS = "tests/e2e/test_file_uploads.py::test_pdf_url_succeeds"
PDF_REJECTION = "tests/e2e/test_file_uploads.py::test_pdf_url_rejected"
ERROR_NORMALIZATION = "tests/e2e/test_errors.py::test_error_normalization"
FIRST_PARTY_MODEL_PROFILES = first_party_model_profiles()

CAPABILITIES = (
    ModelCapability("sync_chat", "model-dependent"),
    ModelCapability("pdf_url", "model-dependent"),
    ModelCapability("error_normalization", "always-on"),
)


def _scenario_catalogue() -> ScenarioCatalogue:
    """Small route inventory; tuple entries preserve duplicates for validation."""
    return ScenarioCatalogue(
        positive=(
            CapabilityScenario("sync_chat", SYNC_CHAT),
            CapabilityScenario("pdf_url", PDF_SUCCESS),
        ),
        exceptions=(
            ExceptionScenario("pdf_url", "rejected_before_transport", PDF_REJECTION),
        ),
        always_on=(CapabilityScenario("error_normalization", ERROR_NORMALIZATION),),
    )


def _model(name: str, exceptions: Sequence[dict[str, str]] | None = ()) -> ModelSpec:
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
        data["capability_exceptions"] = list(exceptions)
    return ModelSpec.from_dict(name, data)


def _exception(behavior_id: str, behavior: str = "Provider limitation") -> dict[str, str]:
    return {
        "capability_id": "pdf_url",
        "behavior_id": behavior_id,
        "behavior": behavior,
    }


def _select(
    organization: str,
    model: ModelSpec,
    catalogue: ScenarioCatalogue | None = None,
) -> tuple[str, ...]:
    # Import inside the helper so pytest collects every red contract case even
    # before the selector module exists.
    from tests.capability_selection import select_model_scenarios

    return select_model_scenarios(
        organization=organization,
        model=model,
        scenarios=catalogue or _scenario_catalogue(),
        capabilities=CAPABILITIES,
    )


def _assert_selected(actual: tuple[str, ...], expected: set[str]) -> None:
    assert set(actual) == expected
    assert len(actual) == len(expected)


@pytest.mark.unit
def test_empty_profile_selects_every_applicable_positive_and_always_on_scenario():
    selected = _select("mistral", _model("mistral-small-2603"))

    _assert_selected(selected, {SYNC_CHAT, PDF_SUCCESS, ERROR_NORMALIZATION})


@pytest.mark.unit
def test_mistral_pdf_pass_uses_the_shared_success_scenario():
    model = _model("mistral-small-2603", [_exception("pass", "PDF via OCR")])

    selected = _select("mistral", model)

    _assert_selected(selected, {SYNC_CHAT, PDF_SUCCESS, ERROR_NORMALIZATION})
    assert PDF_REJECTION not in selected


@pytest.mark.unit
def test_pdf_rejection_replaces_only_pdf_success_and_never_interprets_prose():
    model = _model(
        "pdf-limited-model",
        [_exception("rejected_before_transport", "PDF via OCR succeeds")],
    )

    selected = _select("example", model)

    _assert_selected(selected, {SYNC_CHAT, PDF_REJECTION, ERROR_NORMALIZATION})
    assert PDF_SUCCESS not in selected


@pytest.mark.unit
def test_replacement_routes_can_be_scoped_to_an_organization():
    zai_route = "packages/organizations/zai/tests/e2e/test_zai.py::test_pdf_rejected"
    kimi_route = "packages/organizations/kimi/tests/e2e/test_kimi.py::test_pdf_rejected"
    catalogue = replace(
        _scenario_catalogue(),
        exceptions=(
            ExceptionScenario(
                "pdf_url",
                "rejected_before_transport",
                zai_route,
                organization="zai",
            ),
            ExceptionScenario(
                "pdf_url",
                "rejected_before_transport",
                kimi_route,
                organization="kimi",
            ),
        ),
    )
    model = _model(
        "pdf-limited-model",
        [_exception("rejected_before_transport")],
    )

    _assert_selected(
        _select("zai", model, catalogue),
        {SYNC_CHAT, zai_route, ERROR_NORMALIZATION},
    )
    _assert_selected(
        _select("kimi", model, catalogue),
        {SYNC_CHAT, kimi_route, ERROR_NORMALIZATION},
    )

    with pytest.raises(ValueError) as error:
        _select("deepseek", model, catalogue)

    message = str(error.value)
    assert "pdf-limited-model" in message
    assert "pdf_url" in message
    assert "rejected_before_transport" in message


@pytest.mark.unit
def test_legacy_model_without_profile_cannot_be_certified():
    model = _model("legacy-model", None)

    with pytest.raises(ValueError) as error:
        _select("example", model)

    assert "legacy-model" in str(error.value)
    assert "profile" in str(error.value)


@pytest.mark.unit
def test_unknown_behavior_pair_reports_model_capability_and_behavior_id():
    model = _model("odd-model", [_exception("unexpected_outcome")])

    with pytest.raises(ValueError) as error:
        _select("example", model)

    message = str(error.value)
    assert "odd-model" in message
    assert "pdf_url" in message
    assert "unexpected_outcome" in message


@pytest.mark.unit
def test_missing_positive_scenario_is_a_coverage_error():
    catalogue = replace(
        _scenario_catalogue(),
        positive=(CapabilityScenario("sync_chat", SYNC_CHAT),),
    )

    with pytest.raises(ValueError) as error:
        _select("example", _model("complete-model"), catalogue)

    assert "complete-model" in str(error.value)
    assert "pdf_url" in str(error.value)


@pytest.mark.unit
def test_missing_replacement_scenario_is_a_coverage_error():
    model = _model("pdf-limited-model", [_exception("rejected_before_transport")])
    catalogue = replace(_scenario_catalogue(), exceptions=())

    with pytest.raises(ValueError) as error:
        _select("example", model, catalogue)

    message = str(error.value)
    assert "pdf-limited-model" in message
    assert "pdf_url" in message
    assert "rejected_before_transport" in message


@pytest.mark.unit
@pytest.mark.parametrize("route_type", ("positive", "exceptions"))
def test_duplicate_evidence_route_is_rejected(route_type: str):
    catalogue = _scenario_catalogue()
    duplicate_routes = getattr(catalogue, route_type)
    catalogue = replace(
        catalogue,
        **{route_type: duplicate_routes + (duplicate_routes[-1],)},
    )
    model = _model(
        "mistral-small-2603",
        [_exception("rejected_before_transport")] if route_type != "positive" else [],
    )

    with pytest.raises(ValueError, match="duplicate"):
        _select("mistral", model, catalogue)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("organization", "model"),
    [
        pytest.param(
            organization,
            model,
            id=f"{organization}-{model.name}",
        )
        for organization, model in FIRST_PARTY_MODEL_PROFILES
    ],
)
def test_every_first_party_profile_has_routes_for_its_shared_capabilities(
    organization: str,
    model: ModelSpec,
):
    from tests.capability_selection import select_model_scenarios

    gaps = []
    declared = {
        exception.capability_id: exception
        for exception in model.require_capability_profile()
    }
    for capability in E2E_SCENARIO_CAPABILITIES:
        try:
            select_model_scenarios(
                organization=organization,
                model=model,
                capabilities=(capability,),
            )
        except (TypeError, ValueError) as error:
            exception = declared.get(capability.id)
            behavior_id = (
                exception.behavior_id
                if exception is not None
                else "always-on"
                if capability.scope == "always-on"
                else "baseline"
            )
            gaps.append(
                f"{organization}/{model.name}: capability_id={capability.id}, "
                f"behavior_id={behavior_id}: {error}"
            )

    assert not gaps, "\n".join(gaps)

    selected = set(
        select_model_scenarios(
            organization=organization,
            model=model,
            capabilities=E2E_SCENARIO_CAPABILITIES,
        )
    )
    shared_capabilities = {item.id: item for item in E2E_SCENARIO_CAPABILITIES}
    for capability_id in ("sync_chat", "application_tools"):
        expected_node = next(
            route.node_id
            for route in SCENARIO_CATALOGUE.positive
            if route.capability_id == capability_id
        )
        assert shared_capabilities[capability_id].scope == "model-dependent"
        assert expected_node in selected, (
            f"{organization}/{model.name}: capability_id={capability_id}, "
            "behavior_id=baseline shared scenario was not selected"
        )
