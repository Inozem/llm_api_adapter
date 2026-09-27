"""Deterministic coverage checks for the test-only E2E scenario inventory."""

from dataclasses import replace

import pytest

from llm_api_adapter.llm_registry.model_capabilities import ModelCapability
from tests.capability_scenarios import (
    CapabilityScenario,
    DeclaredException,
    E2E_SCENARIO_CAPABILITIES,
    ExceptionScenario,
    PassSupplement,
    SCENARIO_CATALOGUE,
    ScenarioCatalogue,
    first_party_declared_exceptions,
    validate_scenario_catalogue,
)


CAPABILITIES = (
    ModelCapability("sync_chat", "model-dependent"),
    ModelCapability("pdf_url", "model-dependent"),
    ModelCapability("error_normalization", "always-on"),
)
DECLARED_EXCEPTIONS = (
    DeclaredException("example", "limited-pdf", "pdf_url", "rejected_before_transport"),
    DeclaredException("mistral", "mistral-small-2603", "pdf_url", "pass"),
)


def _synthetic_catalogue() -> ScenarioCatalogue:
    return ScenarioCatalogue(
        positive=(
            CapabilityScenario("sync_chat", "tests/e2e/test_chat.py::test_sync_chat"),
            CapabilityScenario("pdf_url", "tests/e2e/test_files.py::test_pdf_succeeds"),
        ),
        exceptions=(
            ExceptionScenario(
                "pdf_url",
                "rejected_before_transport",
                "tests/e2e/test_files.py::test_pdf_rejected",
            ),
        ),
        supplements=(
            PassSupplement(
                "mistral",
                "mistral-small-2603",
                "pdf_url",
                "tests/e2e/test_mistral.py::test_pdf_uses_ocr",
            ),
        ),
        always_on=(
            CapabilityScenario(
                "error_normalization",
                "tests/e2e/test_errors.py::test_error_normalization",
            ),
        ),
    )


@pytest.mark.unit
def test_checked_in_catalogue_covers_capabilities_and_declared_profiles():
    validate_scenario_catalogue(SCENARIO_CATALOGUE)

    declared = first_party_declared_exceptions()
    e2e_capability_ids = {
        capability.id for capability in E2E_SCENARIO_CAPABILITIES
    }
    mapped_pairs = {
        (route.capability_id, route.behavior_id)
        for route in SCENARIO_CATALOGUE.exceptions
    }
    mapped_supplements = {
        (route.organization, route.model, route.capability_id)
        for route in SCENARIO_CATALOGUE.supplements
    }
    assert mapped_pairs == {
        (item.capability_id, item.behavior_id)
        for item in declared
        if item.capability_id in e2e_capability_ids and item.behavior_id != "pass"
    }
    assert mapped_supplements == {
        (item.organization, item.model, item.capability_id)
        for item in declared
        if item.capability_id in e2e_capability_ids and item.behavior_id == "pass"
    }


@pytest.mark.unit
def test_validator_accepts_a_complete_synthetic_catalogue():
    validate_scenario_catalogue(
        _synthetic_catalogue(),
        capabilities=CAPABILITIES,
        declared_exceptions=DECLARED_EXCEPTIONS,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("positive", "missing baseline-positive"),
        ("exceptions", "missing exception evidence"),
        ("supplements", "missing exact-model pass supplements"),
        ("always_on", "missing unconditional"),
    ],
)
def test_validator_rejects_missing_evidence(field: str, message: str):
    catalogue = _synthetic_catalogue()
    catalogue = replace(catalogue, **{field: getattr(catalogue, field)[:-1]})

    with pytest.raises(ValueError, match=message):
        validate_scenario_catalogue(
            catalogue,
            capabilities=CAPABILITIES,
            declared_exceptions=DECLARED_EXCEPTIONS,
        )


@pytest.mark.unit
@pytest.mark.parametrize("field", ("positive", "exceptions", "supplements", "always_on"))
def test_validator_rejects_duplicate_evidence(field: str):
    catalogue = _synthetic_catalogue()
    routes = getattr(catalogue, field)
    catalogue = replace(catalogue, **{field: routes + (routes[-1],)})

    with pytest.raises(ValueError, match="duplicate"):
        validate_scenario_catalogue(
            catalogue,
            capabilities=CAPABILITIES,
            declared_exceptions=DECLARED_EXCEPTIONS,
        )


@pytest.mark.unit
def test_validator_keeps_model_dependent_scenarios_out_of_unconditional_routes():
    catalogue = _synthetic_catalogue()
    catalogue = replace(
        catalogue,
        always_on=catalogue.always_on
        + (CapabilityScenario("pdf_url", "tests/e2e/test_files.py::test_pdf_succeeds"),),
    )

    with pytest.raises(ValueError, match="cannot have an unconditional route"):
        validate_scenario_catalogue(
            catalogue,
            capabilities=CAPABILITIES,
            declared_exceptions=DECLARED_EXCEPTIONS,
        )


@pytest.mark.unit
def test_validator_rejects_pass_as_a_replacement_behavior():
    catalogue = _synthetic_catalogue()
    catalogue = replace(
        catalogue,
        exceptions=(
            ExceptionScenario("pdf_url", "pass", "tests/e2e/test_files.py::test_pdf"),
        ),
    )

    with pytest.raises(ValueError, match="non-pass behavior_id"):
        validate_scenario_catalogue(
            catalogue,
            capabilities=CAPABILITIES,
            declared_exceptions=DECLARED_EXCEPTIONS,
        )


@pytest.mark.unit
def test_validator_rejects_routes_without_a_pytest_node_id():
    catalogue = _synthetic_catalogue()
    catalogue = replace(
        catalogue,
        positive=(CapabilityScenario("sync_chat", "test_sync_chat"),)
        + catalogue.positive[1:],
    )

    with pytest.raises(ValueError, match="valid pytest node ID"):
        validate_scenario_catalogue(
            catalogue,
            capabilities=CAPABILITIES,
            declared_exceptions=DECLARED_EXCEPTIONS,
        )
