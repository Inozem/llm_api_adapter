"""Contract tests for exact-model capability scenario selection.

The synthetic catalogue keeps routing rules separate from live E2E collection.
T021 supplies the real scenario inventory; T022 implements the selector.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import pytest

from llm_api_adapter.llm_registry.llm_registry import ModelSpec
from llm_api_adapter.llm_registry.model_capabilities import ModelCapability


SYNC_CHAT = "tests/e2e/test_llm_adapter_chat.py::test_sync_chat"
PDF_SUCCESS = "tests/e2e/test_file_uploads.py::test_pdf_url_succeeds"
PDF_REJECTION = "tests/e2e/test_file_uploads.py::test_pdf_url_rejected"
MISTRAL_OCR = "tests/e2e/test_mistral_ocr_costs.py::test_mistral_pdf_ocr_exposes_cost_breakdown"
PACKAGE_OCR = "packages/organizations/example/tests/e2e/test_pdf.py::test_ocr"
ERROR_NORMALIZATION = "tests/e2e/test_errors.py::test_error_normalization"

CAPABILITIES = (
    ModelCapability("sync_chat", "model-dependent"),
    ModelCapability("pdf_url", "model-dependent"),
    ModelCapability("error_normalization", "always-on"),
)


@dataclass(frozen=True)
class _ScenarioCatalogue:
    """Small route inventory; tuple entries preserve duplicate routes for validation."""

    positive: tuple[tuple[str, str], ...] = (
        ("sync_chat", SYNC_CHAT),
        ("pdf_url", PDF_SUCCESS),
    )
    exceptions: tuple[tuple[str, str, str], ...] = (
        ("pdf_url", "rejected_before_transport", PDF_REJECTION),
    )
    supplements: tuple[tuple[str, str, str, str], ...] = (
        ("mistral", "mistral-small-2603", "pdf_url", MISTRAL_OCR),
    )
    always_on: tuple[tuple[str, str], ...] = (
        ("error_normalization", ERROR_NORMALIZATION),
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
    catalogue: _ScenarioCatalogue = _ScenarioCatalogue(),
) -> tuple[str, ...]:
    # Import inside the helper so pytest collects every red contract case even
    # before the selector module exists.
    from tests.capability_selection import select_model_scenarios

    return select_model_scenarios(
        organization=organization,
        model=model,
        scenarios=catalogue,
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
def test_mistral_pdf_pass_keeps_success_and_adds_exact_model_ocr_evidence():
    model = _model("mistral-small-2603", [_exception("pass", "PDF via OCR")])

    selected = _select("mistral", model)

    _assert_selected(
        selected,
        {SYNC_CHAT, PDF_SUCCESS, MISTRAL_OCR, ERROR_NORMALIZATION},
    )
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
        _ScenarioCatalogue(),
        positive=(("sync_chat", SYNC_CHAT),),
    )

    with pytest.raises(ValueError) as error:
        _select("example", _model("complete-model"), catalogue)

    assert "complete-model" in str(error.value)
    assert "pdf_url" in str(error.value)


@pytest.mark.unit
def test_missing_replacement_scenario_is_a_coverage_error():
    model = _model("pdf-limited-model", [_exception("rejected_before_transport")])
    catalogue = replace(_ScenarioCatalogue(), exceptions=())

    with pytest.raises(ValueError) as error:
        _select("example", model, catalogue)

    message = str(error.value)
    assert "pdf-limited-model" in message
    assert "pdf_url" in message
    assert "rejected_before_transport" in message


@pytest.mark.unit
def test_pass_without_exact_model_supplement_is_a_coverage_error():
    model = _model("mistral-small-2603", [_exception("pass")])
    catalogue = replace(_ScenarioCatalogue(), supplements=())

    with pytest.raises(ValueError) as error:
        _select("mistral", model, catalogue)

    message = str(error.value)
    assert "mistral-small-2603" in message
    assert "pdf_url" in message
    assert "pass" in message


@pytest.mark.unit
@pytest.mark.parametrize("route_type", ("positive", "exceptions", "supplements"))
def test_duplicate_evidence_route_is_rejected(route_type: str):
    catalogue = _ScenarioCatalogue()
    duplicate_routes = getattr(catalogue, route_type)
    catalogue = replace(
        catalogue,
        **{route_type: duplicate_routes + (duplicate_routes[-1],)},
    )
    model = _model(
        "mistral-small-2603",
        [_exception("pass" if route_type == "supplements" else "rejected_before_transport")]
        if route_type != "positive"
        else [],
    )

    with pytest.raises(ValueError, match="duplicate"):
        _select("mistral", model, catalogue)


@pytest.mark.unit
def test_package_supplements_are_scoped_to_exact_organization_and_model():
    catalogue = replace(
        _ScenarioCatalogue(),
        supplements=(
            ("mistral", "mistral-small-2603", "pdf_url", MISTRAL_OCR),
            ("example", "mistral-small-2603", "pdf_url", PACKAGE_OCR),
        ),
    )
    model = _model("mistral-small-2603", [_exception("pass")])

    mistral = _select("mistral", model, catalogue)
    example = _select("example", model, catalogue)

    _assert_selected(mistral, {SYNC_CHAT, PDF_SUCCESS, MISTRAL_OCR, ERROR_NORMALIZATION})
    _assert_selected(example, {SYNC_CHAT, PDF_SUCCESS, PACKAGE_OCR, ERROR_NORMALIZATION})


@pytest.mark.unit
def test_another_model_cannot_borrow_a_package_supplement():
    model = _model("mistral-medium-3.5", [_exception("pass")])

    with pytest.raises(ValueError) as error:
        _select("mistral", model)

    assert "mistral-medium-3.5" in str(error.value)
    assert "pdf_url" in str(error.value)
