"""Specify the exact-model exception profile parser contract."""

import pytest

from src.llm_api_adapter.llm_registry.llm_registry import ModelSpec


def _model_data(*, capability_exceptions="not-provided"):
    data = {
        "limits": {
            "context_window_tokens": 128_000,
            "max_output_tokens": 16_384,
        },
        "pricing_tiers": [
            {
                "up_to_prompt_tokens": None,
                "input_per_1m": 1_000,
                "output_per_1m": 2_000,
            }
        ],
    }
    if capability_exceptions != "not-provided":
        data["capability_exceptions"] = capability_exceptions
    return data


@pytest.mark.unit
def test_empty_exception_list_is_an_explicit_profile_but_absence_is_legacy():
    explicit = ModelSpec.from_dict(
        "explicit-model",
        _model_data(capability_exceptions=[]),
    )
    legacy = ModelSpec.from_dict("legacy-model", _model_data())

    assert explicit.capability_exceptions == ()
    assert legacy.capability_exceptions is None


@pytest.mark.unit
def test_model_profile_parses_a_declared_capability_exception():
    model = ModelSpec.from_dict(
        "pdf-limited-model",
        _model_data(
            capability_exceptions=[
                {
                    "capability_id": "pdf_url",
                    "behavior_id": "rejected_before_transport",
                    "behavior": "rejects remote PDF URLs before submission",
                }
            ]
        ),
    )

    assert len(model.capability_exceptions) == 1
    exception = model.capability_exceptions[0]
    assert exception.capability_id == "pdf_url"
    assert exception.behavior_id == "rejected_before_transport"
    assert exception.behavior == "rejects remote PDF URLs before submission"


@pytest.mark.unit
def test_compensated_exception_accepts_the_explicit_pass_behavior_id():
    model = ModelSpec.from_dict(
        "ocr-model",
        _model_data(
            capability_exceptions=[
                {
                    "capability_id": "pdf_bytes",
                    "behavior_id": "pass",
                    "behavior": "handles PDF bytes through OCR before chat",
                }
            ]
        ),
    )

    assert model.capability_exceptions[0].behavior_id == "pass"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("exceptions", "capability_id"),
    [
        (None, None),
        ({}, None),
        ([None], None),
        ([{"behavior": "does not support this"}], None),
        (
            [
                {
                    "capability_id": "unregistered_provider_shortcut",
                    "behavior_id": "rejected_before_transport",
                    "behavior": "unsupported",
                }
            ],
            "unregistered_provider_shortcut",
        ),
        (
            [
                {
                    "capability_id": "pdf_url",
                    "behavior_id": "rejected_before_transport",
                    "behavior": "unsupported",
                },
                {
                    "capability_id": "pdf_url",
                    "behavior_id": "rejected_before_transport",
                    "behavior": "also unsupported",
                },
            ],
            "pdf_url",
        ),
        (
            [
                {
                    "capability_id": "request_rule_fidelity",
                    "behavior_id": "rejected_before_transport",
                    "behavior": "unsupported",
                }
            ],
            "request_rule_fidelity",
        ),
        ([{"capability_id": "pdf_url"}], "pdf_url"),
        (
            [
                {
                    "capability_id": "pdf_url",
                    "behavior_id": "rejected_before_transport",
                    "behavior": "",
                }
            ],
            "pdf_url",
        ),
        (
            [
                {
                    "capability_id": "pdf_url",
                    "behavior_id": "rejected_before_transport",
                    "behavior": "rejects remote PDFs",
                    "variant_limits": {"pdf_bytes": "supported"},
                }
            ],
            "pdf_url",
        ),
    ],
)
def test_invalid_exception_profiles_fail_with_model_and_capability_named(
    exceptions,
    capability_id,
):
    with pytest.raises(ValueError) as error:
        ModelSpec.from_dict(
            "invalid-model-profile",
            _model_data(capability_exceptions=exceptions),
        )

    message = str(error.value)
    assert "invalid-model-profile" in message
    if capability_id is not None:
        assert capability_id in message


@pytest.mark.unit
@pytest.mark.parametrize(
    "invalid_behavior_id",
    (None, "", " ", "Pass", "before-transport", "2nd_route", "has space", 7, True),
)
def test_declared_exception_requires_a_valid_behavior_id(invalid_behavior_id):
    with pytest.raises(ValueError) as error:
        ModelSpec.from_dict(
            "invalid-behavior-model",
            _model_data(
                capability_exceptions=[
                    {
                        "capability_id": "pdf_url",
                        "behavior_id": invalid_behavior_id,
                        "behavior": "rejects PDF URLs before submission",
                    }
                ]
            ),
        )

    message = str(error.value)
    assert "invalid-behavior-model" in message
    assert "pdf_url" in message
    assert "behavior_id" in message


@pytest.mark.unit
def test_declared_exception_rejects_a_missing_behavior_id():
    with pytest.raises(ValueError, match="behavior_id"):
        ModelSpec.from_dict(
            "missing-behavior-model",
            _model_data(
                capability_exceptions=[
                    {
                        "capability_id": "pdf_url",
                        "behavior": "rejects PDF URLs before submission",
                    }
                ]
            ),
        )
