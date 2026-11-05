"""Specify the exact-model exception profile parser contract."""

import pytest

from src.llm_api_adapter.llm_registry.llm_registry import ModelSpec
from src.llm_api_adapter.llm_registry.request_rules import OpenAIRequestRuleRegistry


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


def _api_exception(
    capability_id="pdf_url",
    behavior_id="rejected_before_transport",
):
    return {
        "capability_id": capability_id,
        "behavior_id": behavior_id,
        "behavior": "rejects remote PDF URLs before submission",
    }


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


@pytest.mark.unit
@pytest.mark.parametrize(
    "api_profiles",
    [
        None,
        [],
        "chat_completions",
        True,
        {"chat_completions": None},
        {"chat_completions": ()},
        {"chat_completions": "pdf_url"},
        {"chat_completions": {}},
        {"chat_completions": [_api_exception(), None]},
        {"chat_completions": [{}]},
        {"chat_completions": [{"capability_id": "pdf_url"}]},
        {"chat_completions": [{**_api_exception(), "behavior": ""}]},
        {"chat_completions": [{**_api_exception(), "behavior_id": "Bad ID"}]},
        {"chat_completions": [_api_exception(), _api_exception()]},
        {"chat_completions": [_api_exception("unregistered_capability")]},
        {"chat_completions": [_api_exception("request_rule_fidelity")]},
        {"chat_completions": [{**_api_exception(), "pytest_id": "fake-case"}]},
        {"chat_completions": [{**_api_exception(), "exact_model": "fake-model"}]},
        {"unsupported_api": []},
        {"": []},
        {" ": []},
        {None: []},
        {1: []},
    ],
    ids=[
        "null-field",
        "list-field",
        "string-field",
        "boolean-field",
        "null-entries",
        "tuple-entries",
        "scalar-entries",
        "object-entries",
        "malformed-entry",
        "missing-entry-fields",
        "missing-behavior-id",
        "empty-behavior",
        "invalid-behavior-id",
        "duplicate-capability",
        "unknown-capability",
        "always-on-capability",
        "pytest-id-extra",
        "exact-value-extra",
        "unknown-api",
        "empty-api-key",
        "whitespace-api-key",
        "null-api-key",
        "integer-api-key",
    ],
)
def test_api_exception_profiles_reject_invalid_profile_shapes(api_profiles):
    """Keys use nonempty owner-schema variants.

    Entries reuse the existing exception fields and validation, with no extra
    route IDs or exact-value metadata.
    """
    data = _model_data()
    data["api_capability_exceptions"] = api_profiles

    with pytest.raises(ValueError):
        ModelSpec.from_dict(
            "invalid-api-profile-model",
            data,
            request_rule_registry=OpenAIRequestRuleRegistry(),
        )


@pytest.mark.unit
def test_nonempty_api_profile_mapping_requires_owning_request_rule_schema():
    data = _model_data()
    data["api_capability_exceptions"] = {"chat_completions": []}

    with pytest.raises(ValueError, match="schema"):
        ModelSpec.from_dict("api-profile-without-schema", data)


@pytest.mark.unit
def test_api_profile_omission_and_empty_mapping_are_no_explicit_api_profiles():
    omitted = ModelSpec.from_dict("omitted-api-profiles", _model_data())
    explicit_empty = ModelSpec.from_dict(
        "empty-api-profiles", {**_model_data(), "api_capability_exceptions": {}}
    )

    assert omitted.api_capability_exceptions == {}
    assert explicit_empty.api_capability_exceptions == {}
    assert (
        omitted.api_capability_exceptions
        is not explicit_empty.api_capability_exceptions
    )


@pytest.mark.unit
def test_api_profiles_parse_to_immutable_exceptions_and_copy_input_containers():
    native = [_api_exception()]
    api_entry = _api_exception()
    api_entries = [api_entry]
    api_profiles = {"chat_completions": api_entries}
    data = _model_data(capability_exceptions=native)
    data["api_capability_exceptions"] = api_profiles
    model = ModelSpec.from_dict(
        "copied-api-profile-model",
        data,
        request_rule_registry=OpenAIRequestRuleRegistry(),
    )

    native.clear()
    api_entry["behavior"] = "mutated after parsing"
    api_entries.append(_api_exception("image_url"))
    api_profiles.clear()

    assert (
        model.capability_exceptions[0].behavior
        == "rejects remote PDF URLs before submission"
    )
    assert model.api_capability_exceptions["chat_completions"] == (
        ModelSpec.from_dict(
            "comparison-model",
            _model_data(capability_exceptions=[_api_exception()]),
        ).capability_exceptions[0],
    )
    assert (
        model.api_capability_exceptions["chat_completions"]
        is not model.capability_exceptions
    )


@pytest.mark.unit
def test_api_profiles_support_empty_declared_tuple_and_independent_defaults():
    data = _model_data()
    data["api_capability_exceptions"] = {"chat_completions": []}
    first = ModelSpec.from_dict(
        "first-api-profile-model",
        data,
        request_rule_registry=OpenAIRequestRuleRegistry(),
    )
    second = ModelSpec.from_dict("second-api-profile-model", _model_data())

    assert first.api_capability_exceptions == {"chat_completions": ()}
    assert second.api_capability_exceptions == {}
    assert first.api_capability_exceptions is not second.api_capability_exceptions
