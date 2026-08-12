"""Unit tests for execution of validated registry request rules."""

from __future__ import annotations

import logging
import warnings

import pytest

from src.llm_api_adapter.llm_registry.request_rules import (
    AppliedRequestRule,
    RequestRule,
    RequestRuleRegistry,
    RequestRules,
    apply_request_rules,
)


@pytest.mark.unit
def test_runner_renames_parameters_without_mutating_the_source_payload():
    payload = {"max_tokens": 256, "messages": []}
    request_rules = RequestRules(
        rules=(
            RequestRule(
                handler=RequestRuleRegistry.RENAME_PARAMETER,
                arguments={"from": "max_tokens", "to": "max_completion_tokens"},
            ),
        )
    )

    transformed, diagnostics = apply_request_rules(
        payload,
        request_rules,
        model="gpt-4.1-mini",
    )

    assert transformed == {"max_completion_tokens": 256, "messages": []}
    assert payload == {"max_tokens": 256, "messages": []}
    assert diagnostics == (
        AppliedRequestRule(
            handler="rename_parameter",
            path="max_tokens",
            target_path="max_completion_tokens",
        ),
    )


@pytest.mark.unit
def test_runner_quietly_omits_default_values(caplog):
    payload = {"top_p": 1.0}
    request_rules = RequestRules(
        rules=(
            RequestRule(
                handler=RequestRuleRegistry.DROP_PARAMETER,
                arguments={"path": "top_p", "default": 1.0},
            ),
        )
    )

    with caplog.at_level(
        logging.WARNING,
        logger="src.llm_api_adapter.llm_registry.request_rules",
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            transformed, diagnostics = apply_request_rules(
                payload,
                request_rules,
                model="gpt-5",
            )

    assert transformed == {}
    assert payload == {"top_p": 1.0}
    assert caught == []
    assert caplog.records == []
    assert diagnostics == (
        AppliedRequestRule(
            handler="drop_parameter",
            path="top_p",
        ),
    )


@pytest.mark.unit
def test_runner_warns_and_logs_once_for_non_default_values(caplog):
    request_rules = RequestRules(
        rules=(
            RequestRule(
                handler=RequestRuleRegistry.DROP_PARAMETER,
                arguments={"path": "top_p", "default": 1.0},
            ),
        )
    )

    with caplog.at_level(
        logging.WARNING,
        logger="src.llm_api_adapter.llm_registry.request_rules",
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            transformed, diagnostics = apply_request_rules(
                {"top_p": 0.2},
                request_rules,
                model="gpt-5",
            )

    assert transformed == {}
    assert len(caught) == 1
    assert caught[0].category is UserWarning
    assert "top_p" in str(caught[0].message)
    assert "gpt-5" in str(caught[0].message)
    assert [record.message for record in caplog.records] == [
        "Parameter 'top_p' is not supported for model 'gpt-5' and will be ignored."
    ]
    assert diagnostics == (
        AppliedRequestRule(
            handler="drop_parameter",
            path="top_p",
            warning_emitted=True,
        ),
    )


@pytest.mark.unit
def test_runner_uses_a_deep_copy_for_nested_parameter_paths():
    payload = {
        "generationConfig": {
            "maxOutputTokens": None,
            "temperature": 1.0,
        }
    }
    request_rules = RequestRules(
        rules=(
            RequestRule(
                handler=RequestRuleRegistry.DROP_PARAMETER,
                arguments={
                    "path": "generationConfig.maxOutputTokens",
                    "default": None,
                },
            ),
        )
    )

    transformed, diagnostics = apply_request_rules(
        payload,
        request_rules,
        model="gemini-2.5-flash",
    )

    assert transformed == {"generationConfig": {"temperature": 1.0}}
    assert payload == {
        "generationConfig": {
            "maxOutputTokens": None,
            "temperature": 1.0,
        }
    }
    assert diagnostics == (
        AppliedRequestRule(
            handler="drop_parameter",
            path="generationConfig.maxOutputTokens",
        ),
    )


@pytest.mark.unit
def test_api_variant_rules_are_not_payload_transformations():
    request_rules = RequestRules(
        rules=(
            RequestRule(
                handler=RequestRuleRegistry.SELECT_API_VARIANT,
                arguments={"variant": "responses"},
            ),
        )
    )

    transformed, diagnostics = apply_request_rules(
        {"model": "gpt-5"},
        request_rules,
        model="gpt-5",
    )

    assert transformed == {"model": "gpt-5"}
    assert diagnostics == ()
