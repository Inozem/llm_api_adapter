"""Unit tests for execution of validated registry request rules."""

from __future__ import annotations

import logging
import warnings
from itertools import permutations

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
def test_runner_keeps_conditional_parameters_when_the_condition_matches():
    request_rules = RequestRules(
        rules=(
            RequestRule(
                handler=RequestRuleRegistry.DROP_PARAMETER_UNLESS,
                arguments={
                    "path": "temperature",
                    "condition_path": "reasoning.effort",
                    "allowed_values": ["none"],
                    "default": 1.0,
                },
            ),
        )
    )

    transformed, diagnostics = apply_request_rules(
        {"reasoning": {"effort": "none"}, "temperature": 0.2},
        request_rules,
        model="gpt-6-sol",
    )

    assert transformed == {"reasoning": {"effort": "none"}, "temperature": 0.2}
    assert diagnostics == ()


@pytest.mark.unit
def test_runner_omits_conditional_parameters_when_the_condition_does_not_match():
    request_rules = RequestRules(
        rules=(
            RequestRule(
                handler=RequestRuleRegistry.DROP_PARAMETER_UNLESS,
                arguments={
                    "path": "temperature",
                    "condition_path": "reasoning.effort",
                    "allowed_values": ["none"],
                    "default": 1.0,
                },
            ),
        )
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        transformed, diagnostics = apply_request_rules(
            {"reasoning": {"effort": "medium"}, "temperature": 0.2},
            request_rules,
            model="gpt-6-sol",
        )

    assert transformed == {"reasoning": {"effort": "medium"}}
    assert len(caught) == 1
    assert diagnostics == (
        AppliedRequestRule(
            handler="drop_parameter_unless",
            path="temperature",
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
        model="synthetic-google-model",
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


@pytest.mark.unit
def test_tool_choice_restriction_is_registry_metadata_not_a_payload_change():
    request_rules = RequestRules(
        rules=(
            RequestRule(
                handler=RequestRuleRegistry.RESTRICT_TOOL_CHOICE,
                arguments={"allowed_values": ["auto", "none"]},
            ),
        )
    )

    transformed, diagnostics = apply_request_rules(
        {"model": "claude-fable-5-1"},
        request_rules,
        model="claude-fable-5-1",
    )

    assert request_rules.allowed_tool_choice_modes == frozenset({"auto", "none"})
    assert transformed == {"model": "claude-fable-5-1"}
    assert diagnostics == ()


_DISABLE_FORCED_TOOL_HANDLER = "disable_reasoning_for_forced_tools"


class _SemanticRequestRuleRegistry(RequestRuleRegistry):
    """Generic test schema for phase/order interaction of the semantic rule."""

    organization_name = "qwen-test"
    supported_handlers = frozenset(
        {
            _DISABLE_FORCED_TOOL_HANDLER,
            RequestRuleRegistry.RESTRICT_TOOL_CHOICE,
            RequestRuleRegistry.DROP_PARAMETER,
        }
    )
    supported_tool_choice_modes = frozenset({"auto", "none", "any", "tool"})
    droppable_parameter_defaults = {"top_p": 1.0}


_SEMANTIC_RULE_REGISTRY = _SemanticRequestRuleRegistry()


def _forced_tool_rule(modes):
    # Direct construction supplies a controlled semantic rule object without
    # substituting test-side validation for the package schema's parser tests.
    return RequestRule(
        handler=_DISABLE_FORCED_TOOL_HANDLER,
        arguments={"modes": list(modes)},
    )


@pytest.mark.unit
@pytest.mark.parametrize("modes", [("any",), ("tool",), ("any", "tool")])
def test_forced_tool_semantic_rule_preserves_exact_modes_without_payload_effects(
    modes,
    caplog,
):
    payload = {
        "thinking": {"type": "enabled", "budget_tokens": 2048},
        "output_config": {"format": {"type": "json_schema", "schema": {"type": "object"}}, "effort": "high"},
    }
    request_rules = RequestRules(rules=(_forced_tool_rule(modes),))

    with caplog.at_level(logging.WARNING):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            transformed, diagnostics = apply_request_rules(
                payload,
                request_rules,
                model="qwen-test-profile",
            )

    assert request_rules.disabled_reasoning_tool_choice_modes == frozenset(modes)
    assert request_rules.rules[0].affected_paths == ()
    assert transformed == payload
    assert transformed is not payload
    assert transformed["output_config"] is not payload["output_config"]
    assert diagnostics == ()
    assert caught == []
    assert caplog.records == []


@pytest.mark.unit
@pytest.mark.parametrize(
    ("allowed_modes", "disabled_modes"),
    [
        ({"auto", "none", "any"}, {"any"}),
        ({"auto", "none", "tool"}, {"tool"}),
        ({"auto", "none", "any", "tool"}, {"any", "tool"}),
    ],
)
def test_semantic_policy_and_restrictions_are_compatible_in_either_rule_order(
    allowed_modes,
    disabled_modes,
):
    restrict = {
        "handler": RequestRuleRegistry.RESTRICT_TOOL_CHOICE,
        "arguments": {"allowed_values": sorted(allowed_modes)},
    }
    semantic = {
        "handler": _DISABLE_FORCED_TOOL_HANDLER,
        "arguments": {"modes": sorted(disabled_modes)},
    }

    for ordered_rules in permutations((restrict, semantic)):
        request_rules = RequestRules.from_dict(
            list(ordered_rules),
            rule_registry=_SEMANTIC_RULE_REGISTRY,
        )
        assert request_rules.allowed_tool_choice_modes == frozenset(allowed_modes)
        assert request_rules.disabled_reasoning_tool_choice_modes == frozenset(
            disabled_modes
        )


@pytest.mark.unit
@pytest.mark.parametrize("semantic_first", [False, True])
def test_semantic_rule_rejects_modes_excluded_by_tool_restriction(semantic_first):
    semantic = {
        "handler": _DISABLE_FORCED_TOOL_HANDLER,
        "arguments": {"modes": ["any"]},
    }
    restrict = {
        "handler": RequestRuleRegistry.RESTRICT_TOOL_CHOICE,
        "arguments": {"allowed_values": ["auto", "none", "tool"]},
    }
    valid = {
        "handler": RequestRuleRegistry.RESTRICT_TOOL_CHOICE,
        "arguments": {"allowed_values": ["auto", "none", "any", "tool"]},
    }
    admitted = RequestRules.from_dict(
        [semantic, valid] if semantic_first else [valid, semantic],
        rule_registry=_SEMANTIC_RULE_REGISTRY,
    )
    assert admitted.disabled_reasoning_tool_choice_modes == frozenset({"any"})

    rules = [semantic, restrict] if semantic_first else [restrict, semantic]
    with pytest.raises(ValueError, match="allowed|compatible|conflict"):
        RequestRules.from_dict(rules, rule_registry=_SEMANTIC_RULE_REGISTRY)


@pytest.mark.unit
def test_semantic_rule_order_around_parameter_drop():
    semantic = {
        "handler": _DISABLE_FORCED_TOOL_HANDLER,
        "arguments": {"modes": ["any"]},
    }
    drop = {
        "handler": RequestRuleRegistry.DROP_PARAMETER,
        "arguments": {"path": "top_p"},
    }
    admitted = RequestRules.from_dict(
        [semantic, drop],
        rule_registry=_SEMANTIC_RULE_REGISTRY,
    )
    assert admitted.disabled_reasoning_tool_choice_modes == frozenset({"any"})
    with pytest.raises(ValueError, match="order|phase"):
        RequestRules.from_dict([drop, semantic], rule_registry=_SEMANTIC_RULE_REGISTRY)


@pytest.mark.unit
def test_semantic_forced_tool_rule_does_not_interact_with_other_payload_rules():
    semantic = _forced_tool_rule(("any", "tool"))
    drop = RequestRule(
        handler=RequestRuleRegistry.DROP_PARAMETER,
        arguments={"path": "top_p", "default": 1.0},
    )
    request_rules = RequestRules(rules=(semantic, drop))

    with pytest.warns(UserWarning, match="top_p") as caught:
        transformed, diagnostics = apply_request_rules(
            {"top_p": 0.2, "thinking": {"type": "enabled", "budget_tokens": 64}},
            request_rules,
            model="qwen-test-profile",
        )

    assert "top_p" not in transformed
    assert transformed["thinking"] == {"type": "enabled", "budget_tokens": 64}
    assert len(caught) == 1
    assert diagnostics == (
        AppliedRequestRule(
            handler=RequestRuleRegistry.DROP_PARAMETER,
            path="top_p",
            warning_emitted=True,
        ),
    )
