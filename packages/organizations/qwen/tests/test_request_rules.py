"""Qwen package policy declarations stay separate from Core request rules."""

from __future__ import annotations

import copy
import importlib
import logging
from pathlib import Path
import sys
import warnings

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[2]
for source in (PACKAGE_ROOT / "src", REPOSITORY_ROOT / "src"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from llm_api_adapter_qwen.request_rules import (
    forced_tool_reasoning_modes,
    parse_qwen_request_rules,
    validate_qwen_policy_catalogue,
)


HANDLER = "disable_reasoning_for_forced_tools"


def _declaration(modes):
    return {"handler": HANDLER, "arguments": {"modes": list(modes)}}


@pytest.mark.unit
@pytest.mark.parametrize("modes", [("any",), ("tool",), ("any", "tool")])
def test_qwen_policy_preserves_exact_modes_without_payload_effects(modes, caplog):
    declarations = [_declaration(modes)]
    before = copy.deepcopy(declarations)

    with caplog.at_level(logging.WARNING):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            parsed = parse_qwen_request_rules(declarations)

    assert parsed == frozenset(modes)
    assert declarations == before
    assert type(parsed) is frozenset
    assert caught == []
    assert caplog.records == []


@pytest.mark.unit
def test_qwen_policy_does_not_interact_with_existing_core_payload_rules():
    from llm_api_adapter.llm_registry.request_rules import (
        AppliedRequestRule,
        RequestRule,
        RequestRuleRegistry,
        RequestRules,
        apply_request_rules,
    )

    qwen_modes = parse_qwen_request_rules([_declaration(["any", "tool"])])
    core_rules = RequestRules(
        rules=(
            RequestRule(
                handler=RequestRuleRegistry.DROP_PARAMETER,
                arguments={"path": "top_p", "default": 1.0},
            ),
        )
    )
    payload = {
        "top_p": 0.2,
        "thinking": {"type": "enabled", "budget_tokens": 64},
    }

    with pytest.warns(UserWarning, match="top_p") as caught:
        transformed, diagnostics = apply_request_rules(
            payload,
            core_rules,
            model="qwen-test-profile",
        )

    assert qwen_modes == frozenset({"any", "tool"})
    assert transformed == {"thinking": {"type": "enabled", "budget_tokens": 64}}
    assert transformed["thinking"] == payload["thinking"]
    assert payload == {
        "top_p": 0.2,
        "thinking": {"type": "enabled", "budget_tokens": 64},
    }
    assert len(caught) == 1
    assert diagnostics == (
        AppliedRequestRule(
            handler=RequestRuleRegistry.DROP_PARAMETER,
            path="top_p",
            warning_emitted=True,
        ),
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("allowed_modes", "policy_modes"),
    [
        ({"auto", "none", "any"}, ["any"]),
        ({"auto", "none", "tool"}, ["tool"]),
        ({"auto", "none", "any", "tool"}, ["any", "tool"]),
    ],
)
def test_policy_is_compatible_with_existing_tool_restrictions(
    allowed_modes, policy_modes
):
    assert parse_qwen_request_rules(
        [_declaration(policy_modes)],
        allowed_tool_choice_modes=frozenset(allowed_modes),
    ) == frozenset(policy_modes)


@pytest.mark.unit
@pytest.mark.parametrize("policy_first", [False, True])
def test_policy_conflicts_with_tool_modes_excluded_by_existing_restrictions(
    policy_first,
):
    qwen_policy = {"request_rules": [_declaration(["any"])]}
    allowed = {
        "handler": "restrict_tool_choice",
        "arguments": {"allowed_values": ["auto", "none"]},
    }
    valid_allowed = {
        "handler": "restrict_tool_choice",
        "arguments": {"allowed_values": ["auto", "none", "any", "tool"]},
    }
    model_data = (
        {"qwen_policy": qwen_policy, "request_rules": [valid_allowed]}
        if policy_first
        else {"request_rules": [valid_allowed], "qwen_policy": qwen_policy}
    )
    assert validate_qwen_policy_catalogue({"models": {"qwen-test": model_data}}) == {
        "qwen-test": frozenset({"any"})
    }

    model_data = (
        {"qwen_policy": qwen_policy, "request_rules": [allowed]}
        if policy_first
        else {"request_rules": [allowed], "qwen_policy": qwen_policy}
    )
    with pytest.raises(ValueError, match="allowed|compatible|conflict"):
        validate_qwen_policy_catalogue({"models": {"qwen-test": model_data}})


@pytest.mark.unit
def test_qwen_policy_extension_does_not_enter_core_model_rules():
    data = {
        "currency": "CNY",
        "models": {
            "qwen-test": {
                "qwen_policy": {"request_rules": [_declaration(["any"])]},
                "limits": {"context_window_tokens": 1, "max_output_tokens": 1},
                "pricing_tiers": [
                    {
                        "up_to_prompt_tokens": None,
                        "input_per_1m": 1.0,
                        "output_per_1m": 1.0,
                    }
                ],
            }
        },
    }
    assert validate_qwen_policy_catalogue(data) == {"qwen-test": frozenset({"any"})}
    from llm_api_adapter.llm_registry.llm_registry import ModelSpec

    core_model = ModelSpec.from_dict("qwen-test", data["models"]["qwen-test"])
    assert core_model.request_rules.rules == ()
    assert not hasattr(core_model.request_rules, "disabled_reasoning_tool_choice_modes")


@pytest.mark.unit
def test_loaded_qwen_catalogue_registers_normally_without_core_policy_mutation():
    qwen_registry = importlib.import_module("llm_api_adapter_qwen.registry")
    core_registry = importlib.import_module(
        "llm_api_adapter.llm_registry.llm_registry"
    )
    metadata = qwen_registry.MODEL_METADATA
    fresh = core_registry.RegistrySpec()

    assert metadata.organization == "qwen"
    assert metadata.request_rule_registry is None
    assert fresh.register_organization_metadata(metadata) is True
    registered = fresh.organizations["qwen"]
    assert tuple(registered.models) == (
        "qwen4-max",
        "qwen3.8-max",
        "qwen3.8-flash",
        "qwen3.7-plus",
        "qwen3.7-flash",
    )
    for model_name in registered.models:
        assert registered.models[model_name].request_rules.rules == ()
        assert not hasattr(
            registered.models[model_name].request_rules,
            "disabled_reasoning_tool_choice_modes",
        )
    for model_name in (
        "qwen3.8-max",
        "qwen3.8-flash",
        "qwen3.7-plus",
        "qwen3.7-flash",
    ):
        assert forced_tool_reasoning_modes(model_name) == frozenset({"any", "tool"})
    assert forced_tool_reasoning_modes("qwen4-max") is None


@pytest.mark.unit
def test_missing_or_unknown_model_has_no_borrowed_qwen_policy():
    assert forced_tool_reasoning_modes("qwen-unknown") is None
    assert forced_tool_reasoning_modes("") is None


@pytest.mark.unit
def test_absent_and_empty_qwen_policies_admit_no_model_rule():
    catalogue = {
        "models": {
            "qwen-absent": {},
            "qwen-empty-policy": {"qwen_policy": {"request_rules": []}},
        }
    }
    assert validate_qwen_policy_catalogue(catalogue) == {}


@pytest.mark.unit
def test_qwen_catalogue_and_declaration_boundary_is_closed():
    invalid_policies = [
        ("not-an-object", "qwen_policy must contain only request_rules"),
        ({"other": []}, "qwen_policy must contain only request_rules"),
        ({"request_rules": [], "extra": True}, "qwen_policy must contain only request_rules"),
        ({"request_rules": None}, "request_rules must be an array"),
        ({"request_rules": [{}]}, "request rule 0 must contain only handler and arguments"),
        (
            {"request_rules": [{"handler": HANDLER, "arguments": None}]},
            "arguments must be an object",
        ),
        (
            {"request_rules": [{"handler": HANDLER, "arguments": {"modes": ["any"]}, "extra": 1}]},
            "only handler and arguments",
        ),
        (
            {"request_rules": [{"handler": "unknown", "arguments": {}}]},
            "handler is unknown",
        ),
    ]
    for policy, message in invalid_policies:
        with pytest.raises(ValueError, match=message):
            validate_qwen_policy_catalogue(
                {"models": {"qwen-test": {"qwen_policy": policy}}}
            )


@pytest.mark.unit
def test_qwen_policy_schema_admits_only_supported_unique_modes():
    for modes in (["any"], ["tool"], ["any", "tool"], ["tool", "any"]):
        assert parse_qwen_request_rules([_declaration(modes)]) == frozenset(modes)

    invalid = [
        ({"modes": []}, "non-empty"),
        ({}, "modes"),
        ({"modes": "any"}, "array"),
        ({"modes": [True]}, "strings"),
        ({"modes": ["any", 1]}, "strings"),
        ({"modes": ["auto"]}, "any.*tool"),
        ({"modes": ["any", "any"]}, "duplicate"),
        ({"modes": ["any"], "extra": True}, "only modes"),
    ]
    for arguments, message in invalid:
        with pytest.raises(ValueError, match=message):
            parse_qwen_request_rules(
                [{"handler": HANDLER, "arguments": arguments}]
            )

    with pytest.raises(ValueError, match="only.*handler|unknown"):
        parse_qwen_request_rules([{"handler": "other", "arguments": {}}])
    with pytest.raises(ValueError, match="more than once|duplicate"):
        parse_qwen_request_rules(
            [_declaration(["any"]), {"handler": HANDLER, "arguments": {"modes": ["tool"]}}]
        )


@pytest.mark.unit
def test_duplicate_qwen_policy_and_catalogue_conflicts_are_rejected():
    for other_modes in (["any"], ["tool"]):
        with pytest.raises(ValueError, match="more than once|duplicate"):
            parse_qwen_request_rules(
                [_declaration(["any"]), _declaration(other_modes)]
            )

    malformed_catalogue = {
        "currency": "CNY",
        "models": {
            "qwen-test": {
                "request_rules": [
                    {
                        "handler": "restrict_tool_choice",
                        "arguments": {"allowed_values": ["auto", "none"]},
                    }
                ],
                "qwen_policy": {"request_rules": [_declaration(["any"])]},
            }
        },
    }
    with pytest.raises(ValueError, match="allowed|compatible|conflict"):
        validate_qwen_policy_catalogue(malformed_catalogue)
