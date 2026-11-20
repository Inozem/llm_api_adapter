"""Contracts for optional Qwen-local policy and non-token meter metadata."""

from __future__ import annotations

import copy
import dataclasses
import inspect
from pathlib import Path
import sys

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[2]
for source in (PACKAGE_ROOT / "src", REPOSITORY_ROOT / "src"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from llm_api_adapter_qwen.request_rules import validate_qwen_policy_catalogue


def _base_model(**overrides):
    model = {
        "limits": {"context_window_tokens": 100, "max_output_tokens": 80},
        "pricing_tiers": [
            {
                "up_to_prompt_tokens": None,
                "input_per_1m": 1.25,
                "output_per_1m": 2.5,
            }
        ],
    }
    model.update(overrides)
    return model


def _reasoning_modes(default="enabled"):
    return {
        "default": default,
        "enabled": {"max_input_tokens": 80, "max_output_tokens": 70},
        "disabled": {"max_input_tokens": 60, "max_output_tokens": 40},
    }


def _catalogue(model_data):
    return {"currency": "QWEN-TEST", "models": {"qwen-test-model": model_data}}


@pytest.mark.unit
@pytest.mark.parametrize("default", ["enabled", "disabled"])
def test_reasoning_modes_accept_each_supported_default_and_independent_maxima(default):
    model = _base_model()
    model["qwen_policy"] = {"reasoning_modes": _reasoning_modes(default)}

    assert validate_qwen_policy_catalogue(_catalogue(model)) == {}
    modes = model["qwen_policy"]["reasoning_modes"]
    assert (
        modes["enabled"]["max_input_tokens"]
        + modes["enabled"]["max_output_tokens"]
        > model["limits"]["context_window_tokens"]
    )


@pytest.mark.unit
@pytest.mark.parametrize("extension", ["reasoning", "accounting", "combined"])
def test_optional_qwen_extensions_are_independently_admitted(extension):
    request_rules = [
        {
            "handler": "disable_reasoning_for_forced_tools",
            "arguments": {"modes": ["any", "tool"]},
        }
    ]
    policy = {}
    if extension in {"reasoning", "combined"}:
        policy["reasoning_modes"] = _reasoning_modes(
            "disabled" if extension == "combined" else "enabled"
        )
    if extension in {"accounting", "combined"}:
        policy["input_token_accounting"] = "ordinary_plus_cache_read_and_write"
    if extension == "combined":
        policy["request_rules"] = copy.deepcopy(request_rules)
    model = _base_model(qwen_policy=policy)

    expected = {"qwen-test-model": frozenset({"any", "tool"})} if extension == "combined" else {}
    assert validate_qwen_policy_catalogue(_catalogue(model)) == expected
    if extension == "combined":
        assert model["qwen_policy"]["request_rules"] == request_rules


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["enabled", "disabled"])
@pytest.mark.parametrize("cap", ["max_input_tokens", "max_output_tokens"])
@pytest.mark.parametrize("value", [True, False, 12.5, "12", None, 0, -1])
def test_reasoning_caps_reject_nonpositive_or_noninteger_values(mode, cap, value):
    modes = _reasoning_modes()
    modes[mode][cap] = value
    if cap == "max_input_tokens":
        expected_error = "max_input_tokens"
    else:
        expected_error = "max_output_tokens"
    model = _base_model(qwen_policy={"reasoning_modes": modes})

    with pytest.raises(ValueError, match=expected_error):
        validate_qwen_policy_catalogue(_catalogue(model))


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["enabled", "disabled"])
@pytest.mark.parametrize("cap", ["max_input_tokens", "max_output_tokens"])
def test_reasoning_caps_accept_exact_registered_boundaries(mode, cap):
    model = _base_model()
    modes = _reasoning_modes()
    boundary = (
        model["limits"]["context_window_tokens"]
        if cap == "max_input_tokens"
        else model["limits"]["max_output_tokens"]
    )
    modes[mode][cap] = boundary
    model["qwen_policy"] = {"reasoning_modes": modes}

    assert validate_qwen_policy_catalogue(_catalogue(model)) == {}


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["enabled", "disabled"])
@pytest.mark.parametrize("cap", ["max_input_tokens", "max_output_tokens"])
def test_reasoning_caps_reject_values_above_registered_boundaries(mode, cap):
    model = _base_model()
    modes = _reasoning_modes()
    boundary = (
        model["limits"]["context_window_tokens"]
        if cap == "max_input_tokens"
        else model["limits"]["max_output_tokens"]
    )
    modes[mode][cap] = boundary + 1
    model["qwen_policy"] = {"reasoning_modes": modes}

    with pytest.raises(ValueError, match=cap):
        validate_qwen_policy_catalogue(_catalogue(model))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_default", "default"),
        ("invalid_default_type", "default"),
        ("invalid_default_value", "default"),
        ("missing_enabled", "enabled"),
        ("missing_disabled", "disabled"),
        ("malformed_mode", "enabled"),
        ("missing_input_cap", "max_input_tokens"),
        ("missing_output_cap", "max_output_tokens"),
        ("unknown_mode", "reasoning_modes"),
        ("unknown_mode_member", "reasoning_modes"),
    ],
)
def test_reasoning_modes_reject_incomplete_or_open_records(mutation, message):
    modes = _reasoning_modes()
    if mutation == "missing_default":
        modes.pop("default")
    elif mutation == "invalid_default_type":
        modes["default"] = True
    elif mutation == "invalid_default_value":
        modes["default"] = "automatic"
    elif mutation == "missing_enabled":
        modes.pop("enabled")
    elif mutation == "missing_disabled":
        modes.pop("disabled")
    elif mutation == "malformed_mode":
        modes["enabled"] = None
    elif mutation == "missing_input_cap":
        modes["enabled"].pop("max_input_tokens")
    elif mutation == "missing_output_cap":
        modes["disabled"].pop("max_output_tokens")
    elif mutation == "unknown_mode":
        modes["other"] = {"max_input_tokens": 1, "max_output_tokens": 1}
    elif mutation == "unknown_mode_member":
        modes["enabled"]["extra"] = 1
    model = _base_model(qwen_policy={"reasoning_modes": modes})

    with pytest.raises(ValueError, match=message):
        validate_qwen_policy_catalogue(_catalogue(model))


@pytest.mark.unit
@pytest.mark.parametrize(
    "value",
    [None, True, False, "ordinary + cache", {"formula": "x + y"}, 1],
)
def test_input_token_accounting_rejects_values_outside_closed_policy(value):
    model = _base_model(qwen_policy={"input_token_accounting": value})

    with pytest.raises(ValueError, match="input_token_accounting"):
        validate_qwen_policy_catalogue(_catalogue(model))


@pytest.mark.unit
def test_qwen_policy_rejects_unknown_outer_member():
    model = _base_model(
        qwen_policy={
            "input_token_accounting": "ordinary_plus_cache_read_and_write",
            "unexpected_extension": "value",
        }
    )

    with pytest.raises(ValueError, match="unexpected_extension"):
        validate_qwen_policy_catalogue(_catalogue(model))


def _qwen_meter_catalogue(
    *,
    rate_a=0.375,
    rate_b=1.125,
    unit_a="page",
    currency="QWEN-QA",
    include_accounting=False,
):
    model_a_policy = {"request_rules": []}
    if include_accounting:
        model_a_policy["input_token_accounting"] = (
            "ordinary_plus_cache_read_and_write"
        )
    return {
        "currency": currency,
        "models": {
            "qwen-test-chat-model-a": _base_model(
                qwen_policy=model_a_policy
            ),
            "qwen-test-chat-model-b": _base_model(),
        },
        "metered_operations": {
            "pdf_pages_a": {
                "model": "qwen-test-chat-model-a",
                "unit": unit_a,
                "rate": rate_a,
            },
            "pdf_pages_b": {
                "model": "qwen-test-chat-model-b",
                "unit": "page",
                "rate": rate_b,
            },
        },
    }


@pytest.mark.unit
@pytest.mark.parametrize("include_accounting", [False, True])
def test_qwen_metadata_registration_and_meter_resolution_are_isolated_from_core(
    include_accounting,
):
    from llm_api_adapter.llm_registry.llm_registry import (
        LLM_REGISTRY,
        ModelLimits,
        ModelSpec,
        OrganizationModelMetadata,
        RegistrySpec,
        resolve_metered_operation_spec,
    )

    data = _qwen_meter_catalogue(include_accounting=include_accounting)
    before_global = dict(LLM_REGISTRY.organizations)
    registry = RegistrySpec()
    core_inventory_before = {
        name: tuple(organization.models)
        for name, organization in registry.organizations.items()
    }
    assert validate_qwen_policy_catalogue(data) == {}
    metadata = OrganizationModelMetadata("qwen", data)
    assert registry.register_organization_metadata(metadata) is True

    qwen = registry.organizations["qwen"]
    assert set(qwen.models) == set(data["models"])
    assert qwen.models[data["metered_operations"]["pdf_pages_a"]["model"]].request_rules.rules == ()
    for operation_name, operation_data in data["metered_operations"].items():
        meter = resolve_metered_operation_spec(registry, "qwen", operation_name)
        assert meter is not None
        assert (meter.model, meter.unit, meter.rate, meter.currency) == (
            operation_data["model"],
            operation_data["unit"],
            operation_data["rate"],
            data["currency"],
        )
        assert meter.model in data["models"]
    assert resolve_metered_operation_spec(registry, "qwen", "missing") is None
    assert resolve_metered_operation_spec(registry, "qwen", "pdf_pages_b") != resolve_metered_operation_spec(
        registry, "qwen", "pdf_pages_a"
    )
    assert {
        name: tuple(organization.models)
        for name, organization in registry.organizations.items()
        if name != "qwen"
    } == core_inventory_before
    assert dict(LLM_REGISTRY.organizations) == before_global

    assert tuple(field.name for field in dataclasses.fields(ModelLimits)) == (
        "context_window_tokens",
        "max_output_tokens",
    )
    assert tuple(field.name for field in dataclasses.fields(ModelSpec)) == (
        "name",
        "limits",
        "pricing_tiers",
        "reasoning_capability",
        "is_adaptive_thinking",
        "request_rules",
        "capability_exceptions",
        "provider_model",
    )
    assert tuple(inspect.signature(ModelLimits).parameters) == (
        "context_window_tokens",
        "max_output_tokens",
    )
    assert tuple(inspect.signature(ModelSpec).parameters) == (
        "name",
        "limits",
        "pricing_tiers",
        "reasoning_capability",
        "is_adaptive_thinking",
        "request_rules",
        "capability_exceptions",
        "provider_model",
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"rate_a": True}, "rate"),
        ({"rate_a": -0.01}, "rate"),
        ({"unit_a": ""}, "unit"),
        ({"currency": ""}, "currency"),
    ],
)
def test_qwen_meter_catalogue_reuses_core_meter_validation(change, message):
    from llm_api_adapter.llm_registry.llm_registry import (
        OrganizationModelMetadata,
        RegistrySpec,
    )

    data = _qwen_meter_catalogue(**change)
    assert validate_qwen_policy_catalogue(data) == {}
    with pytest.raises(ValueError, match=message):
        RegistrySpec().register_organization_metadata(
            OrganizationModelMetadata("qwen", data)
        )


@pytest.mark.unit
def test_qwen_zero_rate_meter_is_valid():
    from llm_api_adapter.llm_registry.llm_registry import (
        OrganizationModelMetadata,
        RegistrySpec,
        resolve_metered_operation_spec,
    )

    data = _qwen_meter_catalogue(rate_a=0)
    assert validate_qwen_policy_catalogue(data) == {}
    registry = RegistrySpec()
    registry.register_organization_metadata(OrganizationModelMetadata("qwen", data))

    meter = resolve_metered_operation_spec(registry, "qwen", "pdf_pages_a")
    assert meter is not None
    assert meter.model == data["metered_operations"]["pdf_pages_a"]["model"]
    assert meter.rate == 0


@pytest.mark.unit
def test_absent_qwen_extensions_preserve_loaded_profiles_and_core_registry():
    from llm_api_adapter.llm_registry.llm_registry import (
        LLM_REGISTRY,
        RegistrySpec,
        resolve_metered_operation_spec,
    )
    from llm_api_adapter_qwen.registry import MODEL_METADATA
    from llm_api_adapter_qwen.request_rules import (
        forced_tool_reasoning_modes,
        parse_qwen_request_rules,
    )

    before_global = dict(LLM_REGISTRY.organizations)
    registry = RegistrySpec()
    assert registry.register_organization_metadata(MODEL_METADATA) is True
    loaded = registry.organizations["qwen"]
    catalogue_models = MODEL_METADATA.organization_data["models"]
    assert tuple(loaded.models) == tuple(catalogue_models)
    assert loaded.metered_operations == {}
    assert resolve_metered_operation_spec(registry, "qwen", "pdf_pages_a") is None
    for model_name, model_data in catalogue_models.items():
        declared = model_data.get("qwen_policy", {}).get("request_rules", [])
        assert forced_tool_reasoning_modes(model_name) == parse_qwen_request_rules(declared)
        assert loaded.models[model_name].request_rules.rules == ()
    assert dict(LLM_REGISTRY.organizations) == before_global
