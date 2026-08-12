import json
from datetime import date
from pathlib import Path

import pytest

from src.llm_api_adapter.llm_registry.llm_registry import (
    CategoricalReasoningCapability,
    DEFAULT_REGISTRY_PATH,
    ModelLimits,
    ModelSpec,
    NumericReasoningCapability,
    Pricing,
    PricingTier,
    ProviderSpec,
    RegistrySpec,
    LLM_REGISTRY,
    resolve_model_spec,
)
from src.llm_api_adapter.llm_registry.request_rules import (
    AnthropicRequestRuleRegistry,
    GoogleRequestRuleRegistry,
    OpenAIRequestRuleRegistry,
    RequestRule,
    RequestRuleRegistry,
    RequestRules,
    SamplingRequestRuleRegistry,
)


SONNET_5_STANDARD_PRICING_DATE = date(2026, 9, 1)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("provider_name", "model_name", "expected_base_name"),
    (
        ("anthropic", "claude-sonnet-4-5-20250929", "claude-sonnet-4-5"),
        ("openai", "gpt-5-2025-08-07", "gpt-5"),
        ("openai", "gpt-4.1-2025-04-14", "gpt-4.1"),
        ("google", "gemini-2.5-flash-preview-04-17", None),
        ("anthropic", "claude-sonnet-4-5-20251301", None),
        ("openai", "gpt-5-2025-13-01", None),
        ("openai", "ft:gpt-5:example:custom-2025-08-07", None),
    ),
)
def test_resolve_model_spec_supports_only_verified_provider_snapshots(
    provider_name,
    model_name,
    expected_base_name,
):
    model_spec = resolve_model_spec(LLM_REGISTRY, provider_name, model_name)

    if expected_base_name is None:
        assert model_spec is None
    else:
        assert model_spec is not None
        assert model_spec.name == expected_base_name


def _model_data(*, tiers=None):
    return {
        "limits": {
            "context_window_tokens": 128_000,
            "max_output_tokens": 16_384,
        },
        "pricing_tiers": tiers
        if tiers is not None
        else [
            {
                "up_to_prompt_tokens": None,
                "input_per_1m": 1_000,
                "output_per_1m": 2_000,
            }
        ],
    }


@pytest.mark.unit
def test_pricing_overrides_apply_to_every_tier():
    pricing = Pricing.from_dict(
        [
            {
                "up_to_prompt_tokens": 200_000,
                "input_per_1m": 1_500,
                "output_per_1m": 2_500,
            },
            {
                "up_to_prompt_tokens": None,
                "input_per_1m": 3_000,
                "output_per_1m": 4_000,
            },
        ],
        currency="USD",
    )

    pricing.set_in_per_1m(150.0)
    pricing.set_out_per_1m(250.0)
    pricing.set_currency("EUR")

    assert [tier.in_per_token for tier in pricing.tiers] == [150 / 1_000_000] * 2
    assert [tier.out_per_token for tier in pricing.tiers] == [250 / 1_000_000] * 2
    assert pricing.currency == "EUR"


@pytest.mark.unit
def test_pricing_override_allows_zero_rate_to_disable_one_cost_component():
    pricing = Pricing.from_dict(
        [{"up_to_prompt_tokens": None, "input_per_1m": 1, "output_per_1m": 1}],
        currency="USD",
    )

    pricing.set_in_per_1m(0)

    assert pricing.tiers[0].in_per_token == 0


@pytest.mark.unit
def test_pricing_tier_for_prompt_tokens_uses_inclusive_boundaries():
    pricing = Pricing.from_dict(
        [
            {
                "up_to_prompt_tokens": 200,
                "input_per_1m": 1,
                "output_per_1m": 2,
            },
            {
                "up_to_prompt_tokens": None,
                "input_per_1m": 3,
                "output_per_1m": 4,
            },
        ],
        currency="USD",
    )

    assert pricing.tier_for_prompt_tokens(200) is pricing.tiers[0]
    assert pricing.tier_for_prompt_tokens(201) is pricing.tiers[1]
    with pytest.raises(ValueError, match="non-negative integer"):
        pricing.tier_for_prompt_tokens(-1)


@pytest.mark.unit
def test_model_and_provider_from_dict():
    model_data = _model_data()

    model = ModelSpec.from_dict("gpt-test", model_data)

    assert model.name == "gpt-test"
    assert model.limits == ModelLimits(
        context_window_tokens=128_000,
        max_output_tokens=16_384,
    )
    assert model.pricing_tiers.tiers == (
        PricingTier(
            up_to_prompt_tokens=None,
            in_per_token=1_000 / 1_000_000,
            out_per_token=2_000 / 1_000_000,
        ),
    )

    provider = ProviderSpec.from_dict(
        "prov",
        {"currency": "EUR", "models": {"gpt-test": model_data}},
    )

    assert provider.name == "prov"
    assert provider.currency == "EUR"
    assert "gpt-test" in provider.models
    assert isinstance(provider.models["gpt-test"], ModelSpec)
    assert provider.models["gpt-test"].pricing_tiers.currency == "EUR"


@pytest.mark.unit
def test_model_request_rules_load_as_validated_metadata():
    model_data = _model_data()
    model_data["request_rules"] = [
        {
            "handler": "select_api_variant",
            "arguments": {"variant": "responses"},
        },
        {
            "handler": "drop_parameter",
            "arguments": {"path": "top_p", "default": 1.0},
        },
    ]

    model = ModelSpec.from_dict(
        "gpt-test",
        model_data,
        request_rule_registry=OpenAIRequestRuleRegistry(),
    )

    assert model.request_rules == RequestRules(
        rules=(
            RequestRule(
                handler="select_api_variant",
                arguments={"variant": "responses"},
            ),
            RequestRule(
                handler="drop_parameter",
                arguments={"path": "top_p", "default": 1.0},
            ),
        )
    )
    assert model.request_rules.api_variant == "responses"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("request_rules", "message"),
    [
        (None, "must be an array"),
        (
            [{"handler": "custom", "arguments": {}}],
            "unknown request rule handler",
        ),
        (
            [{"handler": "select_api_variant", "arguments": {"variant": "other"}}],
            "must be one of: chat_completions, responses",
        ),
        (
            [{"handler": "rename_parameter", "arguments": {"from": "top_p", "to": "topP"}}],
            "unsupported request parameter rename",
        ),
        (
            [{"handler": "drop_parameter", "arguments": {"path": "messages[0]"}}],
            "unsupported request parameter path",
        ),
        (
            [
                {
                    "handler": "drop_parameter",
                    "arguments": {"path": "top_p", "default": 0.5},
                }
            ],
            "default for 'top_p' must be 1.0",
        ),
        (
            [
                {
                    "handler": "drop_parameter",
                    "arguments": {"path": "top_p", "default": True},
                }
            ],
            "default for 'top_p' must be 1.0",
        ),
        (
            [
                {
                    "handler": "drop_parameter",
                    "arguments": {"path": "top_p", "custom": {}},
                }
            ],
            "must contain path and optional default",
        ),
    ],
)
def test_invalid_request_rule_metadata_is_rejected(request_rules, message):
    model_data = _model_data()
    model_data["request_rules"] = request_rules

    with pytest.raises(ValueError, match=message):
        ModelSpec.from_dict(
            "invalid-model",
            model_data,
            request_rule_registry=OpenAIRequestRuleRegistry(),
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("request_rules", "message"),
    [
        (
            [
                {
                    "handler": "drop_parameter",
                    "arguments": {"path": "top_p"},
                },
                {
                    "handler": "select_api_variant",
                    "arguments": {"variant": "responses"},
                },
            ],
            "conflicting execution order",
        ),
        (
            [
                {
                    "handler": "select_api_variant",
                    "arguments": {"variant": "responses"},
                },
                {
                    "handler": "select_api_variant",
                    "arguments": {"variant": "chat_completions"},
                },
            ],
            "more than one API variant",
        ),
        (
            [
                {
                    "handler": "drop_parameter",
                    "arguments": {"path": "top_p"},
                },
                {
                    "handler": "drop_parameter",
                    "arguments": {"path": "top_p", "default": 1.0},
                },
            ],
            "conflict on parameter path",
        ),
    ],
)
def test_conflicting_request_rule_order_is_rejected(request_rules, message):
    model_data = _model_data()
    model_data["request_rules"] = request_rules

    with pytest.raises(ValueError, match=message):
        ModelSpec.from_dict(
            "invalid-model",
            model_data,
            request_rule_registry=OpenAIRequestRuleRegistry(),
        )


@pytest.mark.unit
def test_request_rule_schemas_are_scoped_to_their_provider():
    model_data = _model_data()
    model_data["request_rules"] = [
        {
            "handler": "select_api_variant",
            "arguments": {"variant": "responses"},
        }
    ]

    with pytest.raises(ValueError, match="not supported for provider 'anthropic'"):
        ModelSpec.from_dict(
            "claude-test",
            model_data,
            request_rule_registry=AnthropicRequestRuleRegistry(),
        )

    model_data["request_rules"] = [
        {
            "handler": "drop_parameter",
            "arguments": {"path": "top_p", "default": 1.0},
        }
    ]
    with pytest.raises(ValueError, match="unsupported request parameter path"):
        ModelSpec.from_dict(
            "gemini-test",
            model_data,
            request_rule_registry=GoogleRequestRuleRegistry(),
        )


@pytest.mark.unit
def test_sampling_rule_schema_is_shared_by_openai_and_anthropic():
    assert AnthropicRequestRuleRegistry.droppable_parameter_defaults == {
        "top_p": 1.0
    }
    assert OpenAIRequestRuleRegistry.droppable_parameter_defaults["top_p"] == 1.0
    assert SamplingRequestRuleRegistry.DROP_PARAMETER == (
        RequestRuleRegistry.DROP_PARAMETER
    )


@pytest.mark.unit
def test_request_rules_need_a_provider_schema_when_present():
    model_data = _model_data()
    model_data["request_rules"] = [
        {
            "handler": "drop_parameter",
            "arguments": {"path": "top_p", "default": 1.0},
        }
    ]

    with pytest.raises(ValueError, match="without a provider schema"):
        ModelSpec.from_dict("model-with-rules", model_data)


@pytest.mark.unit
def test_model_reasoning_capabilities_load_as_verified_contracts():
    categorical_data = _model_data()
    categorical_data["reasoning_capability"] = {
        "allowed_values": ["none", "low", "medium", "high", "xhigh"],
    }
    categorical = ModelSpec.from_dict("categorical-model", categorical_data)

    assert categorical.is_reasoning is True
    assert categorical.reasoning_capability == CategoricalReasoningCapability(
        allowed_values=("none", "low", "medium", "high", "xhigh"),
    )

    numeric_data = _model_data()
    numeric_data["reasoning_capability"] = {
        "min_budget_tokens": 128,
        "max_budget_tokens": 32_768,
    }
    numeric = ModelSpec.from_dict("numeric-model", numeric_data)

    assert numeric.is_reasoning is True
    assert numeric.reasoning_capability == NumericReasoningCapability(
        min_budget_tokens=128,
        max_budget_tokens=32_768,
    )


@pytest.mark.unit
def test_non_reasoning_model_has_no_reasoning_capability():
    model = ModelSpec.from_dict("non-reasoning-model", _model_data())

    assert model.is_reasoning is False
    assert model.reasoning_capability is None


@pytest.mark.unit
@pytest.mark.parametrize(
    ("reasoning_capability", "message"),
    [
        (None, "must be an object"),
        ({}, "numeric reasoning_capability"),
        ({"allowed_values": []}, "non-empty array"),
        ({"allowed_values": ["low", "low"]}, "must not contain duplicates"),
        (
            {"min_budget_tokens": -1, "max_budget_tokens": 1},
            "non-negative integer",
        ),
        (
            {"min_budget_tokens": 2, "max_budget_tokens": 1},
            "must not exceed",
        ),
        (
            {"allowed_values": ["low"], "max_budget_tokens": 1},
            "categorical reasoning_capability",
        ),
    ],
)
def test_invalid_reasoning_capabilities_are_rejected(reasoning_capability, message):
    model_data = _model_data()
    model_data["reasoning_capability"] = reasoning_capability

    with pytest.raises(ValueError, match=message):
        ModelSpec.from_dict("invalid-model", model_data)


@pytest.mark.unit
def test_legacy_reasoning_flag_and_orphaned_adaptive_flag_are_rejected():
    legacy_data = _model_data()
    legacy_data["is_reasoning"] = True
    with pytest.raises(ValueError, match="legacy is_reasoning"):
        ModelSpec.from_dict("legacy-model", legacy_data)

    adaptive_data = _model_data()
    adaptive_data["is_adaptive_thinking"] = True
    with pytest.raises(ValueError, match="without reasoning_capability"):
        ModelSpec.from_dict("invalid-adaptive-model", adaptive_data)

    adaptive_data["reasoning_capability"] = {
        "min_budget_tokens": 0,
        "max_budget_tokens": 1,
    }
    with pytest.raises(ValueError, match="non-categorical"):
        ModelSpec.from_dict("invalid-adaptive-model", adaptive_data)


@pytest.mark.unit
def test_registry_reads_json_and_module_loads_llm_registry(tmp_path):
    providers_dir = tmp_path / "providers"
    providers_dir.mkdir()
    provider_path = providers_dir / "example_provider.json"
    provider_path.write_text(
        json.dumps({"models": {"example-model": _model_data()}}),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 42,
        "effective_date": "2030-01-01",
        "provider_files": {
            "example_provider": "providers/example_provider.json",
        },
    }
    json_path = tmp_path / "llm_registry.json"
    json_path.write_text(json.dumps(manifest), encoding="utf-8")

    registry = RegistrySpec(path=str(json_path))

    assert registry.schema_version == 42
    assert registry.effective_date == "2030-01-01"
    model = registry.providers["example_provider"].models["example-model"]
    assert model.limits.context_window_tokens == 128_000
    assert model.limits.max_output_tokens == 16_384
    assert model.pricing_tiers.tiers[0].in_per_token == pytest.approx(0.001)
    assert model.pricing_tiers.tiers[0].out_per_token == pytest.approx(0.002)


@pytest.mark.unit
def test_registry_rejects_missing_provider_file(tmp_path):
    json_path = tmp_path / "llm_registry.json"
    json_path.write_text(
        json.dumps(
            {
                "schema_version": 42,
                "effective_date": "2030-01-01",
                "provider_files": {"missing": "providers/missing.json"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Provider registry file is missing"):
        RegistrySpec(path=str(json_path))


@pytest.mark.unit
def test_legacy_flat_pricing_is_rejected():
    legacy_model = {
        "pricing": {"in_per_1m": 1_000, "out_per_1m": 2_000},
    }

    with pytest.raises(ValueError, match="legacy pricing"):
        ModelSpec.from_dict("legacy-model", legacy_model)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("tiers", "message"),
    [
        ([], "non-empty array"),
        (
            [
                {
                    "up_to_prompt_tokens": None,
                    "input_per_1m": 1,
                    "output_per_1m": 1,
                },
                {
                    "up_to_prompt_tokens": 200,
                    "input_per_1m": 1,
                    "output_per_1m": 1,
                },
            ],
            "null only for the final",
        ),
        (
            [
                {
                    "up_to_prompt_tokens": 200,
                    "input_per_1m": 1,
                    "output_per_1m": 1,
                },
                {
                    "up_to_prompt_tokens": 200,
                    "input_per_1m": 1,
                    "output_per_1m": 1,
                },
                {
                    "up_to_prompt_tokens": None,
                    "input_per_1m": 1,
                    "output_per_1m": 1,
                },
            ],
            "strictly increasing",
        ),
        (
            [
                {
                    "up_to_prompt_tokens": None,
                    "input_per_1m": 0,
                    "output_per_1m": 1,
                }
            ],
            "input_per_1m must be a positive number",
        ),
    ],
)
def test_invalid_pricing_tiers_are_rejected(tiers, message):
    with pytest.raises(ValueError, match=message):
        ModelSpec.from_dict("invalid-model", _model_data(tiers=tiers))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("limits", "message"),
    [
        ({"context_window_tokens": 0, "max_output_tokens": 1}, "context_window"),
        ({"context_window_tokens": 1, "max_output_tokens": 0}, "max_output"),
        ({"context_window_tokens": True, "max_output_tokens": 1}, "context_window"),
    ],
)
def test_invalid_limits_are_rejected(limits, message):
    model_data = _model_data()
    model_data["limits"] = limits

    with pytest.raises(ValueError, match=message):
        ModelSpec.from_dict("invalid-model", model_data)


@pytest.mark.unit
def test_default_registry_json_exists_and_uses_tiered_schema():
    assert isinstance(DEFAULT_REGISTRY_PATH, Path)
    assert DEFAULT_REGISTRY_PATH.is_file()

    manifest = json.loads(DEFAULT_REGISTRY_PATH.read_text(encoding="utf-8"))
    assert set(manifest["provider_files"]) == {"openai", "anthropic", "google"}
    for relative_path in manifest["provider_files"].values():
        assert (DEFAULT_REGISTRY_PATH.parent / relative_path).is_file()

    registry = RegistrySpec(path=str(DEFAULT_REGISTRY_PATH))
    assert registry.schema_version == 11
    for provider in registry.providers.values():
        for model in provider.models.values():
            assert model.limits.context_window_tokens > 0
            assert model.limits.max_output_tokens > 0
            assert model.pricing_tiers.tiers[-1].up_to_prompt_tokens is None
            if model.is_reasoning:
                assert model.reasoning_capability is not None
            else:
                assert model.reasoning_capability is None


@pytest.mark.unit
@pytest.mark.parametrize(
    ("provider", "model_name", "expected_rules"),
    [
        (
            "openai",
            "gpt-5",
            (
                ("select_api_variant", {"variant": "responses"}),
                (
                    "drop_parameter",
                    {"path": "top_p", "default": 1.0},
                ),
            ),
        ),
        (
            "openai",
            "gpt-5-nano",
            (
                ("select_api_variant", {"variant": "responses"}),
                (
                    "drop_parameter",
                    {"path": "top_p", "default": 1.0},
                ),
                (
                    "drop_parameter",
                    {"path": "temperature", "default": 1.0},
                ),
            ),
        ),
        (
            "openai",
            "gpt-4.1-mini",
            (
                (
                    "rename_parameter",
                    {"from": "max_tokens", "to": "max_completion_tokens"},
                ),
            ),
        ),
        (
            "anthropic",
            "claude-sonnet-4-5",
            (
                (
                    "drop_parameter",
                    {"path": "top_p", "default": 1.0},
                ),
            ),
        ),
        (
            "google",
            "gemini-2.5-flash",
            (
                (
                    "drop_parameter",
                    {"path": "generationConfig.maxOutputTokens", "default": None},
                ),
            ),
        ),
    ],
)
def test_default_registry_contains_current_request_rules(
    provider,
    model_name,
    expected_rules,
):
    registry = RegistrySpec(path=str(DEFAULT_REGISTRY_PATH))

    rules = registry.providers[provider].models[model_name].request_rules.rules

    assert tuple((rule.handler, rule.arguments) for rule in rules) == expected_rules


@pytest.mark.unit
@pytest.mark.parametrize(
    ("provider", "model_name", "expected_capability"),
    [
        (
            "openai",
            "gpt-5.6-sol",
            CategoricalReasoningCapability(
                ("none", "low", "medium", "high", "xhigh", "max")
            ),
        ),
        (
            "openai",
            "gpt-5.6-terra",
            CategoricalReasoningCapability(
                ("none", "low", "medium", "high", "xhigh", "max")
            ),
        ),
        (
            "openai",
            "gpt-5.6-luna",
            CategoricalReasoningCapability(
                ("none", "low", "medium", "high", "xhigh", "max")
            ),
        ),
        (
            "openai",
            "gpt-5.5",
            CategoricalReasoningCapability(
                ("none", "low", "medium", "high", "xhigh")
            ),
        ),
        (
            "openai",
            "gpt-5.4",
            CategoricalReasoningCapability(
                ("none", "low", "medium", "high", "xhigh")
            ),
        ),
        (
            "openai",
            "gpt-5.4-mini",
            CategoricalReasoningCapability(
                ("none", "low", "medium", "high", "xhigh")
            ),
        ),
        (
            "openai",
            "gpt-5.4-nano",
            CategoricalReasoningCapability(
                ("none", "low", "medium", "high", "xhigh")
            ),
        ),
        (
            "openai",
            "gpt-5.2",
            CategoricalReasoningCapability(
                ("none", "low", "medium", "high", "xhigh")
            ),
        ),
        (
            "openai",
            "gpt-5.1",
            CategoricalReasoningCapability(("none", "low", "medium", "high")),
        ),
        (
            "openai",
            "gpt-5",
            CategoricalReasoningCapability(("minimal", "low", "medium", "high")),
        ),
        (
            "openai",
            "gpt-5-mini",
            CategoricalReasoningCapability(("minimal", "low", "medium", "high")),
        ),
        (
            "openai",
            "gpt-5-nano",
            CategoricalReasoningCapability(("minimal", "low", "medium", "high")),
        ),
        (
            "anthropic",
            "claude-fable-5",
            CategoricalReasoningCapability(
                ("low", "medium", "high", "xhigh", "max")
            ),
        ),
        (
            "anthropic",
            "claude-opus-5",
            CategoricalReasoningCapability(
                ("low", "medium", "high", "xhigh", "max")
            ),
        ),
        (
            "anthropic",
            "claude-sonnet-5",
            CategoricalReasoningCapability(
                ("low", "medium", "high", "xhigh", "max")
            ),
        ),
        (
            "anthropic",
            "claude-opus-4-8",
            CategoricalReasoningCapability(
                ("low", "medium", "high", "xhigh", "max")
            ),
        ),
        (
            "anthropic",
            "claude-opus-4-7",
            CategoricalReasoningCapability(
                ("low", "medium", "high", "xhigh", "max")
            ),
        ),
        (
            "anthropic",
            "claude-opus-4-6",
            CategoricalReasoningCapability(("low", "medium", "high", "max")),
        ),
        (
            "anthropic",
            "claude-sonnet-4-6",
            CategoricalReasoningCapability(("low", "medium", "high", "max")),
        ),
        (
            "anthropic",
            "claude-opus-4-5",
            NumericReasoningCapability(1_024, 128_000),
        ),
        (
            "anthropic",
            "claude-sonnet-4-5",
            NumericReasoningCapability(1_024, 64_000),
        ),
        (
            "anthropic",
            "claude-haiku-4-5",
            NumericReasoningCapability(1_024, 64_000),
        ),
        (
            "google",
            "gemini-3.5-flash",
            CategoricalReasoningCapability(("minimal", "low", "medium", "high")),
        ),
        (
            "google",
            "gemini-3.1-pro-preview",
            CategoricalReasoningCapability(("low", "medium", "high")),
        ),
        (
            "google",
            "gemini-3.1-flash-lite",
            CategoricalReasoningCapability(("minimal", "low", "medium", "high")),
        ),
        (
            "google",
            "gemini-3-flash-preview",
            CategoricalReasoningCapability(("minimal", "low", "medium", "high")),
        ),
        (
            "google",
            "gemini-2.5-pro",
            NumericReasoningCapability(128, 32_768),
        ),
        (
            "google",
            "gemini-2.5-flash",
            NumericReasoningCapability(0, 24_576),
        ),
    ],
)
def test_default_registry_contains_verified_reasoning_capabilities(
    provider,
    model_name,
    expected_capability,
):
    registry = RegistrySpec(path=str(DEFAULT_REGISTRY_PATH))

    assert (
        registry.providers[provider].models[model_name].reasoning_capability
        == expected_capability
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("provider", "model_name", "context_window_tokens", "expected_tiers"),
    [
        ("openai", "gpt-5.6-sol", 1_050_000, ((272_000, 5.0, 30.0), (None, 10.0, 45.0))),
        ("openai", "gpt-5.6-terra", 1_050_000, ((272_000, 2.0, 12.0), (None, 4.0, 18.0))),
        ("openai", "gpt-5.6-luna", 1_050_000, ((272_000, 0.2, 1.2), (None, 0.4, 1.8))),
        ("openai", "gpt-5.5", 1_050_000, ((272_000, 5.0, 30.0), (None, 10.0, 45.0))),
        ("openai", "gpt-5.4", 1_050_000, ((272_000, 2.5, 15.0), (None, 5.0, 22.5))),
        ("openai", "gpt-5.4-mini", 400_000, ((None, 0.75, 4.5),)),
        ("openai", "gpt-5.4-nano", 400_000, ((None, 0.2, 1.25),)),
        ("openai", "gpt-5.2", 400_000, ((None, 1.75, 14.0),)),
        ("openai", "gpt-5.1", 400_000, ((None, 1.25, 10.0),)),
        ("openai", "gpt-5", 400_000, ((None, 1.25, 10.0),)),
        ("openai", "gpt-5-mini", 400_000, ((None, 0.25, 2.0),)),
        ("openai", "gpt-5-nano", 400_000, ((None, 0.05, 0.4),)),
        ("google", "gemini-3.1-pro-preview", 1_048_576, ((200_000, 2.0, 12.0), (None, 4.0, 18.0))),
        ("google", "gemini-2.5-pro", 1_048_576, ((200_000, 1.25, 10.0), (None, 2.5, 15.0))),
    ],
)
def test_default_registry_contains_verified_limits_and_pricing_tiers(
    provider,
    model_name,
    context_window_tokens,
    expected_tiers,
):
    registry = RegistrySpec(path=str(DEFAULT_REGISTRY_PATH))
    model = registry.providers[provider].models[model_name]

    assert model.limits.context_window_tokens == context_window_tokens
    assert len(model.pricing_tiers.tiers) == len(expected_tiers)
    for tier, (up_to_prompt_tokens, input_per_1m, output_per_1m) in zip(
        model.pricing_tiers.tiers,
        expected_tiers,
        strict=True,
    ):
        assert tier.up_to_prompt_tokens == up_to_prompt_tokens
        assert tier.in_per_token * 1_000_000 == pytest.approx(input_per_1m)
        assert tier.out_per_token * 1_000_000 == pytest.approx(output_per_1m)


@pytest.mark.unit
def test_default_registry_excludes_retired_claude_opus_4_1():
    registry = RegistrySpec(path=str(DEFAULT_REGISTRY_PATH))

    assert "claude-opus-4-1" not in registry.providers["anthropic"].models


@pytest.mark.unit
def test_sonnet_5_temporary_pricing_is_updated_after_promotion():
    registry = RegistrySpec(path=str(DEFAULT_REGISTRY_PATH))
    sonnet_5 = registry.providers["anthropic"].models["claude-sonnet-5"]
    if date.today() < SONNET_5_STANDARD_PRICING_DATE:
        expected_input, expected_output = 2.0, 10.0
    else:
        expected_input, expected_output = 3.0, 15.0
    tier = sonnet_5.pricing_tiers.tiers[0]
    assert pytest.approx(tier.in_per_token * 1_000_000) == expected_input
    assert pytest.approx(tier.out_per_token * 1_000_000) == expected_output
