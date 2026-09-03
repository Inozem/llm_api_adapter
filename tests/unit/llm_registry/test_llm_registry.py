import json
from pathlib import Path

import pytest

from src.llm_api_adapter.llm_registry.llm_registry import (
    CategoricalReasoningCapability,
    DEFAULT_REGISTRY_PATH,
    ModelLimits,
    ModelSpec,
    MeteredOperationSpec,
    NumericReasoningCapability,
    Pricing,
    PricingTier,
    OrganizationModelMetadata,
    OrganizationSpec,
    RegistrySpec,
    LLM_REGISTRY,
    resolve_metered_operation_spec,
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


def _metered_operation_data(**overrides):
    data = {
        "model": "metered-test-model",
        "unit": "page",
        "rate": 0.004,
    }
    data.update(overrides)
    return data


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
def test_model_and_organization_from_dict():
    model_data = _model_data()

    model = ModelSpec.from_dict("test-model", model_data)

    assert model.name == "test-model"
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

    organization = OrganizationSpec.from_dict(
        "test-organization",
        {"currency": "EUR", "models": {"test-model": model_data}},
    )

    assert organization.name == "test-organization"
    assert organization.currency == "EUR"
    assert "test-model" in organization.models
    assert isinstance(organization.models["test-model"], ModelSpec)
    assert organization.models["test-model"].pricing_tiers.currency == "EUR"
    assert organization.metered_operations == {}


@pytest.mark.unit
def test_organization_loads_optional_metered_operations_separately_from_models():
    organization = OrganizationSpec.from_dict(
        "test-organization",
        {
            "currency": "EUR",
            "models": {"test-model": _model_data()},
            "metered_operations": {"ocr": _metered_operation_data()},
        },
    )

    meter = organization.metered_operations["ocr"]
    assert meter == MeteredOperationSpec(
        name="ocr",
        model="metered-test-model",
        unit="page",
        rate=0.004,
        currency="EUR",
    )
    assert isinstance(organization.models["test-model"], ModelSpec)
    assert not isinstance(organization.models["test-model"], MeteredOperationSpec)


@pytest.mark.unit
def test_metered_operation_allows_zero_rate():
    meter = MeteredOperationSpec.from_dict(
        "ocr",
        _metered_operation_data(rate=0),
        currency="USD",
    )

    assert meter.rate == 0


@pytest.mark.unit
def test_current_token_priced_organizations_load_without_metered_operations():
    registry = RegistrySpec(path=str(DEFAULT_REGISTRY_PATH))

    assert all(
        not organization.metered_operations
        for organization in registry.organizations.values()
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("name", "data", "currency", "message"),
    [
        (
            "",
            _metered_operation_data(),
            "USD",
            "metered operation name must be a non-empty string",
        ),
        (
            "ocr",
            None,
            "USD",
            "metered operation 'ocr' must be an object",
        ),
        (
            "ocr",
            {"unit": "page", "rate": 0.004},
            "USD",
            "metered operation must contain only model, unit, and rate",
        ),
        (
            "ocr",
            _metered_operation_data(model=""),
            "USD",
            "metered operation model must be a non-empty string",
        ),
        (
            "ocr",
            _metered_operation_data(unit=""),
            "USD",
            "metered operation unit must be a non-empty string",
        ),
        (
            "ocr",
            _metered_operation_data(rate=-0.004),
            "USD",
            "metered operation rate must be a non-negative number",
        ),
        (
            "ocr",
            _metered_operation_data(rate=True),
            "USD",
            "metered operation rate must be a non-negative number",
        ),
        (
            "ocr",
            _metered_operation_data(),
            "",
            "metered operation currency must be a non-empty string",
        ),
    ],
)
def test_metered_operation_spec_rejects_invalid_data(name, data, currency, message):
    with pytest.raises(ValueError, match=message):
        MeteredOperationSpec.from_dict(name, data, currency=currency)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("metered_operations", "message"),
    [
        (None, "metered_operations must be an object"),
        ([], "metered_operations must be an object"),
        ({"": _metered_operation_data()}, "name must be a non-empty string"),
        ({"ocr": None}, "metered operation 'ocr' must be an object"),
    ],
)
def test_organization_rejects_malformed_metered_operations(
    metered_operations,
    message,
):
    with pytest.raises(ValueError, match=message):
        OrganizationSpec.from_dict(
            "test-organization",
            {
                "models": {"test-model": _model_data()},
                "metered_operations": metered_operations,
            },
        )


@pytest.mark.unit
def test_chat_model_metadata_cannot_be_parsed_as_a_metered_operation():
    with pytest.raises(
        ValueError,
        match="metered operation must contain only model, unit, and rate",
    ):
        MeteredOperationSpec.from_dict(
            "ocr",
            _model_data(),
            currency="USD",
        )


@pytest.mark.unit
def test_plugin_organization_metadata_uses_the_existing_validation_and_lifecycle():
    class TestPluginRequestRuleRegistry(RequestRuleRegistry):
        organization_name = "test-plugin"
        supported_handlers = frozenset({RequestRuleRegistry.DROP_PARAMETER})
        droppable_parameter_defaults = {"temperature": 1.0}

    model_data = _model_data()
    model_data["request_rules"] = [
        {
            "handler": "drop_parameter",
            "arguments": {"path": "temperature", "default": 1.0},
        }
    ]
    metadata = OrganizationModelMetadata(
        organization="test-plugin",
        organization_data={
            "currency": "EUR",
            "models": {"test-model": model_data},
            "metered_operations": {"ocr": _metered_operation_data()},
        },
        request_rule_registry=TestPluginRequestRuleRegistry(),
    )
    registry = RegistrySpec()

    assert registry.register_organization_metadata(metadata) is True
    assert registry.register_organization_metadata(metadata) is False

    model = resolve_model_spec(registry, "test-plugin", "test-model")
    assert model is not None
    assert model.pricing_tiers.currency == "EUR"
    assert model.request_rules.rules[0].handler == "drop_parameter"
    meter = resolve_metered_operation_spec(registry, "test-plugin", "ocr")
    assert meter is not None
    assert meter.model == "metered-test-model"
    assert meter.rate == 0.004
    assert meter.currency == "EUR"
    assert resolve_metered_operation_spec(registry, "test-plugin", "missing") is None


@pytest.mark.unit
def test_invalid_plugin_organization_metadata_does_not_mutate_the_registry():
    registry = RegistrySpec()
    organizations_before = registry.organizations
    invalid_metadata = OrganizationModelMetadata(
        organization="test-plugin",
        organization_data={"models": {"test-model": {}}},
    )

    with pytest.raises(ValueError, match="Invalid organization metadata for 'test-plugin'"):
        registry.register_organization_metadata(invalid_metadata)

    assert registry.organizations is organizations_before
    assert "test-plugin" not in registry.organizations


@pytest.mark.unit
def test_conflicting_plugin_organization_metadata_is_rejected():
    registry = RegistrySpec()
    metadata = OrganizationModelMetadata(
        organization="test-plugin",
        organization_data={"models": {"test-model": _model_data()}},
    )
    changed_metadata = OrganizationModelMetadata(
        organization="test-plugin",
        organization_data={
            "models": {
                "test-model": _model_data(
                    tiers=[
                        {
                            "up_to_prompt_tokens": None,
                            "input_per_1m": 2_000,
                            "output_per_1m": 2_000,
                        }
                    ]
                )
            }
        },
    )

    assert registry.register_organization_metadata(metadata) is True
    with pytest.raises(ValueError, match="Organization metadata already registered"):
        registry.register_organization_metadata(changed_metadata)


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
        "test-model",
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
def test_anthropic_tool_choice_restriction_loads_as_validated_metadata():
    model_data = _model_data()
    model_data["request_rules"] = [
        {
            "handler": "restrict_tool_choice",
            "arguments": {"allowed_values": ["auto", "none"]},
        }
    ]

    model = ModelSpec.from_dict(
        "test-model",
        model_data,
        request_rule_registry=AnthropicRequestRuleRegistry(),
    )

    assert model.request_rules.allowed_tool_choice_modes == frozenset(
        {"auto", "none"}
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"allowed_values": "auto"}, "must be a non-empty array"),
        ({"allowed_values": ["auto", "forced"]}, "must contain only"),
        ({"allowed_values": ["auto", "auto"]}, "must not contain duplicates"),
        ({"allowed": ["auto"]}, "must contain only allowed_values"),
    ],
)
def test_anthropic_tool_choice_restriction_rejects_invalid_metadata(
    arguments,
    message,
):
    model_data = _model_data()
    model_data["request_rules"] = [
        {
            "handler": "restrict_tool_choice",
            "arguments": arguments,
        }
    ]

    with pytest.raises(ValueError, match=message):
        ModelSpec.from_dict(
            "test-model",
            model_data,
            request_rule_registry=AnthropicRequestRuleRegistry(),
        )


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
def test_request_rule_schemas_are_scoped_to_their_organization():
    model_data = _model_data()
    model_data["request_rules"] = [
        {
            "handler": "select_api_variant",
            "arguments": {"variant": "responses"},
        }
    ]

    with pytest.raises(ValueError, match="not supported for organization 'anthropic'"):
        ModelSpec.from_dict(
            "test-model",
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
            "test-model",
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
        "can_disable_thinking": True,
    }
    numeric = ModelSpec.from_dict("numeric-model", numeric_data)

    assert numeric.is_reasoning is True
    assert numeric.reasoning_capability == NumericReasoningCapability(
        min_budget_tokens=128,
        max_budget_tokens=32_768,
        can_disable_thinking=True,
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
            {
                "min_budget_tokens": 1,
                "max_budget_tokens": 2,
                "can_disable_thinking": 0,
            },
            "can_disable_thinking must be a boolean",
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
    organizations_dir = tmp_path / "organizations"
    organizations_dir.mkdir()
    organization_path = organizations_dir / "example_organization.json"
    organization_path.write_text(
        json.dumps({"models": {"example-model": _model_data()}}),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 42,
        "effective_date": "2030-01-01",
        "organization_files": {
            "example_organization": "organizations/example_organization.json",
        },
    }
    json_path = tmp_path / "llm_registry.json"
    json_path.write_text(json.dumps(manifest), encoding="utf-8")

    registry = RegistrySpec(path=str(json_path))

    assert registry.schema_version == 42
    assert registry.effective_date == "2030-01-01"
    model = registry.organizations["example_organization"].models["example-model"]
    assert model.limits.context_window_tokens == 128_000
    assert model.limits.max_output_tokens == 16_384
    assert model.pricing_tiers.tiers[0].in_per_token == pytest.approx(0.001)
    assert model.pricing_tiers.tiers[0].out_per_token == pytest.approx(0.002)


@pytest.mark.unit
def test_registry_rejects_missing_organization_file(tmp_path):
    json_path = tmp_path / "llm_registry.json"
    json_path.write_text(
        json.dumps(
            {
                "schema_version": 42,
                "effective_date": "2030-01-01",
                "organization_files": {"missing": "organizations/missing.json"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Organization registry file is missing"):
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
    organization_files = manifest["organization_files"]
    assert organization_files
    for relative_path in organization_files.values():
        assert (DEFAULT_REGISTRY_PATH.parent / relative_path).is_file()

    registry = RegistrySpec(path=str(DEFAULT_REGISTRY_PATH))
    assert set(registry.organizations) == set(organization_files)
    assert registry.schema_version == 13
    for organization in registry.organizations.values():
        for model in organization.models.values():
            assert model.limits.context_window_tokens > 0
            assert model.limits.max_output_tokens > 0
            assert model.pricing_tiers.tiers[-1].up_to_prompt_tokens is None
            if model.is_reasoning:
                assert model.reasoning_capability is not None
            else:
                assert model.reasoning_capability is None

    assert registry.organizations
    assert all(
        organization.models for organization in registry.organizations.values()
    )
