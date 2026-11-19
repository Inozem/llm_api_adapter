"""Validate capability profiles across every first-party model catalogue."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CORE_SOURCE = REPOSITORY_ROOT / "src"
KIMI_SOURCE = REPOSITORY_ROOT / "packages" / "organizations" / "kimi" / "src"
ZAI_SOURCE = REPOSITORY_ROOT / "packages" / "organizations" / "zai" / "src"
QWEN_SOURCE = REPOSITORY_ROOT / "packages" / "organizations" / "qwen" / "src"
for source in (CORE_SOURCE, KIMI_SOURCE, ZAI_SOURCE, QWEN_SOURCE):
    source_path = str(source)
    if source_path not in sys.path:
        sys.path.insert(0, source_path)

from llm_api_adapter.llm_registry.llm_registry import (
    CategoricalReasoningCapability,
    NumericReasoningCapability,
    OrganizationSpec,
)
_EXPECTED_ORGANIZATIONS = {
    "openai",
    "anthropic",
    "google",
    "mistral",
    "xai",
    "qwen",
    "kimi",
    "deepseek",
    "zai",
}
_CATALOGUE_PATHS = sorted(
    (REPOSITORY_ROOT / "src/llm_api_adapter/llm_registry/organizations").glob("*.json")
) + sorted(
    (REPOSITORY_ROOT / "packages/organizations").glob(
        "*/src/*/registry/organizations/*.json"
    )
)
FIRST_PARTY_CATALOGUES = tuple((path.stem, path) for path in _CATALOGUE_PATHS)

PACKAGE_REQUEST_RULE_REGISTRIES = {
    "kimi": importlib.import_module(
        "llm_api_adapter_kimi.request_rules"
    ).KIMI_REQUEST_RULE_REGISTRY,
    "zai": importlib.import_module(
        "llm_api_adapter_zai.request_rules"
    ).ZAI_REQUEST_RULE_REGISTRY,
}

TOOL_CHOICE_EXCEPTION_BY_RULE_MODE = {
    "auto": "tool_choice_auto",
    "none": "tool_choice_none",
    "any": "tool_choice_any",
    "tool": "tool_choice_named",
}

STANDARD_BEHAVIOR_IDS = {
    "pass",
    "ignored",
    "rejected_before_transport",
    "none_falls_back_to_minimum",
}
CUSTOM_BEHAVIOR_KEYS = {
    ("reasoning_control", "minimum_fallback_with_effort_alias"),
    ("provider_continuation", "stateless_reasoning_replay"),
}


def _catalogue_specs():
    for organization_name, catalogue_path in FIRST_PARTY_CATALOGUES:
        assert catalogue_path.is_file(), (
            f"missing first-party catalogue for {organization_name}: "
            f"{catalogue_path}"
        )
        organization_data = json.loads(catalogue_path.read_text(encoding="utf-8"))
        raw_models = organization_data.get("models")
        assert isinstance(raw_models, dict) and raw_models, (
            f"{organization_name} catalogue must define models"
        )

        missing_profiles = sorted(
            model_name
            for model_name, model_data in raw_models.items()
            if not isinstance(model_data, dict)
            or "capability_exceptions" not in model_data
        )
        assert not missing_profiles, (
            f"{organization_name} models missing explicit capability_exceptions: "
            f"{', '.join(missing_profiles)}"
        )

        if organization_name == "qwen":
            # Read the schema from the package metadata owner. Do not fabricate
            # a fallback schema for inventory parsing.
            qwen_metadata = importlib.import_module(
                "llm_api_adapter_qwen.registry"
            ).MODEL_METADATA
            request_rule_registry = qwen_metadata.request_rule_registry
        else:
            request_rule_registry = PACKAGE_REQUEST_RULE_REGISTRIES.get(
                organization_name
            )
        organization_spec = OrganizationSpec.from_dict(
            organization_name,
            organization_data,
            request_rule_registry=request_rule_registry,
        )
        assert tuple(organization_spec.models) == tuple(raw_models)
        yield organization_name, organization_data, organization_spec


@pytest.mark.unit
def test_mistral_glm_5_3_flash_has_inventory_profile():
    mistral_catalogue = next(
        (
            (organization_data, organization_spec)
            for organization_name, organization_data, organization_spec
            in _catalogue_specs()
            if organization_name == "mistral"
        ),
        None,
    )
    assert mistral_catalogue is not None, "Mistral catalogue is missing"
    organization_data, organization_spec = mistral_catalogue

    model_name = "glm-5.3-flash"
    assert model_name in organization_spec.models, (
        f"mistral/{model_name} is missing from the parsed catalogue"
    )
    raw_model_data = organization_data["models"][model_name]
    model_spec = organization_spec.models[model_name]

    assert model_spec.provider_model == raw_model_data.get("provider_model")
    assert model_spec.provider_model == "zai-glm-5-3-flash"
    assert model_spec.reasoning_capability.allowed_values == ("low", "high", "max")
    assert {
        (exception.capability_id, exception.behavior_id)
        for exception in model_spec.capability_exceptions
    } == {
        ("pdf_url", "pass"),
        ("pdf_bytes", "pass"),
        ("provider_continuation", "ignored"),
        ("reasoning_control", "none_falls_back_to_minimum"),
    }


@pytest.mark.unit
def test_every_first_party_model_has_a_valid_explicit_exception_profile():
    # Catalogue sizes evolve; validate every current entry instead of pinning the
    # current total of 57 models as a release limit.
    organizations = set()
    model_count = 0
    for organization_name, organization_data, organization_spec in _catalogue_specs():
        organizations.add(organization_name)
        model_count += len(organization_spec.models)
        for model_name, model_spec in organization_spec.models.items():
            assert model_spec.provider_model == organization_data["models"][
                model_name
            ].get("provider_model")
            assert model_spec.capability_exceptions is not None, (
                f"{organization_name}/{model_name} has no certified profile"
            )
            raw_exceptions = organization_data["models"][model_name][
                "capability_exceptions"
            ]
            assert all(
                isinstance(exception, dict) and "behavior_id" in exception
                for exception in raw_exceptions
            ), f"{organization_name}/{model_name} has an exception without behavior_id"
            assert tuple(
                (exception.capability_id, exception.behavior_id)
                for exception in model_spec.capability_exceptions
            ) == tuple(
                (exception["capability_id"], exception["behavior_id"])
                for exception in raw_exceptions
            )

    assert organizations == _EXPECTED_ORGANIZATIONS
    assert model_count > 0


@pytest.mark.unit
def test_reasoning_control_exceptions_match_reasoning_metadata():
    for organization_name, _, organization_spec in _catalogue_specs():
        for model_name, model_spec in organization_spec.models.items():
            exceptions = {
                exception.capability_id: exception
                for exception in model_spec.capability_exceptions
            }
            capability = model_spec.reasoning_capability
            if capability is None:
                expected_behavior_ids = {"ignored"}
            elif isinstance(capability, CategoricalReasoningCapability):
                expected_behavior_ids = (
                    {None}
                    if "none" in capability.allowed_values
                    else {
                        "none_falls_back_to_minimum",
                        "minimum_fallback_with_effort_alias",
                    }
                )
            elif isinstance(capability, NumericReasoningCapability):
                expected_behavior_ids = (
                    {None}
                    if capability.can_disable_thinking
                    else {
                        "none_falls_back_to_minimum",
                        "minimum_fallback_with_effort_alias",
                    }
                )
            else:  # pragma: no cover - union exhaustiveness guard
                raise AssertionError(f"unsupported reasoning capability: {capability!r}")

            actual_exception = exceptions.get("reasoning_control")
            actual_behavior_id = (
                actual_exception.behavior_id if actual_exception is not None else None
            )
            assert actual_behavior_id in expected_behavior_ids, (
                f"{organization_name}/{model_name} reasoning_control profile "
                f"disagrees with reasoning_capability: expected one of "
                f"{expected_behavior_ids!r}, got {actual_behavior_id!r}"
            )


@pytest.mark.unit
def test_provider_continuation_exceptions_match_transport_metadata():
    for organization_name, _, organization_spec in _catalogue_specs():
        for model_name, model_spec in organization_spec.models.items():
            exceptions = {
                exception.capability_id: exception
                for exception in model_spec.capability_exceptions
            }
            if model_spec.request_rules.api_variant == "responses":
                expected_behavior_ids = {None}
            else:
                expected_behavior_ids = {
                    "ignored",
                    "stateless_reasoning_replay",
                }

            actual_exception = exceptions.get("provider_continuation")
            actual_behavior_id = (
                actual_exception.behavior_id if actual_exception is not None else None
            )
            assert actual_behavior_id in expected_behavior_ids, (
                f"{organization_name}/{model_name} provider_continuation profile "
                f"disagrees with its transport: expected one of "
                f"{expected_behavior_ids!r}, got {actual_behavior_id!r}"
            )


@pytest.mark.unit
def test_exception_behavior_ids_use_the_shared_vocabulary_or_declared_custom_case():
    observed_custom_behaviors = set()
    for organization_name, _, organization_spec in _catalogue_specs():
        for model_name, model_spec in organization_spec.models.items():
            for exception in model_spec.capability_exceptions:
                if exception.behavior_id in STANDARD_BEHAVIOR_IDS:
                    continue

                behavior_key = (exception.capability_id, exception.behavior_id)
                assert behavior_key in CUSTOM_BEHAVIOR_KEYS, (
                    f"{organization_name}/{model_name} uses undeclared custom "
                    f"behavior {behavior_key!r}"
                )
                observed_custom_behaviors.add(behavior_key)

    assert observed_custom_behaviors == CUSTOM_BEHAVIOR_KEYS


@pytest.mark.unit
def test_tool_choice_exceptions_match_restricted_request_rule_metadata():
    for organization_name, _, organization_spec in _catalogue_specs():
        for model_name, model_spec in organization_spec.models.items():
            allowed_modes = model_spec.request_rules.allowed_tool_choice_modes
            if allowed_modes is None:
                continue

            exception_ids = {
                exception.capability_id
                for exception in model_spec.capability_exceptions
            }
            for mode, capability_id in TOOL_CHOICE_EXCEPTION_BY_RULE_MODE.items():
                is_exceptional = capability_id in exception_ids
                is_allowed = mode in allowed_modes
                assert is_exceptional == (not is_allowed), (
                    f"{organization_name}/{model_name} tool-choice profile for "
                    f"{capability_id} disagrees with restrict_tool_choice: "
                    f"allowed={sorted(allowed_modes)}"
                )


@pytest.mark.unit
def test_qwen_package_metadata_owns_the_schema_used_for_catalogue_parsing():
    qwen_registry = importlib.import_module("llm_api_adapter_qwen.registry")
    metadata = qwen_registry.MODEL_METADATA
    request_rules_module = importlib.import_module(
        "llm_api_adapter_qwen.request_rules"
    )
    qwen_rule_registry = request_rules_module.QWEN_REQUEST_RULE_REGISTRY

    assert metadata.organization == "qwen"
    assert metadata.request_rule_registry is qwen_rule_registry

    qwen_catalogue = next(
        (data, spec)
        for name, data, spec in _catalogue_specs()
        if name == "qwen"
    )
    organization_data, organization_spec = qwen_catalogue
    assert organization_data == metadata.organization_data
    registered = OrganizationSpec.from_dict(
        "qwen",
        metadata.organization_data,
        request_rule_registry=metadata.request_rule_registry,
    )
    fresh_registry = importlib.import_module(
        "llm_api_adapter.llm_registry.llm_registry"
    ).RegistrySpec()
    assert fresh_registry.register_organization_metadata(metadata) is True
    registered_via_registry = fresh_registry.organizations["qwen"]
    assert registered_via_registry == registered == organization_spec
    assert tuple(organization_spec.models) == (
        "qwen3.8-max",
        "qwen3.8-flash",
        "qwen3.7-plus",
        "qwen3.7-flash",
    )
    for model_name, model_spec in organization_spec.models.items():
        raw_model = organization_data["models"][model_name]
        assert model_spec.limits.context_window_tokens == raw_model["limits"][
            "context_window_tokens"
        ]
        assert model_spec.limits.max_output_tokens == raw_model["limits"][
            "max_output_tokens"
        ]
        assert model_spec.provider_model == raw_model.get("provider_model")
        assert model_spec.capability_exceptions is not None
        assert model_spec.reasoning_capability is not None
        assert model_spec.request_rules.disabled_reasoning_tool_choice_modes == frozenset(
            {"any", "tool"}
        )
        assert len(model_spec.pricing_tiers.tiers) == len(raw_model["pricing_tiers"])


@pytest.mark.unit
def test_qwen_forced_tool_schema_admits_only_supported_unique_modes():
    from llm_api_adapter.llm_registry.request_rules import (
        OpenAIRequestRuleRegistry,
        AnthropicRequestRuleRegistry,
        GoogleRequestRuleRegistry,
        RequestRule,
        RequestRuleRegistry,
        RequestRules,
    )

    qwen_registry = importlib.import_module("llm_api_adapter_qwen.registry")
    metadata = qwen_registry.MODEL_METADATA
    schema = metadata.request_rule_registry
    assert schema is not None, "Qwen package metadata must supply its rule schema"

    # This admission check ensures malformed cases below do not pass merely
    # because the handler is unknown to the package schema.
    semantic_rule = "disable_reasoning_for_forced_tools"
    for modes in (["any"], ["tool"], ["any", "tool"], ["tool", "any"]):
        valid_rules = RequestRules.from_dict(
            [{"handler": semantic_rule, "arguments": {"modes": modes}}],
            rule_registry=schema,
        )
        assert isinstance(valid_rules.rules[0], RequestRule)
        assert valid_rules.disabled_reasoning_tool_choice_modes == frozenset(modes)
    absent_policy = RequestRules.from_dict([], rule_registry=schema)
    assert absent_policy.disabled_reasoning_tool_choice_modes is None

    class ControlledSemanticSchema(RequestRuleRegistry):
        organization_name = "qwen-test"
        supported_handlers = frozenset({semantic_rule})

    controlled_schema = ControlledSemanticSchema()
    admitted = RequestRule.from_dict(
        {"handler": semantic_rule, "arguments": {"modes": ["any"]}},
        rule_registry=controlled_schema,
    )
    assert admitted.handler == semantic_rule
    for built_in_schema in (
        AnthropicRequestRuleRegistry(),
        OpenAIRequestRuleRegistry(),
        GoogleRequestRuleRegistry(),
    ):
        with pytest.raises(ValueError, match="not supported"):
            built_in_schema.validate_arguments(semantic_rule, {"modes": ["any"]})

    invalid_arguments = [
        ({"modes": []}, "non-empty"),
        ({}, "modes"),
        ({"modes": "any"}, "array"),
        ({"modes": [True]}, "strings"),
        ({"modes": ["any", 1]}, "strings"),
        ({"modes": ["auto"]}, "any.*tool"),
        ({"modes": ["any", "any"]}, "duplicate"),
        ({"modes": ["any"], "extra": True}, "only modes"),
    ]
    for arguments, message in invalid_arguments:
        with pytest.raises(ValueError, match=message):
            RequestRule.from_dict(
                {
                    "handler": "disable_reasoning_for_forced_tools",
                    "arguments": arguments,
                },
                rule_registry=schema,
            )


@pytest.mark.unit
def test_qwen_forced_tool_rules_reject_duplicate_and_conflicting_declarations():
    from llm_api_adapter.llm_registry.request_rules import RequestRules

    schema = importlib.import_module(
        "llm_api_adapter_qwen.registry"
    ).MODEL_METADATA.request_rule_registry
    assert schema is not None, "Qwen package metadata must supply its rule schema"

    semantic = {
        "handler": "disable_reasoning_for_forced_tools",
        "arguments": {"modes": ["any"]},
    }
    admitted = RequestRules.from_dict([semantic], rule_registry=schema)
    assert admitted.disabled_reasoning_tool_choice_modes == frozenset({"any"})
    for other_modes in (["any"], ["tool"]):
        duplicate = [semantic, {**semantic, "arguments": {"modes": other_modes}}]
        with pytest.raises(ValueError, match="more than once|duplicate"):
            RequestRules.from_dict(duplicate, rule_registry=schema)
