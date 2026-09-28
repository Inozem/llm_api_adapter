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
for source in (CORE_SOURCE, KIMI_SOURCE, ZAI_SOURCE):
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
def test_every_first_party_model_has_a_valid_explicit_exception_profile():
    # Catalogue sizes evolve; validate every current entry instead of pinning the
    # current total of 57 models as a release limit.
    organizations = set()
    model_count = 0
    for organization_name, organization_data, organization_spec in _catalogue_specs():
        organizations.add(organization_name)
        model_count += len(organization_spec.models)
        for model_name, model_spec in organization_spec.models.items():
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
