"""Select shared E2E scenarios for one exact model profile."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from llm_api_adapter.llm_registry.llm_registry import ModelSpec
from llm_api_adapter.llm_registry.model_capabilities import (
    CAPABILITY_CATALOGUE,
    ModelCapability,
)
from tests.capability_scenarios import (
    ALWAYS_ON_SCENARIOS,
    BASELINE_SCENARIOS,
    E2E_SCENARIO_CAPABILITIES,
    EXCEPTION_SCENARIOS,
)


_CAPABILITY_SCOPES = {item.id: item.scope for item in CAPABILITY_CATALOGUE}


def _requested_capabilities(
    capabilities: Iterable[ModelCapability],
) -> tuple[ModelCapability, ...]:
    requested = tuple(capabilities)
    ids = [item.id for item in requested if isinstance(item, ModelCapability)]
    if len(ids) != len(requested):
        raise ValueError("capabilities must contain ModelCapability values")
    if len(ids) != len(set(ids)):
        raise ValueError("requested capabilities must be unique")
    for capability in requested:
        if _CAPABILITY_SCOPES.get(capability.id) != capability.scope:
            raise ValueError(f"unknown or incorrectly scoped capability: {capability.id!r}")
    return requested


def select_model_scenarios(
    *,
    organization: str,
    model: ModelSpec,
    capabilities: Iterable[ModelCapability] = E2E_SCENARIO_CAPABILITIES,
    baseline_scenarios: Mapping[str, str] = BASELINE_SCENARIOS,
    exception_scenarios: Mapping[tuple[str, str, str | None], str] = EXCEPTION_SCENARIOS,
    always_on_scenarios: Mapping[str, str] = ALWAYS_ON_SCENARIOS,
) -> tuple[str, ...]:
    """Return baseline routes, replacing only declared non-pass deviations."""
    if not isinstance(organization, str) or not organization:
        raise ValueError("organization must be a non-empty string")
    if not isinstance(model, ModelSpec):
        raise ValueError("model must be a ModelSpec")

    exceptions = {
        exception.capability_id: exception
        for exception in model.require_capability_profile()
    }
    selected = []
    for capability in _requested_capabilities(capabilities):
        capability_id = capability.id
        if capability.scope == "always-on":
            node_id = always_on_scenarios.get(capability_id)
            behavior_id = "always-on"
        else:
            exception = exceptions.get(capability_id)
            behavior_id = exception.behavior_id if exception else "baseline"
            if exception is None or behavior_id == "pass":
                node_id = baseline_scenarios.get(capability_id)
            else:
                node_id = exception_scenarios.get(
                    (capability_id, behavior_id, organization)
                ) or exception_scenarios.get((capability_id, behavior_id, None))

        if node_id is None:
            raise ValueError(
                f"{organization}/{model.name}: no scenario for capability_id="
                f"{capability_id}, behavior_id={behavior_id}"
            )
        selected.append(node_id)

    return tuple(dict.fromkeys(selected))


__all__ = ["select_model_scenarios"]
