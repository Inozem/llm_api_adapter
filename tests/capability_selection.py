"""Select deterministic conformance evidence for one exact model profile."""

from __future__ import annotations

from collections.abc import Iterable
import re

from llm_api_adapter.llm_registry.llm_registry import CapabilityException, ModelSpec
from llm_api_adapter.llm_registry.model_capabilities import (
    CAPABILITY_CATALOGUE,
    ModelCapability,
)
from tests.capability_scenarios import (
    CapabilityScenario,
    ExceptionScenario,
    PassSupplement,
    SCENARIO_CATALOGUE,
    ScenarioCatalogue,
)


_CAPABILITY_SCOPES = {entry.id: entry.scope for entry in CAPABILITY_CATALOGUE}
_BEHAVIOR_ID_PATTERN = re.compile(r"[a-z][a-z0-9_]*\Z")


def _index_routes(
    records: Iterable,
    *,
    route_name: str,
    record_type: type,
    key,
) -> dict:
    """Build an index while reporting malformed and duplicate evidence clearly."""
    indexed = {}
    try:
        iterator = iter(records)
    except TypeError as exc:
        raise ValueError(f"{route_name} evidence must be iterable") from exc

    for record in iterator:
        if not isinstance(record, record_type):
            raise ValueError(f"{route_name} evidence contains an invalid record")
        route_key = key(record)
        try:
            duplicate = route_key in indexed
        except TypeError as exc:
            raise ValueError(f"{route_name} evidence contains an invalid key") from exc
        if duplicate:
            raise ValueError(f"duplicate {route_name} evidence: {route_key!r}")
        indexed[route_key] = record
    return indexed


def _validate_capabilities(
    capabilities: Iterable[ModelCapability],
) -> tuple[ModelCapability, ...]:
    try:
        entries = tuple(capabilities)
    except TypeError as exc:
        raise ValueError("capabilities must be iterable") from exc

    seen = set()
    for capability in entries:
        if not isinstance(capability, ModelCapability):
            raise ValueError("capability entries must be ModelCapability values")
        if capability.id in seen:
            raise ValueError(f"duplicate requested capability: {capability.id!r}")
        seen.add(capability.id)
        canonical_scope = _CAPABILITY_SCOPES.get(capability.id)
        if canonical_scope is None:
            raise ValueError(f"unknown requested capability: {capability.id!r}")
        if capability.scope != canonical_scope:
            raise ValueError(
                f"incorrect scope for requested capability {capability.id!r}: "
                f"{capability.scope!r}"
            )
    return entries


def _validate_route_scopes(
    positive: dict[str, CapabilityScenario],
    always_on: dict[str, CapabilityScenario],
    exceptions: dict[tuple[str, str], ExceptionScenario],
    supplements: dict[tuple[str, str, str], PassSupplement],
) -> None:
    for capability_id, route in positive.items():
        if _CAPABILITY_SCOPES.get(capability_id) != "model-dependent":
            raise ValueError(
                f"positive evidence has unknown or non-model capability "
                f"{capability_id!r}"
            )
        if not isinstance(route.node_id, str) or not route.node_id:
            raise ValueError(f"positive evidence for {capability_id!r} has no node ID")

    for capability_id, route in always_on.items():
        if _CAPABILITY_SCOPES.get(capability_id) != "always-on":
            raise ValueError(
                f"always-on evidence has unknown or model-dependent capability "
                f"{capability_id!r}"
            )
        if not isinstance(route.node_id, str) or not route.node_id:
            raise ValueError(f"always-on evidence for {capability_id!r} has no node ID")

    for (capability_id, behavior_id), route in exceptions.items():
        if _CAPABILITY_SCOPES.get(capability_id) != "model-dependent":
            raise ValueError(
                f"exception evidence has unknown or non-model capability "
                f"{capability_id!r}"
            )
        if (
            not isinstance(behavior_id, str)
            or not _BEHAVIOR_ID_PATTERN.fullmatch(behavior_id)
            or behavior_id == "pass"
        ):
            raise ValueError(
                f"exception evidence for {capability_id!r} requires a non-pass "
                "behavior_id"
            )
        if not isinstance(route.node_id, str) or not route.node_id:
            raise ValueError(
                f"exception evidence for {capability_id}/{behavior_id} has no node ID"
            )

    for key, route in supplements.items():
        organization, model_name, capability_id = key
        if not all(isinstance(value, str) and value for value in key):
            raise ValueError(f"pass supplement requires an exact model key: {key!r}")
        if _CAPABILITY_SCOPES.get(capability_id) != "model-dependent":
            raise ValueError(f"pass supplement has unknown or non-model key: {key!r}")
        if not isinstance(route.node_id, str) or not route.node_id:
            raise ValueError(
                "pass supplement for "
                f"{organization}/{model_name}/{capability_id} has no node ID"
            )


def select_model_scenarios(
    *,
    organization: str,
    model: ModelSpec,
    scenarios: ScenarioCatalogue = SCENARIO_CATALOGUE,
    capabilities: Iterable[ModelCapability] = CAPABILITY_CATALOGUE,
) -> tuple[str, ...]:
    """Return the selected pytest node IDs for one exact model profile.

    Every requested model-dependent capability keeps its baseline-positive
    evidence unless its profile declares a non-``pass`` behavior. A ``pass``
    exception retains the baseline and adds only the supplement keyed by the
    exact organization, model name, and capability. Requested always-on routes
    are always included.
    """
    if not isinstance(organization, str) or not organization:
        raise ValueError("organization must be a non-empty string")
    if not isinstance(model, ModelSpec):
        raise ValueError("model must be a ModelSpec")
    if not isinstance(scenarios, ScenarioCatalogue):
        raise ValueError("scenarios must be a ScenarioCatalogue")

    requested = _validate_capabilities(capabilities)
    positive = _index_routes(
        scenarios.positive,
        route_name="baseline-positive",
        record_type=CapabilityScenario,
        key=lambda route: route.capability_id,
    )
    always_on = _index_routes(
        scenarios.always_on,
        route_name="always-on",
        record_type=CapabilityScenario,
        key=lambda route: route.capability_id,
    )
    exception_routes = _index_routes(
        scenarios.exceptions,
        route_name="exception",
        record_type=ExceptionScenario,
        key=lambda route: (route.capability_id, route.behavior_id),
    )
    supplements = _index_routes(
        scenarios.supplements,
        route_name="pass supplement",
        record_type=PassSupplement,
        key=lambda route: (route.organization, route.model, route.capability_id),
    )
    _validate_route_scopes(positive, always_on, exception_routes, supplements)

    declared_exceptions = model.require_capability_profile()
    exceptions_by_capability: dict[str, CapabilityException] = {}
    for exception in declared_exceptions:
        if not isinstance(exception, CapabilityException):
            raise ValueError(
                f"Model '{model.name}' has a malformed capability exception profile"
            )
        capability_id = exception.capability_id
        if _CAPABILITY_SCOPES.get(capability_id) != "model-dependent":
            raise ValueError(
                f"Model '{model.name}' declares an unknown or non-model capability "
                f"exception {capability_id!r}"
            )
        if capability_id in exceptions_by_capability:
            raise ValueError(
                f"Model '{model.name}' has duplicate capability exception "
                f"'{capability_id}'"
            )
        if (
            not isinstance(exception.behavior_id, str)
            or not _BEHAVIOR_ID_PATTERN.fullmatch(exception.behavior_id)
            or not isinstance(exception.behavior, str)
            or not exception.behavior.strip()
        ):
            raise ValueError(
                f"Model '{model.name}' has a malformed exception for "
                f"capability '{capability_id}'"
            )
        exceptions_by_capability[capability_id] = exception

    selected: list[str] = []
    selected_nodes = set()

    def add(node_id: str) -> None:
        # A single pytest case can provide evidence for more than one capability.
        if node_id not in selected_nodes:
            selected_nodes.add(node_id)
            selected.append(node_id)

    for capability in requested:
        capability_id = capability.id
        if capability.scope == "always-on":
            route = always_on.get(capability_id)
            if route is None:
                raise ValueError(
                    f"Model '{model.name}' has a coverage gap: missing unconditional "
                    f"scenario for capability '{capability_id}'"
                )
            add(route.node_id)
            continue

        baseline = positive.get(capability_id)
        if baseline is None:
            raise ValueError(
                f"Model '{model.name}' has a coverage gap: missing baseline-positive "
                f"scenario for capability '{capability_id}'"
            )

        exception = exceptions_by_capability.get(capability_id)
        if exception is None or exception.behavior_id == "pass":
            add(baseline.node_id)
            if exception is not None:
                exact_key = (organization, model.name, capability_id)
                supplement = supplements.get(exact_key)
                if supplement is None:
                    raise ValueError(
                        f"Model '{model.name}' in organization '{organization}' has "
                        "a coverage gap: missing exact-model pass supplement for "
                        f"capability '{capability_id}' (behavior_id 'pass')"
                    )
                add(supplement.node_id)
            continue

        route_key = (capability_id, exception.behavior_id)
        replacement = exception_routes.get(route_key)
        if replacement is None:
            raise ValueError(
                f"Model '{model.name}' has a coverage gap: missing exception "
                f"scenario for capability '{capability_id}' with behavior_id "
                f"'{exception.behavior_id}'"
            )
        add(replacement.node_id)

    return tuple(selected)


__all__ = ["select_model_scenarios"]
