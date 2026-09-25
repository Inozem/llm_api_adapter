"""Stable capability identifiers shared by model profiles and conformance checks.

The catalogue describes which behaviors can vary by exact model. Test scenario
identifiers and provider-specific decisions belong outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal


CapabilityScope = Literal["model-dependent", "always-on"]


@dataclass(frozen=True)
class ModelCapability:
    """One stable capability identifier and its profile scope."""

    id: str
    scope: CapabilityScope


CAPABILITY_CATALOGUE: tuple[ModelCapability, ...] = (
    ModelCapability("sync_chat", "model-dependent"),
    ModelCapability("async_chat", "model-dependent"),
    ModelCapability("sync_streaming", "model-dependent"),
    ModelCapability("async_streaming", "model-dependent"),
    ModelCapability("application_tools", "model-dependent"),
    ModelCapability("tool_choice_auto", "model-dependent"),
    ModelCapability("tool_choice_none", "model-dependent"),
    ModelCapability("tool_choice_any", "model-dependent"),
    ModelCapability("tool_choice_named", "model-dependent"),
    ModelCapability("structured_output_schema", "model-dependent"),
    ModelCapability("structured_output_model", "model-dependent"),
    ModelCapability("image_url", "model-dependent"),
    ModelCapability("image_bytes", "model-dependent"),
    ModelCapability("image_data_url", "model-dependent"),
    ModelCapability("pdf_url", "model-dependent"),
    ModelCapability("pdf_bytes", "model-dependent"),
    ModelCapability("reasoning_control", "model-dependent"),
    ModelCapability("reasoning_events", "model-dependent"),
    ModelCapability("provider_continuation", "model-dependent"),
    ModelCapability("usage_reporting", "model-dependent"),
    ModelCapability("refusal_outcome", "model-dependent"),
    ModelCapability("incomplete_outcome", "model-dependent"),
    ModelCapability("facade_discovery", "always-on"),
    ModelCapability("message_normalization", "always-on"),
    ModelCapability("response_normalization", "always-on"),
    ModelCapability("transport_parity", "always-on"),
    ModelCapability("stream_cleanup", "always-on"),
    ModelCapability("tool_validation", "always-on"),
    ModelCapability("schema_validation", "always-on"),
    ModelCapability("error_normalization", "always-on"),
    ModelCapability("registry_exactness", "always-on"),
    ModelCapability("request_rule_fidelity", "always-on"),
    ModelCapability("pricing_correctness", "always-on"),
    ModelCapability("missing_usage_honesty", "always-on"),
)

_CANONICAL_SCOPES = {entry.id: entry.scope for entry in CAPABILITY_CATALOGUE}
if len(_CANONICAL_SCOPES) != len(CAPABILITY_CATALOGUE):
    raise ValueError("duplicate capability identifier in canonical catalogue")

MODEL_DEPENDENT_CAPABILITY_IDS = frozenset(
    entry.id for entry in CAPABILITY_CATALOGUE if entry.scope == "model-dependent"
)
ALWAYS_ON_CAPABILITY_IDS = frozenset(
    entry.id for entry in CAPABILITY_CATALOGUE if entry.scope == "always-on"
)


def validate_capability_catalogue(entries: Iterable[ModelCapability]) -> None:
    """Reject IDs or scopes that differ from the version-controlled catalogue."""

    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, ModelCapability):
            raise ValueError("capability entry must be a ModelCapability")
        if entry.id in seen:
            raise ValueError(f"duplicate capability identifier: {entry.id!r}")
        seen.add(entry.id)
        if entry.id not in _CANONICAL_SCOPES:
            raise ValueError(f"unknown capability identifier: {entry.id!r}")
        if entry.scope not in ("model-dependent", "always-on"):
            raise ValueError(f"invalid capability scope for {entry.id!r}: {entry.scope!r}")
        if entry.scope != _CANONICAL_SCOPES[entry.id]:
            raise ValueError(f"incorrect capability scope for {entry.id!r}: {entry.scope!r}")

    missing = _CANONICAL_SCOPES.keys() - seen
    if missing:
        raise ValueError(f"missing capability identifiers: {', '.join(sorted(missing))}")


__all__ = [
    "ALWAYS_ON_CAPABILITY_IDS",
    "CAPABILITY_CATALOGUE",
    "CapabilityScope",
    "MODEL_DEPENDENT_CAPABILITY_IDS",
    "ModelCapability",
    "validate_capability_catalogue",
]
