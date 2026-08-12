"""Shared registry lookups for provider-client request payloads."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ..llm_registry.llm_registry import LLM_REGISTRY, ModelSpec
from ..llm_registry.request_rules import (
    AppliedRequestRule,
    RequestRules,
    apply_request_rules,
)


def model_spec_for(provider_name: str, model_name: str) -> Optional[ModelSpec]:
    """Return the verified model metadata for a provider, if it is known."""
    provider = LLM_REGISTRY.providers.get(provider_name)
    return provider.models.get(model_name) if provider else None


def model_api_variant(provider_name: str, model_name: str) -> Optional[str]:
    """Return the API variant selected by a model's registered request rules."""
    model_spec = model_spec_for(provider_name, model_name)
    return model_spec.request_rules.api_variant if model_spec else None


def apply_model_request_rules(
    provider_name: str,
    model_name: str,
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[AppliedRequestRule, ...]]:
    """Apply a verified model's request rules, or no rules for an unknown model."""
    model_spec = model_spec_for(provider_name, model_name)
    request_rules = model_spec.request_rules if model_spec else RequestRules()
    return apply_request_rules(payload, request_rules, model=model_name)


__all__ = [
    "apply_model_request_rules",
    "model_api_variant",
    "model_spec_for",
]
