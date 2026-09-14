"""Validated Kimi cache-hit/cache-miss price metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class KimiCachePricing:
    """One model's independently billed cached and uncached input rates."""

    cache_hit_input_per_token: float
    cache_miss_input_per_token: float
    output_per_token: float


def load_cache_pricing(organization_data: Mapping[str, Any]) -> dict[str, KimiCachePricing]:
    """Load the package extension which Core's generic pricing cannot express."""
    raw_models = organization_data.get("models")
    if not isinstance(raw_models, Mapping):
        raise ValueError("Kimi organization metadata must define models")

    pricing: dict[str, KimiCachePricing] = {}
    for model, raw_model in raw_models.items():
        if not isinstance(model, str) or not isinstance(raw_model, Mapping):
            raise ValueError("Kimi model metadata must be an object keyed by model ID")
        raw_pricing = raw_model.get("cache_pricing")
        if not isinstance(raw_pricing, Mapping) or set(raw_pricing) != {
            "cache_hit_input_per_1m",
            "cache_miss_input_per_1m",
            "output_per_1m",
        }:
            raise ValueError(f"Kimi cache_pricing is invalid for {model!r}")
        rates = tuple(raw_pricing.values())
        if any(
            isinstance(rate, bool) or not isinstance(rate, (int, float)) or rate < 0
            for rate in rates
        ):
            raise ValueError(f"Kimi cache_pricing rates must be non-negative for {model!r}")
        pricing[model] = KimiCachePricing(
            cache_hit_input_per_token=float(raw_pricing["cache_hit_input_per_1m"])
            / 1_000_000,
            cache_miss_input_per_token=float(raw_pricing["cache_miss_input_per_1m"])
            / 1_000_000,
            output_per_token=float(raw_pricing["output_per_1m"]) / 1_000_000,
        )
    return pricing


__all__ = ["KimiCachePricing", "load_cache_pricing"]
