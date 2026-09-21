"""Package-local Z.ai cached-input pricing extensions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ZaiCachePricing:
    """Rates for one model when the provider reports cached prompt tokens."""

    cache_hit_input_per_token: float
    cache_miss_input_per_token: float
    output_per_token: float


# Core's generic registry intentionally stores only the standard input/output
# tier.  Z.ai's cached-input rate is kept here so it cannot change the shared
# registry schema or affect other providers.
CACHE_PRICING: dict[str, ZaiCachePricing] = {
    "glm-5.3-flash": ZaiCachePricing(
        cache_hit_input_per_token=0.03 / 1_000_000,
        cache_miss_input_per_token=0.15 / 1_000_000,
        output_per_token=0.50 / 1_000_000,
    ),
}


__all__ = ["CACHE_PRICING", "ZaiCachePricing"]
