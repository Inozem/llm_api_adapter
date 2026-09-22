"""Package-local Z.ai cached-input pricing extensions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real


@dataclass(frozen=True)
class ZaiCacheCost:
    """Validated token cost returned by a cache-aware pricing calculation."""

    input_cost: float
    output_cost: float
    total_cost: float


@dataclass(frozen=True)
class ZaiCachePricing:
    """Rates for one model when the provider reports cached prompt tokens."""

    cache_hit_input_per_token: float
    cache_miss_input_per_token: float
    output_per_token: float

    def __post_init__(self) -> None:
        for field_name in (
            "cache_hit_input_per_token",
            "cache_miss_input_per_token",
            "output_per_token",
        ):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise ValueError(
                    f"Z.ai cache pricing {field_name} must be a finite "
                    "non-negative number"
                )

    def calculate(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int | None,
    ) -> ZaiCacheCost | None:
        """Calculate costs only when every cache split value is valid."""

        token_values = (input_tokens, output_tokens)
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in token_values
        ):
            return None
        if (
            cached_tokens is None
            or isinstance(cached_tokens, bool)
            or not isinstance(cached_tokens, int)
            or cached_tokens < 0
            or cached_tokens > input_tokens
        ):
            return None

        input_cost = (
            cached_tokens * self.cache_hit_input_per_token
            + (input_tokens - cached_tokens) * self.cache_miss_input_per_token
        )
        output_cost = output_tokens * self.output_per_token
        return ZaiCacheCost(
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=input_cost + output_cost,
        )


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


__all__ = ["CACHE_PRICING", "ZaiCacheCost", "ZaiCachePricing"]
