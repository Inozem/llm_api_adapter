"""Validated, package-local DeepSeek Flash time-of-use pricing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timezone
from math import isfinite
from typing import Optional


_PER_MILLION = 1_000_000


@dataclass(frozen=True)
class DeepSeekFlashPricing:
    """One validated USD rate set, expressed per token."""

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
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or value < 0
            ):
                raise ValueError(
                    f"DeepSeek Flash {field_name} must be a non-negative finite number",
                )
            object.__setattr__(self, field_name, float(value))

    def is_valid(self) -> bool:
        """Re-check rates at use time in case an extension was altered."""
        return all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and isfinite(value)
            and value >= 0
            for value in (
                self.cache_hit_input_per_token,
                self.cache_miss_input_per_token,
                self.output_per_token,
            )
        )

    @classmethod
    def from_per_million(
        cls,
        *,
        cache_hit_input_per_1m: float,
        cache_miss_input_per_1m: float,
        output_per_1m: float,
    ) -> "DeepSeekFlashPricing":
        """Build rates from the published USD-per-million-token values."""
        values = (
            cache_hit_input_per_1m,
            cache_miss_input_per_1m,
            output_per_1m,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
            or value < 0
            for value in values
        ):
            raise ValueError(
                "DeepSeek Flash published rates must be non-negative finite numbers",
            )
        return cls(
            cache_hit_input_per_token=cache_hit_input_per_1m / _PER_MILLION,
            cache_miss_input_per_token=cache_miss_input_per_1m / _PER_MILLION,
            output_per_token=output_per_1m / _PER_MILLION,
        )


# DeepSeek publishes weekday peak windows in UTC.  Weekends use off-peak rates.
PEAK_PRICING = DeepSeekFlashPricing.from_per_million(
    cache_hit_input_per_1m=0.006,
    cache_miss_input_per_1m=0.3,
    output_per_1m=1.2,
)
OFF_PEAK_PRICING = DeepSeekFlashPricing.from_per_million(
    cache_hit_input_per_1m=0.003,
    cache_miss_input_per_1m=0.15,
    output_per_1m=0.6,
)


def pricing_for_dispatch(
    dispatch_time: datetime,
) -> Optional[DeepSeekFlashPricing]:
    """Select a verified rate set for an aware request-dispatch timestamp.

    Naive or otherwise unconvertible timestamps are deliberately treated as
    unverifiable instead of guessing a billing window.
    """
    if not isinstance(dispatch_time, datetime):
        return None
    if dispatch_time.tzinfo is None or dispatch_time.utcoffset() is None:
        return None
    try:
        utc_time = dispatch_time.astimezone(timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None

    if utc_time.weekday() >= 5:
        selected = OFF_PEAK_PRICING
    else:
        current = utc_time.time()
        is_peak = (
            time(1, 0) <= current < time(4, 0)
            or time(6, 0) <= current < time(10, 0)
        )
        selected = PEAK_PRICING if is_peak else OFF_PEAK_PRICING
    return (
        selected
        if isinstance(selected, DeepSeekFlashPricing) and selected.is_valid()
        else None
    )


__all__ = [
    "DeepSeekFlashPricing",
    "OFF_PEAK_PRICING",
    "PEAK_PRICING",
    "pricing_for_dispatch",
]
