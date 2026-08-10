"""Model-aware reasoning-level resolution.

The resolver intentionally knows only the public canonical scale and the
capability metadata stored on a model. Provider payload shape remains an
adapter concern.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import ceil
from typing import Optional

from ..errors.config_errors import LLMReasoningLevelError
from .llm_registry import (
    CategoricalReasoningCapability,
    ModelSpec,
    NumericReasoningCapability,
)


CANONICAL_REASONING_LEVELS = (
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "very_high",
)


@dataclass(frozen=True)
class ReasoningResolution:
    """The provider-ready result of resolving one public reasoning level."""

    provider_value: Optional[str | int]
    reason: str
    warning: Optional[str] = None


def resolve_reasoning_level(
    model_spec: ModelSpec,
    reasoning_level: str | int,
) -> ReasoningResolution:
    """Resolve a public string or integer level against one model's capability."""
    if isinstance(reasoning_level, bool) or not isinstance(
        reasoning_level,
        (str, int),
    ):
        raise LLMReasoningLevelError(
            message="Invalid reasoning level.",
            detail="reasoning_level must be a string or integer, not a boolean.",
        )

    capability = model_spec.reasoning_capability
    if capability is None:
        return ReasoningResolution(
            provider_value=None,
            reason="unsupported_model",
            warning=(
                f"Model '{model_spec.name}' does not support reasoning; "
                "reasoning is disabled."
            ),
        )
    if isinstance(capability, CategoricalReasoningCapability):
        return _resolve_categorical(
            model_spec,
            capability,
            reasoning_level,
        )
    if isinstance(capability, NumericReasoningCapability):
        return _resolve_numeric(capability, reasoning_level)
    raise TypeError(f"Unsupported reasoning capability: {type(capability).__name__}")


def _resolve_categorical(
    model_spec: ModelSpec,
    capability: CategoricalReasoningCapability,
    reasoning_level: str | int,
) -> ReasoningResolution:
    if isinstance(reasoning_level, str):
        if reasoning_level in capability.allowed_values:
            return ReasoningResolution(
                provider_value=reasoning_level,
                reason="categorical_direct_match",
            )

        if reasoning_level == "none":
            provider_value = _minimum_categorical_value(capability.allowed_values)
            return ReasoningResolution(
                provider_value=provider_value,
                reason="none_to_minimum",
                warning=(
                    f"Model '{model_spec.name}' cannot disable reasoning; "
                    f"using its minimum supported level '{provider_value}'."
                ),
            )

        position = _canonical_position(reasoning_level)
        provider_value = _categorical_value_for_percentage(
            capability.allowed_values,
            Fraction(position, len(CANONICAL_REASONING_LEVELS) - 1),
        )
        return ReasoningResolution(
            provider_value=provider_value,
            reason="categorical_canonical_projection",
        )

    context_window_tokens = model_spec.limits.context_window_tokens
    percentage = Fraction(
        min(max(reasoning_level, 0), context_window_tokens),
        context_window_tokens,
    )
    provider_value = _categorical_value_for_percentage(
        capability.allowed_values,
        percentage,
    )
    return ReasoningResolution(
        provider_value=provider_value,
        reason="categorical_context_window_percentage",
    )


def _resolve_numeric(
    capability: NumericReasoningCapability,
    reasoning_level: str | int,
) -> ReasoningResolution:
    if isinstance(reasoning_level, str):
        if reasoning_level == "none":
            if capability.min_budget_tokens == 0:
                return ReasoningResolution(
                    provider_value=0,
                    reason="numeric_disable",
                )
            return ReasoningResolution(
                provider_value=capability.min_budget_tokens,
                reason="none_to_minimum",
                warning=(
                    "The model cannot disable reasoning; using its minimum "
                    f"budget of {capability.min_budget_tokens} tokens."
                ),
            )

        position = _canonical_position(reasoning_level)
        provider_value = _interpolate_numeric_budget(capability, position)
        return ReasoningResolution(
            provider_value=provider_value,
            reason="numeric_canonical_interpolation",
        )

    if reasoning_level < capability.min_budget_tokens:
        return ReasoningResolution(
            provider_value=capability.min_budget_tokens,
            reason="numeric_minimum_fallback",
            warning=(
                f"Reasoning budget {reasoning_level} is below the model minimum "
                f"of {capability.min_budget_tokens}; using the minimum."
            ),
        )
    return ReasoningResolution(
        provider_value=reasoning_level,
        reason="numeric_budget",
    )


def _canonical_position(reasoning_level: str) -> int:
    try:
        return CANONICAL_REASONING_LEVELS.index(reasoning_level)
    except ValueError as error:
        raise LLMReasoningLevelError(
            message="Invalid reasoning level.",
            detail=(
                f"Unknown reasoning level {reasoning_level!r}. Valid canonical "
                f"levels: {list(CANONICAL_REASONING_LEVELS)}."
            ),
        ) from error


def _minimum_categorical_value(allowed_values: tuple[str, ...]) -> str:
    """Return the first usable thinking value, excluding an optional ``none``."""
    return next(
        (value for value in allowed_values if value != "none"),
        allowed_values[0],
    )


def _categorical_value_for_percentage(
    allowed_values: tuple[str, ...],
    percentage: Fraction,
) -> str:
    """Return the first native category at or above a normalized percentage.

    ``none`` is a zero-only value. When a model has no such value, its first
    native category starts the working scale, so small positive percentages and
    canonical ``minimal`` both resolve to that minimum instead of skipping it.
    """
    if percentage == 0 and "none" in allowed_values:
        return "none"

    working_values = tuple(value for value in allowed_values if value != "none")
    if not working_values:
        return allowed_values[0]

    index = max(ceil(percentage * len(working_values)) - 1, 0)
    return working_values[index]


def _interpolate_numeric_budget(
    capability: NumericReasoningCapability,
    position: int,
) -> int:
    # ``none`` is handled before interpolation. The remaining public levels
    # span the model's actual numeric interval from minimum to maximum.
    percentage = (position - 1) / (len(CANONICAL_REASONING_LEVELS) - 2)
    budget_range = capability.max_budget_tokens - capability.min_budget_tokens
    return capability.min_budget_tokens + ceil(percentage * budget_range)
