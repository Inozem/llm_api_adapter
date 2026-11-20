"""Closed, package-local Qwen model policy declarations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast


_SUPPORTED_HANDLERS = frozenset({"disable_reasoning_for_forced_tools"})
_SUPPORTED_MODES = frozenset({"any", "tool"})
_SUPPORTED_REASONING_MODES = frozenset({"enabled", "disabled"})
_SUPPORTED_POLICY_MEMBERS = frozenset(
    {"request_rules", "reasoning_modes", "input_token_accounting"}
)
_SUPPORTED_INPUT_TOKEN_ACCOUNTING = "ordinary_plus_cache_read_and_write"
ReasoningMode = Literal["enabled", "disabled"]
InputTokenAccounting = Literal["ordinary_plus_cache_read_and_write"]


@dataclass(frozen=True)
class ReasoningModeLimits:
    """One Qwen reasoning mode's published request limits."""

    max_input_tokens: int
    max_output_tokens: int


@dataclass(frozen=True)
class ReasoningModes:
    """Validated enabled/disabled reasoning limits and selected default."""

    default: ReasoningMode
    enabled: ReasoningModeLimits
    disabled: ReasoningModeLimits


@dataclass(frozen=True)
class QwenModelPolicy:
    """Validated immutable Qwen-only metadata for one exact model ID."""

    request_rule_modes: frozenset[str] | None = None
    reasoning_modes: ReasoningModes | None = None
    input_token_accounting: InputTokenAccounting | None = None


def _positive_policy_limit(value: Any, field: str, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Qwen reasoning_modes {field} must be a positive integer")
    if value > maximum:
        raise ValueError(f"Qwen reasoning_modes {field} exceeds the model limit")
    return value


def _parse_reasoning_modes(
    raw_modes: Any,
    model_data: Mapping[str, Any],
) -> ReasoningModes:
    if not isinstance(raw_modes, Mapping):
        raise ValueError("Qwen reasoning_modes must be an object")
    if set(raw_modes) != {"default", "enabled", "disabled"}:
        raise ValueError(
            "Qwen reasoning_modes must contain only default, enabled, and disabled"
        )
    default = raw_modes["default"]
    if not isinstance(default, str) or default not in _SUPPORTED_REASONING_MODES:
        raise ValueError("Qwen reasoning_modes default must be enabled or disabled")
    default = cast(ReasoningMode, default)

    # Use Core's existing limit validation only when the new optional metadata
    # is present. Legacy request-rule-only callers need no limits at all.
    from llm_api_adapter.llm_registry.llm_registry import ModelLimits

    raw_limits = model_data.get("limits")
    if not isinstance(raw_limits, Mapping):
        raise ValueError("Qwen reasoning_modes requires model limits")
    limits = ModelLimits.from_dict(dict(raw_limits))

    parsed: dict[str, ReasoningModeLimits] = {}
    for mode in ("enabled", "disabled"):
        record = raw_modes[mode]
        if not isinstance(record, Mapping):
            raise ValueError(f"Qwen reasoning_modes {mode} must be an object")
        if set(record) != {"max_input_tokens", "max_output_tokens"}:
            raise ValueError(
                f"Qwen reasoning_modes {mode} must contain only max_input_tokens "
                "and max_output_tokens"
            )
        parsed[mode] = ReasoningModeLimits(
            max_input_tokens=_positive_policy_limit(
                record["max_input_tokens"],
                "max_input_tokens",
                maximum=limits.context_window_tokens,
            ),
            max_output_tokens=_positive_policy_limit(
                record["max_output_tokens"],
                "max_output_tokens",
                maximum=limits.max_output_tokens,
            ),
        )
    return ReasoningModes(
        default=default,
        enabled=parsed["enabled"],
        disabled=parsed["disabled"],
    )


def parse_qwen_request_rules(
    raw_rules: Any,
    *,
    allowed_tool_choice_modes: frozenset[str] | None = None,
) -> frozenset[str] | None:
    """Validate a Qwen-only declaration and return its exact forced modes."""
    if not isinstance(raw_rules, list):
        raise ValueError("Qwen qwen_policy.request_rules must be an array")

    if not raw_rules:
        return None

    forced_tool_modes: frozenset[str] | None = None
    for index, declaration in enumerate(raw_rules):
        if not isinstance(declaration, Mapping):
            raise ValueError(f"Qwen request rule {index} must be an object")
        if set(declaration) != {"handler", "arguments"}:
            raise ValueError(
                f"Qwen request rule {index} must contain only handler and arguments"
            )
        handler = declaration["handler"]
        if not isinstance(handler, str) or handler not in _SUPPORTED_HANDLERS:
            raise ValueError(f"Qwen request rule handler is unknown: {handler!r}")
        if forced_tool_modes is not None:
            raise ValueError("Qwen request rule is declared more than once")

        arguments = declaration["arguments"]
        if not isinstance(arguments, Mapping):
            raise ValueError("Qwen forced-tool rule arguments must be an object")
        if set(arguments) != {"modes"}:
            raise ValueError("Qwen forced-tool rule arguments must contain only modes")
        modes = arguments["modes"]
        if not isinstance(modes, list):
            raise ValueError("Qwen forced-tool modes must be an array")
        if not modes:
            raise ValueError("Qwen forced-tool modes must be non-empty")
        if any(not isinstance(mode, str) or not mode for mode in modes):
            raise ValueError("Qwen forced-tool modes must contain only non-empty strings")
        if len(modes) != len(set(modes)):
            raise ValueError("Qwen forced-tool modes contain a duplicate")
        if not set(modes).issubset(_SUPPORTED_MODES):
            raise ValueError("Qwen forced-tool modes must be any and/or tool")

        forced_tool_modes = frozenset(modes)

    if (
        forced_tool_modes is not None
        and allowed_tool_choice_modes is not None
        and not forced_tool_modes.issubset(allowed_tool_choice_modes)
    ):
        raise ValueError(
            "Qwen forced-tool policy conflicts with allowed tool-choice modes"
        )
    return forced_tool_modes


def _restricted_modes(raw_core_rules: Any) -> frozenset[str] | None:
    """Read valid existing Core restrictions for cross-policy conflict checks."""
    if not isinstance(raw_core_rules, list):
        return None
    restricted_modes: list[frozenset[str]] = []
    for declaration in raw_core_rules:
        if not isinstance(declaration, Mapping):
            continue
        if declaration.get("handler") != "restrict_tool_choice":
            continue
        arguments = declaration.get("arguments")
        if not isinstance(arguments, Mapping):
            continue
        allowed = arguments.get("allowed_values")
        if (
            isinstance(allowed, list)
            and all(isinstance(mode, str) and mode for mode in allowed)
        ):
            restricted_modes.append(frozenset(allowed))
    if not restricted_modes:
        return None
    return frozenset.intersection(*restricted_modes)


def validate_qwen_policy_catalogue(
    organization_data: Any,
) -> dict[str, frozenset[str]]:
    """Validate Qwen extensions and preserve the legacy forced-mode projection."""
    request_policies, _ = _parse_qwen_policy_catalogue(organization_data)
    return request_policies


def _parse_qwen_policy_catalogue(
    organization_data: Any,
) -> tuple[dict[str, frozenset[str]], dict[str, QwenModelPolicy]]:
    """Parse once into the legacy request-rule view and typed optional metadata."""
    if not isinstance(organization_data, Mapping):
        raise ValueError("Qwen organization metadata must be an object")
    models = organization_data.get("models")
    if not isinstance(models, Mapping) or not models:
        raise ValueError("Qwen organization metadata must define models")

    policies: dict[str, frozenset[str]] = {}
    typed_policies: dict[str, QwenModelPolicy] = {}
    for model_name, model_data in models.items():
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("Qwen model names must be non-empty strings")
        if not isinstance(model_data, Mapping):
            raise ValueError(f"Qwen model '{model_name}' metadata must be an object")
        if "qwen_policy" not in model_data:
            continue

        policy = model_data["qwen_policy"]
        if not isinstance(policy, Mapping):
            raise ValueError(
                f"Qwen model '{model_name}' qwen_policy must contain only request_rules, "
                "reasoning_modes, and input_token_accounting; offending members: "
                f"{type(policy).__name__}"
            )
        unknown_members = set(policy) - _SUPPORTED_POLICY_MEMBERS
        if unknown_members:
            raise ValueError(
                f"Qwen model '{model_name}' qwen_policy must contain only request_rules, "
                "reasoning_modes, and input_token_accounting; offending members: "
                f"{', '.join(sorted(str(member) for member in unknown_members))}"
            )
        modes = None
        if "request_rules" in policy:
            modes = parse_qwen_request_rules(
                policy["request_rules"],
                allowed_tool_choice_modes=_restricted_modes(
                    model_data.get("request_rules", [])
                ),
            )
        reasoning_modes = (
            _parse_reasoning_modes(policy["reasoning_modes"], model_data)
            if "reasoning_modes" in policy
            else None
        )
        accounting: InputTokenAccounting | None = None
        if "input_token_accounting" in policy:
            raw_accounting = policy["input_token_accounting"]
            if (
                not isinstance(raw_accounting, str)
                or raw_accounting != _SUPPORTED_INPUT_TOKEN_ACCOUNTING
            ):
                raise ValueError(
                    "Qwen input_token_accounting must be "
                    "ordinary_plus_cache_read_and_write"
                )
            accounting = cast(InputTokenAccounting, raw_accounting)
        if modes is not None:
            policies[model_name] = modes
        if reasoning_modes is not None or accounting is not None:
            typed_policies[model_name] = QwenModelPolicy(
                request_rule_modes=modes,
                reasoning_modes=reasoning_modes,
                input_token_accounting=accounting,
            )
    return policies, typed_policies


def forced_tool_reasoning_modes(model_name: str) -> frozenset[str] | None:
    """Return only the exact model's declared Qwen forced-tool policy."""
    if not isinstance(model_name, str) or not model_name:
        return None
    from .registry import QWEN_MODEL_POLICIES

    return QWEN_MODEL_POLICIES.get(model_name)


__all__ = [
    "forced_tool_reasoning_modes",
    "parse_qwen_request_rules",
    "validate_qwen_policy_catalogue",
]
