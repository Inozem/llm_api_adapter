"""Closed, package-local Qwen model policy declarations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


_SUPPORTED_HANDLERS = frozenset({"disable_reasoning_for_forced_tools"})
_SUPPORTED_MODES = frozenset({"any", "tool"})


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
    """Validate all Qwen extensions before Core metadata admission."""
    if not isinstance(organization_data, Mapping):
        raise ValueError("Qwen organization metadata must be an object")
    models = organization_data.get("models")
    if not isinstance(models, Mapping) or not models:
        raise ValueError("Qwen organization metadata must define models")

    policies: dict[str, frozenset[str]] = {}
    for model_name, model_data in models.items():
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("Qwen model names must be non-empty strings")
        if not isinstance(model_data, Mapping):
            raise ValueError(f"Qwen model '{model_name}' metadata must be an object")
        if "qwen_policy" not in model_data:
            continue

        policy = model_data["qwen_policy"]
        if not isinstance(policy, Mapping) or set(policy) != {"request_rules"}:
            raise ValueError(
                f"Qwen model '{model_name}' qwen_policy must contain only request_rules"
            )
        modes = parse_qwen_request_rules(
            policy["request_rules"],
            allowed_tool_choice_modes=_restricted_modes(
                model_data.get("request_rules", [])
            ),
        )
        if modes is not None:
            policies[model_name] = modes
    return policies


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
