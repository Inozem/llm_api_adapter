"""Organization-scoped schemas for registry-backed request rule metadata.

The registry JSON selects only known handler IDs and validated data. It never
imports code, evaluates expressions, or carries arbitrary callback payloads.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import logging
from typing import Any, ClassVar, Dict, Mapping, Optional
import warnings


logger = logging.getLogger(__name__)
_MISSING = object()


class RequestRuleRegistry:
    """Base schema shared by organization-specific request rule registries."""

    SELECT_API_VARIANT: ClassVar[str] = "select_api_variant"
    RENAME_PARAMETER: ClassVar[str] = "rename_parameter"
    DROP_PARAMETER: ClassVar[str] = "drop_parameter"
    RESTRICT_TOOL_CHOICE: ClassVar[str] = "restrict_tool_choice"

    HANDLERS: ClassVar[frozenset[str]] = frozenset(
        {
            SELECT_API_VARIANT,
            RENAME_PARAMETER,
            DROP_PARAMETER,
            RESTRICT_TOOL_CHOICE,
        }
    )
    HANDLER_PHASES: ClassVar[Dict[str, int]] = {
        SELECT_API_VARIANT: 0,
        RESTRICT_TOOL_CHOICE: 0,
        RENAME_PARAMETER: 1,
        DROP_PARAMETER: 2,
    }

    organization_name: ClassVar[str] = "base"
    supported_handlers: ClassVar[frozenset[str]] = frozenset()
    supported_api_variants: ClassVar[frozenset[str]] = frozenset()
    supported_parameter_renames: ClassVar[frozenset[tuple[str, str]]] = (
        frozenset()
    )
    droppable_parameter_defaults: ClassVar[Dict[str, Any]] = {}
    supported_tool_choice_modes: ClassVar[frozenset[str]] = frozenset()

    def validate_arguments(self, handler: str, arguments: Dict[str, Any]) -> None:
        """Validate one rule against this organization's supported payload fields."""
        if handler not in self.HANDLERS:
            raise ValueError(f"unknown request rule handler: {handler!r}")
        if handler not in self.supported_handlers:
            raise ValueError(
                f"request rule handler {handler!r} is not supported for organization "
                f"{self.organization_name!r}"
            )

        if handler == self.SELECT_API_VARIANT:
            self._validate_api_variant(arguments)
        elif handler == self.RENAME_PARAMETER:
            self._validate_parameter_rename(arguments)
        elif handler == self.DROP_PARAMETER:
            self._validate_parameter_drop(arguments)
        else:
            self._validate_tool_choice_restriction(arguments)

    def _validate_api_variant(self, arguments: Dict[str, Any]) -> None:
        if set(arguments) != {"variant"}:
            raise ValueError(
                "select_api_variant arguments must contain only variant"
            )
        if arguments["variant"] not in self.supported_api_variants:
            supported = ", ".join(sorted(self.supported_api_variants))
            raise ValueError(
                "select_api_variant.variant must be one of: "
                f"{supported}"
            )

    def _validate_parameter_rename(self, arguments: Dict[str, Any]) -> None:
        if set(arguments) != {"from", "to"}:
            raise ValueError(
                "rename_parameter arguments must contain only from and to"
            )
        source = arguments["from"]
        target = arguments["to"]
        if (source, target) not in self.supported_parameter_renames:
            raise ValueError(
                f"unsupported request parameter rename: {source!r} -> {target!r}"
            )

    def _validate_parameter_drop(self, arguments: Dict[str, Any]) -> None:
        if set(arguments) not in ({"path"}, {"path", "default"}):
            raise ValueError(
                "drop_parameter arguments must contain path and optional default"
            )
        path = arguments["path"]
        self._validate_droppable_path(path)
        if "default" not in arguments:
            return
        default = arguments["default"]
        expected_default = self.droppable_parameter_defaults[path]
        if isinstance(default, bool) or default != expected_default:
            raise ValueError(
                f"request rule default for {path!r} must be {expected_default!r}"
            )

    def _validate_droppable_path(self, path: Any) -> None:
        if (
            not isinstance(path, str)
            or path not in self.droppable_parameter_defaults
        ):
            raise ValueError(f"unsupported request parameter path: {path!r}")

    def _validate_tool_choice_restriction(self, arguments: Dict[str, Any]) -> None:
        if set(arguments) != {"allowed_values"}:
            raise ValueError(
                "restrict_tool_choice arguments must contain only allowed_values"
            )
        allowed_values = arguments["allowed_values"]
        if not isinstance(allowed_values, list) or not allowed_values:
            raise ValueError(
                "restrict_tool_choice.allowed_values must be a non-empty array"
            )
        if any(
            not isinstance(value, str) or value not in self.supported_tool_choice_modes
            for value in allowed_values
        ):
            supported = ", ".join(sorted(self.supported_tool_choice_modes))
            raise ValueError(
                "restrict_tool_choice.allowed_values must contain only: "
                f"{supported}"
            )
        if len(set(allowed_values)) != len(allowed_values):
            raise ValueError(
                "restrict_tool_choice.allowed_values must not contain duplicates"
            )


class SamplingRequestRuleRegistry(RequestRuleRegistry):
    """Shared sampling-parameter rules used by OpenAI and Anthropic."""

    supported_handlers = frozenset(
        {
            RequestRuleRegistry.DROP_PARAMETER,
        }
    )
    droppable_parameter_defaults = {"top_p": 1.0}


class OpenAIRequestRuleRegistry(SamplingRequestRuleRegistry):
    organization_name = "openai"
    supported_handlers = SamplingRequestRuleRegistry.supported_handlers | frozenset(
        {
            RequestRuleRegistry.SELECT_API_VARIANT,
            RequestRuleRegistry.RENAME_PARAMETER,
        }
    )
    supported_api_variants = frozenset({"chat_completions", "responses"})
    supported_parameter_renames = frozenset(
        {
            ("max_tokens", "max_completion_tokens"),
        }
    )
    droppable_parameter_defaults = {
        **SamplingRequestRuleRegistry.droppable_parameter_defaults,
        "temperature": 1.0,
    }


class AnthropicRequestRuleRegistry(SamplingRequestRuleRegistry):
    organization_name = "anthropic"
    supported_handlers = SamplingRequestRuleRegistry.supported_handlers | frozenset(
        {RequestRuleRegistry.RESTRICT_TOOL_CHOICE}
    )
    supported_tool_choice_modes = frozenset({"auto", "none", "any", "tool"})


class GoogleRequestRuleRegistry(RequestRuleRegistry):
    organization_name = "google"
    supported_handlers = frozenset(
        {
            RequestRuleRegistry.DROP_PARAMETER,
        }
    )
    droppable_parameter_defaults = {
        "generationConfig.maxOutputTokens": None,
        "generationConfig.temperature": 1.0,
        "generationConfig.topP": 1.0,
    }


ORGANIZATION_REQUEST_RULE_REGISTRIES: Dict[str, RequestRuleRegistry] = {
    "openai": OpenAIRequestRuleRegistry(),
    "anthropic": AnthropicRequestRuleRegistry(),
    "google": GoogleRequestRuleRegistry(),
}
REQUEST_RULE_HANDLERS = RequestRuleRegistry.HANDLERS


@dataclass(frozen=True)
class RequestRule:
    """One validated request transformation selected by a model's metadata."""

    handler: str
    arguments: Dict[str, Any]

    @classmethod
    def from_dict(
        cls,
        data: Any,
        *,
        rule_registry: RequestRuleRegistry,
    ) -> "RequestRule":
        if not isinstance(data, dict) or set(data) != {"handler", "arguments"}:
            raise ValueError("request rule must contain only handler and arguments")

        handler = data["handler"]
        arguments = data["arguments"]
        if not isinstance(handler, str):
            raise ValueError(f"unknown request rule handler: {handler!r}")
        if not isinstance(arguments, dict):
            raise ValueError("request rule arguments must be an object")

        rule_registry.validate_arguments(handler, arguments)
        return cls(handler=handler, arguments=dict(arguments))

    @property
    def affected_paths(self) -> tuple[str, ...]:
        """Return the payload paths that this rule reads or writes."""
        if self.handler == RequestRuleRegistry.RENAME_PARAMETER:
            return (self.arguments["from"], self.arguments["to"])
        if self.handler == RequestRuleRegistry.DROP_PARAMETER:
            return (self.arguments["path"],)
        return ()


@dataclass(frozen=True)
class RequestRules:
    """Ordered, conflict-free request rules attached to one ``ModelSpec``."""

    rules: tuple[RequestRule, ...] = ()

    @classmethod
    def from_dict(
        cls,
        data: Any,
        *,
        rule_registry: RequestRuleRegistry,
    ) -> "RequestRules":
        if not isinstance(data, list):
            raise ValueError("request_rules must be an array")
        rules = tuple(
            RequestRule.from_dict(rule_data, rule_registry=rule_registry)
            for rule_data in data
        )
        cls._validate_rule_order(rules)
        return cls(rules=rules)

    @staticmethod
    def _validate_rule_order(rules: tuple[RequestRule, ...]) -> None:
        last_phase = -1
        selected_api_variant = False
        restricted_tool_choice = False
        affected_paths: set[str] = set()
        for rule in rules:
            phase = RequestRuleRegistry.HANDLER_PHASES[rule.handler]
            if phase < last_phase:
                raise ValueError("request rules have a conflicting execution order")
            last_phase = phase

            if rule.handler == RequestRuleRegistry.SELECT_API_VARIANT:
                if selected_api_variant:
                    raise ValueError("request rules select more than one API variant")
                selected_api_variant = True
            if rule.handler == RequestRuleRegistry.RESTRICT_TOOL_CHOICE:
                if restricted_tool_choice:
                    raise ValueError("request rules restrict tool choice more than once")
                restricted_tool_choice = True

            conflicts = affected_paths.intersection(rule.affected_paths)
            if conflicts:
                raise ValueError(
                    "request rules conflict on parameter path(s): "
                    f"{', '.join(sorted(conflicts))}"
                )
            affected_paths.update(rule.affected_paths)

    @property
    def api_variant(self) -> Optional[str]:
        """Return the model-selected OpenAI API variant, if one is declared."""
        for rule in self.rules:
            if rule.handler == RequestRuleRegistry.SELECT_API_VARIANT:
                return rule.arguments["variant"]
        return None

    @property
    def allowed_tool_choice_modes(self) -> Optional[frozenset[str]]:
        """Return model-specific normalized tool-choice modes, when restricted."""
        for rule in self.rules:
            if rule.handler == RequestRuleRegistry.RESTRICT_TOOL_CHOICE:
                return frozenset(rule.arguments["allowed_values"])
        return None


@dataclass(frozen=True)
class AppliedRequestRule:
    """One request rule that changed a payload during execution."""

    handler: str
    path: str
    target_path: Optional[str] = None
    warning_emitted: bool = False


def apply_request_rules(
    payload: Mapping[str, Any],
    request_rules: RequestRules,
    *,
    model: str,
) -> tuple[dict[str, Any], tuple[AppliedRequestRule, ...]]:
    """Apply registry request rules to a copied payload.

    API variant selection and tool-choice restrictions are metadata, so they
    are intentionally not payload transformations. A parameter with a
    declared default is removed quietly when its value matches that default;
    any other removed value emits one warning and one log record.
    """
    transformed_payload = copy.deepcopy(dict(payload))
    applied_rules: list[AppliedRequestRule] = []

    for rule in request_rules.rules:
        if rule.handler == RequestRuleRegistry.RENAME_PARAMETER:
            source = rule.arguments["from"]
            target = rule.arguments["to"]
            value = transformed_payload.pop(source, _MISSING)
            if value is _MISSING:
                continue
            transformed_payload[target] = value
            applied_rules.append(
                AppliedRequestRule(
                    handler=rule.handler,
                    path=source,
                    target_path=target,
                )
            )
            continue

        if rule.handler == RequestRuleRegistry.DROP_PARAMETER:
            path = rule.arguments["path"]
            value = _pop_payload_path(transformed_payload, path)
            if value is _MISSING:
                continue

            warning_emitted = False
            if "default" in rule.arguments and value != rule.arguments["default"]:
                _warn_ignored_parameter(path, model)
                warning_emitted = True
            applied_rules.append(
                AppliedRequestRule(
                    handler=rule.handler,
                    path=path,
                    warning_emitted=warning_emitted,
                )
            )

    return transformed_payload, tuple(applied_rules)


def _pop_payload_path(payload: dict[str, Any], path: str) -> Any:
    """Pop a validated dotted path, returning a sentinel when it is absent."""
    target = payload
    *parents, leaf = path.split(".")
    for parent in parents:
        nested = target.get(parent)
        if not isinstance(nested, dict):
            return _MISSING
        target = nested
    return target.pop(leaf, _MISSING)


def _warn_ignored_parameter(path: str, model: str) -> None:
    message = (
        f"Parameter {path!r} is not supported for model {model!r} and will be ignored."
    )
    warnings.warn(message, UserWarning, stacklevel=3)
    logger.warning(message)


def request_rule_registry_for_organization(
    organization_name: str,
) -> Optional[RequestRuleRegistry]:
    """Return an organization's closed rule schema, if it defines one."""
    return ORGANIZATION_REQUEST_RULE_REGISTRIES.get(organization_name)


__all__ = [
    "AppliedRequestRule",
    "AnthropicRequestRuleRegistry",
    "GoogleRequestRuleRegistry",
    "OpenAIRequestRuleRegistry",
    "REQUEST_RULE_HANDLERS",
    "ORGANIZATION_REQUEST_RULE_REGISTRIES",
    "RequestRule",
    "RequestRuleRegistry",
    "RequestRules",
    "SamplingRequestRuleRegistry",
    "apply_request_rules",
    "request_rule_registry_for_organization",
]
