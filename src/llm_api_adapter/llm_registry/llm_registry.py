from dataclasses import dataclass, field, replace
from datetime import date
import json
from pathlib import Path
import re
from typing import Any, Dict, Mapping, Optional, Sequence

from .request_rules import (
    RequestRuleRegistry,
    RequestRules,
    request_rule_registry_for_organization,
)

DEFAULT_REGISTRY_PATH = Path(__file__).with_name("llm_registry.json")

_ANTHROPIC_SNAPSHOT_ID = re.compile(r"^(claude-[a-z]+-\d+-\d+)-(\d{8})$")
_OPENAI_SNAPSHOT_ID = re.compile(r"^(gpt-[A-Za-z0-9.-]+)-(\d{4}-\d{2}-\d{2})$")


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _positive_rate(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{field_name} must be a positive number")
    return float(value)


def _non_negative_rate(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative number")
    return float(value)


def _non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class ModelLimits:
    """Provider-published context and output limits for one model."""

    context_window_tokens: int
    max_output_tokens: int

    @classmethod
    def from_dict(cls, data: Any) -> "ModelLimits":
        if not isinstance(data, dict):
            raise ValueError("limits must be an object")
        return cls(
            context_window_tokens=_positive_int(
                data.get("context_window_tokens"),
                "limits.context_window_tokens",
            ),
            max_output_tokens=_positive_int(
                data.get("max_output_tokens"),
                "limits.max_output_tokens",
            ),
        )


@dataclass(frozen=True)
class CategoricalReasoningCapability:
    """Ordered provider values for a model with categorical reasoning control."""

    allowed_values: tuple[str, ...]

    @classmethod
    def from_dict(cls, data: Any) -> "CategoricalReasoningCapability":
        if not isinstance(data, dict) or set(data) != {"allowed_values"}:
            raise ValueError(
                "categorical reasoning_capability must contain only allowed_values"
            )

        allowed_values = data["allowed_values"]
        if not isinstance(allowed_values, list) or not allowed_values:
            raise ValueError(
                "reasoning_capability.allowed_values must be a non-empty array"
            )
        if any(
            not isinstance(value, str) or not value
            for value in allowed_values
        ):
            raise ValueError(
                "reasoning_capability.allowed_values must contain non-empty strings"
            )
        if len(set(allowed_values)) != len(allowed_values):
            raise ValueError(
                "reasoning_capability.allowed_values must not contain duplicates"
            )

        return cls(allowed_values=tuple(allowed_values))


@dataclass(frozen=True)
class NumericReasoningCapability:
    """Provider token-budget bounds and whether thinking can be disabled."""

    min_budget_tokens: int
    max_budget_tokens: int
    can_disable_thinking: bool = False

    @classmethod
    def from_dict(cls, data: Any) -> "NumericReasoningCapability":
        required_fields = {"min_budget_tokens", "max_budget_tokens"}
        allowed_fields = required_fields | {
            "can_disable_thinking",
        }
        if not isinstance(data, dict) or not required_fields.issubset(data) or (
            set(data) - allowed_fields
        ):
            raise ValueError(
                "numeric reasoning_capability must contain "
                "min_budget_tokens, max_budget_tokens, and optional "
                "can_disable_thinking"
            )

        min_budget_tokens = _non_negative_int(
            data["min_budget_tokens"],
            "reasoning_capability.min_budget_tokens",
        )
        max_budget_tokens = _positive_int(
            data["max_budget_tokens"],
            "reasoning_capability.max_budget_tokens",
        )
        if min_budget_tokens > max_budget_tokens:
            raise ValueError(
                "reasoning_capability.min_budget_tokens must not exceed "
                "max_budget_tokens"
            )
        can_disable_thinking = data.get("can_disable_thinking", False)
        if not isinstance(can_disable_thinking, bool):
            raise ValueError(
                "reasoning_capability.can_disable_thinking must be a boolean"
            )
        return cls(
            min_budget_tokens=min_budget_tokens,
            max_budget_tokens=max_budget_tokens,
            can_disable_thinking=can_disable_thinking,
        )


ReasoningCapability = CategoricalReasoningCapability | NumericReasoningCapability


def _reasoning_capability_from_dict(data: Any) -> ReasoningCapability:
    if not isinstance(data, dict):
        raise ValueError("reasoning_capability must be an object")
    if "allowed_values" in data:
        return CategoricalReasoningCapability.from_dict(data)
    return NumericReasoningCapability.from_dict(data)


@dataclass(frozen=True)
class PricingTier:
    """One standard-rate band selected by reported prompt-token usage."""

    up_to_prompt_tokens: Optional[int]
    in_per_token: float
    out_per_token: float

    @classmethod
    def from_dict(cls, data: Any) -> "PricingTier":
        if not isinstance(data, dict):
            raise ValueError("pricing tier must be an object")

        boundary = data.get("up_to_prompt_tokens")
        if boundary is not None:
            boundary = _positive_int(boundary, "up_to_prompt_tokens")

        return cls(
            up_to_prompt_tokens=boundary,
            in_per_token=_positive_rate(data.get("input_per_1m"), "input_per_1m")
            / 1_000_000,
            out_per_token=_positive_rate(data.get("output_per_1m"), "output_per_1m")
            / 1_000_000,
        )


@dataclass(frozen=True)
class Pricing:
    """Ordered standard-rate tiers with public whole-model overrides."""

    tiers: tuple[PricingTier, ...]
    currency: str = "USD"

    @classmethod
    def from_dict(cls, data: Any, *, currency: str) -> "Pricing":
        if not isinstance(data, list) or not data:
            raise ValueError("pricing_tiers must be a non-empty array")

        tiers = tuple(PricingTier.from_dict(tier_data) for tier_data in data)
        cls._validate_tiers(tiers)
        return cls(tiers=tiers, currency=currency)

    @staticmethod
    def _validate_tiers(tiers: Sequence[PricingTier]) -> None:
        previous_boundary: Optional[int] = None
        for index, tier in enumerate(tiers):
            boundary = tier.up_to_prompt_tokens
            if boundary is None:
                if index != len(tiers) - 1:
                    raise ValueError(
                        "up_to_prompt_tokens may be null only for the final pricing tier"
                    )
                continue
            if previous_boundary is not None and boundary <= previous_boundary:
                raise ValueError("pricing tier boundaries must be strictly increasing")
            previous_boundary = boundary

    def set_in_per_1m(self, value: float) -> None:
        rate = _non_negative_rate(value, "input_per_1m") / 1_000_000
        object.__setattr__(
            self,
            "tiers",
            tuple(replace(tier, in_per_token=rate) for tier in self.tiers),
        )

    def set_out_per_1m(self, value: float) -> None:
        rate = _non_negative_rate(value, "output_per_1m") / 1_000_000
        object.__setattr__(
            self,
            "tiers",
            tuple(replace(tier, out_per_token=rate) for tier in self.tiers),
        )

    def set_currency(self, value: str) -> None:
        if not isinstance(value, str) or not value:
            raise ValueError("currency must be a non-empty string")
        object.__setattr__(self, "currency", value)

    def tier_for_prompt_tokens(self, prompt_tokens: int) -> PricingTier:
        """Return the rate tier selected by provider-reported prompt tokens."""
        if (
            isinstance(prompt_tokens, bool)
            or not isinstance(prompt_tokens, int)
            or prompt_tokens < 0
        ):
            raise ValueError("prompt_tokens must be a non-negative integer")

        for tier in self.tiers:
            if (
                tier.up_to_prompt_tokens is None
                or prompt_tokens <= tier.up_to_prompt_tokens
            ):
                return tier

        raise ValueError("pricing tiers do not cover the prompt token count")


@dataclass(frozen=True)
class MeteredOperationSpec:
    """Verified non-token price for one organization operation."""

    name: str
    model: str
    unit: str
    rate: float
    currency: str

    @classmethod
    def from_dict(
        cls,
        name: str,
        data: Any,
        *,
        currency: str,
    ) -> "MeteredOperationSpec":
        if not isinstance(name, str) or not name:
            raise ValueError("metered operation name must be a non-empty string")
        if not isinstance(data, Mapping):
            raise ValueError(f"metered operation '{name}' must be an object")
        if set(data) != {"model", "unit", "rate"}:
            raise ValueError(
                "metered operation must contain only model, unit, and rate"
            )

        model = data["model"]
        if not isinstance(model, str) or not model:
            raise ValueError("metered operation model must be a non-empty string")
        unit = data["unit"]
        if not isinstance(unit, str) or not unit:
            raise ValueError("metered operation unit must be a non-empty string")
        if not isinstance(currency, str) or not currency:
            raise ValueError("metered operation currency must be a non-empty string")

        return cls(
            name=name,
            model=model,
            unit=unit,
            rate=_non_negative_rate(data["rate"], "metered operation rate"),
            currency=currency,
        )


@dataclass(frozen=True)
class ModelSpec:
    name: str
    limits: ModelLimits
    pricing_tiers: Pricing
    reasoning_capability: Optional[ReasoningCapability] = None
    is_adaptive_thinking: bool = False
    request_rules: RequestRules = RequestRules()

    @property
    def is_reasoning(self) -> bool:
        """Backward-compatible marker derived from verified capability metadata."""
        return self.reasoning_capability is not None

    @classmethod
    def from_dict(
        cls,
        name: str,
        data: Mapping[str, Any],
        *,
        currency: str = "USD",
        request_rule_registry: Optional[RequestRuleRegistry] = None,
    ) -> "ModelSpec":
        if not isinstance(name, str) or not name:
            raise ValueError("model name must be a non-empty string")
        if not isinstance(data, Mapping):
            raise ValueError(f"Model '{name}' metadata must be an object")
        if "pricing" in data:
            raise ValueError(
                f"Model '{name}' uses legacy pricing; use pricing_tiers instead"
            )
        if "is_reasoning" in data:
            raise ValueError(
                f"Model '{name}' uses legacy is_reasoning; "
                "use reasoning_capability instead"
            )
        reasoning_capability = (
            _reasoning_capability_from_dict(data["reasoning_capability"])
            if "reasoning_capability" in data
            else None
        )
        is_adaptive_thinking = data.get("is_adaptive_thinking", False)
        if not isinstance(is_adaptive_thinking, bool):
            raise ValueError(
                f"Model '{name}' is_adaptive_thinking must be a boolean"
            )
        raw_request_rules = data.get("request_rules", [])
        if request_rule_registry is None:
            if not isinstance(raw_request_rules, list):
                raise ValueError("request_rules must be an array")
            if raw_request_rules:
                raise ValueError(
                    f"Model '{name}' defines request_rules without a provider schema"
                )
            request_rules = RequestRules()
        else:
            request_rules = RequestRules.from_dict(
                raw_request_rules,
                rule_registry=request_rule_registry,
            )
        if is_adaptive_thinking and reasoning_capability is None:
            raise ValueError(
                f"Model '{name}' enables adaptive thinking without "
                "reasoning_capability"
            )
        if is_adaptive_thinking and not isinstance(
            reasoning_capability,
            CategoricalReasoningCapability,
        ):
            raise ValueError(
                f"Model '{name}' enables adaptive thinking with a non-categorical "
                "reasoning_capability"
            )
        return cls(
            name=name,
            limits=ModelLimits.from_dict(data.get("limits")),
            pricing_tiers=Pricing.from_dict(
                data.get("pricing_tiers"),
                currency=currency,
            ),
            reasoning_capability=reasoning_capability,
            is_adaptive_thinking=is_adaptive_thinking,
            request_rules=request_rules,
        )


@dataclass(frozen=True)
class OrganizationSpec:
    name: str
    currency: str = "USD"
    models: Dict[str, ModelSpec] = field(default_factory=dict)
    metered_operations: Dict[str, MeteredOperationSpec] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls,
        name: str,
        data: Mapping[str, Any],
        *,
        request_rule_registry: Optional[RequestRuleRegistry] = None,
    ) -> "OrganizationSpec":
        if not isinstance(name, str) or not name:
            raise ValueError("organization name must be a non-empty string")
        if not isinstance(data, Mapping):
            raise ValueError(f"Organization '{name}' metadata must be an object")
        currency = data.get("currency", "USD")
        if not isinstance(currency, str) or not currency:
            raise ValueError(f"Organization '{name}' has an invalid currency")
        if request_rule_registry is None:
            request_rule_registry = request_rule_registry_for_organization(name)
        elif request_rule_registry.organization_name != name:
            raise ValueError(
                "request rule schema organization must match organization metadata: "
                f"{name}"
            )
        raw_models = data.get("models")
        if not isinstance(raw_models, Mapping) or not raw_models:
            raise ValueError(
                f"Organization '{name}' must define a non-empty models object"
            )
        if any(
            not isinstance(model_name, str) or not model_name
            for model_name in raw_models
        ):
            raise ValueError(
                f"Organization '{name}' model names must be non-empty strings"
            )
        models = {
            model_name: ModelSpec.from_dict(
                model_name,
                model_spec,
                currency=currency,
                request_rule_registry=request_rule_registry,
            )
            for model_name, model_spec in raw_models.items()
        }
        raw_metered_operations = data.get("metered_operations", {})
        if not isinstance(raw_metered_operations, Mapping):
            raise ValueError(
                f"Organization '{name}' metered_operations must be an object"
            )
        metered_operations = {
            operation_name: MeteredOperationSpec.from_dict(
                operation_name,
                operation_spec,
                currency=currency,
            )
            for operation_name, operation_spec in raw_metered_operations.items()
        }
        return cls(
            name=name,
            currency=currency,
            models=models,
            metered_operations=metered_operations,
        )


@dataclass(frozen=True)
class OrganizationModelMetadata:
    """Validated model catalogue supplied by one external organization plugin."""

    organization: str
    organization_data: Mapping[str, Any]
    request_rule_registry: Optional[RequestRuleRegistry] = None


@dataclass(frozen=True, init=False)
class RegistrySpec:
    schema_version: int
    effective_date: str
    organizations: Dict[str, OrganizationSpec]

    def __init__(self, path: str | Path = DEFAULT_REGISTRY_PATH) -> None:
        manifest_path = Path(path)
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        organization_files = data.get("organization_files")
        if not isinstance(organization_files, dict) or not organization_files:
            raise ValueError("registry manifest must define a non-empty organization_files object")

        organizations = {
            organization_name: OrganizationSpec.from_dict(
                organization_name,
                self._load_organization_data(
                    manifest_path,
                    organization_name,
                    organization_file,
                ),
            )
            for organization_name, organization_file in organization_files.items()
        }
        object.__setattr__(self, "schema_version", int(data["schema_version"]))
        object.__setattr__(self, "effective_date", str(data["effective_date"]))
        object.__setattr__(self, "organizations", organizations)

    @staticmethod
    def _load_organization_data(
        manifest_path: Path,
        organization_name: str,
        organization_file: Any,
    ) -> Dict[str, Any]:
        if not isinstance(organization_name, str) or not organization_name:
            raise ValueError("organization_files keys must be non-empty strings")
        if not isinstance(organization_file, str) or not organization_file:
            raise ValueError(
                f"Organization '{organization_name}' must reference a non-empty relative path"
            )

        organization_path = Path(organization_file)
        if organization_path.is_absolute():
            raise ValueError(
                f"Organization '{organization_name}' must use a relative registry path"
            )
        organization_path = manifest_path.parent / organization_path
        try:
            organization_data = json.loads(organization_path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise ValueError(
                f"Organization registry file is missing for '{organization_name}': {organization_path}"
            ) from error

        if not isinstance(organization_data, dict):
            raise ValueError(f"Organization registry data for '{organization_name}' must be an object")
        return organization_data

    def register_organization_metadata(
        self,
        metadata: OrganizationModelMetadata,
    ) -> bool:
        """Validate and atomically register one external organization catalogue."""
        if not isinstance(metadata, OrganizationModelMetadata):
            raise TypeError("metadata must be an OrganizationModelMetadata instance")
        if not isinstance(metadata.organization, str) or not metadata.organization:
            raise ValueError("organization metadata organization must be a non-empty string")

        try:
            organization = OrganizationSpec.from_dict(
                metadata.organization,
                metadata.organization_data,
                request_rule_registry=metadata.request_rule_registry,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid organization metadata for '{metadata.organization}': {error}"
            ) from error

        existing_organization = self.organizations.get(metadata.organization)
        if existing_organization is None:
            object.__setattr__(
                self,
                "organizations",
                {**self.organizations, metadata.organization: organization},
            )
            return True
        if existing_organization == organization:
            return False
        raise ValueError(
            "Organization metadata already registered: "
            f"{metadata.organization}"
        )


def _is_valid_snapshot_date(value: str, *, compact: bool) -> bool:
    """Return whether an organization snapshot suffix is a real calendar date."""
    if compact:
        value = f"{value[:4]}-{value[4:6]}-{value[6:]}"
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _snapshot_base_model_name(organization_name: str, model_name: str) -> Optional[str]:
    """Return a supported direct-organization snapshot's base model ID."""
    if organization_name == "anthropic":
        match = _ANTHROPIC_SNAPSHOT_ID.fullmatch(model_name)
        if match and _is_valid_snapshot_date(match.group(2), compact=True):
            return match.group(1)
    elif organization_name == "openai":
        match = _OPENAI_SNAPSHOT_ID.fullmatch(model_name)
        if match and _is_valid_snapshot_date(match.group(2), compact=False):
            return match.group(1)
    return None


def resolve_model_spec(
    registry: RegistrySpec,
    organization_name: str,
    model_name: str,
) -> Optional[ModelSpec]:
    """Resolve an exact model or a supported organization snapshot to its base spec.

    The caller must continue to send ``model_name`` to the organization API. This
    resolver only supplies verified registry metadata for direct Anthropic and
    OpenAI snapshot IDs whose unsuffixed base is registered.
    """
    organization = registry.organizations.get(organization_name)
    if not organization:
        return None

    model_spec = organization.models.get(model_name)
    if model_spec:
        return model_spec

    base_model_name = _snapshot_base_model_name(organization_name, model_name)
    return organization.models.get(base_model_name) if base_model_name else None


def resolve_metered_operation_spec(
    registry: RegistrySpec,
    organization_name: str,
    operation_name: str,
) -> Optional[MeteredOperationSpec]:
    """Resolve a named non-token meter from registered organization metadata."""
    organization = registry.organizations.get(organization_name)
    return organization.metered_operations.get(operation_name) if organization else None


LLM_REGISTRY = RegistrySpec()
