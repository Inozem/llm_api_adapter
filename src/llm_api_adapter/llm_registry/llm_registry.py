from dataclasses import dataclass, field, replace
from datetime import date
import json
from pathlib import Path
import re
from typing import Any, Dict, Optional, Sequence

from .request_rules import (
    RequestRuleRegistry,
    RequestRules,
    request_rule_registry_for_provider,
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
        data: Dict[str, Any],
        *,
        currency: str = "USD",
        request_rule_registry: Optional[RequestRuleRegistry] = None,
    ) -> "ModelSpec":
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
        is_adaptive_thinking = bool(data.get("is_adaptive_thinking", False))
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
class ProviderSpec:
    name: str
    currency: str = "USD"
    models: Dict[str, ModelSpec] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> "ProviderSpec":
        currency = data.get("currency", "USD")
        if not isinstance(currency, str) or not currency:
            raise ValueError(f"Provider '{name}' has an invalid currency")
        request_rule_registry = request_rule_registry_for_provider(name)
        models = {
            model_name: ModelSpec.from_dict(
                model_name,
                model_spec,
                currency=currency,
                request_rule_registry=request_rule_registry,
            )
            for model_name, model_spec in (data.get("models") or {}).items()
        }
        return cls(name=name, currency=currency, models=models)


@dataclass(frozen=True, init=False)
class RegistrySpec:
    schema_version: int
    effective_date: str
    providers: Dict[str, ProviderSpec]

    def __init__(self, path: str | Path = DEFAULT_REGISTRY_PATH) -> None:
        manifest_path = Path(path)
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        provider_files = data.get("provider_files")
        if not isinstance(provider_files, dict) or not provider_files:
            raise ValueError("registry manifest must define a non-empty provider_files object")

        providers = {
            provider_name: ProviderSpec.from_dict(
                provider_name,
                self._load_provider_data(
                    manifest_path,
                    provider_name,
                    provider_file,
                ),
            )
            for provider_name, provider_file in provider_files.items()
        }
        object.__setattr__(self, "schema_version", int(data["schema_version"]))
        object.__setattr__(self, "effective_date", str(data["effective_date"]))
        object.__setattr__(self, "providers", providers)

    @staticmethod
    def _load_provider_data(
        manifest_path: Path,
        provider_name: str,
        provider_file: Any,
    ) -> Dict[str, Any]:
        if not isinstance(provider_name, str) or not provider_name:
            raise ValueError("provider_files keys must be non-empty strings")
        if not isinstance(provider_file, str) or not provider_file:
            raise ValueError(
                f"Provider '{provider_name}' must reference a non-empty relative path"
            )

        provider_path = Path(provider_file)
        if provider_path.is_absolute():
            raise ValueError(
                f"Provider '{provider_name}' must use a relative registry path"
            )
        provider_path = manifest_path.parent / provider_path
        try:
            provider_data = json.loads(provider_path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise ValueError(
                f"Provider registry file is missing for '{provider_name}': {provider_path}"
            ) from error

        if not isinstance(provider_data, dict):
            raise ValueError(f"Provider registry data for '{provider_name}' must be an object")
        return provider_data


def _is_valid_snapshot_date(value: str, *, compact: bool) -> bool:
    """Return whether a provider snapshot suffix is a real calendar date."""
    if compact:
        value = f"{value[:4]}-{value[4:6]}-{value[6:]}"
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _snapshot_base_model_name(provider_name: str, model_name: str) -> Optional[str]:
    """Return a supported direct-provider snapshot's unsuffixed model ID."""
    if provider_name == "anthropic":
        match = _ANTHROPIC_SNAPSHOT_ID.fullmatch(model_name)
        if match and _is_valid_snapshot_date(match.group(2), compact=True):
            return match.group(1)
    elif provider_name == "openai":
        match = _OPENAI_SNAPSHOT_ID.fullmatch(model_name)
        if match and _is_valid_snapshot_date(match.group(2), compact=False):
            return match.group(1)
    return None


def resolve_model_spec(
    registry: RegistrySpec,
    provider_name: str,
    model_name: str,
) -> Optional[ModelSpec]:
    """Resolve an exact model or a supported provider snapshot to its base spec.

    The caller must continue to send ``model_name`` to the provider. This
    resolver only supplies verified registry metadata for direct Anthropic and
    OpenAI snapshot IDs whose unsuffixed base is registered.
    """
    provider = registry.providers.get(provider_name)
    if not provider:
        return None

    model_spec = provider.models.get(model_name)
    if model_spec:
        return model_spec

    base_model_name = _snapshot_base_model_name(provider_name, model_name)
    return provider.models.get(base_model_name) if base_model_name else None


LLM_REGISTRY = RegistrySpec()
