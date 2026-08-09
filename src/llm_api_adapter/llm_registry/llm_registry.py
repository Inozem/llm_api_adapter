from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


DEFAULT_REGISTRY_PATH = Path(__file__).with_name("llm_registry.json")


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


@dataclass(frozen=True)
class ModelSpec:
    name: str
    limits: ModelLimits
    pricing_tiers: Pricing
    is_reasoning: bool = False
    is_adaptive_thinking: bool = False

    @classmethod
    def from_dict(
        cls,
        name: str,
        data: Dict[str, Any],
        *,
        currency: str = "USD",
    ) -> "ModelSpec":
        if "pricing" in data:
            raise ValueError(
                f"Model '{name}' uses legacy pricing; use pricing_tiers instead"
            )
        return cls(
            name=name,
            limits=ModelLimits.from_dict(data.get("limits")),
            pricing_tiers=Pricing.from_dict(
                data.get("pricing_tiers"),
                currency=currency,
            ),
            is_reasoning=bool(data.get("is_reasoning", False)),
            is_adaptive_thinking=bool(data.get("is_adaptive_thinking", False)),
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
        models = {
            model_name: ModelSpec.from_dict(
                model_name,
                model_spec,
                currency=currency,
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


LLM_REGISTRY = RegistrySpec()
