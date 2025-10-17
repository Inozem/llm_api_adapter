from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_REGISTRY_PATH = Path(__file__).with_name("llm_registry.json")


@dataclass(frozen=True)
class Pricing:
    in_per_token: float
    out_per_token: float
    currency: str = "USD"

    def set_in_per_1m(self, value: float) -> None:
        object.__setattr__(self, "in_per_token", value / 1_000_000)

    def set_out_per_1m(self, value: float) -> None:
        object.__setattr__(self, "out_per_token", value / 1_000_000)

    def set_currency(self, value: str) -> None:
        object.__setattr__(self, "currency", value)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    pricing: Optional[Pricing] = None

    @classmethod
    def from_dict(cls, name: str, d: Dict[str, Any]) -> "ModelSpec":
        pricing_data = d.get("pricing")
        if pricing_data:
            in_per_token = pricing_data["in_per_1m"] / 1_000_000
            out_per_token = pricing_data["out_per_1m"] / 1_000_000
            pricing = Pricing(in_per_token, out_per_token)
        else:
            pricing = None
        return cls(name=name, pricing=pricing)


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    models: Dict[str, ModelSpec] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, name: str, d: Dict[str, Any]) -> "ProviderSpec":
        models = {
            model_name: ModelSpec.from_dict(model_name, model_spec)
            for model_name, model_spec in (d.get("models") or {}).items()
        }
        return cls(name=name, models=models)


@dataclass(frozen=True, init=False)
class RegistrySpec:
    schema_version: int
    effective_date: str
    providers: Dict[str, ProviderSpec]

    def __init__(self, path: str | Path = DEFAULT_REGISTRY_PATH) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        providers = {
            provider_name: ProviderSpec.from_dict(provider_name, provider_spec)
            for provider_name, provider_spec
            in (data.get("providers") or {}).items()
        }
        object.__setattr__(self, "schema_version", int(data["schema_version"]))
        object.__setattr__(self, "effective_date", str(data["effective_date"]))
        object.__setattr__(self, "providers", providers)
            
LLM_REGISTRY = RegistrySpec()
