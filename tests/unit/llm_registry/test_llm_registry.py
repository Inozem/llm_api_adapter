import json
from pathlib import Path

import pytest

from src.llm_api_adapter.llm_registry.llm_registry import (
    DEFAULT_REGISTRY_PATH, ModelSpec, Pricing, ProviderSpec, RegistrySpec
)

def test_pricing_setters():
    p = Pricing(in_per_token=0.0, out_per_token=0.0)
    p.set_in_per_1m(1500.0)
    p.set_out_per_1m(2500.0)
    p.set_currency("EUR")
    assert pytest.approx(p.in_per_token, rel=1e-9) == 1500.0 / 1_000_000
    assert pytest.approx(p.out_per_token, rel=1e-9) == 2500.0 / 1_000_000
    assert p.currency == "EUR"

def test_model_and_provider_from_dict():
    model_data = {"pricing": {"in_per_1m": 1000, "out_per_1m": 2000}}
    model = ModelSpec.from_dict("gpt-test", model_data)
    assert model.name == "gpt-test"
    assert model.pricing is not None
    assert pytest.approx(model.pricing.in_per_token, rel=1e-9) == 1000 / 1_000_000
    assert pytest.approx(model.pricing.out_per_token, rel=1e-9) == 2000 / 1_000_000
    provider = ProviderSpec.from_dict("prov", {"models": {"gpt-test": model_data}})
    assert provider.name == "prov"
    assert "gpt-test" in provider.models
    assert isinstance(provider.models["gpt-test"], ModelSpec)

def test_registry_reads_json_and_module_loads_llm_registry(tmp_path):
    content = {
        "schema_version": 42,
        "effective_date": "2030-01-01",
        "providers": {
            "example_provider": {
                "models": {
                    "example-model": {
                        "pricing": {
                            "in_per_1m": 1234,
                            "out_per_1m": 5678
                        }
                    }
                }
            }
        }
    }
    json_path = tmp_path / "llm_registry.json"
    json_path.write_text(json.dumps(content), encoding="utf-8")
    registry = RegistrySpec(path=str(json_path))
    assert registry.schema_version == 42
    assert registry.effective_date == "2030-01-01"
    assert "example_provider" in registry.providers
    prov = registry.providers["example_provider"]
    assert "example-model" in prov.models
    model = prov.models["example-model"]
    assert pytest.approx(model.pricing.in_per_token, rel=1e-9) == 1234 / 1_000_000
    assert pytest.approx(model.pricing.out_per_token, rel=1e-9) == 5678 / 1_000_000

def test_default_registry_json_exists():
    assert isinstance(DEFAULT_REGISTRY_PATH, Path)
    assert DEFAULT_REGISTRY_PATH.exists(), f"Expected JSON at {DEFAULT_REGISTRY_PATH}"
    assert DEFAULT_REGISTRY_PATH.is_file(), f"Expected a file at {DEFAULT_REGISTRY_PATH}"
