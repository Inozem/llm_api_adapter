import json
from datetime import date
from pathlib import Path

import pytest

from src.llm_api_adapter.llm_registry.llm_registry import (
    DEFAULT_REGISTRY_PATH,
    ModelLimits,
    ModelSpec,
    Pricing,
    PricingTier,
    ProviderSpec,
    RegistrySpec,
)


SONNET_5_STANDARD_PRICING_DATE = date(2026, 9, 1)


def _model_data(*, tiers=None):
    return {
        "limits": {
            "context_window_tokens": 128_000,
            "max_output_tokens": 16_384,
        },
        "pricing_tiers": tiers
        if tiers is not None
        else [
            {
                "up_to_prompt_tokens": None,
                "input_per_1m": 1_000,
                "output_per_1m": 2_000,
            }
        ],
    }


@pytest.mark.unit
def test_pricing_overrides_apply_to_every_tier():
    pricing = Pricing.from_dict(
        [
            {
                "up_to_prompt_tokens": 200_000,
                "input_per_1m": 1_500,
                "output_per_1m": 2_500,
            },
            {
                "up_to_prompt_tokens": None,
                "input_per_1m": 3_000,
                "output_per_1m": 4_000,
            },
        ],
        currency="USD",
    )

    pricing.set_in_per_1m(150.0)
    pricing.set_out_per_1m(250.0)
    pricing.set_currency("EUR")

    assert [tier.in_per_token for tier in pricing.tiers] == [150 / 1_000_000] * 2
    assert [tier.out_per_token for tier in pricing.tiers] == [250 / 1_000_000] * 2
    assert pricing.currency == "EUR"


@pytest.mark.unit
def test_pricing_override_allows_zero_rate_to_disable_one_cost_component():
    pricing = Pricing.from_dict(
        [{"up_to_prompt_tokens": None, "input_per_1m": 1, "output_per_1m": 1}],
        currency="USD",
    )

    pricing.set_in_per_1m(0)

    assert pricing.tiers[0].in_per_token == 0


@pytest.mark.unit
def test_model_and_provider_from_dict():
    model_data = _model_data()

    model = ModelSpec.from_dict("gpt-test", model_data)

    assert model.name == "gpt-test"
    assert model.limits == ModelLimits(
        context_window_tokens=128_000,
        max_output_tokens=16_384,
    )
    assert model.pricing_tiers.tiers == (
        PricingTier(
            up_to_prompt_tokens=None,
            in_per_token=1_000 / 1_000_000,
            out_per_token=2_000 / 1_000_000,
        ),
    )

    provider = ProviderSpec.from_dict(
        "prov",
        {"currency": "EUR", "models": {"gpt-test": model_data}},
    )

    assert provider.name == "prov"
    assert provider.currency == "EUR"
    assert "gpt-test" in provider.models
    assert isinstance(provider.models["gpt-test"], ModelSpec)
    assert provider.models["gpt-test"].pricing_tiers.currency == "EUR"


@pytest.mark.unit
def test_registry_reads_json_and_module_loads_llm_registry(tmp_path):
    providers_dir = tmp_path / "providers"
    providers_dir.mkdir()
    provider_path = providers_dir / "example_provider.json"
    provider_path.write_text(
        json.dumps({"models": {"example-model": _model_data()}}),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 42,
        "effective_date": "2030-01-01",
        "provider_files": {
            "example_provider": "providers/example_provider.json",
        },
    }
    json_path = tmp_path / "llm_registry.json"
    json_path.write_text(json.dumps(manifest), encoding="utf-8")

    registry = RegistrySpec(path=str(json_path))

    assert registry.schema_version == 42
    assert registry.effective_date == "2030-01-01"
    model = registry.providers["example_provider"].models["example-model"]
    assert model.limits.context_window_tokens == 128_000
    assert model.limits.max_output_tokens == 16_384
    assert model.pricing_tiers.tiers[0].in_per_token == pytest.approx(0.001)
    assert model.pricing_tiers.tiers[0].out_per_token == pytest.approx(0.002)


@pytest.mark.unit
def test_registry_rejects_missing_provider_file(tmp_path):
    json_path = tmp_path / "llm_registry.json"
    json_path.write_text(
        json.dumps(
            {
                "schema_version": 42,
                "effective_date": "2030-01-01",
                "provider_files": {"missing": "providers/missing.json"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Provider registry file is missing"):
        RegistrySpec(path=str(json_path))


@pytest.mark.unit
def test_legacy_flat_pricing_is_rejected():
    legacy_model = {
        "pricing": {"in_per_1m": 1_000, "out_per_1m": 2_000},
    }

    with pytest.raises(ValueError, match="legacy pricing"):
        ModelSpec.from_dict("legacy-model", legacy_model)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("tiers", "message"),
    [
        ([], "non-empty array"),
        (
            [
                {
                    "up_to_prompt_tokens": None,
                    "input_per_1m": 1,
                    "output_per_1m": 1,
                },
                {
                    "up_to_prompt_tokens": 200,
                    "input_per_1m": 1,
                    "output_per_1m": 1,
                },
            ],
            "null only for the final",
        ),
        (
            [
                {
                    "up_to_prompt_tokens": 200,
                    "input_per_1m": 1,
                    "output_per_1m": 1,
                },
                {
                    "up_to_prompt_tokens": 200,
                    "input_per_1m": 1,
                    "output_per_1m": 1,
                },
                {
                    "up_to_prompt_tokens": None,
                    "input_per_1m": 1,
                    "output_per_1m": 1,
                },
            ],
            "strictly increasing",
        ),
        (
            [
                {
                    "up_to_prompt_tokens": None,
                    "input_per_1m": 0,
                    "output_per_1m": 1,
                }
            ],
            "input_per_1m must be a positive number",
        ),
    ],
)
def test_invalid_pricing_tiers_are_rejected(tiers, message):
    with pytest.raises(ValueError, match=message):
        ModelSpec.from_dict("invalid-model", _model_data(tiers=tiers))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("limits", "message"),
    [
        ({"context_window_tokens": 0, "max_output_tokens": 1}, "context_window"),
        ({"context_window_tokens": 1, "max_output_tokens": 0}, "max_output"),
        ({"context_window_tokens": True, "max_output_tokens": 1}, "context_window"),
    ],
)
def test_invalid_limits_are_rejected(limits, message):
    model_data = _model_data()
    model_data["limits"] = limits

    with pytest.raises(ValueError, match=message):
        ModelSpec.from_dict("invalid-model", model_data)


@pytest.mark.unit
def test_default_registry_json_exists_and_uses_tiered_schema():
    assert isinstance(DEFAULT_REGISTRY_PATH, Path)
    assert DEFAULT_REGISTRY_PATH.is_file()

    manifest = json.loads(DEFAULT_REGISTRY_PATH.read_text(encoding="utf-8"))
    assert set(manifest["provider_files"]) == {"openai", "anthropic", "google"}
    for relative_path in manifest["provider_files"].values():
        assert (DEFAULT_REGISTRY_PATH.parent / relative_path).is_file()

    registry = RegistrySpec(path=str(DEFAULT_REGISTRY_PATH))
    assert registry.schema_version == 9
    for provider in registry.providers.values():
        for model in provider.models.values():
            assert model.limits.context_window_tokens > 0
            assert model.limits.max_output_tokens > 0
            assert model.pricing_tiers.tiers[-1].up_to_prompt_tokens is None


@pytest.mark.unit
def test_sonnet_5_temporary_pricing_is_updated_after_promotion():
    registry = RegistrySpec(path=str(DEFAULT_REGISTRY_PATH))
    sonnet_5 = registry.providers["anthropic"].models["claude-sonnet-5"]
    if date.today() < SONNET_5_STANDARD_PRICING_DATE:
        expected_input, expected_output = 2.0, 10.0
    else:
        expected_input, expected_output = 3.0, 15.0
    tier = sonnet_5.pricing_tiers.tiers[0]
    assert pytest.approx(tier.in_per_token * 1_000_000) == expected_input
    assert pytest.approx(tier.out_per_token * 1_000_000) == expected_output
