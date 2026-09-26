"""Deterministic, credential-free DeepSeek capability-discovery contracts."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[2]
CORE_SOURCE = REPOSITORY_ROOT / "src"
PACKAGE_SOURCE = PACKAGE_ROOT / "src"
for source in (
    str(PACKAGE_SOURCE),
    str(CORE_SOURCE),
    str(REPOSITORY_ROOT),
):
    if source not in sys.path:
        sys.path.insert(0, source)

from packages.organizations.deepseek.tests.fixtures.deepseek_capability_discovery import (
    CANDIDATE_MODELS,
    CLOSED_MODEL_IDS,
    DEEPSEEK_CAPABILITY_DISCOVERY,
    EXPECTED_CAPABILITIES,
    EXPECTED_CAPABILITY_EXCEPTIONS,
    EXPECTED_LIMITS,
    EXPECTED_THINKING_MODES,
    MATRIX_CAPABILITIES,
    UNSUPPORTED_CAPABILITIES,
)
from llm_api_adapter.llm_registry.llm_registry import (
    RegistrySpec,
    resolve_model_spec,
)
from llm_api_adapter_deepseek.registry import MODEL_METADATA


@pytest.fixture(scope="module")
def discovery_record() -> dict:
    return DEEPSEEK_CAPABILITY_DISCOVERY


@pytest.fixture(scope="module")
def deepseek_registry() -> RegistrySpec:
    registry = RegistrySpec()
    assert registry.register_organization_metadata(MODEL_METADATA) is True
    return registry


@pytest.mark.unit
def test_discovery_exposes_only_exact_flash_model_and_no_aliases(
    discovery_record,
    deepseek_registry,
):
    model_data = MODEL_METADATA.organization_data["models"]

    assert discovery_record["candidate_models"] == CANDIDATE_MODELS
    assert tuple(model_data) == CANDIDATE_MODELS
    assert tuple(discovery_record["models"]) == CANDIDATE_MODELS
    assert not set(model_data).intersection(CLOSED_MODEL_IDS)

    for alias in ("deepseek-flash-latest", "deepseek-v4-pro"):
        assert resolve_model_spec(deepseek_registry, "deepseek", alias) is None


@pytest.mark.unit
def test_flash_limits_and_thinking_modes_are_exact(discovery_record):
    expected_model = discovery_record["models"]["deepseek-flash"]
    model_data = MODEL_METADATA.organization_data["models"]["deepseek-flash"]

    assert set(model_data) == {
        "capability_exceptions",
        "limits",
        "pricing_tiers",
        "reasoning_capability",
    }
    assert {
        exception["capability_id"]: exception["behavior"]
        for exception in model_data["capability_exceptions"]
    } == EXPECTED_CAPABILITY_EXCEPTIONS
    assert "cache_pricing" not in model_data
    assert model_data["limits"] == EXPECTED_LIMITS
    assert expected_model["limits"] == EXPECTED_LIMITS
    assert model_data["reasoning_capability"]["allowed_values"] == list(
        EXPECTED_THINKING_MODES
    )
    assert expected_model["reasoning_modes"] == EXPECTED_THINKING_MODES


@pytest.mark.unit
def test_dynamic_peak_and_off_peak_pricing_stays_package_owned():
    from llm_api_adapter_deepseek.registry import (
        OFF_PEAK_PRICING,
        PEAK_PRICING,
        pricing_for_dispatch,
    )

    assert PEAK_PRICING.cache_miss_input_per_token == 0.3 / 1_000_000
    assert OFF_PEAK_PRICING.cache_miss_input_per_token == 0.15 / 1_000_000
    assert pricing_for_dispatch(
        datetime(2026, 10, 13, 1, 0, tzinfo=timezone.utc)
    ) == PEAK_PRICING
    assert pricing_for_dispatch(
        datetime(2026, 10, 13, 4, 0, tzinfo=timezone.utc)
    ) == OFF_PEAK_PRICING


@pytest.mark.unit
def test_flash_capability_matrix_is_closed_and_verified(discovery_record):
    expected_model = discovery_record["models"]["deepseek-flash"]

    assert tuple(expected_model["capabilities"]) == MATRIX_CAPABILITIES
    assert expected_model["capabilities"] == EXPECTED_CAPABILITIES
    assert all(
        expected_model["capabilities"][capability] == "supported"
        for capability in MATRIX_CAPABILITIES
    )


@pytest.mark.unit
def test_flash_declares_every_unsupported_capability_explicitly(discovery_record):
    expected_model = discovery_record["models"]["deepseek-flash"]

    assert tuple(expected_model["unsupported_capabilities"]) == (
        *UNSUPPORTED_CAPABILITIES,
    )
    assert not set(expected_model["unsupported_capabilities"]).intersection(
        MATRIX_CAPABILITIES
    )
