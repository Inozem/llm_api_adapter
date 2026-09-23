"""Deterministic, credential-free Z.ai capability-discovery contracts."""

from __future__ import annotations

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

from packages.organizations.zai.tests.fixtures.zai_capability_discovery import (
    CANDIDATE_MODELS,
    CLOSED_MODEL_IDS,
    EXPECTED_LIMITS,
    EXPECTED_CAPABILITIES,
    EXPECTED_PRICING_PER_1M_USD,
    EXPECTED_PRICING_TIER,
    EXPECTED_THINKING_MODES,
    MATRIX_CAPABILITIES,
    UNSUPPORTED_CAPABILITIES,
    ZAI_CAPABILITY_DISCOVERY,
)
from llm_api_adapter.llm_registry.llm_registry import (
    RegistrySpec,
    resolve_model_spec,
)
from llm_api_adapter_zai.registry import MODEL_METADATA


@pytest.fixture(scope="module")
def discovery_record() -> dict:
    return ZAI_CAPABILITY_DISCOVERY


@pytest.fixture(scope="module")
def zai_registry() -> RegistrySpec:
    registry = RegistrySpec()
    assert registry.register_organization_metadata(MODEL_METADATA) is True
    return registry


@pytest.mark.unit
def test_discovery_exposes_only_exact_flash_model_and_no_aliases(
    discovery_record,
    zai_registry,
):
    model_data = MODEL_METADATA.organization_data["models"]

    assert discovery_record["candidate_models"] == CANDIDATE_MODELS
    assert tuple(model_data) == CANDIDATE_MODELS
    assert tuple(discovery_record["models"]) == CANDIDATE_MODELS
    assert not set(model_data).intersection(CLOSED_MODEL_IDS)

    for alias in ("glm-5.3-flash-latest", "glm-5.3-flashx"):
        assert resolve_model_spec(zai_registry, "zai", alias) is None


@pytest.mark.unit
def test_flash_limits_pricing_and_reasoning_modes_are_exact(discovery_record):
    expected_model = discovery_record["models"]["glm-5.3-flash"]
    model_data = MODEL_METADATA.organization_data["models"]["glm-5.3-flash"]

    assert MODEL_METADATA.organization_data["currency"] == "USD"
    assert set(model_data) == {
        "limits",
        "pricing_tiers",
        "reasoning_capability",
        "request_rules",
    }
    assert model_data["limits"] == EXPECTED_LIMITS
    assert expected_model["limits"] == EXPECTED_LIMITS
    assert model_data["pricing_tiers"] == [EXPECTED_PRICING_TIER]
    assert expected_model["pricing_per_1m_usd"] == EXPECTED_PRICING_PER_1M_USD
    assert model_data["reasoning_capability"]["allowed_values"] == list(
        EXPECTED_THINKING_MODES
    )
    assert model_data["request_rules"] == [
        {
            "handler": "restrict_tool_choice",
            "arguments": {"allowed_values": ["auto"]},
        }
    ]
    assert expected_model["reasoning_modes"] == EXPECTED_THINKING_MODES


@pytest.mark.unit
def test_flash_capability_matrix_is_closed_and_verified(discovery_record):
    expected_model = discovery_record["models"]["glm-5.3-flash"]

    assert tuple(expected_model["capabilities"]) == MATRIX_CAPABILITIES
    assert expected_model["capabilities"] == EXPECTED_CAPABILITIES
    assert all(
        expected_model["capabilities"][capability] == "supported"
        for capability in MATRIX_CAPABILITIES
    )


@pytest.mark.unit
def test_flash_declares_every_unsupported_capability_explicitly(discovery_record):
    expected_model = discovery_record["models"]["glm-5.3-flash"]

    assert tuple(expected_model["unsupported_capabilities"]) == (
        *UNSUPPORTED_CAPABILITIES,
    )
    assert not set(expected_model["unsupported_capabilities"]).intersection(
        MATRIX_CAPABILITIES
    )
