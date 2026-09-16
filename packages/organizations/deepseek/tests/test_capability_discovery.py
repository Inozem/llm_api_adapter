"""Deterministic, credential-free DeepSeek capability-discovery contracts."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[2]
CORE_SOURCE = REPOSITORY_ROOT / "src"
PACKAGE_SOURCE = PACKAGE_ROOT / "src"
for source in (str(PACKAGE_SOURCE), str(CORE_SOURCE), str(REPOSITORY_ROOT)):
    if source not in sys.path:
        sys.path.insert(0, source)

from fixtures.deepseek_capability_discovery import (
    CANDIDATE_MODELS,
    CLOSED_MODEL_IDS,
    DEEPSEEK_CAPABILITY_DISCOVERY,
    EXPECTED_ALIASES,
    EXPECTED_CAPABILITIES,
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
    assert model_data["deepseek-flash"]["aliases"] == list(EXPECTED_ALIASES)
    assert not set(model_data).intersection(CLOSED_MODEL_IDS)

    for alias in ("deepseek-flash-latest", "deepseek-v4-pro"):
        assert resolve_model_spec(deepseek_registry, "deepseek", alias) is None


@pytest.mark.unit
def test_flash_limits_and_thinking_modes_are_exact(discovery_record):
    expected_model = discovery_record["models"]["deepseek-flash"]
    model_data = MODEL_METADATA.organization_data["models"]["deepseek-flash"]

    assert model_data["limits"] == EXPECTED_LIMITS
    assert expected_model["limits"] == EXPECTED_LIMITS
    assert model_data["reasoning_capability"]["allowed_values"] == list(
        EXPECTED_THINKING_MODES
    )
    assert expected_model["reasoning_modes"] == EXPECTED_THINKING_MODES


@pytest.mark.unit
def test_flash_capability_matrix_is_closed_and_verified(discovery_record):
    expected_model = discovery_record["models"]["deepseek-flash"]
    model_data = MODEL_METADATA.organization_data["models"]["deepseek-flash"]

    assert tuple(expected_model["capabilities"]) == MATRIX_CAPABILITIES
    assert set(model_data["capabilities"]) == set(MATRIX_CAPABILITIES)
    assert model_data["capabilities"] == EXPECTED_CAPABILITIES
    assert all(
        expected_model["capabilities"][capability] == "supported"
        for capability in MATRIX_CAPABILITIES
    )


@pytest.mark.unit
def test_flash_declares_every_unsupported_capability_explicitly(discovery_record):
    expected_model = discovery_record["models"]["deepseek-flash"]
    model_data = MODEL_METADATA.organization_data["models"]["deepseek-flash"]

    assert tuple(expected_model["unsupported_capabilities"]) == (
        *UNSUPPORTED_CAPABILITIES,
    )
    assert model_data["unsupported_capabilities"] == list(UNSUPPORTED_CAPABILITIES)
    assert not set(model_data["unsupported_capabilities"]).intersection(
        MATRIX_CAPABILITIES
    )
