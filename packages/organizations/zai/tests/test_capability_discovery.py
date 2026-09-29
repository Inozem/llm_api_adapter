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

from typing import Final


CANDIDATE_MODELS: Final = ("glm-5.3-flash",)
CLOSED_MODEL_IDS: Final = (
    "glm-5.3-flashx",
    "glm-5.3",
    "glm-5.2",
)
EXPECTED_ALIASES: Final = ()
EXPECTED_LIMITS: Final = {
    "context_window_tokens": 1_000_000,
    "max_output_tokens": 131_072,
}
EXPECTED_THINKING_MODES: Final = ("low", "high", "max")
EXPECTED_CAPABILITY_EXCEPTIONS: Final = {
    "pdf_bytes": (
        "The Z.ai package rejects DocumentPart bytes, including PDF content, before "
        "HTTP; document files are outside this adapter's supported input serialization."
    ),
    "pdf_url": (
        "The Z.ai package rejects DocumentPart URLs, including PDF URLs, before "
        "HTTP; document files are outside this adapter's supported input serialization."
    ),
    "provider_continuation": (
        "previous_response is accepted for Core API compatibility but ignored; the "
        "request uses caller-provided messages and sends no provider continuation "
        "identifier."
    ),
    "reasoning_control": (
        "Reasoning cannot be disabled; Core uses the minimum declared by "
        "reasoning_capability and emits a warning."
    ),
    "structured_output_model": (
        "The adapter rejects response_model before HTTP; Z.ai documents JSON object "
        "mode but no model-bound output enforcement, so this package does not expose "
        "Pydantic response models."
    ),
    "structured_output_schema": (
        "The adapter rejects portable json_schema before HTTP; response_format "
        "supports json_object only, with schema guidance and validation handled in "
        "the application."
    ),
    "tool_choice_any": (
        "The adapter rejects this unsupported tool-choice mode before transport "
        "according to the exact-model request rules."
    ),
    "tool_choice_named": (
        "The adapter rejects this unsupported tool-choice mode before transport "
        "according to the exact-model request rules."
    ),
    "tool_choice_none": (
        "The adapter rejects this unsupported tool-choice mode before transport "
        "according to the exact-model request rules."
    ),
}
EXPECTED_PRICING_PER_1M_USD: Final = {
    "cache_hit_input": 0.03,
    "cache_miss_input": 0.15,
    "output": 0.50,
}
EXPECTED_PRICING_TIER: Final = {
    "up_to_prompt_tokens": None,
    "input_per_1m": EXPECTED_PRICING_PER_1M_USD["cache_miss_input"],
    "output_per_1m": EXPECTED_PRICING_PER_1M_USD["output"],
    "cache_read_input_per_1m": EXPECTED_PRICING_PER_1M_USD["cache_hit_input"],
}

MATRIX_CAPABILITIES: Final = (
    "endpoint",
    "reasoning",
    "tools",
    "image_bytes",
    "image_url",
    "streaming",
    "usage",
)
UNSUPPORTED_CAPABILITIES: Final = (
    "json_schema",
    "response_model",
    "document_bytes",
    "document_url",
    "non_image_file",
    "ocr",
    "file_upload",
    "provider_builtin_tools",
    "parallel_tool_calls",
    "server_continuation_id",
    "arbitrary_endpoint",
    "deployments",
    "video",
)
EXPECTED_CAPABILITIES: Final = {
    capability: "supported" for capability in MATRIX_CAPABILITIES
}


ZAI_CAPABILITY_DISCOVERY: Final = {
    "recorded_on": "2026-09-18",
    "candidate_models": CANDIDATE_MODELS,
    "admission_policy": {
        "initial_public_models": CANDIDATE_MODELS,
        "initial_status": "admitted",
        "closed_model_ids": CLOSED_MODEL_IDS,
    },
    "models": {
        "glm-5.3-flash": {
            "aliases": EXPECTED_ALIASES,
            "limits": EXPECTED_LIMITS,
            "pricing_per_1m_usd": EXPECTED_PRICING_PER_1M_USD,
            "reasoning_modes": EXPECTED_THINKING_MODES,
            "capability_exceptions": EXPECTED_CAPABILITY_EXCEPTIONS,
            "capabilities": EXPECTED_CAPABILITIES,
            "unsupported_capabilities": UNSUPPORTED_CAPABILITIES,
        },
    },
}

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
        "capability_exceptions",
        "limits",
        "pricing_tiers",
        "reasoning_capability",
        "request_rules",
    }
    assert {
        exception["capability_id"]: exception["behavior"]
        for exception in model_data["capability_exceptions"]
    } == EXPECTED_CAPABILITY_EXCEPTIONS
    assert expected_model["capability_exceptions"] == EXPECTED_CAPABILITY_EXCEPTIONS
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
