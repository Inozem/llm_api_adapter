"""Reviewed, credential-free expectations for the initial Z.ai model."""

from __future__ import annotations

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
EXPECTED_PRICING_PER_1M_USD: Final = {
    "cache_hit_input": 0.03,
    "cache_miss_input": 0.15,
    "output": 0.50,
}
EXPECTED_PRICING_TIER: Final = {
    "up_to_prompt_tokens": None,
    "input_per_1m": EXPECTED_PRICING_PER_1M_USD["cache_miss_input"],
    "output_per_1m": EXPECTED_PRICING_PER_1M_USD["output"],
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
            "capabilities": EXPECTED_CAPABILITIES,
            "unsupported_capabilities": UNSUPPORTED_CAPABILITIES,
        },
    },
}
