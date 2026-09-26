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
        "previous_response is accepted for Core API compatibility but ignored; each "
        "request uses the caller-provided messages and sends no provider continuation "
        "identifier."
    ),
    "reasoning_control": (
        "GLM-5.3-Flash cannot disable thinking; reasoning_level='none' falls back to "
        "'low' with a warning, and the supported effort levels are low, high, and max."
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
        "The model request rule permits only tool_choice='auto'; a forced any-tool "
        "choice is rejected before transport."
    ),
    "tool_choice_named": (
        "The model request rule permits only tool_choice='auto'; a forced named-tool "
        "choice is rejected before transport."
    ),
    "tool_choice_none": (
        "The model request rule permits only tool_choice='auto'; an explicit no-tools "
        "choice is rejected before transport."
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
