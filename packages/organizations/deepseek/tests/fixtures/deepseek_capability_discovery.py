"""Reviewed DeepSeek Flash capability-discovery expectations.

The record is deliberately credential-free and contains only the public model
boundary that the package is expected to publish. Provider aliases and
unsupported request forms stay explicit so later adapter work cannot silently
broaden the contract.
"""

from __future__ import annotations

from typing import Final


CANDIDATE_MODELS: Final = ("deepseek-flash",)
CLOSED_MODEL_IDS: Final = (
    "deepseek-chat",
    "deepseek-reasoner",
    "deepseek-v4-pro",
)
EXPECTED_ALIASES: Final = ()
EXPECTED_LIMITS: Final = {
    "context_window_tokens": 1_000_000,
    "max_output_tokens": 384_000,
}
EXPECTED_THINKING_MODES: Final = ("none", "low", "high", "max")
EXPECTED_CAPABILITY_EXCEPTIONS: Final = {
    "provider_continuation": (
        "The Responses API is stateless and does not support previous_response_id; "
        "callers send full messages history, while previous_response carries only "
        "matching opaque reasoning replay material."
    ),
    "tool_choice_named": (
        "For a forced named function choice, the adapter sets reasoning_level='none' "
        "and warns when it must disable thinking, following DeepSeek's documented "
        "thinking-mode restriction for forced tool choice."
    ),
    "pdf_url": (
        "DocumentPart URLs, including PDFs, are rejected before transport; the "
        "Responses API does not support file inputs and this package accepts image "
        "inputs only."
    ),
    "pdf_bytes": (
        "DocumentPart bytes, including PDFs, are rejected before transport; the "
        "Responses API does not support file inputs and this package accepts image "
        "inputs only."
    ),
}

MATRIX_CAPABILITIES: Final = (
    "endpoint",
    "reasoning",
    "tools",
    "json_schema",
    "image_bytes",
    "image_url",
    "streaming",
    "usage",
)
UNSUPPORTED_CAPABILITIES: Final = (
    "document_bytes",
    "document_url",
    "non_image_file",
    "ocr",
    "file_upload",
    "provider_builtin_tools",
    "parallel_tool_calls",
    "server_continuation_id",
)

EXPECTED_CAPABILITIES: Final = {
    capability: "supported" for capability in MATRIX_CAPABILITIES
}

DEEPSEEK_CAPABILITY_DISCOVERY: Final = {
    "recorded_on": "2026-09-16",
    "candidate_models": CANDIDATE_MODELS,
    "admission_policy": {
        "initial_public_models": CANDIDATE_MODELS,
        "initial_status": "admitted",
        "closed_model_ids": CLOSED_MODEL_IDS,
    },
    "models": {
        "deepseek-flash": {
            "aliases": EXPECTED_ALIASES,
            "limits": EXPECTED_LIMITS,
            "reasoning_modes": EXPECTED_THINKING_MODES,
            "capability_exceptions": EXPECTED_CAPABILITY_EXCEPTIONS,
            "capabilities": EXPECTED_CAPABILITIES,
            "unsupported_capabilities": UNSUPPORTED_CAPABILITIES,
        },
    },
}
