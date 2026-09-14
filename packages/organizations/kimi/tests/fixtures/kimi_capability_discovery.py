"""Sanitized Kimi protocol fixtures captured during capability discovery.

These fixtures intentionally describe only documented Chat Completions and Files
wire shapes. They are not an adapter implementation and contain no credentials
or live-provider assertions. Later package tests reuse them to prove the
capabilities admitted during discovery.
"""

from __future__ import annotations

from typing import Final


CANDIDATE_MODELS: Final = ("kimi-k3", "kimi-k2.7-code", "kimi-k2.6")
MATRIX_CAPABILITIES: Final = (
    "endpoint",
    "reasoning",
    "tools",
    "json_schema",
    "image_bytes",
    "image_url",
    "pdf_bytes",
    "pdf_url",
    "streaming",
    "usage",
)

# This is deliberately a fixture rather than registry metadata.  It closes the
# discovery scope without advertising Kimi or registering a model before the
# package has proved the corresponding wire contracts.
KIMI_CAPABILITY_DISCOVERY: Final = {
    "recorded_on": "2026-09-13",
    "candidate_models": CANDIDATE_MODELS,
    "admission_policy": {
        "initial_public_models": (),
        "initial_status": "blocked",
        "blocking_conditions": (
            "Public image URLs have no supported Kimi wire path; only data URIs and uploaded ms:// file IDs are documented.",
            "No native public PDF URL attachment path is documented; downloading a caller URL would violate the shared DocumentPart contract.",
            "Kimi prices cache-hit and cache-miss prompt tokens differently, so Core cannot report an exact total until that split is represented.",
        ),
    },
    "official_sources": {
        "api_overview": "https://platform.kimi.ai/docs/api/overview",
        "chat_api": "https://platform.kimi.ai/docs/api/chat",
        "model_list": "https://platform.kimi.ai/docs/models",
        "parameter_reference": "https://platform.kimi.ai/docs/api/models-overview",
        "k3_guide": "https://platform.kimi.ai/docs/guide/kimi-k3-quickstart",
        "k27_guide": "https://platform.kimi.ai/docs/guide/kimi-k2-7-code-quickstart",
        "k26_guide": "https://platform.kimi.ai/docs/guide/kimi-k2-6-quickstart",
        "vision": "https://platform.kimi.ai/docs/guide/use-kimi-vision-model",
        "structured_output": "https://platform.kimi.ai/docs/guide/response_format",
        "context_caching": "https://platform.kimi.ai/docs/guide/use-context-caching-feature-of-kimi-api",
        "files": "https://platform.kimi.ai/docs/api/files-upload",
        "pricing": "https://platform.kimi.ai/",
    },
    "models": {
        "kimi-k3": {
            "context_window_tokens": 1_048_576,
            "max_output_tokens": 1_048_576,
            "pricing_per_1m_usd": {
                "cache_hit_input": 0.30,
                "cache_miss_input": 3.00,
                "output": 15.00,
            },
            "capabilities": {
                "endpoint": ("pending_conformance", ("api_overview", "chat_api"), "KIMI_CHAT_COMPLETION_RESPONSE"),
                "reasoning": ("pending_conformance", ("parameter_reference", "k3_guide"), "KIMI_CHAT_COMPLETION_RESPONSE"),
                "tools": ("pending_conformance", ("chat_api", "parameter_reference", "k3_guide"), "KIMI_TOOL_CALL_RESPONSE"),
                "json_schema": ("pending_conformance", ("chat_api", "structured_output", "k3_guide"), "KIMI_STRUCTURED_OUTPUT_RESPONSE"),
                "image_bytes": ("pending_conformance", ("vision", "k3_guide"), "KIMI_IMAGE_DATA_URI_MESSAGE"),
                "image_url": ("excluded", ("vision", "k3_guide"), "KIMI_PUBLIC_IMAGE_URL"),
                "pdf_bytes": ("pending_manual_e2e", ("files",), "KIMI_FILE_EXTRACTION_RESPONSE"),
                "pdf_url": ("excluded", ("files",), "KIMI_PUBLIC_IMAGE_URL"),
                "streaming": ("pending_conformance", ("chat_api", "k3_guide"), "KIMI_STREAM_EVENTS"),
                "usage": ("pending_conformance", ("chat_api",), "KIMI_STREAM_EVENTS"),
            },
        },
        "kimi-k2.7-code": {
            "context_window_tokens": 262_144,
            "max_output_tokens": None,
            "documented_default_output_tokens": 32_768,
            "pricing_per_1m_usd": {
                "cache_hit_input": 0.19,
                "cache_miss_input": 0.95,
                "output": 4.00,
            },
            "capabilities": {
                "endpoint": ("pending_conformance", ("api_overview", "chat_api"), "KIMI_CHAT_COMPLETION_RESPONSE"),
                "reasoning": ("pending_conformance", ("parameter_reference", "k27_guide"), "KIMI_CHAT_COMPLETION_RESPONSE"),
                "tools": ("pending_conformance", ("chat_api", "parameter_reference", "k27_guide"), "KIMI_TOOL_CALL_RESPONSE"),
                "json_schema": ("pending_conformance", ("chat_api", "structured_output"), "KIMI_STRUCTURED_OUTPUT_RESPONSE"),
                "image_bytes": ("pending_conformance", ("vision", "k27_guide"), "KIMI_IMAGE_DATA_URI_MESSAGE"),
                "image_url": ("excluded", ("vision",), "KIMI_PUBLIC_IMAGE_URL"),
                "pdf_bytes": ("pending_manual_e2e", ("files",), "KIMI_FILE_EXTRACTION_RESPONSE"),
                "pdf_url": ("excluded", ("files",), "KIMI_PUBLIC_IMAGE_URL"),
                "streaming": ("pending_conformance", ("chat_api",), "KIMI_STREAM_EVENTS"),
                "usage": ("pending_conformance", ("chat_api",), "KIMI_STREAM_EVENTS"),
            },
        },
        "kimi-k2.6": {
            "context_window_tokens": 262_144,
            "max_output_tokens": None,
            "documented_default_output_tokens": 32_768,
            "pricing_per_1m_usd": {
                "cache_hit_input": 0.16,
                "cache_miss_input": 0.95,
                "output": 4.00,
            },
            "capabilities": {
                "endpoint": ("pending_conformance", ("api_overview", "chat_api"), "KIMI_CHAT_COMPLETION_RESPONSE"),
                "reasoning": ("pending_conformance", ("parameter_reference", "k26_guide"), "KIMI_CHAT_COMPLETION_RESPONSE"),
                "tools": ("pending_conformance", ("chat_api", "parameter_reference", "k26_guide"), "KIMI_TOOL_CALL_RESPONSE"),
                "json_schema": ("pending_manual_e2e", ("chat_api", "structured_output"), "KIMI_STRUCTURED_OUTPUT_RESPONSE"),
                "image_bytes": ("pending_conformance", ("vision", "k26_guide"), "KIMI_IMAGE_DATA_URI_MESSAGE"),
                "image_url": ("excluded", ("vision",), "KIMI_PUBLIC_IMAGE_URL"),
                "pdf_bytes": ("pending_manual_e2e", ("files",), "KIMI_FILE_EXTRACTION_RESPONSE"),
                "pdf_url": ("excluded", ("files",), "KIMI_PUBLIC_IMAGE_URL"),
                "streaming": ("pending_conformance", ("chat_api",), "KIMI_STREAM_EVENTS"),
                "usage": ("pending_conformance", ("chat_api",), "KIMI_STREAM_EVENTS"),
            },
        },
    },
}

KIMI_CHAT_COMPLETION_RESPONSE: Final = {
    "id": "cmpl-kimi-discovery",
    "object": "chat.completion",
    "created": 1_789_721_600,
    "model": "kimi-k3",
    "choices": [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": "Kimi discovery fixture.",
            "reasoning_content": "Provider-readable reasoning fixture.",
        },
        "finish_reason": "stop",
    }],
    "usage": {
        "prompt_tokens": 19,
        "completion_tokens": 13,
        "total_tokens": 32,
        "cached_tokens": 12,
    },
}

KIMI_TOOL_CALL_RESPONSE: Final = {
    "id": "cmpl-kimi-tool",
    "object": "chat.completion",
    "model": "kimi-k3",
    "choices": [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_weather",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city":"Beijing"}',
                },
            }],
        },
        "finish_reason": "tool_calls",
    }],
}
KIMI_TOOL_RESULT_MESSAGE: Final = {
    "role": "tool",
    "tool_call_id": "call_weather",
    "content": "Sunny, 25C",
}

KIMI_STRUCTURED_OUTPUT_RESPONSE: Final = {
    "model": "kimi-k3",
    "choices": [{
        "message": {
            "role": "assistant",
            "content": '{"answer":"ok"}',
            "reasoning_content": "This field must not be JSON-parsed.",
        },
        "finish_reason": "stop",
    }],
}

KIMI_STREAM_EVENTS: Final = (
    {
        "data": {
            "id": "cmpl-kimi-stream",
            "object": "chat.completion.chunk",
            "model": "kimi-k3",
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": None,
            }],
        },
    },
    {
        "data": {
            "id": "cmpl-kimi-stream",
            "object": "chat.completion.chunk",
            "model": "kimi-k3",
            "choices": [{
                "index": 0,
                "delta": {"reasoning_content": "Reasoning. ", "content": "Hello"},
                "finish_reason": None,
            }],
        },
    },
    {
        "data": {
            "id": "cmpl-kimi-stream",
            "object": "chat.completion.chunk",
            "model": "kimi-k3",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 19,
                "completion_tokens": 13,
                "total_tokens": 32,
                "cached_tokens": 12,
            },
        },
    },
    {"data": "[DONE]"},
)

KIMI_IMAGE_DATA_URI_MESSAGE: Final = {
    "role": "user",
    "content": [
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,aW1hZ2U="},
        },
        {"type": "text", "text": "Describe this image."},
    ],
}
KIMI_PUBLIC_IMAGE_URL: Final = "https://example.test/image.png"

KIMI_FILE_EXTRACTION_RESPONSE: Final = {
    "id": "file_kimi_discovery",
    "object": "file",
    "bytes": 1024,
    "created_at": 1_789_721_600,
    "filename": "fixture.pdf",
    "purpose": "file-extract",
    "status": "ready",
    "status_details": "",
}
