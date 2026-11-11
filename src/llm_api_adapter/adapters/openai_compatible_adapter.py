"""Backward-compatible imports for the OpenAI-compatible adapter."""

from .openai_compatible.adapter import (
    OpenAICompatibleAdapter,
    _attach_compatible_cache_usage_validity,
    _cache_token_counts,
    _compatible_cache_usage_is_valid,
    _normalize_compatible_usage,
    _redact_compatible_diagnostic,
    _token_count,
    _validate_api_root,
    _validate_compatible_pricing,
)
