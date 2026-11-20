"""Qwen-only validation around the shared Messages usage contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from llm_api_adapter.adapters.anthropic_compatible.usage import (
    normalize_messages_usage,
)
from llm_api_adapter.errors.llm_api_error import LLMAPIClientError
from llm_api_adapter.models.responses.chat_response import Usage


def normalize_qwen_accounted_usage(raw: Any) -> Usage | None:
    """Reject malformed reported counters, then use the common normalizer."""
    if not isinstance(raw, Mapping):
        return None
    for key in (
        "input_tokens",
        "output_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
    ):
        if key in raw:
            _require_count(raw[key], key)
    details = raw.get("prompt_tokens_details")
    if isinstance(details, Mapping) and "cached_tokens" in details:
        _require_count(details["cached_tokens"], "cached_tokens")
    return normalize_messages_usage(raw)


def _require_count(value: Any, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LLMAPIClientError(
            detail=f"Qwen {field} must be a non-negative integer"
        )


__all__ = ["normalize_qwen_accounted_usage"]
