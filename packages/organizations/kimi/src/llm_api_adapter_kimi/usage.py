"""Kimi's native reported usage normalization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from llm_api_adapter.errors.llm_api_error import LLMAPIClientError
from llm_api_adapter.models.responses.chat_response import Usage


@dataclass
class KimiUsage(Usage):
    """Normalized Kimi usage retaining the cache-hit split used for pricing."""

    cached_tokens: int | None = None


def parse_kimi_usage(raw_usage: Any) -> KimiUsage | None:
    """Normalize only counts Kimi reported, retaining omitted values."""
    if not isinstance(raw_usage, Mapping):
        return None

    def read_count(value: Any, field_name: str) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise LLMAPIClientError(
                detail=(
                    "Kimi Chat Completions usage token counts: "
                    f"{field_name} must be a non-negative integer"
                ),
            )
        return value

    input_tokens = read_count(raw_usage.get("prompt_tokens"), "prompt_tokens")
    output_tokens = read_count(raw_usage.get("completion_tokens"), "completion_tokens")
    total_tokens = read_count(raw_usage.get("total_tokens"), "total_tokens")
    if input_tokens is None and output_tokens is None and total_tokens is None:
        raise LLMAPIClientError(
            detail="Kimi Chat Completions usage token counts must include a count",
        )
    if input_tokens is not None and output_tokens is not None:
        total_tokens = total_tokens if total_tokens is not None else input_tokens + output_tokens

    prompt_details = raw_usage.get("prompt_tokens_details")
    details = prompt_details if isinstance(prompt_details, Mapping) else {}
    cached_tokens = read_count(
        details.get("cached_tokens")
        if details.get("cached_tokens") is not None
        else raw_usage.get("cached_tokens"),
        "cached_tokens",
    )
    cache_write_tokens = read_count(details.get("cache_write_tokens"), "cache_write_tokens")
    if input_tokens is not None:
        cache_components = sum(
            count for count in (cached_tokens, cache_write_tokens) if count is not None
        )
        if cache_components > input_tokens:
            raise LLMAPIClientError(
                detail=(
                    "Kimi Chat Completions cached and cache-write tokens "
                    "must not exceed prompt_tokens"
                ),
            )

    return KimiUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_tokens=cached_tokens,
        cache_write_tokens=cache_write_tokens,
    )


__all__ = ["KimiUsage", "parse_kimi_usage"]
