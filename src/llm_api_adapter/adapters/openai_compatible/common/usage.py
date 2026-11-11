"""Shared token usage primitives for OpenAI-compatible protocols."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable, Optional

from ....models.responses.chat_response import ChatResponse, Usage


def _token_count(value: Any) -> Optional[int]:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _cache_token_counts(
    raw_usage: Any,
    *,
    details_field: str,
) -> tuple[Optional[int], Optional[int]]:
    if not isinstance(raw_usage, Mapping):
        return None, None
    details = raw_usage.get(details_field)
    if not isinstance(details, Mapping):
        return None, None
    return (
        _token_count(details.get("cached_tokens")),
        _token_count(details.get("cache_write_tokens")),
    )


def _apply_cache_usage(
    chat_response: ChatResponse,
    response: Mapping[str, Any],
    *,
    details_field: str,
) -> ChatResponse:
    if chat_response.usage is not None:
        (
            chat_response.usage.cached_tokens,
            chat_response.usage.cache_write_tokens,
        ) = _cache_token_counts(
            response.get("usage"),
            details_field=details_field,
        )
    return chat_response


def _normalize_openai_stream_usage(
    raw_usage: Any,
    *,
    input_field: str,
    output_field: str,
    token_counter: Callable[[Any], Optional[int]] = _token_count,
) -> Optional[Usage]:
    """Normalize reported OpenAI-style stream usage without estimating values."""
    if not isinstance(raw_usage, Mapping):
        return None

    input_tokens = token_counter(raw_usage.get(input_field))
    output_tokens = token_counter(raw_usage.get(output_field))
    total_tokens = token_counter(raw_usage.get("total_tokens"))
    details_field = (
        "input_tokens_details"
        if input_field == "input_tokens"
        else "prompt_tokens_details"
    )
    cached_tokens, cache_write_tokens = _cache_token_counts(
        raw_usage,
        details_field=details_field,
    )
    if all(
        value is None
        for value in (
            input_tokens,
            output_tokens,
            total_tokens,
            cached_tokens,
            cache_write_tokens,
        )
    ):
        return None

    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_tokens=cached_tokens,
        cache_write_tokens=cache_write_tokens,
    )
