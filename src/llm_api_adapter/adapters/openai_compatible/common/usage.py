"""Shared token usage primitives for OpenAI-compatible protocols."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

from ....models.responses.chat_response import ChatResponse


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
