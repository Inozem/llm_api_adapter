"""Reported-only token usage normalization for generic Messages connections."""

from __future__ import annotations

from typing import Any, Mapping

from ...models.responses.chat_response import Usage


def _reported_count(value: Any) -> int | None:
    """Return a valid reported counter, preserving missing and invalid values."""
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _cache_read(usage: Mapping[str, Any]) -> int | None:
    """Read the cache counter, accepting its documented mirror alias."""
    primary_present = "cache_read_input_tokens" in usage
    primary = _reported_count(usage.get("cache_read_input_tokens"))

    details = usage.get("prompt_tokens_details")
    alias_present = isinstance(details, Mapping) and "cached_tokens" in details
    alias = _reported_count(details.get("cached_tokens")) if alias_present else None

    if primary_present and primary is None:
        return None
    if alias_present and alias is None:
        return None
    if primary_present and alias_present and primary != alias:
        return None
    if primary_present:
        return primary
    if alias_present:
        return alias
    return None


def normalize_messages_usage(raw_usage: Any) -> Usage | None:
    """Normalize independently reported Messages counters without estimation.

    Generic inclusive input is known only when ordinary input, cache reads and
    cache writes are all independently reported as valid nonnegative integers.
    Output remains independently available when one of those input components
    is unknown.
    """
    if not isinstance(raw_usage, Mapping):
        return None

    ordinary_input = _reported_count(raw_usage.get("input_tokens"))
    output = _reported_count(raw_usage.get("output_tokens"))
    cache_read = _cache_read(raw_usage)
    cache_write = _reported_count(raw_usage.get("cache_creation_input_tokens"))

    inclusive_input = (
        ordinary_input + cache_read + cache_write
        if ordinary_input is not None and cache_read is not None and cache_write is not None
        else None
    )
    if all(value is None for value in (inclusive_input, output, cache_read, cache_write)):
        return None

    return Usage(
        input_tokens=inclusive_input,
        output_tokens=output,
        total_tokens=(
            inclusive_input + output
            if inclusive_input is not None and output is not None
            else None
        ),
        cached_tokens=cache_read,
        cache_write_tokens=cache_write,
    )


def merge_messages_usage_snapshots(
    snapshots: list[tuple[str, Any]] | tuple[tuple[str, Any], ...],
) -> dict[str, Any] | None:
    """Project split stream observations into one reported raw usage mapping.

    A later observation of either cache-read representation replaces an earlier
    observation of the other representation. If both representations occur in
    the same snapshot, their values must agree. A non-mapping snapshot is an
    explicit unknown update and clears earlier usage.
    """
    latest: dict[str, Any] = {}
    has_usage = False
    for _source, snapshot in snapshots:
        if not isinstance(snapshot, Mapping):
            latest.clear()
            has_usage = False
            continue
        has_usage = True
        for field in ("input_tokens", "output_tokens", "cache_creation_input_tokens"):
            if field in snapshot:
                latest[field] = snapshot[field]

        details = snapshot.get("prompt_tokens_details")
        alias_present = isinstance(details, Mapping) and "cached_tokens" in details
        read_present = "cache_read_input_tokens" in snapshot or alias_present
        if read_present:
            latest.pop("cache_read_input_tokens", None)
            latest["cache_read_input_tokens"] = _cache_read(snapshot)

    if not has_usage or not latest:
        return None
    return latest


def normalize_messages_response_usage(response: Mapping[str, Any]) -> Usage | None:
    """Extract and normalize usage from a completed Messages response."""
    return normalize_messages_usage(response.get("usage"))


__all__ = [
    "normalize_messages_response_usage",
    "normalize_messages_usage",
    "merge_messages_usage_snapshots",
]
