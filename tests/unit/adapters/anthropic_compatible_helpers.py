"""Test-only helpers for Anthropic-compatible behavior policies."""

from typing import Any


def first_text_block(blocks: list[dict[str, Any]]) -> str | None:
    """Return the first text block's text, matching native Claude policy."""
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            return block.get("text") if isinstance(block.get("text"), str) else None
    return None
