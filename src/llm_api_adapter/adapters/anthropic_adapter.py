"""Compatibility import for the packaged Anthropic adapter."""

from .anthropic.adapter import AnthropicAdapter
from ..llms.anthropic.async_client import ClaudeAsyncClient
from ..llms.anthropic.sync_client import ClaudeSyncClient

__all__ = ["AnthropicAdapter", "ClaudeAsyncClient", "ClaudeSyncClient"]
