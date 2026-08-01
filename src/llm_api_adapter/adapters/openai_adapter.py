"""Compatibility import for the packaged OpenAI adapter."""

from .openai.adapter import OpenAIAdapter
from ..llms.openai.async_client import OpenAIAsyncClient
from ..llms.openai.sync_client import OpenAISyncClient

__all__ = ["OpenAIAdapter", "OpenAIAsyncClient", "OpenAISyncClient"]
