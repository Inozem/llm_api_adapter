"""Compatibility import for the packaged Google adapter."""

from .google.adapter import GoogleAdapter
from ..llms.google.async_client import GeminiAsyncClient
from ..llms.google.sync_client import GeminiSyncClient

__all__ = ["GoogleAdapter", "GeminiAsyncClient", "GeminiSyncClient"]
