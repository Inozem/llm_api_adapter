"""Reusable E2E harness shared by Core and organization-package checks."""

import asyncio
from collections.abc import Mapping
import time
from typing import Any

import pytest

from llm_api_adapter.errors import (
    LLMAPIRateLimitError,
    LLMAPIServerError,
    LLMAPITimeoutError,
)
from llm_api_adapter.llm_registry.llm_registry import LLM_REGISTRY
from llm_api_adapter.models.messages.chat_message import UserMessage
from llm_api_adapter.models.messages.file_parts import DocumentPart
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter


_RETRY_DELAYS = (2, 4, 8)
_MAX_ATTEMPTS = len(_RETRY_DELAYS) + 1
_TRANSIENT_ERRORS = (
    LLMAPIServerError,
    LLMAPIRateLimitError,
    LLMAPITimeoutError,
)


class ProfiledE2EAdapter:
    """Forward profile-specific operation kwargs to every facade operation."""

    def __init__(
        self,
        adapter: Any,
        operation_kwargs: Mapping[str, object],
    ) -> None:
        self._adapter = adapter
        self._operation_kwargs = dict(operation_kwargs)

    def _with_profile_kwargs(self, kwargs: Mapping[str, object]) -> dict[str, object]:
        return {**kwargs, **self._operation_kwargs}

    def chat(self, **kwargs: object) -> Any:
        return self._adapter.chat(**self._with_profile_kwargs(kwargs))

    def stream_chat(self, **kwargs: object) -> Any:
        return self._adapter.stream_chat(**self._with_profile_kwargs(kwargs))

    async def achat(self, **kwargs: object) -> Any:
        return await self._adapter.achat(**self._with_profile_kwargs(kwargs))

    def astream_chat(self, **kwargs: object) -> Any:
        return self._adapter.astream_chat(**self._with_profile_kwargs(kwargs))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._adapter, name)


def create_e2e_adapter(organization, model: str, *, transport: str = "requests"):
    """Create a Universal facade with the profile's operation kwargs."""
    adapter = UniversalLLMAPIAdapter(
        organization=organization["name"],
        model=model,
        api_key=organization["api_key"],
        transport=transport,
    )
    return ProfiledE2EAdapter(
        adapter,
        operation_kwargs=organization["operation_kwargs"],
    )


def select_tool_choice_for_model(
    organization_name: str,
    model_name: str,
    tool_name: str,
) -> str:
    """Select the strongest portable tool-choice mode for one model."""
    model_spec = LLM_REGISTRY.organizations[organization_name].models[model_name]
    allowed_modes = model_spec.request_rules.allowed_tool_choice_modes
    if allowed_modes is None or "tool" in allowed_modes:
        return tool_name
    if "any" in allowed_modes:
        return "any"
    if "auto" in allowed_modes:
        return "auto"
    raise pytest.UsageError(
        f"{organization_name}/{model_name} has no tool-call mode enabled "
        "in its registered request rules"
    )


def chat_with_transient_retry(adapter, **kwargs):
    """Retry ``adapter.chat()`` on transient errors or model refusals."""
    for attempt in range(_MAX_ATTEMPTS):
        try:
            response = adapter.chat(**kwargs)
        except _TRANSIENT_ERRORS:
            if attempt == _MAX_ATTEMPTS - 1:
                raise
            time.sleep(_RETRY_DELAYS[attempt])
            continue
        if response.finish_reason != "refusal" or attempt == _MAX_ATTEMPTS - 1:
            return response
        time.sleep(_RETRY_DELAYS[attempt])
    return response


def stream_with_transient_retry(adapter, **kwargs) -> list[str]:
    """Retry a complete synchronous stream on transient provider errors."""
    on_retry = kwargs.pop("on_retry", None)
    for attempt in range(_MAX_ATTEMPTS):
        try:
            return list(adapter.stream_chat(**kwargs))
        except _TRANSIENT_ERRORS:
            if attempt == _MAX_ATTEMPTS - 1:
                raise
            if on_retry is not None:
                on_retry()
            time.sleep(_RETRY_DELAYS[attempt])
    return []


async def async_chat_with_transient_retry(adapter, **kwargs):
    """Retry ``adapter.achat()`` on transient errors or model refusals."""
    for attempt in range(_MAX_ATTEMPTS):
        try:
            response = await adapter.achat(**kwargs)
        except _TRANSIENT_ERRORS:
            if attempt == _MAX_ATTEMPTS - 1:
                raise
            await asyncio.sleep(_RETRY_DELAYS[attempt])
            continue
        if response.finish_reason != "refusal" or attempt == _MAX_ATTEMPTS - 1:
            return response
        await asyncio.sleep(_RETRY_DELAYS[attempt])
    return response


async def async_stream_with_transient_retry(adapter, **kwargs) -> list[str]:
    """Retry a complete asynchronous stream on transient provider errors."""
    on_retry = kwargs.pop("on_retry", None)
    for attempt in range(_MAX_ATTEMPTS):
        try:
            chunks = []
            async for chunk in adapter.astream_chat(**kwargs):
                chunks.append(chunk)
            return chunks
        except _TRANSIENT_ERRORS:
            if attempt == _MAX_ATTEMPTS - 1:
                raise
            if on_retry is not None:
                on_retry()
            await asyncio.sleep(_RETRY_DELAYS[attempt])
    return []


def make_document_message(prompt: str, document: DocumentPart) -> UserMessage:
    """Build one portable user message carrying a PDF document part."""
    return UserMessage(prompt, files=[document])

