"""Conformance checks for registry-backed payload rules at the adapter boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
from unittest.mock import AsyncMock, Mock, patch
import warnings

import pytest

from src.llm_api_adapter.adapters.anthropic.adapter import AnthropicAdapter
from src.llm_api_adapter.adapters.google.adapter import GoogleAdapter
from src.llm_api_adapter.adapters.openai.adapter import OpenAIAdapter
from src.llm_api_adapter.llms.anthropic.async_client import ClaudeAsyncClient
from src.llm_api_adapter.llms.anthropic.sync_client import ClaudeSyncClient
from src.llm_api_adapter.llms.google.async_client import GeminiAsyncClient
from src.llm_api_adapter.llms.google.sync_client import GeminiSyncClient
from src.llm_api_adapter.llms.openai.async_client import OpenAIAsyncClient
from src.llm_api_adapter.llms.openai.sync_client import OpenAISyncClient
from src.llm_api_adapter.models.messages.chat_message import UserMessage


@dataclass(frozen=True)
class _Response:
    data: dict[str, Any]

    def json(self) -> dict[str, Any]:
        return self.data


@dataclass(frozen=True)
class RequestRuleConformanceCase:
    name: str
    adapter_class: type[Any]
    sync_client_class: type[Any]
    async_client_class: type[Any]
    model: str
    message_key: str
    ignored_paths: tuple[str, ...]
    default_kwargs: dict[str, Any]
    non_default_kwargs: dict[str, Any]
    response_factory: Callable[[], dict[str, Any]]


def _openai_response() -> dict[str, Any]:
    return {
        "id": "resp_123",
        "model": "gpt-5-nano",
        "created_at": 123,
        "status": "completed",
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        "output": [{
            "type": "message",
            "content": [{"type": "output_text", "text": "ok"}],
        }],
    }


def _anthropic_response() -> dict[str, Any]:
    return {
        "id": "msg_123",
        "model": "claude-sonnet-4-5",
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": "ok"}],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def _google_response() -> dict[str, Any]:
    return {
        "modelVersion": "gemini-2.5-flash",
        "candidates": [{
            "content": {"parts": [{"text": "ok"}]},
            "finishReason": "STOP",
        }],
        "usageMetadata": {
            "promptTokenCount": 1,
            "candidatesTokenCount": 1,
            "totalTokenCount": 2,
        },
    }


CASES = (
    pytest.param(
        RequestRuleConformanceCase(
            name="openai",
            adapter_class=OpenAIAdapter,
            sync_client_class=OpenAISyncClient,
            async_client_class=OpenAIAsyncClient,
            model="gpt-5-nano",
            message_key="input",
            ignored_paths=("top_p", "temperature"),
            default_kwargs={"max_tokens": 64, "top_p": 1.0, "temperature": 1.0},
            non_default_kwargs={"max_tokens": 64, "top_p": 0.2, "temperature": 0.2},
            response_factory=_openai_response,
        ),
        id="openai",
    ),
    pytest.param(
        RequestRuleConformanceCase(
            name="anthropic",
            adapter_class=AnthropicAdapter,
            sync_client_class=ClaudeSyncClient,
            async_client_class=ClaudeAsyncClient,
            model="claude-sonnet-4-5",
            message_key="messages",
            ignored_paths=("top_p",),
            default_kwargs={"max_tokens": 64, "top_p": 1.0},
            non_default_kwargs={"max_tokens": 64, "top_p": 0.2},
            response_factory=_anthropic_response,
        ),
        id="anthropic",
    ),
    pytest.param(
        RequestRuleConformanceCase(
            name="google",
            adapter_class=GoogleAdapter,
            sync_client_class=GeminiSyncClient,
            async_client_class=GeminiAsyncClient,
            model="gemini-2.5-flash",
            message_key="contents",
            ignored_paths=("generationConfig.maxOutputTokens",),
            default_kwargs={"max_tokens": None},
            non_default_kwargs={"max_tokens": 64},
            response_factory=_google_response,
        ),
        id="google",
    ),
    pytest.param(
        RequestRuleConformanceCase(
            name="google-gemini-3-7",
            adapter_class=GoogleAdapter,
            sync_client_class=GeminiSyncClient,
            async_client_class=GeminiAsyncClient,
            model="gemini-3.7-flash",
            message_key="contents",
            ignored_paths=(
                "generationConfig.temperature",
                "generationConfig.topP",
            ),
            default_kwargs={"max_tokens": 64},
            non_default_kwargs={
                "max_tokens": 64,
                "temperature": 0.2,
                "top_p": 0.2,
            },
            response_factory=_google_response,
        ),
        id="google-gemini-3-7",
    ),
)


def _make_adapter(case: RequestRuleConformanceCase):
    return case.adapter_class(api_key="test_api_key", model=case.model)


def _sync_chat(case: RequestRuleConformanceCase, kwargs: dict[str, Any]):
    transport = Mock(return_value=_Response(case.response_factory()))
    with patch.object(case.sync_client_class, "_send_request", new=transport):
        response = _make_adapter(case).chat([UserMessage("hi")], **kwargs)
    return response, transport.call_args.args[1]


async def _async_chat(case: RequestRuleConformanceCase, kwargs: dict[str, Any]):
    transport = AsyncMock(return_value=case.response_factory())
    with patch.object(case.async_client_class, "_send_request", new=transport):
        response = await _make_adapter(case).achat([UserMessage("hi")], **kwargs)
    return response, transport.await_args.args[1]


def _assert_paths_are_absent(payload: dict[str, Any], paths: tuple[str, ...]) -> None:
    for path in paths:
        target = payload
        *parents, leaf = path.split(".")
        for parent in parents:
            target = target[parent]
        assert leaf not in target


def _expected_warning_messages(case: RequestRuleConformanceCase) -> list[str]:
    return [
        f"Parameter {path!r} is not supported for model {case.model!r} and will be ignored."
        for path in case.ignored_paths
    ]


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("case", CASES)
async def test_adapter_chat_silently_omits_registered_default_values(case):
    with warnings.catch_warnings(record=True) as sync_warnings:
        warnings.simplefilter("always")
        sync_response, sync_payload = _sync_chat(
            case,
            dict(case.default_kwargs),
        )
    with warnings.catch_warnings(record=True) as async_warnings:
        warnings.simplefilter("always")
        async_response, async_payload = await _async_chat(
            case,
            dict(case.default_kwargs),
        )

    assert sync_response.content == async_response.content == "ok"
    assert sync_payload == async_payload
    assert sync_payload["model"] == case.model
    assert sync_payload[case.message_key]
    _assert_paths_are_absent(sync_payload, case.ignored_paths)
    assert sync_warnings == []
    assert async_warnings == []


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("case", CASES)
async def test_adapter_chat_warns_for_registered_non_default_values(case):
    with warnings.catch_warnings(record=True) as sync_warnings:
        warnings.simplefilter("always")
        sync_response, sync_payload = _sync_chat(
            case,
            dict(case.non_default_kwargs),
        )
    with warnings.catch_warnings(record=True) as async_warnings:
        warnings.simplefilter("always")
        async_response, async_payload = await _async_chat(
            case,
            dict(case.non_default_kwargs),
        )

    assert sync_response.content == async_response.content == "ok"
    assert sync_payload == async_payload
    _assert_paths_are_absent(sync_payload, case.ignored_paths)
    expected_messages = _expected_warning_messages(case)
    assert [str(warning.message) for warning in sync_warnings] == expected_messages
    assert [str(warning.message) for warning in async_warnings] == expected_messages
