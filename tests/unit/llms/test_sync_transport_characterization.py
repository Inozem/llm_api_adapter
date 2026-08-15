"""Characterize the current synchronous ``requests`` transport contract.

These tests intentionally exercise the three built-in sync clients through
their public raw-client methods. A later transport abstraction must preserve
the same HTTP call shape, exception conversion, SSE framing, and cleanup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
from unittest.mock import Mock, patch

import pytest
import requests

from src.llm_api_adapter.errors.llm_api_error import (
    LLMAPIClientError,
    LLMAPIRateLimitError,
    LLMAPITimeoutError,
)
from src.llm_api_adapter.llms.anthropic.sync_client import ClaudeSyncClient
from src.llm_api_adapter.llms.google.sync_client import GeminiSyncClient
from src.llm_api_adapter.llms.openai.sync_client import OpenAISyncClient
from src.llm_api_adapter.llms.streaming import SSEEvent
from src.llm_api_adapter.models.messages.chat_message import UserMessage
from src.llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter


class FakeResponse:
    """Small ``requests.Response`` double with observable stream cleanup."""

    def __init__(
        self,
        *,
        body: dict[str, Any] | None = None,
        lines: list[str] | None = None,
        status_code: int = 200,
        error_payload: dict[str, Any] | None = None,
    ) -> None:
        self.body = body or {}
        self.lines = lines or []
        self.status_code = status_code
        self.error_payload = error_payload or {}
        self.close = Mock()

    def iter_lines(self, decode_unicode: bool = True):
        _ = decode_unicode
        return iter(self.lines)

    def json(self) -> dict[str, Any]:
        return self.error_payload if self.error_payload else self.body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(response=self)


@dataclass(frozen=True)
class SyncClientCase:
    name: str
    client_factory: Callable[[], Any]
    patch_target: str
    model: str
    request_kwargs: dict[str, Any]
    json_url: str
    stream_url: str
    headers: dict[str, str]
    stream_payload: dict[str, Any]
    rate_limit_payload: dict[str, Any]


CASES = (
    SyncClientCase(
        name="openai",
        client_factory=lambda: OpenAISyncClient(api_key="test-key"),
        patch_target="src.llm_api_adapter.llms.openai.sync_client.requests.post",
        model="gpt-4o",
        request_kwargs={"messages": [{"role": "user", "content": "Hello"}]},
        json_url="https://api.openai.com/v1/chat/completions",
        stream_url="https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": "Bearer test-key",
            "Content-Type": "application/json",
        },
        stream_payload={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        },
        rate_limit_payload={
            "error": {"type": "rate_limit_exceeded", "message": "Slow down"}
        },
    ),
    SyncClientCase(
        name="anthropic",
        client_factory=lambda: ClaudeSyncClient(api_key="test-key"),
        patch_target="src.llm_api_adapter.llms.anthropic.sync_client.requests.post",
        model="claude-opus-4-8",
        request_kwargs={
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 32,
        },
        json_url="https://api.anthropic.com/v1/messages",
        stream_url="https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": "test-key",
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        stream_payload={
            "model": "claude-opus-4-8",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 32,
            "stream": True,
        },
        rate_limit_payload={
            "error": {"type": "rate_limit_error", "message": "Slow down"}
        },
    ),
    SyncClientCase(
        name="google",
        client_factory=lambda: GeminiSyncClient(api_key="test-key"),
        patch_target="src.llm_api_adapter.llms.google.sync_client.requests.post",
        model="gemini-2.5-flash",
        request_kwargs={
            "contents": [{"role": "user", "parts": [{"text": "Hello"}]}]
        },
        json_url=(
            "https://generativelanguage.googleapis.com/v1beta/"
            "models/gemini-2.5-flash:generateContent"
        ),
        stream_url=(
            "https://generativelanguage.googleapis.com/v1beta/"
            "models/gemini-2.5-flash:streamGenerateContent?alt=sse"
        ),
        headers={
            "x-goog-api-key": "test-key",
            "Content-Type": "application/json",
        },
        stream_payload={
            "model": "gemini-2.5-flash",
            "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
        },
        rate_limit_payload={
            "error": {"status": "RESOURCE_EXHAUSTED", "message": "Slow down"}
        },
    ),
)


def _complete(case: SyncClientCase):
    return case.client_factory().chat_completion(
        case.model,
        2.5,
        **case.request_kwargs,
    )


def _stream(case: SyncClientCase):
    return case.client_factory().stream(
        case.model,
        2.5,
        **case.request_kwargs,
    )


@pytest.mark.unit
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_sync_json_post_uses_requests_with_provider_owned_wire_contract(case):
    response = FakeResponse(body={"provider": case.name})

    with patch(case.patch_target, return_value=response) as mock_post:
        assert _complete(case) == {"provider": case.name}

    mock_post.assert_called_once_with(
        case.json_url,
        headers=case.headers,
        json={"model": case.model, **case.request_kwargs},
        timeout=2.5,
    )


@pytest.mark.unit
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
@pytest.mark.parametrize(
    ("request_error", "expected_error"),
    (
        (requests.exceptions.Timeout("timed out"), LLMAPITimeoutError),
        (requests.exceptions.RequestException("connection failed"), LLMAPIClientError),
    ),
)
def test_sync_json_post_converts_requests_errors(case, request_error, expected_error):
    with patch(case.patch_target, side_effect=request_error):
        with pytest.raises(expected_error):
            _complete(case)


@pytest.mark.unit
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_sync_sse_post_preserves_framing_and_closes_response(case):
    response = FakeResponse(
        lines=['data: {"transport_contract": true}', ""],
    )

    with patch(case.patch_target, return_value=response) as mock_post:
        events = list(_stream(case))

    assert events == [SSEEvent(event=None, data={"transport_contract": True})]
    mock_post.assert_called_once_with(
        case.stream_url,
        headers=case.headers,
        json=case.stream_payload,
        timeout=2.5,
        stream=True,
    )
    response.close.assert_called_once_with()


@pytest.mark.unit
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_sync_sse_closes_response_when_consumer_stops_early(case):
    response = FakeResponse(
        lines=[
            'data: {"sequence": 1}',
            "",
            'data: {"sequence": 2}',
            "",
        ],
    )

    with patch(case.patch_target, return_value=response):
        events = _stream(case)
        assert next(events) == SSEEvent(event=None, data={"sequence": 1})
        events.close()

    response.close.assert_called_once_with()


@pytest.mark.unit
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_sync_sse_uses_provider_specific_http_error_mapping(case):
    response = FakeResponse(
        status_code=429,
        error_payload=case.rate_limit_payload,
    )

    with patch(case.patch_target, return_value=response):
        with pytest.raises(LLMAPIRateLimitError, match="Slow down"):
            list(_stream(case))

    response.close.assert_called_once_with()


@pytest.mark.unit
def test_public_adapter_default_sync_path_uses_requests():
    response = FakeResponse(
        body={
            "id": "chatcmpl_test",
            "model": "gpt-4o",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hello"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )

    with patch(
        "src.llm_api_adapter.llms.openai.sync_client.requests.post",
        return_value=response,
    ) as mock_post:
        adapter = UniversalLLMAPIAdapter(
            organization="openai",
            model="gpt-4o",
            api_key="test-key",
        )
        result = adapter.chat([UserMessage("Hello")])

    assert result.content == "Hello"
    assert mock_post.call_args.args[0] == (
        "https://api.openai.com/v1/chat/completions"
    )
