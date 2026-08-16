"""Tests for the opt-in synchronous HTTPX transport."""

from __future__ import annotations

import builtins
from unittest.mock import patch

import httpx
import pytest

from src.llm_api_adapter.errors.llm_api_error import (
    LLMAPIClientError,
    LLMAPIRateLimitError,
    LLMAPITimeoutError,
)
from src.llm_api_adapter.llms.httpx_transport import HttpxSyncTransport
from src.llm_api_adapter.llms.requests_transport import RequestsSyncTransport
from src.llm_api_adapter.llms.transports import (
    SSEEvent,
    TransportRequest,
    create_sync_transport,
)
from src.llm_api_adapter.models.messages.chat_message import UserMessage
from src.llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter


class FakeResponse:
    def __init__(
        self,
        *,
        body: dict | None = None,
        lines: list[str] | None = None,
        status_code: int = 200,
    ) -> None:
        self.body = body or {}
        self.lines = lines or []
        self.status_code = status_code
        self.request = httpx.Request("POST", "https://example.test")
        self.close_calls = 0

    def json(self) -> dict:
        return self.body

    def iter_lines(self):
        return iter(self.lines)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=self.request,
                response=self,
            )

    def close(self) -> None:
        self.close_calls += 1


class FakeHttpxClient:
    def __init__(self, *, response: FakeResponse | None = None, post_error=None) -> None:
        self.response = response
        self.post_error = post_error
        self.post_calls = []
        self.build_request_calls = []
        self.send_calls = []
        self.close_calls = 0

    def post(self, url, **kwargs):
        self.post_calls.append((url, kwargs))
        if self.post_error is not None:
            raise self.post_error
        return self.response

    def build_request(self, method, url, **kwargs):
        self.build_request_calls.append((method, url, kwargs))
        return httpx.Request(
            method,
            url,
            headers=kwargs.get("headers"),
            json=kwargs.get("json"),
        )

    def send(self, request, *, stream=False):
        self.send_calls.append((request, stream))
        return self.response

    def close(self) -> None:
        self.close_calls += 1


@pytest.mark.unit
def test_httpx_sync_transport_posts_json_and_closes_resources():
    response = FakeResponse(body={"answer": "ok"})
    client = FakeHttpxClient(response=response)
    request = TransportRequest(
        url="https://example.test/messages",
        headers={"Authorization": "Bearer test"},
        payload={"message": "Hello"},
        timeout=3.0,
    )

    with patch.object(httpx, "Client", return_value=client):
        assert HttpxSyncTransport().post_json(request).json() == {"answer": "ok"}

    assert client.post_calls == [
        (
            "https://example.test/messages",
            {
                "headers": {"Authorization": "Bearer test"},
                "json": {"message": "Hello"},
                "timeout": 3.0,
            },
        )
    ]
    assert response.close_calls == 1
    assert client.close_calls == 1


@pytest.mark.unit
def test_httpx_sync_transport_uses_provider_http_error_handler():
    response = FakeResponse(status_code=429)
    client = FakeHttpxClient(response=response)
    request = TransportRequest(url="https://example.test/messages")
    observed = []

    def provider_handler(error):
        observed.append(error.response)
        raise LLMAPIRateLimitError(detail="provider mapping")

    with patch.object(httpx, "Client", return_value=client):
        with pytest.raises(LLMAPIRateLimitError, match="provider mapping"):
            HttpxSyncTransport().post_json(
                request,
                http_error_handler=provider_handler,
            )

    assert observed == [response]
    assert response.close_calls == 1
    assert client.close_calls == 1


@pytest.mark.unit
def test_httpx_sync_transport_maps_timeout_and_closes_client():
    client = FakeHttpxClient(post_error=httpx.TimeoutException("timed out"))

    with patch.object(httpx, "Client", return_value=client):
        with pytest.raises(LLMAPITimeoutError):
            HttpxSyncTransport().post_json(TransportRequest(url="https://example.test"))

    assert client.close_calls == 1


@pytest.mark.unit
def test_httpx_sync_transport_streams_sse_and_closes_on_early_close():
    response = FakeResponse(
        lines=[
            'data: {"sequence": 1}',
            "",
            'data: {"sequence": 2}',
            "",
        ]
    )
    client = FakeHttpxClient(response=response)
    request = TransportRequest(
        url="https://example.test/stream",
        payload={"message": "Hello"},
        timeout=3.0,
    )

    with patch.object(httpx, "Client", return_value=client):
        events = HttpxSyncTransport().post_sse(request)
        assert next(events) == SSEEvent(event=None, data={"sequence": 1})
        events.close()

    assert client.build_request_calls[0] == (
        "POST",
        "https://example.test/stream",
        {"headers": {}, "json": {"message": "Hello"}, "timeout": 3.0},
    )
    assert client.send_calls[0][1] is True
    assert response.close_calls == 1
    assert client.close_calls == 1


@pytest.mark.unit
def test_transport_factory_selects_requests_or_httpx():
    assert isinstance(create_sync_transport("requests"), RequestsSyncTransport)
    assert isinstance(create_sync_transport("httpx"), HttpxSyncTransport)


@pytest.mark.unit
def test_requests_selection_does_not_import_httpx():
    real_import = builtins.__import__

    def missing_httpx(name, *args, **kwargs):
        if name == "httpx":
            raise ImportError("httpx is missing")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=missing_httpx):
        assert isinstance(create_sync_transport("requests"), RequestsSyncTransport)


@pytest.mark.unit
def test_httpx_selection_explains_missing_optional_dependency():
    real_import = builtins.__import__

    def missing_httpx(name, *args, **kwargs):
        if name == "httpx":
            raise ImportError("httpx is missing")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=missing_httpx):
        with pytest.raises(ImportError, match=r"llm-api-adapter\[httpx\]"):
            create_sync_transport("httpx")


@pytest.mark.unit
def test_public_httpx_selection_reports_missing_optional_dependency():
    real_import = builtins.__import__

    def missing_httpx(name, *args, **kwargs):
        if name == "httpx":
            raise ImportError("httpx is missing")
        return real_import(name, *args, **kwargs)

    adapter = UniversalLLMAPIAdapter(
        organization="openai",
        model="gpt-4o",
        api_key="test-key",
        transport="httpx",
    )

    with patch("builtins.__import__", side_effect=missing_httpx):
        with pytest.raises(ImportError, match=r"llm-api-adapter\[httpx\]"):
            adapter.chat([UserMessage("Hello")])


@pytest.mark.unit
def test_transport_factory_rejects_unsupported_values():
    with pytest.raises(ValueError, match="requests.*httpx"):
        create_sync_transport("urllib3")


@pytest.mark.unit
def test_httpx_sync_transport_converts_request_errors():
    client = FakeHttpxClient(post_error=httpx.RequestError("connection failed"))

    with patch.object(httpx, "Client", return_value=client):
        with pytest.raises(LLMAPIClientError):
            HttpxSyncTransport().post_json(TransportRequest(url="https://example.test"))

    assert client.close_calls == 1


@pytest.mark.unit
def test_public_adapter_uses_httpx_only_when_explicitly_selected():
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
    client = FakeHttpxClient(response=response)

    with patch.object(httpx, "Client", return_value=client):
        adapter = UniversalLLMAPIAdapter(
            organization="openai",
            model="gpt-4o",
            api_key="test-key",
            transport="httpx",
        )
        result = adapter.chat([UserMessage("Hello")])

    assert result.content == "Hello"
    assert client.post_calls[0][0] == "https://api.openai.com/v1/chat/completions"
