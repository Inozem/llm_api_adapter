"""Deterministic wire-policy checks for the private compatible client mode."""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest
import requests

from src.llm_api_adapter.errors.llm_api_error import LLMAPIClientError
from src.llm_api_adapter.llms.openai.async_client import OpenAIAsyncClient
from src.llm_api_adapter.llms.openai.sync_client import OpenAISyncClient
from tests.fixtures.openai_compatible import (
    text_response,
    text_sse_lines,
    zero_usage_payload,
)


BASE = "https://compatible.test/prefix/v1"
MESSAGES = [{"role": "user", "content": "hello"}]


class _RequestsResponse:
    def __init__(self, *, status_code=200, body=None, lines=None):
        self.status_code = status_code
        self.body = body if body is not None else text_response()
        self.lines = lines if lines is not None else text_sse_lines()

    def json(self):
        return self.body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError("rejected", response=self)

    def iter_lines(self, decode_unicode=True):
        return iter(self.lines)

    def close(self):
        return None


def _httpx_response(request, *, streaming, status_code=200, usage=None):
    if status_code >= 400:
        return httpx.Response(
            status_code,
            json={"error": {"type": "invalid_request_error", "message": "unsupported parameter"}},
            request=request,
        )
    if streaming:
        sse_lines = text_sse_lines() if usage is None else text_sse_lines(usage=usage)
        return httpx.Response(
            status_code,
            content="".join(f"{line}\n" for line in sse_lines),
            request=request,
        )
    return httpx.Response(status_code, json=text_response(), request=request)


@pytest.mark.unit
@pytest.mark.parametrize("model", ["unregistered-name", "gpt-5", "gpt-4.1-mini"])
def test_compatible_payload_maps_fixed_policy_without_registry_lookup(model):
    client = OpenAISyncClient(api_key="test-key")
    client._set_official_behavior_enabled(False)
    options = {"vendor_option": "keep"}
    kwargs = {
        "messages": MESSAGES,
        "max_tokens": 19,
        "max_completion_tokens": 4,
        "temperature": 0.3,
        "top_p": 0.7,
        "stream_options": options,
    }
    with (
        patch("src.llm_api_adapter.llms.openai.sync_client.model_api_variant", side_effect=AssertionError),
        patch("src.llm_api_adapter.llms.openai.sync_client.model_spec_for", side_effect=AssertionError),
        patch("src.llm_api_adapter.llms.openai.sync_client.apply_model_request_rules", side_effect=AssertionError),
    ):
        ordinary = client._prepare_chat_payload_for_model(model, kwargs)
        streamed = client._prepare_chat_payload_for_model(model, kwargs, streaming=True)

    assert ordinary == {
        "model": model,
        "messages": MESSAGES,
        "max_completion_tokens": 19,
        "temperature": 0.3,
        "top_p": 0.7,
        "stream_options": options,
    }
    assert streamed["stream_options"] == {"vendor_option": "keep", "include_usage": True}
    assert streamed["stream_options"] is not options
    assert "max_tokens" not in ordinary and "max_tokens" not in streamed
    assert options == {"vendor_option": "keep"}
    assert kwargs["max_tokens"] == 19


@pytest.mark.unit
@pytest.mark.parametrize("value", [None, 1, 1.0, True, "1", 0.4])
def test_compatible_limit_and_sampling_defaults_are_exact(value):
    client = OpenAISyncClient(api_key="test-key")
    client._set_official_behavior_enabled(False)
    payload = client._prepare_chat_payload_for_model(
        "gpt-4.1-mini",
        {"messages": MESSAGES, "max_tokens": value, "temperature": value, "top_p": value},
    )
    assert "max_tokens" not in payload
    assert ("max_completion_tokens" in payload) is (value is not None)
    for key in ("temperature", "top_p"):
        assert (key in payload) is not (type(value) in (int, float) and value == 1)
        if key in payload:
            assert payload[key] == value


@pytest.mark.unit
def test_compatible_unset_limit_is_omitted():
    client = OpenAISyncClient(api_key="test-key")
    client._set_official_behavior_enabled(False)
    payload = client._prepare_chat_payload_for_model("unknown", {"messages": MESSAGES})
    assert "max_tokens" not in payload
    assert "max_completion_tokens" not in payload


@pytest.mark.unit
@pytest.mark.parametrize("transport", ["requests", "httpx"])
@pytest.mark.parametrize("streaming", [False, True])
def test_sync_compatible_transport_sends_expected_wire_payload(transport, streaming):
    calls = []
    client = OpenAISyncClient(api_key="test-key", endpoint=BASE, transport=transport)
    client._set_official_behavior_enabled(False)
    params = {"messages": MESSAGES, "max_tokens": 8, "temperature": 1.0, "top_p": 0.25}
    expected = {
        "model": "gpt-5",
        "messages": MESSAGES,
        "max_completion_tokens": 8,
        "top_p": 0.25,
    }
    if streaming:
        expected.update(stream_options={"include_usage": True}, stream=True)
    if transport == "requests":
        def post(url, **kwargs):
            calls.append((url, json.loads(json.dumps(kwargs["json"]))))
            return _RequestsResponse(
                lines=text_sse_lines() if streaming else None
            )
        context = patch("src.llm_api_adapter.llms.requests_transport.requests.post", side_effect=post)
    else:
        def handler(request):
            calls.append((str(request.url), json.loads(request.content)))
            return _httpx_response(request, streaming=streaming)

        real_client = httpx.Client
        context = patch.object(
            httpx,
            "Client",
            side_effect=lambda: real_client(transport=httpx.MockTransport(handler)),
        )

    with context:
        if streaming:
            list(client.stream("gpt-5", **params))
        else:
            client.complete("gpt-5", **params)

    assert calls == [(f"{BASE}/chat/completions", expected)]
    assert params == {"messages": MESSAGES, "max_tokens": 8, "temperature": 1.0, "top_p": 0.25}


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("streaming", [False, True])
async def test_async_compatible_httpx_sends_expected_wire_payload(streaming):
    calls = []
    client = OpenAIAsyncClient(api_key="test-key", endpoint=BASE)
    client._set_official_behavior_enabled(False)
    params = {"messages": MESSAGES, "max_tokens": 8, "temperature": 1.0, "top_p": 0.25}

    real_async_client = httpx.AsyncClient

    def handler(request):
        calls.append((str(request.url), json.loads(request.content)))
        if streaming:
            return httpx.Response(
                200,
                content="".join(f"{line}\n" for line in text_sse_lines()),
                request=request,
            )
        return httpx.Response(200, json=text_response(), request=request)

    def make_client():
        return real_async_client(transport=httpx.MockTransport(handler))

    with patch.object(httpx, "AsyncClient", side_effect=make_client):
        if streaming:
            [event async for event in client.stream("arbitrary", **params)]
        else:
            await client.complete("arbitrary", **params)

    expected = {
        "model": "arbitrary",
        "messages": MESSAGES,
        "max_completion_tokens": 8,
        "top_p": 0.25,
    }
    if streaming:
        expected.update(stream_options={"include_usage": True}, stream=True)
    assert calls == [(f"{BASE}/chat/completions", expected)]


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("streaming", [False, True])
async def test_async_compatible_server_rejection_is_not_retried(streaming):
    calls = []
    client = OpenAIAsyncClient(api_key="test-key", endpoint=BASE)
    client._set_official_behavior_enabled(False)
    real_async_client = httpx.AsyncClient

    def handler(request):
        calls.append((str(request.url), json.loads(request.content)))
        return _httpx_response(request, streaming=streaming, status_code=400)

    def make_client():
        return real_async_client(transport=httpx.MockTransport(handler))

    with patch.object(httpx, "AsyncClient", side_effect=make_client):
        with pytest.raises(LLMAPIClientError, match="unsupported parameter"):
            if streaming:
                [event async for event in client.stream("gpt-5", **_rejection_params())]
            else:
                await client.complete("gpt-5", **_rejection_params())

    assert calls == [
        (f"{BASE}/chat/completions", _rejection_wire_payload("gpt-5", streaming))
    ]


@pytest.mark.unit
@pytest.mark.parametrize("transport", ["requests", "httpx"])
@pytest.mark.parametrize(
    ("reported_usage", "expected_usage"),
    [(None, None), (zero_usage_payload(), zero_usage_payload())],
    ids=("missing-usage", "reported-zero-usage"),
)
def test_compatible_stream_returns_reported_usage_without_inventing_it(
    transport, reported_usage, expected_usage
):
    calls = []
    client = OpenAISyncClient(api_key="test-key", endpoint=BASE, transport=transport)
    client._set_official_behavior_enabled(False)
    sse_lines = (
        text_sse_lines()
        if reported_usage is None
        else text_sse_lines(usage=reported_usage)
    )
    if transport == "requests":
        def post(url, **kwargs):
            calls.append((url, json.loads(json.dumps(kwargs["json"]))))
            return _RequestsResponse(lines=sse_lines)
        context = patch("src.llm_api_adapter.llms.requests_transport.requests.post", side_effect=post)
    else:
        def handler(request):
            calls.append((str(request.url), json.loads(request.content)))
            return _httpx_response(
                request,
                streaming=True,
                usage=reported_usage,
            )
        real_client = httpx.Client
        context = patch.object(
            httpx,
            "Client",
            side_effect=lambda: real_client(transport=httpx.MockTransport(handler)),
        )

    with context:
        events = list(client.stream("unknown-model", messages=MESSAGES))

    assert calls == [
        (
            f"{BASE}/chat/completions",
            {
                "model": "unknown-model",
                "messages": MESSAGES,
                "stream_options": {"include_usage": True},
                "stream": True,
            },
        )
    ]
    usage_events = [
        event.data["usage"] for event in events if "usage" in event.data
    ]
    if expected_usage is None:
        assert usage_events == []
    else:
        assert usage_events == [expected_usage]


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize(
    ("reported_usage", "expected_usage"),
    [(None, None), (zero_usage_payload(), zero_usage_payload())],
    ids=("missing-usage", "reported-zero-usage"),
)
async def test_async_compatible_stream_returns_reported_usage_without_inventing_it(
    reported_usage, expected_usage
):
    calls = []
    client = OpenAIAsyncClient(api_key="test-key", endpoint=BASE)
    client._set_official_behavior_enabled(False)
    real_async_client = httpx.AsyncClient

    def handler(request):
        calls.append((str(request.url), json.loads(request.content)))
        return _httpx_response(
            request,
            streaming=True,
            usage=reported_usage,
        )

    def make_client():
        return real_async_client(transport=httpx.MockTransport(handler))

    with patch.object(httpx, "AsyncClient", side_effect=make_client):
        events = [
            event
            async for event in client.stream(
                "unknown-model",
                messages=MESSAGES,
            )
        ]

    assert calls == [
        (
            f"{BASE}/chat/completions",
            {
                "model": "unknown-model",
                "messages": MESSAGES,
                "stream_options": {"include_usage": True},
                "stream": True,
            },
        )
    ]
    usage_events = [
        event.data["usage"] for event in events if "usage" in event.data
    ]
    if expected_usage is None:
        assert usage_events == []
    else:
        assert usage_events == [expected_usage]


@pytest.mark.unit
@pytest.mark.parametrize("transport", ["requests", "httpx"])
@pytest.mark.parametrize("streaming", [False, True])
def test_compatible_server_rejection_is_not_retried(transport, streaming):
    calls = []
    client = OpenAISyncClient(api_key="test-key", endpoint=BASE, transport=transport)
    client._set_official_behavior_enabled(False)
    if transport == "requests":
        def post(url, **kwargs):
            calls.append((url, json.loads(json.dumps(kwargs["json"]))))
            return _RequestsResponse(
                status_code=400,
                body={"error": {"type": "invalid_request_error", "message": "unsupported parameter"}},
            )
        context = patch("src.llm_api_adapter.llms.requests_transport.requests.post", side_effect=post)
    else:
        def handler(request):
            calls.append((str(request.url), json.loads(request.content)))
            return _httpx_response(request, streaming=streaming, status_code=400)
        real_client = httpx.Client
        context = patch.object(
            httpx,
            "Client",
            side_effect=lambda: real_client(transport=httpx.MockTransport(handler)),
        )

    with context, pytest.raises(LLMAPIClientError, match="unsupported parameter"):
        if streaming:
            list(client.stream("gpt-5", **_rejection_params()))
        else:
            client.complete("gpt-5", **_rejection_params())
    assert calls == [
        (f"{BASE}/chat/completions", _rejection_wire_payload("gpt-5", streaming))
    ]


def _rejection_params():
    return {
        "messages": MESSAGES,
        "max_tokens": 5,
        "temperature": 0.25,
        "top_p": 0.6,
    }


def _rejection_wire_payload(model, streaming):
    payload = {
        "model": model,
        "messages": MESSAGES,
        "max_completion_tokens": 5,
        "temperature": 0.25,
        "top_p": 0.6,
    }
    if streaming:
        payload.update(stream_options={"include_usage": True}, stream=True)
    return payload
