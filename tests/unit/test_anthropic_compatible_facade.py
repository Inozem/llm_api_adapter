"""Facade-level routing evidence for Anthropic-compatible Messages."""

import asyncio
import json
import socket

import httpx
import pytest
import requests
import requests_mock

import src.llm_api_adapter.adapters.base_adapter as base_adapter_module
import src.llm_api_adapter.llm_registry.llm_registry as registry_module
import src.llm_api_adapter.universal_adapter as universal_module
from src.llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter
from tests.fixtures.anthropic_compatible import (
    events_to_sse_lines,
    messages_event_trace,
    messages_response,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("method", "transport"),
    [
        ("chat", "requests"),
        ("stream_chat", "requests"),
        ("chat", "httpx"),
        ("stream_chat", "httpx"),
        ("achat", "requests"),
        ("astream_chat", "requests"),
    ],
)
def test_facade_routes_successful_messages_without_native_lookup_or_fallback(
    method, transport, monkeypatch
):
    """Exercise the actual facade, adapter and transport with controlled replies."""
    raw_sync_client = httpx.Client
    raw_async_client = httpx.AsyncClient
    model = "claude-sonnet-4-5-20250929"
    api_key = "t012%2fSynthetic+Facade-Key"
    root = "https://messages.invalid/gateway/tenant/v1/"
    endpoint = root.rstrip("/") + "/messages"
    dispatches = []
    sync_httpx_clients = []
    async_httpx_clients = []
    httpx_responses = []
    requests_responses_closed = []

    def forbidden(*_args, **_kwargs):
        raise AssertionError("generic facade setup must not resolve providers or models")

    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(universal_module.ORGANIZATION_PLUGIN_DISCOVERY, "discover", forbidden)
    monkeypatch.setattr(base_adapter_module, "resolve_model_spec", forbidden)
    monkeypatch.setattr(registry_module, "resolve_model_spec", forbidden)
    original_response_close = requests.Response.close

    def track_requests_response_close(response):
        requests_responses_closed.append(response)
        return original_response_close(response)

    monkeypatch.setattr(requests.Response, "close", track_requests_response_close)

    def response_payload(streaming):
        if streaming:
            body = "\n".join(
                events_to_sse_lines(
                    messages_event_trace(
                        blocks=[
                            {"type": "text", "text": "Hello world"},
                            {"type": "thinking", "thinking": "Synthetic reasoning"},
                        ],
                        text_fragments={0: ["Hello", " world"]},
                        model=model,
                    )
                )
            )
            return body.encode(), "text/event-stream"
        body = json.dumps(
            messages_response(
                content=[
                    {"type": "text", "text": "Hello world"},
                    {"type": "thinking", "thinking": "Synthetic reasoning"},
                ],
                model=model,
            )
        )
        return body.encode(), "application/json"

    def sync_factory(*args, **kwargs):
        kwargs.pop("transport", None)

        def handler(request):
            dispatches.append(request)
            body, content_type = response_payload(
                bool(json.loads(request.content).get("stream"))
            )
            response = httpx.Response(
                200, content=body, headers={"content-type": content_type}, request=request
            )
            httpx_responses.append(response)
            return response

        client = raw_sync_client(
            *args, transport=httpx.MockTransport(handler), **kwargs
        )
        sync_httpx_clients.append(client)
        return client

    def async_factory(*args, **kwargs):
        async def handler(request):
            dispatches.append(request)
            body, content_type = response_payload(
                bool(json.loads(request.content).get("stream"))
            )
            response = httpx.Response(
                200, content=body, headers={"content-type": content_type}, request=request
            )
            httpx_responses.append(response)
            return response

        client = raw_async_client(
            *args, transport=httpx.MockTransport(handler), **kwargs
        )
        async_httpx_clients.append(client)
        return client

    monkeypatch.setattr(httpx, "Client", sync_factory)
    monkeypatch.setattr(httpx, "AsyncClient", async_factory)

    facade = UniversalLLMAPIAdapter(
        organization="anthropic_compatible",
        model=model,
        api_key=api_key,
        transport=transport,
        base_url=root,
    )
    assert facade.adapter.model_spec is None
    assert facade.adapter.pricing is None
    assert facade.adapter.base_url == endpoint.removesuffix("/messages")

    request_args = {
        "messages": [{"role": "user", "content": "Say hello"}],
        "max_tokens": 32,
        "capture_reasoning": True,
    }
    def execute_facade_call():
        if method == "chat":
            return facade.chat(**request_args)
        if method == "stream_chat":
            completed = []
            chunks = list(
                facade.stream_chat(
                    **request_args, buffer_chars=2, on_done=completed.append
                )
            )
            assert "".join(chunks) == "Hello world"
            assert all(0 < len(chunk) <= 2 for chunk in chunks)
            assert len(completed) == 1
            return completed[0]
        if method == "achat":
            return asyncio.run(facade.achat(**request_args))

        async def collect_stream():
            completed = []
            chunks = [
                chunk
                async for chunk in facade.astream_chat(
                    **request_args, buffer_chars=2, on_done=completed.append
                )
            ]
            return chunks, completed

        chunks, completed = asyncio.run(collect_stream())
        assert "".join(chunks) == "Hello world"
        assert all(0 < len(chunk) <= 2 for chunk in chunks)
        assert len(completed) == 1
        return completed[0]

    with requests_mock.Mocker(real_http=False) as requests_http:
        body, content_type = response_payload(
            method in ("stream_chat", "astream_chat")
        )
        requests_http.register_uri(
            "POST", endpoint, text=body.decode(),
            headers={"content-type": content_type},
        )
        result = execute_facade_call()
        dispatches.extend(requests_http.request_history)

    assert result.content == "Hello world"
    assert result.model == model
    assert result.reasoning_events
    assert len(dispatches) == 1
    request = dispatches[0]
    assert request.method == "POST"
    assert str(request.url) == endpoint
    assert request.headers["x-api-key"] == api_key
    raw_body = request.content if isinstance(request, httpx.Request) else request.body
    if isinstance(raw_body, str):
        raw_body = raw_body.encode()
    assert json.loads(raw_body)["model"] == model
    if method in ("chat", "stream_chat") and transport == "requests":
        assert len(requests_http.request_history) == 1
        assert sync_httpx_clients == []
        assert async_httpx_clients == []
        assert httpx_responses == []
        assert requests_responses_closed
    elif method in ("chat", "stream_chat"):
        assert requests_http.request_history == []
        assert len(sync_httpx_clients) == 1
        assert async_httpx_clients == []
        assert len(httpx_responses) == 1
        assert sync_httpx_clients[0].is_closed
        assert httpx_responses[0].is_closed
    else:
        assert requests_http.request_history == []
        assert sync_httpx_clients == []
        assert len(async_httpx_clients) == 1
        assert len(httpx_responses) == 1
        assert async_httpx_clients[0].is_closed
        assert httpx_responses[0].is_closed


