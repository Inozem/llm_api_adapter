"""Generic Messages usage merge behavior on real mocked stream transports."""

import asyncio
import json
import socket

import httpx
import pytest
import requests
import requests_mock

from src.llm_api_adapter.adapters.anthropic_compatible.adapter import (
    AnthropicCompatibleAdapter,
)
from src.llm_api_adapter.models.messages.chat_message import UserMessage
from tests.fixtures.anthropic_compatible import messages_event_trace


ROOT = "https://messages.example.invalid/proxy/v1"
URL = ROOT + "/messages"
MODEL = "vendor/usage-stream-contract"
KEY = "synthetic-usage-key"
START_USAGE = {
    "input_tokens": 100,
    "output_tokens": 2,
    "cache_read_input_tokens": 20,
    "cache_creation_input_tokens": 10,
    "prompt_tokens_details": {"cached_tokens": 20},
}
EXPECTED_UPDATED = (135, 5, 140, 25, 10)
_NO_USAGE = object()
pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _block_network_resolution(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("usage stream tests must not resolve network hosts")
        ),
    )


def _sse_body(events, *, include_event_header=True):
    return "".join(
        (f"event: {event.event}\n" if include_event_header else "")
        + f"data: {json.dumps(event.data)}\n\n"
        for event in events
    ).encode("utf-8")


def _httpx_client(monkeypatch, body, *, asynchronous):
    requests_seen = []
    clients = []

    def response_for(request):
        requests_seen.append(request)
        return httpx.Response(
            200,
            content=body,
            headers={"content-type": "text/event-stream"},
            request=request,
        )

    if asynchronous:
        original = httpx.AsyncClient

        def make_client(*args, **kwargs):
            async def handler(request):
                return response_for(request)

            kwargs.update(transport=httpx.MockTransport(handler), trust_env=False)
            client = original(*args, **kwargs)
            clients.append(client)
            return client

        monkeypatch.setattr(httpx, "AsyncClient", make_client)
    else:
        original = httpx.Client

        def make_client(*args, **kwargs):
            kwargs.update(transport=httpx.MockTransport(response_for), trust_env=False)
            client = original(*args, **kwargs)
            clients.append(client)
            return client

        monkeypatch.setattr(httpx, "Client", make_client)
    return requests_seen, clients


def _events(usage_delta):
    options = {"blocks": [{"type": "text", "text": "tail"}]}
    if usage_delta is not _NO_USAGE:
        options.update(usage_start=START_USAGE, usage_delta=usage_delta)
    return messages_event_trace(**options)


@pytest.mark.parametrize("lane", ["requests", "httpx", "async-httpx"])
@pytest.mark.parametrize("include_event_header", [True, False], ids=["event-name", "data-type-only"])
@pytest.mark.parametrize(
    ("usage_delta", "expected"),
    [
        ({"output_tokens": 5, "cache_read_input_tokens": 25}, EXPECTED_UPDATED),
        ({"output_tokens": 5, "prompt_tokens_details": {"cached_tokens": 25}}, EXPECTED_UPDATED),
        (None, None),
        (["invalid-usage-shape"], None),
        (_NO_USAGE, None),
    ],
    ids=[
        "late-primary-read", "late-alias-read", "unknown-final-update",
        "malformed-final-update", "no-usage-ever",
    ],
)
def test_late_usage_updates_replace_alias_group_and_clear_unknown(
    lane, include_event_header, usage_delta, expected, monkeypatch
):
    body = _sse_body(_events(usage_delta), include_event_header=include_event_header)
    async_mode = lane == "async-httpx"
    transport = "requests" if lane == "requests" else "httpx"
    adapter = AnthropicCompatibleAdapter(
        api_key=KEY, model=MODEL, base_url=ROOT, transport=transport
    )
    if lane == "requests":
        server = requests_mock.Mocker(real_http=False)
        server.start()
        server.register_uri(
            "POST", URL,
            content=body,
            headers={"content-type": "text/event-stream"},
        )
    else:
        requests_seen, clients = _httpx_client(
            monkeypatch, body, asynchronous=async_mode
        )

    chunks, completed = [], []
    kwargs = {"max_tokens": 32, "buffer_chars": 100, "on_chunk": chunks.append,
              "on_done": completed.append}
    try:
        if async_mode:
            async def run():
                return [chunk async for chunk in adapter.astream_chat([UserMessage("hi")], **kwargs)]
            text = asyncio.run(run())
            assert requests_seen and len(requests_seen) == 1
            assert all(client.is_closed for client in clients)
        else:
            text = list(adapter.stream_chat([UserMessage("hi")], **kwargs))
            if lane == "requests":
                assert len(server.request_history) == 1
                assert server.request_history[0].url == URL
            else:
                assert len(requests_seen) == 1
                assert all(client.is_closed for client in clients)
        assert text == ["tail"]
        assert len(chunks) == len(completed) == 1
        for usage in (completed[0].usage, chunks[0].usage):
            if expected is None:
                if usage_delta is _NO_USAGE:
                    assert usage is None
                else:
                    assert usage is None or all(
                        value is None for value in vars(usage).values()
                    )
            else:
                assert (
                    usage.input_tokens,
                    usage.output_tokens,
                    usage.total_tokens,
                    usage.cached_tokens,
                    usage.cache_write_tokens,
                ) == expected
    finally:
        if lane == "requests":
            server.stop()
