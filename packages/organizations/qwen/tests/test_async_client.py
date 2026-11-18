"""Direct deterministic checks for Qwen's asynchronous Messages client."""

from pathlib import Path
import inspect
import sys
from types import SimpleNamespace

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[2]
for source in (PACKAGE_ROOT / "src", REPOSITORY_ROOT / "src", REPOSITORY_ROOT):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from llm_api_adapter.errors.llm_api_error import LLMAPIAuthorizationError
from llm_api_adapter.llms.transports import SSEEvent
from llm_api_adapter_qwen.clients.sync_client import QwenMessagesSyncClient
from llm_api_adapter_qwen.clients import async_client as async_client_module
from llm_api_adapter_qwen.clients.async_client import QwenMessagesAsyncClient


class FakeHTTPError(Exception):
    def __init__(self, status_code, detail):
        super().__init__(detail)
        self.response = SimpleNamespace(status_code=status_code, json=lambda: {})


@pytest.mark.asyncio
@pytest.mark.unit
async def test_async_client_zero_argument_construction_and_native_helper_forwarding(
    monkeypatch,
):
    client = QwenMessagesAsyncClient()
    constructor = inspect.signature(QwenMessagesAsyncClient)
    assert not constructor.parameters
    with pytest.raises(TypeError):
        QwenMessagesAsyncClient(object())
    for method in (client.chat, client.stream):
        parameters = inspect.signature(method).parameters
        assert list(parameters) == ["api_key", "workspace_id", "payload", "timeout_s"]
        assert all(
            parameter.kind is inspect.Parameter.KEYWORD_ONLY
            for parameter in parameters.values()
        )
        assert all(parameter.default is inspect.Parameter.empty for parameter in parameters.values())
    with pytest.raises(TypeError):
        client.chat("synthetic-key", "workspace-one", {}, None)
    calls = []
    stream_calls = []

    async def fake_async_request(url, **kwargs):
        calls.append((url, kwargs))
        return {"ok": True}

    def fake_async_stream_request(url, **kwargs):
        stream_calls.append((url, kwargs))

        async def events():
            yield SSEEvent(event="message_stop", data={"type": "message_stop"})

        return events()

    monkeypatch.setattr(async_client_module, "async_request", fake_async_request)
    monkeypatch.setattr(
        async_client_module,
        "async_stream_request",
        fake_async_stream_request,
    )
    payload = {"model": "qwen3.8-max"}
    assert await client.chat(
        api_key="synthetic-key",
        workspace_id="workspace-one",
        payload=payload,
        timeout_s=4,
    ) == {"ok": True}
    assert calls[0][0] == (
        "https://workspace-one.eu-central-1.maas.aliyuncs.com/apps/anthropic/v1/messages"
    )
    assert calls[0][1]["headers"] == {
        "x-api-key": "synthetic-key",
        "Content-Type": "application/json",
    }
    assert calls[0][1]["payload"] == payload
    assert calls[0][1]["timeout"] == 4
    http_handler = calls[0][1]["http_error_handler"]
    assert http_handler.__self__ is QwenMessagesSyncClient
    assert http_handler.__func__ is QwenMessagesSyncClient._handle_http_error.__func__
    with pytest.raises(LLMAPIAuthorizationError, match="403"):
        http_handler(FakeHTTPError(403, "HTTP 403"))

    assert [event async for event in client.stream(
        api_key="async-stream-key",
        workspace_id="workspace-two",
        payload=payload,
        timeout_s=7,
    )] == [SSEEvent(event="message_stop", data={"type": "message_stop"})]
    stream_url, stream_kwargs = stream_calls[0]
    assert stream_url.endswith("workspace-two.eu-central-1.maas.aliyuncs.com/apps/anthropic/v1/messages")
    assert stream_kwargs["headers"] == {
        "x-api-key": "async-stream-key",
        "Content-Type": "application/json",
    }
    assert stream_kwargs["payload"] == {**payload, "stream": True}


@pytest.mark.asyncio
@pytest.mark.unit
async def test_async_client_stream_copies_payload_and_forwards_native_handlers(monkeypatch):
    client = QwenMessagesAsyncClient()
    calls = []

    def fake_async_stream_request(url, **kwargs):
        calls.append((url, kwargs))

        async def events():
            yield SSEEvent(event="message_stop", data={"type": "message_stop"})

        return events()

    monkeypatch.setattr(
        async_client_module,
        "async_stream_request",
        fake_async_stream_request,
    )
    payload = {"model": "qwen3.8-max", "stream": False}
    events = client.stream(
        api_key="async-stream-key",
        workspace_id="workspace-one",
        payload=payload,
        timeout_s=5,
    )
    assert [event async for event in events] == [
        SSEEvent(event="message_stop", data={"type": "message_stop"})
    ]
    url, kwargs = calls[0]
    assert url.endswith("workspace-one.eu-central-1.maas.aliyuncs.com/apps/anthropic/v1/messages")
    assert kwargs["payload"] == {"model": "qwen3.8-max", "stream": True}
    assert payload == {"model": "qwen3.8-max", "stream": False}
    assert kwargs["headers"] == {
        "x-api-key": "async-stream-key",
        "Content-Type": "application/json",
    }
    assert kwargs["timeout"] == 5
    http_handler = kwargs["http_error_handler"]
    stream_handler = kwargs["stream_error_handler"]
    assert http_handler.__self__ is QwenMessagesSyncClient
    assert http_handler.__func__ is QwenMessagesSyncClient._handle_http_error.__func__
    assert stream_handler.__self__ is QwenMessagesSyncClient
    assert stream_handler.__func__ is QwenMessagesSyncClient._handle_stream_error.__func__
    with pytest.raises(LLMAPIAuthorizationError, match="async synthetic denied"):
        http_handler(FakeHTTPError(403, "async synthetic denied"))
    with pytest.raises(LLMAPIAuthorizationError, match="stream synthetic denied"):
        stream_handler(
            SSEEvent(
                event="error",
                data={
                    "type": "error",
                    "error": {
                        "type": "authentication_error",
                        "message": "stream synthetic denied",
                    },
                },
            )
        )


