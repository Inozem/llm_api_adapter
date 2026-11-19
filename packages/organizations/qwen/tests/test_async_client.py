"""Direct deterministic checks for Qwen's asynchronous Messages client."""

import asyncio
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[2]
for source in (PACKAGE_ROOT / "src", REPOSITORY_ROOT / "src", REPOSITORY_ROOT):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from llm_api_adapter.errors.llm_api_error import LLMAPIAuthorizationError
from llm_api_adapter.llms.transports import SSEEvent, TransportRequest
from llm_api_adapter.llms.anthropic_compatible.async_client import (
    AnthropicCompatibleAsyncClient,
    _AsyncMessagesExecutionMixin,
)
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
    assert issubclass(QwenMessagesAsyncClient, AnthropicCompatibleAsyncClient)
    assert "endpoint" not in repr(client) and "api_key" not in repr(client)
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


@pytest.mark.asyncio
@pytest.mark.unit
async def test_async_client_uses_shared_prepared_execution_and_routes_overlapping_calls(
    monkeypatch,
):
    client = QwenMessagesAsyncClient()
    json_calls = []
    stream_calls = []
    shared_json_calls = []
    shared_stream_calls = []
    sources = []

    async def fake_async_request(url, **kwargs):
        json_calls.append((url, kwargs))
        await asyncio.sleep(0)
        return {"workspace": url.split("//", 1)[1].split(".", 1)[0]}

    def fake_async_stream_request(url, **kwargs):
        stream_calls.append((url, kwargs))

        async def events():
            yield SSEEvent(event="message_stop", data={"url": url})

        source = events()
        sources.append(source)
        return source

    original_json = _AsyncMessagesExecutionMixin._execute_prepared_request
    original_stream = _AsyncMessagesExecutionMixin._execute_prepared_stream

    async def observed_json(self, request: TransportRequest):
        shared_json_calls.append(request)
        return await original_json(self, request)

    def observed_stream(self, request: TransportRequest):
        shared_stream_calls.append(request)
        return original_stream(self, request)

    monkeypatch.setattr(async_client_module, "async_request", fake_async_request)
    monkeypatch.setattr(
        async_client_module, "async_stream_request", fake_async_stream_request
    )
    monkeypatch.setattr(
        _AsyncMessagesExecutionMixin, "_execute_prepared_request", observed_json
    )
    monkeypatch.setattr(
        _AsyncMessagesExecutionMixin, "_execute_prepared_stream", observed_stream
    )

    first_stream = client.stream(
        api_key="first-key",
        workspace_id="first-workspace",
        payload={"model": "qwen3.8-max"},
        timeout_s=3,
    )
    second_stream = client.stream(
        api_key="second-key",
        workspace_id="second-workspace",
        payload={"model": "qwen3.8-max"},
        timeout_s=9,
    )
    assert stream_calls == []
    first_response, second_response = await asyncio.gather(
        client.chat(
            api_key="first-key",
            workspace_id="first-workspace",
            payload={"model": "qwen3.8-max"},
            timeout_s=3,
        ),
        client.chat(
            api_key="second-key",
            workspace_id="second-workspace",
            payload={"model": "qwen3.8-max"},
            timeout_s=9,
        ),
    )
    assert first_response == {"workspace": "first-workspace"}
    assert second_response == {"workspace": "second-workspace"}
    first_events, second_events = await asyncio.gather(
        _collect(first_stream), _collect(second_stream)
    )
    assert len(stream_calls) == 2

    assert first_events[0].data["url"].startswith(
        "https://first-workspace.eu-central-1."
    )
    assert second_events[0].data["url"].startswith(
        "https://second-workspace.eu-central-1."
    )
    assert len(shared_json_calls) == 2
    assert len(shared_stream_calls) == 2
    assert json_calls[0][1]["headers"]["x-api-key"] == "first-key"
    assert json_calls[1][1]["headers"]["x-api-key"] == "second-key"
    assert stream_calls[0][1]["headers"]["x-api-key"] == "first-key"
    assert stream_calls[1][1]["headers"]["x-api-key"] == "second-key"
    assert all(source.ag_frame is None for source in sources)
    assert not any(
        name in vars(client)
        for name in ("api_key", "endpoint", "workspace_id", "_async_client")
    )


async def _collect(source):
    return [event async for event in source]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_async_client_forwards_headers_through_native_header_hook(monkeypatch):
    client = QwenMessagesAsyncClient()
    header_calls = []
    request_calls = []

    def headers(api_key):
        header_calls.append(api_key)
        return {"x-api-key": api_key, "X-Test-Hook": "preserved"}

    async def fake_async_request(url, **kwargs):
        request_calls.append((url, kwargs))
        return {"ok": True}

    def fake_async_stream_request(url, **kwargs):
        request_calls.append((url, kwargs))

        async def events():
            yield SSEEvent(event="message_stop", data={"type": "message_stop"})

        return events()

    monkeypatch.setattr(client, "_headers", headers)
    monkeypatch.setattr(async_client_module, "async_request", fake_async_request)
    monkeypatch.setattr(
        async_client_module, "async_stream_request", fake_async_stream_request
    )

    await client.chat(
        api_key="chat-key",
        workspace_id="workspace-one",
        payload={},
        timeout_s=None,
    )
    events = client.stream(
        api_key="stream-key",
        workspace_id="workspace-two",
        payload={},
        timeout_s=None,
    )
    await _collect(events)

    assert header_calls == ["chat-key", "stream-key"]
    assert [call[1]["headers"]["X-Test-Hook"] for call in request_calls] == [
        "preserved",
        "preserved",
    ]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_async_client_close_before_iteration_does_not_create_source(monkeypatch):
    class Source:
        def __init__(self):
            self.closed = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        async def aclose(self):
            self.closed = True

    source = Source()
    calls = []
    monkeypatch.setattr(
        async_client_module,
        "async_stream_request",
        lambda *_args, **_kwargs: calls.append(source) or source,
    )
    stream = QwenMessagesAsyncClient().stream(
        api_key="synthetic-key",
        workspace_id="workspace-one",
        payload={},
        timeout_s=None,
    )
    assert calls == []
    await stream.aclose()
    assert calls == []
    assert not source.closed


@pytest.mark.asyncio
@pytest.mark.unit
async def test_async_client_shared_stream_cleanup_preserves_primary_error(monkeypatch):
    primary_error = RuntimeError("primary stream failure")
    cleanup_error = RuntimeError("cleanup failure")

    class BrokenSource:
        def __init__(self):
            self.closed = False
            self.close_count = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise primary_error

        async def aclose(self):
            self.closed = True
            self.close_count += 1
            raise cleanup_error

    source = BrokenSource()
    monkeypatch.setattr(
        async_client_module, "async_stream_request", lambda *_args, **_kwargs: source
    )
    stream = QwenMessagesAsyncClient().stream(
        api_key="synthetic-key",
        workspace_id="workspace-one",
        payload={},
        timeout_s=None,
    )
    with pytest.raises(RuntimeError) as raised:
        await anext(stream)
    assert raised.value is primary_error
    assert source.closed
    assert source.close_count == 1


@pytest.mark.asyncio
@pytest.mark.unit
async def test_async_client_shared_stream_cleanup_on_early_close_and_cancellation(
    monkeypatch,
):
    entered_events = []
    sources = []

    def fake_async_stream_request(*_args, **_kwargs):
        class Source:
            def __init__(self):
                self.closed = False
                self.read_count = 0
                self.entered = asyncio.Event()
                entered_events.append(self.entered)
                self.block_first = len(sources) == 1

            def __aiter__(self):
                return self

            async def __anext__(self):
                self.read_count += 1
                if self.block_first and self.read_count == 1:
                    self.entered.set()
                    await asyncio.Future()
                if self.read_count == 1:
                    return SSEEvent(
                        event="content_block_delta", data={"text": "x"}
                    )
                await asyncio.Future()

            async def aclose(self):
                self.closed = True

        source = Source()
        sources.append(source)
        return source

    monkeypatch.setattr(
        async_client_module, "async_stream_request", fake_async_stream_request
    )
    client = QwenMessagesAsyncClient()
    early = client.stream(
        api_key="early-key",
        workspace_id="early-workspace",
        payload={},
        timeout_s=None,
    )
    assert (await anext(early)).data == {"text": "x"}
    await early.aclose()
    assert sources[0].closed

    cancelled = client.stream(
        api_key="cancel-key",
        workspace_id="cancel-workspace",
        payload={},
        timeout_s=None,
    )
    task = asyncio.create_task(anext(cancelled))
    await asyncio.sleep(0)
    await entered_events[1].wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert sources[1].closed


