"""DeepSeek sync and async clients must use Core's compatible request engines."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
import requests

from llm_api_adapter.errors.llm_api_error import (
    LLMAPIClientError,
    LLMAPIRateLimitError,
)
from llm_api_adapter.llms.openai_compatible.sync_client import (
    OpenAICompatibleSyncClient,
)
from llm_api_adapter.llms.openai_compatible.async_client import (
    OpenAICompatibleAsyncClient,
)
from llm_api_adapter.llms.transports import JSONResponse, SSEEvent, TransportRequest
from llm_api_adapter_deepseek.clients.sync_client import (
    DEEPSEEK_RESPONSES_URL,
    DeepSeekResponsesSyncClient,
)
from llm_api_adapter_deepseek.clients.async_client import (
    DeepSeekResponsesAsyncClient,
)


class CaptureTransport:
    def __init__(self) -> None:
        self.json_calls: list[tuple[TransportRequest, Any]] = []
        self.sse_calls: list[tuple[TransportRequest, Any, Any]] = []
        self.http_calls: list[str] = []
        self.source: Any = None
        self.events = [
            SSEEvent(
                event="response.completed",
                data={"type": "response.completed"},
            ),
        ]

    def post_json(self, request: TransportRequest, *, http_error_handler=None):
        self.json_calls.append((request, http_error_handler))
        return JSONResponse({"id": "resp_test"})

    def post_sse(
        self,
        request: TransportRequest,
        *,
        http_error_handler=None,
        stream_error_handler=None,
    ):
        self.sse_calls.append(
            (request, http_error_handler, stream_error_handler),
        )
        self.source = _LazyEvents(self) if self.source is None else self.source
        return self.source


class _LazyEvents:
    def __init__(self, transport: CaptureTransport) -> None:
        self.transport = transport
        self._events = iter(transport.events)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        if not self.transport.http_calls:
            self.transport.http_calls.append("opened")
        return next(self._events)

    def close(self) -> None:
        self.closed = True
        self.transport.http_calls.append("closed")


@pytest.mark.unit
def test_sync_client_directly_inherits_and_runs_core_json_engine(monkeypatch):
    assert DeepSeekResponsesSyncClient.__bases__ == (OpenAICompatibleSyncClient,)
    calls = []
    inherited_send = OpenAICompatibleSyncClient._send_request

    def trace_send(self, url, payload, timeout=None):
        calls.append((self, url, payload, timeout))
        return inherited_send(self, url, payload, timeout)

    monkeypatch.setattr(OpenAICompatibleSyncClient, "_send_request", trace_send)
    transport = CaptureTransport()
    client = DeepSeekResponsesSyncClient(
        "test-key",
        endpoint="https://gateway.example/v1/responses",
    )
    client._sync_transport = transport

    result = client.create(model="deepseek-flash", input=[], timeout=2.5)

    request, handler = transport.json_calls[0]
    assert result == {"id": "resp_test"}
    assert calls == [
        (client, "https://gateway.example/v1/responses", {
            "model": "deepseek-flash", "input": [],
        }, 2.5),
    ]
    assert request.url == "https://gateway.example/v1/responses"
    assert request.payload == {"model": "deepseek-flash", "input": []}
    assert request.headers_dict() == {
        "Authorization": "Bearer test-key",
        "Content-Type": "application/json",
    }
    assert request.timeout == 2.5
    assert handler.__self__ is DeepSeekResponsesSyncClient


@pytest.mark.unit
def test_sync_client_runs_core_sse_engine_and_keeps_native_open_timing(monkeypatch):
    calls = []
    inherited_stream = OpenAICompatibleSyncClient._stream_request

    def trace_stream(self, url, payload, timeout=None):
        def run():
            calls.append((self, url, payload, timeout))
            yield from inherited_stream(self, url, payload, timeout)

        return run()

    monkeypatch.setattr(OpenAICompatibleSyncClient, "_stream_request", trace_stream)
    transport = CaptureTransport()
    client = DeepSeekResponsesSyncClient("test-key")
    client._sync_transport = transport

    events = client.stream(model="deepseek-flash", input=[])

    assert len(transport.sse_calls) == 1  # raw helper invocation stays immediate
    assert transport.http_calls == []  # the actual HTTP opening stays lazy
    assert calls == []  # the inherited generator itself is still lazy
    request, http_handler, stream_handler = transport.sse_calls[0]
    assert request.url == DEEPSEEK_RESPONSES_URL
    assert request.payload == {
        "model": "deepseek-flash", "input": [], "stream": True,
    }
    assert request.headers_dict() == {
        "Authorization": "Bearer test-key",
        "Content-Type": "application/json",
    }
    assert http_handler.__self__ is DeepSeekResponsesSyncClient
    assert stream_handler.__self__ is DeepSeekResponsesSyncClient

    received = list(events)
    assert received == transport.events
    assert len(calls) == 1
    owner, url, payload, timeout = calls[0]
    assert isinstance(owner, DeepSeekResponsesSyncClient)
    assert owner is not client
    assert (url, payload, timeout) == (
        DEEPSEEK_RESPONSES_URL,
        {"model": "deepseek-flash", "input": [], "stream": True},
        None,
    )
    assert transport.http_calls == ["opened", "closed"]


@pytest.mark.unit
def test_sync_stream_yields_native_response_failed_and_accepts_plain_iterators():
    failure_event = SSEEvent(
        event="response.failed",
        data={"type": "response.failed", "response": {"status": "failed"}},
    )
    transport = CaptureTransport()
    transport.events = [failure_event]
    client = DeepSeekResponsesSyncClient("test-key")
    client._sync_transport = transport

    assert list(client.stream(model="deepseek-flash", input=[])) == [failure_event]

    transport = CaptureTransport()
    transport.source = iter(transport.events)  # ordinary iterators need no close()
    client._sync_transport = transport
    assert list(client.stream(model="deepseek-flash", input=[])) == transport.events


@pytest.mark.unit
def test_sync_stream_close_before_and_after_consumption_closes_once():
    transport = CaptureTransport()
    client = DeepSeekResponsesSyncClient("test-key")
    client._sync_transport = transport
    stream = client.stream(model="deepseek-flash", input=[])
    source = transport.source

    stream.close()
    stream.close()
    assert source.closed is True
    assert transport.http_calls == ["closed"]

    transport = CaptureTransport()
    transport.events = [
        SSEEvent(event="response.output_text.delta", data={"delta": "one"}),
        SSEEvent(event="response.output_text.delta", data={"delta": "two"}),
    ]
    client._sync_transport = transport
    stream = client.stream(model="deepseek-flash", input=[])
    source = transport.source
    assert next(stream) == transport.events[0]
    stream.close()
    stream.close()
    assert source.closed is True
    assert transport.http_calls == ["opened", "closed"]


@pytest.mark.unit
def test_sync_stream_iteration_error_remains_primary_when_close_fails():
    failure = RuntimeError("iteration failed")

    class FailingSource:
        def __iter__(self):
            return self

        def __next__(self):
            raise failure

        def close(self):
            raise RuntimeError("cleanup failed")

    transport = CaptureTransport()
    transport.source = FailingSource()
    client = DeepSeekResponsesSyncClient("test-key")
    client._sync_transport = transport

    with pytest.raises(RuntimeError) as caught:
        list(client.stream(model="deepseek-flash", input=[]))
    assert caught.value is failure


@pytest.mark.unit
def test_sync_stream_explicit_close_propagates_cleanup_error_once():
    failure = RuntimeError("explicit cleanup failed")

    class CloseFailingSource:
        def __init__(self):
            self._events = iter([SSEEvent(event="response.output_text.delta")])
            self.close_calls = 0

        def __iter__(self):
            return self

        def __next__(self):
            return next(self._events)

        def close(self):
            self.close_calls += 1
            raise failure

    transport = CaptureTransport()
    transport.source = CloseFailingSource()
    client = DeepSeekResponsesSyncClient("test-key")
    client._sync_transport = transport
    stream = client.stream(model="deepseek-flash", input=[])
    source = transport.source
    assert next(stream).event == "response.output_text.delta"

    with pytest.raises(RuntimeError) as caught:
        stream.close()
    assert caught.value is failure
    assert source.close_calls == 1


@pytest.mark.unit
def test_sync_stream_snapshots_native_headers_once_and_preserves_repr_match_args():
    transport = CaptureTransport()
    client = DeepSeekResponsesSyncClient("private-test-key")
    client._sync_transport = transport
    header_calls = []

    def headers():
        header_calls.append(True)
        return {"Authorization": "Bearer snapshot", "X-Native": "yes"}

    client._headers = headers
    stream = client.stream(model="deepseek-flash", input=[])
    assert len(header_calls) == 1
    assert transport.sse_calls[0][0].headers_dict() == {
        "Authorization": "Bearer snapshot", "X-Native": "yes",
    }
    list(stream)
    assert len(header_calls) == 1

    assert repr(client) == object.__repr__(client)
    assert DeepSeekResponsesSyncClient.__match_args__ == (
        "api_key", "transport", "endpoint",
    )
    matched = False
    match client:
        case DeepSeekResponsesSyncClient("private-test-key", "requests", _):
            matched = True
    assert matched


@pytest.mark.unit
def test_requests_http_error_uses_native_mapper_after_inheritance(monkeypatch):
    class Response:
        status_code = 429

        def raise_for_status(self):
            error = requests.exceptions.HTTPError("rate limited")
            error.response = self
            raise error

        def json(self):
            return {"error": {"type": "rate_limit_error", "message": "slow"}}

        def close(self):
            return None

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: Response())
    client = DeepSeekResponsesSyncClient("test-key")

    with pytest.raises(LLMAPIRateLimitError, match="slow"):
        client.create(model="deepseek-flash", input=[])


@pytest.mark.unit
def test_httpx_http_error_uses_native_mapper_after_inheritance(monkeypatch):
    httpx = pytest.importorskip("httpx")

    class Client:
        def post(self, url, *, headers, json, timeout):
            del headers, json, timeout
            request = httpx.Request("POST", url)
            response = httpx.Response(
                429,
                request=request,
                json={"error": {"type": "rate_limit_error", "message": "slow"}},
            )
            raise httpx.HTTPStatusError(
                "rate limited",
                request=request,
                response=response,
            )

        def close(self):
            return None

    monkeypatch.setattr(httpx, "Client", Client)
    client = DeepSeekResponsesSyncClient("test-key", transport="httpx")

    with pytest.raises(LLMAPIRateLimitError, match="slow"):
        client.create(model="deepseek-flash", input=[])


@pytest.mark.unit
def test_async_client_directly_inherits_and_runs_core_json_engine(monkeypatch):
    assert DeepSeekResponsesAsyncClient.__bases__ == (OpenAICompatibleAsyncClient,)
    calls = []
    dispatches = []

    async def request(url, **kwargs):
        dispatches.append((url, kwargs))
        return {"id": "resp_async"}

    monkeypatch.setattr(
        "llm_api_adapter_deepseek.clients.async_client.async_request",
        request,
    )
    inherited_send = OpenAICompatibleAsyncClient._send_request

    async def trace_send(self, url, payload, timeout=None):
        calls.append((self, url, payload, timeout))
        return await inherited_send(self, url, payload, timeout)

    monkeypatch.setattr(OpenAICompatibleAsyncClient, "_send_request", trace_send)
    endpoint = "https://gateway.example/v1/responses"
    client = DeepSeekResponsesAsyncClient("test-key", endpoint=endpoint)
    assert not hasattr(client, "_sync_transport")

    result = asyncio.run(
        client.create(model="deepseek-flash", input=[], timeout=2.5),
    )

    assert result == {"id": "resp_async"}
    assert calls == [
        (client, endpoint, {"model": "deepseek-flash", "input": []}, 2.5),
    ]
    assert dispatches[0][0] == endpoint
    assert dispatches[0][1]["payload"] == {
        "model": "deepseek-flash", "input": [],
    }
    assert dispatches[0][1]["headers"] == {
        "Authorization": "Bearer test-key",
        "Content-Type": "application/json",
    }
    assert dispatches[0][1]["http_error_handler"] is client._handle_http_error


@pytest.mark.unit
def test_async_client_runs_core_sse_engine_with_immediate_helper_and_snapshot(
    monkeypatch,
):
    dispatches = []
    engine_calls = []
    closed = []

    class Source:
        def __init__(self):
            self._events = iter([
                SSEEvent(event="response.output_text.delta", data={"delta": "hello"}),
            ])

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self._events)
            except StopIteration:
                raise StopAsyncIteration

        async def aclose(self):
            closed.append("closed")

    source = Source()

    def stream_request(url, **kwargs):
        dispatches.append((url, kwargs))
        return source

    monkeypatch.setattr(
        "llm_api_adapter_deepseek.clients.async_client.async_stream_request",
        stream_request,
    )
    inherited_stream = OpenAICompatibleAsyncClient._stream_request

    async def trace_stream(self, url, payload, timeout=None):
        engine_calls.append((self, url, payload, timeout))
        async for event in inherited_stream(self, url, payload, timeout):
            yield event

    monkeypatch.setattr(
        OpenAICompatibleAsyncClient,
        "_stream_request",
        trace_stream,
    )
    endpoint = "https://gateway.example/v1/prefix/responses"
    client = DeepSeekResponsesAsyncClient("test-key", endpoint=endpoint)
    http_error_handler = lambda error: None
    stream_error_handler = lambda event: None
    client._handle_http_error = http_error_handler
    client._handle_stream_error = stream_error_handler

    events = client.stream(model="deepseek-flash", input=[], timeout=3)
    client.endpoint = "https://changed.example/responses"
    client.api_key = "changed-key"
    client._handle_http_error = lambda error: pytest.fail("handler was not captured")
    client._handle_stream_error = lambda event: pytest.fail("handler was not captured")
    assert len(dispatches) == 1
    assert dispatches[0][0] == endpoint
    assert dispatches[0][1]["headers"]["Authorization"] == "Bearer test-key"
    assert dispatches[0][1]["http_error_handler"] is http_error_handler
    assert dispatches[0][1]["stream_error_handler"] is stream_error_handler
    assert engine_calls == []

    async def consume_one_and_close():
        event = await events.__anext__()
        await events.aclose()
        await events.aclose()
        return event

    received = asyncio.run(consume_one_and_close())
    assert received.data == {"delta": "hello"}
    assert len(engine_calls) == 1
    view, url, payload, timeout = engine_calls[0]
    assert isinstance(view, DeepSeekResponsesAsyncClient)
    assert isinstance(view, OpenAICompatibleAsyncClient)
    assert view is not client
    assert repr(view) == object.__repr__(view)
    assert (url, payload, timeout) == (
        endpoint,
        {"model": "deepseek-flash", "input": [], "stream": True},
        3,
    )
    assert dispatches[0][1]["payload"] == payload
    assert dispatches[0][1]["stream_error_handler"] is stream_error_handler
    assert closed == ["closed"]


async def _collect_async(events):
    return [event async for event in events]


@pytest.mark.unit
def test_async_stream_close_before_consumption_and_non_object_json(monkeypatch):
    closed = []

    class Source:
        def __aiter__(self):
            return self

        async def __anext__(self):
            await asyncio.sleep(0)
            return SSEEvent(event="response.output_text.delta")

        async def aclose(self):
            closed.append(True)

    monkeypatch.setattr(
        "llm_api_adapter_deepseek.clients.async_client.async_stream_request",
        lambda *args, **kwargs: Source(),
    )
    events = DeepSeekResponsesAsyncClient("test-key").stream(
        model="deepseek-flash",
        input=[],
    )

    async def close_unconsumed():
        await events.aclose()
        await events.aclose()

    asyncio.run(close_unconsumed())
    assert closed == [True]

    async def request(url, **kwargs):
        return []

    monkeypatch.setattr(
        "llm_api_adapter_deepseek.clients.async_client.async_request",
        request,
    )
    with pytest.raises(LLMAPIClientError, match="non-object"):
        asyncio.run(DeepSeekResponsesAsyncClient("test-key").create(
            model="deepseek-flash",
        ))


@pytest.mark.unit
def test_async_stream_supports_plain_source_and_preserves_response_failed(monkeypatch):
    failure_event = SSEEvent(
        event="response.failed",
        data={"type": "response.failed", "response": {"status": "failed"}},
    )

    class PlainSource:
        def __init__(self):
            self._events = iter([failure_event])

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self._events)
            except StopIteration:
                raise StopAsyncIteration

    monkeypatch.setattr(
        "llm_api_adapter_deepseek.clients.async_client.async_stream_request",
        lambda *args, **kwargs: PlainSource(),
    )
    events = DeepSeekResponsesAsyncClient("test-key").stream(
        model="deepseek-flash",
        input=[],
    )
    assert asyncio.run(_collect_async(events)) == [failure_event]


@pytest.mark.unit
def test_async_stream_iteration_failure_remains_primary_when_close_fails(
    monkeypatch,
):
    iteration_failure = RuntimeError("iteration failed")

    class FailingSource:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise iteration_failure

        async def aclose(self):
            raise RuntimeError("cleanup failed")

    monkeypatch.setattr(
        "llm_api_adapter_deepseek.clients.async_client.async_stream_request",
        lambda *args, **kwargs: FailingSource(),
    )
    events = DeepSeekResponsesAsyncClient("test-key").stream(
        model="deepseek-flash",
        input=[],
    )

    with pytest.raises(RuntimeError, match="iteration failed") as caught:
        asyncio.run(_collect_async(events))
    assert caught.value is iteration_failure


@pytest.mark.unit
def test_async_stream_cancellation_remains_primary_when_close_fails(monkeypatch):
    cancellation = asyncio.CancelledError("request cancelled")

    class CancelledSource:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise cancellation

        async def aclose(self):
            raise RuntimeError("cleanup failed")

    monkeypatch.setattr(
        "llm_api_adapter_deepseek.clients.async_client.async_stream_request",
        lambda *args, **kwargs: CancelledSource(),
    )
    events = DeepSeekResponsesAsyncClient("test-key").stream(
        model="deepseek-flash",
        input=[],
    )

    async def consume_cancelled_stream():
        with pytest.raises(asyncio.CancelledError) as caught:
            await _collect_async(events)
        assert caught.value is cancellation

    asyncio.run(consume_cancelled_stream())


@pytest.mark.unit
def test_async_stream_explicit_close_propagates_source_close_failure_once(
    monkeypatch,
):
    close_failure = RuntimeError("explicit cleanup failed")

    class CloseFailingSource:
        def __init__(self):
            self.close_calls = 0
            self._events = iter([
                SSEEvent(event="response.output_text.delta", data={"delta": "one"}),
            ])

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self._events)
            except StopIteration:
                raise StopAsyncIteration

        async def aclose(self):
            self.close_calls += 1
            raise close_failure

    source = CloseFailingSource()
    monkeypatch.setattr(
        "llm_api_adapter_deepseek.clients.async_client.async_stream_request",
        lambda *args, **kwargs: source,
    )
    events = DeepSeekResponsesAsyncClient("test-key").stream(
        model="deepseek-flash",
        input=[],
    )

    async def consume_and_close():
        await events.__anext__()
        with pytest.raises(RuntimeError, match="explicit cleanup failed") as caught:
            await events.aclose()
        assert caught.value is close_failure
        await events.aclose()

    asyncio.run(consume_and_close())
    assert source.close_calls == 1


@pytest.mark.unit
def test_async_stream_close_while_pull_is_running_does_not_close_source(
    monkeypatch,
):
    class WaitingSource:
        def __init__(self):
            self.entered = asyncio.Event()
            self.release = asyncio.Event()
            self.close_calls = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            self.entered.set()
            await self.release.wait()
            return SSEEvent(event="response.output_text.delta", data={"delta": "ok"})

        async def aclose(self):
            self.close_calls += 1

    source = WaitingSource()
    monkeypatch.setattr(
        "llm_api_adapter_deepseek.clients.async_client.async_stream_request",
        lambda *args, **kwargs: source,
    )
    events = DeepSeekResponsesAsyncClient("test-key").stream(
        model="deepseek-flash",
        input=[],
    )

    async def close_during_pull():
        pending_pull = asyncio.create_task(events.__anext__())
        await source.entered.wait()
        with pytest.raises(RuntimeError, match="already running"):
            await events.aclose()
        assert source.close_calls == 0
        source.release.set()
        assert (await pending_pull).data == {"delta": "ok"}
        await events.aclose()
        assert source.close_calls == 1

    asyncio.run(close_during_pull())


@pytest.mark.unit
def test_async_json_dispatch_uses_native_error_mapping(monkeypatch):
    class Response:
        status_code = 429

        def json(self):
            return {"error": {"type": "rate_limit_error", "message": "slow"}}

    class Error(Exception):
        response = Response()

    async def request(url, **kwargs):
        del url
        kwargs["http_error_handler"](Error("rate limited"))

    monkeypatch.setattr(
        "llm_api_adapter_deepseek.clients.async_client.async_request",
        request,
    )
    with pytest.raises(LLMAPIRateLimitError, match="slow"):
        asyncio.run(DeepSeekResponsesAsyncClient("test-key").create(
            model="deepseek-flash",
        ))
