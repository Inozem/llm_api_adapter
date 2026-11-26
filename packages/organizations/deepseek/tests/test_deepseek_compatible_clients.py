"""DeepSeek sync clients must use Core's compatible request engines."""

from __future__ import annotations

from typing import Any

import pytest
import requests

from llm_api_adapter.errors.llm_api_error import LLMAPIRateLimitError
from llm_api_adapter.llms.openai_compatible.sync_client import (
    OpenAICompatibleSyncClient,
)
from llm_api_adapter.llms.transports import JSONResponse, SSEEvent, TransportRequest
from llm_api_adapter_deepseek.clients.sync_client import (
    DEEPSEEK_RESPONSES_URL,
    DeepSeekResponsesSyncClient,
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

