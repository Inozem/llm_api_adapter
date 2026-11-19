"""Direct deterministic checks for Qwen's synchronous Messages client."""

from pathlib import Path
import inspect
import json
import sys
from types import SimpleNamespace

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[2]
for source in (PACKAGE_ROOT / "src", REPOSITORY_ROOT / "src", REPOSITORY_ROOT):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from llm_api_adapter.errors.llm_api_error import (
    LLMAPIAuthorizationError,
    LLMAPIClientError,
    LLMAPITimeoutError,
    LLMAPITokenLimitError,
    LLMAPIUsageLimitError,
)
from llm_api_adapter.llms.transports import JSONResponse, SSEEvent
from llm_api_adapter.llms.anthropic_compatible.sync_client import (
    _SyncMessagesExecutionMixin,
)
from llm_api_adapter_qwen.clients.sync_client import (
    QwenMessagesSyncClient,
    messages_url,
    validate_workspace_id,
)


class RecordingTransport:
    def __init__(self):
        self.json_requests = []
        self.sse_requests = []
        self.json_handler = None
        self.stream_handler = None

    def post_json(self, request, *, http_error_handler=None):
        self.json_requests.append(request)
        self.json_handler = http_error_handler
        return JSONResponse({"ok": True})

    def post_sse(
        self,
        request,
        *,
        http_error_handler=None,
        stream_error_handler=None,
    ):
        self.sse_requests.append(request)
        self.json_handler = http_error_handler
        self.stream_handler = stream_error_handler
        return iter([SSEEvent(event="message_stop", data={"type": "message_stop"})])


class FakeHTTPResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self.payload = payload

    def json(self):
        return self.payload


class FakeHTTPError(Exception):
    def __init__(self, status_code, payload):
        super().__init__(f"HTTP {status_code}")
        self.response = FakeHTTPResponse(status_code, payload)


@pytest.mark.unit
def test_sync_client_keeps_transport_only_constructor_and_request_local_routing():
    transport = RecordingTransport()
    client = QwenMessagesSyncClient(transport)
    constructor = inspect.signature(QwenMessagesSyncClient)
    assert list(constructor.parameters) == ["transport"]
    assert constructor.parameters["transport"].default is inspect.Parameter.empty
    with pytest.raises(TypeError):
        QwenMessagesSyncClient()
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
    payload = {"model": "qwen3.8-max", "stream": False}

    assert client.chat(
        api_key="synthetic-key",
        workspace_id="workspace-one",
        payload=payload,
        timeout_s=3,
    ) == {"ok": True}
    assert list(
        client.stream(
            api_key="stream-key",
            workspace_id="workspace-two",
            payload=payload,
            timeout_s=None,
        )
    )

    chat_request = transport.json_requests[0]
    stream_request = transport.sse_requests[0]
    assert chat_request.url == messages_url("workspace-one")
    assert stream_request.url == messages_url("workspace-two")
    assert chat_request.headers_dict() == {
        "x-api-key": "synthetic-key",
        "Content-Type": "application/json",
    }
    assert stream_request.headers_dict() == {
        "x-api-key": "stream-key",
        "Content-Type": "application/json",
    }
    assert chat_request.timeout == 3
    assert stream_request.timeout is None
    assert stream_request.payload == {**payload, "stream": True}
    assert payload == {"model": "qwen3.8-max", "stream": False}
    for handler in (transport.json_handler, transport.stream_handler):
        assert handler.__self__ is QwenMessagesSyncClient
    assert transport.json_handler.__func__ is QwenMessagesSyncClient._handle_http_error.__func__
    assert transport.stream_handler.__func__ is QwenMessagesSyncClient._handle_stream_error.__func__


@pytest.mark.unit
def test_sync_client_validates_workspace_and_maps_native_http_and_sse_errors():
    with pytest.raises(ValueError, match="workspace_id"):
        validate_workspace_id("Workspace/one")

    transport = RecordingTransport()
    client = QwenMessagesSyncClient(transport)
    list(
        client.stream(
            api_key="synthetic-key",
            workspace_id="workspace-one",
            payload={},
            timeout_s=None,
        )
    )
    with pytest.raises(LLMAPIAuthorizationError, match="synthetic denied"):
        transport.json_handler(FakeHTTPError(
            403,
            {"error": {"type": "permission_denied", "message": "synthetic denied"}},
        ))
    with pytest.raises(LLMAPIAuthorizationError, match="synthetic stream denied"):
        transport.stream_handler(
            SSEEvent(
                event="error",
                data={
                    "type": "error",
                    "error": {
                        "type": "authentication_error",
                        "message": "synthetic stream denied",
                    },
                },
            )
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("status_code", "payload", "expected_error", "detail"),
    [
        (408, {}, LLMAPITimeoutError, "HTTP 408"),
        (504, {}, LLMAPITimeoutError, "HTTP 504"),
        (400, {"code": "max_output_tokens_exceeded", "message": "token code"}, LLMAPITokenLimitError, "token code"),
        (400, {"code": "quota_exceeded", "message": "quota code"}, LLMAPIUsageLimitError, "quota code"),
        (400, {}, LLMAPIClientError, "HTTP 400"),
    ],
)
def test_sync_client_maps_status_only_native_errors_and_code_fallbacks(
    status_code,
    payload,
    expected_error,
    detail,
):
    transport = RecordingTransport()
    client = QwenMessagesSyncClient(transport)
    client.chat(
        api_key="synthetic-key",
        workspace_id="workspace-one",
        payload={},
        timeout_s=None,
    )

    with pytest.raises(expected_error, match=detail):
        transport.json_handler(FakeHTTPError(status_code, payload))


@pytest.mark.unit
def test_sync_client_dispatches_json_and_sse_through_shared_execution(monkeypatch):
    transport = RecordingTransport()
    client = QwenMessagesSyncClient(transport)
    dispatched = []
    execute_json = _SyncMessagesExecutionMixin._execute_prepared_request
    execute_stream = _SyncMessagesExecutionMixin._execute_prepared_stream

    def record_json(self, request):
        dispatched.append(("json", request))
        return execute_json(self, request)

    def record_stream(self, request):
        dispatched.append(("stream", request))
        return execute_stream(self, request)

    monkeypatch.setattr(
        _SyncMessagesExecutionMixin, "_execute_prepared_request", record_json
    )
    monkeypatch.setattr(
        _SyncMessagesExecutionMixin, "_execute_prepared_stream", record_stream
    )

    client.chat(
        api_key="synthetic-key",
        workspace_id="workspace-one",
        payload={"model": "qwen3.8-max"},
        timeout_s=None,
    )
    list(
        client.stream(
            api_key="synthetic-key",
            workspace_id="workspace-one",
            payload={"model": "qwen3.8-max"},
            timeout_s=None,
        )
    )

    assert [method for method, _ in dispatched] == ["json", "stream"]
    assert [request.url for _, request in dispatched] == [
        messages_url("workspace-one"),
        messages_url("workspace-one"),
    ]


@pytest.mark.unit
def test_sync_stream_binds_transport_when_stream_is_prepared():
    first_transport = RecordingTransport()
    second_transport = RecordingTransport()
    client = QwenMessagesSyncClient(first_transport)
    stream = client.stream(
        api_key="synthetic-key",
        workspace_id="workspace-one",
        payload={"model": "qwen3.8-max"},
        timeout_s=None,
    )

    client.transport = second_transport
    assert list(stream)
    assert len(first_transport.sse_requests) == 1
    assert second_transport.sse_requests == []


@pytest.mark.unit
@pytest.mark.parametrize("transport_name", ["requests", "httpx"])
def test_sync_client_uses_real_transport_contract_with_mocked_http(
    monkeypatch, transport_name
):
    import httpx

    from llm_api_adapter.llms import httpx_transport, requests_transport
    from llm_api_adapter.llms.transports import create_sync_transport

    sent = []
    response_json = {"id": "msg_1", "type": "message"}
    sse_body = (
        b'event: message_start\ndata: {"type":"message_start"}\n\n'
        b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
    )

    if transport_name == "requests":
        class Response:
            status_code = 200
            text = ""
            content = sse_body

            def __init__(self, stream=False):
                self.stream = stream
                self.closed = False

            def raise_for_status(self):
                return None

            def json(self):
                return response_json

            def iter_lines(self, **kwargs):
                return iter(sse_body.splitlines())

            def close(self):
                self.closed = True

        def post(url, **kwargs):
            sent.append((url, kwargs))
            return Response(stream=kwargs.get("stream", False))

        monkeypatch.setattr(requests_transport.requests, "post", post)
    else:
        def handler(request):
            sent.append(request)
            if json.loads(request.content).get("stream") is not True:
                return httpx.Response(
                    200,
                    json=response_json,
                    headers={"content-type": "application/json"},
                    request=request,
                )
            return httpx.Response(
                200,
                content=sse_body,
                headers={"content-type": "text/event-stream"},
                request=request,
            )

        real_client = httpx.Client
        monkeypatch.setattr(
            httpx_transport,
            "_require_httpx",
            lambda: SimpleNamespace(
                Client=lambda: real_client(transport=httpx.MockTransport(handler)),
                TimeoutException=httpx.TimeoutException,
                HTTPStatusError=httpx.HTTPStatusError,
                RequestError=httpx.RequestError,
            ),
        )

    client = QwenMessagesSyncClient(create_sync_transport(transport_name))
    assert client.chat(
        api_key="synthetic-key",
        workspace_id="workspace-one",
        payload={"model": "qwen3.8-max"},
        timeout_s=2,
    ) == response_json
    assert [event.event for event in client.stream(
        api_key="synthetic-key",
        workspace_id="workspace-one",
        payload={"model": "qwen3.8-max"},
        timeout_s=2,
    )] == ["message_start", "message_stop"]
    assert len(sent) == 2
    if transport_name == "requests":
        json_url, json_kwargs = sent[0]
        stream_url, stream_kwargs = sent[1]
        assert json_url == stream_url == messages_url("workspace-one")
        assert json_kwargs["headers"] == stream_kwargs["headers"] == {
            "x-api-key": "synthetic-key",
            "Content-Type": "application/json",
        }
        assert json_kwargs["json"] == {"model": "qwen3.8-max"}
        assert stream_kwargs["json"] == {"model": "qwen3.8-max", "stream": True}
        assert stream_kwargs["stream"] is True
    else:
        assert [request.url for request in sent] == [
            messages_url("workspace-one"),
            messages_url("workspace-one"),
        ]
        assert all(request.headers["x-api-key"] == "synthetic-key" for request in sent)
        assert all("anthropic-version" not in request.headers for request in sent)
        assert json.loads(sent[0].content) == {"model": "qwen3.8-max"}
        assert json.loads(sent[1].content) == {
            "model": "qwen3.8-max",
            "stream": True,
        }


@pytest.mark.unit
@pytest.mark.parametrize("transport_name", ["requests", "httpx"])
def test_sync_client_real_transport_preserves_native_http_error_mapping(
    monkeypatch, transport_name
):
    import httpx
    import requests

    from llm_api_adapter.llms import httpx_transport, requests_transport
    from llm_api_adapter.llms.transports import create_sync_transport

    payload = {"error": {"type": "permission_denied", "message": "synthetic denied"}}
    if transport_name == "requests":
        class Response:
            status_code = 403

            def json(self):
                return payload

            def raise_for_status(self):
                raise requests.HTTPError("forbidden", response=self)

            def close(self):
                pass

        monkeypatch.setattr(
            requests_transport.requests, "post", lambda *_args, **_kwargs: Response()
        )
    else:
        def handler(request):
            return httpx.Response(403, json=payload, request=request)

        real_client = httpx.Client
        monkeypatch.setattr(
            httpx_transport,
            "_require_httpx",
            lambda: SimpleNamespace(
                Client=lambda: real_client(transport=httpx.MockTransport(handler)),
                TimeoutException=httpx.TimeoutException,
                HTTPStatusError=httpx.HTTPStatusError,
                RequestError=httpx.RequestError,
            ),
        )

    client = QwenMessagesSyncClient(create_sync_transport(transport_name))
    with pytest.raises(LLMAPIAuthorizationError, match="synthetic denied"):
        client.chat(
            api_key="synthetic-key",
            workspace_id="workspace-one",
            payload={"model": "qwen3.8-max"},
            timeout_s=None,
        )


@pytest.mark.unit
def test_sync_stream_preserves_primary_error_and_closes_source():
    primary_error = RuntimeError("primary stream failure")
    close_error = RuntimeError("cleanup failure")

    class BrokenSource:
        def __iter__(self):
            return self

        def __next__(self):
            raise primary_error

        def close(self):
            self.closed = True
            raise close_error

    class SourceTransport(RecordingTransport):
        def post_sse(self, request, *, http_error_handler=None, stream_error_handler=None):
            self.sse_requests.append(request)
            self.stream_handler = stream_error_handler
            self.source = BrokenSource()
            return self.source

    transport = SourceTransport()
    stream = QwenMessagesSyncClient(transport).stream(
        api_key="synthetic-key",
        workspace_id="workspace-one",
        payload={},
        timeout_s=None,
    )

    with pytest.raises(RuntimeError) as raised:
        next(stream)
    assert raised.value is primary_error
    assert transport.source.closed


@pytest.mark.unit
def test_sync_stream_close_before_iteration_closes_eager_source():
    class ClosableSource:
        def __iter__(self):
            return self

        def __next__(self):
            raise StopIteration

        def close(self):
            self.closed = True

    class SourceTransport(RecordingTransport):
        def post_sse(self, request, *, http_error_handler=None, stream_error_handler=None):
            self.sse_requests.append(request)
            self.source = ClosableSource()
            return self.source

    transport = SourceTransport()
    stream = QwenMessagesSyncClient(transport).stream(
        api_key="synthetic-key",
        workspace_id="workspace-one",
        payload={},
        timeout_s=None,
    )
    stream.close()
    assert transport.source.closed
