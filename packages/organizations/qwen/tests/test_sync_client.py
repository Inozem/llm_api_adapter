"""Direct deterministic checks for Qwen's synchronous Messages client."""

from pathlib import Path
import inspect
import sys

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
