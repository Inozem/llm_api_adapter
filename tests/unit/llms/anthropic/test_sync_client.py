from unittest.mock import patch, Mock
from dataclasses import fields

import pytest
import requests

from src.llm_api_adapter.errors.llm_api_error import (
    LLMAPIAuthorizationError,
    LLMAPIRateLimitError,
    LLMAPIClientError,
    LLMAPIServerError,
    LLMAPITimeoutError,
)
from src.llm_api_adapter.llms.anthropic.sync_client import ClaudeSyncClient
from src.llm_api_adapter.adapters.anthropic_adapter import ClaudeSyncClient as ShimSyncClient
from src.llm_api_adapter.llms.streaming import SSEEvent


class StreamingResponse:
    def __init__(self, lines, status_code=200, error_payload=None):
        self.lines = lines
        self.status_code = status_code
        self.error_payload = error_payload or {}
        self.close = Mock()

    def iter_lines(self, decode_unicode=True):
        return iter(self.lines)

    def json(self):
        return self.error_payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(response=self)

@pytest.fixture
def client():
    return ClaudeSyncClient(api_key="test_api_key")

@pytest.fixture
def mock_post_success():
    mock_response = Mock()
    mock_response.json.return_value = {
        "completion": "Hello"
    }
    mock_response.raise_for_status = Mock()
    with patch(
        "src.llm_api_adapter.llms.anthropic.sync_client.requests.post",
        return_value=mock_response
    ) as mock_post:
        yield mock_post, mock_response

@pytest.mark.unit
def test_chat_completion_success(client, mock_post_success):
    mock_post, _ = mock_post_success
    result = client.chat_completion(
        "claude-sonnet-4-5", messages=[{"role": "user", "content": "Hi"}]
    )
    assert isinstance(result, dict)
    assert "completion" in result
    assert result["completion"] == "Hello"
    mock_post.assert_called_once()
    headers = mock_post.call_args[1]["headers"]
    assert headers["x-api-key"] == "test_api_key"

@pytest.mark.parametrize("exception,expected_exception", [
    (requests.exceptions.Timeout("timeout"), LLMAPITimeoutError),
    (requests.exceptions.RequestException("generic error"), LLMAPIClientError),
])
@pytest.mark.unit
@patch("src.llm_api_adapter.llms.anthropic.sync_client.requests.post")
def test_send_request_exceptions(
    mock_post, client, exception, expected_exception
):
    mock_post.side_effect = exception
    with pytest.raises(expected_exception):
        client._send_request("http://example.com", {})

@pytest.mark.parametrize("status_code, error_type, expected_exception", [
    (401, "invalid_api_key", LLMAPIAuthorizationError),
    (429, "rate_limit_exceeded", LLMAPIRateLimitError),
    (400, "bad_request", LLMAPIClientError),
    (500, "server_error", LLMAPIServerError),
])
@pytest.mark.unit
@patch("src.llm_api_adapter.llms.openai.sync_client.requests.post")
def test_send_request_http_errors(
    mock_post, client, status_code, error_type, expected_exception
):
    mock_response = Mock()
    mock_response.status_code = status_code
    mock_response.json.return_value = {
        "error": {"type": error_type, "message": "Error message"}
    }
    http_err = requests.exceptions.HTTPError(response=mock_response)
    mock_post.side_effect = http_err
    with pytest.raises(expected_exception):
        client._send_request("http://example.com", {})

@pytest.mark.unit
@patch("src.llm_api_adapter.llms.anthropic.sync_client.requests.post")
def test_send_request_fallback_error_parsing(mock_post, client):
    mock_response = Mock()
    mock_response.status_code = 400
    mock_response.json.side_effect = ValueError("Invalid JSON")
    http_err = requests.exceptions.HTTPError(response=mock_response)
    mock_post.side_effect = http_err
    with pytest.raises(LLMAPIClientError):
        client._send_request("http://example.com", {})


# ---------------------------
# _prepare_chat_payload_for_model
# ---------------------------

@pytest.mark.unit
def test_prepare_payload_adaptive_thinking_strips_top_p_and_sets_thinking(client):
    kwargs = {"messages": [], "top_p": 1.0, "is_adaptive_thinking": True, "effort": "high"}
    payload = client._prepare_chat_payload_for_model("claude-opus-4-8", kwargs)
    assert "top_p" not in payload
    assert "is_adaptive_thinking" not in payload
    assert payload["thinking"] == {"type": "adaptive"}
    assert payload["output_config"]["effort"] == "high"


@pytest.mark.unit
def test_prepare_payload_adaptive_thinking_no_effort_strips_top_p_only(client):
    kwargs = {"messages": [], "top_p": 1.0, "is_adaptive_thinking": True}
    payload = client._prepare_chat_payload_for_model("claude-opus-4-8", kwargs)
    assert "top_p" not in payload
    assert "thinking" not in payload
    assert "is_adaptive_thinking" not in payload


@pytest.mark.unit
def test_prepare_payload_legacy_sets_budget_tokens_thinking(client):
    kwargs = {"messages": [], "budget_tokens": 4096, "is_adaptive_thinking": False}
    payload = client._prepare_chat_payload_for_model("claude-opus-4-5", kwargs)
    assert payload["thinking"] == {"type": "enabled", "budget_tokens": 4096}
    assert "is_adaptive_thinking" not in payload


@pytest.mark.unit
def test_prepare_payload_capture_reasoning_requests_summarized_thinking(client):
    kwargs = {
        "messages": [],
        "budget_tokens": 4096,
        "is_adaptive_thinking": False,
        "capture_reasoning": True,
    }
    payload = client._prepare_chat_payload_for_model("claude-opus-4-5", kwargs)
    assert payload["thinking"] == {
        "type": "enabled",
        "budget_tokens": 4096,
        "display": "summarized",
    }
    assert "capture_reasoning" not in payload


@pytest.mark.unit
def test_prepare_payload_is_adaptive_thinking_never_in_result(client):
    for flag in (True, False):
        kwargs = {"messages": [], "is_adaptive_thinking": flag}
        payload = client._prepare_chat_payload_for_model("claude-opus-4-8", kwargs)
        assert "is_adaptive_thinking" not in payload


@pytest.mark.unit
@patch("src.llm_api_adapter.llms.streaming.requests.post")
def test_stream_yields_raw_messages_events_and_ignores_ping(mock_post, client):
    response = StreamingResponse([
        "event: message_start",
        'data: {"type":"message_start","message":{"id":"msg_123","content":[]}}',
        "",
        "event: ping",
        'data: {"type":"ping"}',
        "",
        "event: content_block_delta",
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}',
        "",
        "event: message_stop",
        'data: {"type":"message_stop"}',
        "",
    ])
    mock_post.return_value = response

    events = list(client.stream(
        "claude-opus-4-8",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=64,
    ))

    assert events == [
        SSEEvent(
            event="message_start",
            data={"type": "message_start", "message": {"id": "msg_123", "content": []}},
        ),
        SSEEvent(
            event="content_block_delta",
            data={
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hello"},
            },
        ),
        SSEEvent(event="message_stop", data={"type": "message_stop"}),
    ]
    assert mock_post.call_args.args[0] == "https://api.anthropic.com/v1/messages"
    assert mock_post.call_args.kwargs["json"] == {
        "model": "claude-opus-4-8",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 64,
        "stream": True,
    }
    assert mock_post.call_args.kwargs["stream"] is True
    response.close.assert_called_once()


@pytest.mark.unit
@patch("src.llm_api_adapter.llms.streaming.requests.post")
def test_stream_reuses_adaptive_thinking_payload_preparation(mock_post, client):
    response = StreamingResponse([])
    mock_post.return_value = response

    assert list(client.stream(
        "claude-opus-4-8",
        messages=[],
        top_p=1.0,
        is_adaptive_thinking=True,
        effort="high",
    )) == []

    assert mock_post.call_args.kwargs["json"] == {
        "model": "claude-opus-4-8",
        "messages": [],
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "high"},
        "stream": True,
    }


@pytest.mark.unit
@patch("src.llm_api_adapter.llms.streaming.requests.post")
def test_stream_preserves_input_json_deltas_for_later_tool_assembly(mock_post, client):
    response = StreamingResponse([
        "event: content_block_delta",
        'data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\\\"city\\\": \\\"Tel"}}',
        "",
        "event: content_block_delta",
        'data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":" Aviv\\\"}"}}',
        "",
        "event: content_block_stop",
        'data: {"type":"content_block_stop","index":1}',
        "",
    ])
    mock_post.return_value = response

    events = list(client.stream("claude-opus-4-8", messages=[]))

    assert [event.data["delta"]["partial_json"] for event in events[:2]] == [
        '{"city": "Tel',
        ' Aviv"}',
    ]
    assert events[-1] == SSEEvent(
        event="content_block_stop",
        data={"type": "content_block_stop", "index": 1},
    )


@pytest.mark.unit
@patch("src.llm_api_adapter.llms.streaming.requests.post")
def test_stream_closes_response_when_consumer_stops_early(mock_post, client):
    response = StreamingResponse([
        "event: content_block_delta",
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}',
        "",
        "event: content_block_delta",
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}',
        "",
    ])
    mock_post.return_value = response

    events = client.stream("claude-opus-4-8", messages=[])
    next(events)
    events.close()

    response.close.assert_called_once()


@pytest.mark.unit
@patch("src.llm_api_adapter.llms.streaming.requests.post")
def test_stream_maps_http_errors_through_anthropic_error_handler(mock_post, client):
    response = StreamingResponse(
        [],
        status_code=429,
        error_payload={"error": {"type": "rate_limit_error", "message": "Slow down"}},
    )
    mock_post.return_value = response

    with pytest.raises(LLMAPIRateLimitError, match="Slow down"):
        list(client.stream("claude-opus-4-8", messages=[]))

    response.close.assert_called_once()


@pytest.mark.unit
@patch("src.llm_api_adapter.llms.streaming.requests.post")
def test_stream_maps_in_stream_error_through_anthropic_error_handler(mock_post, client):
    response = StreamingResponse([
        "event: error",
        'data: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}',
        "",
    ])
    mock_post.return_value = response

    with pytest.raises(LLMAPIServerError, match="Overloaded"):
        list(client.stream("claude-opus-4-8", messages=[]))

    response.close.assert_called_once()


@pytest.mark.unit
@patch("src.llm_api_adapter.llms.streaming.requests.post")
def test_stream_passes_unknown_event_types_through(mock_post, client):
    response = StreamingResponse([
        "event: future_event",
        'data: {"type":"future_event","value":"kept"}',
        "",
    ])
    mock_post.return_value = response

    assert list(client.stream("claude-opus-4-8", messages=[])) == [
        SSEEvent(event="future_event", data={"type": "future_event", "value": "kept"})
    ]


@pytest.mark.unit
def test_sync_client_positional_dataclass_defaults_repr_and_transport_patch_point():
    transport, equivalent_transport = Mock(name="first transport"), Mock(name="second transport")
    with patch(
        "src.llm_api_adapter.llms.anthropic.sync_client.create_sync_transport",
        side_effect=[transport, equivalent_transport, Mock(), Mock(), Mock()],
    ) as factory:
        client = ClaudeSyncClient("secret-key", "https://proxy.test/v1", "v-test", "requests")
        equivalent = ClaudeSyncClient("secret-key", "https://proxy.test/v1", "v-test", "requests")
        different_endpoint = ClaudeSyncClient("secret-key", "https://other.test/v1", "v-test", "requests")
        different_version = ClaudeSyncClient("secret-key", "https://proxy.test/v1", "v-other", "requests")
        different_transport = ClaudeSyncClient("secret-key", "https://proxy.test/v1", "v-test", "httpx")

    assert ShimSyncClient is ClaudeSyncClient
    assert [call.args[0] for call in factory.call_args_list] == [
        "requests", "requests", "requests", "requests", "httpx",
    ]
    assert client == equivalent
    assert client._sync_transport is transport
    assert equivalent._sync_transport is equivalent_transport
    assert client._sync_transport is not equivalent._sync_transport
    assert client != different_endpoint
    assert client != different_version
    assert client != different_transport
    assert "secret-key" not in repr(client)
    assert client.endpoint == "https://proxy.test/v1"
    assert client.api_version == "v-test"
    assert [field.name for field in fields(client)] == [
        "api_key", "endpoint", "api_version", "transport", "_sync_transport",
    ]
    internal_transport = fields(client)[-1]
    assert (internal_transport.init, internal_transport.repr, internal_transport.compare) == (
        False, False, False,
    )
    with patch("src.llm_api_adapter.llms.anthropic.sync_client.create_sync_transport", return_value=Mock()):
        defaults = ClaudeSyncClient("test_api_key")
    assert defaults.endpoint == "https://api.anthropic.com/v1"
    assert defaults.api_version == "2023-06-01"
    assert defaults.transport == "requests"


@pytest.mark.unit
def test_native_headers_and_endpoint_stay_module_patchable(client):
    assert client._headers() == {
        "x-api-key": "test_api_key",
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    with patch.object(client, "_send_request", return_value=Mock(json=lambda: {"id": "msg"})) as send:
        assert client.chat_completion("claude-sonnet-4-5", messages=[])["id"] == "msg"
    assert send.call_args.args[0] == "https://api.anthropic.com/v1/messages"


@pytest.mark.unit
def test_sync_custom_endpoint_version_timeout_and_bound_error_handlers():
    transport = Mock()
    response = Mock()
    response.json.return_value = {"id": "msg_custom"}
    transport.post_json.return_value = response
    events = Mock()
    events.__iter__ = Mock(return_value=iter([]))
    events.close = Mock()
    transport.post_sse.return_value = events
    with patch(
        "src.llm_api_adapter.llms.anthropic.sync_client.create_sync_transport",
        return_value=transport,
    ):
        client = ClaudeSyncClient(
            "custom-key", endpoint="https://gateway.test/prefix/v9",
            api_version="2025-01-01", transport="requests",
        )
        assert client.chat_completion("custom-model", timeout_s=7, messages=[]) == {"id": "msg_custom"}
        request, = transport.post_json.call_args.args
        http_handler = transport.post_json.call_args.kwargs["http_error_handler"]
        assert request.url == "https://gateway.test/prefix/v9/messages"
        assert request.timeout == 7
        assert request.headers_dict() == {
            "x-api-key": "custom-key",
            "anthropic-version": "2025-01-01",
            "Content-Type": "application/json",
        }
        assert http_handler.__self__ is client
        assert http_handler.__func__ is ClaudeSyncClient._handle_http_error

        assert list(client.stream("custom-model", timeout_s=11, messages=[])) == []
        stream_request, = transport.post_sse.call_args.args
        stream_kwargs = transport.post_sse.call_args.kwargs
        assert stream_request.url == "https://gateway.test/prefix/v9/messages"
        assert stream_request.timeout == 11
        assert stream_request.headers_dict()["x-api-key"] == "custom-key"
        assert stream_request.headers_dict()["anthropic-version"] == "2025-01-01"
        assert stream_kwargs["http_error_handler"].__self__ is client
        assert stream_kwargs["stream_error_handler"].__self__ is client
        assert stream_kwargs["http_error_handler"].__func__ is ClaudeSyncClient._handle_http_error
        assert stream_kwargs["stream_error_handler"].__func__ is ClaudeSyncClient._handle_stream_error
    events.close.assert_called_once()


@pytest.mark.unit
def test_shared_messages_native_json_and_stream_dispatch(monkeypatch, client):
    from src.llm_api_adapter.llms.anthropic_compatible.sync_client import (
        _SyncMessagesExecutionMixin,
    )

    response = Mock()
    response.json.return_value = {"id": "native-minimal"}

    class RawEvents:
        def __iter__(self):
            return iter([SSEEvent(event="future_native_event", data={"kept": True})])

        def close(self):
            pass

    raw_events = RawEvents()
    transport = Mock()
    transport.post_json.return_value = response
    transport.post_sse.return_value = raw_events
    client._sync_transport = transport

    json_requests = []
    stream_requests = []
    execute_json = _SyncMessagesExecutionMixin._execute_prepared_request
    execute_stream = _SyncMessagesExecutionMixin._execute_prepared_stream

    def observe_json(self, request):
        json_requests.append(request)
        return execute_json(self, request)

    def observe_stream(self, request):
        stream_requests.append(request)
        yield from execute_stream(self, request)

    monkeypatch.setattr(_SyncMessagesExecutionMixin, "_execute_prepared_request", observe_json)
    monkeypatch.setattr(_SyncMessagesExecutionMixin, "_execute_prepared_stream", observe_stream)

    result = client.chat_completion(
        "claude-opus-4-8", timeout_s=7, messages=[{"role": "user", "content": "Hi"}]
    )
    assert result == {"id": "native-minimal"}
    assert list(client.stream("claude-opus-4-8", timeout_s=11, messages=[])) == [
        SSEEvent(event="future_native_event", data={"kept": True})
    ]

    assert len(json_requests) == len(stream_requests) == 1
    assert json_requests[0].url == stream_requests[0].url == "https://api.anthropic.com/v1/messages"
    assert json_requests[0].timeout == 7
    assert stream_requests[0].timeout == 11
    assert json_requests[0].payload == {
        "model": "claude-opus-4-8",
        "messages": [{"role": "user", "content": "Hi"}],
    }
    assert stream_requests[0].payload["stream"] is True
    assert json_requests[0].headers_dict()["anthropic-version"] == "2023-06-01"
    assert transport.post_json.call_args.kwargs["http_error_handler"].__self__ is client
    stream_handlers = transport.post_sse.call_args.kwargs
    assert stream_handlers["http_error_handler"].__self__ is client
    assert stream_handlers["stream_error_handler"].__self__ is client


@pytest.mark.unit
def test_shared_messages_native_stream_binds_headers_and_transport_on_iteration(client):
    first_transport = Mock()
    second_transport = Mock()
    response = Mock()
    response.json.return_value = {"id": "late-bound"}
    second_transport.post_json.return_value = response
    source = Mock()
    source.__iter__ = Mock(return_value=iter([]))
    source.close = Mock()
    second_transport.post_sse.return_value = source
    client._sync_transport = first_transport

    events = client.stream("claude-opus-4-8", messages=[])

    client.api_key = "changed-key"
    client.api_version = "changed-version"
    client._sync_transport = second_transport
    assert list(events) == []
    request = second_transport.post_sse.call_args.args[0]
    assert request.headers_dict()["x-api-key"] == "changed-key"
    assert request.headers_dict()["anthropic-version"] == "changed-version"
    first_transport.post_sse.assert_not_called()
    source.close.assert_called_once()


@pytest.mark.unit
def test_shared_messages_native_preserves_public_and_json_error_identity(client):
    import json

    from src.llm_api_adapter.errors.llm_api_error import LLMAPIClientError

    public_error = LLMAPIClientError(detail="native identity")

    class ErrorTransport:
        def post_json(self, request, **kwargs):
            raise public_error

    client._sync_transport = ErrorTransport()
    with pytest.raises(LLMAPIClientError) as public_raised:
        client._send_request("https://gateway.test/messages", {})
    assert public_raised.value is public_error

    decode_error = json.JSONDecodeError("bad JSON", "not JSON", 0)

    class DecodeTransport:
        def post_json(self, request, **kwargs):
            raise decode_error

    client._sync_transport = DecodeTransport()
    with pytest.raises(json.JSONDecodeError) as decode_raised:
        client._send_request("https://gateway.test/messages", {})
    assert decode_raised.value is decode_error


@pytest.mark.unit
def test_shared_messages_native_stream_cleanup_preserves_primary_and_propagates_normal_close(client):
    primary = RuntimeError("native iteration failure")

    class BrokenSource:
        def __iter__(self):
            return self

        def __next__(self):
            raise primary

        def close(self):
            raise ValueError("secondary close failure")

    class BrokenTransport:
        def post_sse(self, request, **kwargs):
            return BrokenSource()

    client._sync_transport = BrokenTransport()
    with pytest.raises(RuntimeError) as iteration_raised:
        list(client._stream_request("https://gateway.test/messages", {}))
    assert iteration_raised.value is primary

    close_error = ValueError("normal close failure")

    class ClosingSource:
        def __iter__(self):
            return iter(())

        def close(self):
            raise close_error

    class ClosingTransport:
        def post_sse(self, request, **kwargs):
            return ClosingSource()

    client._sync_transport = ClosingTransport()
    with pytest.raises(ValueError) as close_raised:
        list(client._stream_request("https://gateway.test/messages", {}))
    assert close_raised.value is close_error
