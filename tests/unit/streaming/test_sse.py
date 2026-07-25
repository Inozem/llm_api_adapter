from unittest.mock import patch

import pytest
import requests

from src.llm_api_adapter.errors.llm_api_error import (
    LLMAPIClientError,
    LLMAPIAuthorizationError,
    LLMAPIServerError,
)
from src.llm_api_adapter.llms.streaming import (
    SSEEvent,
    iter_sse_events,
    stream_request,
)


class FakeResponse:
    def __init__(self, lines, status_code=200, iter_error=None):
        self.lines = lines
        self.status_code = status_code
        self.iter_error = iter_error
        self.closed = False
        self.close_calls = 0

    def iter_lines(self, decode_unicode=True):
        if self.iter_error:
            raise self.iter_error
        return iter(self.lines)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(
                f"HTTP {self.status_code}", response=self
            )

    def close(self):
        self.close_calls += 1
        self.closed = True


@pytest.mark.unit
def test_iter_sse_events_parses_event_and_multiline_json_data():
    response = FakeResponse(
        [
            b"event: message",
            b'data: {"text":',
            b'data: "hello"}',
            b"",
        ]
    )

    assert list(iter_sse_events(response)) == [
        SSEEvent(event="message", data={"text": "hello"})
    ]
    assert response.closed is True


@pytest.mark.unit
def test_iter_sse_events_ignores_comments_and_ping_events():
    response = FakeResponse(
        [
            b": keep-alive",
            b"event: ping",
            b'data: {"type":"ping"}',
            b"",
            b'data: {"text":"ok"}',
            b"",
        ]
    )

    assert list(iter_sse_events(response)) == [
        SSEEvent(event=None, data={"text": "ok"})
    ]


@pytest.mark.unit
def test_iter_sse_events_recognizes_done_sentinel():
    response = FakeResponse([b"data: [DONE]", b"", b'data: {"ignored": true}', b""])

    assert list(iter_sse_events(response)) == [SSEEvent(event=None, done=True)]
    assert response.closed is True


@pytest.mark.unit
def test_iter_sse_events_flushes_complete_event_at_eof():
    response = FakeResponse([b'data: {"text":"eof"}'])

    assert list(iter_sse_events(response)) == [
        SSEEvent(event=None, data={"text": "eof"})
    ]
    assert response.closed is True


@pytest.mark.unit
def test_iter_sse_events_maps_malformed_json_and_closes_response():
    response = FakeResponse([b"data: not-json", b""])

    with pytest.raises(LLMAPIClientError, match="Malformed SSE JSON data"):
        list(iter_sse_events(response))

    assert response.closed is True


@pytest.mark.unit
def test_iter_sse_events_closes_response_when_consumer_stops_early():
    response = FakeResponse([b'data: {"text":"first"}', b"", b'data: {"text":"second"}', b""])
    events = iter_sse_events(response)

    assert next(events).data == {"text": "first"}
    events.close()
    assert response.closed is True


@pytest.mark.unit
def test_stream_request_uses_stream_true_and_closes_response():
    response = FakeResponse([b'data: {"text":"hello"}', b""])

    with patch(
        "src.llm_api_adapter.llms.streaming.requests.post", return_value=response
    ) as mock_post:
        events = list(
            stream_request(
                "https://example.test/stream",
                headers={"Authorization": "Bearer test"},
                payload={"prompt": "hello"},
                timeout=3.0,
            )
        )

    assert events == [SSEEvent(event=None, data={"text": "hello"})]
    assert response.closed is True
    assert response.close_calls == 1
    mock_post.assert_called_once_with(
        "https://example.test/stream",
        headers={"Authorization": "Bearer test"},
        json={"prompt": "hello"},
        timeout=3.0,
        stream=True,
    )


@pytest.mark.unit
def test_stream_request_maps_http_errors_to_unified_hierarchy():
    response = FakeResponse([], status_code=401)

    with patch(
        "src.llm_api_adapter.llms.streaming.requests.post", return_value=response
    ), pytest.raises(LLMAPIAuthorizationError):
        list(stream_request("https://example.test/stream"))

    assert response.closed is True


@pytest.mark.unit
def test_stream_request_uses_provider_http_error_handler():
    response = FakeResponse([], status_code=500)
    provider_error = LLMAPIServerError(detail="provider-specific")

    def handle_http_error(error):
        raise provider_error

    with patch(
        "src.llm_api_adapter.llms.streaming.requests.post", return_value=response
    ), pytest.raises(LLMAPIServerError, match="provider-specific"):
        list(
            stream_request(
                "https://example.test/stream",
                http_error_handler=handle_http_error,
            )
        )

    assert response.closed is True


@pytest.mark.unit
def test_stream_request_maps_generic_in_stream_error():
    response = FakeResponse(
        [
            b"event: error",
            b'data: {"error":{"type":"overloaded_error","message":"Overloaded"}}',
            b"",
        ]
    )

    with patch(
        "src.llm_api_adapter.llms.streaming.requests.post", return_value=response
    ), pytest.raises(LLMAPIClientError, match="overloaded_error: Overloaded"):
        list(stream_request("https://example.test/stream"))

    assert response.closed is True
