"""Deterministic sync-transport parity tests for every provider client.

These tests deliberately use local response doubles instead of provider
credentials or network mocks.  Each scenario runs through both concrete sync
transports and asserts the provider-facing request, result, error mapping,
stream framing, resource cleanup, and public callback order are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
from unittest.mock import patch

import httpx
import pytest
import requests

from src.llm_api_adapter.errors.llm_api_error import LLMAPIRateLimitError
from src.llm_api_adapter.llms.anthropic.sync_client import ClaudeSyncClient
from src.llm_api_adapter.llms.google.sync_client import GeminiSyncClient
from src.llm_api_adapter.llms.openai.sync_client import OpenAISyncClient
from src.llm_api_adapter.llms.transports import SSEEvent
from src.llm_api_adapter.models.messages.chat_message import UserMessage
from src.llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter


@dataclass(frozen=True)
class RequestTrace:
    """The transport-neutral portion of one HTTP operation."""

    url: str
    headers: dict[str, str]
    payload: dict[str, Any]
    timeout: float | None
    stream: bool


@dataclass(frozen=True)
class OrganizationCase:
    name: str
    client_class: type
    organization: str
    model: str
    client_kwargs: dict[str, Any]
    public_stream_kwargs: dict[str, Any]
    expected_json_request: RequestTrace
    expected_stream_request: RequestTrace
    stream_lines: list[str]
    expected_stream_events: list[SSEEvent]
    error_payload: dict[str, Any]
    stream_error_lines: list[str]


_USER_MESSAGES = [{"role": "user", "content": "Hi"}]
_GOOGLE_CONTENTS = [{"role": "user", "parts": [{"text": "Hi"}]}]

ORGANIZATION_CASES = (
    OrganizationCase(
        name="openai",
        client_class=OpenAISyncClient,
        organization="openai",
        model="gpt-4o",
        client_kwargs={"messages": _USER_MESSAGES},
        public_stream_kwargs={},
        expected_json_request=RequestTrace(
            url="https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": "Bearer test-key",
                "Content-Type": "application/json",
            },
            payload={"model": "gpt-4o", "messages": _USER_MESSAGES},
            timeout=None,
            stream=False,
        ),
        expected_stream_request=RequestTrace(
            url="https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": "Bearer test-key",
                "Content-Type": "application/json",
            },
            payload={
                "model": "gpt-4o",
                "messages": _USER_MESSAGES,
                "stream": True,
            },
            timeout=None,
            stream=True,
        ),
        stream_lines=[
            'data: {"choices":[{"delta":{"content":"Hel"}}]}',
            "",
            'data: {"choices":[{"delta":{"content":"lo!"}}]}',
            "",
            "data: [DONE]",
            "",
        ],
        expected_stream_events=[
            SSEEvent(event=None, data={"choices": [{"delta": {"content": "Hel"}}]}),
            SSEEvent(event=None, data={"choices": [{"delta": {"content": "lo!"}}]}),
        ],
        error_payload={
            "error": {"type": "rate_limit_exceeded", "message": "Slow down"}
        },
        stream_error_lines=[
            "event: error",
            'data: {"error":{"code":"rate_limit_exceeded","message":"Slow down"}}',
            "",
        ],
    ),
    OrganizationCase(
        name="anthropic",
        client_class=ClaudeSyncClient,
        organization="anthropic",
        model="claude-sonnet-4-5",
        client_kwargs={"messages": _USER_MESSAGES, "max_tokens": 64},
        public_stream_kwargs={"max_tokens": 64},
        expected_json_request=RequestTrace(
            url="https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": "test-key",
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            payload={
                "model": "claude-sonnet-4-5",
                "messages": _USER_MESSAGES,
                "max_tokens": 64,
            },
            timeout=None,
            stream=False,
        ),
        expected_stream_request=RequestTrace(
            url="https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": "test-key",
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            payload={
                "model": "claude-sonnet-4-5",
                "messages": _USER_MESSAGES,
                "max_tokens": 64,
                "stream": True,
            },
            timeout=None,
            stream=True,
        ),
        stream_lines=[
            "event: message_start",
            'data: {"type":"message_start","message":{"id":"msg_test","content":[]}}',
            "",
            "event: content_block_start",
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
            "",
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hel"}}',
            "",
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"lo!"}}',
            "",
            "event: message_stop",
            'data: {"type":"message_stop"}',
            "",
        ],
        expected_stream_events=[
            SSEEvent(
                event="message_start",
                data={
                    "type": "message_start",
                    "message": {"id": "msg_test", "content": []},
                },
            ),
            SSEEvent(
                event="content_block_start",
                data={
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
            SSEEvent(
                event="content_block_delta",
                data={
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "Hel"},
                },
            ),
            SSEEvent(
                event="content_block_delta",
                data={
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "lo!"},
                },
            ),
            SSEEvent(event="message_stop", data={"type": "message_stop"}),
        ],
        error_payload={"error": {"type": "rate_limit_error", "message": "Slow down"}},
        stream_error_lines=[
            "event: error",
            'data: {"type":"error","error":{"type":"rate_limit_error","message":"Slow down"}}',
            "",
        ],
    ),
    OrganizationCase(
        name="google",
        client_class=GeminiSyncClient,
        organization="google",
        model="gemini-2.5-flash",
        client_kwargs={"contents": _GOOGLE_CONTENTS},
        public_stream_kwargs={},
        expected_json_request=RequestTrace(
            url=(
                "https://generativelanguage.googleapis.com/v1beta/"
                "models/gemini-2.5-flash:generateContent"
            ),
            headers={
                "x-goog-api-key": "test-key",
                "Content-Type": "application/json",
            },
            payload={"model": "gemini-2.5-flash", "contents": _GOOGLE_CONTENTS},
            timeout=None,
            stream=False,
        ),
        expected_stream_request=RequestTrace(
            url=(
                "https://generativelanguage.googleapis.com/v1beta/"
                "models/gemini-2.5-flash:streamGenerateContent?alt=sse"
            ),
            headers={
                "x-goog-api-key": "test-key",
                "Content-Type": "application/json",
            },
            payload={"model": "gemini-2.5-flash", "contents": _GOOGLE_CONTENTS},
            timeout=None,
            stream=True,
        ),
        stream_lines=[
            'data: {"candidates":[{"content":{"parts":[{"text":"Hel"}]}}]}',
            "",
            'data: {"candidates":[{"content":{"parts":[{"text":"lo!"}]},"finishReason":"STOP"}]}',
            "",
        ],
        expected_stream_events=[
            SSEEvent(
                event=None,
                data={"candidates": [{"content": {"parts": [{"text": "Hel"}]}}]},
            ),
            SSEEvent(
                event=None,
                data={
                    "candidates": [
                        {
                            "content": {"parts": [{"text": "lo!"}]},
                            "finishReason": "STOP",
                        }
                    ]
                },
            ),
        ],
        error_payload={
            "error": {"status": "RESOURCE_EXHAUSTED", "message": "Slow down"}
        },
        stream_error_lines=[
            'data: {"error":{"status":"RESOURCE_EXHAUSTED","message":"Slow down"}}',
            "",
        ],
    ),
)


class _ResponseDouble:
    def __init__(
        self,
        *,
        body: dict[str, Any] | None = None,
        lines: list[str] | None = None,
        status_code: int = 200,
    ) -> None:
        self.body = body or {}
        self.lines = lines or []
        self.status_code = status_code
        self.close_calls = 0

    def json(self) -> dict[str, Any]:
        return self.body

    def iter_lines(self, decode_unicode: bool = True):
        _ = decode_unicode
        return iter(self.lines)

    def close(self) -> None:
        self.close_calls += 1


class _RequestsResponse(_ResponseDouble):
    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(
                f"HTTP {self.status_code}",
                response=self,
            )


class _HttpxResponse(_ResponseDouble):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.request = httpx.Request("POST", "https://example.test")

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=self.request,
                response=self,
            )


class _HttpxClientDouble:
    def __init__(self, response: _HttpxResponse) -> None:
        self.response = response
        self.calls: list[RequestTrace] = []
        self._built_request: RequestTrace | None = None
        self.close_calls = 0

    def post(self, url: str, **kwargs: Any) -> _HttpxResponse:
        self.calls.append(
            RequestTrace(
                url=url,
                headers=kwargs["headers"],
                payload=kwargs["json"],
                timeout=kwargs["timeout"],
                stream=False,
            )
        )
        return self.response

    def build_request(self, method: str, url: str, **kwargs: Any) -> httpx.Request:
        assert method == "POST"
        self._built_request = RequestTrace(
            url=url,
            headers=kwargs["headers"],
            payload=kwargs["json"],
            timeout=kwargs["timeout"],
            stream=True,
        )
        return httpx.Request(method, url, headers=kwargs["headers"], json=kwargs["json"])

    def send(self, request: httpx.Request, *, stream: bool) -> _HttpxResponse:
        _ = request
        assert stream is True
        assert self._built_request is not None
        self.calls.append(self._built_request)
        return self.response

    def close(self) -> None:
        self.close_calls += 1


@dataclass
class _Capture:
    result: Any
    error: Exception | None
    call: RequestTrace
    response: _ResponseDouble
    httpx_client_close_calls: int | None


def _run(
    transport: str,
    response: _ResponseDouble,
    invoke: Callable[[], Any],
) -> _Capture:
    """Run one operation through the selected transport and record its shape."""
    result: Any = None
    error: Exception | None = None

    if transport == "requests":
        calls: list[RequestTrace] = []

        def post(url: str, **kwargs: Any) -> _RequestsResponse:
            calls.append(
                RequestTrace(
                    url=url,
                    headers=kwargs["headers"],
                    payload=kwargs["json"],
                    timeout=kwargs["timeout"],
                    stream=kwargs.get("stream", False),
                )
            )
            return response  # type: ignore[return-value]

        with patch(
            "src.llm_api_adapter.llms.requests_transport.requests.post",
            side_effect=post,
        ):
            try:
                result = invoke()
            except Exception as exc:  # Captured for cross-transport comparison.
                error = exc

        assert len(calls) == 1
        return _Capture(result, error, calls[0], response, None)

    client = _HttpxClientDouble(response)  # type: ignore[arg-type]
    with patch.object(httpx, "Client", return_value=client):
        try:
            result = invoke()
        except Exception as exc:  # Captured for cross-transport comparison.
            error = exc

    assert len(client.calls) == 1
    return _Capture(result, error, client.calls[0], response, client.close_calls)


def _response_for(
    transport: str,
    *,
    body: dict[str, Any] | None = None,
    lines: list[str] | None = None,
    status_code: int = 200,
) -> _ResponseDouble:
    response_class = _RequestsResponse if transport == "requests" else _HttpxResponse
    return response_class(body=body, lines=lines, status_code=status_code)


def _assert_lifecycle(capture: _Capture) -> None:
    assert capture.response.close_calls == 1
    if capture.httpx_client_close_calls is not None:
        assert capture.httpx_client_close_calls == 1


@pytest.mark.unit
@pytest.mark.parametrize("case", ORGANIZATION_CASES, ids=lambda case: case.name)
def test_sync_json_provider_requests_are_identical_for_both_transports(
    case: OrganizationCase,
):
    captures = {}
    for transport in ("requests", "httpx"):
        client = case.client_class(api_key="test-key", transport=transport)
        response = _response_for(transport, body={"provider": case.name})
        captures[transport] = _run(
            transport,
            response,
            lambda: client.chat_completion(case.model, **case.client_kwargs),
        )

    requests_capture = captures["requests"]
    httpx_capture = captures["httpx"]
    assert requests_capture.error is None
    assert httpx_capture.error is None
    assert requests_capture.result == httpx_capture.result == {"provider": case.name}
    assert requests_capture.call == httpx_capture.call == case.expected_json_request
    _assert_lifecycle(requests_capture)
    _assert_lifecycle(httpx_capture)


@pytest.mark.unit
@pytest.mark.parametrize("case", ORGANIZATION_CASES, ids=lambda case: case.name)
def test_sync_http_error_mapping_is_identical_for_both_transports(
    case: OrganizationCase,
):
    captures = {}
    for transport in ("requests", "httpx"):
        client = case.client_class(api_key="test-key", transport=transport)
        response = _response_for(
            transport,
            body=case.error_payload,
            status_code=429,
        )
        captures[transport] = _run(
            transport,
            response,
            lambda: client.chat_completion(case.model, **case.client_kwargs),
        )

    requests_capture = captures["requests"]
    httpx_capture = captures["httpx"]
    assert isinstance(requests_capture.error, LLMAPIRateLimitError)
    assert isinstance(httpx_capture.error, LLMAPIRateLimitError)
    assert str(requests_capture.error) == str(httpx_capture.error)
    assert requests_capture.call == httpx_capture.call == case.expected_json_request
    _assert_lifecycle(requests_capture)
    _assert_lifecycle(httpx_capture)


@pytest.mark.unit
@pytest.mark.parametrize("case", ORGANIZATION_CASES, ids=lambda case: case.name)
def test_sync_sse_framing_and_cleanup_are_identical_for_both_transports(
    case: OrganizationCase,
):
    captures = {}
    for transport in ("requests", "httpx"):
        client = case.client_class(api_key="test-key", transport=transport)
        response = _response_for(transport, lines=case.stream_lines)
        captures[transport] = _run(
            transport,
            response,
            lambda: list(client.stream(case.model, **case.client_kwargs)),
        )

    requests_capture = captures["requests"]
    httpx_capture = captures["httpx"]
    assert requests_capture.error is None
    assert httpx_capture.error is None
    assert requests_capture.result == httpx_capture.result == case.expected_stream_events
    assert requests_capture.call == httpx_capture.call == case.expected_stream_request
    _assert_lifecycle(requests_capture)
    _assert_lifecycle(httpx_capture)


@pytest.mark.unit
@pytest.mark.parametrize("case", ORGANIZATION_CASES, ids=lambda case: case.name)
def test_sync_sse_error_mapping_is_identical_for_both_transports(
    case: OrganizationCase,
):
    captures = {}
    for transport in ("requests", "httpx"):
        client = case.client_class(api_key="test-key", transport=transport)
        response = _response_for(transport, lines=case.stream_error_lines)
        captures[transport] = _run(
            transport,
            response,
            lambda: list(client.stream(case.model, **case.client_kwargs)),
        )

    requests_capture = captures["requests"]
    httpx_capture = captures["httpx"]
    assert isinstance(requests_capture.error, LLMAPIRateLimitError)
    assert isinstance(httpx_capture.error, LLMAPIRateLimitError)
    assert str(requests_capture.error) == str(httpx_capture.error)
    assert requests_capture.call == httpx_capture.call == case.expected_stream_request
    _assert_lifecycle(requests_capture)
    _assert_lifecycle(httpx_capture)


@pytest.mark.unit
@pytest.mark.parametrize("case", ORGANIZATION_CASES, ids=lambda case: case.name)
def test_sync_transport_preserves_public_stream_callback_order(
    case: OrganizationCase,
):
    captures = {}
    for transport in ("requests", "httpx"):
        response = _response_for(transport, lines=case.stream_lines)

        def invoke() -> tuple[list[str], list[tuple[str, str]]]:
            adapter = UniversalLLMAPIAdapter(
                organization=case.organization,
                model=case.model,
                api_key="test-key",
                transport=transport,
            )
            order: list[tuple[str, str]] = []
            yielded = []

            def on_chunk(chunk: Any) -> None:
                order.append(("chunk", chunk.text))

            def on_delta(text: str) -> None:
                order.append(("delta", text))

            def on_done(result: Any) -> None:
                order.append(("done", result.content))

            for text in adapter.stream_chat(
                [UserMessage("Hi")],
                buffer_chars=3,
                on_chunk=on_chunk,
                on_delta=on_delta,
                on_done=on_done,
                **case.public_stream_kwargs,
            ):
                yielded.append(text)
                order.append(("yield", text))
            return yielded, order

        captures[transport] = _run(transport, response, invoke)

    expected_result = (
        ["Hel", "lo!"],
        [
            ("chunk", "Hel"),
            ("delta", "Hel"),
            ("yield", "Hel"),
            ("chunk", "lo!"),
            ("delta", "lo!"),
            ("yield", "lo!"),
            ("done", "Hello!"),
        ],
    )
    requests_capture = captures["requests"]
    httpx_capture = captures["httpx"]
    assert requests_capture.error is None
    assert httpx_capture.error is None
    assert requests_capture.result == httpx_capture.result == expected_result
    _assert_lifecycle(requests_capture)
    _assert_lifecycle(httpx_capture)
