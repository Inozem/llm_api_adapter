"""Evidence that Kimi sync requests use the compatible client foundation."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
import sys
from typing import Any
from unittest.mock import Mock

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[2]
for source in (PACKAGE_ROOT / "src", REPOSITORY_ROOT / "src", REPOSITORY_ROOT):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from llm_api_adapter.errors.llm_api_error import (
    LLMAPIAuthorizationError,
    LLMAPIRateLimitError,
)
from llm_api_adapter.llms.openai_compatible.sync_client import (
    OpenAICompatibleSyncClient,
)
from llm_api_adapter.llms.transports import (
    HTTPErrorHandler,
    JSONResponse,
    MultipartForm,
    SSEEvent,
    StreamErrorHandler,
    SyncTransport,
    TransportRequest,
)
from llm_api_adapter_kimi.clients.sync_client import (
    KIMI_CHAT_COMPLETIONS_URL,
    KimiSyncClient,
)


class _RecordingTransport(SyncTransport):
    def __init__(self, label: str = "transport") -> None:
        self.label = label
        self.json_requests: list[TransportRequest] = []
        self.stream_requests: list[TransportRequest] = []

    def post_json(
        self,
        request: TransportRequest,
        *,
        http_error_handler: HTTPErrorHandler | None = None,
    ) -> JSONResponse:
        self.json_requests.append(request)
        return JSONResponse({"choices": []})

    def post_multipart(
        self,
        request: TransportRequest,
        form: MultipartForm,
        *,
        http_error_handler: HTTPErrorHandler | None = None,
    ) -> JSONResponse:
        raise AssertionError("Kimi Chat Completions must not use multipart")

    def post_sse(
        self,
        request: TransportRequest,
        *,
        http_error_handler: HTTPErrorHandler | None = None,
        stream_error_handler: StreamErrorHandler | None = None,
    ) -> Iterator[SSEEvent]:
        self.stream_requests.append(request)
        return iter((SSEEvent(event=None, data={"transport": self.label}),))


class _TwoEventTransport(_RecordingTransport):
    def post_sse(
        self,
        request: TransportRequest,
        *,
        http_error_handler: HTTPErrorHandler | None = None,
        stream_error_handler: StreamErrorHandler | None = None,
    ) -> Iterator[SSEEvent]:
        self.stream_requests.append(request)
        return iter(
            (
                SSEEvent(event=None, data={"transport": self.label, "index": 1}),
                SSEEvent(event=None, data={"transport": self.label, "index": 2}),
            ),
        )


class _SingleEventTransport(_RecordingTransport):
    def __init__(self, event: SSEEvent) -> None:
        super().__init__("single-event")
        self.event = event

    def post_sse(
        self,
        request: TransportRequest,
        *,
        http_error_handler: HTTPErrorHandler | None = None,
        stream_error_handler: StreamErrorHandler | None = None,
    ) -> Iterator[SSEEvent]:
        self.stream_requests.append(request)
        return iter((self.event,))


class _NoCloseIterator:
    def __init__(self) -> None:
        self._events = iter((SSEEvent(event=None, data={"accepted": True}),))

    def __iter__(self) -> _NoCloseIterator:
        return self

    def __next__(self) -> SSEEvent:
        return next(self._events)


class _NoCloseTransport(_RecordingTransport):
    def post_sse(
        self,
        request: TransportRequest,
        *,
        http_error_handler: HTTPErrorHandler | None = None,
        stream_error_handler: StreamErrorHandler | None = None,
    ) -> Iterator[SSEEvent]:
        self.stream_requests.append(request)
        return _NoCloseIterator()  # type: ignore[return-value]


@pytest.mark.unit
def test_sync_client_directly_inherits_compatible_json_and_sse_runners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert KimiSyncClient.__bases__ == (OpenAICompatibleSyncClient,)

    reached: list[str] = []
    inherited_send = OpenAICompatibleSyncClient._send_request
    inherited_stream = OpenAICompatibleSyncClient._stream_request

    def send(
        self: OpenAICompatibleSyncClient,
        url: str,
        payload: dict[str, Any],
        timeout: float | None = None,
    ) -> JSONResponse:
        reached.append("json")
        return inherited_send(self, url, payload, timeout)

    def stream(
        self: OpenAICompatibleSyncClient,
        url: str,
        payload: dict[str, Any],
        timeout: float | None = None,
    ) -> Iterator[SSEEvent]:
        reached.append("sse")
        return inherited_stream(self, url, payload, timeout)

    monkeypatch.setattr(OpenAICompatibleSyncClient, "_send_request", send)
    monkeypatch.setattr(OpenAICompatibleSyncClient, "_stream_request", stream)
    transport = _RecordingTransport()
    client = KimiSyncClient(transport)

    assert client.chat(api_key="runner-key", payload={"model": "raw"}, timeout_s=3) == {
        "choices": [],
    }
    raw_events = client.stream(api_key="runner-key", payload={"model": "raw"}, timeout_s=4)
    assert transport.stream_requests == []
    assert [event.data for event in raw_events] == [{"transport": "transport"}]

    assert reached == ["json", "sse"]
    assert transport.json_requests[0].url == KIMI_CHAT_COMPLETIONS_URL
    assert transport.stream_requests[0].url == KIMI_CHAT_COMPLETIONS_URL
    assert transport.json_requests[0].headers["Authorization"] == "Bearer runner-key"
    assert transport.stream_requests[0].headers["Authorization"] == "Bearer runner-key"


@pytest.mark.unit
def test_sync_stream_adapts_only_sources_without_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _NoCloseTransport()
    client = KimiSyncClient(transport)
    request_view = Mock(wraps=client._request_view)
    monkeypatch.setattr(client, "_request_view", request_view)
    payload = {"model": "raw", "stream_options": {"discarded": True}}
    stream = client.stream(
        api_key="no-close-key",
        payload=payload,
        timeout_s=1.5,
    )

    assert transport.stream_requests == []
    request_view.assert_not_called()
    payload["model"] = "updated-before-consumption"
    events = list(stream)
    request_view.assert_called_once_with("no-close-key")
    assert events[0].data == {"accepted": True}
    request = transport.stream_requests[0]
    assert request.payload == {
        "model": "updated-before-consumption",
        "stream": True,
        "stream_options": {"include_usage": True},
    }


@pytest.mark.unit
def test_interleaved_sync_streams_keep_their_own_key_and_transport() -> None:
    first = _TwoEventTransport("first")
    second = _TwoEventTransport("second")
    client = KimiSyncClient(first)

    first_stream = client.stream(api_key="first-private-key", payload={"model": "m"}, timeout_s=None)
    assert first.stream_requests == []
    first_iterator = iter(first_stream)
    assert next(first_iterator).data == {"transport": "first", "index": 1}
    client.transport = second
    second_stream = client.stream(api_key="second-private-key", payload={"model": "m"}, timeout_s=None)
    second_iterator = iter(second_stream)
    assert next(second_iterator).data == {"transport": "second", "index": 1}

    assert [event.data for event in first_iterator] == [{"transport": "first", "index": 2}]
    assert [event.data for event in second_iterator] == [{"transport": "second", "index": 2}]
    assert first.stream_requests[0].headers["Authorization"] == "Bearer first-private-key"
    assert second.stream_requests[0].headers["Authorization"] == "Bearer second-private-key"


@pytest.mark.unit
def test_request_view_preserves_dynamic_classmethod_error_handlers() -> None:
    calls: list[tuple[str, object]] = []

    class _CustomKimiClient(KimiSyncClient):
        @classmethod
        def _handle_http_error(cls, error: Any) -> None:
            calls.append(("http", error))

        @classmethod
        def _handle_stream_error(cls, event: SSEEvent) -> None:
            calls.append(("stream", event))

    view = _CustomKimiClient(_RecordingTransport("custom"))._request_view("key")
    http_error = object()
    event = SSEEvent(event="error", data={"error": {"type": "custom_error"}})
    view._handle_http_error(http_error)
    view._handle_stream_error(event)

    assert calls == [("http", http_error), ("stream", event)]


@pytest.mark.unit
def test_kimi_sse_passes_through_responses_only_failure_frames() -> None:
    event = SSEEvent(event="response.failed", data={"type": "response.failed"})
    stream = KimiSyncClient(_SingleEventTransport(event)).stream(
        api_key="key",
        payload={"model": "m"},
        timeout_s=None,
    )

    assert list(stream) == [event]


@pytest.mark.unit
def test_kimi_native_sse_error_callback_remains_transport_owned() -> None:
    event = SSEEvent(event="error", data={"error": {"type": "native_error"}})
    calls: list[SSEEvent] = []

    class _NativeErrorTransport(_SingleEventTransport):
        def post_sse(
            self,
            request: TransportRequest,
            *,
            http_error_handler: HTTPErrorHandler | None = None,
            stream_error_handler: StreamErrorHandler | None = None,
        ) -> Iterator[SSEEvent]:
            self.stream_requests.append(request)
            assert stream_error_handler is not None
            stream_error_handler(event)
            return iter((self.event,))

    class _NativeHandlerClient(KimiSyncClient):
        @classmethod
        def _handle_stream_error(cls, stream_event: SSEEvent) -> None:
            calls.append(stream_event)

    stream = _NativeHandlerClient(_NativeErrorTransport(event)).stream(
        api_key="key",
        payload={"model": "m"},
        timeout_s=None,
    )

    assert list(stream) == [event]
    assert calls == [event]


@pytest.mark.unit
@pytest.mark.parametrize("transport", ["requests", "httpx"])
@pytest.mark.parametrize(
    ("status_code", "error_type", "expected_error"),
    [
        (429, "rate_limit_error", LLMAPIRateLimitError),
        (403, "rate_limit_error", LLMAPIAuthorizationError),
    ],
)
def test_real_transport_diagnostics_are_safe_and_native_mapping_wins(
    transport: str,
    status_code: int,
    error_type: str,
    expected_error: type[Exception],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret_key = "kimi-secret-key-should-not-log"
    sensitive_payload = {
        "error": {
            "type": error_type,
            "message": "sensitive-reasoning-and-tool-arguments",
        },
    }

    if transport == "requests":
        import requests

        class _Response:
            def __init__(self) -> None:
                self.status_code = status_code

            def raise_for_status(self) -> None:
                error = requests.exceptions.HTTPError("raw-provider-error-with-sensitive-data")
                error.response = self
                raise error

            def json(self) -> dict[str, Any]:
                return sensitive_payload

            def close(self) -> None:
                pass

        monkeypatch.setattr(requests, "post", lambda *args, **kwargs: _Response())
    else:
        import httpx

        class _Response:
            def __init__(self) -> None:
                self.status_code = status_code

            def raise_for_status(self) -> None:
                request = httpx.Request("POST", KIMI_CHAT_COMPLETIONS_URL)
                response = httpx.Response(
                    status_code,
                    request=request,
                    json=sensitive_payload,
                )
                raise httpx.HTTPStatusError(
                    "raw-provider-error-with-sensitive-data",
                    request=request,
                    response=response,
                )

            def json(self) -> dict[str, Any]:
                return sensitive_payload

            def close(self) -> None:
                pass

        class _Client:
            def post(self, *args: Any, **kwargs: Any) -> _Response:
                return _Response()

            def close(self) -> None:
                pass

        monkeypatch.setattr(httpx, "Client", _Client)

    from llm_api_adapter.llms.transports import create_sync_transport

    client = KimiSyncClient(create_sync_transport(transport))
    with pytest.raises(expected_error, match="sensitive-reasoning"):
        client.chat(api_key=secret_key, payload={"model": "raw"}, timeout_s=1)

    assert "raw-provider-error-with-sensitive-data" not in caplog.text
    assert secret_key not in caplog.text
    assert "sensitive-reasoning-and-tool-arguments" not in caplog.text
    assert "Kimi transport diagnostic" in caplog.text
