"""Evidence that Kimi sync requests use the compatible client foundation."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator, Mapping
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
import llm_api_adapter.adapters.base_adapter as base_adapter_module
from llm_api_adapter.adapters.openai_compatible.adapter import OpenAICompatibleAdapter
from llm_api_adapter.adapters.openai_compatible.chat_completions.streaming import (
    _ChatCompletionsStreamingMixin,
)
from llm_api_adapter.llms.openai_compatible.async_client import (
    OpenAICompatibleAsyncClient,
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
from llm_api_adapter.llm_registry.llm_registry import RegistrySpec
from llm_api_adapter_kimi.clients.sync_client import (
    KIMI_CHAT_COMPLETIONS_URL,
    KimiSyncClient,
)
from llm_api_adapter_kimi.clients.async_client import KimiAsyncClient
from llm_api_adapter_kimi.adapter import KimiAdapter
from llm_api_adapter_kimi.plugin import PLUGIN
from llm_api_adapter.service_provider_registry import ServiceProviderRegistry


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


@pytest.mark.unit
def test_async_client_uses_compatible_runners_without_sync_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert KimiAsyncClient.__bases__ == (
        OpenAICompatibleAsyncClient,
        KimiSyncClient,
    )
    transport_construction: list[str] = []

    def create_sync_transport(self: OpenAICompatibleSyncClient, transport: str) -> Any:
        transport_construction.append(transport)
        raise AssertionError("async client construction must not create sync transport")

    monkeypatch.setattr(
        OpenAICompatibleSyncClient,
        "_create_sync_transport",
        create_sync_transport,
    )
    client = KimiAsyncClient()
    assert transport_construction == []

    from llm_api_adapter_kimi.clients import async_client as async_client_module

    raw_payload = {"model": "raw-model", "native_extension": {"keep": True}}
    calls: list[tuple[str, dict[str, Any]]] = []

    async def async_request(url: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((url, kwargs))
        return {"choices": []}

    monkeypatch.setattr(async_client_module, "async_request", async_request)
    reached: list[str] = []
    inherited_send = OpenAICompatibleAsyncClient._send_request
    inherited_stream = OpenAICompatibleAsyncClient._stream_request

    async def send(
        self: OpenAICompatibleAsyncClient,
        url: str,
        payload: dict[str, Any],
        timeout: float | None = None,
    ) -> Any:
        reached.append("json")
        return await inherited_send(self, url, payload, timeout)

    async def stream(
        self: OpenAICompatibleAsyncClient,
        url: str,
        payload: dict[str, Any],
        timeout: float | None = None,
    ) -> AsyncIterator[SSEEvent]:
        reached.append("sse")
        async for event in inherited_stream(self, url, payload, timeout):
            yield event

    monkeypatch.setattr(OpenAICompatibleAsyncClient, "_send_request", send)
    monkeypatch.setattr(OpenAICompatibleAsyncClient, "_stream_request", stream)
    result = asyncio.run(
        client.chat(api_key="async-key", payload=raw_payload, timeout_s=2.75)
    )
    assert result == {"choices": []}
    assert calls[0][0] == KIMI_CHAT_COMPLETIONS_URL
    assert calls[0][1]["payload"] == raw_payload
    assert calls[0][1]["headers"] == {
        "Authorization": "Bearer async-key",
        "Content-Type": "application/json",
    }
    assert calls[0][1]["timeout"] == 2.75
    assert raw_payload == {"model": "raw-model", "native_extension": {"keep": True}}

    stream_calls: list[tuple[str, dict[str, Any]]] = []

    def async_stream_request(url: str, **kwargs: Any) -> AsyncIterator[SSEEvent]:
        stream_calls.append((url, kwargs))

        async def events() -> AsyncIterator[SSEEvent]:
            yield SSEEvent(event=None, data={"event": "one"})

        return events()

    monkeypatch.setattr(async_client_module, "async_stream_request", async_stream_request)
    stream_payload = {"model": "raw-stream-model", "stream_options": {"custom": True}}
    raw_events = client.stream(
        api_key="async-stream-key",
        payload=stream_payload,
        timeout_s=4.25,
    )
    assert stream_calls == []

    async def collect() -> list[SSEEvent]:
        return [event async for event in raw_events]

    events = asyncio.run(collect())
    assert [event.data for event in events] == [{"event": "one"}]
    assert reached == ["json", "sse"]
    assert stream_calls[0][0] == KIMI_CHAT_COMPLETIONS_URL
    assert stream_calls[0][1]["payload"] == {
        "model": "raw-stream-model",
        "stream_options": {"include_usage": True},
        "stream": True,
    }
    assert stream_calls[0][1]["headers"]["Authorization"] == "Bearer async-stream-key"
    assert stream_calls[0][1]["timeout"] == 4.25
    assert len(stream_calls) == 1
    assert stream_payload == {
        "model": "raw-stream-model",
        "stream_options": {"custom": True},
    }
    assert stream_payload["stream_options"] == {"custom": True}


@pytest.mark.unit
def test_async_stream_resolves_alias_after_creation_and_binds_interleaved_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_api_adapter_kimi.clients import async_client as async_client_module

    client = KimiAsyncClient()
    first = client.stream(api_key="first-async-key", payload={"model": "m"}, timeout_s=None)
    calls: list[tuple[str, dict[str, Any]]] = []

    def async_stream_request(url: str, **kwargs: Any) -> AsyncIterator[SSEEvent]:
        calls.append((url, kwargs))

        async def events() -> AsyncIterator[SSEEvent]:
            try:
                yield SSEEvent(event=None, data={"index": 1})
                yield SSEEvent(event=None, data={"index": 2})
            finally:
                completed.append(kwargs["headers"]["Authorization"])

        return events()

    completed: list[str] = []
    monkeypatch.setattr(async_client_module, "async_stream_request", async_stream_request)
    second = client.stream(api_key="second-async-key", payload={"model": "m"}, timeout_s=None)

    async def consume_interleaved() -> None:
        first_iterator = first.__aiter__()
        assert (await first_iterator.__anext__()).data == {"index": 1}
        second_iterator = second.__aiter__()
        assert (await second_iterator.__anext__()).data == {"index": 1}
        assert (await first_iterator.__anext__()).data == {"index": 2}
        assert (await second_iterator.__anext__()).data == {"index": 2}
        await first_iterator.aclose()
        await second_iterator.aclose()

    asyncio.run(consume_interleaved())
    assert [call[1]["headers"]["Authorization"] for call in calls] == [
        "Bearer first-async-key",
        "Bearer second-async-key",
    ]
    assert [call[0] for call in calls] == [KIMI_CHAT_COMPLETIONS_URL] * 2
    assert completed == ["Bearer first-async-key", "Bearer second-async-key"]


@pytest.mark.unit
def test_async_request_view_preserves_native_handler_dispatch_and_error_classes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_api_adapter.errors.llm_api_error import LLMAPIAuthorizationError
    from llm_api_adapter_kimi.clients import async_client as async_client_module

    calls: list[tuple[str, object]] = []

    class CustomKimiAsyncClient(KimiAsyncClient):
        @classmethod
        def _handle_http_error(cls, error: Any) -> None:
            calls.append(("http", error))

        @classmethod
        def _handle_stream_error(cls, event: SSEEvent) -> None:
            calls.append(("stream", event))

    client = CustomKimiAsyncClient()
    error = object()

    async def async_request(url: str, **kwargs: Any) -> dict[str, Any]:
        kwargs["http_error_handler"](error)
        return {"choices": []}

    monkeypatch.setattr(async_client_module, "async_request", async_request)
    assert asyncio.run(client.chat(api_key="key", payload={}, timeout_s=None)) == {
        "choices": [],
    }

    class FakeResponse:
        status_code = 401

        def json(self) -> dict[str, Any]:
            return {"error": {"type": "invalid_api_key", "message": "denied"}}

    class FakeHTTPError(Exception):
        response = FakeResponse()

    async def failing_request(url: str, **kwargs: Any) -> Any:
        kwargs["http_error_handler"](FakeHTTPError("no secret diagnostic"))

    monkeypatch.setattr(async_client_module, "async_request", failing_request)
    with pytest.raises(LLMAPIAuthorizationError, match="denied"):
        asyncio.run(KimiAsyncClient().chat(api_key="key", payload={}, timeout_s=None))
    assert calls == [("http", error)]


@pytest.mark.unit
def test_async_kimi_sse_passes_through_responses_only_failure_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_api_adapter_kimi.clients import async_client as async_client_module

    event = SSEEvent(event="response.failed", data={"type": "response.failed"})

    async def events() -> AsyncIterator[SSEEvent]:
        yield event

    monkeypatch.setattr(
        async_client_module,
        "async_stream_request",
        lambda *args, **kwargs: events(),
    )
    stream = KimiAsyncClient().stream(
        api_key="key",
        payload={"model": "m"},
        timeout_s=None,
    )

    async def collect() -> list[SSEEvent]:
        return [item async for item in stream]

    assert asyncio.run(collect()) == [event]


@pytest.mark.unit
def test_async_stream_adapts_missing_aclose_and_closes_real_source_on_early_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_api_adapter_kimi.clients import async_client as async_client_module

    client = KimiAsyncClient()

    class NoCloseIterator:
        def __init__(self) -> None:
            self._events = iter((SSEEvent(event=None, data={"accepted": True}),))

        def __aiter__(self) -> NoCloseIterator:
            return self

        async def __anext__(self) -> SSEEvent:
            try:
                return next(self._events)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

    no_close = NoCloseIterator()
    monkeypatch.setattr(
        async_client_module,
        "async_stream_request",
        lambda *args, **kwargs: no_close,
    )
    no_close_stream = client.stream(api_key="key", payload={"model": "m"}, timeout_s=None)

    async def consume_no_close() -> list[SSEEvent]:
        return [event async for event in no_close_stream]

    assert asyncio.run(consume_no_close())[0].data == {"accepted": True}

    closed: list[bool] = []

    async def closeable_events() -> AsyncIterator[SSEEvent]:
        try:
            yield SSEEvent(event=None, data={"index": 1})
            yield SSEEvent(event=None, data={"index": 2})
        finally:
            closed.append(True)

    monkeypatch.setattr(
        async_client_module,
        "async_stream_request",
        lambda *args, **kwargs: closeable_events(),
    )
    stream = client.stream(api_key="key", payload={"model": "m"}, timeout_s=None)

    async def consume_one_then_close() -> None:
        iterator = stream.__aiter__()
        assert (await iterator.__anext__()).data == {"index": 1}
        await iterator.aclose()

    asyncio.run(consume_one_then_close())
    assert closed == [True]


@pytest.mark.unit
def test_registered_adapter_reaches_compatible_execution_and_native_runners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Instrument the actual registered type while each inherited layer calls through."""
    registry = ServiceProviderRegistry()
    model_registry = RegistrySpec()
    assert PLUGIN.model_metadata is not None
    model_registry.register_organization_metadata(PLUGIN.model_metadata)
    monkeypatch.setattr(base_adapter_module, "LLM_REGISTRY", model_registry)
    PLUGIN.register(registry)
    factory = registry.get("kimi")
    assert factory is KimiAdapter
    adapter = factory(api_key="registered-test-key", model="kimi-k3")

    assert KimiAdapter.__bases__ == (OpenAICompatibleAdapter,)
    assert type(adapter) is KimiAdapter
    assert type(adapter._client) is KimiSyncClient
    assert type(adapter._async_client) is KimiAsyncClient
    assert KimiAdapter.chat is OpenAICompatibleAdapter.chat
    assert KimiAdapter.achat is OpenAICompatibleAdapter.achat
    assert KimiAdapter.astream_chat is OpenAICompatibleAdapter.astream_chat

    from llm_api_adapter_kimi.clients import async_client as async_client_module

    response = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
    event = SSEEvent(
        event=None,
        data={"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]},
    )
    class _RegisteredTransport(_SingleEventTransport):
        def post_json(
            self,
            request: TransportRequest,
            *,
            http_error_handler: HTTPErrorHandler | None = None,
        ) -> JSONResponse:
            self.json_requests.append(request)
            return JSONResponse(response)

    transport = _RegisteredTransport(event)
    adapter._sync_transport = transport
    reached: list[str] = []

    inherited_chat = OpenAICompatibleAdapter.chat
    inherited_achat = OpenAICompatibleAdapter.achat
    inherited_stream_chat = OpenAICompatibleAdapter.stream_chat
    inherited_astream = OpenAICompatibleAdapter._astream_chat
    inherited_send = OpenAICompatibleSyncClient._send_request
    inherited_sync_stream = OpenAICompatibleSyncClient._stream_request
    inherited_async_send = OpenAICompatibleAsyncClient._send_request
    inherited_async_stream = OpenAICompatibleAsyncClient._stream_request
    inherited_consume = _ChatCompletionsStreamingMixin._consume_chat_completions_stream
    inherited_async_consume = _ChatCompletionsStreamingMixin._consume_chat_completions_stream_async

    def spy(name: str, function: Any):
        def call_through(self: Any, *args: Any, **kwargs: Any) -> Any:
            reached.append(name)
            return function(self, *args, **kwargs)
        return call_through

    async def async_send(self: Any, *args: Any, **kwargs: Any) -> Any:
        reached.append("async-json-runner")
        return await inherited_async_send(self, *args, **kwargs)

    async def async_stream(self: Any, *args: Any, **kwargs: Any) -> AsyncIterator[SSEEvent]:
        reached.append("async-sse-runner")
        async for item in inherited_async_stream(self, *args, **kwargs):
            yield item

    async def async_request(url: str, **kwargs: Any) -> dict[str, Any]:
        return response

    def async_stream_request(url: str, **kwargs: Any) -> AsyncIterator[SSEEvent]:
        async def events() -> AsyncIterator[SSEEvent]:
            yield event
        return events()

    monkeypatch.setattr(OpenAICompatibleAdapter, "chat", spy("chat", inherited_chat))
    monkeypatch.setattr(OpenAICompatibleAdapter, "achat", spy("achat", inherited_achat))
    monkeypatch.setattr(
        OpenAICompatibleAdapter,
        "stream_chat",
        spy("stream_chat", inherited_stream_chat),
    )
    monkeypatch.setattr(OpenAICompatibleAdapter, "_astream_chat", spy("_astream_chat", inherited_astream))
    monkeypatch.setattr(OpenAICompatibleSyncClient, "_send_request", spy("sync-json-runner", inherited_send))
    monkeypatch.setattr(OpenAICompatibleSyncClient, "_stream_request", spy("sync-sse-runner", inherited_sync_stream))
    monkeypatch.setattr(OpenAICompatibleAsyncClient, "_send_request", async_send)
    monkeypatch.setattr(OpenAICompatibleAsyncClient, "_stream_request", async_stream)
    monkeypatch.setattr(
        _ChatCompletionsStreamingMixin,
        "_consume_chat_completions_stream",
        spy("sync-consumer", inherited_consume),
    )
    monkeypatch.setattr(
        _ChatCompletionsStreamingMixin,
        "_consume_chat_completions_stream_async",
        spy("async-consumer", inherited_async_consume),
    )
    monkeypatch.setattr(async_client_module, "async_request", async_request)
    monkeypatch.setattr(async_client_module, "async_stream_request", async_stream_request)

    assert adapter.chat([{"role": "user", "content": "hi"}]).content == "ok"
    assert list(adapter.stream_chat([{"role": "user", "content": "hi"}])) == ["ok"]
    assert asyncio.run(adapter.achat([{"role": "user", "content": "hi"}])).content == "ok"

    async def consume_async_stream() -> list[str]:
        return [
            text
            async for text in adapter.astream_chat([{"role": "user", "content": "hi"}])
        ]

    assert asyncio.run(consume_async_stream()) == ["ok"]
    assert reached.count("chat") == 1
    assert reached.count("achat") == 1
    assert reached.count("stream_chat") == 1
    assert reached.count("_astream_chat") == 1
    assert reached.count("sync-json-runner") == 1
    assert reached.count("sync-sse-runner") == 1
    assert reached.count("async-json-runner") == 1
    assert reached.count("async-sse-runner") == 1
    assert reached.count("sync-consumer") == 1
    assert reached.count("async-consumer") == 1


@pytest.mark.unit
def test_adapter_bindings_snapshot_distinct_keys_without_client_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_api_adapter_kimi.clients import async_client as async_client_module

    sync_transport = _RecordingTransport()
    sync_client = KimiSyncClient(sync_transport)
    first_sync = sync_client._adapter_binding("first-key")
    second_sync = sync_client._adapter_binding("second-key")

    first_sync.complete("kimi-k3", messages=[])
    second_sync.complete("kimi-k3", messages=[])
    assert [request.headers["Authorization"] for request in sync_transport.json_requests] == [
        "Bearer first-key",
        "Bearer second-key",
    ]
    assert sync_client.api_key == ""

    async_client = KimiAsyncClient()
    async_calls: list[dict[str, Any]] = []

    async def fake_request(url: str, **kwargs: Any) -> dict[str, Any]:
        async_calls.append(kwargs)
        return {"choices": []}

    monkeypatch.setattr(async_client_module, "async_request", fake_request)
    first_async = async_client._adapter_binding("first-async-key")
    second_async = async_client._adapter_binding("second-async-key")

    async def run_requests() -> None:
        await first_async.complete("kimi-k3", messages=[])
        await second_async.complete("kimi-k3", messages=[])

    asyncio.run(run_requests())
    assert [call["headers"]["Authorization"] for call in async_calls] == [
        "Bearer first-async-key",
        "Bearer second-async-key",
    ]
    assert async_client.api_key == ""
