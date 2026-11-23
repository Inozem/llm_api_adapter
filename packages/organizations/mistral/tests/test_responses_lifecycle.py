"""Native Mistral Responses lifecycle and diagnostic regression tests."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator, Mapping
import sys
from pathlib import Path
from typing import Any

import pytest
import requests


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[2]
for source in (str(PACKAGE_ROOT / "src"), str(REPOSITORY_ROOT / "src"), str(REPOSITORY_ROOT)):
    if source not in sys.path:
        sys.path.insert(0, source)

import llm_api_adapter.adapters.base_adapter as base_adapter_module
import llm_api_adapter.llms.requests_transport as requests_transport_module
import llm_api_adapter_mistral.clients.async_client as mistral_async_client_module
from llm_api_adapter.errors.llm_api_error import (
    LLMAPIAuthorizationError,
    LLMAPIClientError,
    LLMAPIRateLimitError,
)
from llm_api_adapter.llm_registry.llm_registry import RegistrySpec
from llm_api_adapter.llms.transports import (
    HTTPErrorHandler,
    JSONResponse,
    MultipartForm,
    SSEEvent,
    StreamErrorHandler,
    SyncTransport,
    TransportRequest,
)
from llm_api_adapter.models.responses.chat_response import ChatResponse
from llm_api_adapter_mistral.adapter import MistralAdapter
from llm_api_adapter_mistral.plugin import PLUGIN


MODEL = "mistral-large-4"
API_KEY = "mistral-lifecycle-secret-key"
QUERY_SECRET = "url-token-lifecycle-secret"
REASONING_MARKER = "synthetic-reasoning-marker"
TOOL_ARGUMENT_MARKER = "synthetic-tool-argument-marker"
DOCUMENT_MARKER = "synthetic-document-marker"
MESSAGE = [{"role": "user", "content": "Answer briefly."}]


@pytest.fixture
def mistral_registry(monkeypatch: pytest.MonkeyPatch) -> RegistrySpec:
    registry = RegistrySpec()
    assert PLUGIN.model_metadata is not None
    assert registry.register_organization_metadata(PLUGIN.model_metadata)
    monkeypatch.setattr(base_adapter_module, "LLM_REGISTRY", registry)
    return registry


def new_adapter() -> MistralAdapter:
    return MistralAdapter(api_key=API_KEY, model=MODEL, company="mistral")


def response_event(event_name: str, payload: Mapping[str, Any]) -> SSEEvent:
    return SSEEvent(event=event_name, data=dict(payload))


class ErrorResponse:
    """Minimal HTTP response that raises the configured requests exception."""

    status_code: int
    message: str

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        self.message = message

    def json(self) -> dict[str, Any]:
        return {"error": {"type": "authentication_error", "message": self.message}}

    def raise_for_status(self) -> None:
        error = requests.HTTPError(self.message, response=self)
        raise error

    def close(self) -> None:
        return None


class FailingSSETransport(SyncTransport):
    """Pass a provider error through the real native callback boundary."""

    event: SSEEvent
    closed: bool

    def __init__(self, event: SSEEvent) -> None:
        self.event = event
        self.closed = False

    def post_json(
        self, request: TransportRequest, *, http_error_handler: HTTPErrorHandler | None = None
    ) -> JSONResponse:
        raise AssertionError("lifecycle stream tests do not submit JSON")

    def post_multipart(
        self,
        request: TransportRequest,
        form: MultipartForm,
        *,
        http_error_handler: HTTPErrorHandler | None = None,
    ) -> JSONResponse:
        raise AssertionError("lifecycle stream tests do not submit multipart")

    def post_sse(
        self,
        request: TransportRequest,
        *,
        http_error_handler: HTTPErrorHandler | None = None,
        stream_error_handler: StreamErrorHandler | None = None,
    ) -> Iterator[SSEEvent]:
        def events() -> Iterator[SSEEvent]:
            try:
                yield response_event("response.output_text.delta", {"delta": "partial"})
                if self.event.event != "response.failed":
                    assert stream_error_handler is not None
                    stream_error_handler(self.event)
                yield self.event
            finally:
                self.closed = True

        return events()


@pytest.mark.integration
@pytest.mark.parametrize("method", ["stream_chat", "astream_chat"])
def test_native_stream_eof_without_terminal_never_completes_or_flushes(
    mistral_registry: RegistrySpec,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    adapter = new_adapter()
    visible: list[str] = []
    completed: list[ChatResponse] = []
    pending = response_event("response.output_text.delta", {"delta": "ABCDE"})

    if method == "stream_chat":
        transport = FailingSSETransport(SSEEvent(event=None, data={}))

        def source(
            request: TransportRequest,
            *,
            http_error_handler: HTTPErrorHandler | None = None,
            stream_error_handler: StreamErrorHandler | None = None,
        ) -> Iterator[SSEEvent]:
            try:
                yield pending
            finally:
                transport.closed = True

        monkeypatch.setattr(transport, "post_sse", source)
        adapter._sync_transport = transport
        with pytest.raises(LLMAPIClientError):
            list(adapter.stream_chat(MESSAGE, buffer_chars=3, on_chunk=lambda chunk: visible.append(chunk.text), on_done=completed.append))
        assert visible == ["ABC"]
        assert transport.closed
    else:
        closed: list[bool] = []

        def stream_request(
            url: str,
            *,
            headers: Mapping[str, str] | None = None,
            payload: Any = None,
            timeout: float | None = None,
            http_error_handler: HTTPErrorHandler | None = None,
            stream_error_handler: StreamErrorHandler | None = None,
        ) -> AsyncIterator[SSEEvent]:
            async def events() -> AsyncIterator[SSEEvent]:
                try:
                    yield pending
                finally:
                    closed.append(True)

            return events()

        monkeypatch.setattr(mistral_async_client_module, "async_stream_request", stream_request)

        async def collect() -> None:
            with pytest.raises(LLMAPIClientError):
                async for _ in adapter.astream_chat(
                    MESSAGE,
                    buffer_chars=3,
                    on_chunk=lambda chunk: visible.append(chunk.text),
                    on_done=completed.append,
                ):
                    pass

        asyncio.run(collect())
        assert visible == ["ABC"]
        assert closed == [True]

    assert completed == []


@pytest.mark.integration
@pytest.mark.parametrize("streaming", [False, True])
def test_sync_http_diagnostics_redact_native_credentials(
    mistral_registry: RegistrySpec,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    streaming: bool,
) -> None:
    adapter = new_adapter()
    message = f"{API_KEY} https://example.invalid/fail?token={QUERY_SECRET}"
    response = ErrorResponse(403, message)
    calls: list[str] = []

    def post(url: str, **kwargs: Any) -> ErrorResponse:
        calls.append(url)
        response.message = message
        return response

    monkeypatch.setattr(requests_transport_module.requests, "post", post)
    with pytest.raises(LLMAPIAuthorizationError) as caught:
        if streaming:
            list(adapter.stream_chat(MESSAGE))
        else:
            adapter.chat(MESSAGE)

    assert len(calls) == 1
    assert API_KEY not in caplog.text
    assert QUERY_SECRET not in caplog.text
    assert API_KEY not in str(caught.value)
    assert QUERY_SECRET not in str(caught.value)
    assert "[redacted URL]" in caplog.text


@pytest.mark.integration
def test_sync_failed_stream_maps_category_and_redacts_provider_detail(
    mistral_registry: RegistrySpec,
) -> None:
    adapter = new_adapter()
    detail = f"{API_KEY} https://example.invalid/fail?token={QUERY_SECRET}"
    provider_error = response_event(
        "response.failed",
        {"response": {"error": {"type": "rate_limit_error", "message": detail}}},
    )
    transport = FailingSSETransport(provider_error)
    adapter._sync_transport = transport
    completed: list[ChatResponse] = []

    with pytest.raises(LLMAPIRateLimitError) as caught:
        list(adapter.stream_chat(MESSAGE, on_done=completed.append))

    assert completed == []
    assert transport.closed
    assert API_KEY not in str(caught.value)
    assert QUERY_SECRET not in str(caught.value)


@pytest.mark.integration
def test_async_failed_stream_maps_category_and_redacts_provider_detail(
    mistral_registry: RegistrySpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = new_adapter()
    detail = f"{API_KEY} https://example.invalid/fail?token={QUERY_SECRET}"
    provider_error = response_event(
        "response.failed",
        {"response": {"error": {"type": "rate_limit_error", "message": detail}}},
    )
    closed: list[bool] = []

    def stream_request(
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        payload: Any = None,
        timeout: float | None = None,
        http_error_handler: HTTPErrorHandler | None = None,
        stream_error_handler: StreamErrorHandler | None = None,
    ) -> AsyncIterator[SSEEvent]:
        async def events() -> AsyncIterator[SSEEvent]:
            try:
                yield response_event(
                    "response.output_text.delta", {"delta": "pending text"}
                )
                yield provider_error
            finally:
                closed.append(True)

        return events()

    monkeypatch.setattr(mistral_async_client_module, "async_stream_request", stream_request)

    async def collect() -> None:
        with pytest.raises(LLMAPIRateLimitError) as caught:
            async for _ in adapter.astream_chat(MESSAGE):
                pass
        assert API_KEY not in str(caught.value)
        assert QUERY_SECRET not in str(caught.value)

    asyncio.run(collect())
    assert closed == [True]


@pytest.mark.integration
@pytest.mark.parametrize("method", ["stream_chat", "astream_chat"])
@pytest.mark.parametrize(
    "message",
    [
        None,
        {
            "reasoning": REASONING_MARKER,
            "tool_arguments": TOOL_ARGUMENT_MARKER,
            "document": DOCUMENT_MARKER,
        },
    ],
)
def test_response_failed_without_scalar_message_does_not_render_raw_event(
    mistral_registry: RegistrySpec,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    method: str,
    message: object,
) -> None:
    adapter = new_adapter()
    error_data: dict[str, Any] = {"type": "rate_limit_error"}
    if message is not None:
        error_data["message"] = message
    provider_error = response_event(
        "response.failed",
        {"response": {"error": error_data, "output": []}},
    )
    completed: list[ChatResponse] = []
    error_text = ""
    chunks: list[str] = []

    if method == "stream_chat":
        transport = FailingSSETransport(provider_error)
        adapter._sync_transport = transport
        with pytest.raises(LLMAPIRateLimitError) as caught:
            list(
                adapter.stream_chat(
                    MESSAGE,
                    buffer_chars=100,
                    on_chunk=lambda chunk: chunks.append(chunk.text),
                    on_done=completed.append,
                )
            )
        assert transport.closed
        error_text = str(caught.value)
    else:
        closed: list[bool] = []

        def stream_request(
            url: str,
            *,
            headers: Mapping[str, str] | None = None,
            payload: Any = None,
            timeout: float | None = None,
            http_error_handler: HTTPErrorHandler | None = None,
            stream_error_handler: StreamErrorHandler | None = None,
        ) -> AsyncIterator[SSEEvent]:
            async def events() -> AsyncIterator[SSEEvent]:
                try:
                    yield response_event(
                        "response.output_text.delta", {"delta": "pending text"}
                    )
                    yield provider_error
                finally:
                    closed.append(True)

            return events()

        monkeypatch.setattr(mistral_async_client_module, "async_stream_request", stream_request)

        async def collect() -> None:
            nonlocal error_text
            with pytest.raises(LLMAPIRateLimitError) as async_caught:
                async for _ in adapter.astream_chat(
                    MESSAGE,
                    buffer_chars=100,
                    on_chunk=lambda chunk: chunks.append(chunk.text),
                    on_done=completed.append,
                ):
                    pass
            error_text = str(async_caught.value)

        asyncio.run(collect())
        assert closed == [True]

    assert completed == []
    assert chunks == []
    assert API_KEY not in error_text
    assert QUERY_SECRET not in error_text
    assert REASONING_MARKER not in error_text
    assert TOOL_ARGUMENT_MARKER not in error_text
    assert DOCUMENT_MARKER not in error_text
    assert API_KEY not in caplog.text
    assert QUERY_SECRET not in caplog.text
    assert REASONING_MARKER not in caplog.text
    assert TOOL_ARGUMENT_MARKER not in caplog.text
    assert DOCUMENT_MARKER not in caplog.text


@pytest.mark.integration
@pytest.mark.parametrize("streaming", [False, True])
def test_async_http_diagnostics_redact_native_credentials(
    mistral_registry: RegistrySpec,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    streaming: bool,
) -> None:
    import httpx

    adapter = new_adapter()
    message = f"{API_KEY} https://example.invalid/fail?token={QUERY_SECRET}"

    class AsyncErrorResponse:
        status_code = 403
        request: httpx.Request

        def __init__(self, url: str) -> None:
            self.request = httpx.Request("POST", url)

        def raise_for_status(self) -> None:
            response = httpx.Response(403, request=self.request)
            raise httpx.HTTPStatusError(message, request=self.request, response=response)

        def json(self) -> dict[str, Any]:
            return {"error": {"type": "authentication_error", "message": message}}

        async def aread(self) -> None:
            return None

        async def aclose(self) -> None:
            return None

        async def aiter_lines(self) -> AsyncIterator[str]:
            if False:
                yield ""

    class AsyncErrorClient:
        async def post(self, url: str, **kwargs: Any) -> AsyncErrorResponse:
            return AsyncErrorResponse(url)

        def build_request(self, method: str, url: str, **kwargs: Any) -> httpx.Request:
            return httpx.Request(method, url)

        async def send(self, request: httpx.Request, *, stream: bool) -> AsyncErrorResponse:
            return AsyncErrorResponse(str(request.url))

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr(httpx, "AsyncClient", AsyncErrorClient)

    async def call() -> None:
        with pytest.raises(LLMAPIAuthorizationError) as caught:
            if streaming:
                async for _ in adapter.astream_chat(MESSAGE):
                    pass
            else:
                await adapter.achat(MESSAGE)
        assert API_KEY not in str(caught.value)
        assert QUERY_SECRET not in str(caught.value)

    asyncio.run(call())
    assert API_KEY not in caplog.text
    assert QUERY_SECRET not in caplog.text
    assert "[redacted URL]" in caplog.text
