"""Deterministic contract tests for Qwen's Frankfurt Messages adapter."""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[2]
CORE_SOURCE = REPOSITORY_ROOT / "src"
PACKAGE_SOURCE = PACKAGE_ROOT / "src"
for source in (str(PACKAGE_SOURCE), str(CORE_SOURCE), str(REPOSITORY_ROOT)):
    if source not in sys.path:
        sys.path.insert(0, source)

import llm_api_adapter.adapters.base_adapter as base_adapter_module
import llm_api_adapter.universal_adapter as universal_module
from llm_api_adapter.errors.llm_api_error import (
    LLMAPIAuthorizationError,
    LLMAPIClientError,
    LLMAPIRateLimitError,
    LLMAPIServerError,
    LLMAPITimeoutError,
)
from llm_api_adapter.llm_registry.llm_registry import RegistrySpec, resolve_model_spec
from llm_api_adapter.llms.transports import JSONResponse, SSEEvent
from llm_api_adapter.service_provider_registry import ServiceProviderRegistry
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter


class FakeSyncTransport:
    def __init__(self, response, *, error=None, events=None) -> None:
        self.response = response
        self.error = error
        self.events = list(events or [])
        self.requests = []
        self.sse_requests = []
        self.sse_closed = False

    def post_json(self, request, *, http_error_handler=None):
        self.requests.append(request)
        if self.error is not None:
            assert http_error_handler is not None
            http_error_handler(self.error)
        return JSONResponse(self.response)

    def post_sse(
        self,
        request,
        *,
        http_error_handler=None,
        stream_error_handler=None,
    ):
        self.sse_requests.append(request)

        def event_iterator():
            try:
                for event in self.events:
                    payload = event.data if isinstance(event.data, dict) else {}
                    if event.event == "error" or payload.get("type") == "error":
                        assert stream_error_handler is not None
                        stream_error_handler(event)
                    yield event
            finally:
                self.sse_closed = True

        return event_iterator()


class FakeHTTPResponse:
    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
        return self._payload


class FakeHTTPError(Exception):
    def __init__(self, status_code: int, payload: dict) -> None:
        super().__init__(f"HTTP {status_code}")
        self.response = FakeHTTPResponse(status_code, payload)


def qwen_messages_sse_events(model="qwen3.8-max"):
    return [
        SSEEvent(
            event="message_start",
            data={
                "type": "message_start",
                "message": {
                    "id": "msg-qwen-stream-1",
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "usage": {"input_tokens": 5, "output_tokens": 0},
                },
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
                "delta": {"type": "text_delta", "text": "Hal"},
            },
        ),
        SSEEvent(
            event="content_block_delta",
            data={
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "lo"},
            },
        ),
        SSEEvent(
            event="message_delta",
            data={
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"input_tokens": 5, "output_tokens": 2},
            },
        ),
        SSEEvent(event="message_stop", data={"type": "message_stop"}),
    ]


@pytest.fixture
def qwen_runtime(monkeypatch):
    from llm_api_adapter_qwen.plugin import PLUGIN

    model_registry = RegistrySpec()
    assert PLUGIN.model_metadata is not None
    assert model_registry.register_organization_metadata(PLUGIN.model_metadata) is True

    service_provider_registry = ServiceProviderRegistry()
    PLUGIN.register(service_provider_registry)
    monkeypatch.setattr(universal_module, "LLM_REGISTRY", model_registry)
    monkeypatch.setattr(
        universal_module,
        "SERVICE_PROVIDER_REGISTRY",
        service_provider_registry,
    )
    monkeypatch.setattr(base_adapter_module, "LLM_REGISTRY", model_registry)
    return model_registry


@pytest.mark.unit
@pytest.mark.parametrize(
    "model",
    ["qwen3.8-max", "qwen3.8-flash", "qwen3.7-plus", "qwen3.7-flash"],
)
def test_qwen_plugin_registers_each_declared_model(qwen_runtime, model):
    from llm_api_adapter_qwen.adapter import QwenAdapter

    adapter = UniversalLLMAPIAdapter(
        organization="qwen",
        model=model,
        api_key="qwen-test-key",
    )

    assert isinstance(adapter.adapter, QwenAdapter)
    assert adapter.adapter.service_provider == "qwen"
    assert resolve_model_spec(qwen_runtime, "qwen", model) is adapter.adapter.model_spec
    assert adapter.adapter.model_spec is not None
    assert adapter.adapter.model_spec.limits.context_window_tokens == 1_000_000
    assert adapter.adapter.model_spec.limits.max_output_tokens == 131_072


@pytest.mark.integration
@pytest.mark.parametrize(
    "model",
    ["qwen3.8-max", "qwen3.8-flash", "qwen3.7-plus", "qwen3.7-flash"],
)
def test_universal_chat_uses_the_frankfurt_messages_endpoint(qwen_runtime, model):
    adapter = UniversalLLMAPIAdapter(
        organization="qwen",
        model=model,
        api_key="qwen-test-key",
    )
    transport = FakeSyncTransport(
        {
            "id": "msg-qwen-1",
            "model": model,
            "content": [{"type": "text", "text": "Hallo"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }
    )
    adapter.adapter._sync_transport = transport

    response = adapter.chat(
        [
            {"role": "system", "content": "Reply in German."},
            {"role": "user", "content": "Hello"},
        ],
        max_tokens=64,
        temperature=0.7,
        top_p=0.8,
        timeout_s=12.5,
        workspace_id="frankfurt-workspace",
    )

    assert response.content == "Hallo"
    assert response.response_id == "msg-qwen-1"
    assert response.usage is not None
    assert response.usage.total_tokens == 30
    assert response.currency == "CNY"

    request = transport.requests[0]
    assert request.url == (
        "https://frankfurt-workspace.eu-central-1.maas.aliyuncs.com/"
        "apps/anthropic/v1/messages"
    )
    assert request.headers_dict() == {
        "x-api-key": "qwen-test-key",
        "Content-Type": "application/json",
    }
    assert request.timeout == 12.5
    assert request.payload == {
        "model": model,
        "system": "Reply in German.",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 64,
        "temperature": 0.7,
        "top_p": 0.8,
    }


@pytest.mark.unit
@pytest.mark.parametrize("workspace_id", [None, "", "wrong/workspace", "wrong.workspace"])
def test_qwen_rejects_missing_or_malformed_workspace_before_transport(
    qwen_runtime,
    workspace_id,
):
    adapter = UniversalLLMAPIAdapter(
        organization="qwen",
        model="qwen3.8-max",
        api_key="qwen-test-key",
    )
    transport = FakeSyncTransport({})
    adapter.adapter._sync_transport = transport

    with pytest.raises(ValueError, match="workspace_id"):
        adapter.chat(
            [{"role": "user", "content": "Hello"}],
            max_tokens=64,
            workspace_id=workspace_id,
        )

    assert transport.requests == []


@pytest.mark.unit
@pytest.mark.parametrize(
    ("status_code", "error_type", "expected_error"),
    [
        (401, "authentication_error", LLMAPIAuthorizationError),
        (400, "invalid_request_error", LLMAPIClientError),
        (429, "rate_limit_error", LLMAPIRateLimitError),
        (504, "timeout_error", LLMAPITimeoutError),
        (500, "api_error", LLMAPIServerError),
    ],
)
def test_qwen_normalizes_messages_http_failures(
    qwen_runtime,
    status_code,
    error_type,
    expected_error,
):
    adapter = UniversalLLMAPIAdapter(
        organization="qwen",
        model="qwen3.8-max",
        api_key="qwen-test-key",
    )
    adapter.adapter._sync_transport = FakeSyncTransport(
        {},
        error=FakeHTTPError(
            status_code,
            {"error": {"type": error_type, "message": "Qwen test failure"}},
        ),
    )

    with pytest.raises(expected_error, match="Qwen test failure"):
        adapter.chat(
            [{"role": "user", "content": "Hello"}],
            max_tokens=64,
            workspace_id="frankfurt-workspace",
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("payload", "detail"),
    [
        ({"id": "msg-qwen-1", "content": "not an array"}, "response.content"),
        (
            {"content": [{"type": "text", "text": 42}]},
            "text content blocks",
        ),
        (
            {
                "content": [],
                "usage": {"input_tokens": "10", "output_tokens": 20},
            },
            "usage token counts",
        ),
    ],
)
def test_qwen_rejects_malformed_messages_response(qwen_runtime, payload, detail):
    adapter = UniversalLLMAPIAdapter(
        organization="qwen",
        model="qwen3.8-max",
        api_key="qwen-test-key",
    )
    adapter.adapter._sync_transport = FakeSyncTransport(
        payload,
    )

    with pytest.raises(LLMAPIClientError, match=detail):
        adapter.chat(
            [{"role": "user", "content": "Hello"}],
            max_tokens=64,
            workspace_id="frankfurt-workspace",
        )


@pytest.mark.integration
def test_universal_stream_chat_reconstructs_qwen_messages_response(qwen_runtime):
    adapter = UniversalLLMAPIAdapter(
        organization="qwen",
        model="qwen3.8-max",
        api_key="qwen-test-key",
    )
    transport = FakeSyncTransport({}, events=qwen_messages_sse_events())
    adapter.adapter._sync_transport = transport
    callback_order = []
    completed = []

    def on_chunk(chunk):
        callback_order.append(("chunk", chunk.text, chunk.usage.output_tokens))

    def on_delta(text):
        callback_order.append(("delta", text))

    def on_done(response):
        completed.append(response)
        callback_order.append(("done", response.content))

    output = []
    for text in adapter.stream_chat(
        [{"role": "user", "content": "Hello"}],
        max_tokens=64,
        timeout_s=12.5,
        workspace_id="frankfurt-workspace",
        on_chunk=on_chunk,
        on_delta=on_delta,
        on_done=on_done,
    ):
        callback_order.append(("yield", text))
        output.append(text)

    assert output == ["Hal", "lo"]
    assert callback_order == [
        ("chunk", "Hal", 0),
        ("delta", "Hal"),
        ("yield", "Hal"),
        ("chunk", "lo", 0),
        ("delta", "lo"),
        ("yield", "lo"),
        ("done", "Hallo"),
    ]
    assert len(completed) == 1
    assert completed[0].response_id == "msg-qwen-stream-1"
    assert completed[0].usage is not None
    assert completed[0].usage.total_tokens == 7
    assert completed[0].currency == "CNY"
    assert transport.sse_closed is True

    request = transport.sse_requests[0]
    assert request.timeout == 12.5
    assert request.payload["stream"] is True
    assert request.payload["model"] == "qwen3.8-max"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_universal_achat_uses_qwen_async_client(qwen_runtime, monkeypatch):
    from llm_api_adapter_qwen.clients import async_client as async_client_module

    requests = []

    async def fake_async_request(url, **kwargs):
        requests.append((url, kwargs))
        return {
            "id": "msg-qwen-async-1",
            "model": "qwen3.8-max",
            "content": [{"type": "text", "text": "Hallo async"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }

    monkeypatch.setattr(async_client_module, "async_request", fake_async_request)
    adapter = UniversalLLMAPIAdapter(
        organization="qwen",
        model="qwen3.8-max",
        api_key="qwen-test-key",
    )

    response = await adapter.achat(
        [{"role": "user", "content": "Hello"}],
        max_tokens=64,
        timeout_s=12.5,
        workspace_id="frankfurt-workspace",
    )

    assert response.content == "Hallo async"
    assert response.usage is not None
    assert response.usage.total_tokens == 30
    assert response.currency == "CNY"
    assert len(requests) == 1
    assert requests[0][0] == (
        "https://frankfurt-workspace.eu-central-1.maas.aliyuncs.com/"
        "apps/anthropic/v1/messages"
    )
    assert requests[0][1]["headers"] == {
        "x-api-key": "qwen-test-key",
        "Content-Type": "application/json",
    }
    assert requests[0][1]["payload"] == {
        "model": "qwen3.8-max",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 64,
        "temperature": 1.0,
        "top_p": 1.0,
    }
    assert requests[0][1]["timeout"] == 12.5


@pytest.mark.asyncio
@pytest.mark.integration
async def test_universal_astream_chat_matches_sync_lifecycle(qwen_runtime, monkeypatch):
    from llm_api_adapter_qwen.clients import async_client as async_client_module

    requests = []
    closed = False

    def fake_async_stream_request(url, **kwargs):
        requests.append((url, kwargs))

        async def events():
            nonlocal closed
            try:
                for event in qwen_messages_sse_events():
                    yield event
            finally:
                closed = True

        return events()

    monkeypatch.setattr(
        async_client_module,
        "async_stream_request",
        fake_async_stream_request,
    )
    adapter = UniversalLLMAPIAdapter(
        organization="qwen",
        model="qwen3.8-max",
        api_key="qwen-test-key",
    )
    callback_order = []
    completed = []

    async def on_chunk(chunk):
        callback_order.append(("chunk", chunk.text, chunk.usage.output_tokens))

    def on_delta(text):
        callback_order.append(("delta", text))

    async def on_done(response):
        completed.append(response)
        callback_order.append(("done", response.content))

    output = []
    async for text in adapter.astream_chat(
        [{"role": "user", "content": "Hello"}],
        max_tokens=64,
        timeout_s=12.5,
        workspace_id="frankfurt-workspace",
        on_chunk=on_chunk,
        on_delta=on_delta,
        on_done=on_done,
    ):
        callback_order.append(("yield", text))
        output.append(text)

    assert output == ["Hal", "lo"]
    assert callback_order == [
        ("chunk", "Hal", 0),
        ("delta", "Hal"),
        ("yield", "Hal"),
        ("chunk", "lo", 0),
        ("delta", "lo"),
        ("yield", "lo"),
        ("done", "Hallo"),
    ]
    assert len(completed) == 1
    assert completed[0].usage is not None
    assert completed[0].usage.total_tokens == 7
    assert closed is True
    assert requests[0][0].endswith("/apps/anthropic/v1/messages")
    assert requests[0][1]["payload"]["stream"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("workspace_id", [None, "", "wrong/workspace"])
async def test_qwen_async_operations_reject_workspace_before_transport(
    qwen_runtime,
    monkeypatch,
    workspace_id,
):
    from llm_api_adapter_qwen.clients import async_client as async_client_module

    requests = []

    async def fake_async_request(*args, **kwargs):
        requests.append((args, kwargs))
        return {}

    monkeypatch.setattr(async_client_module, "async_request", fake_async_request)
    adapter = UniversalLLMAPIAdapter(
        organization="qwen",
        model="qwen3.8-max",
        api_key="qwen-test-key",
    )

    with pytest.raises(ValueError, match="workspace_id"):
        await adapter.achat(
            [{"role": "user", "content": "Hello"}],
            max_tokens=64,
            workspace_id=workspace_id,
        )
    with pytest.raises(ValueError, match="workspace_id"):
        adapter.stream_chat(
            [{"role": "user", "content": "Hello"}],
            max_tokens=64,
            workspace_id=workspace_id,
        )
    with pytest.raises(ValueError, match="workspace_id"):
        adapter.astream_chat(
            [{"role": "user", "content": "Hello"}],
            max_tokens=64,
            workspace_id=workspace_id,
        )

    assert requests == []


@pytest.mark.unit
def test_qwen_stream_error_is_mapped_and_closes_resources(qwen_runtime):
    adapter = UniversalLLMAPIAdapter(
        organization="qwen",
        model="qwen3.8-max",
        api_key="qwen-test-key",
    )
    transport = FakeSyncTransport(
        {},
        events=[
            SSEEvent(
                event="error",
                data={
                    "type": "error",
                    "error": {
                        "type": "rate_limit_error",
                        "message": "Qwen stream rate limit",
                    },
                },
            )
        ],
    )
    adapter.adapter._sync_transport = transport

    with pytest.raises(LLMAPIRateLimitError, match="Qwen stream rate limit"):
        list(
            adapter.stream_chat(
                [{"role": "user", "content": "Hello"}],
                max_tokens=64,
                workspace_id="frankfurt-workspace",
            )
        )

    assert transport.sse_closed is True


@pytest.mark.asyncio
@pytest.mark.unit
async def test_qwen_async_stream_error_is_mapped_and_closes_resources(
    qwen_runtime,
    monkeypatch,
):
    from llm_api_adapter_qwen.clients import async_client as async_client_module

    stream_closed = False

    def fake_async_stream_request(url, **kwargs):
        async def events():
            nonlocal stream_closed
            try:
                event = SSEEvent(
                    event="error",
                    data={
                        "type": "error",
                        "error": {
                            "type": "rate_limit_error",
                            "message": "Qwen async stream rate limit",
                        },
                    },
                )
                kwargs["stream_error_handler"](event)
                yield event  # pragma: no cover - the error handler always raises
            finally:
                stream_closed = True

        return events()

    monkeypatch.setattr(
        async_client_module,
        "async_stream_request",
        fake_async_stream_request,
    )
    adapter = UniversalLLMAPIAdapter(
        organization="qwen",
        model="qwen3.8-max",
        api_key="qwen-test-key",
    )

    with pytest.raises(LLMAPIRateLimitError, match="Qwen async stream rate limit"):
        [
            text
            async for text in adapter.astream_chat(
                [{"role": "user", "content": "Hello"}],
                max_tokens=64,
                workspace_id="frankfurt-workspace",
            )
        ]

    assert stream_closed is True


@pytest.mark.unit
def test_qwen_stream_close_before_completion_closes_resources(qwen_runtime):
    adapter = UniversalLLMAPIAdapter(
        organization="qwen",
        model="qwen3.8-max",
        api_key="qwen-test-key",
    )
    transport = FakeSyncTransport({}, events=qwen_messages_sse_events())
    adapter.adapter._sync_transport = transport
    completed = []

    stream = adapter.stream_chat(
        [{"role": "user", "content": "Hello"}],
        max_tokens=64,
        workspace_id="frankfurt-workspace",
        on_done=completed.append,
    )
    assert next(stream) == "Hal"
    stream.close()

    assert transport.sse_closed is True
    assert completed == []


@pytest.mark.asyncio
@pytest.mark.unit
async def test_qwen_async_stream_cancellation_closes_resources(qwen_runtime, monkeypatch):
    from llm_api_adapter_qwen.clients import async_client as async_client_module

    stream_closed = False
    stream_entered = asyncio.Event()
    never = asyncio.Event()

    def fake_async_stream_request(url, **kwargs):
        async def events():
            nonlocal stream_closed
            try:
                yield SSEEvent(
                    event="message_start",
                    data={
                        "type": "message_start",
                        "message": {
                            "id": "msg-qwen-cancel",
                            "model": "qwen3.8-max",
                            "content": [],
                            "usage": {"input_tokens": 1, "output_tokens": 0},
                        },
                    },
                )
                yield SSEEvent(
                    event="content_block_start",
                    data={
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    },
                )
                yield SSEEvent(
                    event="content_block_delta",
                    data={
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": "Hal"},
                    },
                )
                stream_entered.set()
                await never.wait()
            finally:
                stream_closed = True

        return events()

    monkeypatch.setattr(
        async_client_module,
        "async_stream_request",
        fake_async_stream_request,
    )
    adapter = UniversalLLMAPIAdapter(
        organization="qwen",
        model="qwen3.8-max",
        api_key="qwen-test-key",
    )
    completed = []
    stream = adapter.astream_chat(
        [{"role": "user", "content": "Hello"}],
        max_tokens=64,
        workspace_id="frankfurt-workspace",
        on_done=completed.append,
    )

    assert await stream.__anext__() == "Hal"
    pending_chunk = asyncio.create_task(stream.__anext__())
    await stream_entered.wait()
    pending_chunk.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending_chunk

    assert stream_closed is True
    assert completed == []
