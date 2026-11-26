"""Native DeepSeek contracts that must survive the compatible migration."""

from __future__ import annotations

import asyncio
from dataclasses import fields
from datetime import datetime, timezone
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[2]
for source in (str(PACKAGE_ROOT / "src"), str(REPOSITORY_ROOT / "src")):
    if source not in sys.path:
        sys.path.insert(0, source)

import llm_api_adapter.adapters.base_adapter as base_adapter_module
from llm_api_adapter.adapters.openai_compatible.adapter import OpenAICompatibleAdapter
import llm_api_adapter_deepseek.adapter as deepseek_adapter_module
import llm_api_adapter.universal_adapter as universal_module
from llm_api_adapter.llm_registry.llm_registry import RegistrySpec
from llm_api_adapter.llms.transports import JSONResponse, SSEEvent, TransportRequest
from llm_api_adapter.models.messages.chat_message import UserMessage
from llm_api_adapter.service_provider_registry import ServiceProviderRegistry
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter
from llm_api_adapter_deepseek.clients.async_client import (
    DeepSeekAsyncClient,
    DeepSeekResponsesAsyncClient,
)
from llm_api_adapter_deepseek.adapter import DeepSeekAdapter
from llm_api_adapter_deepseek.clients.sync_client import (
    DEEPSEEK_RESPONSES_URL,
    DeepSeekResponsesSyncClient,
    DeepSeekSyncClient,
)
from llm_api_adapter_deepseek.plugin import PLUGIN

pytestmark = pytest.mark.unit


@pytest.fixture
def deepseek_runtime(monkeypatch):
    model_registry = RegistrySpec()
    assert PLUGIN.model_metadata is not None
    assert model_registry.register_organization_metadata(PLUGIN.model_metadata)
    provider_registry = ServiceProviderRegistry()
    PLUGIN.register(provider_registry)
    monkeypatch.setattr(universal_module, "LLM_REGISTRY", model_registry)
    monkeypatch.setattr(universal_module, "SERVICE_PROVIDER_REGISTRY", provider_registry)
    monkeypatch.setattr(base_adapter_module, "LLM_REGISTRY", model_registry)
    return model_registry


def _adapter() -> UniversalLLMAPIAdapter:
    return UniversalLLMAPIAdapter(
        organization="deepseek",
        model="deepseek-flash",
        api_key="test-key",
    )


def _priced_deepseek_response(response_id: str, text: str) -> dict:
    return {
        "object": "response",
        "id": response_id,
        "model": "deepseek-flash",
        "status": "completed",
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text}],
        }],
        "usage": {
            "input_tokens": 100,
            "input_tokens_details": {"cached_tokens": 25},
            "output_tokens": 40,
            "total_tokens": 140,
        },
    }


def test_native_constructor_fields_aliases_and_patch_points(monkeypatch):
    """Keep native constructor layout and long-standing import aliases stable."""
    assert DeepSeekSyncClient is DeepSeekResponsesSyncClient
    assert DeepSeekAsyncClient is DeepSeekResponsesAsyncClient
    assert [item.name for item in fields(DeepSeekResponsesSyncClient)] == [
        "api_key", "transport", "endpoint", "_sync_transport",
    ]
    assert [item.name for item in fields(DeepSeekResponsesAsyncClient)] == [
        "api_key", "endpoint",
    ]
    assert list(inspect.signature(DeepSeekResponsesSyncClient).parameters) == [
        "api_key", "transport", "endpoint",
    ]
    assert list(inspect.signature(DeepSeekResponsesAsyncClient).parameters) == [
        "api_key", "endpoint",
    ]
    adapter_parameters = inspect.signature(DeepSeekAdapter).parameters
    assert DeepSeekAdapter.__bases__ == (OpenAICompatibleAdapter,)
    assert [
        (parameter.name, parameter.kind, parameter.default)
        for parameter in adapter_parameters.values()
    ] == [
        ("api_key", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
        ("model", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
        ("company", inspect.Parameter.POSITIONAL_OR_KEYWORD, "deepseek"),
        ("transport", inspect.Parameter.POSITIONAL_OR_KEYWORD, "requests"),
        ("pricing", inspect.Parameter.POSITIONAL_OR_KEYWORD, None),
        ("is_reasoning", inspect.Parameter.POSITIONAL_OR_KEYWORD, False),
        ("is_adaptive_thinking", inspect.Parameter.POSITIONAL_OR_KEYWORD, False),
        ("service_provider", inspect.Parameter.POSITIONAL_OR_KEYWORD, None),
        (
            "endpoint",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            DEEPSEEK_RESPONSES_URL,
        ),
    ]
    assert [item.name for item in fields(DeepSeekAdapter)] == [
        "api_key", "model", "company", "transport", "pricing",
        "is_reasoning", "is_adaptive_thinking", "service_provider", "model_spec",
        "endpoint", "_client", "_async_client",
    ]

    calls = []
    marker = object()

    def make_transport(name):
        calls.append(name)
        return marker

    monkeypatch.setattr(
        "llm_api_adapter_deepseek.clients.sync_client.create_sync_transport",
        make_transport,
    )
    client = DeepSeekResponsesSyncClient("test-key")
    assert calls == ["requests"]
    assert client._sync_transport is marker


def test_public_chat_uses_inherited_responses_execution(deepseek_runtime, monkeypatch):
    adapter = _adapter().adapter
    requests = []
    inherited_calls = []
    preparation_calls = []
    core_preparation_calls = []
    normalization_calls = []
    original_prepare = adapter._prepare_responses_parameters
    original_core_prepare = base_adapter_module.LLMAdapterBase._prepare_chat_request
    original_normalize = adapter._normalize_responses_response
    inherited_chat = OpenAICompatibleAdapter.chat

    def prepare(**kwargs):
        preparation_calls.append(True)
        return original_prepare(**kwargs)

    def prepare_core(self, *args, **kwargs):
        core_preparation_calls.append(True)
        return original_core_prepare(self, *args, **kwargs)

    def normalize(response, *, capture_reasoning):
        normalization_calls.append(True)
        return original_normalize(
            response,
            capture_reasoning=capture_reasoning,
        )

    monkeypatch.setattr(adapter, "_prepare_responses_parameters", prepare)
    monkeypatch.setattr(adapter, "_normalize_responses_response", normalize)
    monkeypatch.setattr(
        base_adapter_module.LLMAdapterBase,
        "_prepare_chat_request",
        prepare_core,
    )

    def inherited_executor(self, *args, **kwargs):
        inherited_calls.append(True)
        return inherited_chat(self, *args, **kwargs)

    monkeypatch.setattr(OpenAICompatibleAdapter, "chat", inherited_executor)

    def create(**kwargs):
        requests.append(kwargs)
        return {
            "object": "response",
            "id": "resp-sync",
            "model": "deepseek-flash",
            "status": "completed",
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hello"}],
            }],
            "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        }

    monkeypatch.setattr(adapter._client, "create", create)
    response = adapter.chat([UserMessage("hello")])

    assert response.content == "hello"
    assert inherited_calls == [True]
    assert preparation_calls == [True]
    assert core_preparation_calls == [True]
    assert normalization_calls == [True]
    assert requests == [{
        "model": "deepseek-flash",
        "timeout": None,
        "input": [{"role": "user", "content": "hello"}],
        "temperature": 1.0,
        "top_p": 1.0,
    }]


def test_completed_preflight_errors_stay_outside_inherited_error_handler(
    deepseek_runtime,
    monkeypatch,
):
    adapter = _adapter().adapter
    errors = []

    def handle_error(*args, **kwargs):
        errors.append((args, kwargs))
        raise AssertionError("native preflight must happen before inherited handling")

    monkeypatch.setattr(adapter, "handle_error", handle_error)
    with pytest.raises(NotImplementedError, match="parallel_tool_calls"):
        adapter.chat(
            [UserMessage("hello")],
            parallel_tool_calls=True,
        )
    assert errors == []


def test_public_achat_uses_inherited_responses_execution(deepseek_runtime, monkeypatch):
    adapter = _adapter().adapter
    requests = []
    inherited_calls = []
    inherited_achat = OpenAICompatibleAdapter.achat

    async def inherited_executor(self, *args, **kwargs):
        inherited_calls.append(True)
        return await inherited_achat(self, *args, **kwargs)

    monkeypatch.setattr(OpenAICompatibleAdapter, "achat", inherited_executor)

    async def create(**kwargs):
        requests.append(kwargs)
        return {
            "object": "response",
            "id": "resp-async",
            "model": "deepseek-flash",
            "status": "completed",
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hello async"}],
            }],
            "usage": {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6},
        }

    monkeypatch.setattr(adapter._async_client, "create", create)

    async def run():
        return await adapter.achat([UserMessage("hello")])

    response = asyncio.run(run())

    assert response.content == "hello async"
    assert inherited_calls == [True]
    assert requests == [{
        "model": "deepseek-flash",
        "timeout": None,
        "input": [{"role": "user", "content": "hello"}],
        "temperature": 1.0,
        "top_p": 1.0,
    }]


def test_achat_cancellation_restores_invocation_and_dispatch_context(
    deepseek_runtime,
    monkeypatch,
):
    adapter = _adapter().adapter
    started = asyncio.Event()
    prior_dispatch = datetime(2026, 9, 15, 8, 0, tzinfo=timezone.utc)
    dispatch_token = deepseek_adapter_module._DEEPSEEK_DISPATCH_TIME.set(
        prior_dispatch,
    )

    async def create(**kwargs):
        del kwargs
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(adapter._async_client, "create", create)

    async def run():
        current = asyncio.current_task()

        async def cancel_request():
            await started.wait()
            current.cancel()

        cancellation = asyncio.create_task(cancel_request())
        with pytest.raises(asyncio.CancelledError):
            await adapter.achat([UserMessage("cancel me")])
        await cancellation
        assert deepseek_adapter_module._DEEPSEEK_COMPLETED_INVOCATION.get() is None
        assert deepseek_adapter_module._DEEPSEEK_DISPATCH_TIME.get() == prior_dispatch

    try:
        asyncio.run(run())
    finally:
        deepseek_adapter_module._DEEPSEEK_DISPATCH_TIME.reset(dispatch_token)


def test_nested_completed_calls_keep_dispatch_pricing_isolated(
    deepseek_runtime,
    monkeypatch,
):
    adapter = _adapter().adapter
    dispatch_times = iter((
        datetime(2026, 9, 16, 2, 0, tzinfo=timezone.utc),
        datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc),
    ))
    monkeypatch.setattr(
        deepseek_adapter_module,
        "_utc_now",
        lambda: next(dispatch_times),
    )
    responses = []
    inner_responses = []
    pricing_times = []
    original_multiplier = adapter._dispatch_pricing_multiplier

    def record_multiplier(dispatch_time):
        pricing_times.append(dispatch_time)
        return original_multiplier(dispatch_time)

    monkeypatch.setattr(adapter, "_dispatch_pricing_multiplier", record_multiplier)

    def create(**kwargs):
        del kwargs
        if not responses:
            responses.append("outer")
            outer = deepseek_adapter_module._DEEPSEEK_COMPLETED_INVOCATION.get()
            inner_response = adapter.chat([UserMessage("nested")])
            inner_responses.append(inner_response)
            assert inner_response.content == "inner"
            assert outer.dispatch_time == datetime(
                2026, 9, 16, 2, 0, tzinfo=timezone.utc,
            )
            return _priced_deepseek_response("outer", "outer")
        return _priced_deepseek_response("inner", "inner")

    monkeypatch.setattr(adapter._client, "create", create)
    outer_response = adapter.chat([UserMessage("outer")])

    peak_input = (25 * 0.006 + 75 * 0.3) / 1_000_000
    off_peak_input = (25 * 0.003 + 75 * 0.15) / 1_000_000
    assert outer_response.content == "outer"
    assert pricing_times == [
        datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc),
        datetime(2026, 9, 16, 2, 0, tzinfo=timezone.utc),
    ]
    assert inner_responses[0].cost_input == pytest.approx(off_peak_input)
    assert inner_responses[0].cost_total == pytest.approx(
        off_peak_input + 40 * 0.6 / 1_000_000,
    )
    assert outer_response.cost_input == pytest.approx(peak_input)
    assert outer_response.cost_total == pytest.approx(
        peak_input + 40 * 1.2 / 1_000_000,
    )


def test_stream_inside_completed_send_keeps_its_dispatch_time(
    deepseek_runtime,
    monkeypatch,
):
    adapter = _adapter().adapter
    peak_time = datetime(2026, 9, 16, 2, 0, tzinfo=timezone.utc)
    off_peak_time = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)
    dispatch_times = iter((peak_time, off_peak_time))
    monkeypatch.setattr(
        deepseek_adapter_module,
        "_utc_now",
        lambda: next(dispatch_times),
    )

    stream_response = _priced_deepseek_response("stream", "stream")
    monkeypatch.setattr(
        adapter._client,
        "stream",
        lambda **kwargs: iter((
            SSEEvent(
                event="response.output_text.delta",
                data={"type": "response.output_text.delta", "delta": "stream"},
            ),
            SSEEvent(
                event="response.completed",
                data={"type": "response.completed", "response": stream_response},
            ),
        )),
    )
    streamed = []

    def create(**kwargs):
        del kwargs
        list(adapter.stream_chat(
            [UserMessage("nested stream")],
            on_done=streamed.append,
        ))
        assert deepseek_adapter_module._DEEPSEEK_DISPATCH_TIME.get() is None
        return _priced_deepseek_response("completed", "completed")

    monkeypatch.setattr(adapter._client, "create", create)
    completed = adapter.chat([UserMessage("completed")])

    peak_input = (25 * 0.006 + 75 * 0.3) / 1_000_000
    off_peak_input = (25 * 0.003 + 75 * 0.15) / 1_000_000
    assert len(streamed) == 1
    assert streamed[0].cost_input == pytest.approx(off_peak_input)
    assert completed.cost_input == pytest.approx(peak_input)


def test_chat_inside_stream_callback_restores_stream_dispatch_time(
    deepseek_runtime,
    monkeypatch,
):
    adapter = _adapter().adapter
    peak_time = datetime(2026, 9, 16, 2, 0, tzinfo=timezone.utc)
    off_peak_time = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)
    dispatch_times = iter((peak_time, off_peak_time))
    monkeypatch.setattr(
        deepseek_adapter_module,
        "_utc_now",
        lambda: next(dispatch_times),
    )

    monkeypatch.setattr(
        adapter._client,
        "stream",
        lambda **kwargs: iter((
            SSEEvent(
                event="response.output_text.delta",
                data={"type": "response.output_text.delta", "delta": "stream"},
            ),
            SSEEvent(
                event="response.completed",
                data={
                    "type": "response.completed",
                    "response": _priced_deepseek_response("stream", "stream"),
                },
            ),
        )),
    )
    nested_responses = []
    stream_responses = []
    restored_times = []

    def create(**kwargs):
        del kwargs
        return _priced_deepseek_response("nested", "nested")

    monkeypatch.setattr(adapter._client, "create", create)

    def on_done(stream_response):
        stream_responses.append(stream_response)
        nested_responses.append(adapter.chat([UserMessage("nested")]))
        restored_times.append(deepseek_adapter_module._DEEPSEEK_DISPATCH_TIME.get())
        assert stream_response.content == "stream"

    list(adapter.stream_chat([UserMessage("stream")], on_done=on_done))

    peak_input = (25 * 0.006 + 75 * 0.3) / 1_000_000
    off_peak_input = (25 * 0.003 + 75 * 0.15) / 1_000_000
    assert stream_responses[0].cost_input == pytest.approx(peak_input)
    assert nested_responses[0].cost_input == pytest.approx(off_peak_input)
    assert restored_times == [peak_time]


def test_native_client_preserves_explicit_full_resource_and_wire_defaults():
    class CaptureTransport:
        def __init__(self):
            self.request = None

        def post_json(self, request, *, http_error_handler=None):
            del http_error_handler
            self.request = request
            return JSONResponse({"object": "response"})

        def post_multipart(self, *args, **kwargs):
            raise AssertionError("text generation must not upload")

        def post_sse(self, *args, **kwargs):
            raise AssertionError("not streaming")

    transport = CaptureTransport()
    client = DeepSeekResponsesSyncClient(
        "test-key",
        transport="requests",
        endpoint="https://gateway.example/v1/responses",
    )
    client._sync_transport = transport
    assert client.create(model="deepseek-flash", input=[]) == {"object": "response"}
    request = transport.request
    assert isinstance(request, TransportRequest)
    assert request.url == "https://gateway.example/v1/responses"
    assert request.payload == {"model": "deepseek-flash", "input": []}
    assert request.headers_dict() == {
        "Authorization": "Bearer test-key",
        "Content-Type": "application/json",
    }


def test_sync_stream_helper_does_not_open_http_until_consumed(monkeypatch):
    helper_calls = []
    http_calls = []

    class EmptyResponse:
        def raise_for_status(self):
            return None

        def iter_lines(self):
            return iter(())

        def close(self):
            http_calls.append("closed")

    def post(url, **kwargs):
        http_calls.append((url, kwargs))
        return EmptyResponse()

    monkeypatch.setattr("llm_api_adapter.llms.requests_transport.requests.post", post)
    from llm_api_adapter.llms.requests_transport import RequestsSyncTransport

    transport = RequestsSyncTransport()
    real_post_sse = transport.post_sse

    def post_sse(request, *, http_error_handler=None, stream_error_handler=None):
        helper_calls.append(request)
        return real_post_sse(
            request,
            http_error_handler=http_error_handler,
            stream_error_handler=stream_error_handler,
        )

    transport.post_sse = post_sse
    client = DeepSeekResponsesSyncClient("test-key")
    client._sync_transport = transport
    events = client.stream(model="deepseek-flash", input=[])

    # The native helper runs immediately, while its Requests generator opens on pull.
    assert len(helper_calls) == 1
    assert helper_calls[0].url == DEEPSEEK_RESPONSES_URL
    assert http_calls == []
    with pytest.raises(StopIteration):
        next(events)
    assert http_calls[0][0] == DEEPSEEK_RESPONSES_URL
    assert http_calls[0][1]["json"] == {
        "model": "deepseek-flash", "input": [], "stream": True,
    }
    assert http_calls[0][1]["stream"] is True
    assert http_calls[-1] == "closed"


def test_public_stream_modes_are_lazy_and_preflight_retains_first_error(
    deepseek_runtime,
    monkeypatch,
):
    from llm_api_adapter_deepseek.clients import async_client as native_async_module

    def unexpected_sync_http(*args, **kwargs):
        raise AssertionError("facade construction or preflight opened sync HTTP")

    def unexpected_async_helper(*args, **kwargs):
        raise AssertionError("facade construction or preflight opened async HTTP")

    monkeypatch.setattr(
        "llm_api_adapter.llms.requests_transport.requests.post",
        unexpected_sync_http,
    )
    monkeypatch.setattr(native_async_module, "async_stream_request", unexpected_async_helper)
    adapter = _adapter()
    calls = []

    class NoIOClient:
        def stream(self, **kwargs):
            calls.append(kwargs)
            raise AssertionError("invalid preflight must not open a provider stream")

    adapter.adapter._client = NoIOClient()
    adapter.adapter._async_client = NoIOClient()
    stream = adapter.stream_chat(
        messages=[UserMessage("hello")],
        temperature=2.0,
        parallel_tool_calls=True,
    )
    # Facade generator construction itself performs no validation or I/O.
    assert iter(stream) is stream
    with pytest.raises(NotImplementedError, match="parallel_tool_calls"):
        next(stream)

    async_stream = adapter.astream_chat(
        messages=[UserMessage("hello")],
        temperature=2.0,
        parallel_tool_calls=True,
    )

    async def consume_invalid_stream():
        with pytest.raises(NotImplementedError, match="parallel_tool_calls"):
            await async_stream.__anext__()

    asyncio.run(consume_invalid_stream())
    assert calls == []


def test_native_async_module_aliases_remain_patchable(monkeypatch):
    """Async calls keep dispatch in the native module's two historical seams."""
    from llm_api_adapter_deepseek.clients import async_client as module

    records = []

    async def request(url, **kwargs):
        records.append(("json", url, kwargs))
        return {"object": "response"}

    def stream_request(url, **kwargs):
        records.append(("sse", url, kwargs))

        async def events():
            yield SSEEvent(event="response.completed", data={"type": "response.completed"})

        return events()

    monkeypatch.setattr(module, "async_request", request)
    monkeypatch.setattr(module, "async_stream_request", stream_request)
    endpoint = "https://gateway.example/v1/responses"
    client = DeepSeekResponsesAsyncClient("test-key", endpoint=endpoint)

    async def run():
        result = await client.create(model="deepseek-flash", input=[])
        stream = client.stream(model="deepseek-flash", input=[])
        assert [record[0] for record in records] == ["json", "sse"]
        events = [event async for event in stream]
        return result, events

    result, events = asyncio.run(run())
    assert result == {"object": "response"}
    assert len(events) == 1
    assert [record[0] for record in records] == ["json", "sse"]
    assert all(record[1] == endpoint for record in records)


def test_async_stream_helper_runs_before_http_send_and_uses_full_endpoint(monkeypatch):
    from llm_api_adapter.llms import async_streaming
    from llm_api_adapter_deepseek.clients import async_client as native_module

    endpoint = "https://gateway.example/prefix/responses"
    helper_calls = []
    sends = []
    real_helper = async_streaming.async_stream_request

    class Response:
        def raise_for_status(self):
            return None

        async def aiter_lines(self):
            yield "event: response.output_text.delta"
            yield 'data: {"type":"response.output_text.delta","delta":"hello"}'
            yield ""

        async def aclose(self):
            return None

    class AsyncClient:
        def build_request(self, method, url, *, headers, json, timeout):
            return (method, url, headers, json, timeout)

        async def send(self, request, *, stream):
            sends.append((request, stream))
            return Response()

        async def aclose(self):
            return None

    monkeypatch.setattr(
        async_streaming,
        "_require_httpx",
        lambda: SimpleNamespace(AsyncClient=AsyncClient),
    )

    def helper_spy(url, **kwargs):
        helper_calls.append((url, kwargs))
        return real_helper(url, **kwargs)

    monkeypatch.setattr(native_module, "async_stream_request", helper_spy)
    client = DeepSeekResponsesAsyncClient("test-key", endpoint=endpoint)
    events = client.stream(model="deepseek-flash", input=[])

    assert len(helper_calls) == 1
    assert helper_calls[0][0] == endpoint
    assert sends == []

    async def consume():
        return [event async for event in events]

    received = asyncio.run(consume())
    assert len(received) == 1
    assert sends[0][0][1] == endpoint
    assert sends[0][0][3] == {
        "model": "deepseek-flash", "input": [], "stream": True,
    }
    assert sends[0][1] is True


def test_stream_pricing_uses_dispatch_time_across_peak_boundary(
    deepseek_runtime,
    monkeypatch,
):
    adapter = _adapter()
    dispatch_time = datetime(2026, 9, 16, 2, 0, tzinfo=timezone.utc)
    completion_time = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)
    clock = [dispatch_time]
    monkeypatch.setattr(deepseek_adapter_module, "_utc_now", lambda: clock[0])

    response = {
        "object": "response",
        "id": "stream-deepseek-flash",
        "model": "deepseek-flash",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "hello"}],
            },
        ],
        "usage": {
            "input_tokens": 100,
            "input_tokens_details": {"cached_tokens": 25},
            "output_tokens": 40,
            "total_tokens": 140,
        },
    }

    class BoundaryTransport:
        def post_json(self, *args, **kwargs):
            raise AssertionError("stream expected")

        def post_multipart(self, *args, **kwargs):
            raise AssertionError("text generation must not upload")

        def post_sse(self, request, *, http_error_handler=None, stream_error_handler=None):
            del request, http_error_handler, stream_error_handler

            def events():
                yield SSEEvent(
                    event="response.output_text.delta",
                    data={"type": "response.output_text.delta", "delta": "hello"},
                )
                clock[0] = completion_time
                yield SSEEvent(
                    event="response.completed",
                    data={"type": "response.completed", "response": response},
                )

            return events()

    adapter.adapter._client._sync_transport = BoundaryTransport()
    completed = []
    list(
        adapter.stream_chat(
            messages=[UserMessage("price this")],
            on_done=completed.append,
        ),
    )

    assert clock[0] == completion_time
    assert len(completed) == 1
    assert completed[0].currency == "USD"
    # 02:00 UTC is peak; completion occurs after the 12:00 off-peak boundary.
    assert completed[0].cost_input == pytest.approx(
        (25 * 0.006 + 75 * 0.3) / 1_000_000,
    )
    assert completed[0].cost_output == pytest.approx(40 * 1.2 / 1_000_000)
    assert completed[0].cost_total == pytest.approx(
        (25 * 0.006 + 75 * 0.3 + 40 * 1.2) / 1_000_000,
    )


def test_sync_callback_failure_remains_caller_error_and_closes_stream(deepseek_runtime):
    """A callback exception is not remapped as a provider failure."""
    adapter = _adapter()

    class Transport:
        closed = False

        def post_json(self, *args, **kwargs):
            raise AssertionError("stream expected")

        def post_multipart(self, *args, **kwargs):
            raise AssertionError("text generation must not upload")

        def post_sse(self, request, *, http_error_handler=None, stream_error_handler=None):
            del request, http_error_handler, stream_error_handler

            def events():
                try:
                    yield SSEEvent(
                        event="response.output_text.delta",
                        data={"type": "response.output_text.delta", "delta": "hi"},
                    )
                finally:
                    self.closed = True

            return events()

    transport = Transport()
    adapter.adapter._client._sync_transport = transport
    failure = RuntimeError("caller callback failed")

    def on_delta(_text):
        raise failure

    with pytest.raises(RuntimeError) as caught:
        list(adapter.stream_chat(messages=[UserMessage("hello")], on_delta=on_delta))
    assert caught.value is failure
    assert transport.closed is True

