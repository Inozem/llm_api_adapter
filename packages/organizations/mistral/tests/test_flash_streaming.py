"""Lifecycle contract tests for routed GLM Flash streaming."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[2]
for source in (
    str(PACKAGE_ROOT / "src"),
    str(REPOSITORY_ROOT / "src"),
    str(REPOSITORY_ROOT),
):
    if source not in sys.path:
        sys.path.insert(0, source)

import llm_api_adapter.adapters.base_adapter as base_adapter_module
import llm_api_adapter.universal_adapter as universal_module
import llm_api_adapter_mistral.clients.async_client as mistral_async_client_module
from llm_api_adapter.errors.llm_api_error import LLMAPIClientError
from llm_api_adapter.llm_registry.llm_registry import RegistrySpec
from llm_api_adapter.llms.httpx_transport import HttpxSyncTransport
from llm_api_adapter.llms.requests_transport import RequestsSyncTransport
from llm_api_adapter.llms.transports import JSONResponse, SSEEvent
from llm_api_adapter.models.tools import ToolSpec
from llm_api_adapter.service_provider_registry import ServiceProviderRegistry
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter
from llm_api_adapter_mistral.plugin import PLUGIN


MODEL = "glm-5.3-flash"
WIRE_MODEL = "zai-glm-5-3-flash"
MESSAGES = [{"role": "user", "content": "Answer briefly."}]
LOOKUP_TOOL = ToolSpec(
    name="lookup",
    description="Look up a value.",
    json_schema={"type": "object", "properties": {"key": {"type": "string"}}},
)


def event(content, *, tool_calls=None, finish_reason=None, metadata=False):
    payload = {
        "choices": [{
            "index": 0,
            "delta": {"content": content, **({"tool_calls": tool_calls} if tool_calls else {})},
            "finish_reason": finish_reason,
        }]
    }
    if metadata:
        payload.update({"id": "flash-stream-1", "model": WIRE_MODEL})
    return SSEEvent(event=None, data=payload)


def successful_events():
    return [
        event(
            [
                {"type": "thinking", "thinking": [{"type": "text", "text": "Private thought."}]},
                {"type": "text", "text": "AB"},
            ],
            tool_calls=[{
                "index": 0,
                "id": "call-1",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"key":'},
            }],
            metadata=True,
        ),
        event(
            "CDE",
            tool_calls=[{
                "index": 0,
                "function": {"arguments": ' "alpha"}'},
            }],
            finish_reason="tool_calls",
        ),
    ]


class RecordingTransport:
    def __init__(self, events):
        self.events = events
        self.stream_requests = []
        self.requests = []
        self.closed = []

    def post_json(self, request, *, http_error_handler=None):
        self.requests.append(request)
        return JSONResponse({})

    def post_sse(self, request, *, http_error_handler=None, stream_error_handler=None):
        self.stream_requests.append(request)

        def source():
            try:
                yield from self.events
            finally:
                self.closed.append(True)

        return source()


@pytest.fixture
def mistral_runtime(monkeypatch):
    registry = RegistrySpec()
    assert PLUGIN.model_metadata is not None
    assert registry.register_organization_metadata(PLUGIN.model_metadata)
    providers = ServiceProviderRegistry()
    PLUGIN.register(providers)
    monkeypatch.setattr(universal_module, "LLM_REGISTRY", registry)
    monkeypatch.setattr(universal_module, "SERVICE_PROVIDER_REGISTRY", providers)
    monkeypatch.setattr(base_adapter_module, "LLM_REGISTRY", registry)
    return registry


def new_adapter(transport="requests"):
    adapter = UniversalLLMAPIAdapter(
        organization="zai",
        service_provider="mistral",
        model=MODEL,
        api_key="mistral-test-key",
        transport=transport,
    )
    assert (adapter.organization, adapter.service_provider, adapter.model) == (
        "zai", "mistral", MODEL
    )
    return adapter


def install_sync_transport(adapter, transport):
    expected = RequestsSyncTransport if adapter.adapter.transport == "requests" else HttpxSyncTransport
    assert isinstance(adapter.adapter._sync_transport, expected)
    adapter.adapter._sync_transport = transport


def install_async_source(monkeypatch, event_source, calls):
    def fake_async_stream_request(url, **kwargs):
        calls.append((url, kwargs))
        return event_source()

    monkeypatch.setattr(
        mistral_async_client_module,
        "async_stream_request",
        fake_async_stream_request,
    )


@pytest.mark.integration
@pytest.mark.parametrize("transport", ["requests", "httpx"])
def test_flash_routed_sync_stream_flushes_visible_text_and_completes_after_tools(
    mistral_runtime, transport
):
    adapter = new_adapter(transport)
    source = RecordingTransport(successful_events())
    install_sync_transport(adapter, source)
    callbacks = []
    reasoning = []
    chunks = []
    completed = []

    output = list(adapter.stream_chat(
        MESSAGES,
        tools=[LOOKUP_TOOL],
        buffer_chars=3,
        capture_reasoning=True,
        on_reasoning=reasoning.append,
        on_chunk=chunks.append,
        on_tool_call=lambda call: callbacks.append(("tool", call)),
        on_done=lambda response: (callbacks.append(("done", response)), completed.append(response)),
    ))

    assert output == ["ABC", "DE"]
    assert all(0 < len(text) <= 3 for text in output)
    assert "".join(output) == "ABCDE"
    assert [chunk.text for chunk in chunks] == output
    assert [item.text for item in reasoning] == ["Private thought."]
    assert "Private thought." not in "".join(output)
    assert [kind for kind, _ in callbacks] == ["tool", "done"]
    assert callbacks[0][1].name == "lookup"
    assert callbacks[0][1].arguments == {"key": "alpha"}
    response = completed[0]
    assert response.content == "ABCDE"
    assert response.response_id == "flash-stream-1"
    assert [item.text for item in response.reasoning_events] == ["Private thought."]
    assert callbacks[1][1] is response
    assert len(source.stream_requests) == 1
    assert source.requests == []
    assert source.stream_requests[0].payload["model"] == WIRE_MODEL
    assert source.stream_requests[0].payload["stream"] is True
    assert source.closed == [True]


@pytest.mark.integration
def test_flash_routed_async_stream_flushes_visible_text_and_completes_after_tools(
    mistral_runtime, monkeypatch
):
    adapter = new_adapter()
    closed = []
    calls = []

    def source():
        async def events():
            try:
                for item in successful_events():
                    yield item
            finally:
                closed.append(True)
        return events()

    install_async_source(monkeypatch, source, calls)

    async def collect():
        order = []
        reasoning = []
        completed = []

        async def on_tool(call):
            order.append(("tool", call))

        async def on_done(response):
            order.append(("done", response))
            completed.append(response)

        async def on_reasoning(item):
            reasoning.append(item)

        output = [item async for item in adapter.astream_chat(
            MESSAGES,
            tools=[LOOKUP_TOOL],
            buffer_chars=3,
            capture_reasoning=True,
            on_reasoning=on_reasoning,
            on_tool_call=on_tool,
            on_done=on_done,
        )]
        return output, order, reasoning, completed

    output, order, reasoning, completed = asyncio.run(collect())
    assert output == ["ABC", "DE"]
    assert all(0 < len(text) <= 3 for text in output)
    assert "".join(output) == "ABCDE"
    assert [item.text for item in reasoning] == ["Private thought."]
    assert "Private thought." not in "".join(output)
    assert [kind for kind, _ in order] == ["tool", "done"]
    assert order[0][1].arguments == {"key": "alpha"}
    assert completed[0].content == "ABCDE"
    assert completed[0].response_id == "flash-stream-1"
    assert order[1][1] is completed[0]
    assert [item.text for item in completed[0].reasoning_events] == ["Private thought."]
    assert calls[0][1]["payload"]["model"] == WIRE_MODEL
    assert closed == [True]


@pytest.mark.integration
@pytest.mark.parametrize("transport", ["requests", "httpx"])
def test_flash_sync_callback_error_propagates_unchanged_and_closes_source(
    mistral_runtime, transport
):
    adapter = new_adapter(transport)
    source = RecordingTransport([event("visible", finish_reason="stop")])
    install_sync_transport(adapter, source)
    expected = RuntimeError("caller callback failed")
    completed = []
    stream = adapter.stream_chat(
        MESSAGES,
        buffer_chars=1,
        on_delta=lambda _: (_ for _ in ()).throw(expected),
        on_done=completed.append,
    )

    with pytest.raises(RuntimeError) as caught:
        next(stream)

    assert caught.value is expected
    assert completed == []
    assert source.closed == [True]
    assert len(source.stream_requests) == 1
    assert source.requests == []


@pytest.mark.integration
def test_flash_async_callback_error_propagates_unchanged_and_closes_source(
    mistral_runtime, monkeypatch
):
    adapter = new_adapter()
    closed = []
    calls = []
    expected = RuntimeError("caller callback failed")

    def source():
        async def events():
            try:
                yield event("visible", finish_reason="stop")
                await asyncio.Event().wait()
            finally:
                closed.append(True)
        return events()

    install_async_source(monkeypatch, source, calls)

    async def run():
        completed = []

        async def on_delta(_):
            raise expected

        with pytest.raises(RuntimeError) as caught:
            async for _ in adapter.astream_chat(MESSAGES, buffer_chars=1, on_delta=on_delta, on_done=completed.append):
                pass
        return caught.value, completed

    caught, completed = asyncio.run(run())
    assert caught is expected
    assert completed == []
    assert closed == [True]
    assert len(calls) == 1


@pytest.mark.integration
@pytest.mark.parametrize("transport", ["requests", "httpx"])
def test_flash_closed_sync_stream_discards_pending_text_and_closes_source(
    mistral_runtime, transport
):
    adapter = new_adapter(transport)
    source = RecordingTransport([event("ABCDE", finish_reason="stop")])
    install_sync_transport(adapter, source)
    completed = []
    stream = adapter.stream_chat(MESSAGES, buffer_chars=3, on_done=completed.append)

    assert next(stream) == "ABC"
    stream.close()

    assert completed == []
    assert source.closed == [True]
    assert len(source.stream_requests) == 1
    assert source.requests == []


@pytest.mark.integration
def test_flash_cancelled_async_stream_discards_pending_text_and_closes_source(
    mistral_runtime, monkeypatch
):
    adapter = new_adapter()
    closed = []
    calls = []

    def source():
        async def events():
            try:
                yield event("ABCDE", finish_reason="stop")
                await asyncio.Event().wait()
            finally:
                closed.append(True)
        return events()

    install_async_source(monkeypatch, source, calls)

    async def run():
        completed = []
        with pytest.raises(asyncio.CancelledError):
            async for _ in adapter.astream_chat(MESSAGES, buffer_chars=3, on_done=completed.append):
                raise asyncio.CancelledError
        return completed

    assert asyncio.run(run()) == []
    assert closed == [True]
    assert len(calls) == 1


@pytest.mark.integration
@pytest.mark.parametrize("transport", ["requests", "httpx"])
def test_flash_sync_provider_error_is_preserved_and_does_not_retry(
    mistral_runtime, transport
):
    adapter = new_adapter(transport)
    expected = LLMAPIClientError(detail="synthetic stream failure")
    source = RecordingTransport([])

    def failing_source():
        yield event("partial", finish_reason=None)
        raise expected

    source.events = failing_source()
    install_sync_transport(adapter, source)
    completed = []

    with pytest.raises(LLMAPIClientError) as caught:
        list(adapter.stream_chat(MESSAGES, on_done=completed.append))

    assert caught.value is expected
    assert completed == []
    assert source.closed == [True]
    assert len(source.stream_requests) == 1
    assert source.requests == []


@pytest.mark.integration
def test_flash_async_provider_error_is_preserved_and_does_not_retry(
    mistral_runtime, monkeypatch
):
    adapter = new_adapter()
    expected = LLMAPIClientError(detail="synthetic stream failure")
    closed = []
    calls = []

    def source():
        async def events():
            try:
                yield event("partial")
                raise expected
            finally:
                closed.append(True)
        return events()

    install_async_source(monkeypatch, source, calls)

    async def run():
        completed = []
        with pytest.raises(LLMAPIClientError) as caught:
            async for _ in adapter.astream_chat(MESSAGES, on_done=completed.append):
                pass
        return caught.value, completed

    caught, completed = asyncio.run(run())
    assert caught is expected
    assert completed == []
    assert closed == [True]
    assert len(calls) == 1
