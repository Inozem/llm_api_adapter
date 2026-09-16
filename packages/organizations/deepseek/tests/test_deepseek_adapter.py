"""Credential-free facade contracts for DeepSeek's Responses API adapter."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping

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
from llm_api_adapter.errors.llm_api_error import LLMAPIClientError
from llm_api_adapter.llm_registry.llm_registry import RegistrySpec
from llm_api_adapter.llms.transports import (
    JSONResponse,
    SSEEvent,
    SyncTransport,
    TransportRequest,
)
from llm_api_adapter.models.messages.chat_message import Prompt, UserMessage
from llm_api_adapter.service_provider_registry import ServiceProviderRegistry
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter


@dataclass
class FakeSyncTransport(SyncTransport):
    """Transport double that exposes requests and closes stream iterators."""

    response: Any
    stream_events: list[SSEEvent] = field(default_factory=list)
    requests: list[TransportRequest] = field(default_factory=list)
    sse_closed: bool = False

    def post_json(
        self,
        request: TransportRequest,
        *,
        http_error_handler=None,
    ) -> JSONResponse:
        del http_error_handler
        self.requests.append(request)
        return JSONResponse(self.response)

    def post_multipart(
        self,
        request: TransportRequest,
        form,
        *,
        http_error_handler=None,
    ) -> JSONResponse:
        del request, form, http_error_handler
        raise AssertionError("DeepSeek text contract must not upload files")

    def post_sse(
        self,
        request: TransportRequest,
        *,
        http_error_handler=None,
        stream_error_handler=None,
    ) -> Iterator[SSEEvent]:
        del http_error_handler
        self.requests.append(request)

        def events() -> Iterator[SSEEvent]:
            try:
                for event in self.stream_events:
                    payload = event.data if isinstance(event.data, Mapping) else {}
                    if (
                        stream_error_handler is not None
                        and (
                            event.event == "error"
                            or payload.get("type") == "error"
                        )
                    ):
                        stream_error_handler(event)
                    yield event
            finally:
                self.sse_closed = True

        return events()


def _response() -> dict[str, Any]:
    """Return a completed official Responses API envelope."""
    return {
        "object": "response",
        "id": "resp-deepseek-flash",
        "model": "deepseek-flash",
        "created_at": 1_774_274_151,
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Hello from DeepSeek.",
                    },
                ],
            },
        ],
        "usage": {
            "input_tokens": 20,
            "output_tokens": 5,
            "total_tokens": 25,
        },
    }


def _stream_events() -> list[SSEEvent]:
    response = _response()
    response["id"] = "stream-deepseek-flash"
    return [
        SSEEvent(
            event="response.created",
            data={
                "type": "response.created",
                "response": {
                    "id": response["id"],
                    "model": "deepseek-flash",
                    "object": "response",
                    "status": "in_progress",
                },
            },
        ),
        SSEEvent(
            event="response.output_text.delta",
            data={
                "type": "response.output_text.delta",
                "delta": "Hello ",
            },
        ),
        SSEEvent(
            event="response.output_text.delta",
            data={
                "type": "response.output_text.delta",
                "delta": "from ",
            },
        ),
        SSEEvent(
            event="response.output_text.delta",
            data={
                "type": "response.output_text.delta",
                "delta": "DeepSeek.",
            },
        ),
        SSEEvent(
            event="response.completed",
            data={"type": "response.completed", "response": response},
        ),
    ]


@pytest.fixture
def deepseek_runtime(monkeypatch):
    """Register the package plugin against isolated Core registries."""
    from llm_api_adapter_deepseek.plugin import PLUGIN

    model_registry = RegistrySpec()
    assert PLUGIN.model_metadata is not None
    assert (
        model_registry.register_organization_metadata(PLUGIN.model_metadata) is True
    )

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


def _deepseek_facade() -> UniversalLLMAPIAdapter:
    return UniversalLLMAPIAdapter(
        organization="deepseek",
        model="deepseek-flash",
        api_key="deepseek-test-key",
    )


@pytest.mark.integration
def test_facade_chat_maps_text_to_responses_and_normalizes_output(deepseek_runtime):
    adapter = _deepseek_facade()
    transport = FakeSyncTransport(_response())
    adapter.adapter._client._sync_transport = transport

    response = adapter.chat(
        messages=[Prompt("Be concise."), UserMessage("Hello")],
        max_tokens=12,
        temperature=0.5,
        top_p=0.8,
        timeout_s=3.0,
    )

    assert response.content == "Hello from DeepSeek."
    assert response.response_id == "resp-deepseek-flash"
    assert response.usage is not None
    assert response.usage.total_tokens == 25
    assert len(transport.requests) == 1
    request = transport.requests[0]
    assert request.url.endswith("/responses")
    assert request.headers_dict() == {
        "Authorization": "Bearer deepseek-test-key",
        "Content-Type": "application/json",
    }
    assert request.payload == {
        "model": "deepseek-flash",
        "input": [{"role": "user", "content": "Hello"}],
        "max_output_tokens": 12,
        "temperature": 0.5,
        "top_p": 0.8,
        "instructions": "Be concise.",
    }
    assert request.timeout == 3.0


@pytest.mark.integration
def test_facade_stream_maps_sse_callbacks_completion_and_close(deepseek_runtime):
    adapter = _deepseek_facade()
    transport = FakeSyncTransport({}, stream_events=_stream_events())
    adapter.adapter._client._sync_transport = transport
    callbacks: list[tuple[str, Any]] = []

    output = list(
        adapter.stream_chat(
            messages=[Prompt("Be concise."), UserMessage("Hello")],
            max_tokens=12,
            temperature=0.5,
            top_p=0.8,
            timeout_s=3.0,
            buffer_chars=6,
            on_chunk=lambda chunk: callbacks.append(("chunk", chunk.text)),
            on_delta=lambda text: callbacks.append(("delta", text)),
            on_done=lambda response: callbacks.append(("done", response)),
        )
    )

    assert output == ["Hello ", "from ", "DeepSe", "ek."]
    assert callbacks[:8] == [
        ("chunk", "Hello "),
        ("delta", "Hello "),
        ("chunk", "from "),
        ("delta", "from "),
        ("chunk", "DeepSe"),
        ("delta", "DeepSe"),
        ("chunk", "ek."),
        ("delta", "ek."),
    ]
    assert callbacks[-1][0] == "done"
    assert callbacks[-1][1].content == "Hello from DeepSeek."
    assert callbacks[-1][1].response_id == "stream-deepseek-flash"
    assert transport.sse_closed is True
    assert transport.requests[0].payload["stream"] is True
    assert transport.requests[0].payload["stream_options"] == {
        "include_usage": True,
    }


@pytest.mark.integration
def test_facade_achat_and_astream_match_sync_contract(deepseek_runtime, monkeypatch):
    from llm_api_adapter_deepseek.clients import async_client as async_client_module

    requests: list[dict[str, Any]] = []

    async def fake_async_request(
        url: str,
        *,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout: float | None,
        http_error_handler,
    ) -> dict[str, Any]:
        del http_error_handler
        requests.append(
            {
                "url": url,
                "headers": headers,
                "payload": payload,
                "timeout": timeout,
            }
        )
        return _response()

    def fake_async_stream_request(
        url: str,
        *,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout: float | None,
        http_error_handler,
        stream_error_handler,
    ):
        del http_error_handler, stream_error_handler
        requests.append(
            {
                "url": url,
                "headers": headers,
                "payload": payload,
                "timeout": timeout,
            }
        )

        async def events():
            for event in _stream_events():
                yield event

        return events()

    monkeypatch.setattr(async_client_module, "async_request", fake_async_request)
    monkeypatch.setattr(
        async_client_module,
        "async_stream_request",
        fake_async_stream_request,
    )
    adapter = _deepseek_facade()

    async def exercise():
        response = await adapter.achat(
            messages=[Prompt("Be concise."), UserMessage("Hello")],
            max_tokens=12,
            temperature=0.5,
            top_p=0.8,
            timeout_s=3.0,
        )
        callback_events: list[tuple[str, Any]] = []

        async def on_chunk(chunk):
            callback_events.append(("chunk", chunk.text))

        async def on_delta(text):
            callback_events.append(("delta", text))

        async def on_done(done_response):
            callback_events.append(("done", done_response))

        chunks = []
        async for text in adapter.astream_chat(
            messages=[Prompt("Be concise."), UserMessage("Hello")],
            max_tokens=12,
            temperature=0.5,
            top_p=0.8,
            timeout_s=3.0,
            buffer_chars=6,
            on_chunk=on_chunk,
            on_delta=on_delta,
            on_done=on_done,
        ):
            chunks.append(text)
        return response, chunks, callback_events

    response, chunks, callback_events = asyncio.run(exercise())

    assert response.content == "Hello from DeepSeek."
    assert chunks == ["Hello ", "from ", "DeepSe", "ek."]
    assert callback_events[:2] == [
        ("chunk", "Hello "),
        ("delta", "Hello "),
    ]
    assert callback_events[-1][0] == "done"
    assert callback_events[-1][1].content == "Hello from DeepSeek."
    assert len(requests) == 2
    assert requests[0]["url"].endswith("/responses")
    assert requests[0]["headers"]["Authorization"] == (
        "Bearer deepseek-test-key"
    )
    assert requests[0]["payload"]["model"] == "deepseek-flash"
    assert requests[0]["payload"]["input"] == [
        {"role": "user", "content": "Hello"},
    ]
    assert requests[1]["payload"]["stream"] is True
    assert requests[1]["payload"]["stream_options"] == {
        "include_usage": True,
    }


@pytest.mark.integration
def test_stream_close_before_terminal_event_skips_completion(deepseek_runtime):
    adapter = _deepseek_facade()
    transport = FakeSyncTransport({}, stream_events=_stream_events())
    adapter.adapter._client._sync_transport = transport
    completed = []
    stream = adapter.stream_chat(
        messages=[UserMessage("Hello")],
        on_done=completed.append,
    )

    assert next(stream) == "Hello "
    stream.close()

    assert transport.sse_closed is True
    assert completed == []


@pytest.mark.integration
def test_stream_without_terminal_response_is_a_client_error(deepseek_runtime):
    adapter = _deepseek_facade()
    adapter.adapter._client._sync_transport = FakeSyncTransport(
        {},
        stream_events=_stream_events()[:-1],
    )
    completed = []

    with pytest.raises(LLMAPIClientError):
        list(
            adapter.stream_chat(
                messages=[UserMessage("Hello")],
                on_done=completed.append,
            )
        )

    assert completed == []


@pytest.mark.integration
def test_async_stream_cancellation_closes_resources_and_skips_completion(
    deepseek_runtime,
    monkeypatch,
):
    from llm_api_adapter_deepseek.clients import async_client as async_client_module

    stream_closed = False
    stream_entered = asyncio.Event()
    never = asyncio.Event()

    def fake_async_stream_request(url: str, **kwargs: Any):
        del url, kwargs

        async def events():
            nonlocal stream_closed
            try:
                events = _stream_events()
                yield events[0]
                yield events[1]
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
    adapter = _deepseek_facade()
    completed = []

    async def cancel_stream():
        stream = adapter.astream_chat(
            messages=[UserMessage("Hello")],
            on_done=completed.append,
        )
        assert await stream.__anext__() == "Hello "
        pending_chunk = asyncio.create_task(stream.__anext__())
        await stream_entered.wait()
        pending_chunk.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending_chunk

    asyncio.run(cancel_stream())
    assert stream_closed is True
    assert completed == []
