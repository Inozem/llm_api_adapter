"""Native Kimi contracts that migration tasks must preserve or deliberately update."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
import inspect
from pathlib import Path
import sys
from typing import Any

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[2]
for source in (PACKAGE_ROOT / "src", REPOSITORY_ROOT / "src", REPOSITORY_ROOT):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

import llm_api_adapter.adapters.base_adapter as base_adapter_module
import llm_api_adapter.organization_registry as organization_registry_module
import llm_api_adapter.universal_adapter as universal_module
from llm_api_adapter.errors.llm_api_error import (
    LLMAPIClientError,
    LLMAPIRateLimitError,
    LLMAPIUsageLimitError,
)
from llm_api_adapter.llm_registry.llm_registry import RegistrySpec
from llm_api_adapter.llms.transports import (
    HTTPErrorHandler,
    JSONResponse,
    SSEEvent,
    StreamErrorHandler,
    TransportRequest,
)
from llm_api_adapter.models.messages.chat_message import UserMessage
from llm_api_adapter.organization_registry import (
    ORGANIZATION_PLUGIN_ENTRY_POINT_GROUP,
    OrganizationPluginDiscovery,
    OrganizationPlugin,
)
from llm_api_adapter.service_provider_registry import ServiceProviderRegistry
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter
from llm_api_adapter_kimi.adapter import KimiAdapter
from llm_api_adapter_kimi.clients.async_client import KimiAsyncClient
from llm_api_adapter_kimi.clients.sync_client import (
    KIMI_CHAT_COMPLETIONS_URL,
    KimiSyncClient,
)
from llm_api_adapter_kimi.plugin import PLUGIN


@dataclass(frozen=True)
class _InstalledKimiEntryPoint:
    """Entry point stand-in that loads the real installed Kimi plugin."""

    name: str = "kimi"
    value: str = "llm_api_adapter_kimi.plugin:PLUGIN"

    def load(self) -> OrganizationPlugin:
        return PLUGIN


class RecordingTransport:
    """Small deterministic Requests transport that records native requests."""

    def __init__(
        self,
        response: dict[str, Any],
        events: list[SSEEvent] | None = None,
    ) -> None:
        self.response = response
        self.events = list(events or [])
        self.requests: list[TransportRequest] = []
        self.stream_requests: list[TransportRequest] = []
        self.closed = False

    def post_json(
        self,
        request: TransportRequest,
        *,
        http_error_handler: HTTPErrorHandler | None = None,
    ) -> JSONResponse:
        self.requests.append(request)
        return JSONResponse(self.response)

    def post_sse(
        self,
        request: TransportRequest,
        *,
        http_error_handler: HTTPErrorHandler | None = None,
        stream_error_handler: StreamErrorHandler | None = None,
    ) -> Iterator[SSEEvent]:
        self.stream_requests.append(request)

        def iterate() -> Iterator[SSEEvent]:
            try:
                yield from self.events
            finally:
                self.closed = True

        return iterate()


@pytest.fixture
def discovered_kimi(monkeypatch: pytest.MonkeyPatch) -> UniversalLLMAPIAdapter:
    """Build the public facade through actual plugin discovery, without network I/O."""
    service_registry = ServiceProviderRegistry()
    model_registry = RegistrySpec()

    def entry_points(*, group: str) -> list[_InstalledKimiEntryPoint]:
        assert group == ORGANIZATION_PLUGIN_ENTRY_POINT_GROUP
        return [_InstalledKimiEntryPoint()]

    monkeypatch.setattr(organization_registry_module, "entry_points", entry_points)
    monkeypatch.setattr(universal_module, "SERVICE_PROVIDER_REGISTRY", service_registry)
    monkeypatch.setattr(
        universal_module,
        "ORGANIZATION_PLUGIN_DISCOVERY",
        OrganizationPluginDiscovery(),
    )
    monkeypatch.setattr(universal_module, "LLM_REGISTRY", model_registry)
    monkeypatch.setattr(base_adapter_module, "LLM_REGISTRY", model_registry)
    return UniversalLLMAPIAdapter(
        organization="kimi",
        model="kimi-k3",
        api_key="kimi-characterization-key",
    )


def _response() -> dict[str, Any]:
    return {
        "id": "chatcmpl-native-kimi",
        "created": 1_789_721_600,
        "model": "kimi-k3",
        "choices": [
            {
                "message": {"role": "assistant", "content": "Native reply"},
                "finish_reason": "stop",
            },
        ],
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 7,
            "total_tokens": 27,
            "cached_tokens": 12,
            "prompt_tokens_details": {"cached_tokens": 12, "cache_write_tokens": 3},
        },
    }


@pytest.mark.unit
def test_native_signatures_and_defaults_are_characterized() -> None:
    constructor = inspect.signature(KimiAdapter)
    assert tuple(constructor.parameters) == (
        "api_key", "model", "company", "transport", "pricing", "is_reasoning",
        "is_adaptive_thinking", "service_provider",
    )
    assert [parameter.kind for parameter in constructor.parameters.values()] == [
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ] * len(constructor.parameters)
    assert constructor.parameters["api_key"].default is inspect.Parameter.empty
    assert constructor.parameters["model"].default is inspect.Parameter.empty
    assert constructor.parameters["company"].default == "kimi"
    assert constructor.parameters["transport"].default == "requests"
    assert constructor.parameters["pricing"].default is None
    assert constructor.parameters["is_reasoning"].default is False
    assert constructor.parameters["is_adaptive_thinking"].default is False
    assert constructor.parameters["service_provider"].default is None

    expected_chat_parameters = (
        "self", "messages", "max_tokens", "temperature", "top_p",
        "reasoning_level", "timeout_s", "tools", "tool_choice",
        "parallel_tool_calls", "previous_response", "json_schema", "response_model",
        "capture_reasoning",
    )
    for method in (KimiAdapter.chat, KimiAdapter.achat):
        signature = inspect.signature(method)
        assert tuple(signature.parameters) == expected_chat_parameters
        assert signature.parameters["max_tokens"].default is None
        assert signature.parameters["temperature"].default == 1.0
        assert signature.parameters["top_p"].default == 1.0
        assert signature.parameters["reasoning_level"].default is None
        assert signature.parameters["capture_reasoning"].default is False
        assert signature.parameters["capture_reasoning"].kind is inspect.Parameter.KEYWORD_ONLY

    expected_stream_parameters = (
        *expected_chat_parameters[:-1], "on_delta", "on_tool_call", "on_done",
        "buffer_chars", "on_chunk", "capture_reasoning", "on_reasoning",
    )
    for method in (KimiAdapter.stream_chat, KimiAdapter.astream_chat):
        signature = inspect.signature(method)
        assert tuple(signature.parameters) == expected_stream_parameters
        assert signature.parameters["max_tokens"].default is None
        assert signature.parameters["temperature"].default == 1.0
        assert signature.parameters["top_p"].default == 1.0
        assert signature.parameters["capture_reasoning"].default is False
        assert signature.parameters["on_reasoning"].default is None
        assert signature.parameters["capture_reasoning"].kind is inspect.Parameter.KEYWORD_ONLY
        assert signature.parameters["on_reasoning"].kind is inspect.Parameter.KEYWORD_ONLY

    sync_signature = inspect.signature(KimiSyncClient)
    assert tuple(sync_signature.parameters) == ("transport",)
    for method in (
        KimiSyncClient.chat,
        KimiSyncClient.stream,
        KimiAsyncClient.chat,
        KimiAsyncClient.stream,
    ):
        assert tuple(inspect.signature(method).parameters) == (
            "self", "api_key", "payload", "timeout_s",
        )
        assert inspect.signature(method).parameters["timeout_s"].default is inspect.Parameter.empty
        assert all(
            inspect.signature(method).parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
            for name in ("api_key", "payload", "timeout_s")
        )

    assert inspect.signature(KimiAsyncClient).parameters == {}


@pytest.mark.unit
def test_discovered_adapter_uses_native_chat_route_payload_and_usage(
    discovered_kimi: UniversalLLMAPIAdapter,
) -> None:
    transport = RecordingTransport(_response())
    assert isinstance(discovered_kimi.adapter, KimiAdapter)
    discovered_kimi.adapter._sync_transport = transport

    response = discovered_kimi.chat([UserMessage("Hi")], max_tokens=64)

    request = transport.requests[0]
    assert request.url == KIMI_CHAT_COMPLETIONS_URL
    assert request.url == "https://api.moonshot.ai/v1/chat/completions"
    assert request.headers == {
        "Authorization": "Bearer kimi-characterization-key",
        "Content-Type": "application/json",
    }
    assert request.payload["model"] == "kimi-k3"
    assert request.payload["max_completion_tokens"] == 64
    assert "max_tokens" not in request.payload
    assert "temperature" not in request.payload
    assert response.content == "Native reply"
    assert response.response_id == "chatcmpl-native-kimi"
    assert response.usage is not None
    assert (
        response.usage.input_tokens,
        response.usage.output_tokens,
        response.usage.total_tokens,
    ) == (20, 7, 27)
    assert response.usage.cached_tokens == 12
    assert response.usage.cache_write_tokens == 3
    assert response.cost_total is not None


@pytest.mark.unit
def test_raw_sync_and_async_chat_accept_keyword_only_payloads_without_rewriting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_payload: dict[str, Any] = {"model": "raw-kimi-model", "custom_field": {"x": 1}}
    sync_transport = RecordingTransport({"choices": []})
    sync_result = KimiSyncClient(sync_transport).chat(
        api_key="raw-sync-key",
        payload=raw_payload,
        timeout_s=2.5,
    )
    assert sync_result == {"choices": []}
    sync_request = sync_transport.requests[0]
    assert sync_request.url == KIMI_CHAT_COMPLETIONS_URL
    assert sync_request.payload == {
        "model": "raw-kimi-model",
        "custom_field": {"x": 1},
    }
    assert sync_request.headers["Authorization"] == "Bearer raw-sync-key"
    assert sync_request.timeout == 2.5
    assert raw_payload == {"model": "raw-kimi-model", "custom_field": {"x": 1}}

    from llm_api_adapter_kimi.clients import async_client as async_client_module

    async_calls: list[tuple[str, dict[str, Any]]] = []

    async def async_request(url: str, **kwargs: Any) -> dict[str, Any]:
        async_calls.append((url, kwargs))
        return {"choices": []}

    monkeypatch.setattr(async_client_module, "async_request", async_request)
    async_result = asyncio.run(
        KimiAsyncClient().chat(
            api_key="raw-async-key",
            payload=raw_payload,
            timeout_s=3.5,
        )
    )
    assert async_result == {"choices": []}
    assert async_calls[0][0] == KIMI_CHAT_COMPLETIONS_URL
    assert async_calls[0][1]["payload"] == {
        "model": "raw-kimi-model",
        "custom_field": {"x": 1},
    }
    assert async_calls[0][1]["headers"]["Authorization"] == "Bearer raw-async-key"
    assert async_calls[0][1]["timeout"] == 3.5
    assert raw_payload == {"model": "raw-kimi-model", "custom_field": {"x": 1}}


@pytest.mark.unit
def test_raw_sync_stream_opens_on_first_consumption_and_sets_native_usage_options() -> None:
    transport = RecordingTransport({})
    client = KimiSyncClient(transport)
    raw_payload: dict[str, Any] = {"model": "kimi-k3", "stream_options": {"custom": True}}

    events = client.stream(api_key="sync-key", payload=raw_payload, timeout_s=9.5)

    assert transport.stream_requests == []
    assert list(events) == []
    assert len(transport.stream_requests) == 1
    request = transport.stream_requests[0]
    assert request.payload == {
        "model": "kimi-k3", "stream_options": {"include_usage": True}, "stream": True,
    }
    assert raw_payload == {"model": "kimi-k3", "stream_options": {"custom": True}}
    assert request.timeout == 9.5


@pytest.mark.unit
def test_raw_async_stream_opens_at_factory_and_sets_native_usage_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_api_adapter_kimi.clients import async_client as async_client_module

    calls: list[tuple[str, dict[str, Any]]] = []

    def async_stream_request(url: str, **kwargs: Any) -> AsyncIterator[SSEEvent]:
        calls.append((url, kwargs))

        async def events() -> AsyncIterator[SSEEvent]:
            yield SSEEvent(event=None, data="[DONE]", done=True)

        return events()

    monkeypatch.setattr(async_client_module, "async_stream_request", async_stream_request)
    client = KimiAsyncClient()
    raw_payload: dict[str, Any] = {"model": "kimi-k3", "stream_options": {"custom": True}}

    stream = client.stream(api_key="async-key", payload=raw_payload, timeout_s=4.0)

    # T004 changes this baseline check to assert lazy opening on first consumption.
    assert len(calls) == 1
    assert calls[0][0] == KIMI_CHAT_COMPLETIONS_URL
    assert calls[0][1]["payload"] == {
        "model": "kimi-k3", "stream_options": {"include_usage": True}, "stream": True,
    }
    assert calls[0][1]["headers"]["Authorization"] == "Bearer async-key"
    assert raw_payload == {"model": "kimi-k3", "stream_options": {"custom": True}}

    async def collect() -> list[SSEEvent]:
        return [event async for event in stream]

    assert asyncio.run(collect())[0].done is True


@pytest.mark.unit
def test_public_chat_parallel_error_precedes_sampling(
    discovered_kimi: UniversalLLMAPIAdapter,
) -> None:
    transport = RecordingTransport({})
    discovered_kimi.adapter._sync_transport = transport

    # T005 changes this baseline: inherited validation reports sampling ValueError first.
    with pytest.raises(NotImplementedError, match="parallel_tool_calls"):
        discovered_kimi.chat(
            [UserMessage("Hi")], temperature=-1, parallel_tool_calls=False,
        )
    assert transport.requests == []
    assert transport.stream_requests == []


@pytest.mark.unit
def test_public_sync_stream_validates_eagerly(
    discovered_kimi: UniversalLLMAPIAdapter,
) -> None:
    transport = RecordingTransport({})
    discovered_kimi.adapter._sync_transport = transport

    # T005 changes this baseline to lazy validation on first stream consumption.
    with pytest.raises(ValueError, match="max_tokens"):
        discovered_kimi.stream_chat([UserMessage("Hi")], max_tokens=0)
    assert transport.requests == []
    assert transport.stream_requests == []


@pytest.mark.unit
def test_native_error_mapping_preserves_provider_error_classes() -> None:
    with pytest.raises(LLMAPIRateLimitError, match="slow down"):
        KimiSyncClient._raise_mapped_error(429, "rate_limit_error", "slow down")
    with pytest.raises(LLMAPIUsageLimitError, match="quota"):
        KimiSyncClient._raise_mapped_error(None, "QUOTA_EXCEEDED", "quota")
    with pytest.raises(LLMAPIClientError, match="bad request"):
        KimiSyncClient._raise_mapped_error(400, "invalid_request_error", "bad request")


@pytest.mark.unit
def test_native_stream_callbacks_observe_reasoning_before_visible_text(
    discovered_kimi: UniversalLLMAPIAdapter,
) -> None:
    events = [
        SSEEvent(event=None, data={"id": "s1", "created": 1, "model": "kimi-k3", "choices": [
            {
                "index": 0,
                "delta": {"reasoning_content": "Think", "content": "Hello"},
                "finish_reason": None,
            },
        ]}),
        SSEEvent(event=None, data={"id": "s1", "created": 1, "model": "kimi-k3", "choices": [
            {"index": 0, "delta": {}, "finish_reason": "stop"},
        ]}),
        SSEEvent(
            event=None,
            data={
                "id": "s1",
                "created": 1,
                "model": "kimi-k3",
                "choices": [],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                },
            },
        ),
        SSEEvent(event=None, data="[DONE]", done=True),
    ]
    transport = RecordingTransport({}, events)
    discovered_kimi.adapter._sync_transport = transport
    order: list[tuple[str, Any]] = []

    chunks = list(discovered_kimi.stream_chat(
        [UserMessage("Hi")],
        capture_reasoning=True,
        on_reasoning=lambda event: order.append(("reasoning", event)),
        on_delta=lambda text: order.append(("delta", text)),
        on_done=lambda response: order.append(("done", response)),
    ))

    assert chunks == ["Hello"]
    assert [kind for kind, _ in order] == ["reasoning", "delta", "done"]
    assert order[0][1].text == "Think"
    assert order[1][1] == "Hello"
    assert order[2][1].reasoning_events[0] is order[0][1]
    assert transport.closed is True

