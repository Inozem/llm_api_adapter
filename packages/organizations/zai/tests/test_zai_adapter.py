"""Credential-free facade and transport contracts for the Z.ai adapter."""

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
from llm_api_adapter.errors.config_errors import LLMReasoningLevelError
from llm_api_adapter.errors.llm_api_error import (
    LLMAPIAuthorizationError,
    LLMAPIError,
    LLMAPIRateLimitError,
    LLMAPIServerError,
    ToolChoiceError,
)
from llm_api_adapter.llm_registry.llm_registry import RegistrySpec
from llm_api_adapter.llms.transports import (
    JSONResponse,
    SSEEvent,
    SyncTransport,
    TransportRequest,
)
from llm_api_adapter.models.messages.chat_message import (
    AIMessage,
    ToolMessage,
    UserMessage,
)
from llm_api_adapter.models.messages.file_parts import DocumentPart, ImagePart
from llm_api_adapter.models.tools.tool_call import ToolCall
from llm_api_adapter.models.tools.tool_spec import ToolSpec
from llm_api_adapter.service_provider_registry import ServiceProviderRegistry
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter
from llm_api_adapter_zai.registry import ZaiCachePricing


MODEL = "glm-5.3-flash"
WEATHER_TOOL = ToolSpec(
    name="get_weather",
    description="Return the weather for a city.",
    json_schema={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
)


class DummyResponseModel:
    """Response-model stand-in; Z.ai rejects portable response models locally."""


@dataclass
class FakeSyncTransport(SyncTransport):
    """Transport double that records requests and closes SSE iterators."""

    response: Any
    error: Exception | None = None
    stream_events: list[SSEEvent] = field(default_factory=list)
    requests: list[TransportRequest] = field(default_factory=list)
    sse_closed: bool = False

    def post_json(
        self,
        request: TransportRequest,
        *,
        http_error_handler=None,
    ) -> JSONResponse:
        self.requests.append(request)
        if self.error is not None:
            assert http_error_handler is not None
            http_error_handler(self.error)
        return JSONResponse(self.response)

    def post_multipart(
        self,
        request: TransportRequest,
        form,
        *,
        http_error_handler=None,
    ) -> JSONResponse:
        del request, form, http_error_handler
        raise AssertionError("Z.ai Chat Completions must not upload files")

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


class FakeHTTPResponse:
    def __init__(self, status_code: int, payload: dict[str, Any]) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload


class FakeHTTPError(Exception):
    def __init__(self, status_code: int, payload: dict[str, Any]) -> None:
        super().__init__(f"HTTP {status_code}")
        self.response = FakeHTTPResponse(status_code, payload)


def zai_response(
    *,
    prompt_tokens: int = 19,
    completion_tokens: int = 13,
    cached_tokens: int | None = None,
) -> dict[str, Any]:
    usage: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    if cached_tokens is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
    return {
        "id": "cmpl-zai-test",
        "object": "chat.completion",
        "created": 1_789_721_600,
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Z.ai test."},
                "finish_reason": "stop",
            },
        ],
        "usage": usage,
    }


def zai_stream_events() -> list[SSEEvent]:
    metadata = {
        "id": "cmpl-zai-stream",
        "created": 1_789_721_600,
        "model": MODEL,
    }
    return [
        SSEEvent(
            event=None,
            data={
                **metadata,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": ""},
                        "finish_reason": None,
                    },
                ],
            },
        ),
        SSEEvent(
            event=None,
            data={
                **metadata,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "reasoning_content": "First reason. ",
                            "content": "Hello ",
                        },
                        "finish_reason": None,
                    },
                ],
            },
        ),
        SSEEvent(
            event=None,
            data={
                **metadata,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "world"},
                        "finish_reason": "stop",
                    },
                ],
            },
        ),
        SSEEvent(
            event=None,
            data={
                **metadata,
                "choices": [],
                "usage": {
                    "prompt_tokens": 19,
                    "completion_tokens": 13,
                    "total_tokens": 32,
                },
            },
        ),
        SSEEvent(event=None, data="[DONE]", done=True),
    ]


@pytest.fixture
def zai_runtime(monkeypatch):
    from llm_api_adapter_zai.plugin import PLUGIN

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
    ("request_kwargs", "error_type", "message"),
    [
        (
            {"json_schema": {"type": "object", "properties": {}}},
            NotImplementedError,
            "structured output",
        ),
        (
            {"response_model": DummyResponseModel},
            NotImplementedError,
            "structured output",
        ),
    ],
)
def test_zai_rejects_structured_output_before_http(
    zai_runtime,
    request_kwargs,
    error_type,
    message,
):
    adapter = UniversalLLMAPIAdapter(
        organization="zai",
        model=MODEL,
        api_key="zai-test-key",
    )
    transport = FakeSyncTransport(zai_response())
    adapter.adapter._sync_transport = transport

    with pytest.raises(error_type, match=message):
        adapter.chat([UserMessage("Return JSON")], **request_kwargs)

    assert transport.requests == []


@pytest.mark.unit
def test_zai_rejects_non_auto_tool_choice_before_http(zai_runtime):
    adapter = UniversalLLMAPIAdapter(
        organization="zai",
        model=MODEL,
        api_key="zai-test-key",
    )
    transport = FakeSyncTransport(zai_response())
    adapter.adapter._sync_transport = transport

    with pytest.raises(ToolChoiceError, match="tool_choice|auto"):
        adapter.chat(
            [UserMessage("Use the weather tool")],
            tools=[WEATHER_TOOL],
            tool_choice="none",
        )

    assert transport.requests == []


@pytest.mark.unit
def test_core_e2e_selects_auto_tool_choice_from_zai_registry(
    zai_runtime,
    monkeypatch,
):
    from tests.e2e import harness as e2e_harness

    monkeypatch.setattr(e2e_harness, "LLM_REGISTRY", zai_runtime)

    assert (
        e2e_harness.select_tool_choice_for_model(
            "zai",
            MODEL,
            WEATHER_TOOL.name,
        )
        == "auto"
    )


@pytest.mark.unit
def test_zai_maps_canonical_reasoning_level_before_http(zai_runtime):
    adapter = UniversalLLMAPIAdapter(
        organization="zai",
        model=MODEL,
        api_key="zai-test-key",
    )
    transport = FakeSyncTransport(zai_response())
    adapter.adapter._sync_transport = transport

    adapter.chat(
        [UserMessage("Explain the answer")],
        reasoning_level="medium",
    )

    assert transport.requests[0].payload["thinking"] == {"type": "enabled"}
    assert transport.requests[0].payload["reasoning_effort"] == "high"


@pytest.mark.unit
def test_zai_rejects_unknown_reasoning_level_before_http(zai_runtime):
    adapter = UniversalLLMAPIAdapter(
        organization="zai",
        model=MODEL,
        api_key="zai-test-key",
    )
    transport = FakeSyncTransport(zai_response())
    adapter.adapter._sync_transport = transport

    with pytest.raises(LLMReasoningLevelError, match="Unknown reasoning level"):
        adapter.chat(
            [UserMessage("Explain the answer")],
            reasoning_level="unsupported",
        )

    assert transport.requests == []


@pytest.mark.unit
def test_zai_rejects_unknown_model_capability_before_http(zai_runtime):
    with pytest.warns(UserWarning, match="not verified"):
        adapter = UniversalLLMAPIAdapter(
            organization="zai",
            model="glm-5.3-flashx",
            api_key="zai-test-key",
        )
    transport = FakeSyncTransport(zai_response())
    adapter.adapter._sync_transport = transport

    with pytest.raises(
        (LLMAPIError, NotImplementedError),
        match="not verified|capability|supported",
    ):
        adapter.chat(
            [UserMessage("Use the weather tool")],
            tools=[WEATHER_TOOL],
        )

    assert transport.requests == []


@pytest.mark.unit
@pytest.mark.parametrize(
    "document",
    [
        DocumentPart(url="https://example.test/brief.pdf"),
        DocumentPart(data=b"%PDF-1.7", media_type="application/pdf"),
    ],
    ids=["url", "bytes"],
)
def test_zai_rejects_document_parts_before_http(zai_runtime, document):
    adapter = UniversalLLMAPIAdapter(
        organization="zai",
        model=MODEL,
        api_key="zai-test-key",
    )
    transport = FakeSyncTransport(zai_response())
    adapter.adapter._sync_transport = transport

    with pytest.raises(ValueError, match="DocumentPart|document"):
        adapter.chat(
            [UserMessage("Summarize this document", files=[document])],
        )

    assert transport.requests == []


@pytest.mark.unit
@pytest.mark.parametrize("parallel_tool_calls", [False, True])
def test_zai_rejects_unverified_tool_combination_before_http(
    zai_runtime,
    parallel_tool_calls,
):
    adapter = UniversalLLMAPIAdapter(
        organization="zai",
        model=MODEL,
        api_key="zai-test-key",
    )
    transport = FakeSyncTransport(zai_response())
    adapter.adapter._sync_transport = transport

    with pytest.raises(NotImplementedError, match="parallel_tool_calls"):
        adapter.chat(
            [UserMessage("Use the weather tool")],
            tools=[WEATHER_TOOL],
            parallel_tool_calls=parallel_tool_calls,
        )

    assert transport.requests == []


@pytest.mark.unit
def test_zai_maps_auto_tools_at_the_128_declaration_limit(zai_runtime):
    adapter = UniversalLLMAPIAdapter(
        organization="zai",
        model=MODEL,
        api_key="zai-test-key",
    )
    transport = FakeSyncTransport(zai_response())
    adapter.adapter._sync_transport = transport
    tools = [
        ToolSpec(name=f"tool_{index}", json_schema={"type": "object"})
        for index in range(128)
    ]

    adapter.chat(
        [UserMessage("Choose a tool")],
        tools=tools,
        tool_choice="auto",
    )

    request = transport.requests[0]
    assert request.payload["tool_choice"] == "auto"
    assert len(request.payload["tools"]) == 128
    assert request.payload["tools"][0] == {
        "type": "function",
        "function": {
            "name": "tool_0",
            "parameters": {"type": "object"},
        },
    }


@pytest.mark.unit
def test_zai_rejects_more_than_128_tools_before_http(zai_runtime):
    adapter = UniversalLLMAPIAdapter(
        organization="zai",
        model=MODEL,
        api_key="zai-test-key",
    )
    transport = FakeSyncTransport(zai_response())
    adapter.adapter._sync_transport = transport
    tools = [
        ToolSpec(name=f"tool_{index}", json_schema={"type": "object"})
        for index in range(129)
    ]

    with pytest.raises(ValueError, match="at most 128"):
        adapter.chat([UserMessage("Choose a tool")], tools=tools)

    assert transport.requests == []


@pytest.mark.unit
def test_zai_preserves_tool_result_history_in_openai_payload(zai_runtime):
    adapter = UniversalLLMAPIAdapter(
        organization="zai",
        model=MODEL,
        api_key="zai-test-key",
    )
    transport = FakeSyncTransport(zai_response())
    adapter.adapter._sync_transport = transport

    adapter.chat(
        [
            UserMessage("What is the weather in Paris?"),
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        name="get_weather",
                        arguments={"city": "Paris"},
                        call_id="call-1",
                    ),
                ],
            ),
            ToolMessage(
                content='{"temperature_c": 18}',
                tool_call_id="call-1",
            ),
            UserMessage("Now summarize it."),
        ],
        tools=[WEATHER_TOOL],
        tool_choice="auto",
    )

    assert transport.requests[0].payload["messages"] == [
        {"role": "user", "content": "What is the weather in Paris?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Paris"}',
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": '{"temperature_c": 18}',
        },
        {"role": "user", "content": "Now summarize it."},
    ]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("image", "expected_url"),
    [
        (
            ImagePart(url="https://example.test/cat.png"),
            "https://example.test/cat.png",
        ),
        (
            ImagePart(url="data:image/png;base64,aW1hZ2U="),
            "data:image/png;base64,aW1hZ2U=",
        ),
        (
            ImagePart(data=b"image", media_type="image/png"),
            "data:image/png;base64,aW1hZ2U=",
        ),
    ],
    ids=["url", "data-url", "bytes-as-data-url"],
)
def test_zai_serializes_image_url_and_data_url_parts(
    zai_runtime,
    image,
    expected_url,
):
    adapter = UniversalLLMAPIAdapter(
        organization="zai",
        model=MODEL,
        api_key="zai-test-key",
    )
    transport = FakeSyncTransport(zai_response())
    adapter.adapter._sync_transport = transport

    adapter.chat(
        [UserMessage("Describe this image", files=[image])],
    )

    assert transport.requests[0].payload["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image"},
                {
                    "type": "image_url",
                    "image_url": {"url": expected_url},
                },
            ],
        },
    ]


@pytest.mark.integration
def test_zai_facade_chat_uses_official_endpoint_and_normalizes_response(zai_runtime):
    adapter = UniversalLLMAPIAdapter(
        organization="zai",
        model=MODEL,
        api_key="zai-test-key",
    )
    transport = FakeSyncTransport(zai_response())
    adapter.adapter._sync_transport = transport

    response = adapter.chat(
        [UserMessage("Hello")],
        max_tokens=64,
        timeout_s=12.5,
    )

    assert response.content == "Z.ai test."
    assert response.response_id == "cmpl-zai-test"
    assert response.model == MODEL
    assert response.usage is not None
    assert response.usage.total_tokens == 32

    request = transport.requests[0]
    assert request.url == "https://api.z.ai/api/paas/v4/chat/completions"
    assert request.headers_dict() == {
        "Authorization": "Bearer zai-test-key",
        "Content-Type": "application/json",
    }
    assert request.timeout == 12.5
    assert request.payload["model"] == MODEL
    assert request.payload["messages"] == [{"role": "user", "content": "Hello"}]
    assert request.payload["max_tokens"] == 64
    assert "stream" not in request.payload


@pytest.mark.integration
def test_zai_sync_stream_reconstructs_deltas_and_usage(zai_runtime):
    adapter = UniversalLLMAPIAdapter(
        organization="zai",
        model=MODEL,
        api_key="zai-test-key",
    )
    transport = FakeSyncTransport({}, stream_events=zai_stream_events())
    adapter.adapter._sync_transport = transport
    completed = []
    reasoning = []

    output = list(
        adapter.stream_chat(
            [UserMessage("Hello")],
            max_tokens=64,
            capture_reasoning=True,
            on_done=completed.append,
            on_reasoning=reasoning.append,
        )
    )

    assert output == ["Hello ", "world"]
    assert completed[0].content == "Hello world"
    assert completed[0].usage is not None
    assert completed[0].usage.total_tokens == 32
    assert [event.text for event in completed[0].reasoning_events] == [
        "First reason. ",
    ]
    assert [event.text for event in reasoning] == ["First reason. "]
    assert all("First reason." not in text for text in output)
    assert transport.requests[0].url == (
        "https://api.z.ai/api/paas/v4/chat/completions"
    )
    assert transport.requests[0].payload["stream"] is True
    assert transport.sse_closed is True


def test_zai_httpx_async_chat_matches_sync_request_contract(zai_runtime, monkeypatch):
    from llm_api_adapter_zai.clients import async_client as async_client_module

    requests = []

    async def fake_async_request(url, **kwargs):
        requests.append((url, kwargs))
        return zai_response()

    monkeypatch.setattr(async_client_module, "async_request", fake_async_request)
    adapter = UniversalLLMAPIAdapter(
        organization="zai",
        model=MODEL,
        api_key="zai-test-key",
    )

    response = asyncio.run(
        adapter.achat(
            [UserMessage("Hello")],
            max_tokens=64,
            timeout_s=12.5,
        )
    )

    assert response.content == "Z.ai test."
    assert response.usage is not None
    assert response.usage.total_tokens == 32
    assert requests[0][0] == "https://api.z.ai/api/paas/v4/chat/completions"
    assert requests[0][1]["headers"] == {
        "Authorization": "Bearer zai-test-key",
        "Content-Type": "application/json",
    }
    assert requests[0][1]["payload"] == {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 64,
    }
    assert requests[0][1]["timeout"] == 12.5


def test_zai_httpx_async_stream_matches_sync_sse_lifecycle(
    zai_runtime,
    monkeypatch,
):
    from llm_api_adapter_zai.clients import async_client as async_client_module

    requests = []
    stream_closed = False

    def fake_async_stream_request(url, **kwargs):
        requests.append((url, kwargs))

        async def events():
            nonlocal stream_closed
            try:
                for event in zai_stream_events():
                    yield event
            finally:
                stream_closed = True

        return events()

    monkeypatch.setattr(
        async_client_module,
        "async_stream_request",
        fake_async_stream_request,
    )
    adapter = UniversalLLMAPIAdapter(
        organization="zai",
        model=MODEL,
        api_key="zai-test-key",
    )
    completed = []

    async def collect() -> list[str]:
        output = []
        async for text in adapter.astream_chat(
            [UserMessage("Hello")],
            max_tokens=64,
            capture_reasoning=True,
            on_done=completed.append,
        ):
            output.append(text)
        return output

    output = asyncio.run(collect())

    assert output == ["Hello ", "world"]
    assert completed[0].content == "Hello world"
    assert completed[0].usage is not None
    assert completed[0].usage.total_tokens == 32
    assert stream_closed is True
    assert requests[0][0] == "https://api.z.ai/api/paas/v4/chat/completions"
    assert requests[0][1]["payload"]["stream"] is True


@pytest.mark.parametrize(
    ("status_code", "error_type", "expected_error"),
    [
        (401, "authentication_error", LLMAPIAuthorizationError),
        (429, "rate_limit_error", LLMAPIRateLimitError),
        (500, "api_error", LLMAPIServerError),
    ],
)
def test_zai_normalizes_chat_completions_http_failures(
    zai_runtime,
    status_code,
    error_type,
    expected_error,
):
    adapter = UniversalLLMAPIAdapter(
        organization="zai",
        model=MODEL,
        api_key="zai-test-key",
    )
    adapter.adapter._sync_transport = FakeSyncTransport(
        {},
        error=FakeHTTPError(
            status_code,
            {"error": {"type": error_type, "message": "Z.ai failure"}},
        ),
    )

    with pytest.raises(expected_error, match="Z.ai failure"):
        adapter.chat([UserMessage("Hello")])


def test_zai_stream_error_is_mapped_and_closes_resources(zai_runtime):
    adapter = UniversalLLMAPIAdapter(
        organization="zai",
        model=MODEL,
        api_key="zai-test-key",
    )
    transport = FakeSyncTransport(
        {},
        stream_events=[
            SSEEvent(
                event="error",
                data={
                    "error": {
                        "type": "rate_limit_error",
                        "message": "Z.ai stream rate limit",
                    },
                },
            ),
        ],
    )
    adapter.adapter._sync_transport = transport

    with pytest.raises(LLMAPIRateLimitError, match="Z.ai stream rate limit"):
        list(adapter.stream_chat([UserMessage("Hello")]))

    assert transport.sse_closed is True


@pytest.mark.integration
def test_zai_reports_usage_and_verified_usd_costs(zai_runtime):
    adapter = UniversalLLMAPIAdapter(
        organization="zai",
        model=MODEL,
        api_key="zai-test-key",
    )
    adapter.adapter._sync_transport = FakeSyncTransport(
        zai_response(prompt_tokens=100, completion_tokens=40, cached_tokens=25),
    )

    response = adapter.chat([UserMessage("Price this request")])

    assert response.usage is not None
    assert response.usage.input_tokens == 100
    assert response.usage.output_tokens == 40
    assert response.usage.total_tokens == 140
    assert response.usage.cached_tokens == 25
    assert response.currency == "USD"
    assert response.cost_input == pytest.approx(
        (25 * 0.03 + 75 * 0.15) / 1_000_000,
    )
    assert response.cost_output == pytest.approx(40 * 0.50 / 1_000_000)
    assert response.cost_total == pytest.approx(
        response.cost_input + response.cost_output,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("cached_tokens", "expected_input_rate"),
    [
        (100, 0.03),
        (0, 0.15),
    ],
    ids=["all-cache-hit", "all-cache-miss"],
)
def test_zai_prices_cache_hit_and_cache_miss_rates(
    zai_runtime,
    cached_tokens,
    expected_input_rate,
):
    adapter = UniversalLLMAPIAdapter(
        organization="zai",
        model=MODEL,
        api_key="zai-test-key",
    )
    adapter.adapter._sync_transport = FakeSyncTransport(
        zai_response(
            prompt_tokens=100,
            completion_tokens=40,
            cached_tokens=cached_tokens,
        ),
    )

    response = adapter.chat([UserMessage("Price this request")])

    assert response.usage is not None
    assert response.usage.cached_tokens == cached_tokens
    assert response.cost_input == pytest.approx(
        100 * expected_input_rate / 1_000_000,
    )
    assert response.cost_output == pytest.approx(40 * 0.50 / 1_000_000)
    assert response.cost_total == pytest.approx(
        response.cost_input + response.cost_output,
    )


@pytest.mark.unit
def test_zai_uses_standard_miss_pricing_when_cache_split_is_missing(zai_runtime):
    adapter = UniversalLLMAPIAdapter(
        organization="zai",
        model=MODEL,
        api_key="zai-test-key",
    )
    adapter.adapter._sync_transport = FakeSyncTransport(
        zai_response(prompt_tokens=100, completion_tokens=40),
    )

    response = adapter.chat([UserMessage("Price without cache details")])

    assert response.usage is not None
    assert response.usage.cached_tokens is None
    assert response.cost_input == pytest.approx(100 * 0.15 / 1_000_000)
    assert response.cost_output == pytest.approx(40 * 0.50 / 1_000_000)
    assert response.cost_total == pytest.approx(
        response.cost_input + response.cost_output,
    )


@pytest.mark.unit
def test_zai_leaves_cost_unset_when_usage_is_missing(zai_runtime):
    payload = zai_response()
    payload.pop("usage")
    adapter = UniversalLLMAPIAdapter(
        organization="zai",
        model=MODEL,
        api_key="zai-test-key",
    )
    adapter.adapter._sync_transport = FakeSyncTransport(payload)

    response = adapter.chat([UserMessage("No usage please")])

    assert response.usage is None
    assert response.cost_input is None
    assert response.cost_output is None
    assert response.cost_total is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "usage",
    [
        {},
        {"prompt_tokens": 100, "completion_tokens": 40},
        {
            "prompt_tokens": "100",
            "completion_tokens": 40,
            "total_tokens": 140,
        },
        {"prompt_tokens": -1, "completion_tokens": 40, "total_tokens": 39},
        {"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 999},
    ],
)
def test_zai_leaves_cost_unset_for_malformed_usage(zai_runtime, usage):
    payload = zai_response()
    payload["usage"] = usage
    adapter = UniversalLLMAPIAdapter(
        organization="zai",
        model=MODEL,
        api_key="zai-test-key",
    )
    adapter.adapter._sync_transport = FakeSyncTransport(payload)

    response = adapter.chat([UserMessage("Validate usage")])

    assert response.usage is None
    assert response.cost_input is None
    assert response.cost_output is None
    assert response.cost_total is None


@pytest.mark.unit
@pytest.mark.parametrize("cached_tokens", [True, "25", -1, 101])
def test_zai_does_not_apply_cache_discount_for_invalid_cache_tokens(
    zai_runtime,
    cached_tokens,
):
    payload = zai_response(prompt_tokens=100, completion_tokens=40)
    payload["usage"]["prompt_tokens_details"] = {
        "cached_tokens": cached_tokens,
    }
    adapter = UniversalLLMAPIAdapter(
        organization="zai",
        model=MODEL,
        api_key="zai-test-key",
    )
    adapter.adapter._sync_transport = FakeSyncTransport(payload)

    response = adapter.chat([UserMessage("Ignore malformed cache details")])

    assert response.usage is not None
    assert response.usage.cached_tokens is None
    assert response.cost_input == pytest.approx(100 * 0.15 / 1_000_000)
    assert response.cost_output == pytest.approx(40 * 0.50 / 1_000_000)


@pytest.mark.unit
def test_zai_cache_pricing_calculates_only_validated_token_splits():
    pricing = ZaiCachePricing(
        cache_hit_input_per_token=0.03 / 1_000_000,
        cache_miss_input_per_token=0.15 / 1_000_000,
        output_per_token=0.50 / 1_000_000,
    )

    estimate = pricing.calculate(
        input_tokens=100,
        output_tokens=40,
        cached_tokens=25,
    )

    assert estimate is not None
    assert estimate.input_cost == pytest.approx(
        (25 * 0.03 + 75 * 0.15) / 1_000_000,
    )
    assert estimate.output_cost == pytest.approx(40 * 0.50 / 1_000_000)
    assert estimate.total_cost == pytest.approx(
        estimate.input_cost + estimate.output_cost,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "cached_tokens",
    [None, True, "25", -1, 101],
)
def test_zai_cache_pricing_returns_no_estimate_for_invalid_split(cached_tokens):
    pricing = ZaiCachePricing(
        cache_hit_input_per_token=0.03 / 1_000_000,
        cache_miss_input_per_token=0.15 / 1_000_000,
        output_per_token=0.50 / 1_000_000,
    )

    assert pricing.calculate(
        input_tokens=100,
        output_tokens=40,
        cached_tokens=cached_tokens,
    ) is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "rates",
    [
        (-1.0, 0.15, 0.50),
        (float("nan"), 0.15, 0.50),
        (0.03, float("inf"), 0.50),
        (True, 0.15, 0.50),
    ],
)
def test_zai_cache_pricing_rejects_invalid_rates(rates):
    with pytest.raises(ValueError, match="finite non-negative"):
        ZaiCachePricing(*rates)
