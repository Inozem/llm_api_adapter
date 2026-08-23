"""Contract tests for the official direct Mistral adapter."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CORE_SOURCE = PACKAGE_ROOT.parents[1] / "src"
PACKAGE_SOURCE = PACKAGE_ROOT / "src"
for source in (str(PACKAGE_SOURCE), str(CORE_SOURCE)):
    if source not in sys.path:
        sys.path.insert(0, source)

import llm_api_adapter.adapters.base_adapter as base_adapter_module
import llm_api_adapter.universal_adapter as universal_module
import llm_api_adapter_mistral.adapter as mistral_adapter_module
from llm_api_adapter.errors.llm_api_error import LLMAPITokenLimitError
from llm_api_adapter.llm_registry.llm_registry import RegistrySpec, resolve_model_spec
from llm_api_adapter.llms.transports import JSONResponse, SSEEvent
from llm_api_adapter.models.tools import ToolSpec
from llm_api_adapter.provider_registry import ServiceProviderRegistry
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter
from llm_api_adapter_mistral.adapter import MistralAdapter
from llm_api_adapter_mistral.plugin import PLUGIN


class FakeSyncTransport:
    def __init__(self, response: dict, events=()) -> None:
        self.response = response
        self.events = list(events)
        self.requests = []
        self.stream_requests = []

    def post_json(self, request, *, http_error_handler=None):
        self.requests.append(request)
        return JSONResponse(self.response)

    def post_sse(
        self,
        request,
        *,
        http_error_handler=None,
        stream_error_handler=None,
    ):
        self.stream_requests.append(request)
        return iter(self.events)


@pytest.fixture
def mistral_runtime(monkeypatch):
    model_registry = RegistrySpec()
    assert PLUGIN.model_metadata is not None
    assert model_registry.register_provider_metadata(PLUGIN.model_metadata) is True

    service_registry = ServiceProviderRegistry()
    PLUGIN.register(service_registry)
    monkeypatch.setattr(universal_module, "LLM_REGISTRY", model_registry)
    monkeypatch.setattr(
        universal_module,
        "SERVICE_PROVIDER_REGISTRY",
        service_registry,
    )
    monkeypatch.setattr(base_adapter_module, "LLM_REGISTRY", model_registry)
    return model_registry


@pytest.mark.unit
def test_plugin_registers_verified_models_and_universal_adapter(mistral_runtime):
    adapter = UniversalLLMAPIAdapter(
        organization="mistral",
        model="mistral-small-2603",
        api_key="mistral-test-key",
    )

    assert isinstance(adapter.adapter, MistralAdapter)
    assert adapter.adapter.service_provider == "mistral"
    assert resolve_model_spec(
        mistral_runtime,
        "mistral",
        "mistral-small-2603",
    ) is adapter.adapter.model_spec
    assert resolve_model_spec(
        mistral_runtime,
        "mistral",
        "mistral-large-2512",
    ) is not None


@pytest.mark.unit
def test_universal_chat_builds_direct_mistral_payload_and_finalizes_pricing(
    mistral_runtime,
):
    adapter = UniversalLLMAPIAdapter(
        organization="mistral",
        model="mistral-small-2603",
        api_key="mistral-test-key",
    )
    transport = FakeSyncTransport(
        {
            "id": "cmpl-mistral-1",
            "created": 1_700_000_000,
            "model": "mistral-small-2603",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "Bonjour"},
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
        }
    )
    adapter.adapter._sync_transport = transport

    response = adapter.chat(
        [
            {"role": "system", "content": "Reply in French."},
            {"role": "user", "content": "Hello"},
        ],
        max_tokens=40,
        temperature=0.7,
        top_p=0.8,
        timeout_s=12.5,
    )

    assert response.content == "Bonjour"
    assert response.response_id == "cmpl-mistral-1"
    assert response.usage is not None
    assert response.usage.total_tokens == 30
    assert response.cost_input == pytest.approx(0.0000015)
    assert response.cost_output == pytest.approx(0.000012)
    assert response.cost_total == pytest.approx(0.0000135)

    request = transport.requests[0]
    assert request.url == "https://api.mistral.ai/v1/chat/completions"
    assert request.headers_dict() == {
        "Authorization": "Bearer mistral-test-key",
        "Content-Type": "application/json",
    }
    assert request.timeout == 12.5
    assert request.payload == {
        "model": "mistral-small-2603",
        "messages": [
            {"role": "system", "content": "Reply in French."},
            {"role": "user", "content": "Hello"},
        ],
        "max_tokens": 40,
        "temperature": 0.7,
        "top_p": 0.8,
        "reasoning_effort": "none",
    }


@pytest.mark.unit
def test_mistral_normalizes_tool_calls_and_thinking(
    mistral_runtime,
):
    adapter = UniversalLLMAPIAdapter(
        organization="mistral",
        model="mistral-medium-3-5",
        api_key="mistral-test-key",
    )
    transport = FakeSyncTransport(
        {
            "id": "cmpl-mistral-2",
            "model": "mistral-medium-3-5",
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": [
                            {
                                "type": "thinking",
                                "thinking": [
                                    {"type": "text", "text": "Need weather."}
                                ],
                            },
                            {"type": "text", "text": '{"city":"Paris"}'},
                        ],
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city":"Paris"}',
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 6},
        }
    )
    adapter.adapter._sync_transport = transport
    tool = ToolSpec(
        name="get_weather",
        description="Get a weather report",
        json_schema={"type": "object", "properties": {"city": {"type": "string"}}},
    )

    response = adapter.chat(
        [{"role": "user", "content": "What is the weather in Paris?"}],
        tools=[tool],
        tool_choice="get_weather",
        json_schema=None,
        reasoning_level="high",
        capture_reasoning=True,
    )

    assert response.content == '{"city":"Paris"}'
    assert response.tool_calls is not None
    assert response.tool_calls[0].name == "get_weather"
    assert response.tool_calls[0].arguments == {"city": "Paris"}
    assert [event.text for event in response.reasoning_events] == ["Need weather."]
    assert response.usage is not None
    assert response.usage.total_tokens == 11
    assert transport.requests[0].payload["reasoning_effort"] == "high"
    assert transport.requests[0].payload["tool_choice"] == {
        "type": "function",
        "function": {"name": "get_weather"},
    }


@pytest.mark.unit
def test_mistral_uses_official_json_schema_payload_shape(mistral_runtime):
    adapter = UniversalLLMAPIAdapter(
        organization="mistral",
        model="mistral-small-2603",
        api_key="mistral-test-key",
    )
    transport = FakeSyncTransport(
        {
            "model": "mistral-small-2603",
            "choices": [{"message": {"content": '{"answer":"Bonjour"}'}}],
        }
    )
    adapter.adapter._sync_transport = transport
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
    }

    response = adapter.chat(
        [{"role": "user", "content": "Reply as JSON."}],
        json_schema=schema,
    )

    assert response.parsed_json == {"answer": "Bonjour"}
    assert transport.requests[0].payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "response",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "additionalProperties": False,
            },
        },
    }


@pytest.mark.unit
def test_mistral_streams_text_tools_reasoning_and_usage(mistral_runtime):
    adapter = UniversalLLMAPIAdapter(
        organization="mistral",
        model="mistral-small-2603",
        api_key="mistral-test-key",
    )
    transport = FakeSyncTransport(
        {},
        events=[
            SSEEvent(
                event=None,
                data={
                    "id": "cmpl-mistral-stream-1",
                    "created": 1_700_000_001,
                    "model": "mistral-small-2603",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": [
                                    {
                                        "type": "thinking",
                                        "thinking": [
                                            {"type": "text", "text": "Plan answer."}
                                        ],
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                },
            ),
            SSEEvent(
                event=None,
                data={
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": [{"type": "text", "text": "Bon"}],
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_weather",
                                        "type": "function",
                                        "function": {
                                            "name": "get_weather",
                                            "arguments": '{"city":',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": None,
                        }
                    ]
                },
            ),
            SSEEvent(
                event=None,
                data={
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": "jour",
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {"arguments": '"Paris"}'},
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 5,
                        "completion_tokens": 6,
                        "total_tokens": 11,
                    },
                },
            ),
        ],
    )
    adapter.adapter._sync_transport = transport
    deltas = []
    chunks = []
    reasoning = []
    tools = []
    completed = []

    streamed = list(
        adapter.stream_chat(
            [{"role": "user", "content": "Say hello."}],
            buffer_chars=3,
            capture_reasoning=True,
            on_delta=deltas.append,
            on_chunk=chunks.append,
            on_reasoning=reasoning.append,
            on_tool_call=tools.append,
            on_done=completed.append,
        )
    )

    assert streamed == ["Bon", "jou", "r"]
    assert deltas == streamed
    assert [chunk.text for chunk in chunks] == streamed
    assert chunks[1].usage is not None
    assert chunks[1].usage.total_tokens == 11
    assert [event.text for event in reasoning] == ["Plan answer."]
    assert len(tools) == 1
    assert tools[0].name == "get_weather"
    assert tools[0].arguments == {"city": "Paris"}
    assert len(completed) == 1
    response = completed[0]
    assert response.content == "Bonjour"
    assert response.response_id == "cmpl-mistral-stream-1"
    assert response.usage is not None
    assert response.usage.total_tokens == 11
    assert response.cost_total == pytest.approx(0.00000435)

    stream_request = transport.stream_requests[0]
    assert stream_request.timeout is None
    assert stream_request.payload["stream"] is True
    assert stream_request.payload["stream_options"] == {"include_usage": True}


@pytest.mark.unit
def test_mistral_async_chat_and_stream_use_shared_async_transport(
    mistral_runtime,
    monkeypatch,
):
    adapter = UniversalLLMAPIAdapter(
        organization="mistral",
        model="mistral-small-2603",
        api_key="mistral-test-key",
    )
    request_calls = []
    stream_calls = []

    async def fake_async_request(url, **kwargs):
        request_calls.append((url, kwargs))
        return {
            "id": "cmpl-mistral-async-1",
            "model": "mistral-small-2603",
            "choices": [{"message": {"content": "Async hello"}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3},
        }

    def fake_async_stream_request(url, **kwargs):
        stream_calls.append((url, kwargs))

        async def events():
            yield SSEEvent(
                event=None,
                data={
                    "id": "cmpl-mistral-async-stream-1",
                    "model": "mistral-small-2603",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "Async "},
                            "finish_reason": None,
                        }
                    ],
                },
            )
            yield SSEEvent(
                event=None,
                data={
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "stream"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 2,
                        "completion_tokens": 4,
                        "total_tokens": 6,
                    },
                },
            )

        return events()

    monkeypatch.setattr(
        mistral_adapter_module,
        "async_request",
        fake_async_request,
    )
    monkeypatch.setattr(
        mistral_adapter_module,
        "async_stream_request",
        fake_async_stream_request,
    )

    async def run_requests():
        response = await adapter.achat(
            [{"role": "user", "content": "Say hello."}],
            timeout_s=7.5,
        )
        deltas = []
        completed = []

        async def on_delta(text):
            deltas.append(text)

        async def on_done(chat_response):
            completed.append(chat_response)

        streamed = [
            text
            async for text in adapter.astream_chat(
                [{"role": "user", "content": "Stream hello."}],
                timeout_s=8.5,
                on_delta=on_delta,
                on_done=on_done,
            )
        ]
        return response, streamed, deltas, completed

    response, streamed, deltas, completed = asyncio.run(run_requests())

    assert response.content == "Async hello"
    assert streamed == ["Async ", "stream"]
    assert deltas == streamed
    assert completed[0].content == "Async stream"
    assert completed[0].usage is not None
    assert completed[0].usage.total_tokens == 6
    assert request_calls == [
        (
            "https://api.mistral.ai/v1/chat/completions",
            {
                "headers": {
                    "Authorization": "Bearer mistral-test-key",
                    "Content-Type": "application/json",
                },
                "payload": {
                    "model": "mistral-small-2603",
                    "messages": [{"role": "user", "content": "Say hello."}],
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "reasoning_effort": "none",
                },
                "timeout": 7.5,
                "http_error_handler": adapter.adapter._handle_http_error,
            },
        )
    ]
    assert stream_calls[0][0] == "https://api.mistral.ai/v1/chat/completions"
    assert stream_calls[0][1]["timeout"] == 8.5
    assert stream_calls[0][1]["payload"]["stream"] is True
    assert stream_calls[0][1]["payload"]["stream_options"] == {
        "include_usage": True
    }


@pytest.mark.unit
def test_mistral_maps_context_errors_to_shared_error_hierarchy():
    with pytest.raises(LLMAPITokenLimitError):
        MistralAdapter._raise_mapped_error(
            status_code=400,
            error_type="context_length_exceeded",
            detail="context is too long",
        )
