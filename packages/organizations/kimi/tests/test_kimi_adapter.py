"""Deterministic contracts for Kimi's synchronous Chat Completions slice."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
import sys
import warnings

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[2]
CORE_SOURCE = REPOSITORY_ROOT / "src"
PACKAGE_SOURCE = PACKAGE_ROOT / "src"
for source in (str(PACKAGE_SOURCE), str(CORE_SOURCE), str(REPOSITORY_ROOT)):
    if source not in sys.path:
        sys.path.insert(0, source)

import llm_api_adapter.adapters.base_adapter as base_adapter_module
import llm_api_adapter.organization_registry as organization_registry_module
import llm_api_adapter.universal_adapter as universal_module
from llm_api_adapter.errors.llm_api_error import (
    LLMAPIAuthorizationError,
    LLMAPIClientError,
    LLMAPIRateLimitError,
    LLMAPIServerError,
    LLMAPITimeoutError,
    LLMAPITokenLimitError,
    LLMAPIUsageLimitError,
)
from llm_api_adapter.errors.llm_api_error import JSONSchemaError
from llm_api_adapter.llm_registry.llm_registry import RegistrySpec, resolve_model_spec
from llm_api_adapter.llms.transports import JSONResponse, SSEEvent
from llm_api_adapter.models.messages.chat_message import UserMessage
from llm_api_adapter.models.messages.file_parts import DocumentPart, ImagePart
from llm_api_adapter.models.responses.chat_response import ChatResponse
from llm_api_adapter.models.tools.tool_spec import ToolSpec
from llm_api_adapter.organization_registry import (
    ORGANIZATION_PLUGIN_ENTRY_POINT_GROUP,
    OrganizationPluginDiscovery,
)
from llm_api_adapter.service_provider_registry import ServiceProviderRegistry
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter
from llm_api_adapter_kimi.adapter import KimiAdapter
from llm_api_adapter_kimi.plugin import PLUGIN
from tests.fixtures.structured_output import (
    FLAT_OBJECT_SCHEMA,
    INVALID_JSON_CONTENT,
    NESTED_PYDANTIC_RESPONSE_JSON,
    NestedPydanticResponse,
)


KIMI_MODELS = ("kimi-k3", "kimi-k2.7-code", "kimi-k2.6")
WEATHER_TOOL = ToolSpec(
    name="get_weather",
    description="Return the current weather for a city.",
    json_schema={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
)


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


def kimi_response(model: str, *, cached_tokens: int | None = 12) -> dict:
    usage = {
        "prompt_tokens": 19,
        "completion_tokens": 13,
        "total_tokens": 32,
    }
    if cached_tokens is not None:
        usage["cached_tokens"] = cached_tokens
    return {
        "id": "cmpl-kimi-test",
        "created": 1_789_721_600,
        "model": model,
        "choices": [
            {
                "message": {"role": "assistant", "content": "Kimi test."},
                "finish_reason": "stop",
            },
        ],
        "usage": usage,
    }


def kimi_tool_response(model: str) -> dict:
    response = kimi_response(model)
    response["choices"][0] = {
        "message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_kimi_weather",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city":"Tel Aviv"}',
                    },
                },
            ],
        },
        "finish_reason": "tool_calls",
    }
    return response


def kimi_structured_response(
    model: str,
    content: str | None,
    *,
    finish_reason: str = "stop",
    refusal: str | None = None,
) -> dict:
    response = kimi_response(model)
    message: dict[str, str | None] = {"role": "assistant", "content": content}
    if refusal is not None:
        message["refusal"] = refusal
    response["choices"][0] = {
        "message": message,
        "finish_reason": finish_reason,
    }
    return response


def kimi_chat_sse_events(model: str = "kimi-k3") -> list[SSEEvent]:
    """A complete OpenAI-compatible Kimi stream, including final usage."""
    stream_metadata = {
        "id": "cmpl-kimi-stream-1",
        "created": 1_789_721_600,
        "model": model,
    }
    return [
        SSEEvent(
            event=None,
            data={
                **stream_metadata,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": ""},
                        "finish_reason": None,
                    }
                ],
            },
        ),
        SSEEvent(
            event=None,
            data={
                **stream_metadata,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "reasoning_content": "First reason. ",
                            "content": "Hello ",
                        },
                        "finish_reason": None,
                    }
                ],
            },
        ),
        SSEEvent(
            event=None,
            data={
                **stream_metadata,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "world"},
                        "finish_reason": "stop",
                    }
                ],
            },
        ),
        SSEEvent(
            event=None,
            data={
                **stream_metadata,
                "choices": [],
                "usage": {
                    "prompt_tokens": 19,
                    "completion_tokens": 13,
                    "total_tokens": 32,
                    "cached_tokens": 12,
                },
            },
        ),
        SSEEvent(event=None, data="[DONE]", done=True),
    ]


def kimi_tool_call_sse_events(model: str = "kimi-k3") -> list[SSEEvent]:
    """Kimi chunks whose function arguments arrive in OpenAI-style pieces."""
    stream_metadata = {
        "id": "cmpl-kimi-tool-stream-1",
        "created": 1_789_721_600,
        "model": model,
    }
    return [
        SSEEvent(
            event=None,
            data={
                **stream_metadata,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_kimi_weather",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city":"Tel',
                                    },
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
                **stream_metadata,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": ' Aviv"}'},
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            },
        ),
        SSEEvent(
            event=None,
            data={
                **stream_metadata,
                "choices": [],
                "usage": {
                    "prompt_tokens": 19,
                    "completion_tokens": 13,
                    "total_tokens": 32,
                    "cached_tokens": 12,
                },
            },
        ),
        SSEEvent(event=None, data="[DONE]", done=True),
    ]


@pytest.fixture
def kimi_runtime(monkeypatch):
    from llm_api_adapter_kimi.plugin import PLUGIN

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
@pytest.mark.parametrize("model", KIMI_MODELS)
def test_kimi_plugin_registers_each_declared_model(kimi_runtime, model):
    from llm_api_adapter_kimi.adapter import KimiAdapter

    adapter = UniversalLLMAPIAdapter(
        organization="kimi",
        model=model,
        api_key="kimi-test-key",
    )

    assert isinstance(adapter.adapter, KimiAdapter)
    assert adapter.adapter.service_provider == "kimi"
    assert resolve_model_spec(kimi_runtime, "kimi", model) is adapter.adapter.model_spec
    assert adapter.adapter.model_spec is not None
    assert adapter.adapter.model_spec.limits.context_window_tokens in {
        262_144,
        1_048_576,
    }


@pytest.mark.unit
def test_kimi_metadata_records_the_pricing_snapshot_and_cache_split():
    from llm_api_adapter_kimi.registry import CACHE_PRICING, ORGANIZATION_DATA

    assert tuple(ORGANIZATION_DATA["models"]) == KIMI_MODELS
    for model in KIMI_MODELS:
        assert ORGANIZATION_DATA["models"][model]["pricing_as_of"] == "2026-09-14"
        assert CACHE_PRICING[model].cache_hit_input_per_token > 0
        assert CACHE_PRICING[model].cache_miss_input_per_token > 0


@pytest.mark.integration
@pytest.mark.parametrize("model", KIMI_MODELS)
def test_universal_chat_uses_chat_completions_for_every_declared_model(
    kimi_runtime,
    model,
):
    adapter = UniversalLLMAPIAdapter(
        organization="kimi",
        model=model,
        api_key="kimi-test-key",
    )
    transport = FakeSyncTransport(kimi_response(model))
    adapter.adapter._sync_transport = transport

    response = adapter.chat(
        [
            {"role": "system", "content": "Reply briefly."},
            {"role": "user", "content": "Hello"},
        ],
        max_tokens=64,
        timeout_s=12.5,
    )

    assert response.content == "Kimi test."
    assert response.response_id == "cmpl-kimi-test"
    assert response.usage is not None
    assert response.usage.total_tokens == 32

    request = transport.requests[0]
    assert request.url == "https://api.moonshot.ai/v1/chat/completions"
    assert request.headers_dict() == {
        "Authorization": "Bearer kimi-test-key",
        "Content-Type": "application/json",
    }
    assert request.timeout == 12.5
    expected_payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Reply briefly."},
            {"role": "user", "content": "Hello"},
        ],
    }
    expected_payload["max_completion_tokens" if model == "kimi-k3" else "max_tokens"] = 64
    assert request.payload == expected_payload


@pytest.mark.integration
@pytest.mark.parametrize("model", KIMI_MODELS)
@pytest.mark.parametrize(
    ("image", "expected_url"),
    [
        (
            ImagePart(data=b"kimi-image", media_type="image/png"),
            "data:image/png;base64,a2ltaS1pbWFnZQ==",
        ),
        (
            ImagePart(url="data:image/webp;base64,a2ltaS13ZWJw"),
            "data:image/webp;base64,a2ltaS13ZWJw",
        ),
    ],
)
def test_kimi_maps_image_bytes_and_data_uris_for_every_declared_model(
    kimi_runtime,
    model,
    image,
    expected_url,
):
    adapter = UniversalLLMAPIAdapter(
        organization="kimi",
        model=model,
        api_key="kimi-test-key",
    )
    transport = FakeSyncTransport(kimi_response(model))
    adapter.adapter._sync_transport = transport

    response = adapter.chat([UserMessage("Describe this image.", files=[image])])

    assert response.content == "Kimi test."
    assert transport.requests[0].payload["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image."},
                {"type": "image_url", "image_url": {"url": expected_url}},
            ],
        },
    ]


@pytest.mark.unit
@pytest.mark.parametrize("model", KIMI_MODELS)
@pytest.mark.parametrize(
    ("file_part", "detail"),
    [
        (
            ImagePart(url="https://example.com/kimi.png"),
            "public ImagePart URLs",
        ),
        (
            DocumentPart(data=b"%PDF-kimi", media_type="application/pdf"),
            "does not support DocumentPart",
        ),
        (
            DocumentPart(url="https://example.com/kimi.pdf"),
            "does not support DocumentPart",
        ),
    ],
)
def test_kimi_rejects_file_inputs_that_cannot_meet_the_shared_contract(
    kimi_runtime,
    model,
    file_part,
    detail,
):
    adapter = UniversalLLMAPIAdapter(
        organization="kimi",
        model=model,
        api_key="kimi-test-key",
    )
    transport = FakeSyncTransport(kimi_response(model))
    adapter.adapter._sync_transport = transport

    with pytest.raises(ValueError, match=detail):
        adapter.chat([UserMessage("Read this file.", files=[file_part])])

    assert transport.requests == []


@pytest.mark.integration
@pytest.mark.parametrize("model", KIMI_MODELS)
@pytest.mark.parametrize(
    ("tool_choice", "expected_tool_choice"),
    [
        ("auto", "auto"),
        ("none", "none"),
        ("any", "required"),
        (
            {"type": "tool", "name": "get_weather"},
            {"type": "function", "function": {"name": "get_weather"}},
        ),
    ],
)
def test_kimi_maps_application_tools_and_returns_normalized_tool_calls(
    kimi_runtime,
    model,
    tool_choice,
    expected_tool_choice,
):
    adapter = UniversalLLMAPIAdapter(
        organization="kimi",
        model=model,
        api_key="kimi-test-key",
    )
    transport = FakeSyncTransport(kimi_tool_response(model))
    adapter.adapter._sync_transport = transport

    response = adapter.chat(
        [{"role": "user", "content": "What is the weather in Tel Aviv?"}],
        tools=[WEATHER_TOOL],
        tool_choice=tool_choice,
    )

    assert response.finish_reason == "tool_calls"
    assert response.tool_calls is not None
    assert response.tool_calls[0].name == "get_weather"
    assert response.tool_calls[0].arguments == {"city": "Tel Aviv"}
    assert response.tool_calls[0].call_id == "call_kimi_weather"
    assert transport.requests[0].payload["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Return the current weather for a city.",
                "parameters": WEATHER_TOOL.json_schema,
            },
        },
    ]
    assert transport.requests[0].payload["tool_choice"] == expected_tool_choice


@pytest.mark.integration
@pytest.mark.parametrize("model", KIMI_MODELS)
def test_kimi_uses_explicit_tool_history_and_ignores_previous_response(
    kimi_runtime,
    model,
):
    adapter = UniversalLLMAPIAdapter(
        organization="kimi",
        model=model,
        api_key="kimi-test-key",
    )
    transport = FakeSyncTransport(kimi_response(model))
    adapter.adapter._sync_transport = transport

    adapter.chat(
        [
            {"role": "user", "content": "What is the weather in Tel Aviv?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_kimi_weather",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city":"Tel Aviv"}',
                        },
                    },
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_kimi_weather",
                "content": "Sunny, 25 C",
            },
        ],
        tools=[WEATHER_TOOL],
        previous_response=ChatResponse(response_id="cmpl-previous"),
    )

    payload = transport.requests[0].payload
    assert "previous_response" not in payload
    assert "previous_response_id" not in payload
    assert payload["messages"][1] == {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_kimi_weather",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city": "Tel Aviv"}',
                },
            },
        ],
    }
    assert payload["messages"][2] == {
        "role": "tool",
        "tool_call_id": "call_kimi_weather",
        "content": "Sunny, 25 C",
    }


@pytest.mark.integration
@pytest.mark.parametrize("model", KIMI_MODELS)
def test_kimi_chat_supports_core_portable_json_schema(kimi_runtime, model):
    adapter = UniversalLLMAPIAdapter(
        organization="kimi",
        model=model,
        api_key="kimi-test-key",
    )
    transport = FakeSyncTransport(
        kimi_structured_response(model, '{"answer":"Hello"}'),
    )
    adapter.adapter._sync_transport = transport

    response = adapter.chat(
        [{"role": "user", "content": "Reply as JSON."}],
        json_schema=FLAT_OBJECT_SCHEMA,
    )

    assert response.parsed_json == {"answer": "Hello"}
    assert transport.requests[0].payload["response_format"] == {
        "type": "json_schema",
        "json_schema": FLAT_OBJECT_SCHEMA,
    }


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.parametrize("model", KIMI_MODELS)
async def test_kimi_achat_supports_pydantic_structured_output(
    kimi_runtime,
    monkeypatch,
    model,
):
    from llm_api_adapter_kimi.clients import async_client as async_client_module

    requests = []

    async def fake_async_request(url, **kwargs):
        requests.append((url, kwargs))
        return kimi_structured_response(model, NESTED_PYDANTIC_RESPONSE_JSON)

    monkeypatch.setattr(async_client_module, "async_request", fake_async_request)
    adapter = UniversalLLMAPIAdapter(
        organization="kimi",
        model=model,
        api_key="kimi-test-key",
    )

    response = await adapter.achat(
        [{"role": "user", "content": "Reply with a contact."}],
        response_model=NestedPydanticResponse,
    )

    assert response.parsed_json == {"contact": {"name": "Ada"}}
    assert response.parsed_model == NestedPydanticResponse(
        contact={"name": "Ada"},
    )
    serialized_schema = requests[0][1]["payload"]["response_format"]["json_schema"]
    contact_schema = serialized_schema["properties"]["contact"]
    assert contact_schema["additionalProperties"] is False
    assert contact_schema["properties"]["name"]["type"] == "string"
    assert contact_schema["required"] == ["name"]
    assert "$defs" not in serialized_schema


@pytest.mark.unit
@pytest.mark.parametrize("model", KIMI_MODELS)
@pytest.mark.parametrize(
    ("content", "finish_reason", "refusal", "attribute", "expected"),
    [
        ('{"answer":"refused"}', "stop", "policy", "refusal", "policy"),
        ('{"answer":"partial"}', "length", None, "incomplete_reason", "length"),
    ],
)
def test_kimi_structured_terminal_outcomes_are_not_parsed(
    kimi_runtime,
    model,
    content,
    finish_reason,
    refusal,
    attribute,
    expected,
):
    adapter = UniversalLLMAPIAdapter(
        organization="kimi",
        model=model,
        api_key="kimi-test-key",
    )
    adapter.adapter._sync_transport = FakeSyncTransport(
        kimi_structured_response(
            model,
            content,
            finish_reason=finish_reason,
            refusal=refusal,
        ),
    )

    response = adapter.chat(
        [{"role": "user", "content": "Reply as JSON."}],
        json_schema=FLAT_OBJECT_SCHEMA,
    )

    assert getattr(response, attribute) == expected
    assert response.parsed_json is None
    assert response.parsed_model is None


@pytest.mark.unit
def test_kimi_rejects_invalid_or_incompatible_structured_requests_before_transport(
    kimi_runtime,
):
    adapter = UniversalLLMAPIAdapter(
        organization="kimi",
        model="kimi-k3",
        api_key="kimi-test-key",
    )
    transport = FakeSyncTransport({})
    adapter.adapter._sync_transport = transport

    with pytest.raises(JSONSchemaError, match="Core portable profile"):
        adapter.chat(
            [{"role": "user", "content": "Reply as JSON."}],
            json_schema={"type": "object", "properties": {"answer": True}},
        )
    with pytest.raises(JSONSchemaError, match="json_schema and tools"):
        adapter.chat(
            [{"role": "user", "content": "Reply as JSON."}],
            json_schema=FLAT_OBJECT_SCHEMA,
            tools=[WEATHER_TOOL],
        )

    assert transport.requests == []


@pytest.mark.unit
def test_kimi_rejects_invalid_json_and_pydantic_structured_responses(kimi_runtime):
    adapter = UniversalLLMAPIAdapter(
        organization="kimi",
        model="kimi-k3",
        api_key="kimi-test-key",
    )
    transport = FakeSyncTransport(
        kimi_structured_response("kimi-k3", INVALID_JSON_CONTENT),
    )
    adapter.adapter._sync_transport = transport

    with pytest.raises(JSONSchemaError, match="not valid JSON"):
        adapter.chat(
            [{"role": "user", "content": "Reply as JSON."}],
            json_schema=FLAT_OBJECT_SCHEMA,
        )

    transport.response = kimi_structured_response("kimi-k3", '{"contact": {}}')
    with pytest.raises(JSONSchemaError, match="Pydantic validation"):
        adapter.chat(
            [{"role": "user", "content": "Reply with a contact."}],
            response_model=NestedPydanticResponse,
        )


@pytest.mark.integration
@pytest.mark.parametrize(
    ("model", "cache_hit_rate", "cache_miss_rate", "output_rate"),
    [
        ("kimi-k3", 0.30, 3.00, 15.00),
        ("kimi-k2.7-code", 0.19, 0.95, 4.00),
        ("kimi-k2.6", 0.16, 0.95, 4.00),
    ],
)
def test_kimi_prices_reported_cache_hits_exactly(
    kimi_runtime,
    model,
    cache_hit_rate,
    cache_miss_rate,
    output_rate,
):
    adapter = UniversalLLMAPIAdapter(
        organization="kimi",
        model=model,
        api_key="kimi-test-key",
    )
    adapter.adapter._sync_transport = FakeSyncTransport(kimi_response(model))

    response = adapter.chat([{"role": "user", "content": "Hello"}])

    assert response.usage is not None
    assert response.usage.cached_tokens == 12
    assert response.currency == "USD"
    assert response.cost_input == pytest.approx(
        (12 * cache_hit_rate + 7 * cache_miss_rate) / 1_000_000,
    )
    assert response.cost_output == pytest.approx(13 * output_rate / 1_000_000)
    assert response.cost_total == pytest.approx(
        response.cost_input + response.cost_output,
    )


@pytest.mark.unit
def test_kimi_does_not_publish_a_total_when_the_cache_split_is_missing(kimi_runtime):
    adapter = UniversalLLMAPIAdapter(
        organization="kimi",
        model="kimi-k3",
        api_key="kimi-test-key",
    )
    adapter.adapter._sync_transport = FakeSyncTransport(
        kimi_response("kimi-k3", cached_tokens=None),
    )

    response = adapter.chat([{"role": "user", "content": "Hello"}])

    assert response.cost_input is None
    assert response.cost_total is None


@pytest.mark.unit
@pytest.mark.parametrize("model", KIMI_MODELS)
def test_kimi_drops_fixed_sampling_parameters_from_registry_rules(kimi_runtime, model):
    adapter = UniversalLLMAPIAdapter(
        organization="kimi",
        model=model,
        api_key="kimi-test-key",
    )
    transport = FakeSyncTransport(kimi_response(model))
    adapter.adapter._sync_transport = transport

    with pytest.warns(UserWarning, match="will be ignored"):
        adapter.chat(
            [{"role": "user", "content": "Hello"}],
            temperature=0.25,
            top_p=0.5,
        )

    assert "temperature" not in transport.requests[0].payload
    assert "top_p" not in transport.requests[0].payload


@pytest.mark.unit
def test_kimi_unknown_model_has_no_inferred_metadata_or_request_rules(kimi_runtime):
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        adapter = UniversalLLMAPIAdapter(
            organization="kimi",
            model="kimi-unverified-model",
            api_key="kimi-test-key",
        )
    transport = FakeSyncTransport(kimi_response("kimi-unverified-model"))
    adapter.adapter._sync_transport = transport

    response = adapter.chat(
        [{"role": "user", "content": "Hello"}],
        max_tokens=10,
        temperature=0.25,
        top_p=0.5,
    )

    assert any("not verified" in str(warning.message) for warning in captured)
    assert adapter.adapter.model_spec is None
    assert adapter.adapter.pricing is None
    assert transport.requests[0].payload["max_tokens"] == 10
    assert transport.requests[0].payload["temperature"] == 0.25
    assert transport.requests[0].payload["top_p"] == 0.5
    assert response.currency is None
    assert response.cost_total is None


@pytest.mark.unit
@pytest.mark.parametrize(
    ("status_code", "error_type", "expected_error"),
    [
        (401, "authentication_error", LLMAPIAuthorizationError),
        (400, "invalid_request_error", LLMAPIClientError),
        (429, "rate_limit_error", LLMAPIRateLimitError),
        (504, "timeout_error", LLMAPITimeoutError),
        (500, "api_error", LLMAPIServerError),
        (400, "max_tokens_exceeded", LLMAPITokenLimitError),
        (400, "quota_exceeded", LLMAPIUsageLimitError),
    ],
)
def test_kimi_normalizes_chat_completions_http_failures(
    kimi_runtime,
    status_code,
    error_type,
    expected_error,
):
    adapter = UniversalLLMAPIAdapter(
        organization="kimi",
        model="kimi-k3",
        api_key="kimi-test-key",
    )
    adapter.adapter._sync_transport = FakeSyncTransport(
        {},
        error=FakeHTTPError(
            status_code,
            {"error": {"type": error_type, "message": "Kimi test failure"}},
        ),
    )

    with pytest.raises(expected_error, match="Kimi test failure"):
        adapter.chat([{"role": "user", "content": "Hello"}])


@pytest.mark.unit
@pytest.mark.parametrize(
    ("payload", "detail"),
    [
        ({"choices": []}, "response.choices"),
        ({"choices": [{"message": {"content": 42}}]}, "content must be"),
        (
            {
                "choices": [{"message": {"content": "Hello"}}],
                "usage": {"prompt_tokens": "19", "completion_tokens": 13, "total_tokens": 32},
            },
            "usage token counts",
        ),
        (
            {
                "choices": [{"message": {"content": "Hello"}}],
                "usage": {"prompt_tokens": 19, "completion_tokens": 13, "total_tokens": 32, "cached_tokens": 20},
            },
            "must not exceed",
        ),
    ],
)
def test_kimi_rejects_malformed_chat_completions_responses(kimi_runtime, payload, detail):
    adapter = UniversalLLMAPIAdapter(
        organization="kimi",
        model="kimi-k3",
        api_key="kimi-test-key",
    )
    adapter.adapter._sync_transport = FakeSyncTransport(payload)

    with pytest.raises(LLMAPIClientError, match=detail):
        adapter.chat([{"role": "user", "content": "Hello"}])


@pytest.mark.unit
@pytest.mark.parametrize(
    ("model", "reasoning_level", "expected_options", "expected_warning"),
    [
        ("kimi-k3", "high", {"reasoning_effort": "high"}, None),
        ("kimi-k3", "very_high", {"reasoning_effort": "max"}, None),
        (
            "kimi-k3",
            "none",
            {"reasoning_effort": "low"},
            "cannot disable reasoning",
        ),
        ("kimi-k2.7-code", "high", {}, None),
        ("kimi-k2.7-code", "none", {}, "cannot disable reasoning"),
        ("kimi-k2.6", "high", {}, None),
        ("kimi-k2.6", "none", {"thinking": {"type": "disabled"}}, None),
    ],
)
def test_kimi_resolves_reasoning_controls_from_registry_metadata(
    kimi_runtime,
    model,
    reasoning_level,
    expected_options,
    expected_warning,
):
    """Keep K3 effort and K2 thinking choices out of adapter model branching."""
    adapter = UniversalLLMAPIAdapter(
        organization="kimi",
        model=model,
        api_key="kimi-test-key",
    )
    transport = FakeSyncTransport(kimi_response(model))
    adapter.adapter._sync_transport = transport

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        response = adapter.chat(
            [{"role": "user", "content": "Hello"}],
            reasoning_level=reasoning_level,
        )

    assert response.content == "Kimi test."
    payload = transport.requests[0].payload
    native_options = {
        key: payload[key]
        for key in ("reasoning_effort", "thinking")
        if key in payload
    }
    assert native_options == expected_options
    warning_messages = [str(item.message) for item in captured]
    if expected_warning is None:
        assert warning_messages == []
    else:
        assert any(expected_warning in message for message in warning_messages)


@pytest.mark.unit
def test_kimi_rejects_unsupported_parallel_tool_calls_before_transport(kimi_runtime):
    adapter = UniversalLLMAPIAdapter(
        organization="kimi",
        model="kimi-k3",
        api_key="kimi-test-key",
    )
    transport = FakeSyncTransport(kimi_response("kimi-k3"))
    adapter.adapter._sync_transport = transport

    with pytest.raises(NotImplementedError, match="parallel_tool_calls"):
        adapter.chat(
            [{"role": "user", "content": "Hello"}],
            tools=[WEATHER_TOOL],
            parallel_tool_calls=False,
        )

    assert transport.requests == []


@pytest.mark.integration
@pytest.mark.parametrize("model", KIMI_MODELS)
def test_kimi_stream_reconstructs_chat_completion_and_callback_order(
    kimi_runtime,
    model,
):
    adapter = UniversalLLMAPIAdapter(
        organization="kimi",
        model=model,
        api_key="kimi-test-key",
    )
    transport = FakeSyncTransport({}, events=kimi_chat_sse_events(model))
    adapter.adapter._sync_transport = transport
    callback_order = []
    completed = []

    def on_reasoning(event):
        callback_order.append(("reasoning", event.text))

    def on_chunk(chunk):
        callback_order.append(("chunk", chunk.text))

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
        capture_reasoning=True,
        on_reasoning=on_reasoning,
        on_chunk=on_chunk,
        on_delta=on_delta,
        on_done=on_done,
    ):
        callback_order.append(("yield", text))
        output.append(text)

    assert output == ["Hello ", "world"]
    assert callback_order == [
        ("reasoning", "First reason. "),
        ("chunk", "Hello "),
        ("delta", "Hello "),
        ("yield", "Hello "),
        ("chunk", "world"),
        ("delta", "world"),
        ("yield", "world"),
        ("done", "Hello world"),
    ]
    assert len(completed) == 1
    assert completed[0].response_id == "cmpl-kimi-stream-1"
    assert completed[0].usage is not None
    assert completed[0].usage.total_tokens == 32
    assert completed[0].usage.cached_tokens == 12
    assert completed[0].currency == "USD"
    assert [event.text for event in completed[0].reasoning_events] == [
        "First reason. "
    ]
    assert transport.sse_closed is True

    request = transport.sse_requests[0]
    assert request.url == "https://api.moonshot.ai/v1/chat/completions"
    assert request.headers_dict() == {
        "Authorization": "Bearer kimi-test-key",
        "Content-Type": "application/json",
    }
    assert request.timeout == 12.5
    expected_payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    expected_payload["max_completion_tokens" if model == "kimi-k3" else "max_tokens"] = 64
    assert request.payload == expected_payload


@pytest.mark.integration
def test_kimi_stream_keeps_reasoning_out_of_visible_text_without_capture(kimi_runtime):
    adapter = UniversalLLMAPIAdapter(
        organization="kimi",
        model="kimi-k3",
        api_key="kimi-test-key",
    )
    adapter.adapter._sync_transport = FakeSyncTransport(
        {},
        events=kimi_chat_sse_events(),
    )
    completed = []

    output = list(
        adapter.stream_chat(
            [{"role": "user", "content": "Hello"}],
            max_tokens=64,
            on_done=completed.append,
        )
    )

    assert output == ["Hello ", "world"]
    assert completed[0].content == "Hello world"
    assert completed[0].reasoning_events == []


@pytest.mark.integration
@pytest.mark.parametrize("model", KIMI_MODELS)
def test_kimi_stream_finalizes_core_structured_output(kimi_runtime, model):
    events = kimi_chat_sse_events(model)
    events[1].data["choices"][0]["delta"]["content"] = '{"answer":"Hel'
    events[2].data["choices"][0]["delta"]["content"] = 'lo"}'
    adapter = UniversalLLMAPIAdapter(
        organization="kimi",
        model=model,
        api_key="kimi-test-key",
    )
    transport = FakeSyncTransport({}, events=events)
    adapter.adapter._sync_transport = transport
    completed = []

    output = list(
        adapter.stream_chat(
            [{"role": "user", "content": "Reply as JSON."}],
            json_schema=FLAT_OBJECT_SCHEMA,
            on_done=completed.append,
        )
    )

    assert output == ['{"answer":"Hel', 'lo"}']
    assert completed[0].parsed_json == {"answer": "Hello"}
    assert transport.sse_requests[0].payload["response_format"] == {
        "type": "json_schema",
        "json_schema": FLAT_OBJECT_SCHEMA,
    }


@pytest.mark.asyncio
@pytest.mark.integration
async def test_kimi_astream_finalizes_core_structured_output(
    kimi_runtime,
    monkeypatch,
):
    from llm_api_adapter_kimi.clients import async_client as async_client_module

    events = kimi_chat_sse_events()
    events[1].data["choices"][0]["delta"]["content"] = '{"answer":"Hel'
    events[2].data["choices"][0]["delta"]["content"] = 'lo"}'
    requests = []

    def fake_async_stream_request(url, **kwargs):
        requests.append((url, kwargs))

        async def stream_events():
            for event in events:
                yield event

        return stream_events()

    monkeypatch.setattr(
        async_client_module,
        "async_stream_request",
        fake_async_stream_request,
    )
    adapter = UniversalLLMAPIAdapter(
        organization="kimi",
        model="kimi-k3",
        api_key="kimi-test-key",
    )
    completed = []

    output = [
        text
        async for text in adapter.astream_chat(
            [{"role": "user", "content": "Reply as JSON."}],
            json_schema=FLAT_OBJECT_SCHEMA,
            on_done=completed.append,
        )
    ]

    assert output == ['{"answer":"Hel', 'lo"}']
    assert completed[0].parsed_json == {"answer": "Hello"}
    assert requests[0][1]["payload"]["response_format"] == {
        "type": "json_schema",
        "json_schema": FLAT_OBJECT_SCHEMA,
    }


@pytest.mark.integration
def test_kimi_stream_reconstructs_fragmented_tool_calls_before_done(kimi_runtime):
    adapter = UniversalLLMAPIAdapter(
        organization="kimi",
        model="kimi-k3",
        api_key="kimi-test-key",
    )
    transport = FakeSyncTransport({}, events=kimi_tool_call_sse_events())
    adapter.adapter._sync_transport = transport
    callback_order = []
    completed = []

    def on_tool_call(tool_call):
        callback_order.append(("tool", tool_call.name, tool_call.arguments))

    def on_done(response):
        completed.append(response)
        callback_order.append(("done", response.finish_reason))

    assert list(
        adapter.stream_chat(
            [{"role": "user", "content": "What is the weather?"}],
            max_tokens=64,
            tools=[WEATHER_TOOL],
            tool_choice="auto",
            on_tool_call=on_tool_call,
            on_done=on_done,
        )
    ) == []
    assert callback_order == [
        ("tool", "get_weather", {"city": "Tel Aviv"}),
        ("done", "tool_calls"),
    ]
    assert completed[0].tool_calls is not None
    assert completed[0].tool_calls[0].call_id == "call_kimi_weather"
    assert transport.sse_closed is True
    assert transport.sse_requests[0].payload["tools"][0]["function"]["name"] == "get_weather"
    assert transport.sse_requests[0].payload["tool_choice"] == "auto"


@pytest.mark.integration
@pytest.mark.parametrize("model", KIMI_MODELS)
def test_kimi_captures_nonstream_reasoning_only_when_requested(kimi_runtime, model):
    payload = kimi_response(model)
    payload["choices"][0]["message"]["reasoning_content"] = "First reason."
    adapter = UniversalLLMAPIAdapter(
        organization="kimi",
        model=model,
        api_key="kimi-test-key",
    )
    adapter.adapter._sync_transport = FakeSyncTransport(payload)

    without_capture = adapter.chat(
        [{"role": "user", "content": "Think carefully."}],
        max_tokens=64,
    )
    with_capture = adapter.chat(
        [{"role": "user", "content": "Think carefully."}],
        max_tokens=64,
        capture_reasoning=True,
    )

    assert without_capture.content == with_capture.content == "Kimi test."
    assert without_capture.reasoning_events == []
    assert [event.text for event in with_capture.reasoning_events] == [
        "First reason."
    ]


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.parametrize("model", KIMI_MODELS)
async def test_kimi_achat_uses_httpx_async_client(kimi_runtime, monkeypatch, model):
    from llm_api_adapter_kimi.clients import async_client as async_client_module

    requests = []

    async def fake_async_request(url, **kwargs):
        requests.append((url, kwargs))
        return kimi_response(model)

    monkeypatch.setattr(async_client_module, "async_request", fake_async_request)
    adapter = UniversalLLMAPIAdapter(
        organization="kimi",
        model=model,
        api_key="kimi-test-key",
    )

    response = await adapter.achat(
        [{"role": "user", "content": "Hello"}],
        max_tokens=64,
        timeout_s=12.5,
        tools=[WEATHER_TOOL],
        tool_choice="auto",
    )

    assert response.content == "Kimi test."
    assert response.usage is not None
    assert response.usage.total_tokens == 32
    assert response.currency == "USD"
    assert len(requests) == 1
    assert requests[0][0] == "https://api.moonshot.ai/v1/chat/completions"
    assert requests[0][1]["headers"] == {
        "Authorization": "Bearer kimi-test-key",
        "Content-Type": "application/json",
    }
    expected_payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Hello"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Return the current weather for a city.",
                    "parameters": WEATHER_TOOL.json_schema,
                },
            },
        ],
        "tool_choice": "auto",
    }
    expected_payload["max_completion_tokens" if model == "kimi-k3" else "max_tokens"] = 64
    assert requests[0][1]["payload"] == expected_payload
    assert requests[0][1]["timeout"] == 12.5


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.parametrize("model", KIMI_MODELS)
async def test_kimi_astream_matches_sync_lifecycle(kimi_runtime, monkeypatch, model):
    from llm_api_adapter_kimi.clients import async_client as async_client_module

    requests = []
    stream_closed = False

    def fake_async_stream_request(url, **kwargs):
        requests.append((url, kwargs))

        async def events():
            nonlocal stream_closed
            try:
                for event in kimi_chat_sse_events(model):
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
        organization="kimi",
        model=model,
        api_key="kimi-test-key",
    )
    callback_order = []
    completed = []

    async def on_chunk(chunk):
        callback_order.append(("chunk", chunk.text))

    def on_delta(text):
        callback_order.append(("delta", text))

    async def on_reasoning(event):
        callback_order.append(("reasoning", event.text))

    async def on_done(response):
        completed.append(response)
        callback_order.append(("done", response.content))

    output = []
    async for text in adapter.astream_chat(
        [{"role": "user", "content": "Hello"}],
        max_tokens=64,
        timeout_s=12.5,
        tools=[WEATHER_TOOL],
        tool_choice="auto",
        capture_reasoning=True,
        on_reasoning=on_reasoning,
        on_chunk=on_chunk,
        on_delta=on_delta,
        on_done=on_done,
    ):
        callback_order.append(("yield", text))
        output.append(text)

    assert output == ["Hello ", "world"]
    assert callback_order == [
        ("reasoning", "First reason. "),
        ("chunk", "Hello "),
        ("delta", "Hello "),
        ("yield", "Hello "),
        ("chunk", "world"),
        ("delta", "world"),
        ("yield", "world"),
        ("done", "Hello world"),
    ]
    assert completed[0].usage is not None
    assert completed[0].usage.total_tokens == 32
    assert [event.text for event in completed[0].reasoning_events] == [
        "First reason. "
    ]
    assert stream_closed is True
    assert requests[0][0] == "https://api.moonshot.ai/v1/chat/completions"
    assert requests[0][1]["payload"]["stream"] is True
    assert requests[0][1]["payload"]["stream_options"] == {"include_usage": True}
    assert requests[0][1]["payload"]["tools"][0]["function"]["name"] == "get_weather"
    assert requests[0][1]["payload"]["tool_choice"] == "auto"


@pytest.mark.unit
def test_kimi_stream_error_is_mapped_and_closes_resources(kimi_runtime):
    adapter = UniversalLLMAPIAdapter(
        organization="kimi",
        model="kimi-k3",
        api_key="kimi-test-key",
    )
    transport = FakeSyncTransport(
        {},
        events=[
            SSEEvent(
                event="error",
                data={
                    "error": {
                        "type": "rate_limit_error",
                        "message": "Kimi stream rate limit",
                    }
                },
            )
        ],
    )
    adapter.adapter._sync_transport = transport

    with pytest.raises(LLMAPIRateLimitError, match="Kimi stream rate limit"):
        list(
            adapter.stream_chat(
                [{"role": "user", "content": "Hello"}],
                max_tokens=64,
            )
        )

    assert transport.sse_closed is True


@pytest.mark.asyncio
@pytest.mark.unit
async def test_kimi_async_stream_error_is_mapped_and_closes_resources(
    kimi_runtime,
    monkeypatch,
):
    from llm_api_adapter_kimi.clients import async_client as async_client_module

    stream_closed = False

    def fake_async_stream_request(url, **kwargs):
        async def events():
            nonlocal stream_closed
            try:
                event = SSEEvent(
                    event="error",
                    data={
                        "error": {
                            "type": "rate_limit_error",
                            "message": "Kimi async stream rate limit",
                        }
                    },
                )
                kwargs["stream_error_handler"](event)
                yield event  # pragma: no cover - the handler always raises
            finally:
                stream_closed = True

        return events()

    monkeypatch.setattr(
        async_client_module,
        "async_stream_request",
        fake_async_stream_request,
    )
    adapter = UniversalLLMAPIAdapter(
        organization="kimi",
        model="kimi-k3",
        api_key="kimi-test-key",
    )

    with pytest.raises(LLMAPIRateLimitError, match="Kimi async stream rate limit"):
        [
            text
            async for text in adapter.astream_chat(
                [{"role": "user", "content": "Hello"}],
                max_tokens=64,
            )
        ]

    assert stream_closed is True


@pytest.mark.unit
def test_kimi_stream_close_before_completion_closes_resources(kimi_runtime):
    adapter = UniversalLLMAPIAdapter(
        organization="kimi",
        model="kimi-k3",
        api_key="kimi-test-key",
    )
    transport = FakeSyncTransport({}, events=kimi_chat_sse_events())
    adapter.adapter._sync_transport = transport
    completed = []

    stream = adapter.stream_chat(
        [{"role": "user", "content": "Hello"}],
        max_tokens=64,
        on_done=completed.append,
    )
    assert next(stream) == "Hello "
    stream.close()

    assert transport.sse_closed is True
    assert completed == []


@pytest.mark.asyncio
@pytest.mark.unit
async def test_kimi_async_stream_cancellation_closes_resources(
    kimi_runtime,
    monkeypatch,
):
    from llm_api_adapter_kimi.clients import async_client as async_client_module

    stream_closed = False
    stream_entered = asyncio.Event()
    never = asyncio.Event()

    def fake_async_stream_request(url, **kwargs):
        async def events():
            nonlocal stream_closed
            try:
                yield SSEEvent(
                    event=None,
                    data={
                        "id": "cmpl-kimi-cancel",
                        "model": "kimi-k3",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": "Hello "},
                                "finish_reason": None,
                            }
                        ],
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
        organization="kimi",
        model="kimi-k3",
        api_key="kimi-test-key",
    )
    completed = []
    stream = adapter.astream_chat(
        [{"role": "user", "content": "Hello"}],
        max_tokens=64,
        on_done=completed.append,
    )

    assert await stream.__anext__() == "Hello "
    pending_chunk = asyncio.create_task(stream.__anext__())
    await stream_entered.wait()
    pending_chunk.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending_chunk

    assert stream_closed is True
    assert completed == []


@dataclass(frozen=True)
class _InstalledKimiEntryPoint:
    """Minimal installed-distribution entry point exercised by the facade."""

    name: str = "kimi"
    value: str = "llm_api_adapter_kimi.plugin:PLUGIN"

    def load(self):
        return PLUGIN


@pytest.fixture
def installed_kimi_plugin(monkeypatch):
    """Discover the real Kimi plugin instead of registering it in the test."""
    service_providers = ServiceProviderRegistry()
    model_registry = RegistrySpec()

    def installed_entry_points(*, group: str):
        assert group == ORGANIZATION_PLUGIN_ENTRY_POINT_GROUP
        return [_InstalledKimiEntryPoint()]

    monkeypatch.setattr(
        organization_registry_module,
        "entry_points",
        installed_entry_points,
    )
    monkeypatch.setattr(
        universal_module,
        "SERVICE_PROVIDER_REGISTRY",
        service_providers,
    )
    monkeypatch.setattr(
        universal_module,
        "ORGANIZATION_PLUGIN_DISCOVERY",
        OrganizationPluginDiscovery(),
    )
    monkeypatch.setattr(universal_module, "LLM_REGISTRY", model_registry)
    monkeypatch.setattr(base_adapter_module, "LLM_REGISTRY", model_registry)
    return model_registry


@pytest.mark.unit
@pytest.mark.parametrize("model", KIMI_MODELS)
def test_installed_kimi_plugin_conforms_through_the_public_facade(
    installed_kimi_plugin,
    monkeypatch,
    model,
):
    """Keep plugin discovery plus requests/HTTPX/SSE coverage credential-free."""
    from llm_api_adapter_kimi.clients import async_client as async_client_module

    adapter = UniversalLLMAPIAdapter(
        organization="kimi",
        model=model,
        api_key="kimi-test-key",
    )

    assert isinstance(adapter.adapter, KimiAdapter)
    assert set(installed_kimi_plugin.organizations["kimi"].models) == set(KIMI_MODELS)

    sync_transport = FakeSyncTransport(kimi_response(model))
    adapter.adapter._sync_transport = sync_transport
    sync_response = adapter.chat([UserMessage("Hello")], max_tokens=64)

    assert sync_response.content == "Kimi test."
    assert sync_response.usage is not None
    assert sync_response.usage.total_tokens == 32
    assert sync_response.cost_total is not None
    assert sync_transport.requests[0].payload["model"] == model

    async_requests = []

    async def fake_async_request(url, **kwargs):
        async_requests.append((url, kwargs))
        return kimi_response(model)

    monkeypatch.setattr(async_client_module, "async_request", fake_async_request)
    async_response = asyncio.run(
        adapter.achat([UserMessage("Hello")], max_tokens=64),
    )

    assert async_response.content == sync_response.content
    assert async_response.usage == sync_response.usage
    assert async_response.cost_total == sync_response.cost_total
    assert async_requests[0][0] == "https://api.moonshot.ai/v1/chat/completions"
    assert async_requests[0][1]["payload"]["model"] == model

    stream_transport = FakeSyncTransport({}, events=kimi_chat_sse_events(model))
    adapter.adapter._sync_transport = stream_transport
    sync_completed = []
    sync_output = list(
        adapter.stream_chat(
            [UserMessage("Hello")],
            max_tokens=64,
            on_done=sync_completed.append,
        )
    )

    assert sync_output == ["Hello ", "world"]
    assert sync_completed[0].content == "Hello world"
    assert stream_transport.sse_closed is True

    async_stream_requests = []

    def fake_async_stream_request(url, **kwargs):
        async_stream_requests.append((url, kwargs))

        async def events():
            for event in kimi_chat_sse_events(model):
                yield event

        return events()

    monkeypatch.setattr(
        async_client_module,
        "async_stream_request",
        fake_async_stream_request,
    )

    async def consume_async_stream():
        completed = []
        output = [
            text
            async for text in adapter.astream_chat(
                [UserMessage("Hello")],
                max_tokens=64,
                on_done=completed.append,
            )
        ]
        return output, completed

    async_output, async_completed = asyncio.run(consume_async_stream())

    assert async_output == sync_output
    assert async_completed[0].content == sync_completed[0].content
    assert async_stream_requests[0][0] == "https://api.moonshot.ai/v1/chat/completions"
    assert async_stream_requests[0][1]["payload"]["model"] == model
