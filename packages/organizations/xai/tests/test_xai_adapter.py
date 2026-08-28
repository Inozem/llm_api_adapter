"""Deterministic contract tests for xAI's initial Responses API adapter."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping

import pytest
from pydantic import BaseModel

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CORE_SOURCE = PACKAGE_ROOT.parents[2] / "src"
PACKAGE_SOURCE = PACKAGE_ROOT / "src"
for source in (str(PACKAGE_SOURCE), str(CORE_SOURCE)):
    if source not in sys.path:
        sys.path.insert(0, source)

import llm_api_adapter.adapters.base_adapter as base_adapter_module
import llm_api_adapter.universal_adapter as universal_module
from llm_api_adapter.errors.llm_api_error import (
    InvalidToolSchemaError,
    JSONSchemaError,
    LLMAPIAuthorizationError,
    LLMAPIClientError,
    LLMAPIRateLimitError,
    LLMAPIServerError,
    LLMAPITimeoutError,
)
from llm_api_adapter.llm_registry.llm_registry import OrganizationSpec, RegistrySpec
from llm_api_adapter.llms.transports import (
    JSONResponse,
    MultipartForm,
    SSEEvent,
    SyncTransport,
    TransportRequest,
)
from llm_api_adapter.models.messages.chat_message import (
    AIMessage,
    Prompt,
    ToolMessage,
    UserMessage,
)
from llm_api_adapter.models.tools import ToolSpec
from llm_api_adapter.service_provider_registry import ServiceProviderRegistry
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter

from llm_api_adapter_xai.adapter import XAIAdapter
import llm_api_adapter_xai.clients.async_client as xai_async_client_module
from llm_api_adapter_xai.clients.sync_client import XAIResponsesSyncClient
from llm_api_adapter_xai.plugin import PLUGIN, register
from llm_api_adapter_xai.registry import MODEL_METADATA


class StructuredAnswer(BaseModel):
    answer: str


@dataclass
class FakeSyncTransport(SyncTransport):
    payload: Any
    requests: list[TransportRequest] = field(default_factory=list)
    stream_events: list[SSEEvent] = field(default_factory=list)

    def post_json(
        self,
        request: TransportRequest,
        *,
        http_error_handler=None,
    ) -> JSONResponse:
        self.requests.append(request)
        return JSONResponse(self.payload)

    def post_multipart(
        self,
        request: TransportRequest,
        form: MultipartForm,
        *,
        http_error_handler=None,
    ) -> JSONResponse:
        raise AssertionError("xAI text chat must not upload multipart data")

    def post_sse(
        self,
        request: TransportRequest,
        *,
        http_error_handler=None,
        stream_error_handler=None,
    ) -> Iterator[SSEEvent]:
        self.requests.append(request)
        for event in self.stream_events:
            if (
                stream_error_handler is not None
                and (
                    event.event == "error"
                    or (
                        isinstance(event.data, Mapping)
                        and event.data.get("type") == "error"
                    )
                )
            ):
                stream_error_handler(event)
            yield event


def _response(*, model: str) -> dict[str, Any]:
    return {
        "object": "response",
        "id": f"resp-{model}",
        "model": model,
        "created_at": 1774274151,
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "Hello from xAI."}],
            },
        ],
        "usage": {
            "input_tokens": 20,
            "output_tokens": 5,
            "total_tokens": 25,
        },
    }


def _function_call_response(*, model: str) -> dict[str, Any]:
    response = _response(model=model)
    response["id"] = f"tool-{model}"
    response["output"] = [
        {
            "type": "function_call",
            "id": "fc_weather",
            "call_id": "call_weather",
            "name": "get_weather",
            "arguments": '{"city":"Haifa"}',
        }
    ]
    return response


def _structured_response(*, model: str) -> dict[str, Any]:
    response = _response(model=model)
    response["output"][0]["content"][0]["text"] = '{"answer":"ok"}'
    return response


def _stream_events(
    *,
    model: str,
    include_function_call: bool = False,
) -> list[SSEEvent]:
    response = _response(model=model)
    response["id"] = f"stream-{model}"
    events = [
        SSEEvent(
            event="response.created",
            data={
                "type": "response.created",
                "response": {
                    "id": response["id"],
                    "model": model,
                    "object": "response",
                    "status": "in_progress",
                },
            },
        ),
        SSEEvent(
            event="response.output_text.delta",
            data={"type": "response.output_text.delta", "delta": "Hello "},
        ),
        SSEEvent(
            event="response.output_text.delta",
            data={"type": "response.output_text.delta", "delta": "from xAI."},
        ),
    ]
    if include_function_call:
        function_call = {
            "type": "function_call",
            "id": "fc_456",
            "call_id": "call_456",
            "name": "get_weather",
            "arguments": '{"city":"Haifa"}',
        }
        response["output"].append(function_call)
        events.append(
            SSEEvent(
                event="response.output_item.done",
                data={
                    "type": "response.output_item.done",
                    "output_index": 1,
                    "item": function_call,
                },
            )
        )
    events.append(
        SSEEvent(
            event="response.completed",
            data={"type": "response.completed", "response": response},
        )
    )
    return events


@pytest.fixture
def xai_runtime(monkeypatch):
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
def test_plugin_registers_the_xai_service_provider():
    service_providers = ServiceProviderRegistry()

    register(service_providers)

    assert PLUGIN.model_metadata is MODEL_METADATA
    assert service_providers.get("xai") is XAIAdapter


@pytest.mark.unit
def test_model_metadata_validates_all_fixed_model_ids():
    organization = OrganizationSpec.from_dict("xai", MODEL_METADATA.organization_data)

    assert set(organization.models) == {
        "grok-4.3",
        "grok-4.5",
        "grok-4.6",
        "grok-build-0.1",
    }
    pricing = organization.models["grok-4.6"].pricing_tiers
    assert pricing.tier_for_prompt_tokens(199999).in_per_token == pytest.approx(
        0.000002
    )
    assert pricing.tier_for_prompt_tokens(200000).out_per_token == pytest.approx(
        0.000012
    )
    assert organization.models["grok-4.3"].reasoning_capability.allowed_values == (
        "none",
        "low",
        "medium",
        "high",
    )
    assert organization.models["grok-build-0.1"].reasoning_capability is None


@pytest.mark.unit
@pytest.mark.parametrize(
    ("model", "expected_total_cost"),
    [
        ("grok-4.3", 0.0000375),
        ("grok-4.5", 0.00007),
        ("grok-4.6", 0.00007),
        ("grok-build-0.1", 0.00003),
    ],
)
def test_universal_chat_maps_text_to_responses_api_and_normalizes_output(
    xai_runtime,
    model: str,
    expected_total_cost: float,
):
    adapter = UniversalLLMAPIAdapter(
        organization="xai",
        api_key="test-key",
        model=model,
    )
    assert isinstance(adapter.adapter, XAIAdapter)
    transport = FakeSyncTransport(_response(model=model))
    adapter.adapter._client._sync_transport = transport

    response = adapter.chat(
        messages=[Prompt("Be concise."), UserMessage("Hello")],
        max_tokens=12,
        temperature=0.5,
        top_p=0.8,
        timeout_s=3.0,
    )

    assert transport.requests == [
        TransportRequest(
            url="https://api.x.ai/v1/responses",
            headers={
                "Authorization": "Bearer test-key",
                "Content-Type": "application/json",
            },
            payload={
                "model": model,
                "input": [{"role": "user", "content": "Hello"}],
                "max_output_tokens": 12,
                "temperature": 0.5,
                "top_p": 0.8,
                "instructions": "Be concise.",
            },
            timeout=3.0,
        ),
    ]
    assert response.content == "Hello from xAI."
    assert response.response_id == f"resp-{model}"
    assert response.usage is not None
    assert response.usage.total_tokens == 25
    assert response.cost_total == pytest.approx(expected_total_cost)


@pytest.mark.unit
def test_chat_ignores_previous_response_without_sending_continuation_id(xai_runtime):
    adapter = XAIAdapter(api_key="test-key", model="grok-4.6")
    transport = FakeSyncTransport(_response(model="grok-4.6"))
    adapter._client._sync_transport = transport

    response = adapter.chat(
        messages=[UserMessage("Hello")],
        previous_response=object(),
    )

    assert response.content == "Hello from xAI."
    assert transport.requests[0].payload == {
        "model": "grok-4.6",
        "input": [{"role": "user", "content": "Hello"}],
        "temperature": 1.0,
        "top_p": 1.0,
    }


@pytest.mark.unit
def test_universal_chat_round_trips_application_tools_without_server_state(xai_runtime):
    tool = ToolSpec(
        name="get_weather",
        description="Get the weather for a city.",
        json_schema={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    )
    adapter = UniversalLLMAPIAdapter(
        organization="xai",
        api_key="test-key",
        model="grok-4.6",
    )
    first_transport = FakeSyncTransport(_function_call_response(model="grok-4.6"))
    adapter.adapter._client._sync_transport = first_transport

    first_response = adapter.chat(
        messages=[UserMessage("What is the weather in Haifa?")],
        tools=[tool],
        tool_choice="get_weather",
        parallel_tool_calls=False,
    )

    assert [(call.name, call.arguments, call.call_id) for call in first_response.tool_calls or []] == [
        ("get_weather", {"city": "Haifa"}, "call_weather"),
    ]
    assert first_transport.requests[0].payload == {
        "model": "grok-4.6",
        "input": [{"role": "user", "content": "What is the weather in Haifa?"}],
        "temperature": 1.0,
        "top_p": 1.0,
        "tools": [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get the weather for a city.",
                "parameters": tool.json_schema,
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        "parallel_tool_calls": False,
    }

    second_transport = FakeSyncTransport(_response(model="grok-4.6"))
    adapter.adapter._client._sync_transport = second_transport
    final_response = adapter.chat(
        messages=[
            UserMessage("What is the weather in Haifa?"),
            AIMessage(content="", tool_calls=first_response.tool_calls),
            ToolMessage(
                content='{"city":"Haifa","temperature":25}',
                tool_call_id="call_weather",
            ),
        ],
        tools=[tool],
        previous_response=first_response,
    )

    assert final_response.content == "Hello from xAI."
    second_payload = second_transport.requests[0].payload
    assert second_payload["input"] == [
        {"role": "user", "content": "What is the weather in Haifa?"},
        {
            "type": "function_call",
            "call_id": "call_weather",
            "name": "get_weather",
            "arguments": '{"city": "Haifa"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_weather",
            "output": '{"city":"Haifa","temperature":25}',
        },
    ]
    assert "previous_response_id" not in second_payload
    assert "store" not in second_payload


@pytest.mark.unit
@pytest.mark.parametrize(
    ("tool_choice", "expected"),
    [
        ("auto", "auto"),
        ("none", "none"),
        ("any", "required"),
        (
            "get_weather",
            {"type": "function", "function": {"name": "get_weather"}},
        ),
    ],
)
def test_xai_maps_normalized_tool_choice(tool_choice, expected):
    assert XAIAdapter._map_tool_choice(tool_choice) == expected


@pytest.mark.unit
def test_xai_rejects_function_without_required_description(xai_runtime):
    adapter = XAIAdapter(api_key="test-key", model="grok-4.6")

    with pytest.raises(InvalidToolSchemaError, match="requires a description"):
        adapter.chat(
            messages=[UserMessage("Hello")],
            tools=[
                ToolSpec(
                    name="get_weather",
                    json_schema={"type": "object"},
                )
            ],
        )


@pytest.mark.unit
def test_chat_maps_structured_pydantic_output_and_reasoning(xai_runtime):
    adapter = XAIAdapter(api_key="test-key", model="grok-4.6")
    transport = FakeSyncTransport(_structured_response(model="grok-4.6"))
    adapter._client._sync_transport = transport

    response = adapter.chat(
        messages=[UserMessage("Return an answer.")],
        response_model=StructuredAnswer,
        reasoning_level="xhigh",
    )

    expected_schema = StructuredAnswer.model_json_schema()
    assert response.parsed_json == {"answer": "ok"}
    assert response.parsed_model == StructuredAnswer(answer="ok")
    assert transport.requests[0].payload["text"] == {
        "format": {
            "type": "json_schema",
            "name": "response",
            "schema": expected_schema,
            "strict": True,
        }
    }
    assert transport.requests[0].payload["reasoning"] == {"effort": "xhigh"}


@pytest.mark.unit
@pytest.mark.parametrize(
    "schema",
    [
        {"type": "object", "properties": {"value": True}},
        {"type": "string", "enum": []},
        {"type": "array", "items": [{"type": "string"}]},
        {"type": "string", "pattern": r"\bword\b"},
    ],
)
def test_chat_rejects_documented_invalid_structured_schemas(xai_runtime, schema):
    adapter = XAIAdapter(api_key="test-key", model="grok-4.6")

    with pytest.raises(
        JSONSchemaError,
        match="xAI structured output rejects|does not support",
    ):
        adapter.chat(messages=[UserMessage("Hello")], json_schema=schema)


@pytest.mark.unit
def test_chat_accepts_explicit_additional_properties_in_structured_schema(
    xai_runtime,
):
    adapter = XAIAdapter(api_key="test-key", model="grok-4.6")
    adapter._client._sync_transport = FakeSyncTransport(
        _structured_response(model="grok-4.6"),
    )
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "additionalProperties": False,
    }

    response = adapter.chat(messages=[UserMessage("Hello")], json_schema=schema)

    assert response.parsed_json == {"answer": "ok"}


@pytest.mark.unit
def test_chat_omits_reasoning_for_a_model_without_effort_control(xai_runtime):
    adapter = XAIAdapter(api_key="test-key", model="grok-build-0.1")
    transport = FakeSyncTransport(_response(model="grok-build-0.1"))
    adapter._client._sync_transport = transport

    with pytest.warns(UserWarning, match="does not support reasoning"):
        response = adapter.chat(
            messages=[UserMessage("Hello")],
            reasoning_level="high",
        )

    assert response.content == "Hello from xAI."
    assert "reasoning" not in transport.requests[0].payload


@pytest.mark.unit
def test_chat_uses_grok_43_native_none_reasoning_level(xai_runtime):
    adapter = XAIAdapter(api_key="test-key", model="grok-4.3")
    transport = FakeSyncTransport(_response(model="grok-4.3"))
    adapter._client._sync_transport = transport

    adapter.chat(messages=[UserMessage("Hello")], reasoning_level="none")

    assert transport.requests[0].payload["reasoning"] == {"effort": "none"}


@pytest.mark.unit
def test_chat_warns_when_grok_45_cannot_disable_reasoning(xai_runtime):
    adapter = XAIAdapter(api_key="test-key", model="grok-4.5")
    transport = FakeSyncTransport(_response(model="grok-4.5"))
    adapter._client._sync_transport = transport

    with pytest.warns(UserWarning, match="cannot disable reasoning"):
        adapter.chat(messages=[UserMessage("Hello")], reasoning_level="none")

    assert transport.requests[0].payload["reasoning"] == {"effort": "low"}


@pytest.mark.unit
def test_chat_prefers_exact_xai_reported_cost(xai_runtime):
    adapter = XAIAdapter(api_key="test-key", model="grok-4.6")
    api_response = _response(model="grok-4.6")
    api_response["usage"]["cost_in_usd_ticks"] = 37_756_000
    adapter._client._sync_transport = FakeSyncTransport(api_response)

    response = adapter.chat(messages=[UserMessage("Hello")])

    assert response.currency == "USD"
    assert response.cost_input is None
    assert response.cost_output is None
    assert response.cost_total == pytest.approx(0.0037756)


@pytest.mark.unit
def test_chat_rejects_malformed_responses_payload(xai_runtime):
    adapter = XAIAdapter(api_key="test-key", model="grok-4.6")
    adapter._client._sync_transport = FakeSyncTransport({"object": "response"})

    with pytest.raises(LLMAPIClientError, match="response.output"):
        adapter.chat(messages=[UserMessage("Hello")])


@pytest.mark.unit
@pytest.mark.parametrize(
    ("status_code", "error_type", "error_class"),
    [
        (401, None, LLMAPIAuthorizationError),
        (429, None, LLMAPIRateLimitError),
        (504, None, LLMAPITimeoutError),
        (500, None, LLMAPIServerError),
        (400, "validation_error", LLMAPIClientError),
    ],
)
def test_client_maps_xai_errors(
    status_code: int,
    error_type: str | None,
    error_class: type[Exception],
):
    with pytest.raises(error_class):
        XAIResponsesSyncClient._raise_mapped_error(
            status_code=status_code,
            error_type=error_type,
            detail="test failure",
        )


@pytest.mark.unit
def test_stream_chat_maps_responses_sse_to_shared_lifecycle(xai_runtime):
    adapter = XAIAdapter(api_key="test-key", model="grok-4.6")
    transport = FakeSyncTransport(
        _response(model="grok-4.6"),
        stream_events=_stream_events(model="grok-4.6"),
    )
    adapter._client._sync_transport = transport
    callbacks: list[tuple[str, Any]] = []

    chunks = list(
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

    assert chunks == ["Hello ", "from x", "AI."]
    assert callbacks[:6] == [
        ("chunk", "Hello "),
        ("delta", "Hello "),
        ("chunk", "from x"),
        ("delta", "from x"),
        ("chunk", "AI."),
        ("delta", "AI."),
    ]
    assert callbacks[-1][0] == "done"
    final_response = callbacks[-1][1]
    assert final_response.content == "Hello from xAI."
    assert final_response.response_id == "stream-grok-4.6"
    assert final_response.cost_total == pytest.approx(0.00007)
    assert transport.requests[-1] == TransportRequest(
        url="https://api.x.ai/v1/responses",
        headers={
            "Authorization": "Bearer test-key",
            "Content-Type": "application/json",
        },
        payload={
            "model": "grok-4.6",
            "input": [{"role": "user", "content": "Hello"}],
            "max_output_tokens": 12,
            "temperature": 0.5,
            "top_p": 0.8,
            "instructions": "Be concise.",
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        timeout=3.0,
    )


@pytest.mark.unit
def test_stream_captures_reasoning_and_uses_final_exact_cost(xai_runtime):
    adapter = XAIAdapter(api_key="test-key", model="grok-4.6")
    events = _stream_events(model="grok-4.6")
    events.insert(
        1,
        SSEEvent(
            event="response.reasoning_summary_text.delta",
            data={
                "type": "response.reasoning_summary_text.delta",
                "delta": "Consider the constraints. ",
            },
        ),
    )
    events[-1].data["response"]["usage"]["cost_in_usd_ticks"] = 37_756_000
    adapter._client._sync_transport = FakeSyncTransport(
        _response(model="grok-4.6"),
        stream_events=events,
    )
    reasoning_events = []
    completed = []

    assert list(
        adapter.stream_chat(
            messages=[UserMessage("Hello")],
            capture_reasoning=True,
            on_reasoning=reasoning_events.append,
            on_done=completed.append,
        )
    ) == ["Hello ", "from xAI."]

    assert [event.text for event in reasoning_events] == [
        "Consider the constraints. ",
    ]
    assert completed[0].reasoning_events == reasoning_events
    assert completed[0].cost_total == pytest.approx(0.0037756)
    assert completed[0].cost_input is None
    assert completed[0].cost_output is None


@pytest.mark.unit
def test_stream_reconstructs_complete_function_output_item(xai_runtime):
    adapter = XAIAdapter(api_key="test-key", model="grok-4.6")
    adapter._client._sync_transport = FakeSyncTransport(
        _response(model="grok-4.6"),
        stream_events=[
            SSEEvent(
                event="response.created",
                data={
                    "type": "response.created",
                    "response": {
                        "id": "stream-function-call",
                        "model": "grok-4.6",
                        "status": "in_progress",
                    },
                },
            ),
            SSEEvent(
                event="response.output_text.delta",
                data={"type": "response.output_text.delta", "delta": "Use "},
            ),
            SSEEvent(
                event="response.output_item.done",
                data={
                    "type": "response.output_item.done",
                    "output_index": 1,
                    "item": {
                        "type": "function_call",
                        "id": "fc_123",
                        "call_id": "call_123",
                        "name": "get_weather",
                        "arguments": '{"city":"Haifa"}',
                    },
                },
            ),
            SSEEvent(
                event="response.output_text.delta",
                data={"type": "response.output_text.delta", "delta": "a tool."},
            ),
        ],
    )
    tool_calls = []
    completed = []

    assert list(
        adapter.stream_chat(
            messages=[UserMessage("Hello")],
            on_tool_call=tool_calls.append,
            on_done=completed.append,
        )
    ) == ["Use ", "a tool."]
    assert completed[0].content == "Use a tool."
    assert [(call.name, call.arguments, call.call_id) for call in tool_calls] == [
        ("get_weather", {"city": "Haifa"}, "call_123"),
    ]


@pytest.mark.unit
def test_stream_early_close_skips_completion_callback(xai_runtime):
    adapter = XAIAdapter(api_key="test-key", model="grok-4.6")
    adapter._client._sync_transport = FakeSyncTransport(
        _response(model="grok-4.6"),
        stream_events=_stream_events(model="grok-4.6"),
    )
    completed = []
    stream = adapter.stream_chat(
        messages=[UserMessage("Hello")],
        on_done=completed.append,
    )

    assert next(stream) == "Hello "
    stream.close()
    assert completed == []


@pytest.mark.unit
def test_stream_maps_xai_error_events(xai_runtime):
    adapter = XAIAdapter(api_key="test-key", model="grok-4.6")
    adapter._client._sync_transport = FakeSyncTransport(
        _response(model="grok-4.6"),
        stream_events=[
            SSEEvent(
                event="error",
                data={
                    "type": "error",
                    "error": {
                        "type": "rate_limit_error",
                        "message": "Slow down",
                    },
                },
            )
        ],
    )

    with pytest.raises(LLMAPIRateLimitError, match="Slow down"):
        list(adapter.stream_chat(messages=[UserMessage("Hello")]))


@pytest.mark.unit
def test_async_chat_applies_structured_output_and_reasoning(xai_runtime, monkeypatch):
    requests: list[dict[str, Any]] = []

    async def fake_async_request(
        url: str,
        *,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout: float | None,
        http_error_handler,
    ) -> dict[str, Any]:
        del url, headers, timeout, http_error_handler
        requests.append(payload)
        return _structured_response(model="grok-4.6")

    monkeypatch.setattr(xai_async_client_module, "async_request", fake_async_request)
    adapter = XAIAdapter(api_key="test-key", model="grok-4.6")

    response = asyncio.run(
        adapter.achat(
            messages=[UserMessage("Return an answer.")],
            response_model=StructuredAnswer,
            reasoning_level="high",
        )
    )

    assert response.parsed_json == {"answer": "ok"}
    assert response.parsed_model == StructuredAnswer(answer="ok")
    assert requests[0]["reasoning"] == {"effort": "high"}
    assert requests[0]["text"]["format"]["schema"] == (
        StructuredAnswer.model_json_schema()
    )


@pytest.mark.unit
def test_async_chat_and_stream_match_sync_contract(xai_runtime, monkeypatch):
    requests: list[dict[str, Any]] = []

    async def fake_async_request(
        url: str,
        *,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout: float | None,
        http_error_handler,
    ) -> dict[str, Any]:
        requests.append(
            {
                "url": url,
                "headers": headers,
                "payload": payload,
                "timeout": timeout,
            }
        )
        return _response(model="grok-4.6")

    def fake_async_stream_request(
        url: str,
        *,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout: float | None,
        http_error_handler,
        stream_error_handler,
    ):
        requests.append(
            {
                "url": url,
                "headers": headers,
                "payload": payload,
                "timeout": timeout,
            }
        )

        async def events():
            for event in _stream_events(
                model="grok-4.6",
                include_function_call=True,
            ):
                yield event

        return events()

    monkeypatch.setattr(xai_async_client_module, "async_request", fake_async_request)
    monkeypatch.setattr(
        xai_async_client_module,
        "async_stream_request",
        fake_async_stream_request,
    )
    adapter = XAIAdapter(api_key="test-key", model="grok-4.6")

    async def exercise_async_contract():
        response = await adapter.achat(
            messages=[Prompt("Be concise."), UserMessage("Hello")],
            max_tokens=12,
            temperature=0.5,
            top_p=0.8,
            timeout_s=3.0,
        )
        callback_events = []

        async def on_chunk(chunk):
            callback_events.append(("chunk", chunk.text))

        async def on_delta(text):
            callback_events.append(("delta", text))

        async def on_done(done_response):
            callback_events.append(("done", done_response))

        async def on_tool_call(tool_call):
            callback_events.append(("tool", tool_call))

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
            on_tool_call=on_tool_call,
            on_done=on_done,
        ):
            chunks.append(text)
        return response, chunks, callback_events

    response, chunks, callback_events = asyncio.run(exercise_async_contract())

    assert response.content == "Hello from xAI."
    assert chunks == ["Hello ", "from x", "AI."]
    assert callback_events[:2] == [("chunk", "Hello "), ("delta", "Hello ")]
    assert callback_events[-1][0] == "done"
    assert callback_events[-1][1].content == "Hello from xAI."
    assert callback_events[-2][0] == "tool"
    assert callback_events[-2][1].call_id == "call_456"
    assert requests == [
        {
            "url": "https://api.x.ai/v1/responses",
            "headers": {
                "Authorization": "Bearer test-key",
                "Content-Type": "application/json",
            },
            "payload": {
                "model": "grok-4.6",
                "input": [{"role": "user", "content": "Hello"}],
                "max_output_tokens": 12,
                "temperature": 0.5,
                "top_p": 0.8,
                "instructions": "Be concise.",
            },
            "timeout": 3.0,
        },
        {
            "url": "https://api.x.ai/v1/responses",
            "headers": {
                "Authorization": "Bearer test-key",
                "Content-Type": "application/json",
            },
            "payload": {
                "model": "grok-4.6",
                "input": [{"role": "user", "content": "Hello"}],
                "max_output_tokens": 12,
                "temperature": 0.5,
                "top_p": 0.8,
                "instructions": "Be concise.",
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            "timeout": 3.0,
        },
    ]


@pytest.mark.unit
def test_async_stream_maps_xai_error_events(xai_runtime, monkeypatch):
    def fake_async_stream_request(
        url: str,
        *,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout: float | None,
        http_error_handler,
        stream_error_handler,
    ):
        del url, headers, payload, timeout, http_error_handler

        async def events():
            stream_error_handler(
                SSEEvent(
                    event="error",
                    data={
                        "type": "error",
                        "error": {
                            "type": "rate_limit_error",
                            "message": "Slow down asynchronously",
                        },
                    },
                )
            )
            yield SSEEvent(event=None, done=True)

        return events()

    monkeypatch.setattr(
        xai_async_client_module,
        "async_stream_request",
        fake_async_stream_request,
    )
    adapter = XAIAdapter(api_key="test-key", model="grok-4.6")

    async def consume_error_stream():
        async for _ in adapter.astream_chat(messages=[UserMessage("Hello")]):
            pass

    with pytest.raises(LLMAPIRateLimitError, match="Slow down asynchronously"):
        asyncio.run(consume_error_stream())


@pytest.mark.unit
def test_async_stream_early_close_skips_completion_callback(xai_runtime):
    adapter = XAIAdapter(api_key="test-key", model="grok-4.6")
    completed = []

    async def events():
        for event in _stream_events(model="grok-4.6"):
            yield event

    adapter._async_client.stream = lambda **_: events()

    async def close_stream_early():
        stream = adapter.astream_chat(
            messages=[UserMessage("Hello")],
            on_done=lambda response: completed.append(response),
        )
        assert await anext(stream) == "Hello "
        await stream.aclose()

    asyncio.run(close_stream_early())
    assert completed == []
