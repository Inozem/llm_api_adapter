"""Deterministic contract tests for Qwen's Frankfurt Messages adapter."""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys
import warnings

import pytest
from pydantic import BaseModel, ConfigDict


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
    InvalidToolArgumentsError,
    JSONSchemaError,
    LLMAPIAuthorizationError,
    LLMAPIClientError,
    LLMAPIRateLimitError,
    LLMAPIServerError,
    LLMAPITimeoutError,
    LLMAPITokenLimitError,
    LLMAPIUsageLimitError,
)
from llm_api_adapter.llm_registry.llm_registry import RegistrySpec, resolve_model_spec
from llm_api_adapter.llms.transports import JSONResponse, SSEEvent
from llm_api_adapter.models.messages.file_parts import DocumentPart, ImagePart
from llm_api_adapter.models.tools import ToolSpec
from llm_api_adapter.service_provider_registry import ServiceProviderRegistry
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter
from llm_api_adapter_qwen.streaming import QwenMessagesStreamParser


QWEN_MODELS = (
    "qwen3.8-max",
    "qwen3.8-flash",
    "qwen3.7-plus",
    "qwen3.7-flash",
)


FLAT_OBJECT_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


class StructuredAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str


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


def qwen_messages_tool_sse_events(model="qwen3.8-max"):
    return [
        SSEEvent(
            event="message_start",
            data={
                "type": "message_start",
                "message": {
                    "id": "msg-qwen-tool-stream-1",
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
                "delta": {"type": "text_delta", "text": "Checking "},
            },
        ),
        SSEEvent(
            event="content_block_start",
            data={
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_qwen_1",
                    "name": "get_weather",
                    "input": {},
                },
            },
        ),
        SSEEvent(
            event="content_block_delta",
            data={
                "type": "content_block_delta",
                "index": 1,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": '{"city":"Tel',
                },
            },
        ),
        SSEEvent(
            event="content_block_delta",
            data={
                "type": "content_block_delta",
                "index": 1,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": ' Aviv"}',
                },
            },
        ),
        SSEEvent(
            event="content_block_stop",
            data={"type": "content_block_stop", "index": 1},
        ),
        SSEEvent(
            event="message_delta",
            data={
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {"input_tokens": 5, "output_tokens": 8},
            },
        ),
        SSEEvent(event="message_stop", data={"type": "message_stop"}),
    ]


def qwen_messages_thinking_sse_events(model="qwen3.8-max"):
    return [
        SSEEvent(
            event="message_start",
            data={
                "type": "message_start",
                "message": {
                    "id": "msg-qwen-thinking-stream-1",
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
                "content_block": {"type": "thinking", "thinking": ""},
            },
        ),
        SSEEvent(
            event="content_block_delta",
            data={
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "Plan"},
            },
        ),
        SSEEvent(
            event="content_block_start",
            data={
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        SSEEvent(
            event="content_block_delta",
            data={
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "text_delta", "text": "Visible"},
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
    QWEN_MODELS,
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
    QWEN_MODELS,
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


@pytest.mark.integration
@pytest.mark.parametrize(
    ("model", "input_tokens", "input_rate", "output_rate"),
    [
        ("qwen3.8-max", 10, 12, 36),
        ("qwen3.8-flash", 10, 0.8, 2.7),
        ("qwen3.7-plus", 256_000, 2, 8),
        ("qwen3.7-plus", 256_001, 6, 24),
        ("qwen3.7-flash", 32_000, 0.2, 0.8),
        ("qwen3.7-flash", 32_001, 0.6, 2.4),
        ("qwen3.7-flash", 256_001, 1.2, 4.8),
    ],
)
def test_qwen_finalizes_each_declared_frankfurt_pricing_tier(
    qwen_runtime,
    model,
    input_tokens,
    input_rate,
    output_rate,
):
    output_tokens = 10
    adapter = UniversalLLMAPIAdapter(
        organization="qwen",
        model=model,
        api_key="qwen-test-key",
    )
    adapter.adapter._sync_transport = FakeSyncTransport(
        {
            "id": f"msg-qwen-pricing-{model}",
            "model": model,
            "content": [{"type": "text", "text": "Hallo"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        }
    )

    response = adapter.chat(
        [{"role": "user", "content": "Hello"}],
        max_tokens=64,
        workspace_id="frankfurt-workspace",
    )

    assert response.currency == "CNY"
    assert response.cost_input == pytest.approx(input_tokens * input_rate / 1_000_000)
    assert response.cost_output == pytest.approx(output_tokens * output_rate / 1_000_000)
    assert response.cost_total == pytest.approx(
        response.cost_input + response.cost_output,
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    "model",
    QWEN_MODELS,
)
@pytest.mark.parametrize(
    "image, expected_source",
    [
        (
            ImagePart(url="https://example.com/landscape.jpg"),
            {"type": "url", "url": "https://example.com/landscape.jpg"},
        ),
        (
            ImagePart(data=b"qwen-image", media_type="image/png"),
            {"type": "base64", "media_type": "image/png", "data": "cXdlbi1pbWFnZQ=="},
        ),
    ],
)
def test_qwen_chat_serializes_image_parts_for_every_selected_model(
    qwen_runtime,
    model,
    image,
    expected_source,
):
    adapter = UniversalLLMAPIAdapter(
        organization="qwen",
        model=model,
        api_key="qwen-test-key",
    )
    transport = FakeSyncTransport(
        {
            "id": "msg-qwen-image-1",
            "model": model,
            "content": [{"type": "text", "text": "I see an image."}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        },
    )
    adapter.adapter._sync_transport = transport

    response = adapter.chat(
        [
            {
                "role": "user",
                "content": "Describe this image.",
                "files": [image],
            }
        ],
        max_tokens=64,
        workspace_id="frankfurt-workspace",
    )

    assert response.content == "I see an image."
    assert transport.requests[0].payload["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image."},
                {"type": "image", "source": expected_source},
            ],
        }
    ]


@pytest.mark.unit
@pytest.mark.parametrize("model", QWEN_MODELS)
@pytest.mark.parametrize(
    "document",
    [
        DocumentPart(url="https://example.com/report.pdf"),
        DocumentPart(data=b"%PDF-qwen", media_type="application/pdf"),
    ],
)
def test_qwen_rejects_every_document_part_before_transport(
    qwen_runtime,
    model,
    document,
):
    adapter = UniversalLLMAPIAdapter(
        organization="qwen",
        model=model,
        api_key="qwen-test-key",
    )
    transport = FakeSyncTransport({})
    adapter.adapter._sync_transport = transport

    with pytest.raises(ValueError, match="DocumentPart; PDF and OCR are unavailable"):
        adapter.chat(
            [
                {
                    "role": "user",
                    "content": "Summarize this PDF.",
                    "files": [document],
                }
            ],
            max_tokens=64,
            workspace_id="frankfurt-workspace",
        )
    with pytest.raises(ValueError, match="DocumentPart; PDF and OCR are unavailable"):
        adapter.stream_chat(
            [
                {
                    "role": "user",
                    "content": "Summarize this PDF.",
                    "files": [document],
                }
            ],
            max_tokens=64,
            workspace_id="frankfurt-workspace",
        )

    assert transport.requests == []
    assert transport.sse_requests == []


@pytest.mark.integration
@pytest.mark.parametrize("model", QWEN_MODELS)
def test_qwen_chat_supports_core_portable_json_schema(qwen_runtime, model):
    adapter = UniversalLLMAPIAdapter(
        organization="qwen",
        model=model,
        api_key="qwen-test-key",
    )
    transport = FakeSyncTransport(
        {
            "id": "msg-qwen-json-1",
            "model": model,
            "content": [{"type": "text", "text": '{"answer":"Hallo"}'}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        },
    )
    adapter.adapter._sync_transport = transport

    response = adapter.chat(
        [{"role": "user", "content": "Reply as JSON."}],
        max_tokens=64,
        json_schema=FLAT_OBJECT_SCHEMA,
        workspace_id="frankfurt-workspace",
    )

    assert response.parsed_json == {"answer": "Hallo"}
    assert transport.requests[0].payload["output_config"] == {
        "format": {"type": "json_schema", "schema": FLAT_OBJECT_SCHEMA},
    }


@pytest.mark.integration
@pytest.mark.parametrize("model", QWEN_MODELS)
def test_qwen_chat_supports_pydantic_structured_output(qwen_runtime, model):
    adapter = UniversalLLMAPIAdapter(
        organization="qwen",
        model=model,
        api_key="qwen-test-key",
    )
    transport = FakeSyncTransport(
        {
            "id": "msg-qwen-pydantic-1",
            "model": model,
            "content": [{"type": "text", "text": '{"answer":"Hallo"}'}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        },
    )
    adapter.adapter._sync_transport = transport

    response = adapter.chat(
        [{"role": "user", "content": "Reply as JSON."}],
        max_tokens=64,
        response_model=StructuredAnswer,
        workspace_id="frankfurt-workspace",
    )

    assert response.parsed_json == {"answer": "Hallo"}
    assert response.parsed_model == StructuredAnswer(answer="Hallo")
    assert transport.requests[0].payload["output_config"]["format"] == {
        "type": "json_schema",
        "schema": StructuredAnswer.model_json_schema(),
    }


@pytest.mark.unit
@pytest.mark.parametrize("model", QWEN_MODELS)
@pytest.mark.parametrize(
    "stop_reason, stop_details, attribute, expected",
    [
        ("refusal", {"reason": "policy"}, "refusal", "policy"),
        ("max_tokens", None, "incomplete_reason", "max_tokens"),
    ],
)
def test_qwen_structured_terminal_outcomes_are_not_parsed(
    qwen_runtime,
    model,
    stop_reason,
    stop_details,
    attribute,
    expected,
):
    adapter = UniversalLLMAPIAdapter(
        organization="qwen",
        model=model,
        api_key="qwen-test-key",
    )
    payload = {
        "id": "msg-qwen-terminal-1",
        "model": model,
        "content": [{"type": "text", "text": '{"answer":"partial"}'}],
        "stop_reason": stop_reason,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    if stop_details is not None:
        payload["stop_details"] = stop_details
    adapter.adapter._sync_transport = FakeSyncTransport(payload)

    response = adapter.chat(
        [{"role": "user", "content": "Reply as JSON."}],
        max_tokens=64,
        json_schema=FLAT_OBJECT_SCHEMA,
        workspace_id="frankfurt-workspace",
    )

    assert getattr(response, attribute) == expected
    assert response.parsed_json is None
    assert response.parsed_model is None


@pytest.mark.unit
def test_qwen_rejects_invalid_or_incompatible_structured_requests_before_transport(
    qwen_runtime,
):
    adapter = UniversalLLMAPIAdapter(
        organization="qwen",
        model="qwen3.8-max",
        api_key="qwen-test-key",
    )
    transport = FakeSyncTransport({})
    adapter.adapter._sync_transport = transport

    with pytest.raises(JSONSchemaError, match="Core portable profile"):
        adapter.chat(
            [{"role": "user", "content": "Reply as JSON."}],
            max_tokens=64,
            json_schema={"type": "object", "properties": {"answer": True}},
            workspace_id="frankfurt-workspace",
        )
    with pytest.raises(JSONSchemaError, match="json_schema and tools"):
        adapter.chat(
            [{"role": "user", "content": "Reply as JSON."}],
            max_tokens=64,
            json_schema=FLAT_OBJECT_SCHEMA,
            tools=[ToolSpec(name="get_weather", json_schema={"type": "object"})],
            workspace_id="frankfurt-workspace",
        )

    assert transport.requests == []


@pytest.mark.integration
@pytest.mark.parametrize(
    "model, reasoning_level, expected_options",
    [
        ("qwen3.8-max", "medium", {"output_config": {"effort": "medium"}}),
        ("qwen3.8-flash", "medium", {"output_config": {"effort": "medium"}}),
        (
            "qwen3.7-plus",
            "low",
            {"thinking": {"type": "enabled", "budget_tokens": 65_537}},
        ),
        (
            "qwen3.7-flash",
            "low",
            {"thinking": {"type": "enabled", "budget_tokens": 65_537}},
        ),
    ],
)
def test_qwen_resolves_reasoning_from_registry_metadata(
    qwen_runtime,
    model,
    reasoning_level,
    expected_options,
):
    adapter = UniversalLLMAPIAdapter(
        organization="qwen",
        model=model,
        api_key="qwen-test-key",
    )
    transport = FakeSyncTransport(
        {
            "id": "msg-qwen-reasoning-1",
            "model": model,
            "content": [{"type": "text", "text": "Hallo"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        },
    )
    adapter.adapter._sync_transport = transport

    adapter.chat(
        [{"role": "user", "content": "Think carefully."}],
        max_tokens=131_072,
        reasoning_level=reasoning_level,
        workspace_id="frankfurt-workspace",
    )

    request_payload = transport.requests[0].payload
    assert {key: request_payload[key] for key in expected_options} == expected_options


@pytest.mark.integration
@pytest.mark.parametrize(
    "model",
    QWEN_MODELS,
)
def test_qwen_reasoning_none_disables_thinking_for_every_model(qwen_runtime, model):
    adapter = UniversalLLMAPIAdapter(
        organization="qwen",
        model=model,
        api_key="qwen-test-key",
    )
    transport = FakeSyncTransport(
        {
            "id": "msg-qwen-no-thinking-1",
            "model": model,
            "content": [{"type": "text", "text": "Hallo"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        },
    )
    adapter.adapter._sync_transport = transport

    adapter.chat(
        [{"role": "user", "content": "Do not think aloud."}],
        max_tokens=64,
        reasoning_level="none",
        workspace_id="frankfurt-workspace",
    )

    assert transport.requests[0].payload["thinking"] == {"type": "disabled"}


@pytest.mark.integration
@pytest.mark.parametrize(
    ("model", "reasoning_level", "tool_choice", "expected_tool_choice"),
    [
        ("qwen3.8-max", None, "any", {"type": "any"}),
        (
            "qwen3.8-flash",
            "medium",
            "get_weather",
            {"type": "tool", "name": "get_weather"},
        ),
        ("qwen3.7-plus", "low", "any", {"type": "any"}),
        (
            "qwen3.7-flash",
            "low",
            "get_weather",
            {"type": "tool", "name": "get_weather"},
        ),
    ],
)
def test_qwen_forced_tool_choice_disables_thinking_and_warns(
    qwen_runtime,
    model,
    reasoning_level,
    tool_choice,
    expected_tool_choice,
):
    adapter = UniversalLLMAPIAdapter(
        organization="qwen",
        model=model,
        api_key="qwen-test-key",
    )
    transport = FakeSyncTransport(
        {
            "id": "msg-qwen-forced-tool-1",
            "model": model,
            "content": [{"type": "tool_use", "id": "toolu_qwen_1", "name": "get_weather", "input": {}}],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        },
    )
    adapter.adapter._sync_transport = transport

    with pytest.warns(UserWarning, match="disabled thinking because forced tool_choice"):
        adapter.chat(
            [{"role": "user", "content": "Use the weather tool."}],
            max_tokens=131_072,
            reasoning_level=reasoning_level,
            tools=[ToolSpec(name="get_weather", json_schema={"type": "object"})],
            tool_choice=tool_choice,
            workspace_id="frankfurt-workspace",
        )

    payload = transport.requests[0].payload
    assert payload["tool_choice"] == expected_tool_choice
    assert payload["thinking"] == {"type": "disabled"}
    assert "effort" not in payload.get("output_config", {})


@pytest.mark.unit
def test_qwen_explicit_thinking_disable_with_forced_tool_choice_does_not_warn(
    qwen_runtime,
):
    adapter = UniversalLLMAPIAdapter(
        organization="qwen",
        model="qwen3.8-max",
        api_key="qwen-test-key",
    )
    transport = FakeSyncTransport(
        {
            "id": "msg-qwen-explicit-no-thinking-1",
            "model": "qwen3.8-max",
            "content": [{"type": "tool_use", "id": "toolu_qwen_1", "name": "get_weather", "input": {}}],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        },
    )
    adapter.adapter._sync_transport = transport

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        adapter.chat(
            [{"role": "user", "content": "Use the weather tool."}],
            max_tokens=64,
            reasoning_level="none",
            tools=[ToolSpec(name="get_weather", json_schema={"type": "object"})],
            tool_choice="any",
            workspace_id="frankfurt-workspace",
        )

    assert captured == []
    assert transport.requests[0].payload["thinking"] == {"type": "disabled"}


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
@pytest.mark.parametrize("max_tokens", [None, 0, -1, True, "64"])
def test_qwen_rejects_invalid_output_limits_before_transport(
    qwen_runtime,
    max_tokens,
):
    adapter = UniversalLLMAPIAdapter(
        organization="qwen",
        model="qwen3.8-max",
        api_key="qwen-test-key",
    )
    transport = FakeSyncTransport({})
    adapter.adapter._sync_transport = transport

    with pytest.raises(ValueError, match="max_tokens"):
        adapter.chat(
            [{"role": "user", "content": "Hello"}],
            max_tokens=max_tokens,
            workspace_id="frankfurt-workspace",
        )

    assert transport.requests == []


@pytest.mark.unit
def test_qwen_rejects_parallel_tool_controls_before_transport(qwen_runtime):
    adapter = UniversalLLMAPIAdapter(
        organization="qwen",
        model="qwen3.8-max",
        api_key="qwen-test-key",
    )
    transport = FakeSyncTransport({})
    adapter.adapter._sync_transport = transport

    with pytest.raises(NotImplementedError, match="parallel tool-call"):
        adapter.chat(
            [{"role": "user", "content": "Hello"}],
            max_tokens=64,
            parallel_tool_calls=True,
            workspace_id="frankfurt-workspace",
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
        (400, "max_output_tokens_exceeded", LLMAPITokenLimitError),
        (400, "quota_exceeded", LLMAPIUsageLimitError),
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


@pytest.mark.unit
def test_qwen_rejects_non_object_messages_response_before_normalization(qwen_runtime):
    adapter = UniversalLLMAPIAdapter(
        organization="qwen",
        model="qwen3.8-max",
        api_key="qwen-test-key",
    )
    adapter.adapter._sync_transport = FakeSyncTransport(["not", "an", "object"])

    with pytest.raises(LLMAPIClientError, match="non-object response"):
        adapter.chat(
            [{"role": "user", "content": "Hello"}],
            max_tokens=64,
            workspace_id="frankfurt-workspace",
        )


@pytest.mark.asyncio
@pytest.mark.unit
async def test_qwen_async_client_rejects_non_object_messages_response(
    qwen_runtime,
    monkeypatch,
):
    from llm_api_adapter_qwen.clients import async_client as async_client_module

    async def fake_async_request(*args, **kwargs):
        return ["not", "an", "object"]

    monkeypatch.setattr(async_client_module, "async_request", fake_async_request)
    adapter = UniversalLLMAPIAdapter(
        organization="qwen",
        model="qwen3.8-max",
        api_key="qwen-test-key",
    )

    with pytest.raises(LLMAPIClientError, match="non-object response"):
        await adapter.achat(
            [{"role": "user", "content": "Hello"}],
            max_tokens=64,
            workspace_id="frankfurt-workspace",
        )


@pytest.mark.unit
def test_qwen_stream_parser_ignores_malformed_nonterminal_events():
    state = QwenMessagesStreamParser.new_state(
        buffer_chars=None,
        capture_reasoning=False,
    )

    for event in (
        SSEEvent(event="message_start", data={"type": "message_start"}),
        SSEEvent(
            event="content_block_start",
            data={"index": True, "content_block": {"type": "text"}},
        ),
        SSEEvent(
            event="content_block_delta",
            data={"index": 0, "delta": {"type": "text_delta", "text": "orphan"}},
        ),
        SSEEvent(event="content_block_stop", data={"index": 0}),
    ):
        assert QwenMessagesStreamParser.consume_event(event, state) == (None, None)

    assert QwenMessagesStreamParser.finalize(
        state,
        model="qwen3.8-max",
    ) == {"content": [], "model": "qwen3.8-max"}


@pytest.mark.unit
def test_qwen_stream_rejects_non_object_tool_arguments_before_done(qwen_runtime):
    adapter = UniversalLLMAPIAdapter(
        organization="qwen",
        model="qwen3.8-max",
        api_key="qwen-test-key",
    )
    events = qwen_messages_tool_sse_events()
    events[4].data["delta"]["partial_json"] = "[]"
    events[5].data["delta"]["partial_json"] = ""
    transport = FakeSyncTransport({}, events=events)
    adapter.adapter._sync_transport = transport

    with pytest.raises(InvalidToolArgumentsError, match="must decode to an object"):
        list(
            adapter.stream_chat(
                [{"role": "user", "content": "What is the weather?"}],
                max_tokens=64,
                workspace_id="frankfurt-workspace",
            )
        )

    assert transport.sse_closed is True


@pytest.mark.integration
@pytest.mark.parametrize("model", QWEN_MODELS)
def test_universal_stream_chat_reconstructs_qwen_messages_response(
    qwen_runtime,
    model,
):
    adapter = UniversalLLMAPIAdapter(
        organization="qwen",
        model=model,
        api_key="qwen-test-key",
    )
    transport = FakeSyncTransport({}, events=qwen_messages_sse_events(model))
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
    assert request.payload["model"] == model


@pytest.mark.integration
@pytest.mark.parametrize("model", QWEN_MODELS)
def test_qwen_stream_finalizes_core_structured_output(qwen_runtime, model):
    adapter = UniversalLLMAPIAdapter(
        organization="qwen",
        model=model,
        api_key="qwen-test-key",
    )
    events = qwen_messages_sse_events(model)
    events[2].data["delta"]["text"] = '{"answer":"Hal'
    events[3].data["delta"]["text"] = 'lo"}'
    transport = FakeSyncTransport({}, events=events)
    adapter.adapter._sync_transport = transport
    completed = []

    output = list(
        adapter.stream_chat(
            [{"role": "user", "content": "Reply as JSON."}],
            max_tokens=64,
            json_schema=FLAT_OBJECT_SCHEMA,
            workspace_id="frankfurt-workspace",
            on_done=completed.append,
        )
    )

    assert output == ['{"answer":"Hal', 'lo"}']
    assert completed[0].parsed_json == {"answer": "Hallo"}
    assert transport.sse_requests[0].payload["output_config"] == {
        "format": {"type": "json_schema", "schema": FLAT_OBJECT_SCHEMA},
    }


@pytest.mark.integration
@pytest.mark.parametrize("model", QWEN_MODELS)
def test_qwen_captures_nonstream_thinking_only_on_request(qwen_runtime, model):
    adapter = UniversalLLMAPIAdapter(
        organization="qwen",
        model=model,
        api_key="qwen-test-key",
    )
    transport = FakeSyncTransport(
        {
            "id": "msg-qwen-thinking-1",
            "model": model,
            "content": [
                {"type": "thinking", "thinking": "Plan"},
                {"type": "text", "text": "Visible"},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        },
    )
    adapter.adapter._sync_transport = transport

    without_capture = adapter.chat(
        [{"role": "user", "content": "Think carefully."}],
        max_tokens=64,
        workspace_id="frankfurt-workspace",
    )
    with_capture = adapter.chat(
        [{"role": "user", "content": "Think carefully."}],
        max_tokens=64,
        workspace_id="frankfurt-workspace",
        capture_reasoning=True,
    )

    assert without_capture.content == with_capture.content == "Visible"
    assert without_capture.reasoning_events == []
    assert [event.text for event in with_capture.reasoning_events] == ["Plan"]


@pytest.mark.integration
@pytest.mark.parametrize("model", QWEN_MODELS)
def test_qwen_stream_separates_captured_thinking_from_visible_text(
    qwen_runtime,
    model,
):
    adapter = UniversalLLMAPIAdapter(
        organization="qwen",
        model=model,
        api_key="qwen-test-key",
    )
    transport = FakeSyncTransport(
        {},
        events=qwen_messages_thinking_sse_events(model),
    )
    adapter.adapter._sync_transport = transport
    reasoning = []
    completed = []

    output = list(
        adapter.stream_chat(
            [{"role": "user", "content": "Think carefully."}],
            max_tokens=64,
            workspace_id="frankfurt-workspace",
            capture_reasoning=True,
            on_reasoning=reasoning.append,
            on_done=completed.append,
        )
    )

    assert output == ["Visible"]
    assert [event.text for event in reasoning] == ["Plan"]
    assert completed[0].content == "Visible"
    assert [event.text for event in completed[0].reasoning_events] == ["Plan"]


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.parametrize("model", QWEN_MODELS)
async def test_universal_achat_uses_qwen_async_client(
    qwen_runtime,
    monkeypatch,
    model,
):
    from llm_api_adapter_qwen.clients import async_client as async_client_module

    requests = []

    async def fake_async_request(url, **kwargs):
        requests.append((url, kwargs))
        return {
            "id": "msg-qwen-async-1",
            "model": model,
            "content": [{"type": "text", "text": "Hallo async"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }

    monkeypatch.setattr(async_client_module, "async_request", fake_async_request)
    adapter = UniversalLLMAPIAdapter(
        organization="qwen",
        model=model,
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
        "model": model,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 64,
        "temperature": 1.0,
        "top_p": 1.0,
    }
    assert requests[0][1]["timeout"] == 12.5


@pytest.mark.asyncio
@pytest.mark.integration
async def test_qwen_achat_disables_thinking_for_forced_tool_choice(
    qwen_runtime,
    monkeypatch,
):
    from llm_api_adapter_qwen.clients import async_client as async_client_module

    requests = []

    async def fake_async_request(url, **kwargs):
        requests.append((url, kwargs))
        return {
            "id": "msg-qwen-async-tool-1",
            "model": "qwen3.8-max",
            "content": [{"type": "tool_use", "id": "toolu_qwen_1", "name": "get_weather", "input": {}}],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

    monkeypatch.setattr(async_client_module, "async_request", fake_async_request)
    adapter = UniversalLLMAPIAdapter(
        organization="qwen",
        model="qwen3.8-max",
        api_key="qwen-test-key",
    )

    with pytest.warns(UserWarning, match="disabled thinking because forced tool_choice"):
        await adapter.achat(
            [{"role": "user", "content": "Use the weather tool."}],
            max_tokens=64,
            tools=[ToolSpec(name="get_weather", json_schema={"type": "object"})],
            tool_choice="get_weather",
            workspace_id="frankfurt-workspace",
        )

    assert requests[0][1]["payload"]["thinking"] == {"type": "disabled"}


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.parametrize("model", QWEN_MODELS)
async def test_universal_astream_chat_matches_sync_lifecycle(
    qwen_runtime,
    monkeypatch,
    model,
):
    from llm_api_adapter_qwen.clients import async_client as async_client_module

    requests = []
    closed = False

    def fake_async_stream_request(url, **kwargs):
        requests.append((url, kwargs))

        async def events():
            nonlocal closed
            try:
                for event in qwen_messages_sse_events(model):
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
        model=model,
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
@pytest.mark.integration
async def test_qwen_astream_separates_captured_thinking_from_visible_text(
    qwen_runtime,
    monkeypatch,
):
    from llm_api_adapter_qwen.clients import async_client as async_client_module

    def fake_async_stream_request(url, **kwargs):
        async def events():
            for event in qwen_messages_thinking_sse_events():
                yield event

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
    reasoning = []
    completed = []

    output = [
        text
        async for text in adapter.astream_chat(
            [{"role": "user", "content": "Think carefully."}],
            max_tokens=64,
            workspace_id="frankfurt-workspace",
            capture_reasoning=True,
            on_reasoning=reasoning.append,
            on_done=completed.append,
        )
    ]

    assert output == ["Visible"]
    assert [event.text for event in reasoning] == ["Plan"]
    assert completed[0].content == "Visible"
    assert [event.text for event in completed[0].reasoning_events] == ["Plan"]


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


@pytest.mark.integration
@pytest.mark.parametrize("model", QWEN_MODELS)
@pytest.mark.parametrize(
    ("tool_choice", "expected_tool_choice"),
    [
        ("auto", {"type": "auto"}),
        ("none", {"type": "none"}),
    ],
)
def test_qwen_chat_maps_tools_and_normalized_tool_choice(
    qwen_runtime,
    model,
    tool_choice,
    expected_tool_choice,
):
    adapter = UniversalLLMAPIAdapter(
        organization="qwen",
        model=model,
        api_key="qwen-test-key",
    )
    transport = FakeSyncTransport(
        {
            "id": "msg-qwen-tool-1",
            "model": model,
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_qwen_1",
                    "name": "get_weather",
                    "input": {"city": "Tel Aviv"},
                }
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 20},
        },
    )
    adapter.adapter._sync_transport = transport
    tool = ToolSpec(
        name="get_weather",
        description="Look up the weather for a city.",
        json_schema={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    )

    response = adapter.chat(
        [{"role": "user", "content": "What is the weather in Tel Aviv?"}],
        max_tokens=64,
        tools=[tool],
        tool_choice=tool_choice,
        workspace_id="frankfurt-workspace",
    )

    assert response.content is None
    assert response.finish_reason == "tool_use"
    assert response.tool_calls is not None
    assert response.tool_calls[0].name == "get_weather"
    assert response.tool_calls[0].call_id == "toolu_qwen_1"
    assert response.tool_calls[0].arguments == {"city": "Tel Aviv"}
    assert transport.requests[0].payload["tools"] == [
        {
            "name": "get_weather",
            "description": "Look up the weather for a city.",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    ]
    assert transport.requests[0].payload["tool_choice"] == expected_tool_choice
    assert "thinking" not in transport.requests[0].payload


@pytest.mark.integration
@pytest.mark.parametrize("model", QWEN_MODELS)
def test_qwen_chat_maps_application_controlled_tool_result_messages(
    qwen_runtime,
    model,
):
    adapter = UniversalLLMAPIAdapter(
        organization="qwen",
        model=model,
        api_key="qwen-test-key",
    )
    transport = FakeSyncTransport(
        {
            "id": "msg-qwen-tool-result-1",
            "model": model,
            "content": [{"type": "text", "text": "It is sunny."}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 12, "output_tokens": 4},
        },
    )
    adapter.adapter._sync_transport = transport
    tool = ToolSpec(
        name="get_weather",
        json_schema={"type": "object", "properties": {}},
    )

    response = adapter.chat(
        [
            {"role": "user", "content": "What is the weather in Tel Aviv?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "name": "get_weather",
                        "arguments": {"city": "Tel Aviv"},
                        "call_id": "toolu_qwen_1",
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "toolu_qwen_1",
                "content": "{\"condition\": \"sunny\"}",
            },
        ],
        max_tokens=64,
        tools=[tool],
        workspace_id="frankfurt-workspace",
    )

    assert response.content == "It is sunny."
    assert transport.requests[0].payload["messages"] == [
        {"role": "user", "content": "What is the weather in Tel Aviv?"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_qwen_1",
                    "name": "get_weather",
                    "input": {"city": "Tel Aviv"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_qwen_1",
                    "content": "{\"condition\": \"sunny\"}",
                }
            ],
        },
    ]


@pytest.mark.integration
@pytest.mark.parametrize("model", QWEN_MODELS)
def test_qwen_stream_delivers_completed_tool_call_before_done(qwen_runtime, model):
    adapter = UniversalLLMAPIAdapter(
        organization="qwen",
        model=model,
        api_key="qwen-test-key",
    )
    transport = FakeSyncTransport({}, events=qwen_messages_tool_sse_events(model))
    adapter.adapter._sync_transport = transport
    events = []

    with pytest.warns(UserWarning, match="disabled thinking because forced tool_choice"):
        output = list(
            adapter.stream_chat(
                [{"role": "user", "content": "What is the weather in Tel Aviv?"}],
                max_tokens=64,
                tools=[
                    ToolSpec(
                        name="get_weather",
                        json_schema={"type": "object", "properties": {}},
                    )
                ],
                tool_choice="any",
                workspace_id="frankfurt-workspace",
                on_tool_call=lambda call: events.append(
                    ("tool", call.name, call.arguments, call.call_id),
                ),
                on_done=lambda response: events.append(
                    ("done", response.finish_reason, response.tool_calls),
                ),
            )
        )

    assert output == ["Checking "]
    assert events[0] == (
        "tool",
        "get_weather",
        {"city": "Tel Aviv"},
        "toolu_qwen_1",
    )
    assert events[1][0:2] == ("done", "tool_use")
    assert events[1][2] is not None
    assert events[1][2][0].arguments == {"city": "Tel Aviv"}
    assert transport.sse_requests[0].payload["tool_choice"] == {"type": "any"}
    assert transport.sse_requests[0].payload["thinking"] == {"type": "disabled"}


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.parametrize("model", QWEN_MODELS)
async def test_qwen_astream_delivers_completed_tool_call_before_done(
    qwen_runtime,
    monkeypatch,
    model,
):
    from llm_api_adapter_qwen.clients import async_client as async_client_module

    requests = []

    def fake_async_stream_request(url, **kwargs):
        requests.append((url, kwargs))

        async def events():
            for event in qwen_messages_tool_sse_events(model):
                yield event

        return events()

    monkeypatch.setattr(
        async_client_module,
        "async_stream_request",
        fake_async_stream_request,
    )
    adapter = UniversalLLMAPIAdapter(
        organization="qwen",
        model=model,
        api_key="qwen-test-key",
    )
    callbacks = []

    async def on_tool_call(call):
        callbacks.append(("tool", call.name, call.arguments, call.call_id))

    async def on_done(response):
        callbacks.append(("done", response.finish_reason, response.tool_calls))

    with pytest.warns(UserWarning, match="disabled thinking because forced tool_choice"):
        output = [
            text
            async for text in adapter.astream_chat(
                [{"role": "user", "content": "What is the weather in Tel Aviv?"}],
                max_tokens=64,
                tools=[
                    ToolSpec(
                        name="get_weather",
                        json_schema={"type": "object", "properties": {}},
                    )
                ],
                tool_choice={"type": "tool", "name": "get_weather"},
                workspace_id="frankfurt-workspace",
                on_tool_call=on_tool_call,
                on_done=on_done,
            )
        ]

    assert output == ["Checking "]
    assert callbacks[0] == (
        "tool",
        "get_weather",
        {"city": "Tel Aviv"},
        "toolu_qwen_1",
    )
    assert callbacks[1][0:2] == ("done", "tool_use")
    assert callbacks[1][2] is not None
    assert callbacks[1][2][0].arguments == {"city": "Tel Aviv"}
    assert requests[0][1]["payload"]["thinking"] == {"type": "disabled"}
