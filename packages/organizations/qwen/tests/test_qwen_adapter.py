"""Deterministic contract tests for Qwen's Frankfurt Messages adapter."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import fields
import inspect
import logging
from pathlib import Path
import sys
from types import MappingProxyType
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
        ("qwen3.8-max", 10, 12.0, 36.0),
        ("qwen3.8-flash", 10, 0.8, 2.7),
        ("qwen3.7-plus", 256_000, 2.0, 8.0),
        ("qwen3.7-plus", 256_001, 6.0, 24.0),
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
                "cache_read_input_tokens": 0,
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
@pytest.mark.parametrize("model", ["qwen3.7-plus", "qwen3.7-flash"])
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


@pytest.mark.unit
@pytest.mark.parametrize("model", ["qwen3.7-plus", "qwen3.7-flash"])
def test_qwen_numeric_reasoning_budget_is_not_limited_by_max_tokens(
    qwen_runtime,
    model,
):
    adapter = UniversalLLMAPIAdapter(
        organization="qwen",
        model=model,
        api_key="qwen-test-key",
    )
    budget = adapter.adapter.model_spec.reasoning_capability.max_budget_tokens
    transport = FakeSyncTransport(
        {
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )
    adapter.adapter._sync_transport = transport

    adapter.chat(
        [{"role": "user", "content": "Think."}],
        max_tokens=1,
        reasoning_level=budget,
        workspace_id="frankfurt-workspace",
    )

    assert transport.requests[0].payload["max_tokens"] == 1
    assert transport.requests[0].payload["thinking"] == {
        "type": "enabled",
        "budget_tokens": budget,
    }


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
def test_qwen_stream_parser_safely_ignores_unhashable_event_kinds_and_uses_empty_name_fallback():
    state = QwenMessagesStreamParser.new_state(
        buffer_chars=None,
        capture_reasoning=False,
    )
    malformed_events = (
        SSEEvent(event=None, data={"type": []}),
        SSEEvent(event=None, data={"type": {}}),
    )
    for event in malformed_events:
        assert QwenMessagesStreamParser.consume_event(event, state) == (None, None)

    assert QwenMessagesStreamParser.consume_event(
        SSEEvent(
            event="",
            data={
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": "kept"},
            },
        ),
        state,
    ) == (None, None)
    assert QwenMessagesStreamParser.finalize(
        state,
        model="qwen3.8-max",
    )["content"] == [{"type": "text", "text": "kept"}]


@pytest.mark.unit
def test_qwen_stream_preserves_malformed_start_usage_for_native_validation():
    from llm_api_adapter_qwen.adapter import QwenAdapter

    state = QwenMessagesStreamParser.new_state(
        buffer_chars=None,
        capture_reasoning=False,
    )
    QwenMessagesStreamParser.consume_event(
        SSEEvent(
            event="message_start",
            data={
                "type": "message_start",
                "message": {
                    "id": "synthetic",
                    "content": [],
                    "usage": [],
                },
            },
        ),
        state,
    )

    response = QwenMessagesStreamParser.finalize(state, model="qwen3.8-max")
    assert response["usage"] == []
    with pytest.raises(LLMAPIClientError, match="response.usage must be an object"):
        QwenAdapter._parse_response(response, capture_reasoning=False)


@pytest.mark.unit
def test_qwen_stream_keeps_prior_valid_usage_when_later_start_usage_is_malformed():
    state = QwenMessagesStreamParser.new_state(
        buffer_chars=None,
        capture_reasoning=False,
    )
    for message in (
        {
            "id": "synthetic-valid",
            "content": [],
            "usage": {"input_tokens": 7, "output_tokens": 0},
        },
        {
            "id": "synthetic-invalid",
            "content": [],
            "usage": [],
        },
    ):
        QwenMessagesStreamParser.consume_event(
            SSEEvent(
                event="message_start",
                data={"type": "message_start", "message": message},
            ),
            state,
        )

    response = QwenMessagesStreamParser.finalize(state, model="qwen3.8-max")
    assert response["usage"] == {"input_tokens": 7, "output_tokens": 0}


@pytest.mark.unit
def test_qwen_reuses_shared_messages_assembly_and_stream_reconstruction(
    qwen_runtime,
    monkeypatch,
):
    from llm_api_adapter.adapters.anthropic_compatible import payloads, streaming
    import llm_api_adapter_qwen.adapter as qwen_adapter_module
    import llm_api_adapter_qwen.streaming as qwen_streaming_module

    assembly_calls = []
    consume_calls = []
    finalize_calls = []
    original_build = qwen_adapter_module.build_messages_payload
    original_consume = qwen_streaming_module.consume_messages_event
    original_finalize = qwen_streaming_module.finalize_messages_stream

    def build_spy(**kwargs):
        assembly_calls.append(kwargs)
        return original_build(**kwargs)

    def consume_spy(*args, **kwargs):
        consume_calls.append(args[1])
        return original_consume(*args, **kwargs)

    def finalize_spy(*args, **kwargs):
        finalize_calls.append(args[0])
        return original_finalize(*args, **kwargs)

    monkeypatch.setattr(qwen_adapter_module, "build_messages_payload", build_spy)
    monkeypatch.setattr(qwen_streaming_module, "consume_messages_event", consume_spy)
    monkeypatch.setattr(qwen_streaming_module, "finalize_messages_stream", finalize_spy)
    adapter = UniversalLLMAPIAdapter(
        organization="qwen",
        model="qwen3.8-max",
        api_key="qwen-test-key",
    )
    transport = FakeSyncTransport(
        {}, events=qwen_messages_sse_events("qwen3.8-max")
    )
    adapter.adapter._sync_transport = transport

    list(
        adapter.stream_chat(
            [{"role": "user", "content": "Hello"}],
            max_tokens=64,
            workspace_id="frankfurt-workspace",
        )
    )

    assert len(assembly_calls) == 1
    assert assembly_calls[0]["messages"].items
    assert consume_calls
    assert finalize_calls
    assert payloads.build_messages_payload is original_build
    assert streaming.consume_messages_event is original_consume


@pytest.mark.unit
def test_qwen_shared_stream_policy_keeps_native_tolerated_edges():
    state = QwenMessagesStreamParser.new_state(
        buffer_chars=None,
        capture_reasoning=False,
    )
    events = [
        SSEEvent(event="content_block_start", data={"index": -1, "content_block": {"type": "text", "text": ""}}),
        SSEEvent(event="content_block_delta", data={"index": -1, "delta": {"type": "text_delta", "text": "before"}}),
        SSEEvent(event="content_block_stop", data={"index": -1}),
        SSEEvent(event="content_block_delta", data={"index": -1, "delta": {"type": "text_delta", "text": "after"}}),
        SSEEvent(event="content_block_start", data={"index": 1, "content_block": {"type": "tool_use", "name": "lookup", "input": {"keep": True}}}),
        SSEEvent(event="content_block_delta", data={"index": 1, "delta": {"type": "signature_delta", "signature": "ignored"}}),
        SSEEvent(event="content_block_delta", data={"index": 1, "delta": {"type": "input_json_delta", "partial_json": ""}}),
        SSEEvent(event="content_block_stop", data={"index": 1}),
        SSEEvent(event="message_delta", data={"delta": None, "usage": {"output_tokens": 3}}),
    ]

    outputs = [QwenMessagesStreamParser.consume_event(event, state) for event in events]

    assert outputs[1] == ("before", None)
    assert outputs[2] == (None, None)
    assert outputs[3] == ("after", None)
    response = QwenMessagesStreamParser.finalize(state, model="qwen3.8-max")
    assert response["content"] == [
        {"type": "text", "text": "beforeafter"},
        {"type": "tool_use", "name": "lookup", "input": {"keep": True}},
    ]
    assert response["usage"] == {"output_tokens": 3}


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
@pytest.mark.unit
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
    async_stream = adapter.astream_chat(
        [{"role": "user", "content": "Hello"}],
        max_tokens=64,
        workspace_id=workspace_id,
    )
    with pytest.raises(ValueError, match="workspace_id"):
        await anext(async_stream)

    assert requests == []


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("model", QWEN_MODELS)
async def test_qwen_streams_prepare_at_compatible_times(
    qwen_runtime,
    model,
):
    adapter = UniversalLLMAPIAdapter(
        organization="qwen",
        model=model,
        api_key="qwen-test-key",
    )
    transport = FakeSyncTransport({})
    adapter.adapter._sync_transport = transport

    with pytest.raises(ValueError, match="max_tokens"):
        adapter.stream_chat(
            [{"role": "user", "content": "Hello"}],
            max_tokens=0,
            workspace_id="frankfurt-workspace",
        )
    async_stream = adapter.astream_chat(
        [{"role": "user", "content": "Hello"}],
        max_tokens=0,
        workspace_id="frankfurt-workspace",
    )
    with pytest.raises(ValueError, match="max_tokens"):
        await anext(async_stream)
    assert transport.sse_requests == []


@pytest.mark.unit
@pytest.mark.parametrize("model", QWEN_MODELS)
def test_qwen_adapter_preserves_dataclass_construction_and_transport_patch_surface(
    qwen_runtime,
    monkeypatch,
    model,
):
    from llm_api_adapter_qwen import adapter as qwen_adapter_module
    from llm_api_adapter_qwen.adapter import QwenAdapter
    from llm_api_adapter_qwen.clients.sync_client import QwenMessagesSyncClient

    created_transport = FakeSyncTransport({})
    factory_calls = []
    sync_client_instances = []
    async_client_instances = []

    def fake_create_sync_transport(name):
        factory_calls.append(name)
        return created_transport

    class FakeSyncClient(QwenMessagesSyncClient):
        def __init__(self, transport):
            super().__init__(transport)
            sync_client_instances.append(self)

    class FakeAsyncClient:
        def __init__(self):
            async_client_instances.append(self)

    monkeypatch.setattr(
        qwen_adapter_module,
        "create_sync_transport",
        fake_create_sync_transport,
    )
    monkeypatch.setattr(
        qwen_adapter_module,
        "QwenMessagesSyncClient",
        FakeSyncClient,
    )
    monkeypatch.setattr(
        qwen_adapter_module,
        "QwenMessagesAsyncClient",
        FakeAsyncClient,
    )

    adapter = QwenAdapter(
        "qwen-test-key",
        model,
        "qwen",
        "requests",
        None,
        False,
        False,
        "qwen",
    )
    equal_adapter = QwenAdapter("qwen-test-key", model, "qwen")
    assert factory_calls == ["requests", "requests"]
    assert len(sync_client_instances) == 2
    assert sync_client_instances[0] is not sync_client_instances[1]
    assert len(async_client_instances) == 2
    assert async_client_instances[0] is not async_client_instances[1]
    assert adapter._client.transport is created_transport
    assert adapter._async_client is not equal_adapter._async_client
    assert adapter._sync_transport is created_transport
    assert adapter == equal_adapter

    model_fields = {item.name: item for item in fields(QwenAdapter)}
    assert tuple(model_fields) == (
        "api_key",
        "model",
        "company",
        "transport",
        "pricing",
        "is_reasoning",
        "is_adaptive_thinking",
        "service_provider",
        "model_spec",
        "_client",
        "_async_client",
    )
    assert {
        name: model_fields[name].default
        for name in (
            "company",
            "transport",
            "pricing",
            "is_reasoning",
            "is_adaptive_thinking",
            "service_provider",
        )
    } == {
        "company": "qwen",
        "transport": "requests",
        "pricing": None,
        "is_reasoning": False,
        "is_adaptive_thinking": False,
        "service_provider": None,
    }
    assert not model_fields["model_spec"].init
    assert not model_fields["model_spec"].repr
    for name in ("_client", "_async_client"):
        assert not model_fields[name].init
        assert not model_fields[name].repr
        assert not model_fields[name].compare

    chat_parameters = inspect.signature(QwenAdapter.chat).parameters
    assert list(chat_parameters)[:12] == [
        "self",
        "messages",
        "max_tokens",
        "temperature",
        "top_p",
        "reasoning_level",
        "timeout_s",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "previous_response",
        "json_schema",
    ]
    assert list(chat_parameters)[12:14] == ["response_model", "workspace_id"]
    assert chat_parameters["max_tokens"].default is None
    for method_name in ("chat", "achat", "stream_chat", "astream_chat"):
        method_parameters = inspect.signature(
            getattr(QwenAdapter, method_name)
        ).parameters
        assert method_parameters["max_tokens"].default is None
        assert method_parameters["workspace_id"].kind is inspect.Parameter.KEYWORD_ONLY
        assert method_parameters["capture_reasoning"].kind is inspect.Parameter.KEYWORD_ONLY

    stream_parameters = inspect.signature(QwenAdapter.stream_chat).parameters
    assert stream_parameters["on_done"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert stream_parameters["buffer_chars"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert stream_parameters["on_chunk"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert stream_parameters["workspace_id"].kind is inspect.Parameter.KEYWORD_ONLY
    assert stream_parameters["capture_reasoning"].kind is inspect.Parameter.KEYWORD_ONLY
    assert stream_parameters["on_reasoning"].kind is inspect.Parameter.KEYWORD_ONLY
    with pytest.raises(ValueError, match="max_tokens"):
        adapter.chat(
            [{"role": "user", "content": "Hello"}],
            workspace_id="frankfurt-workspace",
        )

    replacement_transport = FakeSyncTransport({})
    adapter._sync_transport = replacement_transport
    assert adapter._client.transport is replacement_transport
    assert "qwen-test-key" not in repr(adapter)

    positional_done = []
    adapter._sync_transport = FakeSyncTransport({}, events=qwen_messages_sse_events(model))
    list(
        adapter.stream_chat(
            [{"role": "user", "content": "Hello"}],
            64,
            1.0,
            1.0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            positional_done.append,
            None,
            None,
            workspace_id="frankfurt-workspace",
        )
    )
    assert positional_done[0].content == "Hallo"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_qwen_inherits_and_invokes_all_shared_prepared_runners(
    qwen_runtime,
    monkeypatch,
):
    from llm_api_adapter.adapters.anthropic_compatible.adapter import (
        AnthropicCompatibleAdapter,
    )
    from llm_api_adapter.adapters.anthropic_compatible.execution import (
        _MessagesPreparedRunners,
    )
    from llm_api_adapter_qwen.adapter import QwenAdapter
    from llm_api_adapter_qwen.clients import async_client as async_client_module

    assert QwenAdapter.__bases__ == (AnthropicCompatibleAdapter,)
    assert QwenAdapter._consume_prepared_event is _MessagesPreparedRunners._consume_prepared_event
    assert QwenAdapter._consume_prepared_event_async is _MessagesPreparedRunners._consume_prepared_event_async
    assert QwenAdapter._finalize_prepared_stream is _MessagesPreparedRunners._finalize_prepared_stream
    adapter = QwenAdapter("qwen-test-key", "qwen3.8-max")
    prepared_state = adapter._new_messages_stream_state(None, False)
    prepared_state.common.message_data["content"].append(
        {"type": "text", "text": "synthetic provider content"}
    )
    assert "synthetic provider content" not in repr(prepared_state)
    adapter._sync_transport = FakeSyncTransport(
        {
            "content": [{"type": "text", "text": "sync"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
        events=qwen_messages_sse_events(),
    )
    async_requests = []

    async def fake_async_request(url, **kwargs):
        async_requests.append((url, kwargs))
        return {
            "content": [{"type": "text", "text": "async"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    def fake_async_stream_request(url, **kwargs):
        async_requests.append((url, kwargs))

        async def events():
            for event in qwen_messages_sse_events():
                yield event

        return events()

    monkeypatch.setattr(async_client_module, "async_request", fake_async_request)
    monkeypatch.setattr(
        async_client_module,
        "async_stream_request",
        fake_async_stream_request,
    )
    calls = []

    original_chat = _MessagesPreparedRunners._run_prepared_chat
    original_achat = _MessagesPreparedRunners._run_prepared_achat
    original_stream = _MessagesPreparedRunners._run_prepared_stream
    original_astream = _MessagesPreparedRunners._run_prepared_astream

    def run_chat(self, *args, **kwargs):
        calls.append("chat")
        return original_chat(self, *args, **kwargs)

    async def run_achat(self, *args, **kwargs):
        calls.append("achat")
        return await original_achat(self, *args, **kwargs)

    def run_stream(self, *args, **kwargs):
        calls.append("stream")
        return original_stream(self, *args, **kwargs)

    def run_astream(self, *args, **kwargs):
        calls.append("astream")
        return original_astream(self, *args, **kwargs)

    monkeypatch.setattr(_MessagesPreparedRunners, "_run_prepared_chat", run_chat)
    monkeypatch.setattr(_MessagesPreparedRunners, "_run_prepared_achat", run_achat)
    monkeypatch.setattr(_MessagesPreparedRunners, "_run_prepared_stream", run_stream)
    monkeypatch.setattr(_MessagesPreparedRunners, "_run_prepared_astream", run_astream)

    messages = [{"role": "user", "content": "Hello"}]
    adapter.chat(messages, max_tokens=64, workspace_id="frankfurt-workspace")
    await adapter.achat(messages, max_tokens=64, workspace_id="frankfurt-workspace")
    list(adapter.stream_chat(messages, max_tokens=64, workspace_id="frankfurt-workspace"))
    [text async for text in adapter.astream_chat(
        messages, max_tokens=64, workspace_id="frankfurt-workspace"
    )]

    assert calls == ["chat", "achat", "stream", "astream"]
    assert len(async_requests) == 2


@pytest.mark.asyncio
@pytest.mark.unit
async def test_qwen_streams_create_sources_on_first_read(qwen_runtime):
    from llm_api_adapter_qwen.adapter import QwenAdapter

    adapter = QwenAdapter("qwen-test-key", "qwen3.8-max")
    sync_calls = []

    def sync_stream(**kwargs):
        sync_calls.append(kwargs)
        return iter(qwen_messages_sse_events())

    adapter._client.stream = sync_stream
    sync_result = adapter.stream_chat(
        [{"role": "user", "content": "Hello"}],
        max_tokens=64,
        workspace_id="frankfurt-workspace",
    )
    assert sync_calls == []
    assert list(sync_result) == ["Hal", "lo"]
    assert len(sync_calls) == 1

    async_calls = []

    def async_stream(**kwargs):
        async_calls.append(kwargs)

        async def events():
            for event in qwen_messages_sse_events():
                yield event

        return events()

    adapter._async_client.stream = async_stream
    async_result = adapter.astream_chat(
        [{"role": "user", "content": "Hello"}],
        max_tokens=64,
        workspace_id="frankfurt-workspace",
    )
    assert async_calls == []
    assert await anext(async_result) == "Hal"
    assert len(async_calls) == 1
    assert [text async for text in async_result] == ["lo"]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_qwen_stream_validates_buffer_before_opening_native_sources(qwen_runtime):
    from llm_api_adapter_qwen.adapter import QwenAdapter

    adapter = QwenAdapter("qwen-test-key", "qwen3.8-max")
    sync_calls = []
    async_calls = []
    adapter._client.stream = lambda **kwargs: sync_calls.append(kwargs)
    adapter._async_client.stream = lambda **kwargs: async_calls.append(kwargs)
    messages = [{"role": "user", "content": "Hello"}]

    with pytest.raises(ValueError, match="buffer_chars"):
        adapter.stream_chat(
            messages,
            max_tokens=64,
            buffer_chars=0,
            workspace_id="frankfurt-workspace",
        )
    async_result = adapter.astream_chat(
        messages,
        max_tokens=64,
        buffer_chars=0,
        workspace_id="frankfurt-workspace",
    )
    with pytest.raises(ValueError, match="buffer_chars"):
        await anext(async_result)

    assert sync_calls == []
    assert async_calls == []


@pytest.mark.unit
def test_qwen_shared_runner_preserves_native_error_identity(qwen_runtime):
    from llm_api_adapter_qwen.adapter import QwenAdapter

    adapter = QwenAdapter("qwen-test-key", "qwen3.8-max")
    original = LLMAPIClientError(detail="native Qwen failure")

    def fail_chat(**kwargs):
        raise original

    adapter._client.chat = fail_chat
    with pytest.raises(LLMAPIClientError) as caught:
        adapter.chat(
            [{"role": "user", "content": "Hello"}],
            max_tokens=64,
            workspace_id="frankfurt-workspace",
        )

    assert caught.value is original


@pytest.mark.integration
@pytest.mark.parametrize("model", QWEN_MODELS)
def test_qwen_chat_and_stream_final_responses_use_first_text_block(
    qwen_runtime,
    model,
):
    adapter = UniversalLLMAPIAdapter(
        organization="qwen",
        model=model,
        api_key="qwen-test-key",
    )
    response = {
        "id": "msg-qwen-first-text",
        "model": model,
        "content": [
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 2},
    }
    transport = FakeSyncTransport(response)
    adapter.adapter._sync_transport = transport
    assert adapter.chat(
        [{"role": "user", "content": "Hello"}],
        max_tokens=8,
        workspace_id="frankfurt-workspace",
    ).content == "first"

    events = qwen_messages_sse_events(model)
    events[2].data["delta"]["text"] = "first"
    events[3].data["delta"]["text"] = ""
    events.insert(
        -2,
        SSEEvent(
            event="content_block_start",
            data={
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "text", "text": ""},
            },
        ),
    )
    events.insert(
        -2,
        SSEEvent(
            event="content_block_delta",
            data={
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "text_delta", "text": "second"},
            },
        ),
    )
    events = events[:-2]
    transport.events = events
    completed = []
    visible = list(
        adapter.stream_chat(
            [{"role": "user", "content": "Hello"}],
            max_tokens=8,
            workspace_id="frankfurt-workspace",
            on_done=completed.append,
        )
    )
    assert visible == ["first", "second"]
    assert len(completed) == 1
    assert completed[0].content == "first"
    assert completed[0].finish_reason is None


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.parametrize("model", QWEN_MODELS)
async def test_qwen_async_chat_and_stream_final_responses_use_first_text_block(
    qwen_runtime,
    monkeypatch,
    model,
):
    from llm_api_adapter_qwen.clients import async_client as async_client_module

    response = {
        "id": "msg-qwen-first-text-async",
        "model": model,
        "content": [
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 2},
    }

    async def fake_async_request(*args, **kwargs):
        return response

    def fake_async_stream_request(*args, **kwargs):
        async def events():
            yield SSEEvent(
                event="message_start",
                data={
                    "type": "message_start",
                    "message": {
                        "id": "msg-qwen-first-text-async",
                        "model": model,
                        "content": [],
                        "usage": {"input_tokens": 1, "output_tokens": 0},
                    },
                },
            )
            for index, text in ((0, "first"), (1, "second")):
                yield SSEEvent(
                    event="content_block_start",
                    data={
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {"type": "text", "text": ""},
                    },
                )
                yield SSEEvent(
                    event="content_block_delta",
                    data={
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {"type": "text_delta", "text": text},
                    },
                )

        return events()

    monkeypatch.setattr(async_client_module, "async_request", fake_async_request)
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
    assert (await adapter.achat(
        [{"role": "user", "content": "Hello"}],
        max_tokens=8,
        workspace_id="frankfurt-workspace",
    )).content == "first"
    completed = []
    visible = [
        text
        async for text in adapter.astream_chat(
            [{"role": "user", "content": "Hello"}],
            max_tokens=8,
            workspace_id="frankfurt-workspace",
            on_done=completed.append,
        )
    ]
    assert visible == ["first", "second"]
    assert len(completed) == 1
    assert completed[0].content == "first"
    assert completed[0].finish_reason is None


@pytest.mark.asyncio
@pytest.mark.unit
async def test_qwen_overlapping_async_workspace_chats_and_streams_are_isolated(
    qwen_runtime,
    monkeypatch,
):
    from llm_api_adapter_qwen.clients import async_client as async_client_module

    adapter = UniversalLLMAPIAdapter(
        organization="qwen",
        model="qwen3.8-max",
        api_key="qwen-test-key",
    )
    chat_requests = []
    both_chats_started = asyncio.Event()

    async def fake_async_request(url, **kwargs):
        chat_requests.append((url, kwargs))
        if len(chat_requests) == 2:
            both_chats_started.set()
        await both_chats_started.wait()
        workspace = url.split("//", 1)[1].split(".", 1)[0]
        return {
            "id": f"chat-{workspace}",
            "content": [{"type": "text", "text": workspace}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    stream_requests = []

    def fake_async_stream_request(url, **kwargs):
        stream_requests.append((url, kwargs))
        workspace = url.split("//", 1)[1].split(".", 1)[0]

        async def events():
            yield SSEEvent(
                event="message_start",
                data={
                    "type": "message_start",
                    "message": {
                        "id": f"stream-{workspace}",
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
            for delta_text in (workspace, "-end"):
                yield SSEEvent(
                    event="content_block_delta",
                    data={
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": delta_text},
                    },
                )
            yield SSEEvent(
                event="message_delta",
                data={"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
            )
            yield SSEEvent(event="message_stop", data={"type": "message_stop"})

        return events()

    monkeypatch.setattr(async_client_module, "async_request", fake_async_request)
    monkeypatch.setattr(
        async_client_module,
        "async_stream_request",
        fake_async_stream_request,
    )

    async def chat(workspace):
        return await adapter.achat(
            [{"role": "user", "content": f"Hello {workspace}"}],
            max_tokens=8,
            workspace_id=workspace,
        )

    chat_a, chat_b = await asyncio.wait_for(
        asyncio.gather(chat("workspace-a"), chat("workspace-b")),
        timeout=5,
    )
    assert (chat_a.response_id, chat_a.content) == (
        "chat-workspace-a",
        "workspace-a",
    )
    assert (chat_b.response_id, chat_b.content) == (
        "chat-workspace-b",
        "workspace-b",
    )
    assert {
        url: kwargs["payload"]["messages"][0]["content"]
        for url, kwargs in chat_requests
    } == {
        "https://workspace-a.eu-central-1.maas.aliyuncs.com/apps/anthropic/v1/messages": "Hello workspace-a",
        "https://workspace-b.eu-central-1.maas.aliyuncs.com/apps/anthropic/v1/messages": "Hello workspace-b",
    }

    done = {"workspace-a": [], "workspace-b": []}

    def make_stream(workspace):
        return adapter.astream_chat(
            [{"role": "user", "content": f"Hello {workspace}"}],
            max_tokens=8,
            workspace_id=workspace,
            on_done=done[workspace].append,
        )

    stream_a = make_stream("workspace-a")
    stream_b = make_stream("workspace-b")
    assert stream_requests == []
    first_a, first_b = await asyncio.wait_for(
        asyncio.gather(stream_a.__anext__(), stream_b.__anext__()),
        timeout=5,
    )
    assert len(stream_requests) == 2
    assert {
        url: kwargs["payload"]["messages"][0]["content"]
        for url, kwargs in stream_requests
    } == {
        "https://workspace-a.eu-central-1.maas.aliyuncs.com/apps/anthropic/v1/messages": "Hello workspace-a",
        "https://workspace-b.eu-central-1.maas.aliyuncs.com/apps/anthropic/v1/messages": "Hello workspace-b",
    }
    second_a, second_b = await asyncio.wait_for(
        asyncio.gather(stream_a.__anext__(), stream_b.__anext__()),
        timeout=5,
    )
    assert (first_a, first_b) == ("workspace-a", "workspace-b")
    assert (second_a, second_b) == ("-end", "-end")
    with pytest.raises(StopAsyncIteration):
        await stream_a.__anext__()
    with pytest.raises(StopAsyncIteration):
        await stream_b.__anext__()
    assert len(done["workspace-a"]) == 1
    assert len(done["workspace-b"]) == 1
    assert done["workspace-a"][0].content == "workspace-a-end"
    assert done["workspace-a"][0].response_id == "stream-workspace-a"
    assert done["workspace-b"][0].content == "workspace-b-end"
    assert done["workspace-b"][0].response_id == "stream-workspace-b"


@pytest.mark.unit
def test_qwen_sync_workspace_streams_interleave_request_locally(qwen_runtime):
    class WorkspaceTransport(FakeSyncTransport):
        def post_sse(
            self,
            request,
            *,
            http_error_handler=None,
            stream_error_handler=None,
        ):
            self.sse_requests.append(request)
            workspace = request.url.split("//", 1)[1].split(".", 1)[0]
            return iter(
                [
                    SSEEvent(
                        event="message_start",
                        data={
                            "type": "message_start",
                            "message": {
                                "id": f"sync-{workspace}",
                                "model": "qwen3.8-max",
                                "content": [],
                                "usage": {"input_tokens": 1, "output_tokens": 0},
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
                    *[
                        SSEEvent(
                            event="content_block_delta",
                            data={
                                "type": "content_block_delta",
                                "index": 0,
                                "delta": {"type": "text_delta", "text": text},
                            },
                        )
                        for text in (workspace, "-end")
                    ],
                    SSEEvent(
                        event="message_delta",
                        data={
                            "type": "message_delta",
                            "delta": {"stop_reason": "end_turn"},
                        },
                    ),
                    SSEEvent(event="message_stop", data={"type": "message_stop"}),
                ]
            )

    adapter = UniversalLLMAPIAdapter(
        organization="qwen",
        model="qwen3.8-max",
        api_key="qwen-test-key",
    )
    transport = WorkspaceTransport({})
    adapter.adapter._sync_transport = transport
    done_a = []
    done_b = []
    stream_a = adapter.stream_chat(
        [{"role": "user", "content": "Hello alpha"}],
        max_tokens=8,
        workspace_id="workspace-a",
        on_done=done_a.append,
    )
    stream_b = adapter.stream_chat(
        [{"role": "user", "content": "Hello bravo"}],
        max_tokens=8,
        workspace_id="workspace-b",
        on_done=done_b.append,
    )

    assert transport.sse_requests == []
    assert next(stream_a) == "workspace-a"
    assert next(stream_b) == "workspace-b"
    assert len(transport.sse_requests) == 2
    assert {
        request.url: request.payload["messages"][0]["content"]
        for request in transport.sse_requests
    } == {
        "https://workspace-a.eu-central-1.maas.aliyuncs.com/apps/anthropic/v1/messages": "Hello alpha",
        "https://workspace-b.eu-central-1.maas.aliyuncs.com/apps/anthropic/v1/messages": "Hello bravo",
    }
    assert next(stream_a) == "-end"
    assert next(stream_b) == "-end"
    with pytest.raises(StopIteration):
        next(stream_a)
    with pytest.raises(StopIteration):
        next(stream_b)

    assert {request.url for request in transport.sse_requests} == {
        "https://workspace-a.eu-central-1.maas.aliyuncs.com/apps/anthropic/v1/messages",
        "https://workspace-b.eu-central-1.maas.aliyuncs.com/apps/anthropic/v1/messages",
    }
    assert len(done_a) == len(done_b) == 1
    assert (done_a[0].response_id, done_a[0].content) == (
        "sync-workspace-a",
        "workspace-a-end",
    )
    assert (done_b[0].response_id, done_b[0].content) == (
        "sync-workspace-b",
        "workspace-b-end",
    )


@pytest.mark.asyncio
@pytest.mark.unit
async def test_qwen_async_stream_prices_cache_reads_at_inclusive_cny_tier(
    qwen_runtime,
    monkeypatch,
):
    from llm_api_adapter_qwen.clients import async_client as async_client_module
    requests = []
    current_events = []

    def fake_async_stream_request(url, **kwargs):
        requests.append((url, kwargs))

        async def stream_events():
            for event in current_events:
                yield event

        return stream_events()

    monkeypatch.setattr(
        async_client_module,
        "async_stream_request",
        fake_async_stream_request,
    )
    cases = []
    for model in ("qwen3.7-plus", "qwen3.7-flash"):
        spec = resolve_model_spec(qwen_runtime, "qwen", model)
        for tier in spec.pricing_tiers.tiers:
            if tier.up_to_prompt_tokens is not None:
                cases.extend(
                    (model, total_input)
                    for total_input in (
                        tier.up_to_prompt_tokens,
                        tier.up_to_prompt_tokens + 1,
                    )
                )

    results = []
    for model, total_input in cases:
        adapter = UniversalLLMAPIAdapter(
            organization="qwen",
            model=model,
            api_key="qwen-test-key",
        )
        usage = {
            "input_tokens": total_input - 1,
            "output_tokens": 5,
            "cache_read_input_tokens": 1,
            "cache_creation_input_tokens": 7,
        }
        current_events[:] = qwen_messages_sse_events(model)
        current_events[0].data["message"]["usage"] = dict(usage)
        current_events[-2].data["usage"] = dict(usage)
        completed = []
        output = [
            text
            async for text in adapter.astream_chat(
                [{"role": "user", "content": "cached"}],
                max_tokens=16,
                workspace_id="frankfurt-workspace",
                on_done=completed.append,
            )
        ]
        assert len(completed) == 1
        results.append((output, completed[0], usage, model, total_input))

    assert len(cases) == 6
    assert len(requests) == len(cases)
    for output, result, usage, model, total_input in results:
        spec = resolve_model_spec(qwen_runtime, "qwen", model)
        inclusive_input = usage["input_tokens"] + usage["cache_read_input_tokens"]
        tier = spec.pricing_tiers.tier_for_prompt_tokens(inclusive_input)
        expected_input_cost = (
            usage["input_tokens"] * tier.in_per_token
            + usage["cache_read_input_tokens"] * tier.cache_read_in_per_token
        )
        expected_output_cost = usage["output_tokens"] * tier.out_per_token

        assert inclusive_input == total_input
        assert output == ["Hal", "lo"]
        assert result.currency == spec.pricing_tiers.currency == "CNY"
        assert result.usage.input_tokens == total_input
        assert result.usage.output_tokens == usage["output_tokens"]
        assert result.usage.total_tokens == total_input + usage["output_tokens"]
        assert result.usage.cached_tokens == 1
        assert result.usage.cache_write_tokens is None
        assert result.cost_input == pytest.approx(expected_input_cost)
        assert result.cost_output == pytest.approx(expected_output_cost)
        assert result.cost_total == pytest.approx(expected_input_cost + expected_output_cost)


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("model", ["qwen3.7-plus", "qwen3.7-flash"])
@pytest.mark.parametrize(
    "document",
    [
        DocumentPart(url="https://example.com/report.pdf"),
        DocumentPart(data=b"%PDF synthetic", media_type="application/pdf"),
    ],
)
async def test_qwen_async_rejects_documents_before_io(
    qwen_runtime,
    monkeypatch,
    model,
    document,
):
    from llm_api_adapter_qwen.clients import async_client as async_client_module

    chat_requests = []
    stream_requests = []

    async def fake_async_request(*args, **kwargs):
        chat_requests.append((args, kwargs))
        return {}

    def fake_async_stream_request(*args, **kwargs):
        stream_requests.append((args, kwargs))

        async def empty_events():
            if False:
                yield None

        return empty_events()

    monkeypatch.setattr(async_client_module, "async_request", fake_async_request)
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
    messages = [
        {"role": "user", "content": "Read this.", "files": [document]}
    ]
    with pytest.raises(ValueError, match="DocumentPart; PDF and OCR are unavailable"):
        await adapter.achat(
            messages,
            max_tokens=8,
            workspace_id="frankfurt-workspace",
        )
    with pytest.raises(ValueError, match="DocumentPart; PDF and OCR are unavailable"):
        await anext(adapter.astream_chat(
            messages,
            max_tokens=8,
            workspace_id="frankfurt-workspace",
        ))
    assert chat_requests == []
    assert stream_requests == []


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("model", QWEN_MODELS)
@pytest.mark.parametrize("parallel_tool_calls", [False, True])
async def test_qwen_async_rejects_supplied_parallel_control_before_io(
    qwen_runtime,
    monkeypatch,
    model,
    parallel_tool_calls,
):
    from llm_api_adapter_qwen.clients import async_client as async_client_module

    chat_requests = []
    stream_requests = []

    async def fake_async_request(*args, **kwargs):
        chat_requests.append((args, kwargs))
        return {}

    def fake_async_stream_request(*args, **kwargs):
        stream_requests.append((args, kwargs))

        async def empty_events():
            if False:
                yield None

        return empty_events()

    monkeypatch.setattr(async_client_module, "async_request", fake_async_request)
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
    message = [{"role": "user", "content": "Hello"}]

    with pytest.raises(NotImplementedError, match="parallel tool-call"):
        await adapter.achat(
            message,
            max_tokens=8,
            parallel_tool_calls=parallel_tool_calls,
            workspace_id="frankfurt-workspace",
        )
    with pytest.raises(NotImplementedError, match="parallel tool-call"):
        await anext(adapter.astream_chat(
            message,
            max_tokens=8,
            parallel_tool_calls=parallel_tool_calls,
            workspace_id="frankfurt-workspace",
        ))

    assert chat_requests == []
    assert stream_requests == []


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.parametrize("model", QWEN_MODELS)
async def test_qwen_async_chat_serializes_image_and_portable_schema(
    qwen_runtime,
    monkeypatch,
    model,
):
    from llm_api_adapter_qwen.clients import async_client as async_client_module

    requests = []

    async def fake_async_request(url, **kwargs):
        requests.append((url, kwargs))
        return {
            "id": "msg-qwen-async-image-schema",
            "model": model,
            "content": [{"type": "text", "text": '{"answer":"ok"}'}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    monkeypatch.setattr(async_client_module, "async_request", fake_async_request)
    adapter = UniversalLLMAPIAdapter(
        organization="qwen",
        model=model,
        api_key="qwen-test-key",
    )
    image = ImagePart(url="https://example.com/image.png")
    response = await adapter.achat(
        [
            {
                "role": "user",
                "content": "Describe this image as JSON.",
                "files": [image],
            }
        ],
        max_tokens=32,
        json_schema=FLAT_OBJECT_SCHEMA,
        workspace_id="frankfurt-workspace",
    )

    assert response.parsed_json == {"answer": "ok"}
    assert requests[0][1]["payload"]["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image as JSON."},
                {
                    "type": "image",
                    "source": {"type": "url", "url": "https://example.com/image.png"},
                },
            ],
        }
    ]
    assert requests[0][1]["payload"]["output_config"] == {
        "format": {"type": "json_schema", "schema": FLAT_OBJECT_SCHEMA}
    }


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


@pytest.mark.unit
@pytest.mark.parametrize(
    ("usage", "expected_input", "expected_output", "expected_total"),
    [
        ({"output_tokens": 0}, None, 0, None),
        ({"input_tokens": 0, "output_tokens": 0}, 0, 0, 0),
    ],
    ids=["input-omitted", "reported-zero"],
)
def test_qwen_preserves_omitted_and_reported_zero_usage(
    usage,
    expected_input,
    expected_output,
    expected_total,
):
    from llm_api_adapter_qwen.adapter import QwenAdapter

    parsed = QwenAdapter._parse_response(
        {
            "content": [{"type": "text", "text": "ok"}],
            "usage": usage,
        },
        capture_reasoning=False,
    )

    assert parsed.usage is not None
    assert parsed.usage.input_tokens == expected_input
    assert parsed.usage.output_tokens == expected_output
    assert parsed.usage.total_tokens == expected_total
    assert parsed.usage.cached_tokens is None
    stream_usage = QwenMessagesStreamParser._normalize_usage(usage)
    assert stream_usage is not None
    assert stream_usage.input_tokens == expected_input
    assert stream_usage.output_tokens == expected_output
    assert stream_usage.total_tokens == expected_total
    assert stream_usage.cached_tokens is None


@pytest.mark.unit
@pytest.mark.parametrize("cache_read_tokens", [-1, True, 1.5, "2"])
def test_qwen_rejects_malformed_automatic_cache_counts(cache_read_tokens):
    from llm_api_adapter_qwen.adapter import QwenAdapter

    with pytest.raises(LLMAPIClientError, match="cache_read_input_tokens"):
        QwenAdapter._parse_response(
            {
                "content": [{"type": "text", "text": "ok"}],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "cache_read_input_tokens": cache_read_tokens,
                },
            },
            capture_reasoning=False,
        )

    with pytest.raises(LLMAPIClientError, match="cache_read_input_tokens"):
        QwenMessagesStreamParser._normalize_usage(
            {
                "input_tokens": 10,
                "output_tokens": 2,
                "cache_read_input_tokens": cache_read_tokens,
            }
        )


@pytest.mark.integration
def test_qwen_sync_async_and_stream_share_cache_read_pricing(qwen_runtime, monkeypatch):
    from llm_api_adapter_qwen.clients import async_client as async_client_module

    usage = {
        "input_tokens": 256_000,
        "output_tokens": 5,
        "cache_read_input_tokens": 1,
    }
    response_payload = {
        "id": "msg-qwen-cache-parity",
        "model": "qwen3.7-plus",
        "content": [{"type": "text", "text": "cached"}],
        "stop_reason": "end_turn",
        "usage": usage,
    }
    events = qwen_messages_sse_events("qwen3.7-plus")
    events[0].data["message"]["usage"] = usage
    events[-2].data["usage"] = usage

    adapter = UniversalLLMAPIAdapter(
        organization="qwen",
        model="qwen3.7-plus",
        api_key="qwen-test-key",
    )
    adapter.adapter._sync_transport = FakeSyncTransport(
        response_payload,
        events=events,
    )
    sync_response = adapter.chat(
        [{"role": "user", "content": "cached"}],
        max_tokens=16,
        workspace_id="frankfurt-workspace",
    )
    completed = []
    list(
        adapter.stream_chat(
            [{"role": "user", "content": "cached"}],
            max_tokens=16,
            workspace_id="frankfurt-workspace",
            on_done=completed.append,
        )
    )

    async def fake_async_request(url, **kwargs):
        return response_payload

    monkeypatch.setattr(async_client_module, "async_request", fake_async_request)
    async_response = asyncio.run(
        adapter.achat(
            [{"role": "user", "content": "cached"}],
            max_tokens=16,
            workspace_id="frankfurt-workspace",
        )
    )

    expected_input_cost = (256_000 * 6.0 + 1 * 1.2) / 1_000_000
    expected_output_cost = 5 * 24.0 / 1_000_000
    for result in (sync_response, async_response, completed[0]):
        assert result.usage is not None
        assert result.usage.input_tokens == 256_001
        assert result.usage.output_tokens == 5
        assert result.usage.total_tokens == 256_006
        assert result.usage.cached_tokens == 1
        assert result.usage.cache_write_tokens is None
        assert result.cost_input == pytest.approx(expected_input_cost)
        assert result.cost_output == pytest.approx(expected_output_cost)
        assert result.cost_total == pytest.approx(
            expected_input_cost + expected_output_cost
        )


def _install_forced_tool_modes(adapter, modes, monkeypatch):
    from llm_api_adapter_qwen import registry as qwen_registry

    policies = dict(qwen_registry.QWEN_MODEL_POLICIES)
    model = adapter.adapter.model
    if modes is not None:
        policies[model] = frozenset(modes)
    else:
        policies.pop(model, None)
    monkeypatch.setattr(qwen_registry, "QWEN_MODEL_POLICIES", MappingProxyType(policies))


@pytest.mark.unit
@pytest.mark.parametrize(
    (
        "model",
        "modes",
        "reasoning_level",
        "tool_choice",
        "expected_reasoning",
        "should_warn",
        "expected_wire_choice",
    ),
    [
        # Missing rule must not borrow the Qwen-wide forced-tool behavior.
        (
            "qwen3.8-max",
            None,
            "medium",
            "any",
            ("output_config", {"effort": "medium"}),
            False,
            {"type": "any"},
        ),
        # A declared any-only rule does not apply to a named tool choice.
        (
            "qwen3.7-plus",
            ("any",),
            2048,
            "get_weather",
            ("thinking", {"type": "enabled", "budget_tokens": 2048}),
            False,
            {"type": "tool", "name": "get_weather"},
        ),
        # A declared tool-only rule does not apply to any-tool selection.
        (
            "qwen3.8-flash",
            ("tool",),
            "medium",
            "any",
            ("output_config", {"effort": "medium"}),
            False,
            {"type": "any"},
        ),
        # Matching any policy changes thinking and warns exactly once.
        (
            "qwen3.8-max",
            ("any",),
            "medium",
            "any",
            ("thinking", {"type": "disabled"}),
            True,
            {"type": "any"},
        ),
        # Matching named-tool behavior is checked on a second existing profile.
        (
            "qwen3.7-flash",
            ("tool",),
            2048,
            "get_weather",
            ("thinking", {"type": "disabled"}),
            True,
            {"type": "tool", "name": "get_weather"},
        ),
        # Declared forced-tool behavior leaves automatic and disabled choices alone.
        (
            "qwen3.8-flash",
            ("any", "tool"),
            "medium",
            "auto",
            ("output_config", {"effort": "medium"}),
            False,
            {"type": "auto"},
        ),
        (
            "qwen3.8-flash",
            ("any", "tool"),
            "medium",
            "none",
            ("output_config", {"effort": "medium"}),
            False,
            {"type": "none"},
        ),
    ],
)
def test_qwen_forced_tool_policy_uses_only_declared_exact_modes(
    qwen_runtime,
    model,
    modes,
    reasoning_level,
    tool_choice,
    expected_reasoning,
    should_warn,
    expected_wire_choice,
    caplog,
    monkeypatch,
):
    adapter = UniversalLLMAPIAdapter(
        organization="qwen",
        model=model,
        api_key="qwen-test-key",
    )
    _install_forced_tool_modes(adapter, modes, monkeypatch)
    transport = FakeSyncTransport(
        {
            "id": "msg-qwen-forced-policy",
            "model": model,
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_qwen_policy",
                    "name": "get_weather",
                    "input": {},
                }
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )
    adapter.adapter._sync_transport = transport
    caplog.clear()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        adapter.chat(
            [{"role": "user", "content": "Use the tool."}],
            max_tokens=64,
            reasoning_level=reasoning_level,
            tools=[ToolSpec(name="get_weather", json_schema={"type": "object"})],
            tool_choice=tool_choice,
            workspace_id="frankfurt-workspace",
        )

    assert len(caught) == (1 if should_warn else 0)
    if should_warn:
        assert "disabled thinking because forced tool_choice" in str(caught[0].message)
    payload = transport.requests[0].payload
    assert payload[expected_reasoning[0]] == expected_reasoning[1]
    assert payload["tool_choice"] == expected_wire_choice
    policy_logs = [
        record
        for record in caplog.records
        if record.name == "llm_api_adapter_qwen.adapter"
        and "disabled thinking because forced tool_choice" in record.message
    ]
    assert len(policy_logs) == (1 if should_warn else 0)


@pytest.mark.unit
def test_qwen_bound_forced_tool_policy_preserves_schema_format_and_warns_once(
    qwen_runtime,
    caplog,
    monkeypatch,
):
    adapter = UniversalLLMAPIAdapter(
        organization="qwen",
        model="qwen3.8-max",
        api_key="qwen-test-key",
    )
    _install_forced_tool_modes(adapter, ("any", "tool"), monkeypatch)
    schema_format = deepcopy(
        {"type": "json_schema", "schema": FLAT_OBJECT_SCHEMA}
    )
    expected_format = deepcopy(schema_format)
    payload = {
        "thinking": {"type": "enabled", "budget_tokens": 2048},
        "output_config": {"format": schema_format, "effort": "high"},
    }

    expected_payload = deepcopy(payload)
    expected_payload["thinking"] = {"type": "disabled"}
    expected_payload["output_config"].pop("effort")
    with caplog.at_level(logging.WARNING, logger="llm_api_adapter_qwen.adapter"):
        with pytest.warns(
            UserWarning,
            match="disabled thinking because forced tool_choice",
        ) as caught:
            adapter.adapter._disable_thinking_for_forced_tool_choice(
                payload,
                normalized_tool_choice="any",
            )

    assert len(caught) == 1
    assert payload == expected_payload
    assert payload["output_config"]["format"] == expected_format
    assert len(caplog.records) == 1

    caplog.clear()
    with warnings.catch_warnings(record=True) as second_warnings:
        warnings.simplefilter("always")
        adapter.adapter._disable_thinking_for_forced_tool_choice(
            payload,
            normalized_tool_choice="any",
        )
    assert second_warnings == []
    assert caplog.records == []
    assert payload == expected_payload


UNKNOWN_QWEN_MODEL = "qwen-test-unregistered-native-id"


@pytest.mark.unit
@pytest.mark.parametrize("method", ["chat", "stream_chat"])
def test_qwen_unknown_native_model_keeps_unknown_selection_without_policy(
    qwen_runtime,
    method,
):
    with pytest.warns(UserWarning, match="is not verified") as selection_warnings:
        adapter = UniversalLLMAPIAdapter(
            organization="qwen",
            model=UNKNOWN_QWEN_MODEL,
            api_key="qwen-test-key",
        )
    assert len(selection_warnings) == 1
    assert adapter.adapter.model_spec is None
    transport = FakeSyncTransport(
        {
            "id": "msg-qwen-unknown",
            "model": UNKNOWN_QWEN_MODEL,
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
        events=qwen_messages_sse_events(UNKNOWN_QWEN_MODEL),
    )
    adapter.adapter._sync_transport = transport

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        if method == "chat":
            adapter.chat(
                [{"role": "user", "content": "Hello"}],
                max_tokens=64,
                tools=[ToolSpec(name="get_weather", json_schema={"type": "object"})],
                tool_choice="any",
                workspace_id="frankfurt-workspace",
            )
            request = transport.requests[0]
        else:
            list(
                adapter.stream_chat(
                    [{"role": "user", "content": "Hello"}],
                    max_tokens=64,
                    tools=[ToolSpec(name="get_weather", json_schema={"type": "object"})],
                    tool_choice={"type": "tool", "name": "get_weather"},
                    workspace_id="frankfurt-workspace",
                )
            )
            request = transport.sse_requests[0]

    assert captured == []
    assert len(transport.requests) + len(transport.sse_requests) == 1
    assert request.payload["model"] == UNKNOWN_QWEN_MODEL
    assert "thinking" not in request.payload
    assert "output_config" not in request.payload
    assert request.url == (
        "https://frankfurt-workspace.eu-central-1.maas.aliyuncs.com/"
        "apps/anthropic/v1/messages"
    )
    assert request.headers_dict() == {
        "x-api-key": "qwen-test-key",
        "Content-Type": "application/json",
    }
    assert "anthropic-version" not in request.headers_dict()


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("method", ["achat", "astream_chat"])
async def test_qwen_unknown_native_model_has_no_async_forced_tool_inference(
    qwen_runtime,
    monkeypatch,
    method,
):
    from llm_api_adapter_qwen.clients import async_client as async_client_module

    requests = []
    with pytest.warns(UserWarning, match="is not verified") as selection_warnings:
        adapter = UniversalLLMAPIAdapter(
            organization="qwen",
            model=UNKNOWN_QWEN_MODEL,
            api_key="qwen-test-key",
        )
    assert len(selection_warnings) == 1
    assert adapter.adapter.model_spec is None

    async def fake_async_request(url, **kwargs):
        requests.append((url, kwargs))
        return {
            "id": "msg-qwen-unknown-async",
            "model": UNKNOWN_QWEN_MODEL,
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    def fake_async_stream_request(url, **kwargs):
        requests.append((url, kwargs))

        async def events():
            for event in qwen_messages_sse_events(UNKNOWN_QWEN_MODEL):
                yield event

        return events()

    monkeypatch.setattr(async_client_module, "async_request", fake_async_request)
    monkeypatch.setattr(
        async_client_module,
        "async_stream_request",
        fake_async_stream_request,
    )

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        if method == "achat":
            await adapter.achat(
                [{"role": "user", "content": "Hello"}],
                max_tokens=64,
                tools=[ToolSpec(name="get_weather", json_schema={"type": "object"})],
                tool_choice="any",
                workspace_id="frankfurt-workspace",
            )
        else:
            [
                text
                async for text in adapter.astream_chat(
                    [{"role": "user", "content": "Hello"}],
                    max_tokens=64,
                    tools=[ToolSpec(name="get_weather", json_schema={"type": "object"})],
                    tool_choice={"type": "tool", "name": "get_weather"},
                    workspace_id="frankfurt-workspace",
                )
            ]

    assert captured == []
    assert len(requests) == 1
    url, kwargs = requests[0]
    payload = kwargs["payload"]
    assert payload["model"] == UNKNOWN_QWEN_MODEL
    assert "thinking" not in payload
    assert "output_config" not in payload
    assert url == (
        "https://frankfurt-workspace.eu-central-1.maas.aliyuncs.com/"
        "apps/anthropic/v1/messages"
    )
    assert kwargs["headers"] == {
        "x-api-key": "qwen-test-key",
        "Content-Type": "application/json",
    }
