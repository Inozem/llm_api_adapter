"""Contract tests for the official direct Mistral adapter."""

from __future__ import annotations

import asyncio
import logging
import sys
import warnings
from pathlib import Path

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
import llm_api_adapter_mistral.adapter as mistral_adapter_module
import llm_api_adapter_mistral.clients.async_client as mistral_async_client_module
from llm_api_adapter.errors.llm_api_error import (
    LLMAPIClientError,
    LLMAPIAuthorizationError,
    LLMAPITokenLimitError,
    JSONSchemaError,
)
from llm_api_adapter.llm_registry.llm_registry import (
    RegistrySpec,
    resolve_metered_operation_spec,
    resolve_model_spec,
)
from llm_api_adapter.llm_registry.reasoning import resolve_reasoning_level
from llm_api_adapter.llms.transports import JSONResponse, SSEEvent
from llm_api_adapter.llms.httpx_transport import HttpxSyncTransport
from llm_api_adapter.llms.requests_transport import RequestsSyncTransport
from llm_api_adapter.models.messages.chat_message import UserMessage
from llm_api_adapter.models.messages.file_parts import DocumentPart, ImagePart
from llm_api_adapter.models.responses.chat_response import ChatResponse
from llm_api_adapter.models.tools import ToolSpec
from llm_api_adapter.service_provider_registry import ServiceProviderRegistry
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter
from llm_api_adapter_mistral.adapter import MistralAdapter
from llm_api_adapter_mistral.plugin import PLUGIN
from tests.fixtures.structured_output import (
    FLAT_OBJECT_SCHEMA,
    NESTED_PYDANTIC_RESPONSE_JSON,
    NestedPydanticResponse,
)


class FakeSyncTransport:
    def __init__(self, response: dict, events=(), responses=None) -> None:
        self.response = response
        self.responses = list(responses or [])
        self.events = list(events)
        self.requests = []
        self.stream_requests = []

    def post_json(self, request, *, http_error_handler=None):
        self.requests.append(request)
        response = self.responses.pop(0) if self.responses else self.response
        return JSONResponse(response)

    def post_sse(
        self,
        request,
        *,
        http_error_handler=None,
        stream_error_handler=None,
    ):
        self.stream_requests.append(request)
        return iter(self.events)


def _mistral_large_4_messages():
    return [
        UserMessage("Typed user message"),
        {"role": "assistant", "content": "Dict assistant message"},
        UserMessage("Typed follow-up"),
    ]


def _mistral_large_4_expected_messages():
    return [
        {"role": "user", "content": "Typed user message"},
        {"role": "assistant", "content": "Dict assistant message"},
        {"role": "user", "content": "Typed follow-up"},
    ]


def _assert_mistral_large_4_recognized(adapter, caught_warnings, caplog):
    assert adapter.adapter.model_spec is not None
    assert adapter.adapter.model_spec.name == "mistral-large-4"
    assert not any(
        "mistral-large-4" in str(item.message)
        and "not verified" in str(item.message)
        for item in caught_warnings
    )
    assert not any(
        record.levelno >= logging.WARNING
        and "Unverified model used: mistral-large-4" in record.getMessage()
        for record in caplog.records
    )


def _assert_large_4_reasoning_payload_parity(
    sync_transport,
    async_calls,
    expected_provider_value,
):
    sync_payload = sync_transport.requests[0].payload
    async_payload = async_calls[0][1]["payload"]
    assert sync_payload == async_payload
    assert sync_payload["reasoning_effort"] == expected_provider_value


@pytest.fixture
def mistral_runtime(monkeypatch):
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


@pytest.mark.integration
@pytest.mark.parametrize(
    ("transport", "transport_class"),
    [
        ("requests", RequestsSyncTransport),
        ("httpx", HttpxSyncTransport),
    ],
)
def test_mistral_large_4_sync_chat_recognition_and_transport(
    mistral_runtime,
    monkeypatch,
    caplog,
    transport,
    transport_class,
):
    caplog.set_level(logging.WARNING)
    response_payload = {
        "id": "cmpl-mistral-large-4-sync",
        "model": "mistral-large-4",
        "choices": [{"message": {"role": "assistant", "content": "Done"}}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 4, "total_tokens": 11},
    }
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        adapter = UniversalLLMAPIAdapter(
            organization="mistral",
            model="mistral-large-4",
            api_key="mistral-test-key",
            transport=transport,
        )
        assert isinstance(adapter.adapter._sync_transport, transport_class)
        fake_transport = FakeSyncTransport(response_payload)
        adapter.adapter._sync_transport = fake_transport

        response = adapter.chat(_mistral_large_4_messages())

    assert response.response_id == "cmpl-mistral-large-4-sync"
    assert response.usage is not None
    assert response.usage.total_tokens == 11
    request = fake_transport.requests[0]
    assert request.payload["model"] == "mistral-large-4"
    assert request.payload["messages"] == _mistral_large_4_expected_messages()
    _assert_mistral_large_4_recognized(adapter, caught_warnings, caplog)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("request_kwargs", "public_reasoning_level"),
    [
        ({}, "none"),
        ({"reasoning_level": "none"}, "none"),
        ({"reasoning_level": "high"}, "high"),
        ({"reasoning_level": "minimal"}, "minimal"),
        ({"reasoning_level": 1234}, 1234),
    ],
    ids=["omitted", "none", "high", "other-string", "integer"],
)
def test_mistral_large_4_sync_async_reasoning_effort_matches_shared_resolver(
    mistral_runtime,
    monkeypatch,
    caplog,
    request_kwargs,
    public_reasoning_level,
):
    caplog.set_level(logging.WARNING)
    response_payload = {
        "id": "cmpl-mistral-large-4-reasoning",
        "model": "mistral-large-4",
        "choices": [{"message": {"role": "assistant", "content": "Done"}}],
    }
    sync_transport = FakeSyncTransport(response_payload)
    async_calls = []

    async def fake_async_request(url, **kwargs):
        async_calls.append((url, kwargs))
        return response_payload

    monkeypatch.setattr(
        mistral_async_client_module,
        "async_request",
        fake_async_request,
    )

    sync_adapter = UniversalLLMAPIAdapter(
        organization="mistral",
        model="mistral-large-4",
        api_key="mistral-test-key",
    )
    sync_adapter.adapter._sync_transport = sync_transport
    expected = resolve_reasoning_level(
        sync_adapter.adapter.model_spec,
        public_reasoning_level,
    )

    with warnings.catch_warnings(record=True) as sync_warnings:
        warnings.simplefilter("always")
        sync_adapter.chat(_mistral_large_4_messages(), **request_kwargs)

    async_adapter = UniversalLLMAPIAdapter(
        organization="mistral",
        model="mistral-large-4",
        api_key="mistral-test-key",
    )
    with warnings.catch_warnings(record=True) as async_warnings:
        warnings.simplefilter("always")
        asyncio.run(
            async_adapter.achat(_mistral_large_4_messages(), **request_kwargs)
        )

    expected_warnings = [expected.warning] if expected.warning else []
    assert [str(item.message) for item in sync_warnings] == expected_warnings
    assert [str(item.message) for item in async_warnings] == expected_warnings
    _assert_large_4_reasoning_payload_parity(
        sync_transport,
        async_calls,
        expected.provider_value,
    )
    assert sync_adapter.adapter.model_spec is not None
    assert sync_adapter.adapter.model_spec.name == "mistral-large-4"
    assert async_adapter.adapter.model_spec is not None
    assert async_adapter.adapter.model_spec.name == "mistral-large-4"
    assert not any(
        record.levelno >= logging.WARNING
        and "Unverified model used: mistral-large-4" in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.integration
def test_mistral_large_4_sync_async_capture_reasoning_alone_keeps_effort_at_none(
    mistral_runtime,
    monkeypatch,
):
    response_payload = {
        "id": "cmpl-mistral-large-4-capture",
        "model": "mistral-large-4",
        "choices": [{"message": {"role": "assistant", "content": "Done"}}],
    }
    sync_transport = FakeSyncTransport(response_payload)
    async_calls = []

    async def fake_async_request(url, **kwargs):
        async_calls.append((url, kwargs))
        return response_payload

    monkeypatch.setattr(
        mistral_async_client_module,
        "async_request",
        fake_async_request,
    )

    sync_adapter = UniversalLLMAPIAdapter(
        organization="mistral",
        model="mistral-large-4",
        api_key="mistral-test-key",
    )
    sync_adapter.adapter._sync_transport = sync_transport
    async_adapter = UniversalLLMAPIAdapter(
        organization="mistral",
        model="mistral-large-4",
        api_key="mistral-test-key",
    )

    sync_adapter.chat(
        _mistral_large_4_messages(),
        capture_reasoning=True,
    )
    asyncio.run(
        async_adapter.achat(
            _mistral_large_4_messages(),
            capture_reasoning=True,
        )
    )

    _assert_large_4_reasoning_payload_parity(sync_transport, async_calls, "none")


@pytest.mark.integration
def test_mistral_large_4_sync_stream_preserves_response_metadata(
    mistral_runtime,
    caplog,
):
    caplog.set_level(logging.WARNING)
    stream_event = SSEEvent(
        event=None,
        data={
            "id": "cmpl-mistral-large-4-sync-stream",
            "model": "mistral-large-4",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "Streamed"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 6, "completion_tokens": 2, "total_tokens": 8},
        },
    )
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        adapter = UniversalLLMAPIAdapter(
            organization="mistral",
            model="mistral-large-4",
            api_key="mistral-test-key",
        )
        fake_transport = FakeSyncTransport({}, events=[stream_event])
        adapter.adapter._sync_transport = fake_transport
        completed = []
        streamed = list(
            adapter.stream_chat(
                _mistral_large_4_messages(),
                on_done=completed.append,
            )
        )

    assert streamed == ["Streamed"]
    assert len(completed) == 1
    assert completed[0].response_id == "cmpl-mistral-large-4-sync-stream"
    assert completed[0].usage is not None
    assert completed[0].usage.total_tokens == 8
    payload = fake_transport.stream_requests[0].payload
    assert payload["model"] == "mistral-large-4"
    assert payload["messages"] == _mistral_large_4_expected_messages()
    _assert_mistral_large_4_recognized(adapter, caught_warnings, caplog)


@pytest.mark.integration
def test_mistral_large_4_async_chat_and_stream_preserve_request_and_response(
    mistral_runtime,
    monkeypatch,
    caplog,
):
    caplog.set_level(logging.WARNING)
    request_calls = []
    stream_calls = []

    async def fake_async_request(url, **kwargs):
        request_calls.append((url, kwargs))
        return {
            "id": "cmpl-mistral-large-4-async",
            "model": "mistral-large-4",
            "choices": [{"message": {"role": "assistant", "content": "Async done"}}],
            "usage": {
                "prompt_tokens": 9,
                "completion_tokens": 3,
                "total_tokens": 12,
            },
        }

    def fake_async_stream_request(url, **kwargs):
        stream_calls.append((url, kwargs))

        async def events():
            yield SSEEvent(
                event=None,
                data={
                    "id": "cmpl-mistral-large-4-async-stream",
                    "model": "mistral-large-4",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "Async streamed"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 8,
                        "completion_tokens": 5,
                        "total_tokens": 13,
                    },
                },
            )

        return events()

    monkeypatch.setattr(
        mistral_async_client_module,
        "async_request",
        fake_async_request,
    )
    monkeypatch.setattr(
        mistral_async_client_module,
        "async_stream_request",
        fake_async_stream_request,
    )

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        adapter = UniversalLLMAPIAdapter(
            organization="mistral",
            model="mistral-large-4",
            api_key="mistral-test-key",
        )

        async def run_calls():
            response = await adapter.achat(_mistral_large_4_messages())
            completed = []
            streamed = [
                text
                async for text in adapter.astream_chat(
                    _mistral_large_4_messages(),
                    on_done=completed.append,
                )
            ]
            return response, streamed, completed

        response, streamed, completed = asyncio.run(run_calls())

    assert response.response_id == "cmpl-mistral-large-4-async"
    assert response.usage is not None
    assert response.usage.total_tokens == 12
    assert streamed == ["Async streamed"]
    assert len(completed) == 1
    assert completed[0].response_id == "cmpl-mistral-large-4-async-stream"
    assert completed[0].usage is not None
    assert completed[0].usage.total_tokens == 13
    assert request_calls[0][1]["payload"]["model"] == "mistral-large-4"
    assert request_calls[0][1]["payload"]["messages"] == (
        _mistral_large_4_expected_messages()
    )
    assert stream_calls[0][1]["payload"]["model"] == "mistral-large-4"
    assert stream_calls[0][1]["payload"]["messages"] == (
        _mistral_large_4_expected_messages()
    )
    _assert_mistral_large_4_recognized(adapter, caught_warnings, caplog)


@pytest.mark.integration
def test_mistral_large_4_tool_result_history_is_sent_without_tool_execution(
    mistral_runtime,
):
    adapter = UniversalLLMAPIAdapter(
        organization="mistral",
        model="mistral-large-4",
        api_key="mistral-test-key",
    )
    tool = ToolSpec(
        name="get_weather",
        description="Get weather",
        json_schema={"type": "object", "properties": {"city": {"type": "string"}}},
    )
    transport = FakeSyncTransport(
        {
            "model": "mistral-large-4",
            "choices": [{"message": {"content": "It is sunny."}}],
        }
    )
    adapter.adapter._sync_transport = transport

    response = adapter.chat(
        [
            {"role": "user", "content": "Weather in Paris?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_weather_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city":"Paris"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_weather_1",
                "content": '{"temperature_c":21}',
            },
        ],
        tools=[tool],
        tool_choice="auto",
    )

    assert response.content == "It is sunny."
    payload = transport.requests[0].payload
    assert payload["model"] == "mistral-large-4"
    assert payload["messages"][2] == {
        "role": "tool",
        "tool_call_id": "call_weather_1",
        "content": '{"temperature_c":21}',
    }
    assert len(transport.requests) == 1


@pytest.mark.integration
def test_mistral_large_4_normalizes_provider_tool_calls(mistral_runtime):
    adapter = UniversalLLMAPIAdapter(
        organization="mistral",
        model="mistral-large-4",
        api_key="mistral-test-key",
    )
    tool = ToolSpec(name="get_weather", json_schema={"type": "object"})
    transport = FakeSyncTransport(
        {
            "model": "mistral-large-4",
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_large_4",
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
        }
    )
    adapter.adapter._sync_transport = transport

    response = adapter.chat(
        [{"role": "user", "content": "Weather in Paris?"}],
        tools=[tool],
        tool_choice="any",
    )

    assert response.tool_calls is not None
    assert response.tool_calls[0].name == "get_weather"
    assert response.tool_calls[0].arguments == {"city": "Paris"}
    assert response.tool_calls[0].call_id == "call_large_4"
    assert transport.requests[0].payload["model"] == "mistral-large-4"
    assert transport.requests[0].payload["tool_choice"] == "any"


@pytest.mark.integration
@pytest.mark.parametrize(
    ("tool_choice", "expected_payload"),
    [
        ("none", "none"),
        (
            "get_weather",
            {"type": "function", "function": {"name": "get_weather"}},
        ),
    ],
)
def test_mistral_large_4_accepts_none_and_valid_named_tool_choices(
    mistral_runtime, tool_choice, expected_payload
):
    adapter = UniversalLLMAPIAdapter(
        organization="mistral",
        model="mistral-large-4",
        api_key="mistral-test-key",
    )
    tool = ToolSpec(name="get_weather", json_schema={"type": "object"})
    transport = FakeSyncTransport(
        {
            "model": "mistral-large-4",
            "choices": [{"message": {"content": "Done."}}],
        }
    )
    adapter.adapter._sync_transport = transport

    adapter.chat(
        [{"role": "user", "content": "Weather?"}],
        tools=[tool],
        tool_choice=tool_choice,
    )

    assert transport.requests[0].payload["model"] == "mistral-large-4"
    assert transport.requests[0].payload["tool_choice"] == expected_payload


@pytest.mark.integration
def test_mistral_large_4_rejects_unknown_named_tool_before_transport(mistral_runtime):
    from llm_api_adapter.errors.llm_api_error import ToolChoiceError

    adapter = UniversalLLMAPIAdapter(
        organization="mistral",
        model="mistral-large-4",
        api_key="mistral-test-key",
    )
    transport = FakeSyncTransport({})
    adapter.adapter._sync_transport = transport

    with pytest.raises(ToolChoiceError):
        adapter.chat(
            [{"role": "user", "content": "Weather?"}],
            tools=[ToolSpec(name="get_weather", json_schema={"type": "object"})],
            tool_choice="missing_tool",
        )

    assert transport.requests == []


@pytest.mark.integration
def test_mistral_large_4_structured_output_schema_and_response_model(mistral_runtime):
    adapter = UniversalLLMAPIAdapter(
        organization="mistral",
        model="mistral-large-4",
        api_key="mistral-test-key",
    )
    transport = FakeSyncTransport(
        {},
        responses=[
            {
                "model": "mistral-large-4",
                "choices": [{"message": {"content": '{"answer":"ok"}'}}],
            },
            {
                "model": "mistral-large-4",
                "choices": [
                    {"message": {"content": NESTED_PYDANTIC_RESPONSE_JSON}}
                ],
            },
        ],
    )
    adapter.adapter._sync_transport = transport

    schema_response = adapter.chat(
        [{"role": "user", "content": "Answer as JSON."}],
        json_schema=FLAT_OBJECT_SCHEMA,
    )
    model_response = adapter.chat(
        [{"role": "user", "content": "Return contact."}],
        response_model=NestedPydanticResponse,
    )

    assert schema_response.parsed_json == {"answer": "ok"}
    assert model_response.parsed_json == {"contact": {"name": "Ada"}}
    assert model_response.parsed_model == NestedPydanticResponse(
        contact={"name": "Ada"}
    )
    for request in transport.requests:
        assert request.payload["model"] == "mistral-large-4"
        assert request.payload["response_format"]["type"] == "json_schema"
    assert transport.requests[0].payload["response_format"]["json_schema"]["schema"] == (
        FLAT_OBJECT_SCHEMA
    )


@pytest.mark.integration
def test_mistral_large_4_structured_preflight_and_terminal_errors(mistral_runtime):
    adapter = UniversalLLMAPIAdapter(
        organization="mistral",
        model="mistral-large-4",
        api_key="mistral-test-key",
    )
    preflight_transport = FakeSyncTransport({})
    adapter.adapter._sync_transport = preflight_transport
    with pytest.raises(JSONSchemaError):
        adapter.chat(
            [{"role": "user", "content": "Invalid schema."}],
            json_schema={"type": "object", "required": ["missing"]},
        )
    assert preflight_transport.requests == []

    for payload, kwargs in [
        (
            {"model": "mistral-large-4", "choices": [{"message": {"content": "{"}}]},
            {"json_schema": FLAT_OBJECT_SCHEMA},
        ),
        (
            {
                "model": "mistral-large-4",
                "choices": [{"message": {"content": '{"contact":{}}'}}],
            },
            {"response_model": NestedPydanticResponse},
        ),
    ]:
        transport = FakeSyncTransport(payload)
        adapter.adapter._sync_transport = transport
        with pytest.raises(JSONSchemaError):
            adapter.chat([{"role": "user", "content": "Structured."}], **kwargs)
        assert len(transport.requests) == 1
        assert transport.requests[0].payload["model"] == "mistral-large-4"


@pytest.mark.integration
def test_mistral_large_4_incomplete_response_has_no_parsed_result(mistral_runtime):
    adapter = UniversalLLMAPIAdapter(
        organization="mistral",
        model="mistral-large-4",
        api_key="mistral-test-key",
    )
    transport = FakeSyncTransport(
        {
            "model": "mistral-large-4",
            "choices": [{"finish_reason": "length", "message": {"content": None}}],
        }
    )
    adapter.adapter._sync_transport = transport

    response = adapter.chat(
        [{"role": "user", "content": "Structured."}],
        response_model=NestedPydanticResponse,
    )

    assert response.incomplete_reason == "length"
    assert response.parsed_json is None
    assert response.parsed_model is None
    assert transport.requests[0].payload["model"] == "mistral-large-4"


@pytest.mark.integration
def test_mistral_large_4_images_and_pdf_documents_use_existing_adaptations(
    mistral_runtime,
):
    adapter = UniversalLLMAPIAdapter(
        organization="mistral",
        model="mistral-large-4",
        api_key="mistral-test-key",
    )
    transport = FakeSyncTransport(
        {},
        responses=[
            {
                "model": "mistral-large-4",
                "choices": [{"message": {"content": "Image ok"}}],
            },
            {
                "model": "mistral-ocr-4-1",
                "pages": [{"markdown": "# One"}, {"markdown": "# Two"}],
                "usage_info": {"pages_processed": 2},
            },
            {
                "model": "mistral-ocr-4-1",
                "pages": [{"markdown": "# Three"}],
                "usage_info": {"pages_processed": 1},
            },
            {
                "model": "mistral-large-4",
                "choices": [{"message": {"content": "PDF ok"}}],
            },
        ],
    )
    adapter.adapter._sync_transport = transport

    image_response = adapter.chat(
        [
            UserMessage(
                "Describe these.",
                files=[
                    ImagePart(data=b"png", media_type="image/png"),
                    ImagePart(url="https://example.com/image.jpg"),
                ],
            )
        ]
    )
    pdf_response = adapter.chat(
        [
            UserMessage(
                "Summarize PDFs.",
                files=[
                    DocumentPart(data=b"%PDF", media_type="application/pdf"),
                    DocumentPart(url="https://example.com/file.pdf"),
                ],
            )
        ]
    )

    assert image_response.content == "Image ok"
    assert transport.requests[0].payload["model"] == "mistral-large-4"
    assert transport.requests[0].payload["messages"][0]["content"][1:] == [
        {"type": "image_url", "image_url": "data:image/png;base64,cG5n"},
        {"type": "image_url", "image_url": "https://example.com/image.jpg"},
    ]
    assert pdf_response.content == "PDF ok"
    assert [request.url for request in transport.requests[1:]] == [
        "https://api.mistral.ai/v1/ocr",
        "https://api.mistral.ai/v1/ocr",
        "https://api.mistral.ai/v1/chat/completions",
    ]
    assert [request.payload["model"] for request in transport.requests[1:]] == [
        "mistral-ocr-4-1",
        "mistral-ocr-4-1",
        "mistral-large-4",
    ]
    assert transport.requests[1].payload["document"]["document_url"] == (
        "data:application/pdf;base64,JVBERg=="
    )
    assert transport.requests[2].payload["document"]["document_url"] == (
        "https://example.com/file.pdf"
    )
    assert "# One" in transport.requests[3].payload["messages"][0]["content"]
    assert "# Two" in transport.requests[3].payload["messages"][0]["content"]
    assert "# Three" in transport.requests[3].payload["messages"][0]["content"]


@pytest.mark.integration
def test_mistral_large_4_preview_authorization_rejection_does_not_substitute_model(
    mistral_runtime,
):
    from types import SimpleNamespace

    adapter = UniversalLLMAPIAdapter(
        organization="mistral",
        model="mistral-large-4",
        api_key="mistral-test-key",
    )

    class RejectingTransport(FakeSyncTransport):
        def post_json(self, request, *, http_error_handler=None):
            self.requests.append(request)
            error = RuntimeError("preview access denied")
            error.response = SimpleNamespace(
                status_code=403,
                json=lambda: {
                    "error": {
                        "type": "authorization_error",
                        "message": "preview access denied",
                    }
                },
            )
            http_error_handler(error)
            raise AssertionError("authorization handler should raise")

    transport = RejectingTransport({})
    adapter.adapter._sync_transport = transport

    with pytest.raises(LLMAPIAuthorizationError):
        adapter.chat([{"role": "user", "content": "Test preview access."}])

    assert len(transport.requests) == 1
    assert transport.requests[0].payload["model"] == "mistral-large-4"


@pytest.mark.integration
@pytest.mark.parametrize("end_mode", ["failure", "close"])
def test_mistral_large_4_sync_stream_failure_and_close_skip_done_and_cleanup(
    mistral_runtime, end_mode
):
    adapter = UniversalLLMAPIAdapter(
        organization="mistral",
        model="mistral-large-4",
        api_key="mistral-test-key",
    )
    closed = []

    class CloseableStreamTransport:
        def __init__(self):
            self.stream_requests = []

        def post_sse(
            self,
            request,
            *,
            http_error_handler=None,
            stream_error_handler=None,
        ):
            self.stream_requests.append(request)

            def events():
                try:
                    yield SSEEvent(
                        event=None,
                        data={
                            "model": "mistral-large-4",
                            "choices": [
                                {"index": 0, "delta": {"content": "part"}}
                            ],
                        },
                    )
                    if end_mode == "failure":
                        raise LLMAPIClientError(detail="stream failed")
                    yield SSEEvent(
                        event=None,
                        data={
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": "later"},
                                    "finish_reason": "stop",
                                }
                            ]
                        },
                    )
                finally:
                    closed.append(True)

            return events()

    transport = CloseableStreamTransport()
    adapter.adapter._sync_transport = transport
    completed = []
    stream = adapter.stream_chat(
        [{"role": "user", "content": "Stream."}],
        on_done=completed.append,
    )

    if end_mode == "failure":
        with pytest.raises(LLMAPIClientError):
            list(stream)
    else:
        assert next(stream) == "part"
        stream.close()

    assert completed == []
    assert closed == [True]
    assert transport.stream_requests[0].payload["model"] == "mistral-large-4"


@pytest.mark.integration
def test_mistral_large_4_async_stream_cancellation_skips_done_and_closes(
    mistral_runtime, monkeypatch
):
    adapter = UniversalLLMAPIAdapter(
        organization="mistral",
        model="mistral-large-4",
        api_key="mistral-test-key",
    )
    stream_calls = []
    closed = []

    def fake_async_stream_request(url, **kwargs):
        stream_calls.append((url, kwargs))

        async def events():
            try:
                yield SSEEvent(
                    event=None,
                    data={
                        "model": "mistral-large-4",
                        "choices": [
                            {"index": 0, "delta": {"content": "first"}}
                        ],
                    },
                )
                await asyncio.Event().wait()
            finally:
                closed.append(True)

        return events()

    monkeypatch.setattr(
        mistral_async_client_module,
        "async_stream_request",
        fake_async_stream_request,
    )

    async def run_cancelled_stream():
        completed = []
        stream = adapter.astream_chat(
            [{"role": "user", "content": "Stream."}],
            on_done=completed.append,
        )
        with pytest.raises(asyncio.CancelledError):
            async for text in stream:
                raise asyncio.CancelledError
        return completed

    completed = asyncio.run(run_cancelled_stream())

    assert completed == []
    assert closed == [True]
    assert stream_calls[0][1]["payload"]["model"] == "mistral-large-4"


@pytest.mark.integration
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
    meter = resolve_metered_operation_spec(mistral_runtime, "mistral", "ocr")
    assert meter is not None
    assert (meter.model, meter.unit, meter.rate, meter.currency) == (
        "mistral-ocr-4-1",
        "page",
        0.004,
        "USD",
    )


@pytest.mark.integration
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
                "prompt_tokens_details": {"cached_tokens": 0},
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


@pytest.mark.integration
def test_mistral_accepts_previous_response_without_serializing_it(
    mistral_runtime,
):
    adapter = UniversalLLMAPIAdapter(
        organization="mistral",
        model="mistral-small-2603",
        api_key="mistral-test-key",
    )
    transport = FakeSyncTransport(
        {
            "model": "mistral-small-2603",
            "choices": [{"message": {"content": "Bonjour"}}],
        }
    )
    adapter.adapter._sync_transport = transport

    response = adapter.chat(
        [{"role": "user", "content": "Hello"}],
        previous_response=ChatResponse(response_id="cmpl-previous"),
    )

    assert response.content == "Bonjour"
    assert "previous_response" not in transport.requests[0].payload


@pytest.mark.integration
def test_mistral_serializes_image_bytes_and_urls_in_vision_format(
    mistral_runtime,
):
    adapter = UniversalLLMAPIAdapter(
        organization="mistral",
        model="mistral-large-2512",
        api_key="mistral-test-key",
    )
    transport = FakeSyncTransport(
        {
            "model": "mistral-large-2512",
            "choices": [{"message": {"content": "A blue square."}}],
        }
    )
    adapter.adapter._sync_transport = transport

    response = adapter.chat(
        [
            UserMessage(
                "Describe these images.",
                files=[
                    ImagePart(data=b"png", media_type="image/png"),
                    ImagePart(url="https://example.com/image.jpg"),
                ],
            )
        ]
    )

    assert response.content == "A blue square."
    assert transport.requests[0].payload["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe these images."},
                {
                    "type": "image_url",
                    "image_url": "data:image/png;base64,cG5n",
                },
                {
                    "type": "image_url",
                    "image_url": "https://example.com/image.jpg",
                },
            ],
        }
    ]


@pytest.mark.integration
def test_mistral_processes_document_bytes_and_urls_through_ocr(
    mistral_runtime,
):
    adapter = UniversalLLMAPIAdapter(
        organization="mistral",
        model="mistral-large-2512",
        api_key="mistral-test-key",
    )
    transport = FakeSyncTransport(
        {},
        responses=[
            {
                "model": "mistral-ocr-4-1",
                "pages": [{"markdown": "# One"}],
                "usage_info": {"pages_processed": 1},
            },
            {
                "model": "mistral-ocr-4-1",
                "pages": [{"markdown": "# Two"}, {"markdown": "More"}],
                "usage_info": {"pages_processed": 3},
            },
            {
                "model": "mistral-large-2512",
                "choices": [{"message": {"content": "A summary."}}],
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 3,
                    "total_tokens": 5,
                    "prompt_tokens_details": {"cached_tokens": 0},
                },
            },
        ],
    )
    adapter.adapter._sync_transport = transport

    response = adapter.chat(
        [
            UserMessage(
                "Summarize these documents.",
                files=[
                    DocumentPart(data=b"%PDF", media_type="application/pdf"),
                    DocumentPart(url="https://example.com/two.pdf"),
                ],
            )
        ]
    )

    assert response.content == "A summary."
    assert response.cost_input == pytest.approx(0.000001)
    assert response.cost_output == pytest.approx(0.0000045)
    assert response.cost_total == pytest.approx(0.0160055)
    assert response.cost_breakdown is not None
    assert [
        (item.operation, item.model, item.unit, item.quantity, item.rate, item.cost)
        for item in response.cost_breakdown
    ] == [
        ("ocr", "mistral-ocr-4-1", "page", 1.0, 0.004, 0.004),
        ("ocr", "mistral-ocr-4-1", "page", 3.0, 0.004, 0.012),
    ]
    assert [request.url for request in transport.requests] == [
        "https://api.mistral.ai/v1/ocr",
        "https://api.mistral.ai/v1/ocr",
        "https://api.mistral.ai/v1/chat/completions",
    ]
    assert transport.requests[0].payload == {
        "model": "mistral-ocr-4-1",
        "document": {
            "type": "document_url",
            "document_url": "data:application/pdf;base64,JVBERg==",
        },
    }
    assert transport.requests[1].payload == {
        "model": "mistral-ocr-4-1",
        "document": {
            "type": "document_url",
            "document_url": "https://example.com/two.pdf",
        },
    }
    assert transport.requests[2].payload["messages"] == [
        {
            "role": "user",
            "content": (
                "Summarize these documents.\n\n"
                "<document index=\"1\">\n# One\n</document>\n\n"
                "<document index=\"2\">\n# Two\n\n---\n\nMore\n</document>"
            ),
        }
    ]


@pytest.mark.integration
def test_mistral_keeps_known_ocr_costs_when_another_usage_is_unavailable(
    mistral_runtime,
):
    adapter = UniversalLLMAPIAdapter(
        organization="mistral",
        model="mistral-large-2512",
        api_key="mistral-test-key",
    )
    adapter.adapter._sync_transport = FakeSyncTransport(
        {},
        responses=[
            {
                "model": "mistral-ocr-4-1",
                "pages": [{"markdown": "# Known"}],
                "usage_info": {"pages_processed": 1},
            },
            {
                "model": "mistral-ocr-4-1",
                "pages": [{"markdown": "# Unknown"}],
            },
            {
                "model": "mistral-large-2512",
                "choices": [{"message": {"content": "A summary."}}],
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 3,
                    "prompt_tokens_details": {"cached_tokens": 0},
                },
            },
        ],
    )

    response = adapter.chat(
        [
            UserMessage(
                "Summarize these documents.",
                files=[
                    DocumentPart(url="https://example.com/known.pdf"),
                    DocumentPart(url="https://example.com/unknown.pdf"),
                ],
            )
        ]
    )

    assert response.cost_input == pytest.approx(0.000001)
    assert response.cost_output == pytest.approx(0.0000045)
    assert response.cost_total is None
    assert response.cost_breakdown is not None
    assert [item.quantity for item in response.cost_breakdown] == [1.0]


@pytest.mark.integration
def test_mistral_omits_total_when_the_ocr_meter_is_unavailable(
    mistral_runtime,
    monkeypatch,
):
    adapter = UniversalLLMAPIAdapter(
        organization="mistral",
        model="mistral-large-2512",
        api_key="mistral-test-key",
    )
    monkeypatch.setattr(adapter.adapter, "_ocr_meter", lambda: None)
    adapter.adapter._sync_transport = FakeSyncTransport(
        {},
        responses=[
            {
                "model": "mistral-ocr-4-1",
                "pages": [{"markdown": "# Meterless"}],
                "usage_info": {"pages_processed": 1},
            },
            {
                "model": "mistral-large-2512",
                "choices": [{"message": {"content": "A summary."}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3},
            },
        ],
    )

    response = adapter.chat(
        [
            UserMessage(
                "Summarize this document.",
                files=[DocumentPart(url="https://example.com/meterless.pdf")],
            )
        ]
    )

    assert response.cost_breakdown == []
    assert response.cost_total is None


@pytest.mark.unit
def test_mistral_prepared_documents_retain_ocr_usage_and_meter(mistral_runtime):
    adapter = UniversalLLMAPIAdapter(
        organization="mistral",
        model="mistral-large-2512",
        api_key="mistral-test-key",
    )
    transport = FakeSyncTransport(
        {},
        responses=[
            {
                "model": "mistral-ocr-4-1",
                "pages": [{"markdown": "# One"}],
                "usage_info": {"pages_processed": 1},
            },
            {
                "model": "mistral-ocr-4-1",
                "pages": [{"markdown": "# Two"}],
                "usage_info": {"pages_processed": 3},
            },
        ],
    )
    adapter.adapter._sync_transport = transport

    prepared = adapter.adapter._prepare_document_messages(
        [
            UserMessage(
                "Summarize these documents.",
                files=[
                    DocumentPart(data=b"%PDF", media_type="application/pdf"),
                    DocumentPart(url="https://example.com/two.pdf"),
                ],
            )
        ],
        timeout_s=None,
    )

    assert [operation.model for operation in prepared.ocr_operations] == [
        "mistral-ocr-4-1",
        "mistral-ocr-4-1",
    ]
    assert [operation.pages_processed for operation in prepared.ocr_operations] == [
        1,
        3,
    ]
    assert all(operation.meter is not None for operation in prepared.ocr_operations)
    assert [operation.meter.rate for operation in prepared.ocr_operations] == [
        0.004,
        0.004,
    ]
    assert prepared.messages.to_openai()[0]["content"] == (
        "Summarize these documents.\n\n"
        "<document index=\"1\">\n# One\n</document>\n\n"
        "<document index=\"2\">\n# Two\n</document>"
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("usage_info", "expected_pages"),
    [
        ({"pages_processed": 0}, 0),
        (None, None),
        ({}, None),
        ({"pages_processed": -1}, None),
        ({"pages_processed": True}, None),
    ],
)
def test_mistral_prepared_documents_mark_missing_or_malformed_ocr_usage_unavailable(
    mistral_runtime,
    usage_info,
    expected_pages,
):
    adapter = UniversalLLMAPIAdapter(
        organization="mistral",
        model="mistral-large-2512",
        api_key="mistral-test-key",
    )
    ocr_response = {
        "model": "mistral-ocr-4-1",
        "pages": [{"markdown": "# One"}],
    }
    if usage_info is not None:
        ocr_response["usage_info"] = usage_info
    adapter.adapter._sync_transport = FakeSyncTransport({}, responses=[ocr_response])

    prepared = adapter.adapter._prepare_document_messages(
        [
            UserMessage(
                "Summarize this document.",
                files=[DocumentPart(url="https://example.com/one.pdf")],
            )
        ],
        timeout_s=None,
    )

    assert prepared.ocr_operations[0].pages_processed == expected_pages
    assert prepared.ocr_operations[0].meter is not None


@pytest.mark.unit
def test_mistral_prepared_documents_leave_documentless_messages_unchanged(mistral_runtime):
    adapter = UniversalLLMAPIAdapter(
        organization="mistral",
        model="mistral-large-2512",
        api_key="mistral-test-key",
    )

    prepared = adapter.adapter._prepare_document_messages(
        [UserMessage("No documents.")],
        timeout_s=None,
    )

    assert prepared.ocr_operations == ()
    assert prepared.messages.to_openai() == [
        {"role": "user", "content": "No documents."}
    ]


@pytest.mark.integration
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


@pytest.mark.integration
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
    response = adapter.chat(
        [{"role": "user", "content": "Reply as JSON."}],
        json_schema=FLAT_OBJECT_SCHEMA,
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
                    "required": ["answer"],
                    "additionalProperties": False,
                },
        },
    }


@pytest.mark.integration
def test_mistral_streams_text_tools_reasoning_and_usage(mistral_runtime):
    adapter = UniversalLLMAPIAdapter(
        organization="mistral",
        model="mistral-small-2603",
        api_key="mistral-test-key",
    )
    transport = FakeSyncTransport(
        {},
        responses=[
            {
                "model": "mistral-ocr-4-1",
                "pages": [{"markdown": "# Stream document"}],
                "usage_info": {"pages_processed": 2},
            }
        ],
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
                        "prompt_tokens_details": {"cached_tokens": 0},
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
            [
                UserMessage(
                    "Say hello.",
                    files=[DocumentPart(url="https://example.com/stream.pdf")],
                )
            ],
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
    assert response.cost_total == pytest.approx(0.00800435)
    assert response.cost_breakdown is not None
    assert [item.quantity for item in response.cost_breakdown] == [2.0]

    assert [request.url for request in transport.requests] == [
        "https://api.mistral.ai/v1/ocr"
    ]
    stream_request = transport.stream_requests[0]
    assert stream_request.timeout is None
    assert stream_request.payload["stream"] is True
    assert stream_request.payload["stream_options"] == {"include_usage": True}


@pytest.mark.integration
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
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 3,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
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
        mistral_async_client_module,
        "async_request",
        fake_async_request,
    )
    monkeypatch.setattr(
        mistral_async_client_module,
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


@pytest.mark.integration
def test_mistral_async_chat_processes_document_through_ocr(
    mistral_runtime,
    monkeypatch,
):
    adapter = UniversalLLMAPIAdapter(
        organization="mistral",
        model="mistral-large-2512",
        api_key="mistral-test-key",
    )
    request_calls = []

    async def fake_async_request(url, **kwargs):
        request_calls.append((url, kwargs))
        if url == "https://api.mistral.ai/v1/ocr":
            return {
                "model": "mistral-ocr-4-1",
                "pages": [{"markdown": "# Async document"}],
                "usage_info": {"pages_processed": 2},
            }
        return {
            "model": "mistral-large-2512",
            "choices": [{"message": {"content": "Async summary."}}],
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 3,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
        }

    monkeypatch.setattr(
        mistral_async_client_module,
        "async_request",
        fake_async_request,
    )

    response = asyncio.run(
        adapter.achat(
            [
                UserMessage(
                    "Summarize this document.",
                    files=[
                        DocumentPart(
                            data=b"%PDF",
                            media_type="application/pdf",
                        )
                    ],
                )
            ]
        )
    )

    assert response.content == "Async summary."
    assert response.cost_total == pytest.approx(0.0080055)
    assert response.cost_breakdown is not None
    assert [item.quantity for item in response.cost_breakdown] == [2.0]
    assert [url for url, _ in request_calls] == [
        "https://api.mistral.ai/v1/ocr",
        "https://api.mistral.ai/v1/chat/completions",
    ]
    assert request_calls[1][1]["payload"]["messages"][0]["content"] == (
        "Summarize this document.\n\n"
        "<document index=\"1\">\n# Async document\n</document>"
    )


@pytest.mark.integration
def test_mistral_large_4_async_pdf_urls_and_bytes_use_ocr(mistral_runtime, monkeypatch):
    adapter = UniversalLLMAPIAdapter(
        organization="mistral",
        model="mistral-large-4",
        api_key="mistral-test-key",
    )
    request_calls = []

    async def fake_async_request(url, **kwargs):
        request_calls.append((url, kwargs))
        if url == "https://api.mistral.ai/v1/ocr":
            document = kwargs["payload"]["document"]["document_url"]
            markdown = "# Bytes PDF" if document.startswith("data:") else "# URL PDF"
            return {
                "model": "mistral-ocr-4-1",
                "pages": [{"markdown": markdown}],
                "usage_info": {"pages_processed": 1},
            }
        return {
            "id": "cmpl-mistral-large-4-async-pdf",
            "model": "mistral-large-4",
            "choices": [{"message": {"content": "Async PDF summary."}}],
        }

    monkeypatch.setattr(
        mistral_async_client_module,
        "async_request",
        fake_async_request,
    )

    response = asyncio.run(
        adapter.achat(
            [
                UserMessage(
                    "Summarize these PDFs.",
                    files=[
                        DocumentPart(data=b"%PDF", media_type="application/pdf"),
                        DocumentPart(url="https://example.com/async.pdf"),
                    ],
                )
            ]
        )
    )

    assert response.response_id == "cmpl-mistral-large-4-async-pdf"
    assert response.content == "Async PDF summary."
    assert [url for url, _ in request_calls] == [
        "https://api.mistral.ai/v1/ocr",
        "https://api.mistral.ai/v1/ocr",
        "https://api.mistral.ai/v1/chat/completions",
    ]
    assert [call[1]["payload"]["model"] for call in request_calls] == [
        "mistral-ocr-4-1",
        "mistral-ocr-4-1",
        "mistral-large-4",
    ]
    assert request_calls[0][1]["payload"]["document"]["document_url"] == (
        "data:application/pdf;base64,JVBERg=="
    )
    assert request_calls[1][1]["payload"]["document"]["document_url"] == (
        "https://example.com/async.pdf"
    )
    chat_payload = request_calls[2][1]["payload"]
    assert chat_payload["model"] == "mistral-large-4"
    chat_content = chat_payload["messages"][0]["content"]
    assert "# Bytes PDF" in chat_content
    assert "# URL PDF" in chat_content


@pytest.mark.integration
def test_mistral_async_stream_includes_ocr_costs(mistral_runtime, monkeypatch):
    adapter = UniversalLLMAPIAdapter(
        organization="mistral",
        model="mistral-large-2512",
        api_key="mistral-test-key",
    )
    request_calls = []
    stream_calls = []

    async def fake_async_request(url, **kwargs):
        request_calls.append((url, kwargs))
        return {
            "model": "mistral-ocr-4-1",
            "pages": [{"markdown": "# Async stream document"}],
            "usage_info": {"pages_processed": 2},
        }

    def fake_async_stream_request(url, **kwargs):
        stream_calls.append((url, kwargs))

        async def events():
            yield SSEEvent(
                event=None,
                data={
                    "id": "cmpl-mistral-async-stream-ocr",
                    "model": "mistral-large-2512",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "Async summary."},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 2,
                        "completion_tokens": 3,
                        "total_tokens": 5,
                        "prompt_tokens_details": {"cached_tokens": 0},
                    },
                },
            )

        return events()

    monkeypatch.setattr(
        mistral_async_client_module,
        "async_request",
        fake_async_request,
    )
    monkeypatch.setattr(
        mistral_async_client_module,
        "async_stream_request",
        fake_async_stream_request,
    )

    async def stream_document():
        completed = []
        streamed = [
            text
            async for text in adapter.astream_chat(
                [
                    UserMessage(
                        "Summarize this document.",
                        files=[DocumentPart(url="https://example.com/async-stream.pdf")],
                    )
                ],
                on_done=completed.append,
            )
        ]
        return streamed, completed

    streamed, completed = asyncio.run(stream_document())

    assert streamed == ["Async summary."]
    assert [url for url, _ in request_calls] == ["https://api.mistral.ai/v1/ocr"]
    assert stream_calls[0][0] == "https://api.mistral.ai/v1/chat/completions"
    assert len(completed) == 1
    response = completed[0]
    assert response.cost_total == pytest.approx(0.0080055)
    assert response.cost_breakdown is not None
    assert [item.quantity for item in response.cost_breakdown] == [2.0]


@pytest.mark.unit
def test_mistral_maps_context_errors_to_shared_error_hierarchy():
    with pytest.raises(LLMAPITokenLimitError):
        MistralAdapter._raise_mapped_error(
            status_code=400,
            error_type="context_length_exceeded",
            detail="context is too long",
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    (
        "usage",
        "expected_input",
        "expected_output",
        "expected_total",
        "expected_cache",
    ),
    [
        ({"completion_tokens": 13}, None, 13, None, None),
        (
            {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
            0,
            0,
            0,
            0,
        ),
    ],
    ids=["input-omitted", "reported-zero"],
)
def test_mistral_preserves_omitted_and_reported_zero_usage(
    usage,
    expected_input,
    expected_output,
    expected_total,
    expected_cache,
):
    parsed = MistralAdapter._parse_usage(usage)

    assert parsed is not None
    assert parsed.input_tokens == expected_input
    assert parsed.output_tokens == expected_output
    assert parsed.total_tokens == expected_total
    assert parsed.cached_tokens == expected_cache


@pytest.mark.unit
@pytest.mark.parametrize("cached_tokens", [-1, True, 1.5, "2"])
def test_mistral_rejects_malformed_automatic_cache_counts(cached_tokens):
    with pytest.raises(LLMAPIClientError, match="cached_tokens"):
        MistralAdapter._parse_usage(
            {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "prompt_tokens_details": {"cached_tokens": cached_tokens},
            }
        )


@pytest.mark.integration
def test_mistral_sync_async_and_stream_share_cache_read_accounting(
    mistral_runtime,
    monkeypatch,
):
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 5,
        "total_tokens": 105,
        "prompt_tokens_details": {"cached_tokens": 20},
    }
    response_payload = {
        "id": "cmpl-mistral-cache-parity",
        "model": "mistral-small-2603",
        "choices": [{"message": {"content": "cached"}}],
        "usage": usage,
    }
    stream_events = [
        SSEEvent(
            event=None,
            data={
                "id": response_payload["id"],
                "model": response_payload["model"],
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "cached"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": usage,
            },
        )
    ]
    adapter = UniversalLLMAPIAdapter(
        organization="mistral",
        model="mistral-small-2603",
        api_key="mistral-test-key",
    )
    adapter.adapter._sync_transport = FakeSyncTransport(
        response_payload,
        events=stream_events,
    )
    sync_response = adapter.chat([{"role": "user", "content": "cached"}])
    completed = []
    list(
        adapter.stream_chat(
            [{"role": "user", "content": "cached"}],
            on_done=completed.append,
        )
    )

    async def fake_async_request(url, **kwargs):
        return response_payload

    monkeypatch.setattr(
        mistral_async_client_module,
        "async_request",
        fake_async_request,
    )
    async_response = asyncio.run(
        adapter.achat([{"role": "user", "content": "cached"}])
    )

    expected_input_cost = (80 * 0.15 + 20 * 0.015) / 1_000_000
    expected_output_cost = 5 * 0.6 / 1_000_000
    for result in (sync_response, async_response, completed[0]):
        assert result.usage is not None
        assert result.usage.input_tokens == 100
        assert result.usage.output_tokens == 5
        assert result.usage.total_tokens == 105
        assert result.usage.cached_tokens == 20
        assert result.usage.cache_write_tokens is None
        assert result.cost_input == pytest.approx(expected_input_cost)
        assert result.cost_output == pytest.approx(expected_output_cost)
        assert result.cost_total == pytest.approx(
            expected_input_cost + expected_output_cost
        )
