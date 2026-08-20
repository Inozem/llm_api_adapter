"""Contract tests for the official direct Mistral adapter."""

from __future__ import annotations

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
from llm_api_adapter.errors.llm_api_error import LLMAPITokenLimitError
from llm_api_adapter.llm_registry.llm_registry import RegistrySpec, resolve_model_spec
from llm_api_adapter.llms.transports import JSONResponse
from llm_api_adapter.models.tools import ToolSpec
from llm_api_adapter.provider_registry import ServiceProviderRegistry
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter
from llm_api_adapter_mistral.adapter import MistralAdapter
from llm_api_adapter_mistral.plugin import PLUGIN


class FakeSyncTransport:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.requests = []

    def post_json(self, request, *, http_error_handler=None):
        self.requests.append(request)
        return JSONResponse(self.response)


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
def test_mistral_maps_context_errors_to_shared_error_hierarchy():
    with pytest.raises(LLMAPITokenLimitError):
        MistralAdapter._raise_mapped_error(
            status_code=400,
            error_type="context_length_exceeded",
            detail="context is too long",
        )
