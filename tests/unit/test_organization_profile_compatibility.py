"""Portable structured-output compatibility checks for organization packages."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for source in (
    REPOSITORY_ROOT / "src",
    REPOSITORY_ROOT / "packages" / "organizations" / "mistral" / "src",
    REPOSITORY_ROOT / "packages" / "organizations" / "xai" / "src",
):
    source_path = str(source)
    if source_path not in sys.path:
        sys.path.insert(0, source_path)


import llm_api_adapter.adapters.base_adapter as base_adapter_module
from llm_api_adapter.errors.llm_api_error import JSONSchemaError
from llm_api_adapter.llm_registry.llm_registry import RegistrySpec
from llm_api_adapter.llms.transports import JSONResponse, SSEEvent
from llm_api_adapter.models.messages.chat_message import UserMessage
from llm_api_adapter_mistral.adapter import MistralAdapter
from llm_api_adapter_mistral.plugin import PLUGIN as MISTRAL_PLUGIN
from llm_api_adapter_xai.adapter import XAIAdapter
from llm_api_adapter_xai.plugin import PLUGIN as XAI_PLUGIN
from llm_api_adapter_xai.streaming import XAIResponsesStreamParser
from tests.fixtures.structured_output import (
    FLAT_OBJECT_SCHEMA,
    NESTED_PYDANTIC_RESPONSE_JSON,
    NestedPydanticResponse,
    PORTABLE_PROFILE_SCHEMAS,
)


_NESTED_PYDANTIC_PROVIDER_SCHEMA = {
    "type": "object",
    "properties": {
        "contact": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "title": "Name"},
            },
            "required": ["name"],
            "additionalProperties": False,
            "title": "PortableContact",
        },
    },
    "required": ["contact"],
    "additionalProperties": False,
    "title": "NestedPydanticResponse",
}


@dataclass
class _RecordedTransport:
    response: dict[str, Any]
    requests: list[Any] = field(default_factory=list)
    stream_events: list[SSEEvent] = field(default_factory=list)

    def post_json(self, request: Any, *, http_error_handler: Any = None) -> JSONResponse:
        self.requests.append(request)
        return JSONResponse(self.response)

    def post_sse(
        self,
        request: Any,
        *,
        http_error_handler: Any = None,
        stream_error_handler: Any = None,
    ):
        self.requests.append(request)
        return iter(self.stream_events)


@pytest.fixture
def organization_package_registry(monkeypatch):
    registry = RegistrySpec()
    for plugin in (MISTRAL_PLUGIN, XAI_PLUGIN):
        assert plugin.model_metadata is not None
        assert registry.register_organization_metadata(plugin.model_metadata) is True
    monkeypatch.setattr(base_adapter_module, "LLM_REGISTRY", registry)
    return registry


def _mistral_response(content: str) -> dict[str, Any]:
    return {
        "model": "mistral-small-2603",
        "choices": [{"message": {"content": content}}],
    }


def _xai_response(content: str) -> dict[str, Any]:
    return {
        "object": "response",
        "id": "resp-xai-structured-output",
        "model": "grok-4.6",
        "created_at": 1_774_274_151,
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": content}],
            }
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    }


def _structured_adapter(
    organization: str,
    content: str,
) -> tuple[MistralAdapter | XAIAdapter, _RecordedTransport]:
    if organization == "mistral":
        adapter = MistralAdapter(
            api_key="mistral-test-key",
            model="mistral-small-2603",
        )
        transport = _RecordedTransport(_mistral_response(content))
        adapter._sync_transport = transport
        return adapter, transport

    adapter = XAIAdapter(api_key="xai-test-key", model="grok-4.6")
    transport = _RecordedTransport(_xai_response(content))
    adapter._client._sync_transport = transport
    return adapter, transport


def _expected_structured_output_format(
    organization: str,
    schema: dict,
) -> dict[str, Any]:
    if organization == "mistral":
        return {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "strict": True,
                    "schema": schema,
                },
            }
        }
    return {
        "text": {
            "format": {
                "type": "json_schema",
                "name": "response",
                "schema": schema,
                "strict": True,
            }
        }
    }


@pytest.mark.unit
@pytest.mark.parametrize("organization", ("mistral", "xai"))
@pytest.mark.parametrize("fixture_name", sorted(PORTABLE_PROFILE_SCHEMAS))
def test_organization_packages_preserve_every_portable_raw_schema(
    organization,
    fixture_name,
    organization_package_registry,
):
    source_schema = deepcopy(PORTABLE_PROFILE_SCHEMAS[fixture_name])
    original_schema = deepcopy(source_schema)
    adapter, transport = _structured_adapter(organization, '{"answer": "ok"}')

    response = adapter.chat(
        messages=[UserMessage("Return JSON.")],
        json_schema=source_schema,
    )

    assert response.parsed_json == {"answer": "ok"}
    assert source_schema == original_schema
    expected = _expected_structured_output_format(organization, original_schema)
    for key, value in expected.items():
        assert transport.requests[0].payload[key] == value


@pytest.mark.unit
@pytest.mark.parametrize("organization", ("mistral", "xai"))
def test_organization_packages_share_the_normalized_nested_pydantic_schema(
    organization,
    organization_package_registry,
):
    source_schema = NestedPydanticResponse.model_json_schema()
    adapter, transport = _structured_adapter(
        organization,
        NESTED_PYDANTIC_RESPONSE_JSON,
    )

    response = adapter.chat(
        messages=[UserMessage("Return the nested JSON response.")],
        response_model=NestedPydanticResponse,
    )

    assert response.parsed_model == NestedPydanticResponse(
        contact={"name": "Ada"},
    )
    assert NestedPydanticResponse.model_json_schema() == source_schema
    expected = _expected_structured_output_format(
        organization,
        _NESTED_PYDANTIC_PROVIDER_SCHEMA,
    )
    for key, value in expected.items():
        assert transport.requests[0].payload[key] == value


@pytest.mark.unit
def test_xai_rejects_its_documented_boolean_schema_before_http(
    organization_package_registry,
):
    adapter, transport = _structured_adapter("xai", '{"answer": "ok"}')
    schema = {
        "type": "object",
        "properties": {"answer": True},
        "required": ["answer"],
        "additionalProperties": False,
    }

    with pytest.raises(JSONSchemaError, match="xAI structured output rejects boolean schemas"):
        adapter.chat(messages=[UserMessage("Return JSON.")], json_schema=schema)

    assert transport.requests == []


@pytest.mark.unit
@pytest.mark.parametrize("organization", ("mistral", "xai"))
def test_organization_packages_enforce_the_common_profile_before_http(
    organization,
    organization_package_registry,
):
    adapter, transport = _structured_adapter(organization, '{"answer": "ok"}')
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }

    with pytest.raises(
        JSONSchemaError,
        match=(
            f"{organization} structured-output schema at "
            "#/additionalProperties: Core portable profile"
        ),
    ):
        adapter.chat(messages=[UserMessage("Return JSON.")], json_schema=schema)

    assert transport.requests == []


@pytest.mark.unit
@pytest.mark.parametrize("organization", ("mistral", "xai"))
def test_organization_packages_skip_structured_parsing_for_native_truncation(
    organization,
    organization_package_registry,
):
    adapter, transport = _structured_adapter(organization, '{"answer": ')
    if organization == "mistral":
        transport.response["choices"][0]["finish_reason"] = "length"
        expected_reason = "length"
    else:
        transport.response["status"] = "incomplete"
        transport.response["incomplete_details"] = {
            "reason": "max_output_tokens",
        }
        expected_reason = "max_output_tokens"

    response = adapter.chat(
        messages=[UserMessage("Return JSON.")],
        json_schema=FLAT_OBJECT_SCHEMA,
    )

    assert response.incomplete_reason == expected_reason
    assert response.refusal is None
    assert response.parsed_json is None
    assert response.parsed_model is None


@pytest.mark.unit
def test_xai_package_exposes_an_explicit_responses_refusal_without_parsing(
    organization_package_registry,
):
    adapter, transport = _structured_adapter("xai", "unused")
    transport.response["output"][0]["content"] = [{
        "type": "refusal",
        "refusal": "I can't help with that.",
    }]

    response = adapter.chat(
        messages=[UserMessage("Return JSON.")],
        json_schema=FLAT_OBJECT_SCHEMA,
    )

    assert response.refusal == "I can't help with that."
    assert response.incomplete_reason is None
    assert response.parsed_json is None
    assert response.parsed_model is None


@pytest.mark.unit
def test_xai_stream_parser_finalizes_an_incomplete_response(
    organization_package_registry,
):
    state = XAIResponsesStreamParser.new_state(buffer_chars=None)
    final_response = _xai_response('{"answer": ')
    final_response["status"] = "incomplete"
    final_response["incomplete_details"] = {"reason": "max_output_tokens"}

    XAIResponsesStreamParser.consume_event(
        SSEEvent(
            event="response.incomplete",
            data={"response": final_response},
        ),
        state,
    )
    response = XAIResponsesStreamParser.finalize(
        state,
        model="grok-4.6",
    )

    assert response.finish_reason == "incomplete"
    assert response.incomplete_reason == "max_output_tokens"


@pytest.mark.unit
def test_mistral_stream_finalizes_a_truncated_structured_result(
    organization_package_registry,
):
    adapter, transport = _structured_adapter("mistral", '{"answer": ')
    transport.stream_events = [
        SSEEvent(
            event=None,
            data={
                "choices": [{
                    "index": 0,
                    "delta": {"content": '{"answer": '},
                    "finish_reason": "length",
                }],
            },
        ),
    ]
    completed = []

    list(
        adapter.stream_chat(
            messages=[UserMessage("Return JSON.")],
            json_schema=FLAT_OBJECT_SCHEMA,
            on_done=completed.append,
        ),
    )

    assert completed[0].finish_reason == "length"
    assert completed[0].incomplete_reason == "length"
    assert completed[0].parsed_json is None
