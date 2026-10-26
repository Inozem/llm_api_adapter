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
    REPOSITORY_ROOT / "packages" / "organizations" / "zai" / "src",
    REPOSITORY_ROOT / "packages" / "organizations" / "xai" / "src",
):
    source_path = str(source)
    if source_path not in sys.path:
        sys.path.insert(0, source_path)


import llm_api_adapter.adapters.base_adapter as base_adapter_module
import llm_api_adapter.organization_registry as organization_registry_module
import llm_api_adapter.universal_adapter as universal_adapter_module
from llm_api_adapter.errors.llm_api_error import JSONSchemaError
from llm_api_adapter.llm_registry.llm_registry import (
    RegistrySpec,
    resolve_model_spec,
)
from llm_api_adapter.llms.transports import JSONResponse, SSEEvent
from llm_api_adapter.models.messages.chat_message import UserMessage
from llm_api_adapter.organization_registry import OrganizationPluginDiscovery
from llm_api_adapter.service_provider_registry import ServiceProviderRegistry
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter
from llm_api_adapter_mistral.adapter import MistralAdapter
from llm_api_adapter_mistral.plugin import PLUGIN as MISTRAL_PLUGIN
from llm_api_adapter_zai.adapter import ZaiAdapter
from llm_api_adapter_zai.plugin import PLUGIN as ZAI_PLUGIN
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


@dataclass
class _PluginEntryPoint:
    name: str
    value: str
    plugin: Any
    load_calls: int = 0

    def load(self) -> Any:
        self.load_calls += 1
        return self.plugin


@pytest.fixture
def organization_package_registry(monkeypatch):
    registry = RegistrySpec()
    for plugin in (MISTRAL_PLUGIN, XAI_PLUGIN):
        assert plugin.model_metadata is not None
        assert registry.register_organization_metadata(plugin.model_metadata) is True
    monkeypatch.setattr(base_adapter_module, "LLM_REGISTRY", registry)
    return registry


@pytest.mark.unit
def test_profile_parsing_preserves_registry_resolution_plugin_discovery_and_facade(
    monkeypatch,
):
    registry = RegistrySpec()
    base_spec = resolve_model_spec(registry, "openai", "gpt-6-astra")

    assert base_spec is not None
    assert tuple(
        (exception.capability_id, exception.behavior_id)
        for exception in base_spec.capability_exceptions or ()
    ) == (("reasoning_control", "none_falls_back_to_minimum"),)
    assert resolve_model_spec(registry, "openai", "gpt-6-astra") is base_spec
    snapshot_model = "gpt-6-astra-2026-07-01"
    assert resolve_model_spec(registry, "openai", snapshot_model) is base_spec
    assert resolve_model_spec(registry, "openai", "gpt-6-astra-2026-02-30") is None

    providers = ServiceProviderRegistry(
        {"openai": universal_adapter_module.OpenAIAdapter}
    )
    discovery = OrganizationPluginDiscovery()
    entry_point = _PluginEntryPoint(
        name="mistral",
        value="llm_api_adapter_mistral.plugin:PLUGIN",
        plugin=MISTRAL_PLUGIN,
    )

    def get_entry_points(*, group: str):
        assert (
            group
            == organization_registry_module.ORGANIZATION_PLUGIN_ENTRY_POINT_GROUP
        )
        return (entry_point,)

    monkeypatch.setattr(
        organization_registry_module,
        "entry_points",
        get_entry_points,
    )
    monkeypatch.setattr(
        universal_adapter_module,
        "SERVICE_PROVIDER_REGISTRY",
        providers,
    )
    monkeypatch.setattr(
        universal_adapter_module,
        "ORGANIZATION_PLUGIN_DISCOVERY",
        discovery,
    )
    monkeypatch.setattr(universal_adapter_module, "LLM_REGISTRY", registry)
    monkeypatch.setattr(base_adapter_module, "LLM_REGISTRY", registry)

    alias_facade = UniversalLLMAPIAdapter(
        organization="openai",
        model=snapshot_model,
        api_key="openai-test-key",
    )
    assert alias_facade.adapter.model == snapshot_model
    assert alias_facade.adapter.model_spec is base_spec

    facade = UniversalLLMAPIAdapter(
        organization="mistral",
        model="mistral-small-2603",
        api_key="mistral-test-key",
    )
    external_spec = resolve_model_spec(
        registry,
        "mistral",
        "mistral-small-2603",
    )
    assert external_spec is not None
    assert tuple(
        (exception.capability_id, exception.behavior_id)
        for exception in external_spec.capability_exceptions or ()
    ) == (
        ("pdf_url", "pass"),
        ("pdf_bytes", "pass"),
        ("provider_continuation", "ignored"),
    )
    assert resolve_model_spec(registry, "mistral", "unknown-model") is None
    assert isinstance(facade.adapter, MistralAdapter)
    assert facade.adapter.model_spec is external_spec
    assert entry_point.load_calls == 1

    transport = _RecordedTransport(_mistral_response("compatibility"))
    facade.adapter._sync_transport = transport
    response = facade.chat(messages=[UserMessage("Check facade compatibility.")])

    assert response.content == "compatibility"
    assert transport.requests[0].payload["model"] == "mistral-small-2603"


@pytest.mark.unit
def test_facade_separates_model_developer_from_mistral_service_provider(
    monkeypatch,
    organization_package_registry,
):
    monkeypatch.setattr(
        universal_adapter_module,
        "SERVICE_PROVIDER_REGISTRY",
        ServiceProviderRegistry({"mistral": MistralAdapter}),
    )

    facade = UniversalLLMAPIAdapter(
        organization="zai",
        model="glm-5.3-flash",
        api_key="mistral-test-key",
        service_provider="mistral",
    )

    assert isinstance(facade.adapter, MistralAdapter)
    assert facade.adapter.company == "zai"
    assert facade.adapter.service_provider == "mistral"
    assert facade.adapter.model_spec is resolve_model_spec(
        organization_package_registry,
        "mistral",
        "glm-5.3-flash",
    )


@pytest.mark.unit
@pytest.mark.parametrize("order", (("mistral", "zai"), ("zai", "mistral")))
def test_mistral_and_zai_same_named_profiles_are_isolated_across_discovery_orders(
    monkeypatch,
    order,
):
    registry = RegistrySpec()
    providers = ServiceProviderRegistry(
        {"openai": universal_adapter_module.OpenAIAdapter}
    )
    discovery = OrganizationPluginDiscovery()
    plugins = {"mistral": MISTRAL_PLUGIN, "zai": ZAI_PLUGIN}
    entries = {
        name: _PluginEntryPoint(
            name=name,
            value=f"llm_api_adapter_{name}.plugin:PLUGIN",
            plugin=plugins[name],
        )
        for name in plugins
    }
    active_order = list(order)

    def get_entry_points(*, group: str):
        assert group == organization_registry_module.ORGANIZATION_PLUGIN_ENTRY_POINT_GROUP
        return tuple(entries[name] for name in active_order)

    monkeypatch.setattr(organization_registry_module, "entry_points", get_entry_points)
    monkeypatch.setattr(universal_adapter_module, "SERVICE_PROVIDER_REGISTRY", providers)
    monkeypatch.setattr(universal_adapter_module, "ORGANIZATION_PLUGIN_DISCOVERY", discovery)
    monkeypatch.setattr(universal_adapter_module, "LLM_REGISTRY", registry)
    monkeypatch.setattr(base_adapter_module, "LLM_REGISTRY", registry)

    discovery.discover(providers, model_registry=registry)
    discovery.discover(providers, model_registry=registry)
    active_order.reverse()
    discovery.discover(providers, model_registry=registry)

    mistral_spec = resolve_model_spec(registry, "mistral", "glm-5.3-flash")
    zai_spec = resolve_model_spec(registry, "zai", "glm-5.3-flash")
    assert mistral_spec is registry.organizations["mistral"].models[
        "glm-5.3-flash"
    ]
    assert zai_spec is registry.organizations["zai"].models["glm-5.3-flash"]
    assert mistral_spec is not zai_spec
    assert mistral_spec.limits != zai_spec.limits
    assert mistral_spec.provider_model == "zai-glm-5-3-flash"
    assert zai_spec.provider_model is None
    for spec, plugin in (
        (mistral_spec, MISTRAL_PLUGIN),
        (zai_spec, ZAI_PLUGIN),
    ):
        assert plugin.model_metadata is not None
        source_tier = plugin.model_metadata.organization_data["models"][
            "glm-5.3-flash"
        ]["pricing_tiers"][0]
        resolved_tier = spec.pricing_tiers.tiers[0]
        assert resolved_tier.in_per_token == source_tier["input_per_1m"] / 1_000_000
        assert resolved_tier.out_per_token == source_tier["output_per_1m"] / 1_000_000

    mistral_facade = UniversalLLMAPIAdapter(
        organization="zai",
        model="glm-5.3-flash",
        api_key="mistral-test-key",
        service_provider="mistral",
    )
    zai_facade = UniversalLLMAPIAdapter(
        organization="zai",
        model="glm-5.3-flash",
        api_key="zai-test-key",
    )
    assert isinstance(mistral_facade.adapter, MistralAdapter)
    assert mistral_facade.adapter.company == "zai"
    assert mistral_facade.adapter.service_provider == "mistral"
    assert isinstance(zai_facade.adapter, ZaiAdapter)
    assert mistral_facade.adapter.model_spec is registry.organizations["mistral"].models[
        "glm-5.3-flash"
    ]
    assert mistral_facade.adapter.model_spec is mistral_spec
    assert mistral_facade.adapter.model_spec is not zai_spec
    assert zai_facade.adapter.model_spec is zai_spec

    transport = _RecordedTransport(_mistral_response("cross-route ok"))
    mistral_facade.adapter._sync_transport = transport
    response = mistral_facade.chat(messages=[UserMessage("test")])

    assert response.content == "cross-route ok"
    request = transport.requests[0]
    assert request.headers["Authorization"] == "Bearer mistral-test-key"
    assert request.payload["model"] == "zai-glm-5-3-flash"
    assert all(entry.load_calls == 1 for entry in entries.values())


@pytest.mark.unit
def test_direct_zai_provider_defaults_and_explicit_selection_preserve_request_contract(
    monkeypatch,
):
    registry = RegistrySpec()
    providers = ServiceProviderRegistry(
        {"openai": universal_adapter_module.OpenAIAdapter}
    )
    discovery = OrganizationPluginDiscovery()
    entries = (
        _PluginEntryPoint("mistral", "mistral:PLUGIN", MISTRAL_PLUGIN),
        _PluginEntryPoint("zai", "zai:PLUGIN", ZAI_PLUGIN),
    )
    monkeypatch.setattr(
        organization_registry_module,
        "entry_points",
        lambda *, group: entries,
    )
    monkeypatch.setattr(universal_adapter_module, "SERVICE_PROVIDER_REGISTRY", providers)
    monkeypatch.setattr(universal_adapter_module, "ORGANIZATION_PLUGIN_DISCOVERY", discovery)
    monkeypatch.setattr(universal_adapter_module, "LLM_REGISTRY", registry)
    monkeypatch.setattr(base_adapter_module, "LLM_REGISTRY", registry)

    implicit = UniversalLLMAPIAdapter(
        organization="zai",
        model="glm-5.3-flash",
        api_key="zai-secret-key",
    )
    explicit = UniversalLLMAPIAdapter(
        organization="zai",
        model="glm-5.3-flash",
        api_key="zai-secret-key",
        service_provider="zai",
    )
    assert isinstance(implicit.adapter, ZaiAdapter)
    assert isinstance(explicit.adapter, ZaiAdapter)
    assert implicit.adapter.model_spec is registry.organizations["zai"].models[
        "glm-5.3-flash"
    ]
    assert explicit.adapter.model_spec is implicit.adapter.model_spec

    response = {
        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
    }
    implicit_transport = _RecordedTransport(response)
    explicit_transport = _RecordedTransport(response)
    implicit.adapter._sync_transport = implicit_transport
    explicit.adapter._sync_transport = explicit_transport
    messages = [UserMessage("test")]
    assert implicit.chat(messages=messages).content == "ok"
    assert explicit.chat(messages=messages).content == "ok"
    implicit_request = implicit_transport.requests[0]
    explicit_request = explicit_transport.requests[0]
    assert implicit_request == explicit_request
    assert implicit_request.url == "https://api.z.ai/api/paas/v4/chat/completions"
    assert implicit_request.headers["Authorization"] == "Bearer zai-secret-key"
    assert implicit_request.payload["model"] == "glm-5.3-flash"
    assert implicit_request.payload == explicit_request.payload
    assert entries[0].load_calls == entries[1].load_calls == 1


@pytest.mark.unit
def test_mistral_only_discovery_keeps_native_catalogue_request_independent_of_zai(
    monkeypatch,
):
    registry = RegistrySpec()
    providers = ServiceProviderRegistry(
        {"openai": universal_adapter_module.OpenAIAdapter}
    )
    discovery = OrganizationPluginDiscovery()
    mistral_entry = _PluginEntryPoint(
        "mistral", "mistral:PLUGIN", MISTRAL_PLUGIN
    )
    monkeypatch.setattr(
        organization_registry_module,
        "entry_points",
        lambda *, group: (mistral_entry,),
    )
    monkeypatch.setattr(universal_adapter_module, "SERVICE_PROVIDER_REGISTRY", providers)
    monkeypatch.setattr(universal_adapter_module, "ORGANIZATION_PLUGIN_DISCOVERY", discovery)
    monkeypatch.setattr(universal_adapter_module, "LLM_REGISTRY", registry)
    monkeypatch.setattr(base_adapter_module, "LLM_REGISTRY", registry)

    facade = UniversalLLMAPIAdapter(
        organization="mistral",
        model="mistral-large-4",
        api_key="mistral-test-key",
    )
    assert isinstance(facade.adapter, MistralAdapter)
    assert "zai" not in registry.organizations
    assert providers.get("zai") is None
    assert facade.adapter.model_spec is registry.organizations["mistral"].models[
        "mistral-large-4"
    ]
    transport = _RecordedTransport(_mistral_response("native ok"))
    facade.adapter._sync_transport = transport
    assert facade.chat(messages=[UserMessage("test")]).content == "native ok"
    assert transport.requests[0].payload["model"] == "mistral-large-4"
    assert mistral_entry.load_calls == 1


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
