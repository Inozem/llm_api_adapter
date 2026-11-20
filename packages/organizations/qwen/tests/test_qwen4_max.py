"""Behavior contracts for Qwen4-Max before provider registration."""
from __future__ import annotations

import json
import base64
import copy
from contextlib import contextmanager
from io import StringIO
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace
import warnings

import pytest
from pydantic import BaseModel, ConfigDict

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[2]
for source in (PACKAGE_ROOT / "src", REPOSITORY_ROOT / "src"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

import llm_api_adapter.adapters.base_adapter as base_adapter_module
import llm_api_adapter.universal_adapter as universal_module
import llm_api_adapter.llm_registry.llm_registry as registry_module
from llm_api_adapter.llm_registry.llm_registry import (
    RegistrySpec,
    resolve_model_spec,
)
from llm_api_adapter.llm_registry.reasoning import resolve_reasoning_level
from llm_api_adapter.llms.transports import JSONResponse, SSEEvent
from llm_api_adapter.models.messages.file_parts import DocumentPart
from llm_api_adapter.models.tools import ToolSpec
from llm_api_adapter.service_provider_registry import ServiceProviderRegistry
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter
from llm_api_adapter.errors.llm_api_error import LLMAPIClientError

MODEL = "qwen4-max"
WORKSPACE = "frankfurt-workspace"
URL = f"https://{WORKSPACE}.eu-central-1.maas.aliyuncs.com/apps/anthropic/v1/messages"
SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}
_UNSET = object()
_KNOWN_MISSING_POLICY_ERROR = (
    "Qwen model 'qwen4-max' qwen_policy must contain only request_rules"
)


class StructuredAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str


class FakeSyncTransport:
    """Controlled sync HTTP boundary retaining native request objects."""

    def __init__(self, body=_UNSET, events=()):
        self.body = response() if body is _UNSET else body
        self.events = list(events)
        self.requests = []
        self.stream_requests = []

    def post_json(self, request, *, http_error_handler=None):
        self.requests.append(request)
        return JSONResponse(self.body)

    def post_sse(self, request, *, http_error_handler=None, stream_error_handler=None):
        self.stream_requests.append(request)
        return iter(self.events)


def response(model=MODEL, *, content=_UNSET, usage=_UNSET, extra=None):
    data = {
        "id": "msg-q4-test",
        "model": model,
        "content": ([{"type": "text", "text": "ok"}] if content is _UNSET else content),
        "stop_reason": "end_turn",
    }
    if usage is not _UNSET:
        data["usage"] = usage
    else:
        data["usage"] = {
            "input_tokens": 3,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "output_tokens": 2,
        }
    if extra:
        data.update(extra)
    return data


def simple_events(model=MODEL, *, usage=_UNSET, final_usage=_UNSET):
    start = {"type": "message_start", "message": {
        "id": "msg-q4-stream", "type": "message", "role": "assistant", "model": model,
        "content": [],
    }}
    if usage is _UNSET:
        start["message"]["usage"] = {
            "input_tokens": 3,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "output_tokens": 0,
        }
    else:
        start["message"]["usage"] = usage
    delta = {"type": "message_delta", "delta": {"stop_reason": "end_turn"}}
    if final_usage is _UNSET:
        delta["usage"] = {
            "input_tokens": 3,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "output_tokens": 2,
        }
    else:
        delta["usage"] = final_usage
    return [
        SSEEvent(event="message_start", data=start),
        SSEEvent(event="content_block_start", data={"type": "content_block_start", "index": 0,
            "content_block": {"type": "text", "text": ""}}),
        SSEEvent(event="content_block_delta", data={"type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": "ok"}}),
        SSEEvent(event="message_delta", data=delta),
        SSEEvent(event="message_stop", data={"type": "message_stop"}),
    ]


def _synthetic_catalogue():
    return {
        "currency": "QWEN-SYNTHETIC",
        "models": {
            MODEL: {
                "limits": {"context_window_tokens": 20, "max_output_tokens": 12},
                "reasoning_capability": {"allowed_values": ["none", "low", "high", "xhigh"]},
                "pricing_tiers": [{"up_to_prompt_tokens": None, "input_per_1m": 2.0,
                                   "output_per_1m": 4.0, "cache_read_input_per_1m": 1.0}],
                "qwen_policy": {
                    "reasoning_modes": {"default": "enabled",
                        "enabled": {"max_input_tokens": 18, "max_output_tokens": 10},
                        "disabled": {"max_input_tokens": 16, "max_output_tokens": 8}},
                    "input_token_accounting": "ordinary_plus_cache_read_and_write",
                    "request_rules": [],
                },
            },
            "qwen-test-forced-rule": {
            "limits": {"context_window_tokens": 20, "max_output_tokens": 12},
            "reasoning_capability": {"allowed_values": ["none", "low", "high", "xhigh"]},
            "pricing_tiers": [{"up_to_prompt_tokens": None, "input_per_1m": 2.0,
                               "output_per_1m": 4.0}],
            "qwen_policy": {
                "reasoning_modes": {"default": "enabled",
                    "enabled": {"max_input_tokens": 18, "max_output_tokens": 10},
                    "disabled": {"max_input_tokens": 16, "max_output_tokens": 8}},
                "request_rules": [{"handler": "disable_reasoning_for_forced_tools",
                                   "arguments": {"modes": ["any", "tool"]}}],
            },
            },
        },
        "metered_operations": {"pdf_pages": {"model": MODEL, "unit": "page", "rate": 0.25}},
    }


@pytest.fixture
def synthetic_profile(monkeypatch):
    """Load deliberately small fixture metadata through Qwen's package loader."""
    from llm_api_adapter_qwen import registry as qwen_registry
    from llm_api_adapter_qwen.plugin import PLUGIN

    catalogue = _synthetic_catalogue()

    class Resource:
        def joinpath(self, *parts):
            return self

        def open(self, *args, **kwargs):
            return StringIO(json.dumps(catalogue))

    import importlib
    import importlib.resources
    original_files = importlib.resources.files
    original_registries = (
        universal_module.LLM_REGISTRY,
        base_adapter_module.LLM_REGISTRY,
        universal_module.SERVICE_PROVIDER_REGISTRY,
        registry_module.LLM_REGISTRY,
    )
    state = SimpleNamespace(registry=None, error=None, catalogue=catalogue)

    def scoped_files(package):
        if package == "llm_api_adapter_qwen.registry":
            return Resource()
        return original_files(package)

    monkeypatch.setattr(importlib.resources, "files", scoped_files)
    try:
        try:
            qwen_registry = importlib.reload(qwen_registry)
        except ValueError as error:
            if str(error) != _KNOWN_MISSING_POLICY_ERROR:
                raise
            state.error = error
        else:
            state.registry = RegistrySpec()
            assert state.registry.register_organization_metadata(qwen_registry.MODEL_METADATA)
            providers = ServiceProviderRegistry()
            PLUGIN.register(providers)
            monkeypatch.setattr(universal_module, "LLM_REGISTRY", state.registry)
            monkeypatch.setattr(base_adapter_module, "LLM_REGISTRY", state.registry)
            monkeypatch.setattr(universal_module, "SERVICE_PROVIDER_REGISTRY", providers)
            monkeypatch.setattr(registry_module, "LLM_REGISTRY", state.registry)
        yield state
    finally:
        importlib.resources.files = original_files
        try:
            importlib.reload(qwen_registry)
        finally:
            universal_module.LLM_REGISTRY = original_registries[0]
            base_adapter_module.LLM_REGISTRY = original_registries[1]
            universal_module.SERVICE_PROVIDER_REGISTRY = original_registries[2]
            registry_module.LLM_REGISTRY = original_registries[3]


@pytest.fixture
def registered_profile(monkeypatch):
    """Use normal package registration; absence is asserted by each test body."""
    from llm_api_adapter_qwen.plugin import PLUGIN

    registry = RegistrySpec()
    assert PLUGIN.model_metadata is not None
    registry.register_organization_metadata(PLUGIN.model_metadata)
    providers = ServiceProviderRegistry()
    PLUGIN.register(providers)
    monkeypatch.setattr(universal_module, "LLM_REGISTRY", registry)
    monkeypatch.setattr(base_adapter_module, "LLM_REGISTRY", registry)
    monkeypatch.setattr(universal_module, "SERVICE_PROVIDER_REGISTRY", providers)
    monkeypatch.setattr(registry_module, "LLM_REGISTRY", registry)
    return SimpleNamespace(
        registry=registry,
        error=None,
        catalogue=PLUGIN.model_metadata.organization_data,
    )


@contextmanager
def _registered_catalogue_variant(catalogue):
    """Load a copied real catalogue through Qwen's normal metadata loader."""
    from llm_api_adapter_qwen import registry as qwen_registry
    from llm_api_adapter_qwen import plugin as plugin_module

    import importlib
    import importlib.resources

    class Resource:
        def joinpath(self, *parts):
            return self

        def open(self, *args, **kwargs):
            return StringIO(json.dumps(catalogue))

    original_files = importlib.resources.files
    original_registry_namespace = vars(qwen_registry).copy()
    original_plugin_namespace = vars(plugin_module).copy()
    original_registries = (
        universal_module.LLM_REGISTRY,
        base_adapter_module.LLM_REGISTRY,
        universal_module.SERVICE_PROVIDER_REGISTRY,
        registry_module.LLM_REGISTRY,
    )

    def scoped_files(package):
        if package == "llm_api_adapter_qwen.registry":
            return Resource()
        return original_files(package)

    importlib.resources.files = scoped_files
    try:
        importlib.reload(qwen_registry)
        importlib.reload(plugin_module)
        registry = RegistrySpec()
        assert plugin_module.PLUGIN.model_metadata is not None
        assert registry.register_organization_metadata(plugin_module.PLUGIN.model_metadata)
        providers = ServiceProviderRegistry()
        plugin_module.PLUGIN.register(providers)
        universal_module.LLM_REGISTRY = registry
        base_adapter_module.LLM_REGISTRY = registry
        registry_module.LLM_REGISTRY = registry
        universal_module.SERVICE_PROVIDER_REGISTRY = providers
        yield SimpleNamespace(
            registry=registry,
            error=None,
            catalogue=plugin_module.PLUGIN.model_metadata.organization_data,
        )
    finally:
        importlib.resources.files = original_files
        vars(qwen_registry).clear()
        vars(qwen_registry).update(original_registry_namespace)
        vars(plugin_module).clear()
        vars(plugin_module).update(original_plugin_namespace)
        universal_module.LLM_REGISTRY = original_registries[0]
        base_adapter_module.LLM_REGISTRY = original_registries[1]
        universal_module.SERVICE_PROVIDER_REGISTRY = original_registries[2]
        registry_module.LLM_REGISTRY = original_registries[3]


def _adapter(registry, *, transport="requests"):
    assert registry.error is None, f"synthetic Qwen policy extension is unavailable: {registry.error}"
    registry = registry.registry
    organization = registry.organizations.get("qwen")
    assert organization is not None and MODEL in organization.models
    adapter = UniversalLLMAPIAdapter(organization="qwen", model=MODEL,
                                      api_key="qwen-test-key", transport=transport)
    assert resolve_model_spec(registry, "qwen", MODEL) is adapter.adapter.model_spec
    return adapter


def _metadata(registry, model_name=MODEL):
    assert registry.error is None, f"Qwen4-Max is not registered: {registry.error}"
    registry = registry.registry
    organization = registry.organizations.get("qwen")
    assert organization is not None and model_name in organization.models
    return organization.models[model_name]


def _core_registry(profile):
    assert profile.error is None, f"profile is not available: {profile.error}"
    return profile.registry


def _model_data(profile, model=MODEL):
    assert profile.error is None, f"profile is not available: {profile.error}"
    models = profile.catalogue.get("models")
    assert isinstance(models, dict) and model in models, (
        f"selected model metadata is unavailable: {model}"
    )
    return models[model]


def _mode_cap(profile, reasoning_level, mode_key="max_output_tokens", model=MODEL):
    policy = _model_data(profile, model)["qwen_policy"]["reasoning_modes"]
    if reasoning_level is None:
        mode = policy["default"]
    else:
        model_spec = resolve_model_spec(_core_registry(profile), "qwen", model)
        resolution = resolve_reasoning_level(model_spec, reasoning_level)
        mode = "disabled" if resolution.provider_value in {"none", 0} else "enabled"
    return policy[mode][mode_key]


def _page_meter(profile, model=MODEL):
    _model_data(profile, model)
    organization = _core_registry(profile).organizations["qwen"]
    meters = [
        (name, meter)
        for name, meter in organization.metered_operations.items()
        if meter.model == model and meter.unit == "page"
    ]
    assert len(meters) == 1, (
        f"expected one registered page meter for selected model {model}; found {len(meters)}"
    )
    return meters[0]


@pytest.mark.unit
@pytest.mark.parametrize("transport", ["requests", "httpx"])
def test_registered_profile_uses_selected_native_sync_transport(registered_profile, monkeypatch, transport):
    """Exercise concrete selected transport with the outgoing HTTP call controlled."""
    metadata = _metadata(registered_profile)
    assert metadata.limits.context_window_tokens > 0
    adapter = _adapter(registered_profile, transport=transport)
    calls = []

    class FakeHTTPResponse:
        def raise_for_status(self):
            return None
        def json(self):
            return response()
        def close(self):
            return None

    if transport == "requests":
        import requests
        def post(*args, **kwargs):
            calls.append((args, kwargs))
            return FakeHTTPResponse()
        monkeypatch.setattr(requests, "post", post)
    else:
        import httpx
        class FakeClient:
            def post(self, *args, **kwargs):
                calls.append((args, kwargs))
                return FakeHTTPResponse()
            def close(self):
                return None
        monkeypatch.setattr(httpx, "Client", FakeClient)
    result = adapter.chat([{"role": "user", "content": "Hello"}], max_tokens=5,
                          workspace_id=WORKSPACE)
    assert result.content == "ok"
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] == URL
    assert kwargs["headers"] == {"x-api-key": "qwen-test-key", "Content-Type": "application/json"}
    assert kwargs["json"]["max_tokens"] == 5
    assert kwargs["json"]["model"] == MODEL


@pytest.mark.unit
@pytest.mark.parametrize("reasoning", [None, "none", "low", 10])
def test_registered_profile_output_mode_cap_boundary_and_overflow(registered_profile, reasoning):
    _metadata(registered_profile)
    cap = _mode_cap(registered_profile, reasoning)
    adapter = _adapter(registered_profile)
    accepted = FakeSyncTransport(response())
    adapter.adapter._sync_transport = accepted
    kwargs = {} if reasoning is None else {"reasoning_level": reasoning}
    adapter.chat([{"role": "user", "content": "Hello"}], max_tokens=cap,
                 workspace_id=WORKSPACE, **kwargs)
    assert accepted.requests[0].payload["max_tokens"] == cap
    rejected = FakeSyncTransport(response())
    adapter.adapter._sync_transport = rejected
    with pytest.raises(ValueError, match="max_tokens"):
        adapter.chat([{"role": "user", "content": "Hello"}], max_tokens=cap + 1,
                     workspace_id=WORKSPACE, **kwargs)
    assert rejected.requests == []


@pytest.mark.unit
def test_registered_profile_input_mode_caps_are_metadata_without_estimation(registered_profile):
    caps = _model_data(registered_profile)["qwen_policy"]["reasoning_modes"]
    assert caps["enabled"]["max_input_tokens"] > 0
    adapter = _adapter(registered_profile)
    transport = FakeSyncTransport(response())
    adapter.adapter._sync_transport = transport
    adapter.chat([{"role": "user", "content": "x" * 100_000}],
        max_tokens=_mode_cap(registered_profile, None), workspace_id=WORKSPACE)
    assert len(transport.requests) == 1


@pytest.mark.unit
@pytest.mark.parametrize("transport", ["requests", "httpx"])
def test_registered_profile_uses_selected_native_sync_stream(registered_profile, monkeypatch, transport):
    """Exercise the concrete selected SSE transport through a controlled boundary."""
    _metadata(registered_profile)
    adapter = _adapter(registered_profile, transport=transport)
    calls = []
    lines = []
    for event in simple_events():
        lines.extend([f"event: {event.event}", f"data: {json.dumps(event.data)}", ""])

    class FakeHTTPResponse:
        def raise_for_status(self):
            return None
        def iter_lines(self, decode_unicode=True):
            return iter(lines)
        def close(self):
            return None

    if transport == "requests":
        import requests
        def post(*args, **kwargs):
            calls.append((args, kwargs))
            return FakeHTTPResponse()
        monkeypatch.setattr(requests, "post", post)
    else:
        import httpx
        class FakeClient:
            def build_request(self, method, *args, **kwargs):
                calls.append(((method, *args), kwargs))
                return object()
            def send(self, request, stream=False):
                calls.append((("send", request), {"stream": stream}))
                return FakeHTTPResponse()
            def close(self):
                return None
        monkeypatch.setattr(httpx, "Client", FakeClient)

    assert "".join(adapter.stream_chat([{"role": "user", "content": "Hello"}],
                                        max_tokens=5, workspace_id=WORKSPACE)) == "ok"
    if transport == "requests":
        args, kwargs = calls[0]
        assert args[0] == URL and kwargs["stream"] is True
        assert kwargs["json"]["stream"] is True
    else:
        assert calls[0][0][0] == "POST" and calls[0][0][1] == URL
        assert calls[1][1]["stream"] is True


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("lane", [
    "chat_requests", "chat_httpx", "stream_requests", "stream_httpx", "achat", "astream_chat",
])
async def test_existing_qwen38_control_executes_each_native_lane(registered_profile, monkeypatch, lane):
    """Exercise current supported metadata before Qwen4-Max gates can fail."""
    model = "qwen3.8-max"
    assert model in _core_registry(registered_profile).organizations["qwen"].models
    transport_name = "httpx" if lane.endswith("httpx") else "requests"
    adapter = UniversalLLMAPIAdapter(organization="qwen", model=model,
        api_key="qwen-test-key", transport=transport_name)
    calls = []
    if lane.startswith(("chat_", "stream_")):
        streaming = lane.startswith("stream_")
        if transport_name == "requests":
            import requests
            class FakeResponse:
                def raise_for_status(self):
                    return None
                def json(self):
                    return response(model=model)
                def iter_lines(self, decode_unicode=True):
                    lines = []
                    for event in simple_events(model):
                        lines.extend([f"event: {event.event}", f"data: {json.dumps(event.data)}", ""])
                    return iter(lines)
                def close(self):
                    return None
            def post(*args, **kwargs):
                calls.append((args, kwargs))
                return FakeResponse()
            monkeypatch.setattr(requests, "post", post)
        else:
            import httpx
            class FakeResponse:
                def raise_for_status(self):
                    return None
                def json(self):
                    return response(model=model)
                def iter_lines(self, decode_unicode=True):
                    lines = []
                    for event in simple_events(model):
                        lines.extend([f"event: {event.event}", f"data: {json.dumps(event.data)}", ""])
                    return iter(lines)
                def close(self):
                    return None
            class FakeClient:
                def post(self, *args, **kwargs):
                    calls.append((args, kwargs))
                    return FakeResponse()
                def build_request(self, method, *args, **kwargs):
                    calls.append(((method, *args), kwargs))
                    return object()
                def send(self, request, stream=False):
                    calls.append((("send", request), {"stream": stream}))
                    return FakeResponse()
                def close(self):
                    return None
            monkeypatch.setattr(httpx, "Client", FakeClient)
        if streaming:
            assert "".join(adapter.stream_chat([{"role": "user", "content": "Hello"}],
                max_tokens=8, workspace_id=WORKSPACE)) == "ok"
        else:
            assert adapter.chat([{"role": "user", "content": "Hello"}], max_tokens=8,
                                workspace_id=WORKSPACE).content == "ok"
        first = calls[0]
        if transport_name == "requests":
            args, kwargs = first
            assert args[0] == URL
            assert kwargs["headers"] == {"x-api-key": "qwen-test-key", "Content-Type": "application/json"}
            assert kwargs["json"].get("stream", False) is streaming
        elif streaming:
            args, kwargs = first
            assert args[0] == "POST" and args[1] == URL
            assert kwargs["headers"] == {"x-api-key": "qwen-test-key", "Content-Type": "application/json"}
            assert kwargs["json"]["stream"] is True
            assert calls[1][1]["stream"] is True
        else:
            args, kwargs = first
            assert args[0] == URL
            assert kwargs["headers"] == {"x-api-key": "qwen-test-key", "Content-Type": "application/json"}
            assert kwargs["json"].get("stream", False) is False
    else:
        from llm_api_adapter_qwen.clients import async_client
        async def fake_request(url, **kwargs):
            calls.append((url, kwargs))
            return response(model=model)
        def fake_stream(url, **kwargs):
            calls.append((url, kwargs))
            async def events():
                for event in simple_events(model):
                    yield event
            return events()
        monkeypatch.setattr(async_client, "async_request", fake_request)
        monkeypatch.setattr(async_client, "async_stream_request", fake_stream)
        if lane == "achat":
            assert (await adapter.achat([{"role": "user", "content": "Hello"}],
                max_tokens=8, workspace_id=WORKSPACE)).content == "ok"
        else:
            assert "".join([part async for part in adapter.astream_chat(
                [{"role": "user", "content": "Hello"}], max_tokens=8,
                workspace_id=WORKSPACE)]) == "ok"
    assert len(calls) >= 1


@pytest.mark.unit
@pytest.mark.parametrize("reasoning,expected", [("none", {"type": "disabled"}),
    ("low", {"effort": "low"})])
def test_synthetic_profile_resolves_reasoning_and_caps(synthetic_profile, reasoning, expected):
    adapter = _adapter(synthetic_profile)
    transport = FakeSyncTransport(response())
    adapter.adapter._sync_transport = transport
    result = adapter.chat([{"role": "user", "content": "Hello"}], max_tokens=8,
                          reasoning_level=reasoning, workspace_id=WORKSPACE)
    payload = transport.requests[0].payload
    assert result.content == "ok"
    if reasoning == "none":
        assert payload["thinking"] == expected
    else:
        assert payload["output_config"]["effort"] == expected["effort"]
    assert payload["max_tokens"] == 8
    if reasoning == "none":
        assert payload["max_tokens"] == 8
        assert "effort" not in payload.get("output_config", {})


@pytest.mark.unit
@pytest.mark.parametrize("choice", ["any", {"type": "tool", "name": "get_weather"}])
@pytest.mark.parametrize("reasoning", [None, "none", "low"])
def test_synthetic_profile_forced_tools_do_not_invent_rule_or_disable_reasoning(
    synthetic_profile, choice, reasoning
):
    adapter = _adapter(synthetic_profile)
    output = [
        {"type": "thinking", "thinking": "private"},
        {"type": "text", "text": "visible"},
        {"type": "tool_use", "id": "toolu_weather", "name": "get_weather",
         "input": {"city": "Oslo", "date": "today"}},
    ]
    if choice == "any":
        output.append({"type": "tool_use", "id": "toolu_exchange", "name": "get_exchange_rate",
                       "input": {"base": "EUR", "quote": "JPY"}})
    transport = FakeSyncTransport(response(content=output))
    adapter.adapter._sync_transport = transport
    kwargs = {} if reasoning is None else {"reasoning_level": reasoning}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        answer = adapter.chat([{"role": "user", "content": "Weather?"}], max_tokens=8,
        capture_reasoning=True, **kwargs, tools=[
            ToolSpec(name="get_weather", json_schema={"type": "object", "properties": {"city": {"type": "string"}, "date": {"type": "string"}}}),
            ToolSpec(name="get_exchange_rate", json_schema={"type": "object", "properties": {"base": {"type": "string"}, "quote": {"type": "string"}}}),
        ],
        tool_choice=choice, workspace_id=WORKSPACE)
    payload = transport.requests[0].payload
    assert caught == []
    if reasoning is None:
        assert "thinking" not in payload and "output_config" not in payload
    elif reasoning == "none":
        assert payload["thinking"] == {"type": "disabled"}
    else:
        assert payload["output_config"]["effort"] == "low"
    assert [tool["name"] for tool in payload["tools"]] == ["get_weather", "get_exchange_rate"]
    assert payload["tools"][0]["input_schema"]["properties"] == {
        "city": {"type": "string"}, "date": {"type": "string"}
    }
    assert payload["tools"][1]["input_schema"]["properties"] == {
        "base": {"type": "string"}, "quote": {"type": "string"}
    }
    assert payload["tool_choice"] == (
        {"type": "any"} if choice == "any"
        else {"type": "tool", "name": "get_weather"}
    )
    assert answer.content == "visible"
    assert [event.text for event in answer.reasoning_events] == ["private"]
    assert [call.call_id for call in answer.tool_calls] == (
        ["toolu_weather", "toolu_exchange"] if choice == "any" else ["toolu_weather"]
    )
    assert answer.tool_calls[0].arguments == {"city": "Oslo", "date": "today"}
    if choice == "any":
        assert answer.tool_calls[1].arguments == {"base": "EUR", "quote": "JPY"}


@pytest.mark.unit
def test_synthetic_profile_r13_any_low_can_return_one_of_two_tools(synthetic_profile):
    adapter = _adapter(synthetic_profile)
    transport = FakeSyncTransport(response(content=[
        {"type": "thinking", "thinking": "private"},
        {"type": "text", "text": "visible"},
        {"type": "tool_use", "id": "toolu_weather", "name": "get_weather",
         "input": {"city": "Oslo", "date": "today"}},
    ]))
    adapter.adapter._sync_transport = transport
    tools = [
        ToolSpec(name="get_weather", json_schema={"type": "object", "properties": {
            "city": {"type": "string"}, "date": {"type": "string"}}}),
        ToolSpec(name="get_exchange_rate", json_schema={"type": "object", "properties": {
            "base": {"type": "string"}, "quote": {"type": "string"}}}),
    ]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        answer = adapter.chat([{"role": "user", "content": "Weather?"}],
            max_tokens=_mode_cap(synthetic_profile, "low"), reasoning_level="low",
            tools=tools, tool_choice="any", capture_reasoning=True,
            workspace_id=WORKSPACE)
    payload = transport.requests[0].payload
    assert caught == []
    assert [tool["name"] for tool in payload["tools"]] == ["get_weather", "get_exchange_rate"]
    assert payload["tool_choice"] == {"type": "any"}
    assert payload["output_config"]["effort"] == "low"
    assert answer.content == "visible"
    assert [event.text for event in answer.reasoning_events] == ["private"]
    assert len(answer.tool_calls) == 1
    assert answer.tool_calls[0].call_id == "toolu_weather"
    assert answer.tool_calls[0].arguments == {"city": "Oslo", "date": "today"}


@pytest.mark.unit
def test_synthetic_profile_omitted_reasoning_preserves_provider_default_by_omission(synthetic_profile):
    adapter = _adapter(synthetic_profile)
    transport = FakeSyncTransport(response())
    adapter.adapter._sync_transport = transport
    adapter.chat([{"role": "user", "content": "Hello"}],
                 max_tokens=_mode_cap(synthetic_profile, None), workspace_id=WORKSPACE)
    payload = transport.requests[0].payload
    assert payload["max_tokens"] == _mode_cap(synthetic_profile, None)
    assert "thinking" not in payload
    assert "output_config" not in payload

    rejected = FakeSyncTransport(response())
    adapter.adapter._sync_transport = rejected
    with pytest.raises(ValueError, match="max_tokens"):
        adapter.chat([{"role": "user", "content": "Hello"}], workspace_id=WORKSPACE)
    assert rejected.requests == []


@pytest.mark.unit
@pytest.mark.parametrize("reasoning", [None, "none", "low", 10])
def test_synthetic_profile_output_cap_boundary_and_just_over(reasoning, synthetic_profile):
    cap = _mode_cap(synthetic_profile, reasoning)
    adapter = _adapter(synthetic_profile)
    transport = FakeSyncTransport(response())
    adapter.adapter._sync_transport = transport
    adapter.chat([{"role": "user", "content": "Hello"}], max_tokens=cap,
                 reasoning_level=reasoning, workspace_id=WORKSPACE)
    assert transport.requests[0].payload["max_tokens"] == cap

    rejected = FakeSyncTransport(response())
    adapter.adapter._sync_transport = rejected
    with pytest.raises(ValueError, match="max_tokens"):
        adapter.chat([{"role": "user", "content": "Hello"}], max_tokens=cap + 1,
                     reasoning_level=reasoning, workspace_id=WORKSPACE)
    assert rejected.requests == []


@pytest.mark.unit
def test_synthetic_profile_output_cap_uses_post_rule_disabled_mode(synthetic_profile):
    """A declared synthetic rule is applied before selecting the output cap."""
    synthetic_name = "qwen-test-forced-rule"
    assert synthetic_profile.error is None, f"synthetic Qwen policy extension is unavailable: {synthetic_profile.error}"
    assert synthetic_name in _core_registry(synthetic_profile).organizations["qwen"].models
    adapter = UniversalLLMAPIAdapter(organization="qwen", model=synthetic_name,
                                     api_key="qwen-test-key")
    transport = FakeSyncTransport(response(model=synthetic_name))
    adapter.adapter._sync_transport = transport
    disabled_cap = _mode_cap(synthetic_profile, "none", model=synthetic_name)
    adapter.chat([{"role": "user", "content": "Weather"}], max_tokens=disabled_cap,
        reasoning_level="low", tools=[ToolSpec(name="get_weather", json_schema={"type": "object"})],
        tool_choice="any", workspace_id=WORKSPACE)
    payload = transport.requests[0].payload
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["max_tokens"] == disabled_cap
    rejected = FakeSyncTransport(response(model=synthetic_name))
    adapter.adapter._sync_transport = rejected
    with pytest.raises(ValueError, match="max_tokens"):
        adapter.chat([{"role": "user", "content": "Weather"}], max_tokens=disabled_cap + 1,
            reasoning_level="low", tools=[ToolSpec(name="get_weather", json_schema={"type": "object"})],
            tool_choice="any", workspace_id=WORKSPACE)
    assert rejected.requests == []


@pytest.mark.unit
def test_synthetic_profile_does_not_estimate_input_tokens(synthetic_profile):
    adapter = _adapter(synthetic_profile)
    transport = FakeSyncTransport(response())
    adapter.adapter._sync_transport = transport
    adapter.chat([{"role": "user", "content": "x" * 100_000}],
                 max_tokens=_mode_cap(synthetic_profile, None),
                 workspace_id=WORKSPACE)
    assert len(transport.requests) == 1


def _install_sync_http_boundary(monkeypatch, transport, body, events):
    calls = []
    lines = []
    for event in events:
        lines.extend([f"event: {event.event}", f"data: {json.dumps(event.data)}", ""])

    class FakeResponse:
        def raise_for_status(self):
            return None
        def json(self):
            return body
        def iter_lines(self, decode_unicode=True):
            return iter(lines)
        def close(self):
            return None

    if transport == "requests":
        import requests
        def post(*args, **kwargs):
            calls.append((args, kwargs))
            return FakeResponse()
        monkeypatch.setattr(requests, "post", post)
    else:
        import httpx
        class FakeClient:
            def post(self, *args, **kwargs):
                calls.append((args, kwargs))
                return FakeResponse()
            def build_request(self, method, *args, **kwargs):
                calls.append(((method, *args), kwargs))
                return object()
            def send(self, request, stream=False):
                calls.append((("send", request), {"stream": stream}))
                return FakeResponse()
            def close(self):
                return None
        monkeypatch.setattr(httpx, "Client", FakeClient)
    return calls


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("model", [MODEL, "qwen3.8-max", "qwen3.8-flash"])
@pytest.mark.parametrize("document", [
    ("url", DocumentPart(url="https://example.test/a.pdf")),
    ("bytes", DocumentPart(data=b"%PDF synthetic", media_type="application/pdf")),
])
@pytest.mark.parametrize("lane", [
    "chat_requests", "chat_httpx", "stream_requests", "stream_httpx", "achat", "astream_chat",
])
async def test_registered_profiles_serialize_pdf_for_all_native_lanes(
    registered_profile, monkeypatch, model, document, lane
):
    assert model in _core_registry(registered_profile).organizations["qwen"].models
    adapter = UniversalLLMAPIAdapter(organization="qwen", model=model, api_key="qwen-test-key",
        transport="httpx" if lane.endswith("httpx") else "requests")
    document_kind, part = document
    source = (
        {"type": "url", "url": part.url}
        if document_kind == "url"
        else {"type": "base64", "media_type": "application/pdf",
              "data": base64.b64encode(b"%PDF synthetic").decode("ascii")}
    )
    messages = [{"role": "user", "content": "Read this PDF", "files": [part]}]
    body = response(model=model, extra={"x_tools": {"pdf_page_parser": {"count": 2}}})
    completed = []
    if lane.startswith(("chat_", "stream_")):
        calls = _install_sync_http_boundary(
            monkeypatch,
            "httpx" if lane.endswith("httpx") else "requests",
            body,
            simple_events(model),
        )
        if lane.startswith("chat_"):
            result = adapter.chat(messages, max_tokens=8, workspace_id=WORKSPACE)
        else:
            list(adapter.stream_chat(messages, max_tokens=8, workspace_id=WORKSPACE,
                                     on_done=completed.append))
            result = completed[0]
        payload = calls[0][1]["json"]
    else:
        from llm_api_adapter_qwen.clients import async_client
        calls = []
        async def fake_request(url, **kwargs):
            calls.append((url, kwargs))
            return body
        def fake_stream(url, **kwargs):
            calls.append((url, kwargs))
            async def events():
                for event in simple_events(model):
                    yield event
            return events()
        monkeypatch.setattr(async_client, "async_request", fake_request)
        monkeypatch.setattr(async_client, "async_stream_request", fake_stream)
        if lane == "achat":
            result = await adapter.achat(messages, max_tokens=8, workspace_id=WORKSPACE)
        else:
            completed = []
            async for _ in adapter.astream_chat(messages, max_tokens=8, workspace_id=WORKSPACE,
                                                on_done=completed.append):
                pass
            result = completed[0]
        payload = calls[0][1]["payload"]
    assert payload["include_tool_usage"] is True
    assert payload["messages"][0]["content"][1] == {"type": "document", "source": source}
    if lane in {"stream_requests", "stream_httpx", "astream_chat"}:
        assert payload["stream"] is True
    if lane.startswith(("chat_", "stream_")):
        assert calls
    metadata = _metadata(registered_profile, model)
    token_input, token_output = _token_costs(metadata, 3, 0, 2)
    assert result.cost_input == pytest.approx(token_input)
    assert result.cost_output == pytest.approx(token_output)
    operation, meter = _page_meter(registered_profile, model)
    page_items = [line for line in result.cost_breakdown if line.operation == operation]
    if lane in {"stream_requests", "stream_httpx", "astream_chat"}:
        assert page_items == []
        assert result.cost_total is None
    else:
        assert len(page_items) == 1
        page_item = page_items[0]
        assert (page_item.model, page_item.unit, page_item.quantity, page_item.rate,
                page_item.currency) == (meter.model, meter.unit, 2, meter.rate, meter.currency)
        assert result.cost_total == pytest.approx(token_input + token_output + 2 * meter.rate)


def _token_costs(model, ordinary, cache_read, output):
    prompt_tokens = ordinary + cache_read
    tier = next(
        tier
        for tier in model.pricing_tiers.tiers
        if tier.up_to_prompt_tokens is None or prompt_tokens <= tier.up_to_prompt_tokens
    )
    read_rate = tier.cache_read_in_per_token
    input_cost = ordinary * tier.in_per_token
    if cache_read:
        assert read_rate is not None
        input_cost += cache_read * read_rate
    return input_cost, output * tier.out_per_token


@pytest.mark.parametrize("usage,expected_input", [
    ({"input_tokens": 10, "cache_read_input_tokens": 2,
      "cache_creation_input_tokens": 0, "output_tokens": 4}, 12),
    ({"input_tokens": 10, "cache_read_input_tokens": 2,
      "cache_creation_input_tokens": 0,
      "prompt_tokens_details": {"cached_tokens": 2}, "output_tokens": 4}, 12),
    ({"input_tokens": 0, "cache_read_input_tokens": 0,
      "cache_creation_input_tokens": 0, "output_tokens": 0}, 0),
])
@pytest.mark.unit
def test_synthetic_profile_prices_read_alias_once_and_preserves_zero(
    synthetic_profile, usage, expected_input
):
    model_data = _model_data(synthetic_profile)
    assert model_data["qwen_policy"]["input_token_accounting"] == "ordinary_plus_cache_read_and_write"
    adapter = _adapter(synthetic_profile)
    transport = FakeSyncTransport(response(usage=usage))
    adapter.adapter._sync_transport = transport
    answer = adapter.chat([{"role": "user", "content": "Hello"}], max_tokens=8,
                          workspace_id=WORKSPACE)
    model = _metadata(synthetic_profile)
    ordinary = usage["input_tokens"]
    read = usage.get("cache_read_input_tokens", usage.get("prompt_tokens_details", {}).get("cached_tokens", 0))
    input_cost, output_cost = _token_costs(model, ordinary, read, usage["output_tokens"])
    assert answer.usage.input_tokens == expected_input
    assert answer.cost_input == pytest.approx(input_cost)
    assert answer.cost_output == pytest.approx(output_cost)
    assert answer.cost_total == pytest.approx(input_cost + output_cost)


@pytest.mark.unit
def test_synthetic_profile_counts_unpriceable_write_but_withholds_input_total(synthetic_profile):
    adapter = _adapter(synthetic_profile)
    usage = {"input_tokens": 10, "cache_read_input_tokens": 2,
             "cache_creation_input_tokens": 3, "output_tokens": 4}
    transport = FakeSyncTransport(response(usage=usage))
    adapter.adapter._sync_transport = transport
    answer = adapter.chat([{"role": "user", "content": "Hello"}], max_tokens=8,
                          workspace_id=WORKSPACE)
    assert answer.usage.input_tokens == 15
    assert answer.cost_input is None
    model = _metadata(synthetic_profile)
    _, output_cost = _token_costs(model, 0, 0, usage["output_tokens"])
    assert answer.cost_output == pytest.approx(output_cost)
    assert answer.cost_total is None


@pytest.mark.unit
def test_synthetic_profile_missing_and_contradictory_usage_stays_unknown(synthetic_profile):
    adapter = _adapter(synthetic_profile)
    missing = FakeSyncTransport(response(usage={"output_tokens": 4}))
    adapter.adapter._sync_transport = missing
    missing_result = adapter.chat([{"role": "user", "content": "Hello"}], max_tokens=8,
                                  workspace_id=WORKSPACE)
    assert missing_result.usage.input_tokens is None
    assert missing_result.cost_input is None
    assert missing_result.cost_output is not None
    assert missing_result.cost_total is None

    contradictory = FakeSyncTransport(response(usage={
        "input_tokens": 10, "cache_read_input_tokens": 2,
        "cache_creation_input_tokens": 0,
        "prompt_tokens_details": {"cached_tokens": 3}, "output_tokens": 4,
    }))
    adapter.adapter._sync_transport = contradictory
    conflict_result = adapter.chat([{"role": "user", "content": "Hello"}], max_tokens=8,
                                   workspace_id=WORKSPACE)
    assert conflict_result.cost_input is None
    assert conflict_result.cost_total is None

    valid_read = FakeSyncTransport(response(usage={
        "input_tokens": 1, "cache_read_input_tokens": 2,
        "cache_creation_input_tokens": 0, "output_tokens": 4,
    }))
    adapter.adapter._sync_transport = valid_read
    valid_result = adapter.chat([{"role": "user", "content": "Hello"}],
        max_tokens=8, workspace_id=WORKSPACE)
    model = _metadata(synthetic_profile)
    expected_input_cost, expected_output_cost = _token_costs(model, 1, 2, 4)
    assert valid_result.usage.input_tokens == 3
    assert valid_result.cost_input == pytest.approx(expected_input_cost)
    assert valid_result.cost_output == pytest.approx(expected_output_cost)
    assert valid_result.cost_total == pytest.approx(expected_input_cost + expected_output_cost)


@pytest.mark.unit
def test_synthetic_profile_missing_write_count_is_unknown(synthetic_profile):
    adapter = _adapter(synthetic_profile)
    usage = {"input_tokens": 10, "cache_read_input_tokens": 2, "output_tokens": 4}
    adapter.adapter._sync_transport = FakeSyncTransport(response(usage=usage))
    answer = adapter.chat([{"role": "user", "content": "Hello"}], max_tokens=8,
                          workspace_id=WORKSPACE)
    assert answer.usage.input_tokens is None
    assert answer.cost_input is None
    assert answer.cost_output is not None
    assert answer.cost_total is None


@pytest.mark.unit
@pytest.mark.parametrize("field", ["input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"])
@pytest.mark.parametrize("value", [True, -1, 1.5, "1"])
def test_synthetic_profile_rejects_invalid_usage_component_types(synthetic_profile, field, value):
    adapter = _adapter(synthetic_profile)
    usage = {
        "input_tokens": 1,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "output_tokens": 1,
    }
    usage[field] = value
    adapter.adapter._sync_transport = FakeSyncTransport(response(usage=usage))
    with pytest.raises(LLMAPIClientError):
        adapter.chat([{"role": "user", "content": "Hello"}], max_tokens=8,
                     workspace_id=WORKSPACE)


@pytest.mark.unit
def test_synthetic_profile_prices_reported_pages_without_document_admission(synthetic_profile):
    operation, meter = _page_meter(synthetic_profile)
    adapter = _adapter(synthetic_profile)
    transport = FakeSyncTransport(response(extra={"x_tools": {"pdf_page_parser": {"count": 2}}}))
    adapter.adapter._sync_transport = transport
    answer = adapter.chat([{"role": "user", "content": "Ordinary text"}], max_tokens=8,
                          workspace_id=WORKSPACE)
    page_item = next(item for item in answer.cost_breakdown if item.operation == operation)
    assert (page_item.model, page_item.unit, page_item.quantity, page_item.rate) == (
        meter.model, meter.unit, 2, meter.rate
    )
    assert page_item.currency == meter.currency
    assert answer.cost_total == pytest.approx(answer.cost_input + answer.cost_output + meter.rate * 2)


@pytest.mark.unit
def test_synthetic_profile_stream_uses_late_usage_and_keeps_unpriceable_write_partial(synthetic_profile):
    adapter = _adapter(synthetic_profile)
    events = simple_events(
        usage=None,
        final_usage={"input_tokens": 10, "cache_read_input_tokens": 2,
                     "cache_creation_input_tokens": 3, "output_tokens": 4},
    )
    adapter.adapter._sync_transport = FakeSyncTransport(events=events)
    completed = []
    assert "".join(adapter.stream_chat([{"role": "user", "content": "Hello"}],
        max_tokens=8, workspace_id=WORKSPACE, on_done=completed.append)) == "ok"
    result = completed[0]
    assert result.usage.input_tokens == 15
    assert result.usage.output_tokens == 4
    assert result.cost_input is None
    assert result.cost_output is not None
    assert result.cost_total is None


@pytest.mark.unit
@pytest.mark.parametrize("usage,ordinary,cache_read,expected_count", [
    ({"input_tokens": 10, "cache_read_input_tokens": 2,
      "cache_creation_input_tokens": 0, "output_tokens": 4}, 10, 2, 12),
    ({"input_tokens": 10, "cache_creation_input_tokens": 0,
      "prompt_tokens_details": {"cached_tokens": 2},
      "output_tokens": 4}, 10, 2, 12),
    ({"input_tokens": 0, "cache_read_input_tokens": 0,
      "cache_creation_input_tokens": 0, "output_tokens": 0}, 0, 0, 0),
])
def test_registered_profile_prices_selected_metadata_rates_once(
    registered_profile, usage, ordinary, cache_read, expected_count
):
    model = _metadata(registered_profile)
    assert _model_data(registered_profile)["qwen_policy"]["input_token_accounting"] == (
        "ordinary_plus_cache_read_and_write"
    )
    adapter = _adapter(registered_profile)
    adapter.adapter._sync_transport = FakeSyncTransport(response(usage=usage))
    answer = adapter.chat([{"role": "user", "content": "Hello"}], max_tokens=8,
                          workspace_id=WORKSPACE)
    input_cost, output_cost = _token_costs(model, ordinary, cache_read, usage["output_tokens"])
    assert answer.usage.input_tokens == expected_count
    assert answer.cost_input == pytest.approx(input_cost)
    assert answer.cost_output == pytest.approx(output_cost)
    assert answer.cost_total == pytest.approx(input_cost + output_cost)


@pytest.mark.unit
def test_registered_profile_positive_write_has_complete_count_but_partial_cost(registered_profile):
    model = _metadata(registered_profile)
    adapter = _adapter(registered_profile)
    usage = {"input_tokens": 10, "cache_read_input_tokens": 2,
             "cache_creation_input_tokens": 3, "output_tokens": 4}
    adapter.adapter._sync_transport = FakeSyncTransport(response(usage=usage))
    answer = adapter.chat([{"role": "user", "content": "Hello"}], max_tokens=8,
                          workspace_id=WORKSPACE)
    _, output_cost = _token_costs(model, 0, 0, usage["output_tokens"])
    assert answer.usage.input_tokens == 15
    assert answer.cost_input is None
    assert answer.cost_output == pytest.approx(output_cost)
    assert answer.cost_total is None


@pytest.mark.unit
def test_registered_profile_unpriceable_cache_write_preserves_known_output_cost(registered_profile):
    _metadata(registered_profile)
    adapter = _adapter(registered_profile)
    transport = FakeSyncTransport(response(usage={"input_tokens": 8,
        "cache_read_input_tokens": 0, "cache_creation_input_tokens": 3,
        "output_tokens": 4}))
    adapter.adapter._sync_transport = transport
    answer = adapter.chat([{"role": "user", "content": "Hello"}], max_tokens=8, workspace_id=WORKSPACE)
    assert answer.cost_input is None
    assert answer.cost_output is not None
    assert answer.cost_total is None


@pytest.mark.unit
@pytest.mark.parametrize("model_name", [MODEL, "qwen3.8-max", "qwen3.8-flash"])
@pytest.mark.parametrize("method", ["chat", "stream_chat"])
def test_registered_profile_pdf_meter_is_separate_and_metadata_owned(
    registered_profile, method, model_name
):
    model = _metadata(registered_profile, model_name)
    operation, meter = _page_meter(registered_profile, model_name)
    adapter = UniversalLLMAPIAdapter(organization="qwen", model=model_name,
                                     api_key="qwen-test-key")
    transport = FakeSyncTransport(response(model=model_name,
        extra={"x_tools": {"pdf_page_parser": {"count": 2}}}),
        events=simple_events(model_name))
    adapter.adapter._sync_transport = transport
    messages = [{"role": "user", "content": "Read", "files": [DocumentPart(url="https://example.test/a.pdf")]}]
    completed = []
    result = adapter.chat(messages, max_tokens=8, workspace_id=WORKSPACE) if method == "chat" else None
    if method == "stream_chat":
        list(adapter.stream_chat(messages, max_tokens=8, workspace_id=WORKSPACE,
                                 on_done=completed.append))
        result = completed[0]
    token_input, token_output = _token_costs(model, 3, 0, 2)
    assert result.cost_input == pytest.approx(token_input)
    assert result.cost_output == pytest.approx(token_output)
    page_items = [line for line in result.cost_breakdown if line.operation == operation]
    if method == "chat":
        assert len(page_items) == 1
        page_item = page_items[0]
        assert (page_item.model, page_item.unit, page_item.quantity, page_item.rate) == (
            meter.model, meter.unit, 2, meter.rate
        )
        assert page_item.cost == pytest.approx(meter.rate * 2)
        assert result.cost_total == pytest.approx(token_input + token_output + page_item.cost)
    else:
        # No live evidence establishes a stream page-count event location.
        assert page_items == []
        assert result.cost_total is None


@pytest.mark.unit
@pytest.mark.parametrize("page_usage", [
    None,
    {},
    {"x_tools": {}},
    {"x_tools": {"pdf_page_parser": {}}},
    {"x_tools": {"pdf_page_parser": {"count": None}}},
    {"x_tools": {"pdf_page_parser": {"count": True}}},
    {"x_tools": {"pdf_page_parser": {"count": "2"}}},
    {"x_tools": {"pdf_page_parser": {"count": -1}}},
    {"x_tools": {"pdf_page_parser": {"count": 1.5}}},
])
def test_registered_profile_missing_or_invalid_page_count_withholds_total(
    registered_profile, page_usage
):
    model = _metadata(registered_profile)
    operation, _meter = _page_meter(registered_profile)
    adapter = _adapter(registered_profile)
    extra = {} if page_usage is None else page_usage
    transport = FakeSyncTransport(response(extra=extra))
    adapter.adapter._sync_transport = transport
    answer = adapter.chat([{"role": "user", "content": "Read",
        "files": [DocumentPart(url="https://example.test/a.pdf")]}], max_tokens=8,
        workspace_id=WORKSPACE)
    token_input, token_output = _token_costs(model, 3, 0, 2)
    assert answer.cost_input == pytest.approx(token_input)
    assert answer.cost_output == pytest.approx(token_output)
    assert answer.cost_total is None
    assert all(line.operation != operation for line in answer.cost_breakdown)


@pytest.mark.unit
def test_registered_profile_zero_page_count_adds_no_page_charge(registered_profile):
    model = _metadata(registered_profile)
    operation, meter = _page_meter(registered_profile)
    adapter = _adapter(registered_profile)
    transport = FakeSyncTransport(response(extra={"x_tools": {"pdf_page_parser": {"count": 0}}}))
    adapter.adapter._sync_transport = transport
    answer = adapter.chat([{"role": "user", "content": "Read",
        "files": [DocumentPart(data=b"%PDF data", media_type="application/pdf")] }],
        max_tokens=8, workspace_id=WORKSPACE)
    token_input, token_output = _token_costs(model, 3, 0, 2)
    assert answer.cost_input == pytest.approx(token_input)
    assert answer.cost_output == pytest.approx(token_output)
    assert answer.cost_total == pytest.approx(token_input + token_output)
    assert all(line.operation != operation or line.quantity == 0 for line in answer.cost_breakdown)
    assert meter.rate >= 0


@pytest.mark.unit
def test_registered_profile_missing_exact_page_meter_withholds_total(registered_profile):
    model = _metadata(registered_profile)
    operation, _meter = _page_meter(registered_profile)
    from llm_api_adapter_qwen import registry as qwen_registry
    from llm_api_adapter_qwen import plugin as plugin_module

    original_registry_metadata = qwen_registry.MODEL_METADATA
    original_plugin = plugin_module.PLUGIN
    original_plugin_metadata = original_plugin.model_metadata
    assert original_plugin_metadata is not None
    original_registry_catalogue = original_registry_metadata.organization_data
    original_plugin_catalogue = original_plugin_metadata.organization_data
    original_registry_catalogue_snapshot = copy.deepcopy(original_registry_catalogue)
    original_plugin_catalogue_snapshot = copy.deepcopy(original_plugin_catalogue)
    original_meter = copy.deepcopy(
        registered_profile.catalogue["metered_operations"][operation]
    )
    catalogue = copy.deepcopy(registered_profile.catalogue)
    removed_catalogue_meter = catalogue["metered_operations"].pop(operation, None)
    assert removed_catalogue_meter is not None

    try:
        with _registered_catalogue_variant(catalogue) as fault_profile:
            fault_model = _metadata(fault_profile)
            assert operation not in fault_profile.catalogue["metered_operations"]
            adapter = _adapter(fault_profile)
            adapter.adapter._sync_transport = FakeSyncTransport(response(
                extra={"x_tools": {"pdf_page_parser": {"count": 2}}}
            ))
            answer = adapter.chat([{"role": "user", "content": "Read", "files": [
                DocumentPart(url="https://example.test/a.pdf")]}], max_tokens=8,
                workspace_id=WORKSPACE)
            token_input, token_output = _token_costs(fault_model, 3, 0, 2)
            assert answer.cost_input == pytest.approx(token_input)
            assert answer.cost_output == pytest.approx(token_output)
            assert answer.cost_total is None
    finally:
        assert qwen_registry.MODEL_METADATA is original_registry_metadata
        assert plugin_module.PLUGIN is original_plugin
        assert original_plugin.model_metadata is original_plugin_metadata
        assert original_registry_metadata.organization_data is original_registry_catalogue
        assert original_plugin_metadata.organization_data is original_plugin_catalogue
        assert original_registry_catalogue == original_registry_catalogue_snapshot
        assert original_plugin_catalogue == original_plugin_catalogue_snapshot
        assert registered_profile.catalogue["metered_operations"][operation] == original_meter


@pytest.mark.unit
def test_registered_profile_token_page_currency_mismatch_withholds_total(registered_profile):
    model = _metadata(registered_profile)
    operation, meter = _page_meter(registered_profile)
    adapter = _adapter(registered_profile)
    adapter.adapter.pricing = replace(model.pricing_tiers, currency="CALLER-LOCAL")
    adapter.adapter._sync_transport = FakeSyncTransport(response(
        extra={"x_tools": {"pdf_page_parser": {"count": 2}}}
    ))
    answer = adapter.chat([{"role": "user", "content": "Read", "files": [
        DocumentPart(url="https://example.test/a.pdf")]}], max_tokens=8, workspace_id=WORKSPACE)
    token_input, token_output = _token_costs(model, 3, 0, 2)
    page_item = next(item for item in answer.cost_breakdown if item.operation == operation)
    assert answer.cost_input == pytest.approx(token_input)
    assert answer.cost_output == pytest.approx(token_output)
    assert page_item.currency == meter.currency
    assert answer.currency == "CALLER-LOCAL"
    assert answer.cost_total is None


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("streaming", [False, True])
async def test_registered_profile_runs_true_async_native_client(registered_profile, monkeypatch, streaming):
    _metadata(registered_profile)
    from llm_api_adapter_qwen.clients import async_client
    adapter = _adapter(registered_profile)
    calls = []
    async def fake_request(url, **kwargs):
        calls.append((url, kwargs))
        return response()
    def fake_stream(url, **kwargs):
        calls.append((url, kwargs))
        async def events():
            for event in simple_events():
                yield event
        return events()
    monkeypatch.setattr(async_client, "async_request", fake_request)
    monkeypatch.setattr(async_client, "async_stream_request", fake_stream)
    if streaming:
        assert "".join([part async for part in adapter.astream_chat(
            [{"role": "user", "content": "Hello"}], max_tokens=5, workspace_id=WORKSPACE)]) == "ok"
    else:
        result = await adapter.achat([{"role": "user", "content": "Hello"}], max_tokens=5,
                                     workspace_id=WORKSPACE)
        assert result.content == "ok"
    assert len(calls) == 1
    assert calls[0][0] == URL
    assert calls[0][1]["headers"] == {"x-api-key": "qwen-test-key", "Content-Type": "application/json"}


@pytest.mark.unit
@pytest.mark.parametrize("bad", [0, -1, True, "4", None])
@pytest.mark.parametrize("method", ["chat", "stream_chat"])
def test_current_qwen38_invalid_limits_reject_before_sync_dispatch(
    registered_profile, bad, method
):
    model = "qwen3.8-max"
    adapter = UniversalLLMAPIAdapter(organization="qwen", model=model, api_key="qwen-test-key")
    transport = FakeSyncTransport(events=simple_events(model))
    adapter.adapter._sync_transport = transport
    with pytest.raises(ValueError, match="max_tokens"):
        result = getattr(adapter, method)([{"role": "user", "content": "Hello"}],
            max_tokens=bad, workspace_id=WORKSPACE)
        if method == "stream_chat":
            list(result)
    assert transport.requests == [] and transport.stream_requests == []


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("bad", [0, -1, True, "4", None])
@pytest.mark.parametrize("method", ["achat", "astream_chat"])
async def test_current_qwen38_invalid_limits_reject_at_async_execution(
    registered_profile, monkeypatch, bad, method
):
    from llm_api_adapter_qwen.clients import async_client
    adapter = UniversalLLMAPIAdapter(organization="qwen", model="qwen3.8-max",
                                     api_key="qwen-test-key")
    calls = []
    async def fake_request(url, **kwargs):
        calls.append((url, kwargs))
        return response(model="qwen3.8-max")
    def fake_stream(url, **kwargs):
        calls.append((url, kwargs))
        async def events():
            for event in simple_events("qwen3.8-max"):
                yield event
        return events()
    monkeypatch.setattr(async_client, "async_request", fake_request)
    monkeypatch.setattr(async_client, "async_stream_request", fake_stream)
    with pytest.raises(ValueError, match="max_tokens"):
        if method == "achat":
            await adapter.achat([{"role": "user", "content": "Hello"}],
                                max_tokens=bad, workspace_id=WORKSPACE)
        else:
            stream = adapter.astream_chat([{"role": "user", "content": "Hello"}],
                                          max_tokens=bad, workspace_id=WORKSPACE)
            await anext(stream)
    assert calls == []


@pytest.mark.unit
@pytest.mark.parametrize("method", ["chat", "stream_chat"])
def test_current_qwen38_parallel_controls_reject_before_sync_dispatch(registered_profile, method):
    adapter = UniversalLLMAPIAdapter(organization="qwen", model="qwen3.8-max",
                                     api_key="qwen-test-key")
    transport = FakeSyncTransport(events=simple_events("qwen3.8-max"))
    adapter.adapter._sync_transport = transport
    with pytest.raises(NotImplementedError, match="parallel tool-call controls"):
        result = getattr(adapter, method)([{"role": "user", "content": "Hello"}],
            max_tokens=8, parallel_tool_calls=True, workspace_id=WORKSPACE)
        if method == "stream_chat":
            list(result)
    assert transport.requests == [] and transport.stream_requests == []


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("method", ["achat", "astream_chat"])
async def test_current_qwen38_parallel_controls_reject_at_async_execution(
    registered_profile, monkeypatch, method
):
    from llm_api_adapter_qwen.clients import async_client
    adapter = UniversalLLMAPIAdapter(organization="qwen", model="qwen3.8-max",
                                     api_key="qwen-test-key")
    calls = []
    async def fake_request(url, **kwargs):
        calls.append((url, kwargs))
        return response(model="qwen3.8-max")
    def fake_stream(url, **kwargs):
        calls.append((url, kwargs))
        async def events():
            for event in simple_events("qwen3.8-max"):
                yield event
        return events()
    monkeypatch.setattr(async_client, "async_request", fake_request)
    monkeypatch.setattr(async_client, "async_stream_request", fake_stream)
    with pytest.raises(NotImplementedError, match="parallel tool-call controls"):
        if method == "achat":
            await adapter.achat([{"role": "user", "content": "Hello"}], max_tokens=8,
                                parallel_tool_calls=True, workspace_id=WORKSPACE)
        else:
            stream = adapter.astream_chat([{"role": "user", "content": "Hello"}],
                max_tokens=8, parallel_tool_calls=True, workspace_id=WORKSPACE)
            await anext(stream)
    assert calls == []


@pytest.mark.unit
@pytest.mark.parametrize("schema", [SCHEMA])
def test_registered_profile_preserves_structured_output_without_prompt_injection(registered_profile, schema):
    _metadata(registered_profile)
    adapter = _adapter(registered_profile)
    transport = FakeSyncTransport(response(content=[{"type": "text", "text": '{"answer":"ok"}'}]))
    adapter.adapter._sync_transport = transport
    result = adapter.chat([{"role": "user", "content": "Answer"}], max_tokens=8,
                          json_schema=schema, workspace_id=WORKSPACE)
    assert result.parsed_json == {"answer": "ok"}
    payload = transport.requests[0].payload
    assert payload["output_config"]["format"] == {"type": "json_schema", "schema": schema}
    assert payload["messages"] == [{"role": "user", "content": "Answer"}]


@pytest.mark.unit
def test_registered_profile_preserves_tool_result_history_ids_and_arguments(registered_profile):
    _metadata(registered_profile)
    adapter = _adapter(registered_profile)
    transport = FakeSyncTransport(response())
    adapter.adapter._sync_transport = transport
    adapter.chat([
        {"role": "user", "content": "What is the weather?"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "name": "get_weather", "arguments": {"city": "Oslo", "date": "today"},
            "call_id": "toolu_history_q4"}]},
        {"role": "tool", "tool_call_id": "toolu_history_q4", "content": "snow"},
    ], max_tokens=8, tools=[ToolSpec(name="get_weather", json_schema={"type": "object"})],
       workspace_id=WORKSPACE)
    assert transport.requests[0].payload["messages"][1]["content"] == [{
        "type": "tool_use", "id": "toolu_history_q4", "name": "get_weather",
        "input": {"city": "Oslo", "date": "today"}}]
    assert transport.requests[0].payload["messages"][2]["content"] == [{
        "type": "tool_result", "tool_use_id": "toolu_history_q4", "content": "snow"}]


@pytest.mark.unit
def test_registered_profile_pydantic_output_parses_without_prompt_injection(registered_profile):
    _metadata(registered_profile)
    adapter = _adapter(registered_profile)
    transport = FakeSyncTransport(response(content=[{"type": "text", "text": '{"answer":"ok"}'}]))
    adapter.adapter._sync_transport = transport
    result = adapter.chat([{"role": "user", "content": "Answer"}], max_tokens=8,
                          response_model=StructuredAnswer, workspace_id=WORKSPACE)
    assert result.parsed_json == {"answer": "ok"}
    assert transport.requests[0].payload["output_config"]["format"]["schema"] == StructuredAnswer.model_json_schema()
    assert result.parsed_model == StructuredAnswer(answer="ok")


@pytest.mark.unit
@pytest.mark.parametrize(("stop_reason", "stop_details", "attribute", "expected"), [
    ("refusal", {"reason": "policy"}, "refusal", "policy"),
    ("max_tokens", None, "incomplete_reason", "max_tokens"),
])
def test_registered_profile_structured_refusal_and_incomplete_are_not_parsed(
    registered_profile, stop_reason, stop_details, attribute, expected
):
    _metadata(registered_profile)
    adapter = _adapter(registered_profile)
    extra = {"stop_reason": stop_reason}
    if stop_details is not None:
        extra["stop_details"] = stop_details
    transport = FakeSyncTransport(response(content=[{"type": "text", "text": '{"answer":"partial"}'}],
                                           extra=extra))
    adapter.adapter._sync_transport = transport
    result = adapter.chat([{"role": "user", "content": "Answer"}], max_tokens=8,
                          json_schema=SCHEMA, workspace_id=WORKSPACE)
    assert getattr(result, attribute) == expected
    assert result.parsed_json is None
    assert result.parsed_model is None


@pytest.mark.unit
def test_registered_profile_stream_reasoning_capture_is_opt_in_with_callback(registered_profile):
    _metadata(registered_profile)
    adapter = _adapter(registered_profile)
    events = [
        SSEEvent(event="message_start", data={"type": "message_start", "message": {
            "id": "msg-q4-thinking", "type": "message", "role": "assistant", "model": MODEL,
            "content": [], "usage": {"input_tokens": 3, "output_tokens": 0}}}),
        SSEEvent(event="content_block_start", data={"type": "content_block_start", "index": 0,
            "content_block": {"type": "thinking", "thinking": ""}}),
        SSEEvent(event="content_block_delta", data={"type": "content_block_delta", "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "plan"}}),
        SSEEvent(event="content_block_start", data={"type": "content_block_start", "index": 1,
            "content_block": {"type": "text", "text": ""}}),
        SSEEvent(event="content_block_delta", data={"type": "content_block_delta", "index": 1,
            "delta": {"type": "text_delta", "text": "visible"}}),
        SSEEvent(event="message_delta", data={"type": "message_delta", "delta": {"stop_reason": "end_turn"},
            "usage": {"input_tokens": 3, "output_tokens": 2}}),
        SSEEvent(event="message_stop", data={"type": "message_stop"}),
    ]
    transport = FakeSyncTransport(events=events)
    adapter.adapter._sync_transport = transport
    callbacks = []
    completed = []
    visible = "".join(adapter.stream_chat([{"role": "user", "content": "Hello"}],
        max_tokens=8, workspace_id=WORKSPACE, capture_reasoning=True,
        on_reasoning=callbacks.append, on_done=completed.append))
    assert visible == "visible"
    assert [item.text for item in callbacks] == ["plan"]
    assert completed[0].content == "visible"
    assert [item.text for item in completed[0].reasoning_events] == ["plan"]


@pytest.mark.unit
def test_registered_profile_reasoning_capture_is_opt_in_and_separate(registered_profile):
    _metadata(registered_profile)
    adapter = _adapter(registered_profile)
    body = response(content=[{"type": "thinking", "thinking": "private plan"},
                             {"type": "text", "text": "visible"}])
    transport = FakeSyncTransport(body)
    adapter.adapter._sync_transport = transport
    hidden = adapter.chat([{"role": "user", "content": "Hello"}], max_tokens=8, workspace_id=WORKSPACE)
    transport.body = body
    captured = adapter.chat([{"role": "user", "content": "Hello"}], max_tokens=8,
                            workspace_id=WORKSPACE, capture_reasoning=True)
    assert hidden.content == "visible" and hidden.reasoning_events == []
    assert captured.content == "visible"
    assert [event.text for event in captured.reasoning_events] == ["private plan"]


@pytest.mark.unit
@pytest.mark.parametrize("model", ["qwen3.7-plus", "qwen3.7-flash"])
def test_registered_profile_old_models_keep_forced_tool_behavior(model, registered_profile):
    assert model in _core_registry(registered_profile).organizations["qwen"].models
    adapter = UniversalLLMAPIAdapter(organization="qwen", model=model, api_key="qwen-test-key")
    transport = FakeSyncTransport(response(model=model))
    adapter.adapter._sync_transport = transport
    with pytest.warns(UserWarning, match="disabled thinking because forced tool_choice"):
        adapter.chat([{"role": "user", "content": "Weather"}], max_tokens=8,
            reasoning_level="low", tools=[ToolSpec(name="get_weather", json_schema={"type": "object"})],
            tool_choice="any", workspace_id=WORKSPACE)
    assert transport.requests[0].payload["thinking"] == {"type": "disabled"}


@pytest.mark.unit
@pytest.mark.parametrize("model_name", [
    "qwen3.8-max", "qwen3.8-flash", "qwen3.7-plus", "qwen3.7-flash",
])
def test_old_four_profiles_keep_metadata_owned_cny_cache_pricing(registered_profile, model_name):
    core = _core_registry(registered_profile)
    assert model_name in core.organizations["qwen"].models
    model = core.organizations["qwen"].models[model_name]
    adapter = UniversalLLMAPIAdapter(organization="qwen", model=model_name,
                                     api_key="qwen-test-key")
    usage = {"input_tokens": 10, "cache_read_input_tokens": 2,
             "cache_creation_input_tokens": 0, "output_tokens": 4}
    transport = FakeSyncTransport(response(model=model_name, usage=usage))
    adapter.adapter._sync_transport = transport
    answer = adapter.chat([{"role": "user", "content": "Hello"}], max_tokens=8,
                          workspace_id=WORKSPACE)
    expected_input, expected_output = _token_costs(model, 10, 2, 4)
    assert answer.usage.input_tokens == 12
    assert answer.cost_input == pytest.approx(expected_input)
    assert answer.cost_output == pytest.approx(expected_output)
    assert answer.cost_total == pytest.approx(expected_input + expected_output)
    assert answer.currency == registered_profile.catalogue["currency"]
    for tier in model.pricing_tiers.tiers:
        if tier.up_to_prompt_tokens is None:
            continue
        for prompt_tokens in (tier.up_to_prompt_tokens, tier.up_to_prompt_tokens + 1):
            tier_response = FakeSyncTransport(response(model=model_name, usage={
                    "input_tokens": prompt_tokens, "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "output_tokens": 4,
            }))
            adapter.adapter._sync_transport = tier_response
            tier_answer = adapter.chat([{"role": "user", "content": "Hello"}],
                max_tokens=8, workspace_id=WORKSPACE)
            tier_input, tier_output = _token_costs(model, prompt_tokens, 0, 4)
            assert tier_answer.cost_input == pytest.approx(tier_input)
            assert tier_answer.cost_output == pytest.approx(tier_output)
            assert tier_answer.currency == registered_profile.catalogue["currency"]


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("model_name", ["qwen3.7-plus", "qwen3.7-flash", "qwen-unknown-pdf-test"])
@pytest.mark.parametrize("document", [
    DocumentPart(url="https://example.test/legacy.pdf"),
    DocumentPart(data=b"%PDF legacy", media_type="application/pdf"),
])
@pytest.mark.parametrize("method", ["chat", "stream_chat", "achat", "astream_chat"])
async def test_legacy_and_unknown_profiles_reject_pdfs_before_io(
    registered_profile, monkeypatch, model_name, document, method
):
    core = _core_registry(registered_profile)
    if model_name != "qwen-unknown-pdf-test":
        assert model_name in core.organizations["qwen"].models
        adapter = UniversalLLMAPIAdapter(organization="qwen", model=model_name,
                                         api_key="qwen-test-key")
    else:
        with pytest.warns(UserWarning, match="is not verified"):
            adapter = UniversalLLMAPIAdapter(organization="qwen", model=model_name,
                                             api_key="qwen-test-key")
    messages = [{"role": "user", "content": "Read", "files": [document]}]
    if method in {"chat", "stream_chat"}:
        transport = FakeSyncTransport(events=simple_events(model_name))
        adapter.adapter._sync_transport = transport
        with pytest.raises(ValueError, match="DocumentPart; PDF and OCR are unavailable"):
            result = getattr(adapter, method)(messages, max_tokens=8, workspace_id=WORKSPACE)
            if method == "stream_chat":
                list(result)
        assert transport.requests == [] and transport.stream_requests == []
    else:
        from llm_api_adapter_qwen.clients import async_client
        calls = []
        async def fake_request(url, **kwargs):
            calls.append((url, kwargs))
            return response(model=model_name)
        def fake_stream(url, **kwargs):
            calls.append((url, kwargs))
            async def events():
                for event in simple_events(model_name):
                    yield event
            return events()
        monkeypatch.setattr(async_client, "async_request", fake_request)
        monkeypatch.setattr(async_client, "async_stream_request", fake_stream)
        with pytest.raises(ValueError, match="DocumentPart; PDF and OCR are unavailable"):
            if method == "achat":
                await adapter.achat(messages, max_tokens=8, workspace_id=WORKSPACE)
            else:
                stream = adapter.astream_chat(messages, max_tokens=8, workspace_id=WORKSPACE)
                await anext(stream)
        assert calls == []


