"""Pre-migration semantic contract for native Mistral execution.

Wire fixtures here intentionally model the current Chat Completions protocol;
the assertions below describe application-visible meaning and request intent.
"""

from __future__ import annotations

import asyncio
import json
import socket
import sys
import warnings
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[2]
for source in (str(PACKAGE_ROOT / "src"), str(REPOSITORY_ROOT / "src"), str(REPOSITORY_ROOT)):
    if source not in sys.path:
        sys.path.insert(0, source)

import llm_api_adapter.adapters.base_adapter as base_adapter_module
import llm_api_adapter.universal_adapter as universal_module
import llm_api_adapter_mistral.clients.async_client as mistral_async_client_module
from llm_api_adapter.errors import LLMReasoningLevelError
from llm_api_adapter.errors.llm_api_error import (
    JSONSchemaError,
    LLMAPIClientError,
    ToolChoiceError,
)
from llm_api_adapter.llm_registry.llm_registry import RegistrySpec
from llm_api_adapter.llm_registry.llm_registry import resolve_metered_operation_spec
from llm_api_adapter.llm_registry.reasoning import resolve_reasoning_level
from llm_api_adapter.llms.httpx_transport import HttpxSyncTransport
from llm_api_adapter.llms.requests_transport import RequestsSyncTransport
from llm_api_adapter.llms.transports import JSONResponse, SSEEvent
from llm_api_adapter.models.messages.chat_message import UserMessage
from llm_api_adapter.models.messages.file_parts import DocumentPart, ImagePart
from llm_api_adapter.models.responses.chat_response import ChatResponse
from llm_api_adapter.models.tools import ToolSpec
from llm_api_adapter.service_provider_registry import ServiceProviderRegistry
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter
from llm_api_adapter_mistral.plugin import PLUGIN


CHAT_URL = "https://api.mistral.ai/v1/chat/completions"
OCR_URL = "https://api.mistral.ai/v1/ocr"
MODEL_IDS = tuple(PLUGIN.model_metadata.organization_data["models"])
MIXED_HISTORY = [
    {"role": "system", "content": "Use concise answers."},
    {"role": "user", "content": "Find the report."},
    {"role": "assistant", "content": "I will search."},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call-synthetic",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"key":"report"}'},
        }],
    },
    {"role": "tool", "tool_call_id": "call-synthetic", "content": "found"},
    UserMessage("Summarize it."),
]
LOOKUP = ToolSpec(name="lookup", description="Lookup", json_schema={"type": "object"})
EXECUTION_PATHS = (
    ("chat", "requests"),
    ("chat", "httpx"),
    ("stream_chat", "requests"),
    ("stream_chat", "httpx"),
    ("achat", None),
    ("astream_chat", None),
)


@pytest.fixture
def mistral_runtime(monkeypatch):
    registry = RegistrySpec()
    assert PLUGIN.model_metadata is not None
    assert registry.register_organization_metadata(PLUGIN.model_metadata)
    providers = ServiceProviderRegistry()
    PLUGIN.register(providers)
    monkeypatch.setattr(universal_module, "LLM_REGISTRY", registry)
    monkeypatch.setattr(universal_module, "SERVICE_PROVIDER_REGISTRY", providers)
    monkeypatch.setattr(base_adapter_module, "LLM_REGISTRY", registry)
    return registry


@pytest.fixture(autouse=True)
def block_unmocked_network(monkeypatch):
    def block_dns(*args, **kwargs):
        raise AssertionError("characterization tests must not perform network I/O")

    original_connect = socket.socket.connect

    def block_external_connect(sock, address):
        # ProactorEventLoop's Windows socketpair fallback connects to loopback
        # while creating its self-pipe; permit that local runtime operation.
        if isinstance(address, tuple) and str(address[0]) in {"127.0.0.1", "::1", "localhost"}:
            return original_connect(sock, address)
        raise AssertionError("characterization tests must not connect externally")

    monkeypatch.setattr(socket, "getaddrinfo", block_dns)
    monkeypatch.setattr(socket.socket, "connect", block_external_connect)


def _response(model, *, content="synthetic answer", finish="stop", usage=None, extra=None):
    message = {"role": "assistant", "content": content}
    if extra:
        message.update(extra)
    result = {"id": "synthetic-response-id", "model": model, "choices": [
        {"message": message, "finish_reason": finish}
    ]}
    if usage is not None:
        result["usage"] = usage
    return result


def _terminal_event(model, *, content="synthetic answer", finish="stop", usage=None):
    data = {
        "id": "synthetic-stream-id", "model": model,
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": finish}],
    }
    if usage is not None:
        data["usage"] = usage
    return SSEEvent(event=None, data=data)


def _events_for_response(response, *, response_id="synthetic-stream-id"):
    choice = response["choices"][0]
    message = choice["message"]
    delta = {key: message[key] for key in ("content", "tool_calls") if key in message}
    data = {
        "id": response_id,
        "model": response["model"],
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": choice.get("finish_reason"),
        }],
    }
    if "usage" in response:
        data["usage"] = response["usage"]
    return [SSEEvent(event=None, data=data)]


class RecordingSyncTransport:
    def __init__(self, *, chat_result=None, events=(), ocr_results=(), timeline=None):
        self.chat_result = chat_result or {}
        self.events = list(events)
        self.ocr_results = list(ocr_results)
        self.requests = []
        self.stream_requests = []
        self.ocr_requests = []
        self.timeline = timeline if timeline is not None else []

    def post_json(self, request, *, http_error_handler=None):
        if request.url == OCR_URL:
            self.ocr_requests.append(request)
            self.timeline.append(("ocr", request))
            return JSONResponse(self.ocr_results.pop(0) if self.ocr_results else {
                "pages": [{"markdown": "synthetic extracted text"}],
                "usage_info": {"pages_processed": 2},
            })
        self.requests.append(request)
        self.timeline.append(("model", request))
        return JSONResponse(self.chat_result)

    def post_sse(self, request, *, http_error_handler=None, stream_error_handler=None):
        if request.url == OCR_URL:
            self.ocr_requests.append(request)
            raise AssertionError("OCR is submitted as JSON, not a stream")
        self.stream_requests.append(request)
        self.timeline.append(("model", request))
        return iter(self.events)


def _new_adapter(model, transport="requests", *, developer=None):
    if developer is None:
        developer = "mistral"
    return UniversalLLMAPIAdapter(
        organization=developer,
        service_provider="mistral",
        model=model,
        api_key="synthetic-test-key",
        transport=transport or "requests",
    )


def _install_async(monkeypatch, *, model_response, stream_events=(), ocr_results=(), timeline=None):
    calls = {"chat": [], "stream": [], "ocr": []}
    ocr_values = list(ocr_results)

    async def async_request(url, **kwargs):
        bucket = "ocr" if url == OCR_URL else "chat"
        calls[bucket].append((url, kwargs))
        if timeline is not None:
            timeline.append((bucket, kwargs))
        if bucket == "ocr":
            return ocr_values.pop(0) if ocr_values else {
                "pages": [{"markdown": "synthetic extracted text"}],
                "usage_info": {"pages_processed": 2},
            }
        return model_response

    def async_stream_request(url, **kwargs):
        calls["stream"].append((url, kwargs))
        if timeline is not None:
            timeline.append(("stream", kwargs))

        async def source():
            for event in stream_events:
                yield event

        return source()

    monkeypatch.setattr(mistral_async_client_module, "async_request", async_request)
    monkeypatch.setattr(mistral_async_client_module, "async_stream_request", async_stream_request)
    return calls


def _invoke(adapter, method, messages, **kwargs):
    if method == "chat":
        return adapter.chat(messages, **kwargs)
    if method == "stream_chat":
        completed = []
        callback = kwargs.pop("on_done", None)
        def done(response):
            completed.append(response)
            if callback is not None:
                callback(response)
        visible = list(adapter.stream_chat(messages, on_done=done, **kwargs))
        return visible, completed[0]
    if method == "achat":
        return asyncio.run(adapter.achat(messages, **kwargs))
    async def collect():
        completed = []
        callback = kwargs.pop("on_done", None)
        def done(response):
            completed.append(response)
            if callback is not None:
                callback(response)
        visible = [item async for item in adapter.astream_chat(
            messages, on_done=done, **kwargs
        )]
        return visible, completed[0]
    return asyncio.run(collect())


def _execute(adapter, method, messages, monkeypatch, *, model_response, stream_events=(), timeout_s=None, ocr_results=(), timeline=None, on_done=None, **kwargs):
    recorder = RecordingSyncTransport(
        chat_result=model_response, events=stream_events, ocr_results=ocr_results,
        timeline=timeline,
    )
    adapter.adapter._sync_transport = recorder
    async_calls = _install_async(
        monkeypatch, model_response=model_response, stream_events=stream_events,
        ocr_results=ocr_results, timeline=timeline,
    )
    if timeout_s is not None:
        kwargs["timeout_s"] = timeout_s
    if on_done is not None:
        kwargs["on_done"] = on_done
    result = _invoke(adapter, method, messages, **kwargs)
    semantic = result[1] if method in {"stream_chat", "astream_chat"} else result
    visible = result[0] if method in {"stream_chat", "astream_chat"} else None
    return semantic, visible, recorder, async_calls


@pytest.mark.integration
@pytest.mark.parametrize("model", MODEL_IDS)
@pytest.mark.parametrize(("method", "transport"), EXECUTION_PATHS)
def test_every_registered_profile_preserves_four_native_methods_and_transport_meaning(
    mistral_runtime, monkeypatch, model, method, transport
):
    """Every exact profile uses its registered identity, request policy and native route."""
    metadata = mistral_runtime.organizations["mistral"]
    assert tuple(metadata.models) == MODEL_IDS  # inventory and order come from PLUGIN metadata
    spec = metadata.models[model]
    wire_model = spec.provider_model or model
    adapter = _new_adapter(model, transport, developer="mistral")
    assert adapter.organization == "mistral"
    assert adapter.service_provider == adapter.adapter.service_provider == "mistral"
    assert adapter.adapter.model_spec is spec
    assert adapter.adapter.model_spec.limits == spec.limits
    assert adapter.adapter.pricing.currency == spec.pricing_tiers.currency
    assert adapter.adapter.pricing.tiers == spec.pricing_tiers.tiers

    response = _response(wire_model)
    stream = [_terminal_event(wire_model)]
    recorder = RecordingSyncTransport(chat_result=response, events=stream)
    if method in {"chat", "stream_chat"}:
        expected_transport = RequestsSyncTransport if transport == "requests" else HttpxSyncTransport
        assert isinstance(adapter.adapter._sync_transport, expected_transport)
    adapter.adapter._sync_transport = recorder
    async_calls = _install_async(monkeypatch, model_response=response, stream_events=stream)

    if method in {"chat", "stream_chat"}:
        result = _invoke(adapter, method, MIXED_HISTORY, max_tokens=73, timeout_s=9.5)
        request = recorder.requests[0] if method == "chat" else recorder.stream_requests[0]
        assert not async_calls["chat"] and not async_calls["stream"]
        assert request.url == CHAT_URL
        assert request.timeout == 9.5
        headers = request.headers_dict()
        payload = request.payload
    else:
        result = _invoke(adapter, method, MIXED_HISTORY, max_tokens=73, timeout_s=9.5)
        assert recorder.requests == [] and recorder.stream_requests == []
        calls = async_calls["chat"] if method == "achat" else async_calls["stream"]
        assert len(calls) == 1
        url, kwargs = calls[0]
        assert url == CHAT_URL
        assert kwargs["timeout"] == 9.5
        headers = kwargs["headers"]
        payload = kwargs["payload"]
    assert headers["Authorization"] == "Bearer synthetic-test-key"
    assert headers["Content-Type"] == "application/json"
    assert recorder.ocr_requests == [] and async_calls["ocr"] == []

    semantic_response = result[1] if method in {"stream_chat", "astream_chat"} else result
    if method in {"stream_chat", "astream_chat"}:
        assert result[0] == ["synthetic answer"]
    assert semantic_response.content == "synthetic answer"
    assert semantic_response.finish_reason == "stop"
    assert semantic_response.response_id == (
        "synthetic-stream-id" if method in {"stream_chat", "astream_chat"}
        else "synthetic-response-id"
    )
    assert payload["model"] == wire_model
    assert payload["max_tokens"] == 73
    assert payload["temperature"] == 1.0 and payload["top_p"] == 1.0
    assert [item["role"] for item in payload["messages"]] == [
        "system", "user", "assistant", "assistant", "tool", "user"
    ]
    assert payload["messages"][0]["content"] == "Use concise answers."
    assert payload["messages"][1]["content"] == "Find the report."
    assert payload["messages"][2]["content"] == "I will search."
    assert payload["messages"][3]["tool_calls"][0]["id"] == "call-synthetic"
    assert json.loads(payload["messages"][3]["tool_calls"][0]["function"]["arguments"]) == {"key": "report"}
    assert payload["messages"][4] == {
        "role": "tool", "tool_call_id": "call-synthetic", "content": "found"
    }
    assert payload["messages"][5]["content"] == "Summarize it."
    capability = spec.reasoning_capability
    expected_default = capability.allowed_values[0] if capability and capability.allowed_values else None
    assert payload.get("reasoning_effort") == expected_default


@pytest.mark.integration
def test_hosted_developers_retain_mistral_service_provider_ownership(mistral_runtime):
    for developer, model in (("zai", "glm-5.3-flash"), ("deepseek", "deepseek-flash")):
        spec = mistral_runtime.organizations["mistral"].models[model]
        adapter = _new_adapter(model, developer=developer)
        assert adapter.organization == developer
        assert adapter.service_provider == adapter.adapter.service_provider == "mistral"
        assert adapter.adapter.model_spec is spec
        recorder = RecordingSyncTransport(
            chat_result=_response(spec.provider_model or spec.name)
        )
        adapter.adapter._sync_transport = recorder
        adapter.chat([{"role": "user", "content": "synthetic hosted request"}])
        assert recorder.requests[0].payload["model"] == spec.provider_model


@pytest.mark.integration
@pytest.mark.parametrize("method", ("chat", "stream_chat", "achat", "astream_chat"))
def test_previous_response_is_ignored_and_caller_history_is_sent(mistral_runtime, monkeypatch, method):
    model = MODEL_IDS[0]
    adapter = _new_adapter(model)
    response = _response(model)
    events = [_terminal_event(model)]
    recorder = RecordingSyncTransport(chat_result=response, events=events)
    adapter.adapter._sync_transport = recorder
    calls = _install_async(monkeypatch, model_response=response, stream_events=events)
    prior = ChatResponse(content="synthetic earlier result", response_id="opaque-synthetic-id")
    _invoke(adapter, method, MIXED_HISTORY, previous_response=prior)
    payload = (recorder.requests[0] if method == "chat" else recorder.stream_requests[0]).payload if method in {"chat", "stream_chat"} else (calls["chat"] if method == "achat" else calls["stream"])[0][1]["payload"]
    assert payload["messages"][0] == {"role": "system", "content": "Use concise answers."}
    assert payload["messages"][4]["content"] == "found"
    assert "opaque-synthetic-id" not in repr(payload)
    assert not any("previous" in key.lower() and "response" in key.lower() for key in payload)


@pytest.mark.integration
@pytest.mark.parametrize("finish", ("stop", "length", "content_filter"))
def test_native_response_keeps_original_finish_reason_and_terminal_fields(mistral_runtime, finish):
    model = MODEL_IDS[0]
    adapter = _new_adapter(model)
    recorder = RecordingSyncTransport(chat_result=_response(
        model, content="synthetic terminal", finish=finish
    ))
    adapter.adapter._sync_transport = recorder
    response = adapter.chat([{"role": "user", "content": "synthetic request"}])
    assert response.finish_reason == finish
    assert response.response_id == "synthetic-response-id"
    assert response.content == "synthetic terminal"
    assert response.incomplete_reason == ("length" if finish == "length" else None)


@pytest.mark.integration
@pytest.mark.parametrize(("method", "transport"), EXECUTION_PATHS)
def test_native_structured_incomplete_and_tool_outcomes_remain_distinct(
    mistral_runtime, monkeypatch, method, transport
):
    model = MODEL_IDS[0]
    adapter = _new_adapter(model, transport)
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    raw = _response(model, content='{"answer":"ready"}')
    parsed, visible, recorder, calls = _execute(
        adapter, method, [{"role": "user", "content": "structured"}], monkeypatch,
        model_response=raw, stream_events=_events_for_response(raw), json_schema=schema,
    )
    assert parsed.parsed_json == {"answer": "ready"}
    assert parsed.parsed_model is None
    if visible is not None:
        assert visible == ['{"answer":"ready"}']

    from pydantic import BaseModel

    class AnswerModel(BaseModel):
        answer: str
        model_config = {"extra": "forbid"}

    raw = _response(model, content='{"answer":"typed"}')
    typed, _, recorder, calls = _execute(
        adapter, method, [{"role": "user", "content": "typed structured"}], monkeypatch,
        model_response=raw, stream_events=_events_for_response(raw), response_model=AnswerModel,
    )
    assert typed.parsed_json == {"answer": "typed"}
    assert typed.parsed_model == AnswerModel(answer="typed")

    raw = _response(model, content=None, finish="length")
    incomplete, visible, _, _ = _execute(
        adapter, method, [{"role": "user", "content": "short output"}], monkeypatch,
        model_response=raw, stream_events=_events_for_response(raw), json_schema=schema,
    )
    assert incomplete.finish_reason == "length" and incomplete.incomplete_reason == "length"
    assert incomplete.parsed_json is None and incomplete.parsed_model is None
    if visible is not None:
        assert visible == []

    raw = _response(model, content=None, finish="tool_calls", extra={
        "tool_calls": [{
            "id": "call-result",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"key":"value"}'},
        }],
    })
    tool_result, _, recorder, calls = _execute(
        adapter, method, [{"role": "user", "content": "lookup"}], monkeypatch,
        model_response=raw, stream_events=_events_for_response(raw),
        tools=[LOOKUP], tool_choice="lookup",
    )
    assert tool_result.tool_calls[0].name == "lookup"
    assert tool_result.tool_calls[0].arguments == {"key": "value"}
    assert tool_result.tool_calls[0].call_id == "call-result"
    if method == "chat":
        tool_payload = recorder.requests[-1].payload
    elif method == "stream_chat":
        tool_payload = recorder.stream_requests[-1].payload
    else:
        bucket = "chat" if method == "achat" else "stream"
        tool_payload = calls[bucket][-1][1]["payload"]
    assert tool_payload["tool_choice"] == {
        "type": "function", "function": {"name": "lookup"}
    }


@pytest.mark.integration
@pytest.mark.parametrize("method", ("chat", "achat"))
def test_native_refusal_keeps_refusal_and_leaves_structured_output_unavailable(
    mistral_runtime, monkeypatch, method
):
    model = MODEL_IDS[0]
    adapter = _new_adapter(model)
    schema = {
        "type": "object", "properties": {"answer": {"type": "string"}},
        "required": ["answer"], "additionalProperties": False,
    }
    raw = _response(model, content=None, finish="content_filter", extra={
        "refusal": "synthetic refusal"
    })
    refused, _, _, _ = _execute(
        adapter, method, [{"role": "user", "content": "decline"}], monkeypatch,
        model_response=raw, json_schema=schema,
    )
    assert refused.finish_reason == "content_filter"
    assert refused.refusal == "synthetic refusal"
    assert refused.parsed_json is None and refused.parsed_model is None


@pytest.mark.integration
@pytest.mark.parametrize(("method", "transport"), EXECUTION_PATHS)
def test_usage_and_cost_keep_reported_zero_missing_partial_and_cached_components(
    mistral_runtime, monkeypatch, method, transport
):
    model = MODEL_IDS[0]
    adapter = _new_adapter(model, transport)
    spec = mistral_runtime.organizations["mistral"].models[model]
    tier = spec.pricing_tiers.tier_for_prompt_tokens(1000)

    def run_usage(usage):
        raw = _response(model, usage=usage)
        return _execute(
            adapter, method, [{"role": "user", "content": "cost"}], monkeypatch,
            model_response=raw, stream_events=_events_for_response(raw),
        )[0]

    complete = {
        "prompt_tokens": 1000,
        "completion_tokens": 500,
        "total_tokens": 1500,
        "prompt_tokens_details": {"cached_tokens": 200},
    }
    cached = run_usage(complete)
    assert cached.usage.input_tokens == 1000 and cached.usage.cached_tokens == 200
    expected_input = 800 * tier.in_per_token + 200 * tier.cache_read_in_per_token
    assert cached.cost_input == pytest.approx(expected_input)
    assert cached.cost_output == pytest.approx(500 * tier.out_per_token)
    assert cached.cost_total == pytest.approx(cached.cost_input + cached.cost_output)
    assert cached.currency == spec.pricing_tiers.currency

    adapter.adapter.pricing.set_in_per_1m(2.0)
    caller_priced = run_usage(complete)
    assert caller_priced.cost_input == pytest.approx(
        800 * 2.0 / 1_000_000 + 200 * tier.cache_read_in_per_token
    )
    assert tier.in_per_token != 2.0 / 1_000_000

    zero = run_usage({
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
        "prompt_tokens_details": {"cached_tokens": 0},
    })
    assert zero.usage.input_tokens == zero.usage.output_tokens == zero.usage.total_tokens == 0
    assert zero.cost_input == zero.cost_output == zero.cost_total == 0

    partial = run_usage({"prompt_tokens": 1000})
    assert partial.usage.input_tokens == 1000
    assert partial.usage.output_tokens is None and partial.usage.total_tokens is None
    assert partial.cost_input is None and partial.cost_output is None and partial.cost_total is None

    missing = run_usage(None)
    assert missing.usage is None
    assert missing.cost_input is None and missing.cost_output is None and missing.cost_total is None

    derived = run_usage({"prompt_tokens": 4, "completion_tokens": 2})
    assert derived.usage.input_tokens == 4 and derived.usage.output_tokens == 2
    assert derived.usage.total_tokens == 6
    malformed_total = run_usage({
        "prompt_tokens": 4, "completion_tokens": 2, "total_tokens": "malformed",
        "prompt_tokens_details": {"cached_tokens": 0},
    })
    assert malformed_total.usage.input_tokens == 4 and malformed_total.usage.output_tokens == 2
    assert malformed_total.usage.total_tokens == 6
    malformed_component = run_usage({
        "prompt_tokens": "malformed", "completion_tokens": 2,
        "prompt_tokens_details": {"cached_tokens": 0},
    })
    assert malformed_component.usage.input_tokens is None
    assert malformed_component.usage.output_tokens == 2
    assert malformed_component.usage.total_tokens is None
    assert malformed_component.cost_input is None
    assert malformed_component.cost_total is None
    contradictory = run_usage({
        "prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 999,
        "prompt_tokens_details": {"cached_tokens": 0},
    })
    assert contradictory.usage.input_tokens == 4 and contradictory.usage.output_tokens == 2
    assert contradictory.usage.total_tokens == 999
    assert contradictory.cost_input is not None and contradictory.cost_output is not None
    assert contradictory.cost_total == pytest.approx(
        contradictory.cost_input + contradictory.cost_output
    )


@pytest.mark.integration
@pytest.mark.parametrize(("method", "transport"), EXECUTION_PATHS)
def test_invalid_reported_cache_count_remains_a_client_error(
    mistral_runtime, monkeypatch, method, transport
):
    model = MODEL_IDS[0]
    adapter = _new_adapter(model, transport)
    raw = _response(model, usage={
        "prompt_tokens": 10,
        "completion_tokens": 1,
        "prompt_tokens_details": {"cached_tokens": "not-a-count"},
    })
    with pytest.raises(LLMAPIClientError, match="cached_tokens"):
        _execute(
            adapter, method, [{"role": "user", "content": "usage"}], monkeypatch,
            model_response=raw, stream_events=_events_for_response(raw),
        )


@pytest.mark.integration
def test_registry_reasoning_allowed_values_fallback_and_projection(mistral_runtime):
    profiles = mistral_runtime.organizations["mistral"].models.values()
    for spec in profiles:
        if spec.reasoning_capability is None:
            requested_values = ("unrecognized",)
        else:
            requested_values = tuple(dict.fromkeys((
                *spec.reasoning_capability.allowed_values,
                "none",
                "minimal",
                0,
                spec.limits.context_window_tokens,
            )))
        for requested in requested_values:
            expected = resolve_reasoning_level(spec, requested)
            adapter = _new_adapter(spec.name)
            recorder = RecordingSyncTransport(
                chat_result=_response(spec.provider_model or spec.name)
            )
            adapter.adapter._sync_transport = recorder
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                adapter.chat(
                    [{"role": "user", "content": "reasoning"}],
                    reasoning_level=requested,
                )
            assert [str(item.message) for item in caught] == (
                [expected.warning] if expected.warning else []
            )
            payload = recorder.requests[0].payload
            if expected.provider_value is None:
                assert "reasoning_effort" not in payload
            else:
                assert payload["reasoning_effort"] == expected.provider_value


@pytest.mark.integration
@pytest.mark.parametrize("method", ("chat", "stream_chat", "achat", "astream_chat"))
def test_sampling_validation_order_is_temperature_then_top_p(mistral_runtime, monkeypatch, method):
    adapter = _new_adapter(MODEL_IDS[0])
    recorder = RecordingSyncTransport(chat_result={})
    adapter.adapter._sync_transport = recorder
    _install_async(monkeypatch, model_response={})
    for kwargs, expected in [
        ({"temperature": 2, "top_p": -1}, "temperature"),
        ({"temperature": 1, "top_p": 2}, "top_p"),
    ]:
        with pytest.raises(ValueError, match=expected):
            _invoke(adapter, method, [{"role": "user", "content": "synthetic"}], **kwargs)
    assert recorder.requests == [] and recorder.stream_requests == []


@pytest.mark.integration
def test_stream_creation_is_lazy_and_preparation_precedes_ocr(mistral_runtime, monkeypatch):
    model = MODEL_IDS[0]
    adapter = _new_adapter(model)
    pdf_history = [UserMessage("read", files=[DocumentPart(url="https://example.invalid/a.pdf")])]
    empty_terminal = _terminal_event(model, content="")
    recorder = RecordingSyncTransport(chat_result=_response(model), events=(), ocr_results=[{
        "pages": [{"markdown": "synthetic markdown"}], "usage_info": {"pages_processed": 1}
    }])
    adapter.adapter._sync_transport = recorder
    stream = adapter.stream_chat(pdf_history, buffer_chars=0)
    assert recorder.ocr_requests == []
    with pytest.raises(ValueError, match="buffer"):
        next(stream)
    assert recorder.ocr_requests == []

    recorder = RecordingSyncTransport(chat_result=_response(model), events=[empty_terminal], ocr_results=[{
        "pages": [{"markdown": "synthetic markdown"}], "usage_info": {"pages_processed": 1}
    }])
    adapter.adapter._sync_transport = recorder
    completed = []
    stream = adapter.stream_chat(pdf_history, on_done=completed.append)
    assert recorder.ocr_requests == []
    with pytest.raises(StopIteration):
        next(stream)
    assert len(recorder.ocr_requests) == 1
    assert len(recorder.stream_requests) == 1
    assert len(completed) == 1 and completed[0].finish_reason == "stop"

    async_calls = _install_async(monkeypatch, model_response=_response(model), stream_events=[empty_terminal], ocr_results=[{
        "pages": [{"markdown": "synthetic markdown"}], "usage_info": {"pages_processed": 1}
    }])
    async def check_async_lazy():
        iterator = adapter.astream_chat(pdf_history, buffer_chars=0)
        assert async_calls["ocr"] == [] and async_calls["stream"] == []
        with pytest.raises(ValueError, match="buffer"):
            await anext(iterator)
        assert async_calls["ocr"] == [] and async_calls["stream"] == []

        completed_async = []
        iterator = adapter.astream_chat(pdf_history, on_done=completed_async.append)
        assert async_calls["ocr"] == [] and async_calls["stream"] == []
        with pytest.raises(StopAsyncIteration):
            await anext(iterator)
        assert len(completed_async) == 1 and completed_async[0].finish_reason == "stop"
    asyncio.run(check_async_lazy())
    assert len(async_calls["ocr"]) == 1
    assert len(async_calls["stream"]) == 1


@pytest.mark.integration
@pytest.mark.parametrize("method", ("stream_chat", "astream_chat"))
def test_stream_callback_failure_retains_caller_exception_identity(
    mistral_runtime, monkeypatch, method
):
    model = MODEL_IDS[0]
    adapter = _new_adapter(model)
    raw = _response(model)
    events = _events_for_response(raw)
    caller_error = RuntimeError("synthetic callback failure")

    def fail_on_done(response):
        raise caller_error

    with pytest.raises(RuntimeError) as caught:
        _execute(
            adapter, method, [{"role": "user", "content": "callback"}], monkeypatch,
            model_response=raw, stream_events=events, on_done=fail_on_done,
        )
    assert caught.value is caller_error


@pytest.mark.integration
@pytest.mark.parametrize(("method", "transport"), EXECUTION_PATHS)
def test_validation_errors_happen_before_pdf_ocr_in_native_order(
    mistral_runtime, monkeypatch, method, transport
):
    from llm_api_adapter.errors.llm_api_error import InvalidToolSchemaError
    from llm_api_adapter.errors.llm_api_error import ToolChoiceError

    model = MODEL_IDS[0]
    adapter = _new_adapter(model, transport)
    history = [UserMessage("read", files=[DocumentPart(url="https://example.invalid/a.pdf")])]
    recorder = RecordingSyncTransport(chat_result=_response(model))
    adapter.adapter._sync_transport = recorder
    async_calls = _install_async(monkeypatch, model_response=_response(model))
    cases = [
        ({"json_schema": {"type": "object", "required": ["missing"]}}, JSONSchemaError),
        ({"tools": [LOOKUP], "tool_choice": "absent"}, ToolChoiceError),
        ({"reasoning_level": True}, LLMReasoningLevelError),
    ]
    for kwargs, error in cases:
        with pytest.raises(error):
            _invoke(adapter, method, history, **kwargs)
        assert recorder.ocr_requests == [] and recorder.requests == []

    # Sampling precedes portable schema validation; schema validation precedes
    # named-tool selection; reasoning is resolved only after those requests.
    with pytest.raises(ValueError, match="temperature"):
        _invoke(adapter, method, history, temperature=2, json_schema={"type": "invalid"})
    with pytest.raises(JSONSchemaError):
        _invoke(
            adapter, method, history,
            json_schema={"type": "object", "required": ["missing"]},
            tools=[LOOKUP],
            tool_choice="absent",
            reasoning_level=True,
        )
    with pytest.raises(ToolChoiceError):
        _invoke(adapter, method, history, tools=[LOOKUP], tool_choice="absent", reasoning_level=True)
    with pytest.raises(InvalidToolSchemaError):
        _invoke(
            adapter, method, history,
            tools=[None], json_schema={"type": "invalid"},
            tool_choice="absent", reasoning_level=True,
        )
    if method in {"chat", "stream_chat"}:
        assert recorder.ocr_requests == [] and recorder.requests == [] and recorder.stream_requests == []
    else:
        assert async_calls["ocr"] == [] and async_calls["chat"] == [] and async_calls["stream"] == []


@pytest.mark.integration
def test_ocr_failure_stops_before_model_dispatch(mistral_runtime):
    model = MODEL_IDS[0]
    adapter = _new_adapter(model)
    recorder = RecordingSyncTransport(
        chat_result=_response(model),
        ocr_results=[{"pages": [], "usage_info": {"pages_processed": 1}}],
    )
    adapter.adapter._sync_transport = recorder
    with pytest.raises(LLMAPIClientError, match="no document text"):
        adapter.chat([UserMessage(
            "synthetic PDF", files=[DocumentPart(url="https://example.invalid/file.pdf")]
        )])
    assert len(recorder.ocr_requests) == 1
    assert recorder.requests == []


@pytest.mark.integration
@pytest.mark.parametrize("model", MODEL_IDS)
@pytest.mark.parametrize(("method", "transport"), EXECUTION_PATHS)
def test_multiple_pdfs_are_ocr_processed_in_order_before_one_model_request(
    mistral_runtime, monkeypatch, method, transport, model
):
    adapter = _new_adapter(model, transport)
    spec = mistral_runtime.organizations["mistral"].models[model]
    meter = resolve_metered_operation_spec(mistral_runtime, "mistral", "ocr")
    assert meter is not None
    history = [
        {"role": "system", "content": "synthetic system"},
        UserMessage("first", files=[
            DocumentPart(url="https://example.invalid/one.pdf"),
            DocumentPart(data=b"synthetic-pdf", media_type="application/pdf"),
            ImagePart(data=b"synthetic-image", media_type="image/png"),
        ]),
        {"role": "assistant", "content": "between"},
        UserMessage("second", files=[DocumentPart(url="data:application/pdf;base64,c3ludGhldGlj")]),
    ]
    token_usage = {
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "total_tokens": 12,
        "prompt_tokens_details": {"cached_tokens": 0},
    }
    wire_model = spec.provider_model or model
    raw = _response(wire_model, usage=token_usage)
    events = _events_for_response(raw)
    timeline = []
    callback_snapshots = []

    def snapshot(response):
        callback_snapshots.append((
            response.cost_total,
            tuple((line.model, line.unit, line.quantity, line.rate, line.currency, line.cost)
                  for line in response.cost_breakdown),
        ))

    ocr_results = [
        {"pages": [{"markdown": "first OCR"}], "usage_info": {"pages_processed": 2}},
        {"pages": [{"markdown": "second OCR"}], "usage_info": {"pages_processed": 3}},
        {"pages": [{"markdown": "third OCR"}], "usage_info": {"pages_processed": 1}},
    ]
    async_calls = _install_async(
        monkeypatch,
        model_response=raw,
        stream_events=events,
        ocr_results=ocr_results,
        timeline=timeline,
    )
    recorder = RecordingSyncTransport(
        chat_result=raw, events=events, ocr_results=ocr_results, timeline=timeline
    )
    adapter.adapter._sync_transport = recorder
    invoke_kwargs = {"timeout_s": 4}
    if method in {"stream_chat", "astream_chat"}:
        invoke_kwargs["on_done"] = snapshot
    result = _invoke(adapter, method, history, **invoke_kwargs)
    if method in {"chat", "stream_chat"}:
        requests = recorder.requests if method == "chat" else recorder.stream_requests
        assert len(recorder.ocr_requests) == 3 and len(requests) == 1
        assert not async_calls["ocr"] and not async_calls["chat"] and not async_calls["stream"]
        assert all(request.url == OCR_URL and request.payload["model"] == meter.model for request in recorder.ocr_requests)
        assert [request.timeout for request in recorder.ocr_requests] == [4, 4, 4]
        ocr_payloads = [request.payload for request in recorder.ocr_requests]
        sent = requests[0].payload["messages"]
    else:
        assert recorder.ocr_requests == [] and recorder.requests == [] and recorder.stream_requests == []
        bucket = "chat" if method == "achat" else "stream"
        assert len(async_calls["ocr"]) == 3 and len(async_calls[bucket]) == 1
        assert all(call[0] == OCR_URL for call in async_calls["ocr"])
        assert [call[1]["timeout"] for call in async_calls["ocr"]] == [4, 4, 4]
        ocr_payloads = [call[1]["payload"] for call in async_calls["ocr"]]
        sent = async_calls[bucket][0][1]["payload"]["messages"]
    model_dispatch = (
        "model" if method in {"chat", "stream_chat"}
        else "chat" if method == "achat"
        else "stream"
    )
    expected_chronology = ["ocr", "ocr", "ocr", model_dispatch]
    assert [kind for kind, _ in timeline] == expected_chronology
    assert [item["model"] for item in ocr_payloads] == [meter.model] * 3
    assert [item["document"]["document_url"] for item in ocr_payloads] == [
        "https://example.invalid/one.pdf",
        "data:application/pdf;base64,c3ludGhldGljLXBkZg==",
        "data:application/pdf;base64,c3ludGhldGlj",
    ]
    assert sent[0] == {"role": "system", "content": "synthetic system"}
    serialized_content = " ".join(
        part if isinstance(part, str) else part.get("text", "")
        for part in sent[1]["content"]
    )
    assert "first OCR" in serialized_content and "second OCR" in serialized_content
    assert serialized_content.index("first OCR") < serialized_content.index("second OCR")
    image_parts = [
        part for part in sent[1]["content"]
        if isinstance(part, dict) and part.get("type") == "image_url"
    ]
    assert len(image_parts) == 1
    assert image_parts[0]["image_url"] == "data:image/png;base64,c3ludGhldGljLWltYWdl"
    assert sent[2] == {"role": "assistant", "content": "between"}
    third_content = sent[3]["content"]
    third_text = third_content if isinstance(third_content, str) else " ".join(
        part if isinstance(part, str) else part.get("text", "") for part in third_content
    )
    assert "third OCR" in third_text
    semantic_result = result[1] if method in {"stream_chat", "astream_chat"} else result
    token_tier = spec.pricing_tiers.tier_for_prompt_tokens(10)
    expected_total = 10 * token_tier.in_per_token + 2 * token_tier.out_per_token + 6 * meter.rate
    assert semantic_result.cost_total == pytest.approx(expected_total)
    assert semantic_result.currency == meter.currency == spec.pricing_tiers.currency
    ocr = [item for item in semantic_result.cost_breakdown if item.operation == "ocr"]
    assert [
        (item.operation, item.model, item.unit, item.quantity, item.rate, item.currency, item.cost)
        for item in ocr
    ] == [
        (meter.name, meter.model, meter.unit, quantity, meter.rate, meter.currency, quantity * meter.rate)
        for quantity in (2, 3, 1)
    ]
    if method in {"stream_chat", "astream_chat"}:
        assert len(callback_snapshots) == 1
        callback_total, callback_lines = callback_snapshots[0]
        assert callback_total == pytest.approx(expected_total)
        assert callback_lines == tuple(
            (meter.model, meter.unit, quantity, meter.rate, meter.currency, quantity * meter.rate)
            for quantity in (2, 3, 1)
        )


@pytest.mark.integration
@pytest.mark.parametrize(
    ("second_usage", "missing_meter"),
    [
        ({"pages_processed": "unknown"}, False),
        ({}, False),
        ({"pages_processed": 2}, True),
    ],
    ids=("invalid-page-count", "missing-page-count", "missing-meter"),
)
def test_unknown_ocr_accounting_keeps_known_lines_and_total_unavailable(
    mistral_runtime, monkeypatch, second_usage, missing_meter
):
    model = MODEL_IDS[0]
    adapter = _new_adapter(model)
    if missing_meter:
        monkeypatch.setattr(adapter.adapter, "_ocr_meter", lambda: None)
    token_usage = {
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "total_tokens": 12,
        "prompt_tokens_details": {"cached_tokens": 0},
    }
    recorder = RecordingSyncTransport(
        chat_result=_response(model, usage=token_usage),
        ocr_results=[
            {"pages": [{"markdown": "known PDF"}], "usage_info": {"pages_processed": 3}},
            {"pages": [{"markdown": "unknown PDF"}], "usage_info": second_usage},
        ],
    )
    adapter.adapter._sync_transport = recorder
    response = adapter.chat([UserMessage("two PDFs", files=[
        DocumentPart(url="https://example.invalid/known.pdf"),
        DocumentPart(url="https://example.invalid/unknown.pdf"),
    ])])
    assert len(recorder.ocr_requests) == 2 and len(recorder.requests) == 1
    assert response.cost_input is not None and response.cost_output is not None
    assert response.cost_total is None
    lines = [line for line in response.cost_breakdown if line.operation == "ocr"]
    if missing_meter:
        assert lines == []
    else:
        assert [line.quantity for line in lines] == [3]
