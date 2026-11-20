"""Conversation contract evidence for generic Anthropic Messages connections."""

from __future__ import annotations

import asyncio
import base64
from copy import deepcopy
import json
import socket
from typing import Any

import httpx
import pytest
import requests
import requests_mock
from pydantic import BaseModel, ConfigDict

from src.llm_api_adapter.adapters.anthropic_compatible.adapter import AnthropicCompatibleAdapter
from src.llm_api_adapter.errors.config_errors import LLMReasoningLevelError
from src.llm_api_adapter.errors.llm_api_error import (
    InvalidToolArgumentsError, InvalidToolSchemaError, JSONSchemaError,
    LLMAPIClientError, ToolChoiceError,
)
from src.llm_api_adapter.models.messages.chat_message import Prompt, UserMessage
from src.llm_api_adapter.models.messages.file_parts import DocumentPart, FilePart, ImagePart
from src.llm_api_adapter.models.responses.chat_response import ChatResponse
from src.llm_api_adapter.models.tools.tool_spec import ToolSpec
from tests.fixtures.anthropic_compatible import (
    events_to_sse_lines, incomplete_response, messages_event_trace,
    messages_response, premature_eof_trace, refusal_response, tool_response,
)

pytestmark = pytest.mark.unit
ROOT = "https://example.invalid/conversations/v3"
URL = f"{ROOT}/messages"
MODEL = "unregistered/vendor-model"
KEY = "synthetic-conversation-key"
SYNC_LANES = ("requests_chat", "requests_stream", "httpx_chat", "httpx_stream")
ASYNC_LANES = ("async_chat", "async_stream")
LANES = SYNC_LANES + ASYNC_LANES
CANONICAL_LEVELS = ("none", "minimal", "low", "medium", "high", "very_high")
_HTTPX_CLIENT = httpx.Client
_HTTPX_ASYNC_CLIENT = httpx.AsyncClient


def _adapter(lane: str, model: str = MODEL) -> AnthropicCompatibleAdapter:
    transport = "httpx" if lane.startswith("httpx") else "requests"
    return AnthropicCompatibleAdapter(
        api_key=KEY, model=model, base_url=ROOT, transport=transport,
    )


@pytest.fixture(autouse=True)
def _block_dns(monkeypatch):
    def unexpected_resolution(*_args, **_kwargs):
        raise AssertionError("conversation contract tests must not resolve network hosts")
    monkeypatch.setattr(socket, "getaddrinfo", unexpected_resolution)


def _sse(raw: dict) -> str:
    events = messages_event_trace(
        blocks=raw.get("content", []),
        stop_reason=raw.get("stop_reason"),
        response_id=raw.get("id", "synthetic-id"),
        model=raw.get("model", MODEL),
    )
    return "\n".join(events_to_sse_lines(events)) + "\n"


def _invoke(
    lane: str, messages, raw_reply: dict, monkeypatch, *, observations=None,
    done_observations=None, sse_override=None, **kwargs,
):
    """One real public call with controlled requests/HTTPX, asserting its wire envelope."""
    stream = lane.endswith("stream")
    method = {
        "requests_chat": "chat", "requests_stream": "stream_chat",
        "httpx_chat": "chat", "httpx_stream": "stream_chat",
        "async_chat": "achat", "async_stream": "astream_chat",
    }[lane]
    timeout_s = kwargs.pop("timeout_s", 2.5)
    max_tokens = kwargs.get("max_tokens", 19)
    seen: list[tuple[Any, dict]] = observations if observations is not None else []
    done: list[ChatResponse] = done_observations if done_observations is not None else []
    reasoning = kwargs.pop("on_reasoning", None)
    kwargs.setdefault("max_tokens", max_tokens)
    kwargs.setdefault("timeout_s", timeout_s)

    def body_for(payload: dict) -> tuple[str | None, dict | None, dict[str, str]]:
        if payload.get("stream"):
            return sse_override or _sse(raw_reply), None, {"Content-Type": "text/event-stream"}
        return None, deepcopy(raw_reply), {}

    def remember(request: Any, payload: dict) -> tuple[str | None, dict | None, dict[str, str]]:
        seen.append((request, deepcopy(payload)))
        return body_for(payload)

    adapter = _adapter(lane, kwargs.pop("_model", MODEL))
    if lane == "requests_chat" or lane == "requests_stream":
        captured_timeouts = []
        with requests_mock.Mocker(real_http=False) as mock:
            original_request = requests.sessions.Session.request

            def observe_request(*request_args, **request_kwargs):
                captured_timeouts.append(request_kwargs.get("timeout"))
                return original_request(*request_args, **request_kwargs)

            monkeypatch.setattr(requests.sessions.Session, "request", observe_request)
            def callback(request, context):
                payload = request.json()
                body, response_json, headers = remember(request, payload)
                context.headers.update(headers)
                if response_json is not None:
                    return json.dumps(response_json)
                return body
            mock.register_uri("POST", URL, text=callback)
            response, visible = _call_public(adapter, method, messages, kwargs, done, reasoning)
            assert mock.call_count == 1
        assert captured_timeouts == [timeout_s]
    elif lane == "httpx_chat" or lane == "httpx_stream":
        requests_seen = []
        def factory(*args, **client_kwargs):
            def handler(request):
                assert str(request.url) == URL and request.method == "POST"
                payload = json.loads(request.content)
                requests_seen.append(request)
                body, response_json, headers = remember(request, payload)
                if response_json is not None:
                    return httpx.Response(200, json=response_json, headers=headers, request=request)
                return httpx.Response(200, text=body, headers=headers, request=request)
            client_kwargs["transport"] = httpx.MockTransport(handler)
            return _HTTPX_CLIENT(*args, **client_kwargs)
        monkeypatch.setattr(httpx, "Client", factory)
        response, visible = _call_public(adapter, method, messages, kwargs, done, reasoning)
        assert len(requests_seen) == 1
        assert requests_seen[0].extensions["timeout"]["read"] == timeout_s
    else:
        requests_seen = []
        def factory(*args, **client_kwargs):
            async def handler(request):
                assert str(request.url) == URL and request.method == "POST"
                payload = json.loads(request.content)
                requests_seen.append(request)
                body, response_json, headers = remember(request, payload)
                if response_json is not None:
                    return httpx.Response(200, json=response_json, headers=headers, request=request)
                return httpx.Response(200, text=body, headers=headers, request=request)
            client_kwargs["transport"] = httpx.MockTransport(handler)
            return _HTTPX_ASYNC_CLIENT(*args, **client_kwargs)
        monkeypatch.setattr(httpx, "AsyncClient", factory)
        adapter._create_sync_client = lambda: (_ for _ in ()).throw(
            AssertionError("async operation fell back to synchronous I/O")
        )
        response, visible = asyncio.run(
            _call_async(adapter, method, messages, kwargs, done, reasoning)
        )
        assert len(requests_seen) == 1
        assert requests_seen[0].extensions["timeout"]["read"] == timeout_s

    assert len(seen) == 1
    request, payload = seen[0]
    assert str(request.url) == URL and request.method == "POST"
    assert request.headers["x-api-key"] == KEY
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert request.headers["content-type"] == "application/json"
    assert payload["model"] == adapter.model
    assert payload["max_tokens"] == max_tokens
    assert payload.get("stream", False) is stream
    if stream:
        assert len(done) == 1 and done[0] is response
    else:
        assert done == []
    return response, visible, payload, seen


def _call_public(adapter, method, messages, kwargs, done, reasoning):
    call_kwargs = dict(kwargs)
    if method in {"stream_chat", "astream_chat"}:
        call_kwargs["on_done"] = done.append
        if reasoning is not None:
            call_kwargs["on_reasoning"] = reasoning
    result = getattr(adapter, method)(messages, **call_kwargs)
    if method == "stream_chat":
        visible = "".join(result)
        return done[-1], visible
    return result, None


async def _call_async(adapter, method, messages, kwargs, done, reasoning):
    call_kwargs = dict(kwargs)
    if method == "astream_chat":
        call_kwargs["on_done"] = done.append
        if reasoning is not None:
            call_kwargs["on_reasoning"] = reasoning
        visible = "".join([
            part async for part in adapter.astream_chat(messages, **call_kwargs)
        ])
        return done[-1], visible
    return await adapter.achat(messages, **call_kwargs), None


@pytest.mark.parametrize("lane", LANES)
def test_caller_driven_mixed_role_tool_loop_and_ignored_previous_response(lane, monkeypatch):
    tools = [ToolSpec("lookup", {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"], "additionalProperties": False})]
    first_reply = tool_response(tool_id="call-88", name="lookup", arguments={"q": "A"}, model=MODEL)
    first, visible, first_payload, first_seen = _invoke(
        lane, [Prompt("system"), UserMessage("find A")], first_reply, monkeypatch,
        tools=tools, tool_choice="lookup", parallel_tool_calls=False,
    )
    assert first.tool_calls[0].call_id == "call-88"
    assert first.tool_calls[0].name == "lookup"
    assert first.tool_calls[0].arguments == {"q": "A"}
    if lane.endswith("stream"):
        assert visible == ""
    assert first_payload["system"] == "system"
    assert first_payload["tool_choice"] == {"type": "tool", "name": "lookup"}
    assert first_payload["disable_parallel_tool_use"] is True

    history = [
        Prompt("system"), UserMessage("find A"),
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "call-88", "type": "function",
            "function": {"name": "lookup", "arguments": '{"q":"A"}'},
        }]},
        {"role": "tool", "tool_call_id": "call-88", "content": '{"value":"A"}'},
    ]
    history_before = deepcopy(history)
    prior = ChatResponse(
        response_id="opaque-response-id",
        provider_data={"opaque_sentinel": "must-not-replay"},
    )
    second_reply = messages_response(content=[
        {"type": "text", "text": "Found"},
        {"type": "text", "text": " A"},
    ], model=MODEL)
    second, visible, second_payload, second_seen = _invoke(
        lane, history, second_reply, monkeypatch,
        previous_response=prior, tools=tools, tool_choice={"type": "auto"},
        parallel_tool_calls=True,
    )
    assert second.content == "Found A"
    if lane.endswith("stream"):
        assert visible == "Found A"
    assert history == history_before
    assert second_payload["messages"] == [
        {"role": "user", "content": "find A"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "call-88", "name": "lookup", "input": {"q": "A"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call-88", "content": '{"value":"A"}'}]},
    ]
    assert second_payload["tool_choice"] == {"type": "auto"}
    assert second_payload["disable_parallel_tool_use"] is False
    serialized = json.dumps(second_payload)
    for sentinel in ("opaque-response-id", "must-not-replay", "previous_response", "previous_response_id", "store", "reasoning_summary"):
        assert sentinel not in serialized
    assert len(first_seen) + len(second_seen) == 2


@pytest.mark.parametrize("lane", LANES)
def test_malformed_completed_tool_arguments_raise_after_one_response(lane, monkeypatch):
    observed = []
    with pytest.raises(InvalidToolArgumentsError):
        _invoke(
            lane, [UserMessage("run lookup")],
            tool_response(tool_id="bad-call", name="lookup", arguments="not-an-object", model=MODEL),
            monkeypatch, observations=observed,
        )
    assert len(observed) == 1


@pytest.mark.parametrize("lane", ("requests_stream", "httpx_stream", "async_stream"))
def test_truncated_sse_is_an_error_not_a_valid_incomplete_response(lane, monkeypatch):
    events = premature_eof_trace(model=MODEL)
    body = "\n".join(events_to_sse_lines(events)) + "\n"
    observed, done = [], []
    with pytest.raises(LLMAPIClientError):
        _invoke(
            lane, [UserMessage("partial")], messages_response(text="unused", model=MODEL),
            monkeypatch, observations=observed, done_observations=done,
            sse_override=body,
        )
    assert len(observed) == 1 and done == []


@pytest.mark.parametrize("lane", LANES)
def test_all_tool_choice_forms_on_each_execution_lane(lane, monkeypatch):
    tools = [ToolSpec("lookup", {"type": "object"})]
    cases = [
        ("auto", {"type": "auto"}), ({"type": "auto"}, {"type": "auto"}),
        ("none", {"type": "none"}), ({"type": "none"}, {"type": "none"}),
        ("any", {"type": "any"}), ({"type": "any"}, {"type": "any"}),
        ("lookup", {"type": "tool", "name": "lookup"}),
        ({"type": "tool", "name": "lookup"}, {"type": "tool", "name": "lookup"}),
    ]
    count = 0
    for choice, wire in cases:
        _, _, payload, seen = _invoke(
            lane, [UserMessage("choose")], messages_response(text="ok", model=MODEL),
            monkeypatch, tools=tools, tool_choice=choice,
        )
        assert payload["tool_choice"] == wire
        count += len(seen)
    assert count == len(cases)


@pytest.mark.parametrize("lane", LANES)
def test_schema_model_refusal_incomplete_and_invalid_completions_on_each_lane(lane, monkeypatch):
    schema = {
        "type": "object", "$defs": {"answer": {"type": "string"}},
        "properties": {"result": {"$ref": "#/$defs/answer"}},
        "required": ["result"], "additionalProperties": False,
    }
    schema_before = deepcopy(schema)

    class Output(BaseModel):
        model_config = ConfigDict(extra="forbid")
        result: str

    raw_success, _, raw_payload, _ = _invoke(
        lane, [UserMessage("json")], messages_response(text='{"result":"yes"}', model=MODEL),
        monkeypatch, json_schema=schema,
    )
    assert raw_success.parsed_json == {"result": "yes"}
    assert raw_payload["output_config"]["format"]["schema"] == {
        "type": "object", "properties": {"result": {"type": "string"}},
        "required": ["result"], "additionalProperties": False,
    }
    model_success, _, model_payload, _ = _invoke(
        lane, [UserMessage("json")], messages_response(text='{"result":"typed"}', model=MODEL),
        monkeypatch, response_model=Output,
    )
    assert model_success.parsed_model == Output(result="typed")
    assert model_payload["output_config"]["format"] == {
        "type": "json_schema", "schema": Output.model_json_schema(),
    }
    assert raw_payload["output_config"]["format"] == {
        "type": "json_schema", "schema": {
            "type": "object", "properties": {"result": {"type": "string"}},
            "required": ["result"], "additionalProperties": False,
        },
    }
    assert schema == schema_before

    refused, _, _, _ = _invoke(
        lane, [UserMessage("json")], refusal_response(model=MODEL), monkeypatch,
        json_schema=schema,
    )
    incomplete, _, _, _ = _invoke(
        lane, [UserMessage("json")], incomplete_response(text='{"result":', model=MODEL),
        monkeypatch, json_schema=schema,
    )
    refused_model, _, _, _ = _invoke(
        lane, [UserMessage("json")], refusal_response(model=MODEL), monkeypatch,
        response_model=Output,
    )
    incomplete_model, _, _, _ = _invoke(
        lane, [UserMessage("json")], incomplete_response(text='{"result":', model=MODEL),
        monkeypatch, response_model=Output,
    )
    assert refused.refusal and refused.parsed_json is None and refused.parsed_model is None
    assert incomplete.incomplete_reason and incomplete.parsed_json is None and incomplete.parsed_model is None
    assert refused_model.refusal and refused_model.parsed_json is None and refused_model.parsed_model is None
    assert incomplete_model.incomplete_reason and incomplete_model.parsed_json is None and incomplete_model.parsed_model is None

    for reply, options in (
        (messages_response(text='{"result":17}', model=MODEL), {"response_model": Output}),
        (messages_response(text="not-json", model=MODEL), {"json_schema": schema}),
    ):
        observed = []
        with pytest.raises(JSONSchemaError):
            _invoke(lane, [UserMessage("json")], reply, monkeypatch, observations=observed, **options)
        assert len(observed) == 1


@pytest.mark.parametrize("lane", LANES)
def test_typed_and_dictionary_image_pdf_sources_preserve_order(lane, monkeypatch):
    image, pdf = b"synthetic image", b"synthetic pdf"
    typed_files = [
        ImagePart(url="https://assets.invalid/a.png"),
        ImagePart(data=image, media_type="image/png"),
        ImagePart(url="data:image/jpeg;base64,QUJD"),
        DocumentPart(url="https://assets.invalid/a.pdf"),
        DocumentPart(data=pdf, media_type="application/pdf"),
        DocumentPart(url="data:application/pdf;base64,REVG"),
    ]
    typed_snapshot = deepcopy(typed_files)
    _, _, payload, seen = _invoke(
        lane, [UserMessage("ordered", files=typed_files)], messages_response(text="ok", model=MODEL),
        monkeypatch,
    )
    assert payload["messages"][0]["content"] == [
        {"type": "text", "text": "ordered"},
        {"type": "image", "source": {"type": "url", "url": "https://assets.invalid/a.png"}},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(image).decode()}},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "QUJD"}},
        {"type": "document", "source": {"type": "url", "url": "https://assets.invalid/a.pdf"}},
        {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": base64.b64encode(pdf).decode()}},
        {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "REVG"}},
    ]
    assert typed_files == typed_snapshot and len(seen) == 1

    dictionary_history = [{
        "role": "user", "content": "dictionary inputs",
        "files": [
            {"type": "image_url", "image_url": {"url": "https://assets.invalid/dict.png"}},
            {"type": "input_file", "file_url": "https://assets.invalid/dict.pdf"},
        ],
    }]
    dictionary_snapshot = deepcopy(dictionary_history)
    _, _, dictionary_payload, dictionary_seen = _invoke(
        lane, dictionary_history, messages_response(text="ok", model=MODEL), monkeypatch,
    )
    assert dictionary_payload["messages"][0]["content"] == [
        {"type": "text", "text": "dictionary inputs"},
        {"type": "image", "source": {"type": "url", "url": "https://assets.invalid/dict.png"}},
        {"type": "document", "source": {"type": "url", "url": "https://assets.invalid/dict.pdf"}},
    ]
    assert dictionary_history == dictionary_snapshot and len(dictionary_seen) == 1


@pytest.mark.parametrize("lane", LANES)
def test_reasoning_and_previous_response_boundaries_on_each_lane(lane, monkeypatch):
    model = "claude-sonnet-4-5-20250929"
    prior = ChatResponse(
        response_id="opaque-id-192", provider_data={"opaque_sentinel": "never-replay-this"},
    )
    history = [Prompt("caller supplied system"), UserMessage("full caller history")]
    history_before = deepcopy(history)
    reasoning_callback: list[str] = []
    raw = messages_response(content=[
        {"type": "thinking", "thinking": "returned only"},
        {"type": "text", "text": "visible"},
    ], model=model)
    level = CANONICAL_LEVELS[LANES.index(lane)]
    with pytest.warns(UserWarning, match="ignored"):
        response, visible, payload, seen = _invoke(
            lane, history, raw, monkeypatch, _model=model,
            previous_response=prior, reasoning_level=level, capture_reasoning=False,
            on_reasoning=lambda event: reasoning_callback.append(event.text),
        )
    assert _adapter(lane, model).model_spec is None
    assert response.content == "visible" and response.reasoning_events == []
    if lane.endswith("stream"):
        assert visible == "visible" and reasoning_callback == []
    assert payload["system"] == "caller supplied system"
    assert payload["messages"] == [{"role": "user", "content": "full caller history"}]
    assert "output_config" not in payload or "effort" not in payload["output_config"]
    assert not ({"thinking", "reasoning_effort", "budget_tokens", "include_reasoning"} & payload.keys())
    assert not any(token in json.dumps(payload) for token in ("opaque-id-192", "never-replay-this", "previous_response", "store", "reasoning_summary"))

    positive_callback: list[str] = []
    captured, visible, captured_payload, captured_seen = _invoke(
        lane, history, raw, monkeypatch, _model=model,
        previous_response=prior, capture_reasoning=True,
        on_reasoning=lambda event: positive_callback.append(event.text),
    )
    assert [event.text for event in captured.reasoning_events] == ["returned only"]
    if lane.endswith("stream"):
        assert visible == "visible" and positive_callback == ["returned only"]
    assert captured_payload == payload

    for numeric_control in (0, 12):
        with pytest.warns(UserWarning, match="ignored"):
            _, _, numeric_payload, numeric_seen = _invoke(
                lane, history, messages_response(text="numeric control", model=model),
                monkeypatch, _model=model, previous_response=prior,
                reasoning_level=numeric_control,
            )
        assert numeric_payload == payload
        captured_seen.extend(numeric_seen)

    no_thinking_callback: list[str] = []
    absent, _, absent_payload, absent_seen = _invoke(
        lane, history, messages_response(text="ordinary", model=model), monkeypatch,
        _model=model, previous_response=prior, capture_reasoning=True,
        on_reasoning=lambda event: no_thinking_callback.append(event.text),
    )
    assert absent.reasoning_events == []
    assert absent_payload == payload
    if lane.endswith("stream"):
        assert no_thinking_callback == []
    assert history == history_before
    assert len(seen) + len(captured_seen) + len(absent_seen) == 5


@pytest.mark.parametrize("lane", LANES)
def test_sampling_defaults_and_explicit_values_across_lanes(lane, monkeypatch):
    _, _, defaults, default_seen = _invoke(
        lane, [UserMessage("sampling")], messages_response(text="ok", model=MODEL), monkeypatch,
    )
    assert "temperature" not in defaults and "top_p" not in defaults
    _, _, explicit, explicit_seen = _invoke(
        lane, [UserMessage("sampling")], messages_response(text="ok", model=MODEL),
        monkeypatch, temperature=0.4, top_p=0.8,
    )
    assert explicit["temperature"] == 0.4 and explicit["top_p"] == 0.8
    default_forms = []
    for temperature, top_p in ((1, 1.0), (1.0, 1)):
        _, _, payload, seen = _invoke(
            lane, [UserMessage("sampling")], messages_response(text="ok", model=MODEL),
            monkeypatch, temperature=temperature, top_p=top_p,
        )
        assert "temperature" not in payload and "top_p" not in payload
        default_forms.extend(seen)
    assert len(default_seen) + len(explicit_seen) + len(default_forms) == 4


@pytest.mark.parametrize("kwargs,error", [
    ({"tools": [ToolSpec("", {"type": "object"})]}, InvalidToolSchemaError),
    ({"tools": [ToolSpec("x", [])]}, InvalidToolSchemaError),
    ({"tools": [ToolSpec("x", {"type": "object"}), ToolSpec("x", {"type": "object"})]}, InvalidToolSchemaError),
    ({"tools": [ToolSpec("x", {"type": "object"})], "tool_choice": "missing"}, ToolChoiceError),
    ({"tool_choice": "any"}, ToolChoiceError),
    ({"tool_choice": {"type": "tool", "name": "x"}}, ToolChoiceError),
    ({"tool_choice": {"type": "required"}}, ToolChoiceError),
    ({"tools": [ToolSpec("x", {"type": "object"})], "tool_choice": {"type": "function", "function": {"name": "x"}}}, ToolChoiceError),
])
def test_invalid_tool_schema_or_choice_rejected_locally(kwargs, error):
    with requests_mock.Mocker(real_http=False) as mock:
        with pytest.raises(error):
            _adapter("requests_chat").chat([UserMessage("bad request")], max_tokens=5, **kwargs)
        assert mock.call_count == 0


@pytest.mark.parametrize("schema", [
    {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"], "additionalProperties": False, "$ref": "https://schemas.invalid/external.json"},
])
def test_unsupported_portable_schema_fails_before_dispatch(schema):
    with requests_mock.Mocker(real_http=False) as mock:
        with pytest.raises(JSONSchemaError):
            _adapter("requests_chat").chat([UserMessage("schema")], max_tokens=5, json_schema=schema)
        assert mock.call_count == 0


@pytest.mark.parametrize("kwargs", [
    {"tools": [ToolSpec("x", {"type": "object"})], "json_schema": {"type": "object", "additionalProperties": False}},
    {"tools": [ToolSpec("x", {"type": "object"})], "response_model": None},
    {"json_schema": {"type": "object", "additionalProperties": False}, "response_model": None},
])
def test_forbidden_structured_output_combinations_are_local(kwargs):
    class ValidOutput(BaseModel):
        model_config = ConfigDict(extra="forbid")
        result: str
    if "response_model" in kwargs:
        kwargs["response_model"] = ValidOutput
    with requests_mock.Mocker(real_http=False) as mock:
        with pytest.raises(JSONSchemaError):
            _adapter("requests_chat").chat([UserMessage("combined")], max_tokens=5, **kwargs)
        assert mock.call_count == 0


def test_unsupported_file_part_rejected_without_io():
    with requests_mock.Mocker(real_http=False) as mock:
        with pytest.raises(ValueError):
            _adapter("requests_chat").chat(
                [UserMessage("unsupported", files=[FilePart(data=b"audio", media_type="audio/mpeg")])],
                max_tokens=5,
            )
        assert mock.call_count == 0


@pytest.mark.parametrize("reasoning", [True, False, -1, "max", 1.5])
def test_invalid_unknown_model_reasoning_fails_before_io(reasoning):
    with requests_mock.Mocker(real_http=False) as mock:
        with pytest.raises(LLMReasoningLevelError):
            _adapter("requests_chat").chat([UserMessage("hi")], max_tokens=5, reasoning_level=reasoning)
        assert mock.call_count == 0


@pytest.mark.parametrize("kwargs", [
    {"temperature": -0.1}, {"temperature": 2.01},
    {"top_p": -0.1}, {"top_p": 1.01}, {"max_tokens": 0},
])
def test_out_of_range_sampling_and_output_limit_fail_before_io(kwargs):
    parameters = {"max_tokens": 5, **kwargs}
    with requests_mock.Mocker(real_http=False) as mock:
        with pytest.raises(ValueError):
            _adapter("requests_chat").chat([UserMessage("hi")], **parameters)
        assert mock.call_count == 0
