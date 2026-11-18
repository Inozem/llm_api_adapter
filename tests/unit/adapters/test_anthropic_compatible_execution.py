"""Bounded end-to-end wiring and lifecycle checks for compatible Messages."""

from __future__ import annotations

import asyncio
import builtins
import json
import socket
import traceback
from dataclasses import FrozenInstanceError

import httpx
import pytest
import requests_mock

from src.llm_api_adapter.adapters.anthropic_compatible.adapter import AnthropicCompatibleAdapter
from src.llm_api_adapter.adapters.anthropic_compatible.execution import _PreparedMessagesOperation
from src.llm_api_adapter.errors.llm_api_error import InvalidToolArgumentsError, LLMAPIClientError
from src.llm_api_adapter.llms.transports import SSEEvent, TransportRequest
from src.llm_api_adapter.models.messages.chat_message import UserMessage
from src.llm_api_adapter.models.tools.tool_spec import ToolSpec
from tests.fixtures.anthropic_compatible import (
    ControlledAsyncSource,
    ControlledSyncSource,
    events_to_sse_lines,
    messages_event_trace,
    messages_response,
    tool_response,
)

pytestmark = pytest.mark.unit

ROOT = "https://example.invalid/proxy/v1"
URL = f"{ROOT}/messages"
MODEL = "vendor/unregistered-model"
KEY = "synthetic-api-key-012345"


def _adapter(**kwargs):
    return AnthropicCompatibleAdapter(
        api_key=KEY, model=MODEL, base_url=ROOT, **kwargs
    )


def _guard_resolution(monkeypatch):
    import src.llm_api_adapter.adapters.base_adapter as base_module
    import src.llm_api_adapter.llm_registry.llm_registry as registry_module
    import src.llm_api_adapter.universal_adapter as universal_module

    def forbidden(*_args, **_kwargs):
        raise AssertionError("generic operation must not resolve or discover models")

    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(base_module, "resolve_model_spec", forbidden)
    monkeypatch.setattr(registry_module, "resolve_model_spec", forbidden)
    monkeypatch.setattr(universal_module.ORGANIZATION_PLUGIN_DISCOVERY, "discover", forbidden)


def _messages_sse(*, text="hello", model=MODEL):
    events = messages_event_trace(
        blocks=[{"type": "text", "text": text}], model=model
    )
    return "\n".join(events_to_sse_lines(events)) + "\n"


@pytest.mark.parametrize("transport", ["requests", "httpx"])
def test_chat_routes_exactly_once_to_selected_root_with_exact_model_and_key(
    transport, monkeypatch
):
    _guard_resolution(monkeypatch)
    adapter = _adapter(transport=transport)
    reply = messages_response(text="hello")
    requests = []
    if transport == "requests":
        with requests_mock.Mocker(real_http=False) as http:
            http.post(URL, json=reply)
            response = adapter.chat([UserMessage(content="hi")], max_tokens=12)
            assert http.call_count == 1
            request = http.request_history[0]
            assert request.url == URL
            assert request.headers["x-api-key"] == KEY
            sent = request.json()
    else:
        real_client = httpx.Client

        def factory():
            def handler(request):
                requests.append(request)
                return httpx.Response(200, json=reply, request=request)
            return real_client(transport=httpx.MockTransport(handler))

        monkeypatch.setattr(httpx, "Client", factory)
        response = adapter.chat([UserMessage(content="hi")], max_tokens=12)
        assert len(requests) == 1
        request = requests[0]
        assert str(request.url) == URL
        assert request.headers["x-api-key"] == KEY
        sent = json.loads(request.content)
    assert sent["model"] == MODEL
    assert sent["max_tokens"] == 12
    assert response.content == "hello"
    assert adapter.model_spec is None


@pytest.mark.parametrize("transport", ["requests", "httpx"])
def test_stream_routes_once_and_orders_tool_and_completion_callbacks(transport, monkeypatch):
    _guard_resolution(monkeypatch)
    adapter = _adapter(transport=transport)
    blocks = [
        {"type": "text", "text": "visible "},
        {"type": "tool_use", "id": "call-7", "name": "lookup", "input": {"q": "x"}},
        {"type": "thinking", "thinking": "private thought"},
        {"type": "text", "text": "tail"},
    ]
    events = messages_event_trace(blocks=blocks, model=MODEL)
    body = "\n".join(events_to_sse_lines(events)) + "\n"
    seen = []
    if transport == "requests":
        with requests_mock.Mocker(real_http=False) as http:
            http.post(URL, text=body, headers={"Content-Type": "text/event-stream"})
            output = list(adapter.stream_chat(
                [UserMessage(content="hi")], max_tokens=30,
                tools=[ToolSpec(name="lookup", json_schema={"type": "object"})],
                tool_choice="any", capture_reasoning=True,
                on_reasoning=lambda item: seen.append(("reasoning", item.text)),
                on_tool_call=lambda call: seen.append(("tool", call.call_id, call.arguments)),
                on_done=lambda response: seen.append(("done", response.content)),
            ))
            assert http.call_count == 1
            assert http.request_history[0].url == URL
            assert http.request_history[0].headers["x-api-key"] == KEY
            payload = http.request_history[0].json()
    else:
        real_client = httpx.Client
        calls = []

        def factory():
            def handler(request):
                calls.append(request)
                return httpx.Response(200, text=body, headers={
                    "Content-Type": "text/event-stream"
                }, request=request)
            return real_client(transport=httpx.MockTransport(handler))

        monkeypatch.setattr(httpx, "Client", factory)
        output = list(adapter.stream_chat(
            [UserMessage(content="hi")], max_tokens=30,
            tools=[ToolSpec(name="lookup", json_schema={"type": "object"})],
            tool_choice="any", capture_reasoning=True,
            on_reasoning=lambda item: seen.append(("reasoning", item.text)),
            on_tool_call=lambda call: seen.append(("tool", call.call_id, call.arguments)),
            on_done=lambda response: seen.append(("done", response.content)),
        ))
        assert len(calls) == 1 and str(calls[0].url) == URL
        assert calls[0].headers["x-api-key"] == KEY
        payload = json.loads(calls[0].content)
    assert payload["model"] == MODEL and payload["tool_choice"] == {"type": "any"}
    assert "".join(output) == "visible tail"
    assert seen[0] == ("reasoning", "private thought")
    assert seen[1] == ("tool", "call-7", {"q": "x"})
    assert seen[2] == ("done", "visible tail")


def test_structured_response_uses_common_schema_finalizer():
    schema = {
        "type": "object", "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"], "additionalProperties": False,
    }
    reply = messages_response(text='{"ok":true}')
    with requests_mock.Mocker(real_http=False) as http:
        http.post(URL, json=reply)
        response = _adapter().chat(
            [UserMessage(content="return json")], max_tokens=20, json_schema=schema
        )
    assert http.call_count == 1
    assert http.request_history[0].json()["output_config"]["format"]["schema"] == schema
    assert response.parsed_json == {"ok": True}


def test_malformed_tool_arguments_retain_category_without_key_or_tool_data():
    reply = tool_response(name=KEY, arguments="{" + KEY + ": 'secret'}")
    with requests_mock.Mocker(real_http=False) as http:
        http.post(URL, json=reply)
        with pytest.raises(InvalidToolArgumentsError) as caught:
            _adapter().chat([UserMessage(content="hi")], max_tokens=8)
    assert http.call_count == 1
    diagnostic = "\n".join((str(caught.value), repr(caught.value), repr(caught.value.args),
                             "".join(traceback.format_exception(caught.value))))
    assert KEY not in diagnostic
    assert "secret" not in diagnostic
    assert caught.value.__cause__ is None


def test_async_chat_keeps_overlapping_payloads_operation_local(monkeypatch):
    _guard_resolution(monkeypatch)
    async def scenario():
        adapter = _adapter(transport="httpx")
        real_client = httpx.AsyncClient
        requests = []
        ready = asyncio.Event()

        def factory():
            async def handler(request):
                payload = json.loads(request.content)
                requests.append((str(request.url), payload, dict(request.headers)))
                if len(requests) == 2:
                    ready.set()
                await asyncio.wait_for(ready.wait(), timeout=2)
                return httpx.Response(200, json=messages_response(
                    text=f"max-{payload['max_tokens']}"
                ), request=request)
            return real_client(transport=httpx.MockTransport(handler))

        monkeypatch.setattr(httpx, "AsyncClient", factory)
        adapter._create_sync_client = lambda: (_ for _ in ()).throw(
            AssertionError("async execution constructed a sync client")
        )
        one, two = await asyncio.wait_for(asyncio.gather(
            adapter.achat([UserMessage(content="alpha")], max_tokens=5),
            adapter.achat([UserMessage(content="beta")], max_tokens=9),
        ), timeout=3)
        by_prompt = {entry[1]["messages"][0]["content"]: entry for entry in requests}
        assert len(requests) == 2
        assert all(entry[0] == URL for entry in requests)
        assert all(entry[1]["model"] == MODEL for entry in requests)
        assert all(entry[2]["x-api-key"] == KEY for entry in requests)
        assert by_prompt["alpha"][1]["max_tokens"] == 5
        assert by_prompt["beta"][1]["max_tokens"] == 9
        assert one.content == "max-5"
        assert two.content == "max-9"
    asyncio.run(scenario())


def test_async_stream_dispatches_once_and_awaits_completion_callback(monkeypatch):
    _guard_resolution(monkeypatch)
    async def scenario():
        adapter = _adapter()
        real_client = httpx.AsyncClient
        captured = []

        def factory():
            async def handler(request):
                captured.append(request)
                return httpx.Response(200, text=_messages_sse(), headers={
                    "Content-Type": "text/event-stream"
                }, request=request)
            return real_client(transport=httpx.MockTransport(handler))

        monkeypatch.setattr(httpx, "AsyncClient", factory)
        completed = []

        async def on_done(response):
            await asyncio.sleep(0)
            completed.append(response.content)

        output = []
        async for text in adapter.astream_chat(
            [UserMessage(content="hi")], max_tokens=6, on_done=on_done
        ):
            output.append(text)
        assert "".join(output) == "hello"
        assert completed == ["hello"]
        assert len(captured) == 1
        assert str(captured[0].url) == URL
        assert captured[0].headers["x-api-key"] == KEY
        payload = json.loads(captured[0].content)
        assert payload["model"] == MODEL and payload["max_tokens"] == 6
    asyncio.run(scenario())


def test_async_stream_public_aclose_closes_owned_source():
    # In-memory source confirms public aclose immediately reaches the source.
    async def scenario():
        adapter = _adapter()
        events = messages_event_trace(
            blocks=[{"type": "text", "text": "first second"}], model=MODEL
        )
        source = ControlledAsyncSource(events)

        class Client:
            def _headers(self): return {}
            def _execute_prepared_stream(self, _request): return source

        adapter._create_async_client = lambda: Client()
        done = []
        stream = adapter.astream_chat(
            [UserMessage(content="hi")], max_tokens=10, buffer_chars=1,
            on_done=lambda response: done.append(response),
        )
        assert await asyncio.wait_for(stream.__anext__(), timeout=2) == "f"
        await stream.aclose()
        assert source.aclose_calls >= 1
        assert done == []
    asyncio.run(scenario())


def test_async_cancellation_closes_source_and_keeps_cancellation_over_cleanup():
    async def scenario():
        adapter = _adapter()
        entered = asyncio.Event()
        cleanup = RuntimeError("cleanup failure")
        trace = messages_event_trace(blocks=[{"type": "text", "text": "x"}], model=MODEL)

        class BlockedSource:
            def __init__(self):
                self._events = iter(trace)
                self.closed = 0
            def __aiter__(self): return self
            async def __anext__(self):
                try:
                    return next(self._events)
                except StopIteration:
                    entered.set()
                    await asyncio.Future()
            async def aclose(self):
                self.closed += 1
                raise cleanup

        source = BlockedSource()
        class Client:
            def _headers(self): return {}
            def _execute_prepared_stream(self, _request): return source
        adapter._create_async_client = lambda: Client()
        stream = adapter.astream_chat([UserMessage(content="hi")], max_tokens=10)
        assert await stream.__anext__() == "x"
        pending = asyncio.create_task(stream.__anext__())
        await asyncio.wait_for(entered.wait(), timeout=2)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert source.closed >= 1
        await stream.aclose()
    asyncio.run(scenario())


def test_sync_early_close_and_malformed_eof_never_finalize_pending_text():
    adapter = _adapter()
    source = ControlledSyncSource(messages_event_trace(
        blocks=[{"type": "text", "text": "pending"}], model=MODEL
    ))
    class Client:
        def _headers(self): return {}
        def _execute_prepared_stream(self, _request): return source
    adapter._create_sync_client = lambda: Client()
    done = []
    stream = adapter.stream_chat(
        [UserMessage(content="hi")], max_tokens=10, buffer_chars=1,
        on_done=done.append,
    )
    assert next(stream) == "p"
    stream.close()
    assert source.close_calls >= 1 and done == []

    source = ControlledSyncSource(messages_event_trace(
        blocks=[{"type": "text", "text": "hidden pending"}], model=MODEL,
        include_stop_reason=False, include_message_stop=False,
    ))
    adapter._create_sync_client = lambda: Client()
    done.clear()
    stream = adapter.stream_chat(
        [UserMessage(content="hi")], max_tokens=10, buffer_chars=100,
        on_done=done.append,
    )
    with pytest.raises(LLMAPIClientError):
        list(stream)
    assert done == [] and source.close_calls >= 1


def test_positive_max_tokens_and_buffer_validation_precede_transport():
    adapter = _adapter()
    calls = []
    adapter._create_sync_client = lambda: calls.append("client")
    for value in (True, 0):
        with pytest.raises(ValueError):
            adapter.chat([UserMessage(content="hi")], max_tokens=value)
    with pytest.raises(ValueError):
        adapter.stream_chat(
            [UserMessage(content="hi")], max_tokens=5, buffer_chars=0
        )
    assert calls == []


def test_missing_transport_extra_keeps_actionable_import_error(monkeypatch):
    adapter = _adapter(transport="httpx")
    calls = []

    real_import = builtins.__import__
    def blocked_import(name, *args, **kwargs):
        if name == "httpx":
            calls.append(name)
            raise ImportError("Install the optional 'httpx' extra for this transport.")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(ImportError, match="optional 'httpx' dependency"):
        adapter.chat([UserMessage(content="hi")], max_tokens=5)
    assert calls == ["httpx"]


def test_callback_failure_identity_survives_source_cleanup_failure():
    adapter = _adapter()
    original = ValueError("callback failure")
    source = ControlledSyncSource(
        messages_event_trace(blocks=[{"type": "text", "text": "x"}], model=MODEL),
        cleanup_error=RuntimeError("cleanup failure"),
    )
    class Client:
        def _headers(self): return {}
        def _execute_prepared_stream(self, _request): return source
    adapter._create_sync_client = lambda: Client()

    def fail(_value):
        raise original
    with pytest.raises(ValueError) as caught:
        list(adapter.stream_chat(
            [UserMessage(content="hi")], max_tokens=10, on_delta=fail
        ))
    assert caught.value is original
    assert source.close_calls >= 1


def test_async_callback_failure_identity_survives_source_cleanup_failure():
    async def scenario():
        adapter = _adapter()
        original = ValueError("async callback failure")
        source = ControlledAsyncSource(
            messages_event_trace(
                blocks=[{"type": "text", "text": "x"}], model=MODEL
            ),
            cleanup_error=RuntimeError("cleanup failure"),
        )

        class Client:
            def _headers(self): return {}
            def _execute_prepared_stream(self, _request): return source
        adapter._create_async_client = lambda: Client()

        def fail(_value):
            raise original

        stream = adapter.astream_chat(
            [UserMessage(content="hi")], max_tokens=10, on_delta=fail
        )
        with pytest.raises(ValueError) as caught:
            await stream.__anext__()
        assert caught.value is original
        assert source.aclose_calls >= 1
    asyncio.run(scenario())


def test_prepared_and_stream_state_repr_is_complete_and_private():
    adapter = _adapter()
    state = adapter._new_messages_stream_state(8, True)
    state.common.reasoning_response.content = "private reasoning marker"
    operation = adapter._prepare_operation(
        [UserMessage(content="private prompt marker")], max_tokens=7
    )
    rendered = "\n".join((repr(adapter), repr(operation), repr(operation.transport_request),
                           repr(state)))
    for marker in (KEY, "private prompt marker", "private reasoning marker"):
        assert marker not in rendered
    with pytest.raises(FrozenInstanceError):
        operation.capture_reasoning = True
    with pytest.raises(TypeError):
        operation.transport_request.headers["x-api-key"] = "mutated"


def test_native_like_prepared_runner_needs_no_generic_routing_fields():
    class NativeRunner(AnthropicCompatibleAdapter):
        pass

    class NativeClient:
        def _execute_prepared_request(self, request):
            assert request.url == "https://native.invalid/messages"
            return type("Result", (), {"json": lambda self: messages_response(
                text="native result", model="native-model"
            )})()

    generic = _adapter()
    context = generic._prepare_chat_request(
        [UserMessage(content="hi")], None, None, None, None
    )
    native = NativeRunner.__new__(NativeRunner)
    native.api_key = "native-key"
    native.model = "native-model"
    native.company = "native"
    native.service_provider = "native"
    native.pricing = None
    operation = _PreparedMessagesOperation(
        chat_context=context,
        transport_request=TransportRequest(
            "https://native.invalid/messages", headers={}, payload={"model": "native-model"}
        ),
        client=NativeClient(),
    )
    assert native._run_prepared_chat(operation).content == "native result"


@pytest.mark.parametrize("value", [True, False, "1", float("nan"), float("inf")])
def test_invalid_sampling_rejected_before_dispatch(value):
    adapter = _adapter()
    calls = []
    adapter._create_sync_client = lambda: calls.append("client")
    with pytest.raises(ValueError):
        adapter.chat([UserMessage(content="hi")], max_tokens=5, temperature=value)
    assert calls == []


@pytest.mark.parametrize("value", [1, "yes", 0.5])
def test_invalid_parallel_tool_setting_rejected_before_dispatch(value):
    adapter = _adapter()
    calls = []
    adapter._create_sync_client = lambda: calls.append("client")
    with pytest.raises(ValueError):
        adapter.chat(
            [UserMessage(content="hi")], max_tokens=5, parallel_tool_calls=value
        )
    assert calls == []
