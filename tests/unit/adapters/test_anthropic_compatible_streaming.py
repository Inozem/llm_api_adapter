from copy import deepcopy
import asyncio
import json

import pytest

from src.llm_api_adapter.adapters.anthropic_compatible.streaming import (
    MessagesBlockState,
    consume_messages_event,
    finalize_messages_stream,
)
from src.llm_api_adapter.adapters.anthropic_compatible.payloads import normalize_messages_response
from src.llm_api_adapter.errors.llm_api_error import InvalidToolArgumentsError, LLMAPIClientError
from src.llm_api_adapter.llms.transports import SSEEvent
from tests.unit.adapters.anthropic_compatible_helpers import first_text_block
from tests.fixtures.anthropic_compatible import (
    events_to_sse_lines,
    messages_event_trace,
    premature_eof_trace,
    unknown_and_ping_events,
)

pytestmark = pytest.mark.unit
_MISSING = object()


@pytest.fixture(autouse=True)
def _block_network_resolution(monkeypatch):
    import socket

    def forbidden_resolution(*_args, **_kwargs):
        raise AssertionError("unit streaming tests must not resolve network hosts")

    monkeypatch.setattr(socket, "getaddrinfo", forbidden_resolution)


def test_common_accumulator_reconstructs_fragmented_blocks_and_complete_terminal():
    events = messages_event_trace(
        text_fragments={0: ["first", " second"]},
        tool_argument_fragments={1: ['{"x":', '2}']},
        blocks=[
            {"type": "text", "text": "first second"},
            {"type": "tool_use", "id": "tool-1", "name": "lookup", "input": {"x": 2}},
            {"type": "thinking", "thinking": "private"},
            {"type": "text", "text": "tail"},
        ],
    )
    state = MessagesBlockState()
    chunks = [consume_messages_event(state, event) for event in events]
    response = finalize_messages_stream(state)
    assert "".join(text for text, _ in chunks) == "first secondtail"
    assert response["content"][0]["text"] == "first second"
    assert response["content"][1]["input"] == {"x": 2}
    assert state.message_stop and not state.open_indexes
    assert state.usage is None
    normalized = normalize_messages_response(response, capture_reasoning=True)
    assert normalized.content == "first secondtail"
    assert normalized.tool_calls[0].call_id == "tool-1"
    assert normalized.tool_calls[0].arguments == {"x": 2}
    assert normalized.reasoning_events[0].text == "private"
    original_blocks = deepcopy(response["content"])
    native_like = normalize_messages_response(
        response, text_normalizer=first_text_block
    )
    assert native_like.content == "first second"
    assert response["content"] == original_blocks


def test_generic_finalization_rejects_eof_but_native_policy_can_finalize_same_state():
    state = MessagesBlockState()
    for event in premature_eof_trace():
        consume_messages_event(state, event)
    with pytest.raises(LLMAPIClientError):
        finalize_messages_stream(state)
    native_like = finalize_messages_stream(
        state,
        validate_completion=lambda current: None,
    )
    assert native_like["content"][0]["text"] == "Partial fixture text"
    assert normalize_messages_response(
        native_like, text_normalizer=first_text_block
    ).content == "Partial fixture text"


def test_invalid_initial_tool_input_fails_on_close_without_argument_fragments():
    events = messages_event_trace(
        blocks=[{"type": "tool_use", "id": "t", "name": "lookup", "input": []}],
        tool_argument_fragments={0: []},
    )
    start = next(event for event in events if event.event == "content_block_start")
    start.data["content_block"]["input"] = []
    state = MessagesBlockState()
    with pytest.raises(InvalidToolArgumentsError):
        for event in events:
            consume_messages_event(state, event)
    tolerant_state = MessagesBlockState()
    for event in events:
        consume_messages_event(
            tolerant_state,
            event,
            block_policy=lambda index, block: (
                {**block, "input": {}}
                if block.get("type") == "tool_use" and not isinstance(block.get("input"), dict)
                else block
            ),
        )
    assert tolerant_state.blocks[0]["input"] == {}


def test_two_interleaved_tool_argument_indexes_reconstruct_in_index_order():
    events = [
        SSEEvent("message_start", {"type": "message_start", "message": {
            "id": "m", "type": "message", "role": "assistant", "model": "fixture",
            "content": [], "stop_reason": None,
        }}),
        SSEEvent("content_block_start", {"index": 2, "content_block": {
            "type": "tool_use", "id": "later", "name": "later", "input": {},
        }}),
        SSEEvent("content_block_start", {"index": 0, "content_block": {
            "type": "tool_use", "id": "earlier", "name": "earlier", "input": {},
        }}),
        SSEEvent("content_block_delta", {"index": 2, "delta": {"type": "input_json_delta", "partial_json": '{"n":'}}),
        SSEEvent("content_block_delta", {"index": 0, "delta": {"type": "input_json_delta", "partial_json": '{"n":1}'}}),
        SSEEvent("content_block_delta", {"index": 2, "delta": {"type": "input_json_delta", "partial_json": '2}'}}),
        SSEEvent("content_block_stop", {"index": 2}),
        SSEEvent("content_block_stop", {"index": 0}),
        SSEEvent("message_delta", {"delta": {"stop_reason": "tool_use"}}),
        SSEEvent("message_stop", {"type": "message_stop"}),
    ]
    state = MessagesBlockState()
    for event in events:
        consume_messages_event(state, event)
    response = finalize_messages_stream(state)
    assert [(block["id"], block["input"]) for block in response["content"]] == [
        ("earlier", {"n": 1}), ("later", {"n": 2})
    ]
    response["content"][0]["input"]["n"] = 8
    assert state.blocks[0]["input"] == {"n": 1}


@pytest.mark.parametrize("terminal_change", ["no_stop", "no_message_stop", "open_block"])
def test_generic_finalization_requires_stop_reason_block_closure_and_message_stop(terminal_change):
    events = messages_event_trace()
    if terminal_change == "no_stop":
        events = messages_event_trace(include_stop_reason=False)
    elif terminal_change == "no_message_stop":
        events = messages_event_trace(include_message_stop=False)
    else:
        events = messages_event_trace(close_last_block=False)
    state = MessagesBlockState()
    for event in events:
        consume_messages_event(state, event)
    with pytest.raises(LLMAPIClientError):
        finalize_messages_stream(state)


def test_raw_usage_merges_without_filling_missing_or_mutating_input():
    events = messages_event_trace(usage_start={"input_tokens": 0, "provider": None}, usage_delta={"output_tokens": 2})
    state = MessagesBlockState()
    for event in events:
        consume_messages_event(state, event)
    assert state.usage == {"input_tokens": 0, "provider": None, "output_tokens": 2}
    assert finalize_messages_stream(state)["usage"] == state.usage
    assert "thinking" not in repr(state)


@pytest.mark.parametrize(
    ("start_usage", "delta_usage", "expected", "expected_snapshots"),
    [
        (_MISSING, _MISSING, None, []),
        (None, _MISSING, None, [("message_start", None)]),
        ({"input_tokens": 0}, _MISSING, {"input_tokens": 0}, [("message_start", {"input_tokens": 0})]),
        ({"input_tokens": 0}, None, None, [("message_start", {"input_tokens": 0}), ("message_delta", None)]),
        (_MISSING, {"output_tokens": 0}, {"output_tokens": 0}, [("message_delta", {"output_tokens": 0})]),
        (["invalid"], _MISSING, ["invalid"], [("message_start", ["invalid"])]),
    ],
)
def test_usage_snapshots_preserve_null_zero_invalid_shapes_and_presence(
    start_usage, delta_usage, expected, expected_snapshots
):
    options = {}
    if start_usage is not _MISSING:
        options["usage_start"] = start_usage
    if delta_usage is not _MISSING:
        options["usage_delta"] = delta_usage
    events = messages_event_trace(**options)
    state = MessagesBlockState()
    for event in events:
        consume_messages_event(state, event)
    assert state.usage == expected
    assert state.usage_snapshots == expected_snapshots
    final = finalize_messages_stream(state)
    if expected_snapshots:
        assert final["usage"] == expected
    else:
        assert "usage" not in final


def test_usage_snapshots_are_deep_copied_and_state_repr_hides_sensitive_markers():
    events = messages_event_trace(
        blocks=[
            {"type": "thinking", "thinking": "raw-reasoning-marker"},
            {"type": "tool_use", "id": "t", "name": "lookup", "input": {"secret": "raw-tool-marker"}},
        ],
        usage_start={"provider_payload": ["raw-usage-marker"]},
    )
    state = MessagesBlockState()
    for event in events:
        consume_messages_event(state, event)
    events[0].data["message"]["usage"]["provider_payload"].append("mutated")
    assert state.usage_snapshots[0][1] == {"provider_payload": ["raw-usage-marker"]}
    assert "raw-reasoning-marker" not in repr(state)
    assert "raw-usage-marker" not in repr(state)
    assert "raw-tool-marker" not in repr(state)
    final = finalize_messages_stream(state)
    final["content"][1]["input"]["secret"] = "changed-output"
    assert state.blocks[1]["input"]["secret"] == "raw-tool-marker"


@pytest.mark.parametrize("raw_arguments", ['[1, 2]', '{"unfinished":'])
def test_tool_arguments_must_decode_to_json_object(raw_arguments):
    events = messages_event_trace(
        blocks=[{"type": "tool_use", "id": "t", "name": "lookup", "input": {}}],
        tool_argument_fragments={0: [raw_arguments]},
    )
    state = MessagesBlockState()
    with pytest.raises(InvalidToolArgumentsError):
        for event in events:
            consume_messages_event(state, event)


def test_wrong_delta_is_strict_by_default_and_skippable_through_policy():
    start = SSEEvent("content_block_start", {"index": 0, "content_block": {"type": "tool_use", "id": "t", "name": "f", "input": {}}})
    wrong_delta = SSEEvent("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": "visible-marker"}})
    strict = MessagesBlockState()
    consume_messages_event(strict, start)
    with pytest.raises(LLMAPIClientError):
        consume_messages_event(strict, wrong_delta)
    tolerant = MessagesBlockState()
    consume_messages_event(tolerant, start)
    assert consume_messages_event(tolerant, wrong_delta, delta_policy=lambda block, delta: False) == ("", "")
    assert "visible-marker" not in tolerant.blocks[0]["input"]


def test_bool_indexes_are_rejected_and_malformed_events_can_be_skipped_by_policy():
    start = SSEEvent("content_block_start", {"index": True, "content_block": {"type": "text", "text": ""}})
    with pytest.raises(LLMAPIClientError):
        consume_messages_event(MessagesBlockState(), start)
    malformed = SSEEvent("content_block_start", "malformed-marker")
    assert consume_messages_event(
        MessagesBlockState(), malformed,
        event_policy=lambda event: hasattr(event.data, "get"),
    ) == ("", "")


def test_unmatched_numeric_index_is_strict_or_skipped_by_index_policy():
    orphan = SSEEvent("content_block_delta", {
        "index": 9, "delta": {"type": "text_delta", "text": "orphan"},
    })
    with pytest.raises(LLMAPIClientError):
        consume_messages_event(MessagesBlockState(), orphan)
    state = MessagesBlockState()
    assert consume_messages_event(
        state, orphan, index_policy=lambda index, kind, current: False
    ) == ("", "")
    assert state.blocks == {}


@pytest.mark.parametrize(
    "start_message",
    [
        None,
        {"type": "message", "role": "user", "id": "m", "model": "x"},
        {"type": "message", "role": "assistant", "id": "", "model": "x"},
        {"type": "message", "role": "assistant", "id": "m", "model": ""},
    ],
)
def test_finalization_requires_valid_message_start_evidence(start_message):
    state = MessagesBlockState()
    if start_message is not None:
        from src.llm_api_adapter.llms.transports import SSEEvent

        consume_messages_event(state, SSEEvent("message_start", {"message": start_message}))
    for event in messages_event_trace():
        if event.event != "message_start":
            consume_messages_event(state, event)
    with pytest.raises(LLMAPIClientError):
        finalize_messages_stream(state)


@pytest.mark.parametrize("bad_reason", [None, ""])
def test_finalization_rejects_missing_or_empty_stop_reason(bad_reason):
    state = MessagesBlockState()
    for event in messages_event_trace(stop_reason=bad_reason):
        consume_messages_event(state, event)
    with pytest.raises(LLMAPIClientError):
        finalize_messages_stream(state)


@pytest.mark.parametrize(
    "block",
    [
        {"type": "text", "text": "unfinished visible text"},
        {"type": "tool_use", "id": "t", "name": "lookup", "input": {}},
    ],
)
def test_finalization_rejects_open_text_and_tool_blocks(block):
    state = MessagesBlockState()
    events = messages_event_trace(blocks=[block], close_last_block=False)
    for event in events:
        consume_messages_event(state, event)
    with pytest.raises(LLMAPIClientError):
        finalize_messages_stream(state)


def test_unknown_ping_and_late_usage_do_not_replace_terminal_evidence():
    from src.llm_api_adapter.llms.transports import SSEEvent

    trace = messages_event_trace(
        blocks=[{"type": "text", "text": "ok"}],
        usage_start={"input_tokens": 1},
        usage_delta={"output_tokens": 2},
    )
    trace[-1:-1] = unknown_and_ping_events()
    state = MessagesBlockState()
    for event in trace:
        consume_messages_event(state, event)
    assert state.usage_snapshots == [
        ("message_start", {"input_tokens": 1}),
        ("message_delta", {"output_tokens": 2}),
    ]
    assert finalize_messages_stream(state)["usage"] == {
        "input_tokens": 1, "output_tokens": 2
    }
    incomplete = MessagesBlockState()
    for event in [*unknown_and_ping_events(), SSEEvent(None, done=True)]:
        consume_messages_event(incomplete, event)
    with pytest.raises(LLMAPIClientError):
        finalize_messages_stream(incomplete)


def test_fragmented_tool_json_is_not_accepted_until_complete_object():
    good = messages_event_trace(
        blocks=[{"type": "tool_use", "id": "t", "name": "lookup", "input": {}}],
        tool_argument_fragments={0: ['{"a":', '1}']},
    )
    state = MessagesBlockState()
    for event in good:
        consume_messages_event(state, event)
    assert finalize_messages_stream(state)["content"][0]["input"] == {"a": 1}


def _install_http_stream(monkeypatch, transport, body, *, fail_client_close=False, async_gate=None):
    """Use the public transports with deterministic HTTP-boundary responses."""
    observations = {"calls": [], "closed": []}
    if transport == "requests":
        import requests

        class Response:
            status_code = 200

            def raise_for_status(self):
                pass

            def iter_lines(self):
                yield from body.splitlines()

            def close(self):
                observations["closed"].append("response")

        def post(url, **kwargs):
            observations["calls"].append((url, kwargs))
            return Response()

        monkeypatch.setattr(requests, "post", post)
        return observations

    import httpx

    if transport == "httpx":
        class ByteStream(httpx.SyncByteStream):
            def __iter__(self):
                yield body.encode()

            def close(self):
                observations["closed"].append("response")

        real_client = httpx.Client

        def handler(request):
            observations["calls"].append(request)
            return httpx.Response(
                200, stream=ByteStream(),
                headers={"Content-Type": "text/event-stream"}, request=request,
            )

        class Client:
            def __init__(self):
                self._client = real_client(transport=httpx.MockTransport(handler))

            def __getattr__(self, name):
                return getattr(self._client, name)

            def close(self):
                self._client.close()
                observations["closed"].append("client")
                if fail_client_close:
                    raise RuntimeError("secondary sync HTTPX close failure")

        monkeypatch.setattr(httpx, "Client", Client)
        return observations

    if transport == "async":
        class ByteStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield body.encode()

            async def aclose(self):
                observations["closed"].append("response")

        real_client = httpx.AsyncClient

        def factory():
            async def handler(request):
                observations["calls"].append(request)
                if async_gate is not None:
                    started, release = async_gate
                    started.set()
                    await release.wait()
                return httpx.Response(
                    200, stream=ByteStream(),
                    headers={"Content-Type": "text/event-stream"}, request=request,
                )

            class Client:
                def __init__(self):
                    self._client = real_client(transport=httpx.MockTransport(handler))

                def __getattr__(self, name):
                    return getattr(self._client, name)

                async def aclose(self):
                    await self._client.aclose()
                    observations["closed"].append("client")
                    if fail_client_close:
                        raise RuntimeError("secondary async HTTPX close failure")

            return Client()

        monkeypatch.setattr(httpx, "AsyncClient", factory)
        return observations

    raise AssertionError(f"Unexpected transport fixture: {transport}")


def _adapter_for_transport(transport):
    from src.llm_api_adapter.adapters.anthropic_compatible.adapter import AnthropicCompatibleAdapter

    return AnthropicCompatibleAdapter(
        api_key="synthetic-test-key", model="unregistered-stream-model",
        base_url="https://example.invalid/v1",
        transport="httpx" if transport == "async" else transport,
    )


def _assert_one_request(observations, transport):
    calls = observations["calls"]
    assert len(calls) == 1
    if transport == "requests":
        url, kwargs = calls[0]
        payload = kwargs["json"]
    else:
        request = calls[0]
        url = str(request.url)
        payload = json.loads(request.content)
        assert request.method == "POST"
    assert url == "https://example.invalid/v1/messages"
    assert payload["model"] == "unregistered-stream-model"
    assert payload["stream"] is True


def _error_trace(*, with_content=True):
    events = messages_event_trace(
        blocks=[
            {"type": "text", "text": "abcdefghij"},
            {"type": "text", "text": "tail"},
        ] if with_content else [],
        include_stop_reason=False,
        include_message_stop=False,
    )
    events.append(SSEEvent("error", {
        "type": "error",
        "error": {"type": "api_error", "message": "fixture provider failure"},
    }))
    return "\n".join(events_to_sse_lines(events)) + "\n"


def test_public_requests_provider_error_without_cleanup_failure(monkeypatch):
    _assert_public_provider_error_without_cleanup(monkeypatch, "requests")


def test_public_sync_httpx_provider_error_without_cleanup_failure(monkeypatch):
    _assert_public_provider_error_without_cleanup(monkeypatch, "httpx")


def test_public_async_httpx_provider_error_without_cleanup_failure(monkeypatch):
    asyncio.run(_async_public_provider_error_without_cleanup(monkeypatch, fail_client_close=False))


def _assert_public_provider_error_without_cleanup(monkeypatch, transport):
    from src.llm_api_adapter.errors.llm_api_error import LLMAPIServerError
    from src.llm_api_adapter.models.messages.chat_message import UserMessage

    body = _error_trace()
    observations = _install_http_stream(monkeypatch, transport, body)
    adapter = _adapter_for_transport(transport)
    done, output = [], []
    stream = adapter.stream_chat(
        [UserMessage(content="prompt")], max_tokens=12, buffer_chars=3,
        on_done=done.append,
    )
    error = None
    try:
        output.extend(stream)
    except BaseException as exc:
        error = exc
    assert "".join(output) == "abcdefghijta"
    assert all(len(chunk) <= 3 for chunk in output)
    assert done == []
    _assert_one_request(observations, transport)
    assert observations["closed"] == (
        ["response"] if transport == "requests" else ["response", "client"]
    )
    assert isinstance(error, LLMAPIServerError)
    assert "fixture provider failure" in str(error)


async def _async_public_provider_error_without_cleanup(monkeypatch, *, fail_client_close):
    from src.llm_api_adapter.errors.llm_api_error import LLMAPIServerError
    from src.llm_api_adapter.models.messages.chat_message import UserMessage

    observations = _install_http_stream(
        monkeypatch, "async", _error_trace(), fail_client_close=fail_client_close,
    )
    adapter = _adapter_for_transport("async")
    adapter._create_sync_client = lambda: pytest.fail("async path used a sync client")
    done, output, error = [], [], None
    stream = adapter.astream_chat(
        [UserMessage(content="prompt")], max_tokens=12, buffer_chars=3,
        on_done=done.append,
    )
    try:
        async for chunk in stream:
            output.append(chunk)
    except BaseException as exc:
        error = exc
    assert "".join(output) == "abcdefghijta"
    assert all(len(chunk) <= 3 for chunk in output)
    assert done == []
    _assert_one_request(observations, "async")
    assert observations["closed"] == ["response", "client"]
    assert isinstance(error, LLMAPIServerError)
    assert "fixture provider failure" in str(error)


def test_httpx_cleanup_failure_does_not_mask_provider_failure_sync(monkeypatch):
    from src.llm_api_adapter.errors.llm_api_error import LLMAPIServerError
    from src.llm_api_adapter.models.messages.chat_message import UserMessage

    observations = _install_http_stream(
        monkeypatch, "httpx", _error_trace(), fail_client_close=True,
    )
    adapter = _adapter_for_transport("httpx")
    done, output, error = [], [], None
    try:
        output.extend(adapter.stream_chat(
            [UserMessage(content="prompt")], max_tokens=12, buffer_chars=3,
            on_done=done.append,
        ))
    except BaseException as exc:
        error = exc
    assert "".join(output) == "abcdefghijta"
    assert all(len(chunk) <= 3 for chunk in output)
    assert done == []
    _assert_one_request(observations, "httpx")
    assert observations["closed"] == ["response", "client"]
    assert isinstance(error, LLMAPIServerError)
    assert "fixture provider failure" in str(error)


def test_httpx_cleanup_failure_does_not_mask_provider_failure_async(monkeypatch):
    asyncio.run(_async_public_provider_error_without_cleanup(monkeypatch, fail_client_close=True))


def _success_events():
    return messages_event_trace(blocks=[
        {"type": "text", "text": "visible "},
        {"type": "thinking", "thinking": "private reasoning"},
        {"type": "tool_use", "id": "tool-1", "name": "lookup", "input": {"q": 1}},
        {"type": "text", "text": "end"},
    ])


def _tool_specs():
    from src.llm_api_adapter.models.tools.tool_spec import ToolSpec

    return [ToolSpec(name="lookup", json_schema={"type": "object"})]


def _recording_callbacks(order, *, awaited):
    def make(tag):
        if awaited:
            async def callback(value):
                await asyncio.sleep(0)
                order.append((tag, value))
            return callback
        return lambda value: order.append((tag, value))

    return {
        "on_chunk": make("chunk"),
        "on_delta": make("delta"),
        "on_reasoning": make("reasoning"),
        "on_tool_call": make("tool"),
        "on_done": make("done"),
    }


def test_public_success_bounded_order_across_real_transport_paths(monkeypatch):
    from src.llm_api_adapter.models.messages.chat_message import UserMessage

    cases = [
        ("requests", False), ("httpx", False),
        ("async", False), ("async", True),
    ]
    for transport, awaited in cases:
        # Each case gets a fresh monkeypatch context to keep HTTP clients isolated.
        with pytest.MonkeyPatch.context() as local_patch:
            body = "\n".join(events_to_sse_lines(_success_events())) + "\n"
            observations = _install_http_stream(local_patch, transport, body)
            adapter = _adapter_for_transport(transport)
            order, output = [], []
            callbacks = _recording_callbacks(order, awaited=awaited)
            kwargs = dict(
                max_tokens=20, buffer_chars=4,
                tools=_tool_specs(), capture_reasoning=True, **callbacks,
            )
            if transport == "async":
                adapter._create_sync_client = lambda: pytest.fail("async path used a sync client")

                async def run():
                    async for chunk in adapter.astream_chat(
                        [UserMessage(content="prompt")], **kwargs
                    ):
                        output.append(chunk)
                asyncio.run(run())
            else:
                output.extend(adapter.stream_chat(
                    [UserMessage(content="prompt")], **kwargs
                ))
            assert "".join(output) == "visible end"
            assert all(len(chunk) <= 4 for chunk in output)
            tags = [tag for tag, _ in order]
            tool_at, done_at = tags.index("tool"), tags.index("done")
            visible_callback_indexes = [
                i for i, tag in enumerate(tags) if tag in {"chunk", "delta"}
            ]
            assert max(visible_callback_indexes) < tool_at < done_at
            assert order[tool_at][1].call_id == "tool-1"
            assert order[done_at][1].content == "visible end"
            assert any(tag == "reasoning" for tag in tags)
            _assert_one_request(observations, transport)
            assert observations["closed"] == (
                ["response"] if transport == "requests" else ["response", "client"]
            )


def test_public_close_and_aclose_release_real_http_responses_without_callbacks(monkeypatch):
    from src.llm_api_adapter.models.messages.chat_message import UserMessage

    for transport in ("requests", "httpx", "async"):
        with pytest.MonkeyPatch.context() as local_patch:
            trace = messages_event_trace(blocks=[
                {"type": "text", "text": "first pending tail"},
                {"type": "thinking", "thinking": "later reasoning"},
                {"type": "tool_use", "id": "tool-1", "name": "lookup", "input": {"q": 1}},
            ])
            observations = _install_http_stream(
                local_patch, transport,
                "\n".join(events_to_sse_lines(trace)) + "\n",
            )
            adapter = _adapter_for_transport(transport)
            if transport == "async":
                adapter._create_sync_client = lambda: pytest.fail("async path used a sync client")
            callbacks = []
            done = []
            kwargs = dict(
                max_tokens=12, buffer_chars=4,
                tools=_tool_specs(), capture_reasoning=True,
                on_chunk=lambda value: callbacks.append("chunk"),
                on_delta=lambda value: callbacks.append("delta"),
                on_reasoning=lambda value: callbacks.append("reasoning"),
                on_tool_call=lambda value: callbacks.append("tool"),
                on_done=done.append,
            )
            if transport == "async":
                async def close_async():
                    stream = adapter.astream_chat(
                        [UserMessage(content="prompt")], **kwargs
                    )
                    try:
                        first = await asyncio.wait_for(stream.__anext__(), timeout=2)
                        assert first == "firs"
                    finally:
                        await asyncio.wait_for(stream.aclose(), timeout=2)
                asyncio.run(close_async())
            else:
                stream = adapter.stream_chat(
                    [UserMessage(content="prompt")], **kwargs
                )
                try:
                    assert next(stream) == "firs"
                finally:
                    stream.close()
            assert callbacks == ["chunk", "delta"]
            assert done == []
            _assert_one_request(observations, transport)
            assert observations["closed"] == (
                ["response"] if transport == "requests" else ["response", "client"]
            )


@pytest.mark.parametrize("transport", ["requests", "httpx", "async"])
def test_public_invalid_terminal_never_flushes_pending_text_or_calls_done(monkeypatch, transport):
    from src.llm_api_adapter.errors.llm_api_error import LLMAPIClientError
    from src.llm_api_adapter.models.messages.chat_message import UserMessage

    events = messages_event_trace(
        blocks=[{"type": "text", "text": "visible-tailX"}],
        include_stop_reason=False,
        include_message_stop=False,
    )
    body = "\n".join(events_to_sse_lines(events)) + "\n"
    observations = _install_http_stream(monkeypatch, transport, body)
    adapter = _adapter_for_transport(transport)
    if transport == "async":
        adapter._create_sync_client = lambda: pytest.fail("async path used a sync client")
    done, output, error = [], [], None
    if transport == "async":
        async def run():
            nonlocal error
            try:
                async for chunk in adapter.astream_chat(
                    [UserMessage(content="prompt")], max_tokens=8, buffer_chars=4,
                    on_done=done.append,
                ):
                    output.append(chunk)
            except BaseException as exc:
                error = exc
        asyncio.run(run())
    else:
        try:
            output.extend(adapter.stream_chat(
                [UserMessage(content="prompt")], max_tokens=8, buffer_chars=4,
                on_done=done.append,
            ))
        except BaseException as exc:
            error = exc
    assert "".join(output) == "visible-tail"
    assert all(len(chunk) <= 4 for chunk in output)
    assert done == [] and isinstance(error, LLMAPIClientError)
    _assert_one_request(observations, transport)
    assert observations["closed"] == (
        ["response"] if transport == "requests" else ["response", "client"]
    )


@pytest.mark.parametrize(
    ("transport", "callback_name"),
    [(mode, callback) for mode in ("requests", "async") for callback in (
        "on_delta", "on_chunk", "on_reasoning", "on_tool_call", "on_done",
    )],
)
def test_public_callback_failures_preserve_identity_and_stop_success_callbacks(
    monkeypatch, transport, callback_name,
):
    from src.llm_api_adapter.models.messages.chat_message import UserMessage

    body = "\n".join(events_to_sse_lines(_success_events())) + "\n"
    observations = _install_http_stream(monkeypatch, transport, body)
    adapter = _adapter_for_transport(transport)
    if transport == "async":
        adapter._create_sync_client = lambda: pytest.fail("async path used a sync client")
    callback_error = ValueError(f"caller callback failure: {callback_name}")
    attempts, done_attempts, output, seen_error = [], [], [], None

    def failing(value):
        attempts.append(value)
        if callback_name == "on_done":
            done_attempts.append(value)
        raise callback_error

    callbacks = {name: (done_attempts.append if name == "on_done" else lambda _value: None)
                 for name in ("on_delta", "on_chunk", "on_reasoning", "on_tool_call", "on_done")}
    callbacks[callback_name] = failing
    args = dict(
        max_tokens=16, buffer_chars=4,
        tools=_tool_specs(), capture_reasoning=True, **callbacks,
    )
    if transport == "async":
        async def run():
            nonlocal seen_error
            try:
                async for chunk in adapter.astream_chat(
                    [UserMessage(content="prompt")], **args
                ):
                    output.append(chunk)
            except BaseException as exc:
                seen_error = exc
        asyncio.run(run())
    else:
        try:
            output.extend(adapter.stream_chat(
                [UserMessage(content="prompt")], **args
            ))
        except BaseException as exc:
            seen_error = exc
    assert seen_error is callback_error
    assert len(attempts) == 1
    assert len(done_attempts) == (1 if callback_name == "on_done" else 0)
    _assert_one_request(observations, transport)
    assert observations["closed"] == (
        ["response", "client"] if transport == "async" else ["response"]
    )


def test_async_awaited_callback_failure_keeps_identity_over_httpx_close_failure(monkeypatch):
    from src.llm_api_adapter.models.messages.chat_message import UserMessage

    async def scenario():
        body = "\n".join(events_to_sse_lines(_success_events())) + "\n"
        observations = _install_http_stream(
            monkeypatch, "async", body, fail_client_close=True,
        )
        adapter = _adapter_for_transport("async")
        adapter._create_sync_client = lambda: pytest.fail("async path used a sync client")
        callback_error = ValueError("awaited async callback marker")
        calls, done, error = [], [], None

        async def failing_after_pause(_value):
            calls.append("on_delta")
            await asyncio.sleep(0)
            raise callback_error

        async def record_done(value):
            done.append(value)

        stream = adapter.astream_chat(
            [UserMessage(content="prompt")], max_tokens=16, buffer_chars=4,
            on_delta=failing_after_pause, on_done=record_done,
        )
        try:
            async for _chunk in stream:
                pass
        except BaseException as exc:
            error = exc
        _assert_one_request(observations, "async")
        assert observations["closed"] == ["response", "client"]
        assert calls == ["on_delta"] and done == []
        assert error is callback_error

    asyncio.run(scenario())


def test_async_callback_pause_cancellation_keeps_primary_over_source_cleanup():
    from src.llm_api_adapter.adapters.anthropic_compatible.adapter import AnthropicCompatibleAdapter
    from src.llm_api_adapter.models.messages.chat_message import UserMessage
    from tests.fixtures.anthropic_compatible import ControlledAsyncSource

    async def scenario():
        adapter = AnthropicCompatibleAdapter(
            api_key="synthetic-test-key", model="fixture-model",
            base_url="https://example.invalid/v1", transport="httpx",
        )
        source = ControlledAsyncSource(
            messages_event_trace(blocks=[{"type": "text", "text": "visible tail"}]),
            cleanup_error=RuntimeError("secondary source cleanup"),
        )

        class Client:
            def _headers(self): return {}
            def _execute_prepared_stream(self, _request): return source

        adapter._create_async_client = lambda: Client()
        entered, release = asyncio.Event(), asyncio.Event()
        done = []

        async def paused(_chunk):
            entered.set()
            await release.wait()

        stream = adapter.astream_chat(
            [UserMessage(content="prompt")], max_tokens=8, buffer_chars=1,
            on_delta=paused, on_done=done.append,
        )
        pending = asyncio.create_task(stream.__anext__())
        try:
            await asyncio.wait_for(entered.wait(), timeout=2)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(pending, timeout=2)
        finally:
            release.set()
            if not pending.done():
                pending.cancel()
                try:
                    await pending
                except asyncio.CancelledError:
                    pass
            try:
                await stream.aclose()
            except BaseException:
                pass
        assert source.aclose_calls >= 1
        assert done == []

    asyncio.run(scenario())


def test_async_httpx_cancellation_without_cleanup_error_preserves_cancelled_error(monkeypatch):
    asyncio.run(_httpx_io_cancellation_case(monkeypatch, fail_client_close=False))


def test_async_httpx_cancellation_with_cleanup_error_preserves_cancelled_error(monkeypatch):
    asyncio.run(_httpx_io_cancellation_case(monkeypatch, fail_client_close=True))


async def _httpx_io_cancellation_case(monkeypatch, *, fail_client_close):
    from src.llm_api_adapter.models.messages.chat_message import UserMessage

    started, release = asyncio.Event(), asyncio.Event()
    observations = _install_http_stream(
        monkeypatch, "async", "", fail_client_close=fail_client_close,
        async_gate=(started, release),
    )
    adapter = _adapter_for_transport("async")
    adapter._create_sync_client = lambda: pytest.fail("async path used a sync client")
    done, error = [], None
    stream = adapter.astream_chat(
        [UserMessage(content="prompt")], max_tokens=8, on_done=done.append,
    )
    pending = asyncio.create_task(stream.__anext__())
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        pending.cancel()
        try:
            await asyncio.wait_for(pending, timeout=2)
        except BaseException as exc:
            error = exc
    finally:
        release.set()
        if not pending.done():
            pending.cancel()
            try:
                await pending
            except BaseException:
                pass
        try:
            await stream.aclose()
        except BaseException:
            pass
    # Inspect observations before the expected cancellation-class assertion so
    # the precedence test remains diagnostic when it is red.
    _assert_one_request(observations, "async")
    assert observations["closed"] == ["client"]
    assert done == []
    assert isinstance(error, asyncio.CancelledError)


def test_cleanup_failure_surfaces_without_flushing_tail_or_done():
    from src.llm_api_adapter.adapters.anthropic_compatible.adapter import AnthropicCompatibleAdapter
    from src.llm_api_adapter.models.messages.chat_message import UserMessage
    from tests.fixtures.anthropic_compatible import ControlledSyncSource

    source = ControlledSyncSource(
        messages_event_trace(blocks=[{"type": "text", "text": "complete-tailX"}]),
        cleanup_error=RuntimeError("cleanup-only marker"),
    )

    class Client:
        def _headers(self): return {}
        def _execute_prepared_stream(self, _request): return source

    adapter = AnthropicCompatibleAdapter(
        api_key="synthetic-test-key", model="fixture-model",
        base_url="https://example.invalid/v1", transport="requests",
    )
    adapter._create_sync_client = lambda: Client()
    done, output = [], []
    error = None
    try:
        output.extend(adapter.stream_chat(
            [UserMessage(content="prompt")], max_tokens=8, buffer_chars=4,
            on_done=done.append,
        ))
    except BaseException as exc:
        error = exc
    assert "".join(output) == "complete-tai"
    assert all(len(chunk) <= 4 for chunk in output)
    assert isinstance(error, LLMAPIClientError)
    assert source.close_calls >= 1 and done == []


def test_native_public_stream_keeps_first_text_policy_and_eof_tolerance():
    from src.llm_api_adapter.adapters.anthropic.adapter import AnthropicAdapter
    from src.llm_api_adapter.models.messages.chat_message import UserMessage

    native = AnthropicAdapter(api_key="synthetic-test-key", model="claude-sonnet-4-5")
    trace = messages_event_trace(
        blocks=[{"type": "text", "text": "first"}, {"type": "text", "text": "second"}],
        include_stop_reason=False,
        include_message_stop=False,
        close_last_block=False,
    )

    class Client:
        def stream(self, **_kwargs):
            return iter(trace)

    native._create_sync_client = lambda: Client()
    done = []
    output = list(native.stream_chat(
        [UserMessage(content="prompt")], max_tokens=16,
        on_done=done.append,
    ))
    assert "".join(output) == "firstsecond"
    assert done[0].content == "first"


@pytest.mark.parametrize("late_kind", ["text", "tool"])
@pytest.mark.parametrize("transport", ["requests", "httpx", "async"])
def test_public_generic_stream_rejects_delta_after_closed_block(
    monkeypatch, late_kind, transport
):
    from src.llm_api_adapter.models.messages.chat_message import UserMessage

    blocks = [{"type": "text", "text": "abcdef"}]
    if late_kind == "tool":
        blocks.append({
            "type": "tool_use", "id": "tool-1", "name": "lookup", "input": {"n": 1},
        })
    events = messages_event_trace(blocks=blocks)
    late_index = 0 if late_kind == "text" else 1
    late_delta = (
        {"type": "text_delta", "text": "late"}
        if late_kind == "text"
        else {"type": "input_json_delta", "partial_json": "not-json"}
    )
    stop_position = next(
        index for index, event in enumerate(events)
        if event.event == "content_block_stop" and event.data["index"] == late_index
    )
    events.insert(stop_position + 1, SSEEvent(
        "content_block_delta", {"index": late_index, "delta": late_delta},
    ))
    body = "\n".join(events_to_sse_lines(events)) + "\n"
    observations = _install_http_stream(monkeypatch, transport, body)
    adapter = _adapter_for_transport(transport)
    if transport == "async":
        adapter._create_sync_client = lambda: pytest.fail("async path used a sync client")
    order, output, error = [], [], None
    kwargs = dict(
        max_tokens=12, buffer_chars=4, tools=_tool_specs(),
        on_chunk=lambda value: order.append(("chunk", value)),
        on_tool_call=lambda value: order.append(("tool", value)),
        on_done=lambda value: order.append(("done", value)),
    )
    if transport == "async":
        async def run():
            nonlocal error
            try:
                async for chunk in adapter.astream_chat(
                    [UserMessage(content="prompt")], **kwargs
                ):
                    output.append(chunk)
            except BaseException as exc:
                error = exc
        asyncio.run(run())
    else:
        try:
            output.extend(adapter.stream_chat(
                [UserMessage(content="prompt")], **kwargs
            ))
        except BaseException as exc:
            error = exc

    assert isinstance(error, LLMAPIClientError)
    assert "".join(output) == "abcd"
    assert [tag for tag, _ in order] == ["chunk"]
    assert order[0][1].text == "abcd"
    _assert_one_request(observations, transport)
    assert observations["closed"] == (
        ["response"] if transport == "requests" else ["response", "client"]
    )


def test_native_public_stream_retains_closed_tool_delta_tolerance():
    from src.llm_api_adapter.adapters.anthropic.adapter import AnthropicAdapter
    from src.llm_api_adapter.models.messages.chat_message import UserMessage

    trace = messages_event_trace(blocks=[
        {"type": "text", "text": "abcdef"},
        {"type": "tool_use", "id": "tool-1", "name": "lookup", "input": {"n": 1}},
    ])
    stop_position = next(
        index for index, event in enumerate(trace)
        if event.event == "content_block_stop" and event.data["index"] == 1
    )
    trace.insert(stop_position + 1, SSEEvent(
        "content_block_delta", {
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": "not-json"},
        },
    ))

    class Client:
        def stream(self, **_kwargs):
            return iter(trace)

    native = AnthropicAdapter(api_key="synthetic-test-key", model="claude-sonnet-4-5")
    native._create_sync_client = lambda: Client()
    order, output = [], []
    output.extend(native.stream_chat(
        [UserMessage(content="prompt")], max_tokens=12,
        tools=_tool_specs(),
        on_tool_call=lambda value: order.append(("tool", value)),
        on_done=lambda value: order.append(("done", value)),
    ))
    assert "".join(output) == "abcdef"
    assert [tag for tag, _ in order] == ["tool", "done"]
    assert order[0][1].arguments == {"n": 1}
