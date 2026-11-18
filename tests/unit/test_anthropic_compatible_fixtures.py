import pytest

from src.llm_api_adapter.llms.transports import SSEEvent, SSEFrameDecoder
from src.llm_api_adapter.models.responses.chat_response import ChatResponse
from tests.fixtures.anthropic_compatible import (
    ControlledAsyncCallback,
    ControlledAsyncSource,
    ControlledCallback,
    ControlledSyncSource,
    events_to_sse_lines,
    incomplete_response,
    malformed_response,
    malformed_terminal_events,
    messages_event_trace,
    messages_response,
    multiple_blocks_response,
    premature_eof_trace,
    refusal_response,
    text_response,
    tool_response,
    transport_done_events,
    unknown_and_ping_events,
    usage_payload,
)


def decode_lines(lines):
    decoder = SSEFrameDecoder()
    return [event for line in lines if (event := decoder.feed(line)) is not None]


@pytest.mark.unit
def test_raw_messages_factories_keep_provider_envelopes_and_existing_response_factory():
    raw = text_response("hello", usage=usage_payload(input_tokens=3, output_tokens=2))
    assert (raw["type"], raw["role"], raw["model"], raw["stop_reason"]) == (
        "message", "assistant", "fixture-messages-model", "end_turn",
    )
    normalized = ChatResponse.from_anthropic_response(raw)
    assert normalized.content == "hello"
    assert normalized.usage.input_tokens == 3
    assert normalized.usage.output_tokens == 2

    multi = multiple_blocks_response()
    assert [block["type"] for block in multi["content"]] == ["text", "tool_use", "thinking"]
    assert ChatResponse.from_anthropic_response(multi, capture_reasoning=True).content == "Visible fixture text"
    assert ChatResponse.from_anthropic_response(multi, capture_reasoning=True).tool_calls[0].call_id == "toolu_fixture_1"
    assert ChatResponse.from_anthropic_response(multi, capture_reasoning=True).reasoning_events

    tool = ChatResponse.from_anthropic_response(tool_response())
    assert tool.tool_calls[0].name == "lookup"
    assert tool.tool_calls[0].arguments == {"query": "fixture"}
    assert tool.tool_calls[0].call_id == "toolu_fixture_1"
    assert ChatResponse.from_anthropic_response(refusal_response()).refusal == "Synthetic refusal"
    assert ChatResponse.from_anthropic_response(incomplete_response()).incomplete_reason == "max_tokens"

    tuple_blocks = ({"type": "text", "text": "tuple"}, {
        "type": "tool_use", "id": "toolu_tuple", "name": "lookup", "input": {"nested": [1]},
    })
    tuple_response = messages_response(content=tuple_blocks)
    assert isinstance(tuple_response["content"], list)
    tuple_response["content"][1]["input"]["nested"].append(2)
    assert tuple_blocks[1]["input"]["nested"] == [1]


@pytest.mark.unit
def test_usage_and_mutable_inputs_keep_raw_distinctions_without_sharing():
    assert "usage" not in text_response()
    assert text_response(usage=None)["usage"] is None
    assert text_response(usage={})["usage"] == {}
    assert text_response(usage=usage_payload(input_tokens=0, output_tokens=0))["usage"] == {
        "input_tokens": 0, "output_tokens": 0,
    }
    extension = {"state": ["raw"]}
    raw_usage = usage_payload(
        input_tokens=5,
        output_tokens=4,
        cache_read_input_tokens=2,
        cache_read_tokens=8,
        cache_creation_input_tokens=1,
        provider_extension=extension,
    )
    partial = raw_usage
    assert partial["input_tokens"] == 5
    assert partial["output_tokens"] == 4
    assert partial["cache_read_input_tokens"] == 2
    assert partial["cache_read_tokens"] == 8
    assert partial["cache_creation_input_tokens"] == 1
    assert "total_tokens" not in partial
    assert partial["provider_extension"] == {"state": ["raw"]}
    for invalid in (None, True, -1, 1.5, "4"):
        assert usage_payload(input_tokens=invalid)["input_tokens"] == invalid

    first_usage = usage_payload(provider_extension=extension)
    second_usage = usage_payload(provider_extension=extension)
    first_usage["provider_extension"]["state"].append("first-only")
    extension["state"].append("caller-only")
    assert second_usage["provider_extension"] == {"state": ["raw"]}
    original = {"input_tokens": 5, "details": {"cache_read_input_tokens": 2}}
    first = messages_response(usage=original)
    second = messages_response(usage=original)
    first["usage"]["details"]["cache_read_input_tokens"] = 99
    original["details"]["cache_read_input_tokens"] = 77
    assert second["usage"]["details"]["cache_read_input_tokens"] == 2
    assert malformed_response(usage={"input_tokens": True})["usage"]["input_tokens"] is True

    caller_blocks = [
        {"type": "text", "text": "before"},
        {"type": "tool_use", "id": "toolu_isolated", "name": "lookup", "input": {"nested": [1]}},
    ]
    first_blocks = messages_response(content=caller_blocks)
    second_blocks = messages_response(content=caller_blocks)
    first_blocks["content"][0]["text"] = "changed"
    first_blocks["content"][1]["input"]["nested"].append(2)
    caller_blocks[1]["input"]["nested"].append(3)
    assert second_blocks["content"][0]["text"] == "before"
    assert second_blocks["content"][1]["input"]["nested"] == [1]

    fragments = ["{\"nested\":", "[1]}"]
    start_usage = {"input_tokens": 5, "nested": {"count": 1}}
    late_usage = {"output_tokens": 2, "nested": {"count": 2}}
    trace_args = {
        "blocks": [{"type": "tool_use", "id": "toolu_stream", "name": "lookup", "input": {}}],
        "tool_argument_fragments": {0: fragments},
        "usage_start": start_usage,
        "usage_delta": late_usage,
    }
    first_trace = messages_event_trace(**trace_args)
    second_trace = messages_event_trace(**trace_args)
    first_trace[0].data["message"]["usage"]["nested"]["count"] = 99
    first_trace[1].data["content_block"]["id"] = "changed"
    first_fragment = next(event for event in first_trace if event.event == "content_block_delta")
    first_fragment.data["delta"]["partial_json"] = "changed"
    fragments[0] = "mutated caller fragment"
    start_usage["nested"]["count"] = 77
    late_usage["nested"]["count"] = 88
    assert second_trace[0].data["message"]["usage"]["nested"]["count"] == 1
    assert second_trace[1].data["content_block"]["id"] == "toolu_stream"
    assert next(event for event in second_trace if event.event == "content_block_delta").data[
        "delta"
    ]["partial_json"] == '{"nested":'
    late = next(event for event in second_trace if event.event == "message_delta")
    assert late.data["usage"]["nested"]["count"] == 2


@pytest.mark.unit
def test_complete_event_trace_roundtrips_through_shared_sse_decoder():
    blocks = [
        {"type": "text", "text": "hello world"},
        {"type": "thinking", "thinking": "synthetic", "signature": "signature_fixture"},
        {"type": "redacted_thinking", "data": "redacted_fixture"},
        {"type": "tool_use", "id": "toolu_ordered", "name": "lookup", "input": {"query": "fixture"}},
    ]
    events = messages_event_trace(
        blocks,
        text_fragments={0: ["hello ", "world"], 1: ["synthetic"]},
        tool_argument_fragments={3: ['{"query":', '"fixture"}']},
        usage_start={"input_tokens": 7},
        usage_delta={"output_tokens": 4, "cache_read_input_tokens": 2},
    )
    decoded = decode_lines(events_to_sse_lines(events))
    assert decoded == events
    tool_start = next(e.data["content_block"] for e in decoded
                      if e.event == "content_block_start" and e.data["index"] == 3)
    assert (tool_start["id"], tool_start["name"]) == ("toolu_ordered", "lookup")
    fragments = [e.data["delta"]["partial_json"] for e in decoded
                 if e.event == "content_block_delta" and e.data["index"] == 3]
    assert fragments == ['{"query":', '"fixture"}']
    assert "hello world" not in "".join(fragments)
    assert decoded[0].data["message"]["usage"] == {"input_tokens": 7}
    assert next(e.data["usage"] for e in decoded if e.event == "message_delta") == {
        "output_tokens": 4, "cache_read_input_tokens": 2,
    }
    assert decoded[-1] == SSEEvent("message_stop", {"type": "message_stop"}, done=False)
    assert all(event.event != "content_block_stop" or event.data["index"] in range(4)
               for event in decoded)
    assert sum(event.event == "content_block_stop" for event in decoded) == len(blocks)
    assert all(not event.done for event in decoded)
    without_usage = messages_event_trace(include_start_usage=False)
    assert "usage" not in without_usage[0].data["message"]
    with_null_usage = messages_event_trace(usage_start=None)
    assert with_null_usage[0].data["message"]["usage"] is None
    no_synthetic_usage = messages_event_trace()
    assert "usage" not in no_synthetic_usage[0].data["message"]
    assert "usage" not in next(event.data for event in no_synthetic_usage
                               if event.event == "message_delta")

    refusal_stream = decode_lines(events_to_sse_lines(messages_event_trace(
        [{"type": "text", "text": "Synthetic refusal"}], stop_reason="refusal",
    )))
    incomplete_stream = decode_lines(events_to_sse_lines(messages_event_trace(
        [{"type": "text", "text": "Partial fixture answer"}], stop_reason="max_tokens",
    )))
    assert next(event.data["delta"]["stop_reason"] for event in refusal_stream
                if event.event == "message_delta") == "refusal"
    assert next(event.data["delta"]["stop_reason"] for event in incomplete_stream
                if event.event == "message_delta") == "max_tokens"

    explicit_null_terminal = SSEEvent("message_stop", None)
    assert decode_lines(events_to_sse_lines([explicit_null_terminal])) == [explicit_null_terminal]
    assert decode_lines(events_to_sse_lines(transport_done_events())) == [SSEEvent(None, done=True)]


@pytest.mark.unit
def test_negative_terminals_and_malformed_payloads_remain_distinct_raw_evidence():
    missing_stop = messages_event_trace(include_message_stop=False)
    missing_reason = messages_event_trace(include_stop_reason=False)
    open_block = messages_event_trace(close_last_block=False)
    malformed_terminal = malformed_terminal_events()
    done_only = transport_done_events()
    assert not any(event.event == "message_stop" for event in missing_stop)
    assert "stop_reason" not in next(e.data for e in missing_reason if e.event == "message_delta")["delta"]
    assert not any(event.event == "content_block_stop" and event.data["index"] == 2 for event in open_block)
    assert malformed_terminal[-1].data == "malformed terminal"
    assert done_only == [SSEEvent(None, done=True)]
    decoded_done = decode_lines(events_to_sse_lines(done_only))
    assert decoded_done == [SSEEvent(None, done=True)]
    premature = premature_eof_trace()
    assert [event.data["delta"].get("type") for event in premature
            if event.event == "content_block_delta"] == ["text_delta", "input_json_delta"]
    assert not any(event.event in {"message_delta", "message_stop"} for event in premature)
    assert [event.data["index"] for event in premature
            if event.event == "content_block_stop"] == [0]
    normal_prefix = messages_event_trace(
        [
            {"type": "text", "text": "Partial fixture text"},
            {"type": "tool_use", "id": "toolu_partial", "name": "lookup", "input": {}},
        ],
        text_fragments={0: ["Partial fixture text"]},
        tool_argument_fragments={1: ['{"unfinished":']},
    )[:len(premature)]
    assert premature == normal_prefix
    assert malformed_response(type="error", content="not-an-array")["type"] == "error"
    assert [event.event for event in unknown_and_ping_events()] == [
        "ping", "future_fixture_event",
    ]

    partial_tool = messages_event_trace(
        [{"type": "tool_use", "id": "toolu_partial", "name": "lookup", "input": {}}],
        tool_argument_fragments={0: ['{"unfinished":']},
        include_message_stop=False,
        include_stop_reason=False,
        close_last_block=False,
    )
    assert [e.data["delta"]["partial_json"] for e in partial_tool if e.event == "content_block_delta"] == [
        '{"unfinished":',
    ]
    assert not any(e.event == "message_stop" for e in partial_tool)


@pytest.mark.unit
def test_controlled_sync_source_and_callback_preserve_failure_and_close_ownership():
    source_error = RuntimeError("source fixture")
    cleanup_error = RuntimeError("cleanup fixture")
    source = ControlledSyncSource(["partial", "later"], failure=source_error,
                                  fail_after=1, cleanup_error=cleanup_error)
    assert next(source) == "partial"
    with pytest.raises(RuntimeError) as raised:
        next(source)
    assert raised.value is source_error
    assert source.close_calls == 0
    with pytest.raises(RuntimeError) as close_raised:
        source.close()
    assert close_raised.value is cleanup_error
    assert source.close_calls == 1

    before_output = ControlledSyncSource(["unused"], failure=source_error)
    with pytest.raises(RuntimeError) as before_raised:
        next(before_output)
    assert before_raised.value is source_error
    assert before_output.yielded == 0
    assert before_output.close_calls == 0
    exhausted = ControlledSyncSource(["only value"])
    assert next(exhausted) == "only value"
    with pytest.raises(StopIteration):
        next(exhausted)
    assert exhausted.close_calls == 0

    callback_error = RuntimeError("callback fixture")
    callback = ControlledCallback(callback_error, fail_on_call=2)
    callback("first")
    with pytest.raises(RuntimeError) as callback_raised:
        callback("second")
    assert callback_raised.value is callback_error
    assert callback.calls == ["first", "second"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_controlled_async_source_and_awaitable_callback_keep_explicit_lifecycle():
    source_error = RuntimeError("async source fixture")
    cleanup_error = RuntimeError("async cleanup fixture")
    source = ControlledAsyncSource(["partial", "later"], failure=source_error,
                                   fail_after=1, cleanup_error=cleanup_error)
    assert await source.__anext__() == "partial"
    with pytest.raises(RuntimeError) as raised:
        await source.__anext__()
    assert raised.value is source_error
    assert source.aclose_calls == 0
    with pytest.raises(RuntimeError) as close_raised:
        await source.aclose()
    assert close_raised.value is cleanup_error
    assert source.aclose_calls == 1

    callback_error = RuntimeError("async callback fixture")
    callback = ControlledAsyncCallback(callback_error, fail_on_call=2)
    await callback("partial")
    with pytest.raises(RuntimeError) as callback_raised:
        await callback("failure boundary")
    assert callback_raised.value is callback_error
    assert callback.calls == ["partial", "failure boundary"]

