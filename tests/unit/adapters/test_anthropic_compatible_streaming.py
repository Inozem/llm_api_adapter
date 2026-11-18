from copy import deepcopy

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
from tests.fixtures.anthropic_compatible import messages_event_trace, premature_eof_trace

pytestmark = pytest.mark.unit
_MISSING = object()


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
