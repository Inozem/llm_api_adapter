import json

import pytest

from src.llm_api_adapter.llms.transports import SSEFrameDecoder
from src.llm_api_adapter.models.responses.chat_response import ChatResponse
from tests.fixtures.openai_compatible import (
    incomplete_response,
    invalid_structured_result_response,
    invalid_usage_payload,
    partial_usage_payload,
    portable_schema,
    premature_eof_sse_lines,
    refusal_response,
    structured_result_response,
    text_response,
    text_sse_lines,
    tool_call_response,
    tool_call_sse_lines,
    usage_payload,
    zero_usage_payload,
)


def decode_lines(lines):
    decoder = SSEFrameDecoder()
    return [event for line in lines if (event := decoder.feed(line)) is not None]


@pytest.mark.unit
def test_response_factories_normalize_text_tools_structured_refusal_and_incomplete():
    text = ChatResponse.from_openai_response(text_response("hello"))
    assert text.content == "hello"

    call = ChatResponse.from_openai_response(tool_call_response())
    assert call.content is None
    assert call.tool_calls[0].name == "lookup"
    assert call.tool_calls[0].arguments == {"query": "fixture"}
    assert call.tool_calls[0].call_id == "call_fixture_001"

    structured = ChatResponse.from_openai_response(structured_result_response())
    assert json.loads(structured.content) == {"answer": "ok"}

    refused = ChatResponse.from_openai_response(refusal_response())
    assert refused.refusal == "Fixture refusal"
    incomplete = ChatResponse.from_openai_response(incomplete_response())
    assert incomplete.incomplete_reason == "length"

    malformed = ChatResponse.from_openai_response(invalid_structured_result_response())
    assert malformed.content == '{"contact": '


@pytest.mark.unit
def test_usage_factories_preserve_absent_null_partial_zero_and_invalid_values():
    assert "usage" not in text_response()
    assert text_response(usage=None)["usage"] is None
    assert text_response(usage="invalid-shape")["usage"] == "invalid-shape"
    assert partial_usage_payload(prompt_tokens=7) == {"prompt_tokens": 7}
    assert usage_payload(total_tokens=999, prompt_tokens=1)["total_tokens"] == 999
    assert zero_usage_payload() == {
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
    }
    for invalid in (True, -1, 1.5, "4", None):
        assert invalid_usage_payload(invalid)["prompt_tokens"] == invalid

    zero = ChatResponse.from_openai_response(text_response(usage=zero_usage_payload()))
    assert zero.usage.input_tokens == zero.usage.output_tokens == zero.usage.total_tokens == 0
    missing = ChatResponse.from_openai_response(text_response())
    assert missing.usage is None


@pytest.mark.unit
def test_sse_factories_preserve_fragments_and_terminal_usage_evidence():
    text_events = decode_lines(text_sse_lines(
        "hello world", fragments=["hello ", "world"], usage=zero_usage_payload(),
    ))
    chunks = [event.data for event in text_events
              if isinstance(event.data, dict) and event.data.get("choices")]
    assert [chunk["choices"][0]["delta"]["content"] for chunk in chunks[:-1]] == [
        "hello ", "world",
    ]
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    usage_event = next(event.data for event in text_events
                       if isinstance(event.data, dict) and event.data.get("choices") == [])
    assert usage_event["usage"] == zero_usage_payload()
    assert text_events[-1].done is True

    tool_events = decode_lines(tool_call_sse_lines(
        "lookup", '{"query":"fixture"}', name_fragments=["look", "up"],
        argument_fragments=['{"query":', '"fixture"}'],
    ))
    deltas = [event.data["choices"][0]["delta"] for event in tool_events
              if isinstance(event.data, dict) and event.data.get("choices")
              and event.data["choices"][0]["delta"].get("tool_calls")]
    calls = [delta["tool_calls"][0] for delta in deltas]
    assert "".join(call["function"].get("name", "") for call in calls) == "lookup"
    assert "".join(call["function"].get("arguments", "") for call in calls) == '{"query":"fixture"}'
    assert any(event.data["choices"][0]["finish_reason"] == "tool_calls"
               for event in tool_events
               if isinstance(event.data, dict) and event.data.get("choices"))

    no_usage_events = decode_lines(text_sse_lines("ok"))
    assert not any(isinstance(event.data, dict) and event.data.get("choices") == []
                   for event in no_usage_events)


@pytest.mark.unit
def test_premature_eof_and_mutable_inputs_remain_distinct_and_isolated():
    partial_lines = premature_eof_sse_lines(
        "part", content_fragments=["part"], tool_name="lookup",
    )
    partial_events = decode_lines(partial_lines)
    assert partial_events
    assert not any(event.done for event in partial_events)
    assert not any(
        event.data.get("choices")
        and event.data["choices"][0].get("finish_reason") is not None
        for event in partial_events
    )

    original = {"prompt_tokens": 3, "detail": {"cached_tokens": 2}}
    first = text_response(usage=original)
    second = text_response(usage=original)
    first["usage"]["detail"]["cached_tokens"] = 99
    original["detail"]["cached_tokens"] = 77
    assert second["usage"]["detail"]["cached_tokens"] == 2

    schema = portable_schema("flat_object")
    schema["properties"]["answer"]["type"] = "integer"
    assert portable_schema("flat_object")["properties"]["answer"]["type"] == "string"

    first_lines = text_sse_lines("same")
    second_lines = text_sse_lines("same")
    first_lines[0] = "changed"
    assert second_lines[0].startswith("data: {")
