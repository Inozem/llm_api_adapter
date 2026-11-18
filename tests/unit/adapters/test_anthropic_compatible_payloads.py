import pytest

from src.llm_api_adapter.adapters.anthropic_compatible.payloads import (
    build_generic_messages_payload,
    build_messages_payload,
    normalize_messages_response,
)
from src.llm_api_adapter.models.messages.chat_message import (
    AIMessage, Messages, Prompt, ToolMessage, UserMessage,
)
from src.llm_api_adapter.models.responses.chat_response import Usage
from src.llm_api_adapter.models.tools.tool_spec import ToolSpec
from src.llm_api_adapter.models.tools.tool_call import ToolCall
from tests.fixtures.anthropic_compatible import multiple_blocks_response
from tests.unit.adapters.anthropic_compatible_helpers import first_text_block

pytestmark = pytest.mark.unit


def test_payload_assembles_messages_tools_schema_and_explicit_sampling():
    schema = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
    payload = build_messages_payload(
        model="arbitrary-model",
        messages=Messages([Prompt("system"), UserMessage("hello")]),
        max_tokens=321,
        tools=[ToolSpec("lookup", {"type": "object"}, "find")],
        tool_choice="lookup",
        parallel_tool_calls=False,
        response_schema=schema,
        temperature=0.4,
        top_p=0.8,
    )
    assert payload["model"] == "arbitrary-model"
    assert payload["system"] == "system"
    assert payload["messages"] == [{"role": "user", "content": "hello"}]
    assert payload["tools"][0]["name"] == "lookup"
    assert payload["tool_choice"] == {"type": "tool", "name": "lookup"}
    assert payload["disable_parallel_tool_use"] is True
    assert payload["output_config"]["format"]["schema"] == {
        "type": "object", "properties": {}, "required": [], "additionalProperties": False
    }
    payload["output_config"]["format"]["schema"]["properties"]["new"] = {"type": "string"}
    assert schema["properties"] == {}
    assert payload["temperature"] == 0.4 and payload["top_p"] == 0.8
    assert "timeout_s" not in payload and "budget_tokens" not in payload


def test_sampling_policy_can_omit_generic_defaults_or_preserve_native_defaults():
    common = dict(
        model="m", messages=Messages([UserMessage("hi")]), max_tokens=10,
        temperature=1.0, top_p=1.0,
    )
    assert "temperature" not in build_generic_messages_payload(**common)
    assert build_generic_messages_payload(**{**common, "temperature": True})["temperature"] is True
    native_like = build_messages_payload(**common)
    assert native_like["temperature"] == 1.0 and native_like["top_p"] == 1.0


def test_mixed_assistant_tool_result_history_retains_tool_identity_and_isolation():
    messages = Messages([
        UserMessage("find something"),
        AIMessage("", tool_calls=[ToolCall("lookup", {"q": "x"}, "call-7")]),
        ToolMessage('{"ok": true}', tool_call_id="call-7"),
    ])
    payload = build_messages_payload(model="m", messages=messages, max_tokens=12)
    assistant_blocks = payload["messages"][1]["content"]
    result_blocks = payload["messages"][2]["content"]
    assert assistant_blocks[0]["id"] == "call-7"
    assert assistant_blocks[0]["input"] == {"q": "x"}
    assert result_blocks[0]["tool_use_id"] == "call-7"
    assistant_blocks[0]["input"]["q"] = "changed"
    assert messages.items[1].tool_calls[0].arguments == {"q": "x"}


def test_generic_normalizer_leaves_usage_unset_and_calls_explicit_usage_policy():
    raw = multiple_blocks_response(text="reply", usage=None)
    default = normalize_messages_response(raw)
    assert default.usage is None
    calls = []

    def use_provider_usage(copied):
        calls.append(copied)
        return Usage(input_tokens=0, output_tokens=None, total_tokens=None)

    normalized = normalize_messages_response(raw, usage_normalizer=use_provider_usage)
    assert normalized.usage.input_tokens == 0
    assert "usage" in calls[0] and calls[0]["usage"] is None
    missing = multiple_blocks_response(text="reply")
    normalize_messages_response(missing, usage_normalizer=use_provider_usage)
    assert "usage" not in calls[1]
    assert raw["usage"] is None


def test_generic_response_concatenates_text_and_keeps_tools_and_opt_in_thinking():
    raw = multiple_blocks_response(blocks=[
        {"type": "text", "text": "one"},
        {"type": "tool_use", "id": "call-1", "name": "lookup", "input": {"q": 1}},
        {"type": "thinking", "thinking": "secret"},
        {"type": "text", "text": "two"},
    ])
    result = normalize_messages_response(raw)
    assert result.content == "onetwo"
    assert result.tool_calls[0].call_id == "call-1"
    assert result.tool_calls[0].arguments == {"q": 1}
    assert result.reasoning_events == []
    assert normalize_messages_response(raw, capture_reasoning=True).reasoning_events[0].text == "secret"
    native_like = normalize_messages_response(raw, text_normalizer=first_text_block)
    assert native_like.content == "one"
    assert raw["content"][0]["text"] == "one" and raw["content"][3]["text"] == "two"
