import json

import pytest

from src.llm_api_adapter.models.messages.chat_message import (
    AIMessage,
    Message,
    Messages,
    Prompt,
    ToolMessage,
    UserMessage,
)
from src.llm_api_adapter.models.tools import ToolCall


@pytest.mark.unit
def test_message_base_serializers():
    m = UserMessage(content="hello")
    assert m.to_openai() == {"role": "user", "content": "hello"}
    assert m.to_openai_responses_input() == [{"role": "user", "content": "hello"}]
    assert m.to_anthropic() == {"role": "user", "content": "hello"}
    assert m.to_google() == {"role": "user", "parts": [{"text": "hello"}]}


@pytest.mark.unit
def test_prompt_to_anthropic_and_google_and_openai():
    p = Prompt(content="system instructions")
    assert p.to_anthropic() == "system instructions"
    assert p.to_google() == "system instructions"
    assert p.to_openai() == {"role": "system", "content": "system instructions"}


@pytest.mark.unit
def test_user_and_ai_to_google_and_openai():
    u = UserMessage(content="hello user")
    a = AIMessage(content="hello ai")
    assert u.to_google() == {"role": "user", "parts": [{"text": "hello user"}]}
    assert a.to_google() == {"role": "model", "parts": [{"text": "hello ai"}]}
    assert u.to_openai() == {"role": "user", "content": "hello user"}
    assert a.to_openai() == {"role": "assistant", "content": "hello ai"}


@pytest.mark.unit
def test_ai_message_to_openai_with_tool_calls():
    msg = AIMessage(
        content="calling tool",
        tool_calls=[
            ToolCall(name="weather", arguments={"city": "TLV"}, call_id="call_1")
        ],
    )
    assert msg.to_openai() == {
        "role": "assistant",
        "content": "calling tool",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "weather",
                    "arguments": json.dumps({"city": "TLV"}, ensure_ascii=False),
                },
            }
        ],
    }


@pytest.mark.unit
def test_ai_message_to_openai_responses_input_with_content():
    msg = AIMessage(content="assistant text")
    assert msg.to_openai_responses_input() == [
        {"role": "assistant", "content": "assistant text"}
    ]


@pytest.mark.unit
def test_ai_message_to_openai_responses_input_without_content():
    msg = AIMessage(content="")
    assert msg.to_openai_responses_input() == []


@pytest.mark.unit
def test_ai_message_to_anthropic_without_tool_calls():
    msg = AIMessage(content="hello")
    assert msg.to_anthropic() == {"role": "assistant", "content": "hello"}


@pytest.mark.unit
def test_ai_message_to_anthropic_with_tool_calls_and_text():
    msg = AIMessage(
        content="before tool",
        tool_calls=[
            ToolCall(name="weather", arguments={"city": "TLV"}, call_id="call_1")
        ],
    )
    assert msg.to_anthropic() == {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "before tool"},
            {
                "type": "tool_use",
                "id": "call_1",
                "name": "weather",
                "input": {"city": "TLV"},
            },
        ],
    }


@pytest.mark.unit
def test_ai_message_to_anthropic_with_tool_calls_and_empty_arguments():
    msg = AIMessage(
        content="",
        tool_calls=[
            ToolCall(name="weather", arguments=None, call_id="call_1")
        ],
    )
    assert msg.to_anthropic() == {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "call_1",
                "name": "weather",
                "input": {},
            }
        ],
    }


@pytest.mark.unit
def test_ai_message_to_anthropic_requires_call_id():
    msg = AIMessage(
        content="x",
        tool_calls=[ToolCall(name="weather", arguments={"city": "TLV"}, call_id="")],
    )
    with pytest.raises(
        ValueError,
        match="Anthropic tool_use requires a non-empty call_id",
    ):
        msg.to_anthropic()


@pytest.mark.unit
def test_ai_message_to_google_without_parts_falls_back_to_empty_text():
    msg = AIMessage(content="", tool_calls=None)
    assert msg.to_google() == {"role": "model", "parts": [{"text": ""}]}


@pytest.mark.unit
def test_ai_message_to_google_with_text_and_tool_calls():
    msg = AIMessage(
        content="before tool",
        tool_calls=[
            ToolCall(name="weather", arguments={"city": "TLV"}, call_id="ignored")
        ],
    )
    assert msg.to_google() == {
        "role": "model",
        "parts": [
            {"text": "before tool"},
            {
                "functionCall": {
                    "name": "weather",
                    "args": {"city": "TLV"},
                }
            },
        ],
    }


@pytest.mark.unit
def test_ai_message_to_google_with_tool_calls_and_no_args():
    msg = AIMessage(
        content="",
        tool_calls=[ToolCall(name="weather", arguments=None, call_id="ignored")],
    )
    assert msg.to_google() == {
        "role": "model",
        "parts": [
            {
                "functionCall": {
                    "name": "weather",
                    "args": {},
                }
            }
        ],
    }


@pytest.mark.unit
def test_tool_message_to_openai():
    msg = ToolMessage(content='{"ok": true}', tool_call_id="call_1")
    assert msg.to_openai() == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": '{"ok": true}',
    }


@pytest.mark.unit
def test_tool_message_to_openai_responses_input():
    msg = ToolMessage(content='{"ok": true}', tool_call_id="call_1")
    assert msg.to_openai_responses_input() == [
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": '{"ok": true}',
        }
    ]


@pytest.mark.unit
def test_tool_message_to_anthropic():
    msg = ToolMessage(content="tool result", tool_call_id="call_1")
    assert msg.to_anthropic() == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "call_1",
                "content": "tool result",
            }
        ],
    }


@pytest.mark.unit
def test_tool_message_to_anthropic_requires_tool_call_id():
    msg = ToolMessage(content="tool result", tool_call_id="")
    with pytest.raises(
        ValueError,
        match="Anthropic tool_result requires tool_call_id",
    ):
        msg.to_anthropic()


@pytest.mark.unit
def test_tool_message_to_google_with_json_content():
    msg = ToolMessage(content='{"temperature": 30}', tool_call_id="weather")
    assert msg.to_google() == {
        "role": "user",
        "parts": [
            {
                "functionResponse": {
                    "name": "weather",
                    "response": {"temperature": 30},
                }
            }
        ],
    }


@pytest.mark.unit
def test_tool_message_to_google_with_empty_content():
    msg = ToolMessage(content="   ", tool_call_id="weather")
    assert msg.to_google() == {
        "role": "user",
        "parts": [
            {
                "functionResponse": {
                    "name": "weather",
                    "response": {},
                }
            }
        ],
    }


@pytest.mark.unit
def test_tool_message_to_google_with_non_json_content():
    msg = ToolMessage(content="plain text result", tool_call_id="weather")
    assert msg.to_google() == {
        "role": "user",
        "parts": [
            {
                "functionResponse": {
                    "name": "weather",
                    "response": {"content": "plain text result"},
                }
            }
        ],
    }


@pytest.mark.unit
def test_tool_message_to_google_requires_tool_call_id():
    msg = ToolMessage(content="x", tool_call_id="")
    with pytest.raises(
        ValueError,
        match="Google functionResponse requires tool_call_id to be set",
    ):
        msg.to_google()


@pytest.mark.unit
def test_messages_to_anthropic_with_prompt_and_messages():
    items = [Prompt(content="sys"), UserMessage(content="u1"), AIMessage(content="a1")]
    msgs = Messages(items=items)
    prompt, anthropic_messages = msgs.to_anthropic()
    assert prompt == "sys"
    assert anthropic_messages == [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
    ]


@pytest.mark.unit
def test_messages_to_google_with_prompt_and_messages():
    items = [Prompt(content="sysg"), UserMessage(content="ug"), AIMessage(content="ag")]
    msgs = Messages(items=items)
    prompt, google_messages = msgs.to_google()
    assert prompt == "sysg"
    assert google_messages == [
        {"role": "user", "parts": [{"text": "ug"}]},
        {"role": "model", "parts": [{"text": "ag"}]},
    ]


@pytest.mark.unit
def test_messages_to_anthropic_and_google_multiple_prompts_raise():
    items = [Prompt(content="p1"), Prompt(content="p2")]
    msgs = Messages(items=items)

    with pytest.raises(ValueError) as excinfo_an:
        msgs.to_anthropic()
    assert "Multiple system prompts are not allowed for Anthropic." in str(excinfo_an.value)

    with pytest.raises(ValueError) as excinfo_gg:
        msgs.to_google()
    assert "Multiple system prompts are not allowed for Google." in str(excinfo_gg.value)


@pytest.mark.unit
def test_messages_to_openai_returns_list_of_dicts():
    items = [UserMessage(content="u"), AIMessage(content="a")]
    msgs = Messages(items=items)
    openai_messages = msgs.to_openai()
    assert openai_messages == [
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ]


@pytest.mark.unit
def test_messages_to_openai_responses_input_skips_prompt_and_flattens():
    items = [
        Prompt(content="system"),
        UserMessage(content="u"),
        AIMessage(content="a"),
        ToolMessage(content='{"ok": true}', tool_call_id="call_1"),
    ]
    msgs = Messages(items=items)
    assert msgs.to_openai_responses_input() == [
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": '{"ok": true}',
        },
    ]


@pytest.mark.unit
def test_messages_to_openai_responses_instructions_returns_prompt():
    msgs = Messages(items=[Prompt(content="system"), UserMessage(content="hi")])
    assert msgs.to_openai_responses_instructions() == "system"


@pytest.mark.unit
def test_messages_to_openai_responses_instructions_returns_none_without_prompt():
    msgs = Messages(items=[UserMessage(content="hi")])
    assert msgs.to_openai_responses_instructions() is None


@pytest.mark.unit
def test_messages_to_openai_responses_instructions_multiple_prompts_raise():
    msgs = Messages(items=[Prompt(content="p1"), Prompt(content="p2")])
    with pytest.raises(
        ValueError,
        match="Multiple system prompts are not allowed for OpenAI Responses.",
    ):
        msgs.to_openai_responses_instructions()


@pytest.mark.unit
def test_messages_post_init_accepts_message_instances():
    items = [UserMessage(content="u"), AIMessage(content="a")]
    msgs = Messages(items=items)
    assert isinstance(msgs.items[0], UserMessage)
    assert isinstance(msgs.items[1], AIMessage)


@pytest.mark.unit
def test_messages_post_init_unsupported_item_type_raises():
    with pytest.raises(TypeError, match="Unsupported message type"):
        Messages(items=[123])


@pytest.mark.unit
def test_messages_post_init_dict_missing_role_raises():
    with pytest.raises(ValueError) as excinfo:
        Messages(items=[{"content": "no role"}])
    assert "Missing 'role' in message data" in str(excinfo.value)


@pytest.mark.unit
def test_messages_post_init_dict_missing_content_raises():
    with pytest.raises(ValueError) as excinfo:
        Messages(items=[{"role": "user"}])
    assert "Missing 'content' in message data" in str(excinfo.value)


@pytest.mark.unit
def test_messages_post_init_dict_unsupported_role_raises():
    with pytest.raises(ValueError) as excinfo:
        Messages(items=[{"role": "unknown", "content": "x"}])
    assert "Unsupported role: unknown" in str(excinfo.value)


@pytest.mark.unit
def test_messages_normalize_tool_message_from_dict():
    msgs = Messages(
        items=[
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "result",
            }
        ]
    )
    assert isinstance(msgs.items[0], ToolMessage)
    assert msgs.items[0].tool_call_id == "call_1"
    assert msgs.items[0].content == "result"


@pytest.mark.unit
def test_messages_normalize_tool_message_missing_tool_call_id_raises():
    with pytest.raises(ValueError, match="Missing 'tool_call_id' for tool message"):
        Messages(items=[{"role": "tool", "content": "result"}])


@pytest.mark.unit
def test_messages_normalize_tool_message_missing_content_raises():
    with pytest.raises(ValueError, match="Missing 'content' in message data"):
        Messages(items=[{"role": "tool", "tool_call_id": "call_1"}])


@pytest.mark.unit
def test_messages_normalize_assistant_message_with_no_content_defaults_to_empty():
    msgs = Messages(items=[{"role": "assistant"}])
    assert isinstance(msgs.items[0], AIMessage)
    assert msgs.items[0].content == ""
    assert msgs.items[0].tool_calls is None


@pytest.mark.unit
def test_messages_normalize_assistant_message_with_normalized_tool_calls():
    msgs = Messages(
        items=[
            {
                "role": "assistant",
                "content": "working",
                "tool_calls": [
                    {
                        "name": "weather",
                        "arguments": {"city": "TLV"},
                        "call_id": "call_1",
                    }
                ],
            }
        ]
    )
    msg = msgs.items[0]
    assert isinstance(msg, AIMessage)
    assert msg.content == "working"
    assert msg.tool_calls == [
        ToolCall(name="weather", arguments={"city": "TLV"}, call_id="call_1")
    ]


@pytest.mark.unit
def test_messages_normalize_assistant_message_with_normalized_tool_call_arguments_none():
    msgs = Messages(
        items=[
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "name": "weather",
                        "arguments": None,
                        "call_id": "call_1",
                    }
                ],
            }
        ]
    )
    msg = msgs.items[0]
    assert msg.tool_calls == [
        ToolCall(name="weather", arguments={}, call_id="call_1")
    ]


@pytest.mark.unit
def test_messages_normalize_assistant_message_with_legacy_openai_tool_calls():
    msgs = Messages(
        items=[
            {
                "role": "assistant",
                "content": "working",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "weather",
                            "arguments": '{"city": "TLV"}',
                        },
                    }
                ],
            }
        ]
    )
    msg = msgs.items[0]
    assert msg.tool_calls == [
        ToolCall(name="weather", arguments={"city": "TLV"}, call_id="call_1")
    ]


@pytest.mark.unit
def test_messages_parse_assistant_tool_calls_none_returns_none():
    msgs = Messages(items=[])
    assert msgs._parse_assistant_tool_calls(None) is None


@pytest.mark.unit
def test_messages_parse_assistant_tool_calls_requires_list():
    msgs = Messages(items=[])
    with pytest.raises(ValueError, match="assistant.tool_calls must be a list"):
        msgs._parse_assistant_tool_calls("bad")


@pytest.mark.unit
def test_messages_parse_single_tool_call_requires_dict():
    msgs = Messages(items=[])
    with pytest.raises(ValueError, match="assistant.tool_calls items must be dict"):
        msgs._parse_single_tool_call("bad")


@pytest.mark.unit
def test_messages_parse_normalized_tool_call_requires_name():
    msgs = Messages(items=[])
    with pytest.raises(
        ValueError,
        match="assistant.tool_calls.name must be non-empty str",
    ):
        msgs._parse_normalized_tool_call({"name": ""})


@pytest.mark.unit
def test_messages_parse_normalized_tool_call_requires_arguments_dict():
    msgs = Messages(items=[])
    with pytest.raises(
        ValueError,
        match="assistant.tool_calls.arguments must be dict",
    ):
        msgs._parse_normalized_tool_call(
            {"name": "weather", "arguments": "bad"}
        )


@pytest.mark.unit
def test_messages_parse_legacy_openai_tool_call_requires_name():
    msgs = Messages(items=[])
    with pytest.raises(
        ValueError,
        match="assistant.tool_calls.function.name must be non-empty str",
    ):
        msgs._parse_legacy_openai_tool_call({"function": {"arguments": "{}"}})


@pytest.mark.unit
def test_messages_parse_legacy_openai_tool_arguments_from_json_string():
    msgs = Messages(items=[])
    assert msgs._parse_legacy_openai_tool_arguments('{"x": 1}') == {"x": 1}


@pytest.mark.unit
def test_messages_parse_legacy_openai_tool_arguments_from_empty_string():
    msgs = Messages(items=[])
    assert msgs._parse_legacy_openai_tool_arguments("   ") == {}


@pytest.mark.unit
def test_messages_parse_legacy_openai_tool_arguments_from_dict():
    msgs = Messages(items=[])
    assert msgs._parse_legacy_openai_tool_arguments({"x": 1}) == {"x": 1}


@pytest.mark.unit
def test_messages_parse_legacy_openai_tool_arguments_from_other_type_returns_empty_dict():
    msgs = Messages(items=[])
    assert msgs._parse_legacy_openai_tool_arguments(123) == {}


@pytest.mark.unit
def test_messages_parse_legacy_openai_tool_arguments_invalid_json_raises():
    msgs = Messages(items=[])
    with pytest.raises(
        ValueError,
        match="assistant.tool_calls.arguments JSON parse failed",
    ):
        msgs._parse_legacy_openai_tool_arguments("{bad json}")


@pytest.mark.unit
def test_messages_normalize_simple_message_rejects_none_content():
    msgs = Messages(items=[])
    with pytest.raises(ValueError, match="Missing 'content' in message data"):
        msgs._normalize_simple_message({"role": "user"}, UserMessage)


@pytest.mark.unit
def test_messages_normalize_simple_message_rejects_empty_content():
    msgs = Messages(items=[])
    with pytest.raises(ValueError, match="Missing 'content' in message data"):
        msgs._normalize_simple_message({"role": "user", "content": ""}, UserMessage)


@pytest.mark.unit
def test_messages_normalize_simple_message_casts_content_to_str():
    msgs = Messages(items=[])
    msg = msgs._normalize_simple_message({"role": "user", "content": 123}, UserMessage)
    assert isinstance(msg, UserMessage)
    assert msg.content == "123"


@pytest.mark.unit
def test_messages_to_anthropic_with_tool_message_and_ai_tool_calls():
    items = [
        Prompt(content="sys"),
        UserMessage(content="u1"),
        AIMessage(
            content="working",
            tool_calls=[ToolCall(name="weather", arguments={"city": "TLV"}, call_id="call_1")],
        ),
        ToolMessage(content='{"temperature": 30}', tool_call_id="call_1"),
    ]
    msgs = Messages(items=items)
    prompt, anthropic_messages = msgs.to_anthropic()

    assert prompt == "sys"
    assert anthropic_messages == [
        {"role": "user", "content": "u1"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "working"},
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "weather",
                    "input": {"city": "TLV"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "content": '{"temperature": 30}',
                }
            ],
        },
    ]


@pytest.mark.unit
def test_messages_to_google_with_tool_message_and_non_user_message_fallback():
    class DummyOtherMessage(Message):
        role: str = "custom"

    other = DummyOtherMessage(content="fallback text")

    items = [
        Prompt(content="sys"),
        UserMessage(content="u1"),
        AIMessage(
            content="working",
            tool_calls=[ToolCall(name="weather", arguments={"city": "TLV"}, call_id="call_1")],
        ),
        ToolMessage(content='{"temperature": 30}', tool_call_id="weather"),
        other,
    ]
    msgs = Messages(items=items)
    prompt, google_messages = msgs.to_google()

    assert prompt == "sys"
    assert google_messages == [
        {"role": "user", "parts": [{"text": "u1"}]},
        {
            "role": "model",
            "parts": [
                {"text": "working"},
                {
                    "functionCall": {
                        "name": "weather",
                        "args": {"city": "TLV"},
                    }
                },
            ],
        },
        {
            "role": "user",
            "parts": [
                {
                    "functionResponse": {
                        "name": "weather",
                        "response": {"temperature": 30},
                    }
                }
            ],
        },
        {"role": "user", "parts": [{"text": "fallback text"}]},
    ]
