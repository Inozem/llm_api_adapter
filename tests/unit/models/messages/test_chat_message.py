import pytest


from src.llm_api_adapter.models.messages.chat_message import (
    Message, Prompt, UserMessage, AIMessage, Messages
)

def test_prompt_to_anthropic_and_google_and_openai():
    p = Prompt(content="system instructions")
    assert p.to_anthropic() == "system instructions"
    assert p.to_google() == "system instructions"
    assert p.to_openai() == {"role": "system", "content": "system instructions"}

def test_user_and_ai_to_google_and_openai():
    u = UserMessage(content="hello user")
    a = AIMessage(content="hello ai")
    assert u.to_google() == {"role": "user", "parts": [{"text": "hello user"}]}
    assert a.to_google() == {"role": "model", "parts": [{"text": "hello ai"}]}
    assert u.to_openai() == {"role": "user", "content": "hello user"}
    assert a.to_openai() == {"role": "assistant", "content": "hello ai"}

def test_messages_to_anthropic_with_prompt_and_messages():
    items = [Prompt(content="sys"), UserMessage(content="u1"), AIMessage(content="a1")]
    msgs = Messages(items=items)
    prompt, anthropic_messages = msgs.to_anthropic()
    assert prompt == "sys"
    assert anthropic_messages == [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
    ]

def test_messages_to_google_with_prompt_and_messages():
    items = [Prompt(content="sysg"), UserMessage(content="ug"), AIMessage(content="ag")]
    msgs = Messages(items=items)
    prompt, google_messages = msgs.to_google()
    assert prompt == "sysg"
    assert google_messages == [
        {"role": "user", "parts": [{"text": "ug"}]},
        {"role": "model", "parts": [{"text": "ag"}]},
    ]

def test_messages_to_anthropic_and_google_multiple_prompts_raise():
    items = [Prompt(content="p1"), Prompt(content="p2")]
    msgs = Messages(items=items)
    with pytest.raises(ValueError) as excinfo_an:
        msgs.to_anthropic()
    assert "Multiple system prompts are not allowed for Anthropic." in str(excinfo_an.value)
    with pytest.raises(ValueError) as excinfo_gg:
        msgs.to_google()
    assert "Multiple system prompts are not allowed for Google." in str(excinfo_gg.value)

def test_messages_to_openai_returns_list_of_dicts():
    items = [UserMessage(content="u"), AIMessage(content="a")]
    msgs = Messages(items=items)
    openai_messages = msgs.to_openai()
    assert openai_messages == [
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ]

def test_messages_post_init_dict_missing_role_raises():
    with pytest.raises(ValueError) as excinfo:
        Messages(items=[{"content": "no role"}])
    assert "Missing 'role' in message data" in str(excinfo.value)

def test_messages_post_init_dict_missing_content_raises():
    with pytest.raises(ValueError) as excinfo:
        Messages(items=[{"role": "user"}])
    assert "Missing 'content' in message data" in str(excinfo.value)

def test_messages_post_init_dict_unsupported_role_raises():
    with pytest.raises(ValueError) as excinfo:
        Messages(items=[{"role": "unknown", "content": "x"}])
    assert "Unsupported role: unknown" in str(excinfo.value)
