import os

import pytest

from llm_api_adapter.models.messages.chat_message import UserMessage
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter


_STREAMING_SCENARIOS = [
    pytest.param("openai", "gpt-5-nano", "OPENAI_API_KEY", id="openai-responses"),
    pytest.param("openai", "gpt-4.1-mini", "OPENAI_API_KEY", id="openai-chat-completions"),
    pytest.param("anthropic", "claude-haiku-4-5", "ANTHROPIC_API_KEY", id="anthropic"),
    pytest.param("google", "gemini-2.5-flash", "GOOGLE_API_KEY", id="google"),
]


@pytest.mark.e2e
@pytest.mark.parametrize(("organization", "model", "api_key_env"), _STREAMING_SCENARIOS)
def test_stream_chat_returns_text_and_finalized_response(
    organization,
    model,
    api_key_env,
    stream_with_retry,
):
    api_key = os.getenv(api_key_env)
    if not api_key:
        pytest.skip(f"{api_key_env} is not configured")

    adapter = UniversalLLMAPIAdapter(
        organization=organization,
        model=model,
        api_key=api_key,
    )
    completed_responses = []
    chunks = stream_with_retry(
        adapter,
        messages=[UserMessage("Reply with exactly: OK")],
        max_tokens=64,
        temperature=0,
        timeout_s=60,
        on_done=completed_responses.append,
    )

    assert "".join(chunks).strip()
    assert len(completed_responses) == 1

    response = completed_responses[0]
    assert isinstance(response.model, str) and response.model
    assert response.usage is not None
    assert response.usage.input_tokens >= 0
    assert response.usage.output_tokens >= 0
    assert response.usage.total_tokens >= response.usage.input_tokens
    assert isinstance(response.finish_reason, str) and response.finish_reason
