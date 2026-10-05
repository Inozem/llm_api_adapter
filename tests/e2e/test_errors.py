import pytest

from llm_api_adapter.errors.llm_api_error import (
    LLMAPIAuthorizationError,
    LLMAPITimeoutError,
)
from llm_api_adapter.models.messages.chat_message import UserMessage


@pytest.mark.e2e
def test_chat_auth_error_invalid_api_key(organizations, e2e_adapter):
    """
    Verifies that an invalid API key is converted into a LLMAPIAuthorizationError.
    """
    for p in organizations:
        for model in p["models"]:
            adapter = e2e_adapter({**p, "api_key": "NON_VALID_KEY"}, model)
            with pytest.raises(LLMAPIAuthorizationError) as excinfo:
                adapter.chat(messages=[UserMessage("Say 'OK'.")], max_tokens=8)
            print(f"{p['name']=} {model=}: {excinfo.value}")

@pytest.mark.e2e
def test_chat_timeout_error(organizations, e2e_adapter, chat_with_retry):
    """
    Require a normalized timeout, retrying incidental connection failures.
    """
    for p in organizations:
        for model in p["models"]:
            adapter = e2e_adapter(p, model)
            base_kwargs = dict(
                messages=[UserMessage("Say 'OK'.")],
                max_tokens=512,
                temperature=1.0,
            )
            with pytest.raises(LLMAPITimeoutError):
                chat_with_retry(
                    adapter,
                    **base_kwargs,
                    timeout_s=0.1,
                    expected_error=LLMAPITimeoutError,
                )
