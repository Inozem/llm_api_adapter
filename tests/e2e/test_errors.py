import pytest

from llm_api_adapter.errors.llm_api_error import LLMAPIAuthorizationError, LLMAPITimeoutError
from llm_api_adapter.models.messages.chat_message import UserMessage
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter

@pytest.mark.e2e
def test_chat_auth_error_invalid_api_key(providers):
    """
    Verifies that an invalid API key is converted into a LLMAPIAuthorizationError.
    """
    for p in providers:
        for model in p["models"]:
            adapter = UniversalLLMAPIAdapter(
                organization=p["name"],
                model=model,
                api_key="NON_VALID_KEY",
            )
            with pytest.raises(LLMAPIAuthorizationError) as excinfo:
                adapter.chat(messages=[UserMessage("Say 'OK'.")], max_tokens=8)
            print(f"{p['name']=} {model=}: {excinfo.value}")

@pytest.mark.e2e
def test_chat_timeout_error(providers):
    """
    Verifies that an extremely small timeout is converted into a LLMAPITimeoutError.
    """
    for p in providers:
        for model in p["models"]:
            adapter = UniversalLLMAPIAdapter(
                organization=p["name"],
                model=model,
                api_key=p["api_key"],
            )
            base_kwargs = dict(
                messages=[UserMessage("Say 'OK'.")],
                max_tokens=512,
                temperature=1.0,
            )
            with pytest.raises(LLMAPITimeoutError):
                adapter.chat(**base_kwargs, **{"timeout_s": 0.001})
