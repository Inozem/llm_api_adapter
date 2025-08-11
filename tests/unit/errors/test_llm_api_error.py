import pytest

from src.llm_api_adapter.errors.llm_api_error import (
    LLMAPIError,
    LLMAPIAuthorizationError,
    LLMAPIRateLimitError,
    LLMAPITokenLimitError,
    LLMAPIClientError,
    LLMAPIServerError,
    LLMAPITimeoutError,
    LLMAPIUsageLimitError,
)

def test_llmapierror_message_and_detail():
    err = LLMAPIError(message="Error occurred", detail="Detailed info")
    assert str(err) == "Error occurred Detail: Detailed info"

    err_no_detail = LLMAPIError(message="Simple error")
    assert str(err_no_detail) == "Simple error"

@pytest.mark.parametrize(
    "error_class, default_message, openai_errors, google_errors, anthropic_errors",
    [
        (LLMAPIAuthorizationError, "Authentication or authorization failed.",
         ["InvalidAuthenticationError", "AuthenticationError"],
         ["INVALID_ARGUMENT", "PERMISSION_DENIED"],
         ["AuthenticationError", "PermissionError"]),
        (LLMAPIRateLimitError, "Rate limit exceeded.",
         ["RateLimitError"],
         ["RESOURCE_EXHAUSTED"],
         ["RateLimitError"]),
        (LLMAPITokenLimitError, "Token limit exceeded.",
         ["MaxTokensExceededError", "TokenLimitError"],
         [],
         []),
        (LLMAPIClientError, "Client error occurred.",
         ["InvalidRequestError", "BadRequestError"],
         ["INVALID_ARGUMENT", "FAILED_PRECONDITION", "NOT_FOUND"],
         ["InvalidRequestError", "RequestTooLargeError", "NotFoundError"]),
        (LLMAPIServerError, "Server error occurred.",
         ["InternalServerError", "ServiceUnavailableError"],
         ["INTERNAL", "UNAVAILABLE"],
         ["APIError", "OverloadedError"]),
        (LLMAPITimeoutError, "Request timed out.",
         ["TimeoutError"],
         ["DEADLINE_EXCEEDED"],
         []),
        (LLMAPIUsageLimitError, "Usage limit exceeded.",
         ["UsageLimitError", "QuotaExceededError"],
         [],
         []),
    ],
)
def test_error_subclasses_attributes(
    error_class, default_message, openai_errors, google_errors, anthropic_errors
):
    error_instance = error_class()
    assert error_instance.message == default_message
    assert str(error_instance).startswith(default_message)
    assert hasattr(error_class, "openai_api_errors")
    assert error_class.openai_api_errors == openai_errors
    assert hasattr(error_class, "google_api_errors")
    assert error_class.google_api_errors == google_errors
    assert hasattr(error_class, "anthropic_api_errors")
    assert error_class.anthropic_api_errors == anthropic_errors
