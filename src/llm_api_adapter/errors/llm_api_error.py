from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMAPIError(Exception):
    """Base class for API-related errors."""
    message: str = "An API error occurred."
    detail: Optional[str] = None

    def __post_init__(self):
        full_message = self.message
        if self.detail:
            full_message = f"{self.message} Detail: {self.detail}"
        super().__init__(full_message)


@dataclass
class LLMAPIAuthorizationError(LLMAPIError):
    """Raised when authentication or authorization fails."""
    message: str = "Authentication or authorization failed."
    openai_api_errors = ["InvalidAuthenticationError", "AuthenticationError"]
    google_api_errors = ["INVALID_ARGUMENT", "PERMISSION_DENIED"]
    anthropic_api_errors = ["AuthenticationError", "PermissionError"]


@dataclass
class LLMAPIRateLimitError(LLMAPIError):
    """Raised when rate limits are exceeded."""
    message: str = "Rate limit exceeded."
    openai_api_errors = ["RateLimitError"]
    google_api_errors = ["RESOURCE_EXHAUSTED"]
    anthropic_api_errors = ["RateLimitError"]


@dataclass
class LLMAPITokenLimitError(LLMAPIError):
    """Raised when token limits are exceeded."""
    message: str = "Token limit exceeded."
    openai_api_errors = ["MaxTokensExceededError", "TokenLimitError"]
    google_api_errors = []
    anthropic_api_errors = []


@dataclass
class LLMAPIClientError(LLMAPIError):
    """Raised when the client makes an invalid request."""
    message: str = "Client error occurred."
    openai_api_errors = ["InvalidRequestError", "BadRequestError"]
    google_api_errors = [
        "INVALID_ARGUMENT", "FAILED_PRECONDITION", "NOT_FOUND"
    ]
    anthropic_api_errors = [
        "InvalidRequestError", "RequestTooLargeError", "NotFoundError"
    ]


@dataclass
class LLMAPIServerError(LLMAPIError):
    """Raised when the server encounters an error."""
    message: str = "Server error occurred."
    openai_api_errors = ["InternalServerError", "ServiceUnavailableError"]
    google_api_errors = ["INTERNAL", "UNAVAILABLE"]
    anthropic_api_errors = ["APIError", "OverloadedError"]


@dataclass
class LLMAPITimeoutError(LLMAPIError):
    """Raised when a request times out."""
    message: str = "Request timed out."
    openai_api_errors = ["TimeoutError"]
    google_api_errors = ["DEADLINE_EXCEEDED"]
    anthropic_api_errors = []


@dataclass
class LLMAPIUsageLimitError(LLMAPIError):
    """Raised when usage limits are exceeded."""
    message: str = "Usage limit exceeded."
    openai_api_errors = ["UsageLimitError", "QuotaExceededError"]
    google_api_errors = []
    anthropic_api_errors = []
