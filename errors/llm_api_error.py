from typing import List, Optional


class LLMAPIError(Exception):
    """Base class for API-related errors."""
    def __init__(self, message: Optional[str] = None):
        self.message = message or self.__class__.message
        self.openai_api_errors: List[str] = self.__class__.openai_api_errors
        self.google_api_errors: List[str] = self.__class__.google_api_errors
        self.anthropic_api_errors: List[str] = (
            self.__class__.anthropic_api_errors
        )
        super().__init__(self.message)

    message: str = "An API error occurred."
    openai_api_errors: List[str] = []
    google_api_errors: List[str] = []
    anthropic_api_errors: List[str] = []


class LLMAPIAuthorizationError(LLMAPIError):
    """Raised when authentication or authorization fails."""
    message: str = "Authentication or authorization failed."
    openai_api_errors: List[str] = [
        "InvalidAuthenticationError", "AuthenticationError"
    ]
    google_api_errors: List[str] = ["Unauthenticated", "PermissionDenied"]
    anthropic_api_errors: List[str] = [
        "AuthenticationError", "PermissionError"
    ]


class LLMAPIRateLimitError(LLMAPIError):
    """Raised when rate limits are exceeded."""
    message: str = "Rate limit exceeded."
    openai_api_errors: List[str] = ["RateLimitError"]
    google_api_errors: List[str] = ["RateLimitExceeded", "TooManyRequests"]
    anthropic_api_errors: List[str] = ["RateLimitError"]


class LLMAPITokenLimitError(LLMAPIError):
    """Raised when token limits are exceeded."""
    message: str = "Token limit exceeded."
    openai_api_errors: List[str] = [
        "MaxTokensExceededError", "TokenLimitError"
    ]
    google_api_errors: List[str] = ["MaxTokensExceeded"]
    anthropic_api_errors: List[str] = []


class LLMAPIClientError(LLMAPIError):
    """Raised when the client makes an invalid request."""
    message: str = "Client error occurred."
    openai_api_errors: List[str] = ["InvalidRequestError", "BadRequestError"]
    google_api_errors: List[str] = ["InvalidArgument", "BadRequest"]
    anthropic_api_errors: List[str] = [
        "InvalidRequestError", "RequestTooLargeError", "NotFoundError"
    ]


class LLMAPIServerError(LLMAPIError):
    """Raised when the server encounters an error."""
    message: str = "Server error occurred."
    openai_api_errors: List[str] = [
        "InternalServerError", "ServiceUnavailableError"
    ]
    google_api_errors: List[str] = ["InternalError", "ServiceUnavailable"]
    anthropic_api_errors: List[str] = ["APIError", "OverloadedError"]


class LLMAPITimeoutError(LLMAPIError):
    """Raised when a request times out."""
    message: str = "Request timed out."
    openai_api_errors: List[str] = ["TimeoutError"]
    google_api_errors: List[str] = ["RequestTimeout", "DeadlineExceeded"]
    anthropic_api_errors: List[str] = []


class LLMAPIUsageLimitError(LLMAPIError):
    """Raised when usage limits are exceeded."""
    message: str = "Usage limit exceeded."
    openai_api_errors: List[str] = ["UsageLimitError", "QuotaExceededError"]
    google_api_errors: List[str] = ["QuotaExceeded", "ResourceExhausted"]
    anthropic_api_errors: List[str] = []


class LLMLibraryNotInstalledError(Exception):
    """Raised when a required library is not installed."""
    def __init__(self, library_name: str, library_install_name: str):
        message = (f"The library '{library_name}' is not installed. "
                   "Please install it using the command: "
                   f"pip install '{library_install_name}'. ")
        super().__init__(message)
