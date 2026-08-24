from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMConfigError(Exception):
    """Raised when the request configuration is invalid or incompatible."""
    message: str = "Invalid LLM request configuration."
    detail: Optional[str] = None

    def __post_init__(self):
        full_message = self.message
        if self.detail:
            full_message = f"{self.message} Detail: {self.detail}"
        super().__init__(full_message)


@dataclass
class LLMReasoningLevelError(LLMConfigError):
    """Raised when reasoning_level is greater max_tokens."""
    message: str = "Reasoning level is too high: 'max_tokens' must be > 'reasoning_level'."


class ProviderNotInstalledError(LLMConfigError):
    """Raised when a known external provider package is unavailable."""

    def __init__(self, organization: str, distribution: str) -> None:
        self.organization = organization
        self.distribution = distribution
        super().__init__(
            message=(
                f"Provider '{organization}' is not installed. "
                f"Install it with: pip install {distribution}"
            ),
        )
