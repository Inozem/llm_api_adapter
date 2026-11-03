"""Internal adapter for caller-selected OpenAI-compatible API roots."""

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
from urllib.parse import urlsplit

from ..errors.config_errors import LLMConfigError
from ..llms.openai.async_client import OpenAIAsyncClient
from ..llms.openai.sync_client import OpenAISyncClient
from ..llms.transports import validate_sync_transport
from .openai.adapter import OpenAIAdapter


def _validate_api_root(base_url: object) -> str:
    """Validate and normalize an absolute compatible API root without I/O."""
    error = LLMConfigError(
        message="base_url must be an absolute HTTP(S) API root."
    )
    if not isinstance(base_url, str) or not base_url:
        raise error
    if any(
        character.isspace() or ord(character) < 0x20 or ord(character) == 0x7F
        for character in base_url
    ):
        raise error
    if "\\" in base_url or "?" in base_url or "#" in base_url:
        raise error

    try:
        parsed = urlsplit(base_url)
        scheme = parsed.scheme.lower()
        hostname = parsed.hostname
        port = parsed.port
    except (TypeError, ValueError):
        raise error from None

    if (
        scheme not in {"http", "https"}
        or not parsed.netloc
        or not hostname
        or parsed.netloc.endswith(":")
    ):
        raise error
    if "@" in parsed.netloc or parsed.username is not None or parsed.password is not None:
        raise LLMConfigError(message="base_url must not contain user information.")
    if port is not None and not 0 <= port <= 65535:
        raise error

    # urlsplit validates bracket structure but not the IPv6 address itself.
    if parsed.netloc.startswith("["):
        literal = parsed.netloc[1:parsed.netloc.find("]")]
        try:
            ipaddress.IPv6Address(literal)
        except ValueError:
            raise error from None
    elif hostname.count(".") == 3 and all(
        part.isdecimal() for part in hostname.split(".")
    ):
        try:
            ipaddress.IPv4Address(hostname)
        except ValueError:
            raise error from None

    if parsed.path.rstrip("/").endswith("/chat/completions"):
        raise LLMConfigError(
            message="base_url must be an API root, not the Chat Completions resource."
        )

    return base_url.rstrip("/")


@dataclass(repr=False)
class OpenAICompatibleAdapter(OpenAIAdapter):
    """OpenAI Chat Completions behavior for an explicitly selected API root."""

    company: str = "openai_compatible"
    base_url: str = ""

    def __post_init__(self) -> None:
        # The compatible mode must not resolve official registry metadata.
        if not isinstance(self.api_key, str) or not self.api_key:
            raise ValueError("api_key must be a non-empty string")
        if not isinstance(self.model, str) or not self.model:
            raise ValueError("model must be a non-empty string")
        if self.company != "openai_compatible":
            raise LLMConfigError(
                message="OpenAI-compatible mode requires company='openai_compatible'."
            )
        if self.service_provider is None:
            self.service_provider = "openai_compatible"
        elif self.service_provider != "openai_compatible":
            raise LLMConfigError(
                message=(
                    "OpenAI-compatible mode requires "
                    "service_provider='openai_compatible'."
                )
            )
        self.transport = validate_sync_transport(self.transport)
        self.base_url = _validate_api_root(self.base_url)
        self.model_spec = None
        self.pricing = None
        self.is_reasoning = False
        self.is_adaptive_thinking = False

    def _create_sync_client(self) -> OpenAISyncClient:
        client = OpenAISyncClient(
            api_key=self.api_key,
            endpoint=self.base_url,
            transport=self.transport,
        )
        client._set_official_behavior_enabled(False)
        return client

    def _create_async_client(self) -> OpenAIAsyncClient:
        client = OpenAIAsyncClient(
            api_key=self.api_key,
            endpoint=self.base_url,
        )
        client._set_official_behavior_enabled(False)
        return client
