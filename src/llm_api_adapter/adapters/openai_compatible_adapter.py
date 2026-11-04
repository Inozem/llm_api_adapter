"""Internal adapter for caller-selected OpenAI-compatible API roots."""

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import logging
import re
from collections.abc import Mapping
from urllib.parse import quote, urlsplit

from ..errors.config_errors import LLMConfigError
from ..errors.llm_api_error import LLMAPIClientError
from ..llms.openai.async_client import OpenAIAsyncClient
from ..llms.openai.sync_client import OpenAISyncClient
from ..llms.transports import validate_sync_transport
from .openai.adapter import OpenAIAdapter

logger = logging.getLogger(__name__)


_URI_RE = re.compile(r"(?i)\bhttps?://[^\s<>]+")


def _redact_compatible_diagnostic(text: object, api_key: str) -> str:
    """Redact one compatible instance's key and credential-bearing URLs."""
    def redact_url(match: re.Match[str]) -> str:
        original = match.group(0)
        token = original.rstrip(".,;:!?)]")
        suffix = original[len(token):]
        url = token
        try:
            parsed = urlsplit(url)
            has_userinfo = parsed.username is not None or parsed.password is not None
            if has_userinfo or "?" in url or "#" in url:
                return "[redacted URL]" + suffix
        except ValueError:
            return "[redacted URL]" + suffix
        return original

    result = _URI_RE.sub(redact_url, str(text))
    encoded_key = quote(api_key, safe="")
    if "%" in encoded_key:
        pieces = re.split(r"(%[0-9A-Fa-f]{2})", encoded_key)
        pattern = "".join(
            "%"
            + "".join(
                f"[{char.lower()}{char.upper()}]"
                if char.lower() in "abcdef"
                else char
                for char in piece[1:]
            )
            if piece.startswith("%")
            else re.escape(piece)
            for piece in pieces
        )
        result = re.sub(pattern, "[redacted]", result)
    if api_key:
        result = result.replace(api_key, "[redacted]")

    return result


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

    def _format_compatible_diagnostic(self, text: object) -> str:
        return _redact_compatible_diagnostic(text, self.api_key)

    @staticmethod
    def _malformed_response() -> LLMAPIClientError:
        return LLMAPIClientError(
            message="The compatible server returned a malformed Chat Completions response."
        )

    @staticmethod
    def _validate_message_fields(message: Mapping[str, object]) -> None:
        for field in ("content", "refusal"):
            if (
                field in message
                and message[field] is not None
                and not isinstance(message[field], str)
            ):
                raise OpenAICompatibleAdapter._malformed_response()
        if (
            "tool_calls" in message
            and message["tool_calls"] is not None
            and not isinstance(message["tool_calls"], list)
        ):
            raise OpenAICompatibleAdapter._malformed_response()
        if (
            "function_call" in message
            and message["function_call"] is not None
            and not isinstance(message["function_call"], Mapping)
        ):
            raise OpenAICompatibleAdapter._malformed_response()

    def _normalize_chat_completions_response(self, response: dict):
        """Validate the compatible envelope, then use OpenAI's shared normalizer."""
        if not isinstance(response, Mapping):
            raise self._malformed_response()
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise self._malformed_response()
        choice = choices[0]
        if not isinstance(choice, Mapping):
            raise self._malformed_response()
        message = choice.get("message")
        finish_reason = choice.get("finish_reason")
        if (
            not isinstance(message, Mapping)
            or not isinstance(finish_reason, str)
            or not finish_reason.strip()
        ):
            raise self._malformed_response()
        self._validate_message_fields(message)
        return super()._normalize_chat_completions_response(response)

    def _validate_chat_completions_stream_event(self, event: object) -> None:
        """Validate compatible Chat Completions events before shared coercion."""
        if getattr(event, "done", False):
            return
        payload = getattr(event, "data", None)
        if not isinstance(payload, Mapping):
            raise self._malformed_response()
        choices = payload.get("choices")
        if not isinstance(choices, list):
            raise self._malformed_response()
        for choice in choices:
            if not isinstance(choice, Mapping):
                raise self._malformed_response()
            delta = choice.get("delta")
            if delta is not None and not isinstance(delta, Mapping):
                raise self._malformed_response()
            if isinstance(delta, Mapping):
                self._validate_message_fields(delta)
            finish_reason = choice.get("finish_reason")
            if finish_reason is not None and (
                not isinstance(finish_reason, str) or not finish_reason.strip()
            ):
                raise self._malformed_response()

    def __repr__(self) -> str:
        inherited = super().__repr__()
        return self._format_compatible_diagnostic(inherited)

    def handle_error(self, error: Exception, error_message: str | None = None):
        """Keep mapped error identity while making compatible diagnostics safe."""
        safe_message = self._format_compatible_diagnostic(
            error_message if error_message is not None else str(error)
        )
        if hasattr(error, "detail") and error.detail is not None:
            error.detail = self._format_compatible_diagnostic(error.detail)
        if hasattr(error, "message"):
            error.message = self._format_compatible_diagnostic(error.message)
        original_args = error.args
        if original_args:
            error.args = tuple(
                self._format_compatible_diagnostic(value)
                if isinstance(value, str)
                else value
                for value in original_args
            )
        else:
            error.args = (safe_message,)
        logger_message = self._format_compatible_diagnostic(
            f"Error with the provider '{self.company}' the model '{self.model}': {safe_message}. "
        )
        logger.error(logger_message)
        raise error from None

    def _create_sync_client(self) -> OpenAISyncClient:
        client = OpenAISyncClient(
            api_key=self.api_key,
            endpoint=self.base_url,
            transport=self.transport,
        )
        client._set_official_behavior_enabled(False)
        client._set_diagnostic_formatter(self._format_compatible_diagnostic)
        return client

    def _create_async_client(self) -> OpenAIAsyncClient:
        client = OpenAIAsyncClient(
            api_key=self.api_key,
            endpoint=self.base_url,
        )
        client._set_official_behavior_enabled(False)
        client._set_diagnostic_formatter(self._format_compatible_diagnostic)
        return client
