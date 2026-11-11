from dataclasses import dataclass, field
import logging
from typing import Any, Callable, Iterator, Mapping

from ...errors.llm_api_error import (
    LLMAPIAuthorizationError,
    LLMAPIRateLimitError,
    LLMAPITokenLimitError,
    LLMAPIClientError,
    LLMAPIServerError,
    LLMAPITimeoutError,
    LLMAPIUsageLimitError,
)
from ..transports import (
    SSEEvent,
    SyncTransport,
    TransportRequest,
    create_sync_transport,
)

logger = logging.getLogger(__name__)


@dataclass
class OpenAICompatibleSyncClient:
    api_key: str
    endpoint: str
    transport: str = "requests"
    _sync_transport: SyncTransport = field(
        init=False,
        repr=False,
        compare=False,
    )
    _api_variant_override: str | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _diagnostic_formatter: Callable[[object], str] | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        self._sync_transport = self._create_sync_transport(self.transport)

    def _create_sync_transport(self, transport: str) -> SyncTransport:
        return create_sync_transport(transport)

    def __repr__(self) -> str:
        masked = f"{self.api_key[:8]}...{self.api_key[-4:]}" if len(self.api_key) > 12 else "***"
        result = f"OpenAICompatibleSyncClient(api_key='{masked}', endpoint='{self.endpoint}')"
        return self._format_diagnostic(result)

    def _set_diagnostic_formatter(
        self, formatter: Callable[[object], str] | None
    ) -> None:
        self._diagnostic_formatter = formatter

    def _format_diagnostic(self, value: object) -> Any:
        formatter = self._diagnostic_formatter
        return formatter(value) if formatter is not None else value

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def complete(self, model: str, timeout: float | None = None, **kwargs):
        if self._should_use_responses_api(model):
            return self.responses(model=model, timeout=timeout, **kwargs)
        return self.chat_completion(model=model, timeout=timeout, **kwargs)

    def stream(self, model: str, timeout: float | None = None, **kwargs) -> Iterator[SSEEvent]:
        """Stream raw events from the configured API."""
        if self._should_use_responses_api(model):
            return self.stream_responses(model=model, timeout=timeout, **kwargs)
        return self.stream_chat_completion(model=model, timeout=timeout, **kwargs)

    def chat_completion(self, model: str, timeout: float | None = None, **kwargs):
        url = f"{self.endpoint}/chat/completions"
        payload = self._prepare_chat_payload_for_model(model, kwargs)
        response = self._send_request(url, payload, timeout)
        return response.json()

    def responses(self, model: str, timeout: float | None = None, **kwargs):
        url = f"{self.endpoint}/responses"
        payload = self._prepare_responses_payload_for_model(model, kwargs)
        response = self._send_request(url, payload, timeout)
        return response.json()

    def stream_chat_completion(
        self, model: str, timeout: float | None = None, **kwargs
    ) -> Iterator[SSEEvent]:
        """Stream raw Chat Completions SSE events without interpreting them."""
        url = f"{self.endpoint}/chat/completions"
        payload = self._prepare_chat_payload_for_model(model, kwargs, streaming=True)
        payload["stream"] = True
        return self._stream_request(url, payload, timeout)

    def stream_responses(
        self, model: str, timeout: float | None = None, **kwargs
    ) -> Iterator[SSEEvent]:
        """Stream raw Responses API SSE events without interpreting them."""
        url = f"{self.endpoint}/responses"
        payload = self._prepare_responses_payload_for_model(model, kwargs)
        payload["stream"] = True
        return self._stream_request(url, payload, timeout)

    def _should_use_responses_api(self, model: str) -> bool:
        return self._api_variant_override == "responses"

    def _prepare_chat_payload_for_model(
        self, model: str, kwargs: dict, *, streaming: bool = False
    ) -> dict:
        return _prepare_compatible_chat_payload(model, kwargs, streaming=streaming)

    def _set_api_variant(self, api_variant: str | None) -> None:
        """Set the instance-local protocol selected for this connection."""
        self._api_variant_override = api_variant

    def _prepare_responses_payload_for_model(self, model: str, kwargs: dict) -> dict:
        return _prepare_compatible_responses_payload(model, kwargs)

    def _send_request(self, url: str, payload: dict, timeout: float | None = None):
        return self._sync_transport.post_json(
            TransportRequest(
                url=url,
                headers=self._headers(),
                payload=payload,
                timeout=timeout,
            ),
            http_error_handler=self._handle_http_error,
        )

    def _stream_request(
        self, url: str, payload: dict, timeout: float | None = None
    ) -> Iterator[SSEEvent]:
        events = self._sync_transport.post_sse(
            TransportRequest(
                url=url,
                headers=self._headers(),
                payload=payload,
                timeout=timeout,
            ),
            http_error_handler=self._handle_http_error,
            stream_error_handler=self._handle_stream_error,
        )
        try:
            for event in events:
                if self._is_failed_stream_event(event):
                    self._handle_stream_error(event)
                yield event
        finally:
            events.close()

    @staticmethod
    def _is_failed_stream_event(event: SSEEvent) -> bool:
        if event.event == "response.failed":
            return True
        return isinstance(event.data, Mapping) and event.data.get("type") == "response.failed"

    def _handle_stream_error(self, event: SSEEvent) -> None:
        payload = event.data if isinstance(event.data, Mapping) else {}
        response = payload.get("response")
        response_error = response.get("error") if isinstance(response, Mapping) else None
        error = response_error if isinstance(response_error, Mapping) else payload.get("error")
        error = error if isinstance(error, Mapping) else payload

        error_type = error.get("type") or error.get("code")
        error_message = error.get("message")
        detail = str(error_message or error_type or payload)
        self._raise_mapped_error(
            status_code=None,
            error_type=str(error_type) if error_type else None,
            detail=detail,
        )

    def _handle_http_error(self, http_err):
        status_code = http_err.response.status_code
        try:
            error_json = http_err.response.json()
            error = error_json.get("error", {})
            error_type = error.get("type") or error.get("code")
            error_message = error.get("message")
        except Exception as e:
            self._warn_http_parse_fallback(e)
            error_type = None
            error_message = None
        detail = error_message or str(http_err)
        self._raise_mapped_error(status_code, error_type, detail)

    def _warn_http_parse_fallback(self, error: Exception) -> None:
        logger.warning("Failed to parse error response: %s", self._format_diagnostic(error))

    @staticmethod
    def _raise_mapped_error(
        status_code: int | None, error_type: str | None, detail: str
    ) -> None:
        error_map = {
            401: LLMAPIAuthorizationError,
            429: LLMAPIRateLimitError,
        }
        if status_code in error_map:
            raise error_map[status_code](detail=detail)
        if error_type in (
            *LLMAPIAuthorizationError.openai_api_errors,
            "invalid_api_key",
            "authentication_error",
        ):
            raise LLMAPIAuthorizationError(detail=detail)
        if error_type in (
            *LLMAPIRateLimitError.openai_api_errors,
            "rate_limit_exceeded",
        ):
            raise LLMAPIRateLimitError(detail=detail)
        if error_type in LLMAPITokenLimitError.openai_api_errors:
            raise LLMAPITokenLimitError(detail=detail)
        if error_type in LLMAPIUsageLimitError.openai_api_errors:
            raise LLMAPIUsageLimitError(detail=detail)
        if error_type in (
            *LLMAPIServerError.openai_api_errors,
            "server_error",
            "internal_error",
            "overloaded_error",
        ):
            raise LLMAPIServerError(detail=detail)
        if error_type in (*LLMAPITimeoutError.openai_api_errors, "timeout"):
            raise LLMAPITimeoutError(detail=detail)
        if status_code is not None and 400 <= status_code < 500:
            raise LLMAPIClientError(detail=detail)
        if status_code is not None and 500 <= status_code < 600:
            raise LLMAPIServerError(detail=detail)
        raise LLMAPIClientError(detail=detail)


def _prepare_compatible_chat_payload(
    model: str, kwargs: dict, *, streaming: bool
) -> dict:
    """Prepare the fixed compatible Chat Completions wire policy."""
    payload = {"model": model, **dict(kwargs)}

    max_tokens = payload.pop("max_tokens", None)
    if max_tokens is not None:
        payload["max_completion_tokens"] = max_tokens

    _omit_numeric_sampling_defaults(payload)

    if streaming:
        stream_options = payload.get("stream_options")
        payload["stream_options"] = {
            **(dict(stream_options) if isinstance(stream_options, Mapping) else {}),
            "include_usage": True,
        }

    return payload


def _prepare_compatible_responses_payload(model: str, kwargs: dict) -> dict:
    """Prepare stateless Responses fields without Chat or native policy fields."""
    payload = {"model": model, **dict(kwargs)}

    if "messages" in payload:
        payload["input"] = payload.pop("messages")

    max_tokens = payload.pop("max_tokens", None)
    if max_tokens is not None:
        payload["max_output_tokens"] = max_tokens
    elif payload.get("max_output_tokens") is None:
        payload.pop("max_output_tokens", None)

    for key in (
        "max_completion_tokens",
        "stream_options",
        "reasoning_effort",
        "reasoning_level",
        "capture_reasoning",
        "reasoning",
        "previous_response_id",
        "conversation",
    ):
        payload.pop(key, None)

    _omit_numeric_sampling_defaults(payload)

    payload["store"] = False
    return payload


def _omit_numeric_sampling_defaults(payload: dict) -> None:
    """Omit the shared numeric defaults while preserving booleans and overrides."""
    for key in ("temperature", "top_p"):
        value = payload.get(key)
        is_numeric_default = (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value == 1
        )
        if is_numeric_default:
            payload.pop(key)


__all__ = ["OpenAICompatibleSyncClient"]
