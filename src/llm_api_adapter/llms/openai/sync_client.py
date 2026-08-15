from dataclasses import dataclass, field
import logging
from typing import Iterator, Mapping

# Kept as a module attribute for existing white-box patch points. Requests are
# now called by ``RequestsSyncTransport``.
import requests

from ...errors.llm_api_error import (
    LLMAPIAuthorizationError,
    LLMAPIRateLimitError,
    LLMAPITokenLimitError,
    LLMAPIClientError,
    LLMAPIServerError,
    LLMAPITimeoutError,
    LLMAPIUsageLimitError,
)
from ...llm_registry.llm_registry import CategoricalReasoningCapability
from ..request_rules import (
    apply_model_request_rules,
    model_api_variant,
    model_spec_for,
)
from ..requests_transport import RequestsSyncTransport
from ..transports import SSEEvent, SyncTransport, TransportRequest

logger = logging.getLogger(__name__)


@dataclass
class OpenAISyncClient:
    api_key: str
    endpoint: str = "https://api.openai.com/v1"
    _sync_transport: SyncTransport = field(
        default_factory=RequestsSyncTransport,
        init=False,
        repr=False,
        compare=False,
    )

    def __repr__(self) -> str:
        masked = f"{self.api_key[:8]}...{self.api_key[-4:]}" if len(self.api_key) > 12 else "***"
        return f"OpenAISyncClient(api_key='{masked}', endpoint='{self.endpoint}')"

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
        """Stream raw OpenAI events from the API appropriate for ``model``."""
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
        payload = self._prepare_chat_payload_for_model(model, kwargs)
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
        return model_api_variant("openai", model) == "responses"

    def _prepare_chat_payload_for_model(self, model: str, kwargs: dict) -> dict:
        payload, _ = apply_model_request_rules(
            "openai",
            model,
            {"model": model, **kwargs},
        )
        return payload

    def _prepare_responses_payload_for_model(self, model: str, kwargs: dict) -> dict:
        payload = {"model": model, **dict(kwargs)}
        if "messages" in payload:
            payload["input"] = payload.pop("messages")
        if "max_tokens" in payload:
            payload["max_output_tokens"] = payload.pop("max_tokens")
        reasoning_effort = payload.pop("reasoning_effort", None)
        capture_reasoning = payload.pop("capture_reasoning", False)
        reasoning: dict = {}
        if reasoning_effort is not None:
            if self._uses_minimal_reasoning_for_none(model) and reasoning_effort == "none":
                reasoning_effort = "minimal"
            reasoning["effort"] = reasoning_effort
        if capture_reasoning:
            reasoning["summary"] = "auto"
        if reasoning:
            payload["reasoning"] = reasoning
        payload, _ = apply_model_request_rules("openai", model, payload)
        return payload

    @staticmethod
    def _uses_minimal_reasoning_for_none(model: str) -> bool:
        """Map legacy ``none`` to the first native effort level when required."""
        model_spec = model_spec_for("openai", model)
        capability = model_spec.reasoning_capability if model_spec else None
        return (
            isinstance(capability, CategoricalReasoningCapability)
            and "none" not in capability.allowed_values
            and "minimal" in capability.allowed_values
        )

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
            logger.warning("Failed to parse error response: %s", e)
            error_type = None
            error_message = None
        detail = error_message or str(http_err)
        self._raise_mapped_error(status_code, error_type, detail)

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
