from dataclasses import dataclass
import logging
from typing import Iterator, Mapping

import requests

from ...errors.llm_api_error import (
    LLMAPIAuthorizationError,
    LLMAPIRateLimitError,
    LLMAPITokenLimitError,
    LLMAPIClientError,
    LLMAPIServerError,
    LLMAPITimeoutError,
)
from ..streaming import SSEEvent, stream_request

logger = logging.getLogger(__name__)

@dataclass
class ClaudeSyncClient:
    api_key: str
    endpoint: str = "https://api.anthropic.com/v1"
    api_version: str = "2023-06-01"

    def __repr__(self) -> str:
        masked = f"{self.api_key[:8]}...{self.api_key[-4:]}" if len(self.api_key) > 12 else "***"
        return f"ClaudeSyncClient(api_key='{masked}', endpoint='{self.endpoint}', api_version='{self.api_version}')"

    def _headers(self):
        return {
            "x-api-key": self.api_key,
            "anthropic-version": self.api_version,
            "Content-Type": "application/json"
        }

    def chat_completion(self, model: str, timeout_s: float | None = None, **kwargs):
        url = f"{self.endpoint}/messages"
        payload = self._prepare_chat_payload_for_model(model, kwargs)
        response = self._send_request(url, payload, timeout_s)
        return response.json()

    def stream(
        self, model: str, timeout_s: float | None = None, **kwargs
    ) -> Iterator[SSEEvent]:
        """Stream raw Messages API SSE events without interpreting content blocks."""
        url = f"{self.endpoint}/messages"
        payload = self._prepare_chat_payload_for_model(model, kwargs)
        payload["stream"] = True
        return self._stream_request(url, payload, timeout_s)

    def _prepare_chat_payload_for_model(self, model: str, kwargs: dict) -> dict:
        budget_tokens = kwargs.pop("budget_tokens", None)
        effort = kwargs.pop("effort", None)
        is_adaptive_thinking = kwargs.pop("is_adaptive_thinking", False)
        if is_adaptive_thinking:
            kwargs.pop("top_p", None)
            if effort:
                kwargs["thinking"] = {"type": "adaptive"}
                existing = kwargs.get("output_config", {})
                kwargs["output_config"] = {**existing, "effort": effort}
        else:
            if model.startswith(
                ("claude-sonnet-4-5", "claude-opus-4-1", "claude-haiku-4-5", "claude-opus-4-5")
            ):
                kwargs.pop("top_p", None)
            if budget_tokens:
                kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget_tokens}
        return {"model": model, **kwargs}

    def _send_request(self, url: str, payload: dict, timeout_s: float | None = None):
        try:
            response = requests.post(
                url, headers=self._headers(), json=payload, timeout=timeout_s,
            )
            response.raise_for_status()
        except requests.exceptions.Timeout as e:
            logger.error(f"Request timed out: {e}")
            raise LLMAPITimeoutError(detail=str(e))
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP error occurred: {http_err}")
            self._handle_http_error(http_err)
        except requests.exceptions.RequestException as e:
            logger.error(f"Request exception: {e}")
            raise LLMAPIClientError(detail=str(e))
        return response

    def _stream_request(
        self, url: str, payload: dict, timeout_s: float | None = None
    ) -> Iterator[SSEEvent]:
        events = stream_request(
            url,
            headers=self._headers(),
            payload=payload,
            timeout=timeout_s,
            http_error_handler=self._handle_http_error,
            stream_error_handler=self._handle_stream_error,
        )
        try:
            yield from events
        finally:
            events.close()

    def _handle_stream_error(self, event: SSEEvent) -> None:
        payload = event.data if isinstance(event.data, Mapping) else {}
        error = payload.get("error")
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
            error_json = http_err.response.json().get("error")
            error_type = error_json.get("type")
            error_message = error_json.get("message")
        except Exception:
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
            *LLMAPIAuthorizationError.anthropic_api_errors,
            "authentication_error",
            "permission_error",
        ):
            raise LLMAPIAuthorizationError(detail=detail)
        if error_type in (
            *LLMAPIRateLimitError.anthropic_api_errors,
            "rate_limit_error",
        ):
            raise LLMAPIRateLimitError(detail=detail)
        if error_type in LLMAPITokenLimitError.anthropic_api_errors:
            raise LLMAPITokenLimitError(detail=detail)
        if error_type in (
            *LLMAPIServerError.anthropic_api_errors,
            "api_error",
            "overloaded_error",
        ):
            raise LLMAPIServerError(detail=detail)
        if status_code is not None and 400 <= status_code < 500:
            raise LLMAPIClientError(detail=detail)
        if status_code is not None and 500 <= status_code < 600:
            raise LLMAPIServerError(detail=detail)
        raise LLMAPIClientError(detail=detail)
