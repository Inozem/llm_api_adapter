from dataclasses import dataclass
import logging
from typing import Iterator, Mapping

import requests

from ...errors.llm_api_error import (
    LLMAPIAuthorizationError,
    LLMAPIRateLimitError,
    LLMAPIClientError,
    LLMAPIServerError,
    LLMAPITimeoutError,
)
from ..streaming import SSEEvent, stream_request

logger = logging.getLogger(__name__)

@dataclass
class GeminiSyncClient:
    api_key: str
    endpoint: str = "https://generativelanguage.googleapis.com/v1beta"

    def __repr__(self) -> str:
        masked = f"{self.api_key[:8]}...{self.api_key[-4:]}" if len(self.api_key) > 12 else "***"
        return f"GeminiSyncClient(api_key='{masked}', endpoint='{self.endpoint}')"

    def _headers(self):
        return {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json"
        }

    def chat_completion(self, model: str, timeout_s: float | None = None, **kwargs):
        url = f"{self.endpoint}/models/{model}:generateContent"
        payload = self._prepare_chat_payload_for_model(model, kwargs)
        response = self._send_request(url, payload, timeout_s)
        return response.json()

    def stream(
        self, model: str, timeout_s: float | None = None, **kwargs
    ) -> Iterator[SSEEvent]:
        """Stream raw GenerateContentResponse chunks from Gemini."""
        url = f"{self.endpoint}/models/{model}:streamGenerateContent?alt=sse"
        payload = self._prepare_chat_payload_for_model(model, kwargs)
        return self._stream_request(url, payload, timeout_s)

    def _prepare_chat_payload_for_model(self, model: str, kwargs: dict) -> dict:
        gen_cfg = kwargs.get("generationConfig", {})
        if "maxOutputTokens" in gen_cfg:
            if model.startswith(("gemini-2.5")):
                gen_cfg.pop("maxOutputTokens", None)
                kwargs["generationConfig"] = gen_cfg
        return {"model": model, **kwargs}

    def _send_request(self, url: str, payload: dict, timeout_s: float | None = None):
        try:
            response = requests.post(
                url, headers=self._headers(), json=payload,  timeout=timeout_s,
            )
            response.raise_for_status()
        except requests.exceptions.Timeout as e:
            logger.error(f"Request timeout: {e}")
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
            for event in events:
                if self._is_failed_stream_event(event):
                    self._handle_stream_error(event)
                yield event
        finally:
            events.close()

    @staticmethod
    def _is_failed_stream_event(event: SSEEvent) -> bool:
        return (
            isinstance(event.data, Mapping)
            and isinstance(event.data.get("error"), Mapping)
        )

    def _handle_stream_error(self, event: SSEEvent) -> None:
        payload = event.data if isinstance(event.data, Mapping) else {}
        error = payload.get("error")
        error = error if isinstance(error, Mapping) else payload
        error_status = error.get("status") or error.get("code")
        error_message = error.get("message")
        detail = str(error_message or error_status or payload)
        self._raise_mapped_error(
            status_code=None,
            error_status=str(error_status) if error_status else None,
            error_message=str(error_message) if error_message else None,
            detail=detail,
        )

    def _handle_http_error(self, http_err):
        status_code = http_err.response.status_code
        try:
            error_json = http_err.response.json()
            error_status = error_json.get("error", {}).get("status", "")
            error_message = error_json.get("error", {}).get("message")
        except Exception:
            logger.warning("Failed to parse error response JSON", exc_info=True)
            error_status = ""
            error_message = None
        detail = error_message or str(http_err)
        self._raise_mapped_error(status_code, error_status, error_message, detail)

    def _raise_mapped_error(
        self,
        status_code: int | None,
        error_status: str | None,
        error_message: str | None,
        detail: str,
    ) -> None:
        if self._is_google_auth_error(status_code, error_status, error_message):
            raise LLMAPIAuthorizationError(detail=detail)
        if status_code == 429 or error_status in LLMAPIRateLimitError.google_api_errors:
            raise LLMAPIRateLimitError(detail=detail)
        if status_code is not None and 400 <= status_code < 500:
            raise LLMAPIClientError(detail=detail)
        if (
            (status_code is not None and 500 <= status_code < 600)
            or error_status in LLMAPIServerError.google_api_errors
        ):
            raise LLMAPIServerError(detail=detail)
        raise LLMAPIClientError(detail=detail)

    @staticmethod
    def _is_google_auth_error(
        status_code: int | None,
        error_status: str | None,
        error_message: str | None,
    ) -> bool:
        if status_code in (401, 403):
            return True
        if error_status in LLMAPIAuthorizationError.google_api_errors:
            return True
        if error_message:
            msg = error_message.lower()
            return any(
                k in msg
                for k in (
                    "api key not valid",
                    "api key invalid",
                    "api key not found",
                    "invalid api key",
                )
            )
        return False
