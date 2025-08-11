from dataclasses import dataclass
import logging

import requests

from ...errors.llm_api_error import (
    LLMAPIAuthorizationError,
    LLMAPIRateLimitError,
    LLMAPIClientError,
    LLMAPIServerError,
    LLMAPITimeoutError,
)

logger = logging.getLogger(__name__)

@dataclass
class GeminiSyncClient:
    api_key: str
    endpoint: str = "https://generativelanguage.googleapis.com/v1beta"

    def _headers(self):
        return {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json"
        }

    def chat_completion(self, model: str, **kwargs):
        url = f"{self.endpoint}/models/{model}:generateContent"
        payload = {"model": model, **kwargs}
        response = self._send_request(url, payload)
        return response.json()

    def _send_request(self, url, payload):
        try:
            response = requests.post(
                url, headers=self._headers(), json=payload, timeout=30
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
        error_map = {
            401: LLMAPIAuthorizationError,
            429: LLMAPIRateLimitError,
        }
        if status_code in error_map:
            raise error_map[status_code](detail=detail)
        elif error_status in LLMAPIAuthorizationError.google_api_errors:
            raise LLMAPIAuthorizationError(detail=detail)
        elif error_status in LLMAPIRateLimitError.google_api_errors:
            raise LLMAPIRateLimitError(detail=detail)
        elif 400 <= status_code < 500:
            raise LLMAPIClientError(detail=detail)
        elif (
            500 <= status_code < 600
            or error_status in LLMAPIServerError.google_api_errors
        ):
            raise LLMAPIServerError(detail=detail)
        else:
            raise LLMAPIClientError(detail=detail)
