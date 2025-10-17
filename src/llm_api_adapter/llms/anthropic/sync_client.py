from dataclasses import dataclass
import logging

import requests

from ...errors.llm_api_error import (
    LLMAPIAuthorizationError,
    LLMAPIRateLimitError,
    LLMAPITokenLimitError,
    LLMAPIClientError,
    LLMAPIServerError,
    LLMAPITimeoutError,
)

logger = logging.getLogger(__name__)

@dataclass
class ClaudeSyncClient:
    api_key: str
    endpoint: str = "https://api.anthropic.com/v1"
    api_version: str = "2023-06-01"

    def _headers(self):
        return {
            "x-api-key": self.api_key,
            "anthropic-version": self.api_version,
            "Content-Type": "application/json"
        }

    def chat_completion(self, model: str, **kwargs):
        if model.startswith(("claude-sonnet-4-5", "claude-opus-4-1")):
            if "top_p" in kwargs:
                kwargs.pop("top_p", None)
        url = f"{self.endpoint}/messages"
        payload = {"model": model, **kwargs}
        response = self._send_request(url, payload)
        return response.json()

    def _send_request(self, url, payload):
        try:
            response = requests.post(
                url, headers=self._headers(), json=payload
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

    def _handle_http_error(self, http_err):
        status_code = http_err.response.status_code
        try:
            error_json = http_err.response.json()
            error_type = error_json.get("type")
            error_message = error_json.get("message")
        except Exception:
            error_type = None
            error_message = None
        detail = error_message or str(http_err)
        error_map = {
            401: LLMAPIAuthorizationError,
            429: LLMAPIRateLimitError,
        }
        if status_code in error_map:
            raise error_map[status_code](detail=detail)
        elif error_type in LLMAPIAuthorizationError.anthropic_api_errors:
            raise LLMAPIAuthorizationError(detail=detail)
        elif error_type in LLMAPIRateLimitError.anthropic_api_errors:
            raise LLMAPIRateLimitError(detail=detail)
        elif error_type in LLMAPITokenLimitError.anthropic_api_errors:
            raise LLMAPITokenLimitError(detail=detail)
        elif 400 <= status_code < 500:
            raise LLMAPIClientError(detail=detail)
        elif 500 <= status_code < 600:
            raise LLMAPIServerError(detail=detail)
        else:
            raise LLMAPIClientError(detail=detail)
