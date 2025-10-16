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
class OpenAISyncClient:
    api_key: str
    endpoint: str = "https://api.openai.com/v1"

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def chat_completion(self, model: str, **kwargs):
        if model.startswith(("gpt-4.1", "gpt-5", "o1")):
            if "max_tokens" in kwargs:
                kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
        url = f"{self.endpoint}/chat/completions"
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
            logger.error("Timeout error: %s", e)
            raise LLMAPITimeoutError(detail=str(e))
        except requests.exceptions.HTTPError as http_err:
            logger.error("HTTP error: %s", http_err)
            self._handle_http_error(http_err)
        except requests.exceptions.RequestException as e:
            logger.error("Request exception: %s", e)
            raise LLMAPIClientError(detail=str(e))
        return response

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
        error_map = {
            401: LLMAPIAuthorizationError,
            429: LLMAPIRateLimitError,
        }
        if status_code in error_map:
            raise error_map[status_code](detail=detail)
        elif error_type in LLMAPIAuthorizationError.openai_api_errors:
            raise LLMAPIAuthorizationError(detail=detail)
        elif error_type in LLMAPIRateLimitError.openai_api_errors:
            raise LLMAPIRateLimitError(detail=detail)
        elif error_type in LLMAPITokenLimitError.openai_api_errors:
            raise LLMAPITokenLimitError(detail=detail)
        elif 400 <= status_code < 500:
            raise LLMAPIClientError(detail=detail)
        elif 500 <= status_code < 600:
            raise LLMAPIServerError(detail=detail)
        else:
            raise LLMAPIClientError(detail=detail)
