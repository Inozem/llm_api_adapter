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
            "Content-Type": "application/json",
        }

    def complete(self, model: str, timeout: float | None = None, **kwargs):
        if self._should_use_responses_api(model):
            return self.responses(model=model, timeout=timeout, **kwargs)
        return self.chat_completion(model=model, timeout=timeout, **kwargs)

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

    def _should_use_responses_api(self, model: str) -> bool:
        return model.startswith("gpt-5")

    def _prepare_chat_payload_for_model(self, model: str, kwargs: dict) -> dict:
        kwargs = dict(kwargs)
        if model.startswith(("gpt-4.1", "gpt-5", "o1")):
            if "max_tokens" in kwargs:
                kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
        if "reasoning_effort" in kwargs and model in ("gpt-5-nano", "gpt-5-mini"):
            if kwargs["reasoning_effort"] == "none":
                kwargs["reasoning_effort"] = "minimal"
        return {"model": model, **kwargs}

    def _prepare_responses_payload_for_model(self, model: str, kwargs: dict) -> dict:
        payload = {"model": model, **dict(kwargs)}
        if "messages" in payload:
            payload["input"] = payload.pop("messages")
        if "max_tokens" in payload:
            payload["max_output_tokens"] = payload.pop("max_tokens")
        reasoning_effort = payload.pop("reasoning_effort", None)
        if reasoning_effort is not None:
            if model in ("gpt-5-mini", "gpt-5-nano") and reasoning_effort == "none":
                reasoning_effort = "minimal"
            payload["reasoning"] = {"effort": reasoning_effort}
        return payload

    def _send_request(self, url: str, payload: dict, timeout: float | None = None):
        try:
            response = requests.post(
                url, headers=self._headers(), json=payload, timeout=timeout
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
