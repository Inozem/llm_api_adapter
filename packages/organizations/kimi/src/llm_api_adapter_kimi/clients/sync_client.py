"""Synchronous client for Kimi's official Chat Completions endpoint."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from llm_api_adapter.errors.llm_api_error import (
    LLMAPIAuthorizationError,
    LLMAPIClientError,
    LLMAPIRateLimitError,
    LLMAPIServerError,
    LLMAPITimeoutError,
    LLMAPITokenLimitError,
    LLMAPIUsageLimitError,
)
from llm_api_adapter.llms.transports import JSONResponse, SyncTransport, TransportRequest


KIMI_CHAT_COMPLETIONS_URL = "https://api.moonshot.ai/v1/chat/completions"


class KimiSyncClient:
    """Submit Kimi Chat Completions requests through the Core sync transport."""

    def __init__(self, transport: SyncTransport) -> None:
        self.transport = transport

    def chat(
        self,
        *,
        api_key: str,
        payload: dict[str, Any],
        timeout_s: float | None,
    ) -> dict[str, Any]:
        """Post one JSON request and validate the top-level response envelope."""
        response: JSONResponse = self.transport.post_json(
            TransportRequest(
                url=KIMI_CHAT_COMPLETIONS_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                payload=payload,
                timeout=timeout_s,
            ),
            http_error_handler=self._handle_http_error,
        )
        response_data = response.json()
        if not isinstance(response_data, dict):
            raise LLMAPIClientError(
                detail="Kimi Chat Completions returned a non-object response",
            )
        return response_data

    @classmethod
    def _handle_http_error(cls, http_error: Any) -> None:
        response = getattr(http_error, "response", None)
        status_code = getattr(response, "status_code", None)
        detail = str(http_error)
        error_type: str | None = None
        try:
            payload = response.json()
        except Exception:
            payload = None
        if isinstance(payload, Mapping):
            error = payload.get("error", payload)
            if isinstance(error, Mapping):
                raw_type = error.get("type") or error.get("code")
                error_type = raw_type if isinstance(raw_type, str) else None
                raw_detail = error.get("message") or error.get("detail")
                if isinstance(raw_detail, str) and raw_detail:
                    detail = raw_detail
        cls._raise_mapped_error(status_code, error_type, detail)

    @staticmethod
    def _raise_mapped_error(
        status_code: int | None,
        error_type: str | None,
        detail: str,
    ) -> None:
        normalized_type = error_type.lower() if error_type else ""
        if status_code in {401, 403} or normalized_type in {
            "authentication_error",
            "authorization_error",
            "invalid_api_key",
            "invalid_authentication_error",
            "permission_denied",
        }:
            raise LLMAPIAuthorizationError(detail=detail)
        if status_code == 429 or normalized_type in {
            "rate_limit_error",
            "rate_limit_exceeded",
        }:
            raise LLMAPIRateLimitError(detail=detail)
        if status_code in {408, 504} or normalized_type in {
            "timeout",
            "timeout_error",
        }:
            raise LLMAPITimeoutError(detail=detail)
        if normalized_type in {
            "context_length_exceeded",
            "input_too_long",
            "max_tokens_exceeded",
            "max_output_tokens_exceeded",
        }:
            raise LLMAPITokenLimitError(detail=detail)
        if normalized_type in {
            "insufficient_quota",
            "quota_exceeded",
            "usage_limit_exceeded",
        }:
            raise LLMAPIUsageLimitError(detail=detail)
        if status_code is not None and 500 <= status_code < 600:
            raise LLMAPIServerError(detail=detail)
        if normalized_type in {"api_error", "internal_error", "overloaded_error"}:
            raise LLMAPIServerError(detail=detail)
        raise LLMAPIClientError(detail=detail)


__all__ = ["KIMI_CHAT_COMPLETIONS_URL", "KimiSyncClient"]
