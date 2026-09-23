"""Synchronous client for Z.ai's official Chat Completions endpoint."""

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
from llm_api_adapter.llms.transports import (
    JSONResponse,
    SSEEvent,
    SyncTransport,
    TransportRequest,
)


ZAI_BASE_URL = "https://api.z.ai/api/paas/v4"
ZAI_CHAT_COMPLETIONS_URL = f"{ZAI_BASE_URL}/chat/completions"


class ZaiSyncClient:
    """Submit Z.ai Chat Completions requests through Core's sync transport."""

    def __init__(self, transport: SyncTransport) -> None:
        self.transport = transport

    def chat(
        self,
        *,
        api_key: str,
        payload: dict[str, Any],
        timeout_s: float | None,
    ) -> dict[str, Any]:
        """Post one JSON request and validate its top-level response envelope."""
        response: JSONResponse = self.transport.post_json(
            TransportRequest(
                url=ZAI_CHAT_COMPLETIONS_URL,
                headers=self._headers(api_key),
                payload=payload,
                timeout=timeout_s,
            ),
            http_error_handler=self._handle_http_error,
        )
        response_data = response.json()
        if not isinstance(response_data, dict):
            raise LLMAPIClientError(
                detail="Z.ai Chat Completions returned a non-object response",
            )
        return response_data

    def stream(
        self,
        *,
        api_key: str,
        payload: dict[str, Any],
        timeout_s: float | None,
    ):
        """Open one Z.ai SSE response and request terminal usage when available."""
        stream_payload = dict(payload)
        stream_payload["stream"] = True
        stream_payload.setdefault("stream_options", {"include_usage": True})
        return self.transport.post_sse(
            TransportRequest(
                url=ZAI_CHAT_COMPLETIONS_URL,
                headers=self._headers(api_key),
                payload=stream_payload,
                timeout=timeout_s,
            ),
            http_error_handler=self._handle_http_error,
            stream_error_handler=self._handle_stream_error,
        )

    @staticmethod
    def _headers(api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    @classmethod
    def _handle_http_error(cls, http_error: Any) -> None:
        """Map a native HTTP exception to the normalized Core hierarchy."""
        response = getattr(http_error, "response", None)
        raw_status = getattr(response, "status_code", None)
        status_code = raw_status if isinstance(raw_status, int) else None
        detail = str(http_error)
        error_type: str | None = None
        try:
            payload = response.json() if response is not None else None
        except Exception:
            payload = None

        if isinstance(payload, Mapping):
            error = payload.get("error", payload)
            if isinstance(error, Mapping):
                raw_type = error.get("type") or error.get("code")
                error_type = raw_type if isinstance(raw_type, str) else None
                raw_detail = (
                    error.get("message")
                    or error.get("detail")
                    or error.get("error")
                )
                if raw_detail:
                    detail = str(raw_detail)
            elif payload.get("message"):
                detail = str(payload["message"])

        cls._raise_mapped_error(status_code, error_type, detail)

    @classmethod
    def _handle_stream_error(cls, event: SSEEvent) -> None:
        """Map an OpenAI-compatible Z.ai SSE error frame."""
        payload = event.data if isinstance(event.data, Mapping) else {}
        error = payload.get("error", payload)
        if not isinstance(error, Mapping):
            raise LLMAPIClientError(
                detail="Z.ai Chat Completions returned an invalid SSE error event",
            )
        raw_type = error.get("type") or error.get("code")
        error_type = raw_type if isinstance(raw_type, str) else None
        raw_detail = error.get("message") or error.get("detail")
        detail = (
            str(raw_detail)
            if raw_detail
            else "Z.ai Chat Completions returned an SSE error"
        )
        cls._raise_mapped_error(None, error_type, detail)

    @staticmethod
    def _raise_mapped_error(
        status_code: int | None,
        error_type: str | None,
        detail: str,
    ) -> None:
        normalized_type = error_type.strip().lower() if error_type else ""
        normalized_detail = detail.strip().lower()

        if status_code in {401, 403} or normalized_type in {
            "authentication_error",
            "authorization_error",
            "invalid_api_key",
            "invalid_authentication_error",
            "permission_denied",
        } or any(
            marker in normalized_detail
            for marker in (
                "api key",
                "api_key",
                "authentication",
                "authorization",
                "unauthorized",
                "invalid credentials",
            )
        ):
            raise LLMAPIAuthorizationError(detail=detail)

        if status_code == 429 or normalized_type in {
            "rate_limit_error",
            "rate_limit_exceeded",
            "rate_limit_reached",
        }:
            raise LLMAPIRateLimitError(detail=detail)

        if status_code == 402 or normalized_type in {
            "insufficient_balance",
            "insufficient_quota",
            "quota_exceeded",
            "usage_limit_exceeded",
        }:
            raise LLMAPIUsageLimitError(detail=detail)

        if normalized_type in {
            "context_length_exceeded",
            "input_too_long",
            "max_tokens_exceeded",
            "max_output_tokens_exceeded",
        }:
            raise LLMAPITokenLimitError(detail=detail)

        if status_code in {408, 504} or normalized_type in {
            "timeout",
            "timeout_error",
        }:
            raise LLMAPITimeoutError(detail=detail)

        if status_code is not None and 500 <= status_code < 600:
            raise LLMAPIServerError(detail=detail)

        if normalized_type in {"api_error", "internal_error", "overloaded_error"}:
            raise LLMAPIServerError(detail=detail)

        raise LLMAPIClientError(detail=detail)


# Keep the explicit protocol name available to package-local callers.
ZaiChatCompletionsSyncClient = ZaiSyncClient


__all__ = [
    "ZAI_BASE_URL",
    "ZAI_CHAT_COMPLETIONS_URL",
    "ZaiChatCompletionsSyncClient",
    "ZaiSyncClient",
]
