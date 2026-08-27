"""Synchronous client for xAI's official Responses API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from llm_api_adapter.errors.llm_api_error import (
    LLMAPIAuthorizationError,
    LLMAPIClientError,
    LLMAPIError,
    LLMAPIRateLimitError,
    LLMAPIServerError,
    LLMAPITimeoutError,
    LLMAPITokenLimitError,
    LLMAPIUsageLimitError,
)
from llm_api_adapter.llms.transports import (
    JSONResponse,
    SyncTransport,
    TransportRequest,
    create_sync_transport,
)


_XAI_RESPONSES_URL = "https://api.x.ai/v1/responses"


@dataclass(repr=False)
class XAIResponsesSyncClient:
    """Send xAI Responses API requests through a selected core transport."""

    api_key: str
    transport: str = "requests"
    endpoint: str = _XAI_RESPONSES_URL
    _sync_transport: SyncTransport = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._sync_transport = create_sync_transport(self.transport)

    def create(
        self,
        *,
        model: str,
        timeout: float | None = None,
        **parameters: Any,
    ) -> dict[str, Any]:
        """Create one Responses API response and validate its JSON envelope."""
        response: JSONResponse = self._sync_transport.post_json(
            TransportRequest(
                url=self.endpoint,
                headers=self._headers(),
                payload={"model": model, **parameters},
                timeout=timeout,
            ),
            http_error_handler=self._handle_http_error,
        )
        payload = response.json()
        if not isinstance(payload, dict):
            raise LLMAPIClientError(
                detail="xAI Responses API returned a non-object response",
            )
        return payload

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _handle_http_error(self, error: Any) -> None:
        response = getattr(error, "response", None)
        status_code = getattr(response, "status_code", None)
        payload: Mapping[str, Any] = {}
        if response is not None:
            try:
                candidate = response.json()
                if isinstance(candidate, Mapping):
                    payload = candidate
            except Exception:
                pass
        error_data = payload.get("error", payload)
        if not isinstance(error_data, Mapping):
            error_data = {}
        error_type = error_data.get("type") or error_data.get("code")
        detail = error_data.get("message") or str(error)
        self._raise_mapped_error(
            status_code=status_code if isinstance(status_code, int) else None,
            error_type=str(error_type) if error_type else None,
            detail=str(detail),
        )

    @staticmethod
    def _raise_mapped_error(
        *,
        status_code: int | None,
        error_type: str | None,
        detail: str,
    ) -> None:
        normalized_type = (error_type or "").lower()
        if status_code in {401, 403} or normalized_type in {
            "authentication_error",
            "authorization_error",
            "invalid_api_key",
            "permission_denied",
        }:
            raise LLMAPIAuthorizationError(detail=detail)
        if status_code == 429 or normalized_type in {
            "rate_limit_error",
            "rate_limit_exceeded",
        }:
            raise LLMAPIRateLimitError(detail=detail)
        if normalized_type in {
            "context_length_exceeded",
            "input_too_long",
            "max_output_tokens_exceeded",
        }:
            raise LLMAPITokenLimitError(detail=detail)
        if normalized_type in {"insufficient_quota", "usage_limit_exceeded"}:
            raise LLMAPIUsageLimitError(detail=detail)
        if status_code in {408, 504} or normalized_type in {
            "timeout",
            "timeout_error",
        }:
            raise LLMAPITimeoutError(detail=detail)
        if status_code is not None and 500 <= status_code < 600:
            raise LLMAPIServerError(detail=detail)
        raise LLMAPIClientError(detail=detail)


__all__ = ["XAIResponsesSyncClient"]
