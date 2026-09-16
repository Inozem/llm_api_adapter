"""Synchronous client for DeepSeek's official Responses API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping

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
    create_sync_transport,
)


DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_RESPONSES_URL = f"{DEEPSEEK_BASE_URL}/responses"


@dataclass(repr=False)
class DeepSeekResponsesSyncClient:
    """Send Responses requests through one of Core's sync transports."""

    api_key: str
    transport: str = "requests"
    endpoint: str = DEEPSEEK_RESPONSES_URL
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
        """Create one Responses result and validate its JSON envelope."""
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
                detail="DeepSeek Responses API returned a non-object response",
            )
        return payload

    def stream(
        self,
        *,
        model: str,
        timeout: float | None = None,
        **parameters: Any,
    ) -> Iterator[SSEEvent]:
        """Stream semantic Responses events without provider-side retries."""
        return self._sync_transport.post_sse(
            TransportRequest(
                url=self.endpoint,
                headers=self._headers(),
                payload={
                    "model": model,
                    **parameters,
                    "stream": True,
                },
                timeout=timeout,
            ),
            http_error_handler=self._handle_http_error,
            stream_error_handler=self._handle_stream_error,
        )

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @classmethod
    def _handle_http_error(cls, error: Any) -> None:
        status_code, error_type, detail = cls._http_error_details(error)
        cls._raise_mapped_error(
            status_code=status_code,
            error_type=error_type,
            detail=detail,
        )

    @staticmethod
    def _http_error_details(
        error: Any,
    ) -> tuple[int | None, str | None, str]:
        """Extract status, provider code, and useful detail from an HTTP error."""
        response = getattr(error, "response", None)
        raw_status = getattr(response, "status_code", None)
        status_code = raw_status if isinstance(raw_status, int) else None

        payload: Mapping[str, Any] = {}
        if response is not None:
            try:
                candidate = response.json()
                if isinstance(candidate, Mapping):
                    payload = candidate
            except Exception:
                payload = {}

        error_data = payload.get("error")
        if isinstance(error_data, Mapping):
            raw_type = error_data.get("type") or error_data.get("code")
            raw_detail = (
                error_data.get("message")
                or error_data.get("detail")
                or error_data.get("error")
            )
        else:
            raw_type = payload.get("type") or payload.get("code")
            raw_detail = (
                payload.get("message")
                or payload.get("detail")
                or error_data
            )

        if not raw_detail and response is not None:
            raw_detail = getattr(response, "text", None)
        detail = str(raw_detail) if raw_detail else str(error)
        return (
            status_code,
            str(raw_type) if raw_type else None,
            detail,
        )

    @classmethod
    def _handle_stream_error(cls, event: SSEEvent) -> None:
        """Map a provider error event through Core's normalized hierarchy."""
        payload = event.data if isinstance(event.data, Mapping) else {}
        error_data = payload.get("error", payload)
        if not isinstance(error_data, Mapping):
            raise LLMAPIClientError(
                detail="DeepSeek Responses stream returned an invalid error event",
            )
        raw_type = error_data.get("type") or error_data.get("code")
        raw_detail = error_data.get("message") or error_data.get("detail")
        detail = (
            str(raw_detail)
            if raw_detail
            else "DeepSeek Responses stream failed"
        )
        cls._raise_mapped_error(
            status_code=None,
            error_type=str(raw_type) if raw_type else None,
            detail=detail,
        )

    @staticmethod
    def _raise_mapped_error(
        *,
        status_code: int | None,
        error_type: str | None,
        detail: str,
    ) -> None:
        """Map documented DeepSeek failures without introducing retries."""
        normalized_type = (error_type or "").lower()
        normalized_detail = detail.lower()

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
        } or "insufficient balance" in normalized_detail:
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

        raise LLMAPIClientError(detail=detail)


# Keep the short name available for package-local callers while the explicit
# Responses name documents the wire protocol at the public client boundary.
DeepSeekSyncClient = DeepSeekResponsesSyncClient


__all__ = [
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_RESPONSES_URL",
    "DeepSeekResponsesSyncClient",
    "DeepSeekSyncClient",
]
