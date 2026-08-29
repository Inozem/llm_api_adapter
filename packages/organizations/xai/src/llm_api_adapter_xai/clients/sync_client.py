"""Synchronous client for xAI's official Responses API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping

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
    MultipartFile,
    MultipartForm,
    SSEEvent,
    SyncTransport,
    TransportRequest,
    create_sync_transport,
)


_XAI_RESPONSES_URL = "https://api.x.ai/v1/responses"
_XAI_FILES_URL = "https://api.x.ai/v1/files"


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

    def stream(
        self,
        *,
        model: str,
        timeout: float | None = None,
        **parameters: Any,
    ) -> Iterator[SSEEvent]:
        """Stream one Responses API response as provider-neutral SSE events."""
        return self._sync_transport.post_sse(
            TransportRequest(
                url=self.endpoint,
                headers=self._headers(),
                payload={
                    "model": model,
                    **parameters,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
                timeout=timeout,
            ),
            http_error_handler=self._handle_http_error,
            stream_error_handler=self._handle_stream_error,
        )

    def upload_file(
        self,
        *,
        content: bytes,
        filename: str,
        content_type: str,
        expires_after: int,
        timeout: float | None = None,
    ) -> str:
        """Upload one adapter-owned attachment and return its xAI file ID."""
        response: JSONResponse = self._sync_transport.post_multipart(
            TransportRequest(
                url=_XAI_FILES_URL,
                headers=self._headers(),
                timeout=timeout,
            ),
            MultipartForm(
                # xAI requires ``expires_after`` to precede the ``file`` part.
                fields=(("expires_after", str(expires_after)),),
                files=(
                    MultipartFile(
                        field_name="file",
                        filename=filename,
                        content=content,
                        content_type=content_type,
                    ),
                ),
            ),
            http_error_handler=self._handle_http_error,
        )
        payload = response.json()
        file_id = payload.get("id") if isinstance(payload, dict) else None
        if not isinstance(file_id, str) or not file_id:
            raise LLMAPIClientError(
                detail="xAI Files API returned a response without a file id",
            )
        return file_id

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _handle_http_error(self, error: Any) -> None:
        status_code, error_type, detail = self._http_error_details(error)
        self._raise_mapped_error(
            status_code=status_code,
            error_type=error_type,
            detail=detail,
        )

    @staticmethod
    def _http_error_details(
        error: Any,
    ) -> tuple[int | None, str | None, str]:
        """Read both documented xAI error envelope variants.

        Responses API errors may contain a nested ``error`` object, or the
        flat ``{"code": ..., "error": "..."}`` shape returned for an invalid
        API key. Preserve the latter message so it can be classified and shown
        to callers.
        """
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

        error_data = payload.get("error")
        if isinstance(error_data, Mapping):
            error_type = error_data.get("type") or error_data.get("code")
            detail = error_data.get("message") or error_data.get("detail")
        else:
            error_type = payload.get("type") or payload.get("code")
            detail = payload.get("message") or error_data

        return (
            status_code if isinstance(status_code, int) else None,
            str(error_type) if error_type else None,
            str(detail) if detail else str(error),
        )

    def _handle_stream_error(self, event: SSEEvent) -> None:
        """Map an xAI error event through the same public error hierarchy."""
        payload = event.data if isinstance(event.data, Mapping) else {}
        error_data = payload.get("error", payload)
        if not isinstance(error_data, Mapping):
            error_data = {}
        error_type = error_data.get("type") or error_data.get("code")
        detail = error_data.get("message") or "xAI Responses stream failed"
        self._raise_mapped_error(
            status_code=None,
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
        normalized_detail = detail.lower()
        if status_code in {401, 403} or normalized_type in {
            "authentication_error",
            "authorization_error",
            "invalid_api_key",
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
