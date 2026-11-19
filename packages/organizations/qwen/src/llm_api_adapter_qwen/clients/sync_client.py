"""Synchronous HTTP client for Model Studio's Messages-compatible endpoint."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Iterator

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
from llm_api_adapter.llms.anthropic_compatible.sync_client import (
    AnthropicCompatibleSyncClient,
)


FRANKFURT_MESSAGES_URL = (
    "https://{workspace_id}.eu-central-1.maas.aliyuncs.com/"
    "apps/anthropic/v1/messages"
)
_WORKSPACE_ID_PATTERN = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")


def validate_workspace_id(workspace_id: object) -> str:
    """Validate one Model Studio workspace identifier before building a URL."""
    if not isinstance(workspace_id, str) or not _WORKSPACE_ID_PATTERN.fullmatch(
        workspace_id
    ):
        raise ValueError(
            "workspace_id must consist of lowercase letters, digits, and "
            "hyphens only",
        )
    return workspace_id


def messages_url(workspace_id: object) -> str:
    """Return the Qwen 0.1.0 Frankfurt Messages endpoint for a workspace."""
    return FRANKFURT_MESSAGES_URL.format(
        workspace_id=validate_workspace_id(workspace_id),
    )


class QwenMessagesSyncClient(AnthropicCompatibleSyncClient):
    """Submit Qwen Messages requests through one Core synchronous transport."""

    def __init__(self, transport: SyncTransport) -> None:
        self.transport = transport

    def _get_sync_transport(self) -> SyncTransport:
        return self.transport

    def __repr__(self) -> str:
        return object.__repr__(self)

    def _headers(self, api_key: str) -> dict[str, str]:
        return {"x-api-key": api_key, "Content-Type": "application/json"}

    def _raise_execution_error(self, error: BaseException) -> None:
        raise error

    @staticmethod
    def _format_diagnostic(value: object) -> str:
        return str(value)

    def chat(
        self,
        *,
        api_key: str,
        workspace_id: object,
        payload: dict[str, Any],
        timeout_s: float | None,
    ) -> dict[str, Any]:
        response: JSONResponse = self._execute_prepared_request(
            TransportRequest(
                url=messages_url(workspace_id),
                headers=self._headers(api_key),
                payload=payload,
                timeout=timeout_s,
            )
        )
        response_data = response.json()
        if not isinstance(response_data, dict):
            raise LLMAPIClientError(
                detail="Qwen Messages returned a non-object response",
            )
        return response_data

    def stream(
        self,
        *,
        api_key: str,
        workspace_id: object,
        payload: dict[str, Any],
        timeout_s: float | None,
    ) -> Iterator[SSEEvent]:
        """Open one Qwen Messages SSE response through the Core transport."""
        stream_payload = dict(payload)
        stream_payload["stream"] = True
        request = TransportRequest(
            url=messages_url(workspace_id),
            headers=self._headers(api_key),
            payload=stream_payload,
            timeout=timeout_s,
        )
        return self._execute_prepared_stream(request)

    @classmethod
    def _handle_http_error(cls, http_error: Any) -> None:
        response = getattr(http_error, "response", None)
        status_code = getattr(response, "status_code", None)
        error_type: str | None = None
        detail = str(http_error)

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

    @classmethod
    def _handle_stream_error(cls, event: SSEEvent) -> None:
        """Map one Qwen Messages ``error`` SSE event to a public error."""
        payload = event.data if isinstance(event.data, Mapping) else {}
        error = payload.get("error", payload)
        if not isinstance(error, Mapping):
            raise LLMAPIClientError(
                detail="Qwen Messages returned an invalid SSE error event",
            )
        raw_type = error.get("type") or error.get("code")
        error_type = raw_type if isinstance(raw_type, str) else None
        raw_detail = error.get("message") or error.get("detail")
        detail = (
            raw_detail
            if isinstance(raw_detail, str) and raw_detail
            else "Qwen Messages returned an SSE error"
        )
        cls._raise_mapped_error(None, error_type, detail)

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


__all__ = [
    "FRANKFURT_MESSAGES_URL",
    "QwenMessagesSyncClient",
    "messages_url",
    "validate_workspace_id",
]
