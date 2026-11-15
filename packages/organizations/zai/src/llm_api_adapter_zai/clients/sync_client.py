"""Synchronous compatible client with Z.ai's native request policy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
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
from llm_api_adapter.llms.openai_compatible.sync_client import OpenAICompatibleSyncClient
from llm_api_adapter.llms.transports import SSEEvent, SyncTransport

ZAI_BASE_URL = "https://api.z.ai/api/paas/v4"
ZAI_CHAT_COMPLETIONS_URL = f"{ZAI_BASE_URL}/chat/completions"


def _format_zai_diagnostic(_value: object) -> str:
    """Return a fixed private diagnostic without echoing untrusted content."""
    return "Z.ai transport failure (details omitted)"


@dataclass(repr=False)
class ZaiSyncClient(OpenAICompatibleSyncClient):
    """Use Core's sync HTTP/SSE engine with Z.ai's native wire behavior."""

    endpoint: str = field(default=ZAI_BASE_URL, init=False)
    sync_transport: InitVar[SyncTransport | None] = field(default=None, kw_only=True)

    def __post_init__(self, sync_transport: SyncTransport | None) -> None:
        """Reuse an injected transport or initialize Core's selected transport."""
        if sync_transport is None:
            super().__post_init__()
        else:
            self._sync_transport = sync_transport
        self._diagnostic_formatter = _format_zai_diagnostic

    def _prepare_chat_payload_for_model(
        self, model: str, kwargs: dict[str, Any], *, streaming: bool = False
    ) -> dict[str, Any]:
        """Keep Z.ai's max_tokens, sampling fields and stream options unchanged."""
        payload = {"model": model, **dict(kwargs)}
        if streaming:
            payload.setdefault("stream_options", {"include_usage": True})
        return payload

    def chat_completion(
        self, model: str, timeout: float | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Validate the completed response envelope after Core sends the request."""
        return _validated_response(super().chat_completion(model, timeout, **kwargs))

    chat = OpenAICompatibleSyncClient.complete

    def _handle_http_error(self, http_error: Any) -> None:
        """Map Z.ai HTTP failures into the existing public error hierarchy."""
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

        self._raise_mapped_error(status_code, error_type, detail)

    def _handle_stream_error(self, event: SSEEvent) -> None:
        """Map a native OpenAI-compatible Z.ai SSE error frame."""
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
        self._raise_mapped_error(None, error_type, detail)

    @staticmethod
    def _raise_mapped_error(
        status_code: int | None,
        error_type: str | None,
        detail: str,
    ) -> None:
        normalized_type = error_type.strip().lower() if error_type else ""
        normalized_detail = detail.strip().lower()

        if (
            status_code in {401, 403}
            or normalized_type
            in {
                "authentication_error",
                "authorization_error",
                "invalid_api_key",
                "invalid_authentication_error",
                "permission_denied",
            }
            or any(
                marker in normalized_detail
                for marker in (
                    "api key",
                    "api_key",
                    "authentication",
                    "authorization",
                    "unauthorized",
                    "invalid credentials",
                )
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


ZaiChatCompletionsSyncClient = ZaiSyncClient


def _validated_response(response_data: Any) -> dict[str, Any]:
    if not isinstance(response_data, dict):
        raise LLMAPIClientError(detail="Z.ai Chat Completions returned a non-object response")
    return response_data


__all__ = ["ZAI_BASE_URL", "ZAI_CHAT_COMPLETIONS_URL", "ZaiChatCompletionsSyncClient", "ZaiSyncClient"]
