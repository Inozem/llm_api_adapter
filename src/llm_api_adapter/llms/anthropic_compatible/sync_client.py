"""Synchronous execution for generic Anthropic Messages connections."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Mapping
from typing import Any

from ...adapters.compatible_connection import (
    _redact_compatible_diagnostic,
    _validate_api_root,
)
from ...errors.llm_api_error import (
    LLMAPIClientError,
    LLMAPIError,
    LLMAPIAuthorizationError,
    LLMAPIRateLimitError,
    LLMAPIServerError,
)
from ..transports import (
    SSEEvent,
    SyncTransport,
    TransportRequest,
    create_sync_transport,
    raise_default_http_error,
)

logger = logging.getLogger(__name__)


class _SyncMessagesExecutionMixin:
    """Execute already prepared Messages requests without instance routing."""

    def _create_sync_transport(self, transport: str) -> SyncTransport:
        return create_sync_transport(transport)

    def _execute_prepared_request(self, request: TransportRequest) -> Any:
        try:
            return self._get_sync_transport().post_json(
                request, http_error_handler=self._handle_http_error
            )
        except (LLMAPIError, json.JSONDecodeError) as exc:
            self._raise_execution_error(exc)

    def _execute_prepared_stream(
        self, request: TransportRequest
    ) -> Iterator[SSEEvent]:
        events = None
        primary_error: BaseException | None = None
        try:
            events = self._get_sync_transport().post_sse(
                request,
                http_error_handler=self._handle_http_error,
                stream_error_handler=self._handle_stream_error,
            )
            for event in events:
                yield event
        except BaseException as exc:
            primary_error = exc
            if isinstance(exc, LLMAPIError):
                self._raise_execution_error(exc)
            raise
        finally:
            if events is not None:
                try:
                    close = getattr(events, "close", None)
                    if close is not None:
                        close()
                except BaseException as close_error:
                    if primary_error is None:
                        if isinstance(close_error, LLMAPIError):
                            self._raise_execution_error(close_error)
                        raise

    def _get_sync_transport(self) -> SyncTransport:
        return self._sync_transport

    def _prepare_chat_payload_for_model(self, model: str, kwargs: dict) -> dict:
        if not isinstance(model, str) or not model:
            raise ValueError("model must be a nonempty string")
        payload = dict(kwargs)
        payload["model"] = model
        return payload

    def _handle_http_error(self, error: Any) -> None:
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
        error_type = error_message = None
        try:
            value = response.json().get("error", {})
            if isinstance(value, Mapping):
                error_type = value.get("type") or value.get("code")
                error_message = value.get("message")
        except Exception as exc:
            logger.warning("Could not read Messages error body: %s", self._format_diagnostic(type(exc).__name__))

        detail = self._safe_detail(error_type, error_message, error)
        if status is not None:
            raise_default_http_error(status_code=status, detail=detail)
        if error_type:
            normalized = str(error_type).lower()
            if normalized in {"authentication_error", "permission_error", "invalid_api_key"}:
                raise LLMAPIAuthorizationError(detail=detail) from None
            if normalized in {"rate_limit_error", "rate_limit_exceeded"}:
                raise LLMAPIRateLimitError(detail=detail) from None
            if normalized in {"api_error", "overloaded_error", "server_error", "internal_error"}:
                raise LLMAPIServerError(detail=detail) from None
        raise_default_http_error(status_code=status, detail=detail)

    def _handle_stream_error(self, event: SSEEvent) -> None:
        payload = event.data if isinstance(event.data, Mapping) else {}
        value = payload.get("error")
        value = value if isinstance(value, Mapping) else payload
        error_type = value.get("type") or value.get("code")
        message = value.get("message")
        detail = self._safe_detail(error_type, message)
        normalized = str(error_type or "").lower()
        if normalized in {"authentication_error", "permission_error", "invalid_api_key"}:
            raise LLMAPIAuthorizationError(detail=detail) from None
        if normalized in {"rate_limit_error", "rate_limit_exceeded"}:
            raise LLMAPIRateLimitError(detail=detail) from None
        if normalized in {"api_error", "overloaded_error", "server_error", "internal_error"}:
            raise LLMAPIServerError(detail=detail) from None
        raise LLMAPIClientError(detail=detail) from None

    def _safe_detail(self, error_type: Any, message: Any, fallback: Any = None) -> str:
        detail = message if isinstance(message, str) and message else None
        if detail is None and isinstance(error_type, str) and error_type:
            detail = error_type
        if detail is None and fallback is not None:
            # Exception text is useful for transport failures; response/event
            # mappings are deliberately never stringified here.
            detail = str(fallback)
        if detail is None:
            return "Messages request failed."
        return str(self._format_diagnostic(str(detail)))

    def _format_diagnostic(self, value: object) -> str:
        key = getattr(self, "api_key", "")
        if not isinstance(key, str):
            key = ""
        return _redact_compatible_diagnostic(value, key)

    def _raise_safe_execution_error(self, error: LLMAPIError) -> None:
        message = self._format_diagnostic(error.message)
        detail = (
            None
            if error.detail is None
            else self._format_diagnostic(error.detail)
        )
        raise type(error)(message=message, detail=detail) from None

    def _raise_execution_error(self, error: BaseException) -> None:
        """Provider hook; native bridges can preserve their own exception object."""
        raise error


class AnthropicCompatibleSyncClient(_SyncMessagesExecutionMixin):
    """Use a caller supplied Messages API root and exact model identifier."""

    def __init__(
        self, api_key: str, endpoint: str, transport: str = "requests"
    ) -> None:
        self._initialize_compatible(api_key, endpoint, transport)

    def _initialize_compatible(
        self,
        api_key: str,
        endpoint: str,
        transport: str,
        *,
        initialize_sync_transport: bool = True,
    ) -> None:
        if not isinstance(api_key, str) or not api_key:
            raise ValueError("api_key must be a nonempty string")
        self.endpoint = _validate_api_root(endpoint, resource_paths=("/messages",))
        self.api_key = api_key
        self.transport = transport
        if initialize_sync_transport:
            self._sync_transport = self._create_sync_transport(transport)

    def __repr__(self) -> str:
        return self._format_client_repr()

    def _format_client_repr(self) -> str:
        safe_endpoint = self._format_diagnostic(self.endpoint)
        return (
            f"{type(self).__name__}(api_key='***', "
            f"endpoint={safe_endpoint!r})"
        )

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

    def _raise_execution_error(self, error: BaseException) -> None:
        if isinstance(error, json.JSONDecodeError) or (
            isinstance(error, LLMAPIClientError)
            and isinstance(error.__cause__, json.JSONDecodeError)
        ):
            raise LLMAPIClientError(detail="Malformed JSON response.") from None
        assert isinstance(error, LLMAPIError)
        self._raise_safe_execution_error(error)

    def _request_data(self, model: str, kwargs: dict, *, stream: bool) -> tuple[str, dict]:
        """Bind the destination before invoking an overridable payload hook."""
        url = f"{self.endpoint}/messages"
        payload = self._prepare_chat_payload_for_model(model, kwargs)
        if stream:
            payload["stream"] = True
        return url, payload

    def _send_request(self, url: str, payload: dict, timeout_s: float | None = None):
        return self._execute_prepared_request(
            TransportRequest(url=url, headers=self._headers(), payload=dict(payload), timeout=timeout_s)
        )

    def _stream_request(self, url: str, payload: dict, timeout_s: float | None = None):
        return self._execute_prepared_stream(
            TransportRequest(url=url, headers=self._headers(), payload=dict(payload), timeout=timeout_s)
        )

    def chat_completion(self, model: str, timeout_s: float | None = None, **wire_payload: Any):
        url, payload = self._request_data(model, wire_payload, stream=False)
        return self._send_request(url, payload, timeout_s).json()

    def stream(self, model: str, timeout_s: float | None = None, **wire_payload: Any):
        url, payload = self._request_data(model, wire_payload, stream=True)
        return self._stream_request(url, payload, timeout_s)


__all__ = ["AnthropicCompatibleSyncClient"]
