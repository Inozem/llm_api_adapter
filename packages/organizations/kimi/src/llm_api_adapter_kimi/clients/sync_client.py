"""Synchronous client for Kimi's official Chat Completions endpoint."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from llm_api_adapter.errors.llm_api_error import (
    LLMAPIClientError,
    LLMAPIAuthorizationError,
    LLMAPIRateLimitError,
    LLMAPIServerError,
    LLMAPITimeoutError,
    LLMAPITokenLimitError,
    LLMAPIUsageLimitError,
)
from llm_api_adapter.llms.openai_compatible.sync_client import (
    OpenAICompatibleSyncClient,
)
from llm_api_adapter.llms.transports import (
    HTTPErrorHandler,
    JSONResponse,
    MultipartForm,
    SSEEvent,
    StreamErrorHandler,
    SyncTransport,
    TransportRequest,
)

KIMI_CHAT_COMPLETIONS_URL = "https://api.moonshot.ai/v1/chat/completions"


class KimiSyncClient(OpenAICompatibleSyncClient):
    """Use Core's JSON/SSE runners while preserving Kimi's raw client API."""

    def __init__(self, transport: SyncTransport) -> None:
        # Preserve the transport-object constructor without constructing another
        # transport via the compatible dataclass initializer.
        self.api_key = ""
        self.endpoint = KIMI_CHAT_COMPLETIONS_URL
        self.transport = transport
        self._api_variant_override = None
        self._diagnostic_formatter = None

    @property
    def transport(self) -> SyncTransport:
        """Expose the historical transport injection seam."""
        return self._sync_transport

    @transport.setter
    def transport(self, transport: SyncTransport) -> None:
        self._sync_transport = transport

    def chat(
        self,
        *,
        api_key: str,
        payload: dict[str, Any],
        timeout_s: float | None,
    ) -> dict[str, Any]:
        """Post one unchanged raw payload and validate its response envelope."""
        view = self._request_view(api_key)
        response: JSONResponse = view._send_request(
            KIMI_CHAT_COMPLETIONS_URL,
            payload,
            timeout_s,
        )
        response_data = response.json()
        if not isinstance(response_data, dict):
            raise LLMAPIClientError(
                detail="Kimi Chat Completions returned a non-object response",
            )
        return response_data

    def stream(
        self,
        *,
        api_key: str,
        payload: dict[str, Any],
        timeout_s: float | None,
    ) -> Iterator[SSEEvent]:
        """Return Core's lazy SSE runner with Kimi's terminal-usage options."""

        def run() -> Iterator[SSEEvent]:
            stream_payload = dict(payload)
            stream_payload["stream"] = True
            stream_payload["stream_options"] = {"include_usage": True}
            view = self._request_view(api_key)
            yield from view._stream_request(
                KIMI_CHAT_COMPLETIONS_URL,
                stream_payload,
                timeout_s,
            )

        return run()

    def _request_view(self, api_key: str) -> _KimiRequestView:
        """Snapshot credentials and transport for one raw invocation."""
        return _KimiRequestView(
            transport=self._sync_transport,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            http_error_handler=self._handle_http_error,
            stream_error_handler=self._handle_stream_error,
        )

    def _adapter_binding(self, api_key: str) -> _KimiAdapterClientBinding:
        """Expose native raw calls through the adapter's inherited dispatch."""
        return _KimiAdapterClientBinding(self, api_key)

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

    @classmethod
    def _handle_stream_error(cls, event: SSEEvent) -> None:
        """Map a documented Kimi SSE error frame through the public hierarchy."""
        payload = event.data if isinstance(event.data, Mapping) else {}
        error = payload.get("error", payload)
        if not isinstance(error, Mapping):
            raise LLMAPIClientError(
                detail="Kimi Chat Completions returned an invalid SSE error event",
            )
        raw_type = error.get("type") or error.get("code")
        error_type = raw_type if isinstance(raw_type, str) else None
        raw_detail = error.get("message") or error.get("detail")
        detail = (
            raw_detail
            if isinstance(raw_detail, str) and raw_detail
            else "Kimi Chat Completions returned an SSE error"
        )
        cls._raise_mapped_error(None, error_type, detail)

    @staticmethod
    def _format_diagnostic(value: object) -> str:
        """Keep transport logs free of raw provider errors and request content."""
        return f"Kimi transport diagnostic ({type(value).__name__})"

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


class _KimiRequestView(KimiSyncClient):
    """Compatible request view with invocation-local credentials and handlers."""

    _request_headers: dict[str, str]
    _request_http_error_handler: HTTPErrorHandler
    _request_stream_error_handler: StreamErrorHandler

    def __init__(
        self,
        *,
        transport: SyncTransport,
        headers: Mapping[str, str],
        http_error_handler: HTTPErrorHandler,
        stream_error_handler: StreamErrorHandler,
    ) -> None:
        super().__init__(_CloseableSyncTransport(transport))
        self._request_headers = dict(headers)
        self._request_http_error_handler = http_error_handler
        self._request_stream_error_handler = stream_error_handler

    def _headers(self) -> dict[str, str]:
        return dict(self._request_headers)

    def _handle_http_error(self, http_error: Any) -> None:
        self._request_http_error_handler(http_error)

    def _handle_stream_error(self, event: SSEEvent) -> None:
        self._request_stream_error_handler(event)

    @staticmethod
    def _is_failed_stream_event(event: SSEEvent) -> bool:
        """Leave Kimi event classification to its native transport callback."""
        return False


class _KimiAdapterClientBinding:
    """Thin adapter-facing dispatch that preserves native payload semantics."""

    _native: KimiSyncClient
    _api_key: str

    def __init__(self, native: KimiSyncClient, api_key: str) -> None:
        self._native = native
        self._api_key = api_key

    @staticmethod
    def _should_use_responses_api(model: str) -> bool:
        _ = model
        return False

    def complete(self, model: str, timeout: float | None = None, **payload: Any) -> dict[str, Any]:
        payload.setdefault("model", model)
        return self._native.chat(
            api_key=self._api_key,
            payload=payload,
            timeout_s=timeout,
        )

    def stream(
        self,
        model: str,
        timeout: float | None = None,
        **payload: Any,
    ) -> Iterator[SSEEvent]:
        payload.setdefault("model", model)
        return self._native.stream(
            api_key=self._api_key,
            payload=payload,
            timeout_s=timeout,
        )


class _CloseableSSEIterator:
    """Supply the lifecycle hook Core expects while forwarding real closure."""

    _events: Iterator[SSEEvent]

    def __init__(self, events: Iterator[SSEEvent]) -> None:
        self._events = events

    def __iter__(self) -> _CloseableSSEIterator:
        return self

    def __next__(self) -> SSEEvent:
        return next(self._events)

    def close(self) -> None:
        close = getattr(self._events, "close", None)
        if callable(close):
            close()


class _CloseableSyncTransport(SyncTransport):
    """Snapshot a transport and adapt accepted SSE sources missing close()."""

    _transport: SyncTransport

    def __init__(self, transport: SyncTransport) -> None:
        self._transport = transport

    def post_json(
        self,
        request: TransportRequest,
        *,
        http_error_handler: HTTPErrorHandler | None = None,
    ) -> JSONResponse:
        return self._transport.post_json(
            request,
            http_error_handler=http_error_handler,
        )

    def post_multipart(
        self,
        request: TransportRequest,
        form: MultipartForm,
        *,
        http_error_handler: HTTPErrorHandler | None = None,
    ) -> JSONResponse:
        return self._transport.post_multipart(
            request,
            form,
            http_error_handler=http_error_handler,
        )

    def post_sse(
        self,
        request: TransportRequest,
        *,
        http_error_handler: HTTPErrorHandler | None = None,
        stream_error_handler: StreamErrorHandler | None = None,
    ) -> Iterator[SSEEvent]:
        events = self._transport.post_sse(
            request,
            http_error_handler=http_error_handler,
            stream_error_handler=stream_error_handler,
        )
        if callable(getattr(events, "close", None)):
            return events
        return _CloseableSSEIterator(events)


__all__ = ["KIMI_CHAT_COMPLETIONS_URL", "KimiSyncClient"]
