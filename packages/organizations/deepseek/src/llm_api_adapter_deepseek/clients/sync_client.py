"""Synchronous DeepSeek Responses client. The ``DeepSeekSyncClient`` alias remains available."""

from __future__ import annotations

from dataclasses import dataclass, field
from json import JSONDecodeError
from typing import Any, Iterator, Mapping

from llm_api_adapter.errors.llm_api_error import (
    LLMAPIError,
    LLMAPIAuthorizationError,
    LLMAPIClientError,
    LLMAPIRateLimitError,
    LLMAPIServerError,
    LLMAPITimeoutError,
    LLMAPITokenLimitError,
    LLMAPIUsageLimitError,
)
from llm_api_adapter.llms.httpx_transport import HttpxSyncTransport
from llm_api_adapter.llms.openai_compatible.sync_client import (
    OpenAICompatibleSyncClient,
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


DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_RESPONSES_URL = f"{DEEPSEEK_BASE_URL}/responses"
DEEPSEEK_FILES_URL = f"{DEEPSEEK_BASE_URL}/files"
_FILE_UPLOAD_TIMEOUT = 300.0
_FILE_UPLOAD_FAILURE = "DeepSeek file upload failed"


@dataclass(repr=False, init=False)
class DeepSeekResponsesSyncClient(OpenAICompatibleSyncClient):
    """Use Core's compatible JSON/SSE engines with native DeepSeek policy.

    Retain native dataclass order with ``transport`` before ``endpoint`` and
    exclude Core-only fields from native dataclass introspection.
    """

    api_key: str
    transport: str = "requests"
    endpoint: str = DEEPSEEK_RESPONSES_URL
    _sync_transport: SyncTransport = field(init=False, repr=False, compare=False)
    __match_args__ = ("api_key", "transport", "endpoint")

    def __init__(
        self,
        api_key: str,
        transport: str = "requests",
        endpoint: str = DEEPSEEK_RESPONSES_URL,
    ) -> None:
        self.api_key = api_key
        self.transport = transport
        self.endpoint = endpoint
        self._api_variant_override = "responses"
        self._diagnostic_formatter = None
        self.__post_init__()

    def __repr__(self) -> str:
        """Retain the old opaque object representation without credential data."""
        return object.__repr__(self)

    @staticmethod
    def _format_diagnostic(value: object) -> object:
        """Provide Core transports a class-callable native passthrough."""
        return value

    def _create_sync_transport(self, transport: str) -> SyncTransport:
        """Create the transport through the native module patch point."""
        return create_sync_transport(transport)

    def create(
        self,
        *,
        model: str,
        timeout: float | None = None,
        **parameters: Any,
    ) -> dict[str, Any]:
        """Create a Responses result through the inherited JSON engine."""
        payload = {"model": model, **parameters}
        response: JSONResponse = self._send_request(self.endpoint, payload, timeout)
        result = response.json()
        if not isinstance(result, dict):
            raise LLMAPIClientError(
                detail="DeepSeek Responses API returned a non-object response",
            )
        return result

    def _upload_pdf(self, content: bytes, timeout: float | None = None) -> str:
        """Upload one PDF and return its private provider receipt."""
        request = TransportRequest(
            url=DEEPSEEK_FILES_URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=_FILE_UPLOAD_TIMEOUT if timeout is None else timeout,
        )
        form = _pdf_upload_form(content)
        if isinstance(self._sync_transport, HttpxSyncTransport):
            form = _HttpxOrderedMultipartForm(
                fields=form.fields,
                files=form.files,
            )
        try:
            response = self._sync_transport.post_multipart(
                request,
                form,
                http_error_handler=self._handle_file_upload_http_error,
            )
            payload = response.json()
        except LLMAPIError as error:
            raise type(error)(detail=_FILE_UPLOAD_FAILURE) from None
        except (JSONDecodeError, UnicodeDecodeError):
            raise LLMAPIClientError(detail=_FILE_UPLOAD_FAILURE) from None
        return _file_id_from_upload(payload)

    @staticmethod
    def _handle_file_upload_http_error(error: Any) -> None:
        status_code, error_type, detail = (
            DeepSeekResponsesSyncClient._http_error_details(error)
        )
        DeepSeekResponsesSyncClient._raise_mapped_error(
            status_code=status_code,
            error_type=error_type,
            detail=detail,
            public_detail=_FILE_UPLOAD_FAILURE,
        )

    def stream(
        self,
        *,
        model: str,
        timeout: float | None = None,
        **parameters: Any,
    ) -> Iterator[SSEEvent]:
        """Call post_sse immediately and defer HTTP opening until iteration."""
        payload = {"model": model, **parameters, "stream": True}
        request = TransportRequest(
            url=self.endpoint,
            headers=self._headers(),
            payload=payload,
            timeout=timeout,
        )
        events = _BoundEventSource(
            self._sync_transport.post_sse(
                request,
                http_error_handler=self._handle_http_error,
                stream_error_handler=self._handle_stream_error,
            ),
        )
        view = self._request_view(request, events)
        return _CoreStreamIterator(
            view._stream_request(request.url, request.payload, request.timeout),
            events,
        )

    def _request_view(
        self,
        request: TransportRequest,
        events: _BoundEventSource,
    ) -> "_DeepSeekRequestView":
        return _DeepSeekRequestView(self, request, events)

    @staticmethod
    def _is_failed_stream_event(event: SSEEvent) -> bool:
        """Leave DeepSeek's semantic response.failed event to native parsing."""
        del event
        return False

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
        status_code = (
            raw_status
            if isinstance(raw_status, int) and not isinstance(raw_status, bool)
            else None
        )

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
        response_data = payload.get("response")
        response_error = (
            response_data.get("error")
            if isinstance(response_data, Mapping)
            else None
        )
        error_data = payload.get("error")
        if not isinstance(error_data, Mapping):
            error_data = response_error
        if not isinstance(error_data, Mapping):
            error_data = payload
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
        public_detail: str | None = None,
    ) -> None:
        """Map documented DeepSeek failures without introducing retries."""
        normalized_type = (error_type or "").strip().lower()
        normalized_detail = detail.strip().lower()
        safe_detail = public_detail if public_detail is not None else detail

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
            raise LLMAPIAuthorizationError(detail=safe_detail)

        if status_code == 429 or normalized_type in {
            "rate_limit_error",
            "rate_limit_exceeded",
            "rate_limit_reached",
        }:
            raise LLMAPIRateLimitError(detail=safe_detail)

        if status_code == 402 or normalized_type in {
            "insufficient_balance",
            "insufficient_quota",
            "quota_exceeded",
            "usage_limit_exceeded",
        } or "insufficient balance" in normalized_detail:
            raise LLMAPIUsageLimitError(detail=safe_detail)

        if normalized_type in {
            "context_length_exceeded",
            "input_too_long",
            "max_tokens_exceeded",
            "max_output_tokens_exceeded",
        }:
            raise LLMAPITokenLimitError(detail=safe_detail)

        if status_code in {408, 504} or normalized_type in {
            "timeout",
            "timeout_error",
        }:
            raise LLMAPITimeoutError(detail=safe_detail)

        if status_code is not None and 500 <= status_code < 600:
            raise LLMAPIServerError(detail=safe_detail)

        raise LLMAPIClientError(detail=safe_detail)


class _BoundEventSource:
    """Bind one native SSE source to Core and own its idempotent cleanup."""

    def __init__(self, events: Iterator[SSEEvent]) -> None:
        self._events = events
        self._closed = False
        self._iteration_failed = False

    def __iter__(self) -> "_BoundEventSource":
        return self

    def __next__(self) -> SSEEvent:
        try:
            return next(self._events)
        except StopIteration:
            raise
        except BaseException:
            self._iteration_failed = True
            raise

    def post_sse(
        self,
        request: TransportRequest,
        *,
        http_error_handler: Any = None,
        stream_error_handler: Any = None,
    ) -> Iterator[SSEEvent]:
        del request, http_error_handler, stream_error_handler
        return self

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self._events, "close", None)
        if not callable(close):
            return
        try:
            close()
        except BaseException:
            if not self._iteration_failed:
                raise


class _DeepSeekRequestView(DeepSeekResponsesSyncClient):
    """Per-request native view that supplies a captured request to Core."""

    def __init__(
        self,
        client: DeepSeekResponsesSyncClient,
        request: TransportRequest,
        events: _BoundEventSource,
    ) -> None:
        self.api_key = client.api_key
        self.transport = client.transport
        self.endpoint = request.url
        self._api_variant_override = "responses"
        self._diagnostic_formatter = client._diagnostic_formatter
        self._sync_transport = events
        self._request = request

    def _headers(self) -> dict[str, str]:
        return self._request.headers_dict()


class _CoreStreamIterator:
    """Expose Core's iterator while owning cleanup from the instant of return."""

    def __init__(
        self,
        events: Iterator[SSEEvent],
        source: _BoundEventSource,
    ) -> None:
        self._events = events
        self._source = source

    def __iter__(self) -> "_CoreStreamIterator":
        return self

    def __next__(self) -> SSEEvent:
        return next(self._events)

    def close(self) -> None:
        close = getattr(self._events, "close", None)
        try:
            if callable(close):
                close()
        finally:
            self._source.close()


DeepSeekResponsesSyncClient.__dataclass_fields__ = {
    name: DeepSeekResponsesSyncClient.__dataclass_fields__[name]
    for name in ("api_key", "transport", "endpoint", "_sync_transport")
}


DeepSeekSyncClient = DeepSeekResponsesSyncClient


def _pdf_upload_form(content: bytes) -> MultipartForm:
    return MultipartForm(
        fields=(
            ("purpose", "rich_layout_recognition"),
            ("expires_after[anchor]", "created_at"),
            ("expires_after[seconds]", "3600"),
        ),
        files=(
            MultipartFile(
                field_name="file",
                filename="document.pdf",
                content=content,
                content_type="application/pdf",
            ),
        ),
    )


class _HttpxOrderedMultipartForm(MultipartForm):
    def fields_list(self) -> dict[str, str]:  # type: ignore[override]
        return dict(self.fields)


def _file_id_from_upload(payload: Any) -> str:
    file_id = payload.get("id") if isinstance(payload, Mapping) else None
    if not isinstance(file_id, str) or not file_id.strip():
        raise LLMAPIClientError(detail=_FILE_UPLOAD_FAILURE)
    return file_id


__all__ = [
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_RESPONSES_URL",
    "DeepSeekResponsesSyncClient",
    "DeepSeekSyncClient",
]
