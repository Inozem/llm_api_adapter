"""Synchronous transport calls for Mistral's official API."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import Any, Optional

from llm_api_adapter.errors.llm_api_error import (
    LLMAPIClientError,
    LLMAPIAuthorizationError,
    LLMAPIRateLimitError,
    LLMAPIServerError,
    LLMAPITimeoutError,
    LLMAPITokenLimitError,
    LLMAPIUsageLimitError,
)
from llm_api_adapter.llm_registry.llm_registry import (
    CategoricalReasoningCapability,
    ModelSpec,
)
from llm_api_adapter.llm_registry.reasoning import resolve_reasoning_level
from llm_api_adapter.llms.openai_compatible.sync_client import (
    OpenAICompatibleSyncClient,
    _prepare_compatible_responses_payload,
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

from ..registry import MODEL_METADATA

MISTRAL_CHAT_COMPLETIONS_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_RESPONSES_URL = "https://api.mistral.ai/v1/responses"
MISTRAL_OCR_URL = "https://api.mistral.ai/v1/ocr"


class MistralSyncClient(OpenAICompatibleSyncClient):
    """Use inherited compatible execution with Mistral-owned request policy."""

    def __init__(self, transport: SyncTransport) -> None:
        # Preserve the historical transport-object constructor without creating
        # a second default transport through the parent initializer.
        self.api_key = ""
        self.endpoint = MISTRAL_RESPONSES_URL
        self.transport = transport
        self._api_variant_override = "responses"
        self._diagnostic_formatter = None

    @property
    def transport(self) -> SyncTransport:
        """Expose the original transport-object compatibility attribute."""
        return self._sync_transport

    @transport.setter
    def transport(self, transport: SyncTransport) -> None:
        self._sync_transport = transport

    def chat(
        self,
        *,
        endpoint: str,
        headers: Mapping[str, str],
        payload: dict[str, Any],
        timeout_s: Optional[float],
        http_error_handler: Callable[[Any], None],
    ) -> dict[str, Any]:
        """Submit the supplied full URL and raw payload through Core JSON I/O."""
        return self._post_json(
            endpoint,
            headers,
            payload,
            timeout_s,
            http_error_handler,
            "Mistral returned a non-object response",
        )

    def ocr(
        self,
        *,
        headers: Mapping[str, str],
        payload: dict[str, Any],
        timeout_s: Optional[float],
        http_error_handler: Callable[[Any], None],
    ) -> dict[str, Any]:
        """Submit raw OCR payloads to Mistral's unchanged OCR resource."""
        return self._post_json(
            MISTRAL_OCR_URL,
            headers,
            payload,
            timeout_s,
            http_error_handler,
            "Mistral OCR returned a non-object response",
        )

    def stream_chat(
        self,
        *,
        endpoint: str,
        headers: Mapping[str, str],
        payload: dict[str, Any],
        timeout_s: Optional[float],
        http_error_handler: Callable[[Any], None],
        stream_error_handler: Callable[[SSEEvent], None],
    ) -> Iterator[SSEEvent]:
        """Return a lazy stream driven by Core's inherited SSE runner."""
        view = self._request_view(headers, http_error_handler, stream_error_handler)
        return view._stream_request(endpoint, payload, timeout_s)

    def _should_use_responses_api(self, model: str) -> bool:
        """Select Responses for native Mistral generation."""
        return True

    def responses(
        self, model: str, timeout: float | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Submit native Responses payloads to a complete resource URL."""
        endpoint = kwargs.pop("endpoint", self.endpoint)
        headers = kwargs.pop("headers", self._headers())
        http_error_handler = kwargs.pop(
            "http_error_handler", self._handle_http_error
        )
        payload = self._prepare_responses_payload_for_model(model, kwargs)
        return self._post_json(
            endpoint,
            headers,
            payload,
            timeout,
            http_error_handler,
            "Mistral returned a non-object response",
        )

    def stream_responses(
        self, model: str, timeout: float | None = None, **kwargs: Any
    ) -> Iterator[SSEEvent]:
        """Stream native Responses events from the supplied full resource URL."""
        endpoint = kwargs.pop("endpoint", self.endpoint)
        headers = kwargs.pop("headers", self._headers())
        http_error_handler = kwargs.pop(
            "http_error_handler", self._handle_http_error
        )
        stream_error_handler = kwargs.pop(
            "stream_error_handler", self._handle_stream_error
        )
        payload = self._prepare_responses_payload_for_model(model, kwargs)
        payload["stream"] = True
        return self.stream_chat(
            endpoint=endpoint,
            headers=headers,
            payload=payload,
            timeout_s=timeout,
            http_error_handler=http_error_handler,
            stream_error_handler=stream_error_handler,
        )

    def _prepare_responses_payload_for_model(
        self, model: str, kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """Reuse shared stateless conversion, restoring native wire differences."""
        model_spec = _model_spec(model)
        native_kwargs = dict(kwargs)
        for key in ("previous_response", "json_schema", "response_model"):
            native_kwargs.pop(key, None)
        payload = _prepare_compatible_responses_payload(model, native_kwargs)
        provider_model = model_spec.provider_model if model_spec else None
        payload["model"] = provider_model or model

        # Native Mistral transmits its numeric sampling defaults explicitly.
        for key in ("temperature", "top_p"):
            if key in kwargs:
                payload[key] = kwargs[key]

        effort = _native_reasoning_effort(model_spec, kwargs)
        if effort is not None:
            payload["reasoning"] = {"effort": effort}
        return payload

    def _post_json(
        self,
        url: str,
        headers: Mapping[str, str],
        payload: dict[str, Any],
        timeout_s: Optional[float],
        http_error_handler: HTTPErrorHandler,
        malformed_detail: str,
    ) -> dict[str, Any]:
        """Use the inherited JSON runner and validate the returned envelope."""
        view = self._request_view(headers, http_error_handler)
        response: JSONResponse = view._send_request(url, payload, timeout_s)
        response_data = response.json()
        if not isinstance(response_data, dict):
            raise LLMAPIClientError(detail=malformed_detail)
        return response_data

    def _request_view(
        self,
        headers: Mapping[str, str],
        http_error_handler: Optional[HTTPErrorHandler] = None,
        stream_error_handler: Optional[StreamErrorHandler] = None,
    ) -> _NativeRequestView:
        """Snapshot request state into a per-call native subclass view."""
        transport = _CloseableSyncTransport(self._sync_transport)
        return _NativeRequestView(
            transport=transport,
            headers=dict(headers),
            http_error_handler=http_error_handler,
            stream_error_handler=stream_error_handler,
        )

    def _handle_http_error(self, http_error: Any) -> None:
        """Map default Mistral HTTP errors through the native error hierarchy."""
        response = getattr(http_error, "response", None)
        raw_status = getattr(response, "status_code", None)
        status_code = raw_status if isinstance(raw_status, int) else None
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
        raw_type = error_data.get("type") or error_data.get("code")
        raw_detail = error_data.get("message") or str(http_error)
        self._raise_mapped_error(
            status_code=status_code,
            error_type=raw_type if isinstance(raw_type, str) else None,
            detail=str(raw_detail),
        )

    def _handle_stream_error(self, event: SSEEvent) -> None:
        """Map default Mistral Responses error events."""
        payload = event.data if isinstance(event.data, Mapping) else {}
        response = payload.get("response")
        response_error = (
            response.get("error") if isinstance(response, Mapping) else None
        )
        error = response_error if isinstance(response_error, Mapping) else payload.get(
            "error", payload
        )
        if not isinstance(error, Mapping):
            error = payload
        raw_type = error.get("type") or error.get("code")
        raw_detail = error.get("message") or str(error)
        self._raise_mapped_error(
            status_code=None,
            error_type=raw_type if isinstance(raw_type, str) else None,
            detail=str(raw_detail),
        )

    @staticmethod
    def _raise_mapped_error(
        status_code: Optional[int], error_type: Optional[str], detail: str
    ) -> None:
        normalized_type = (error_type or "").lower()
        if status_code in {401, 403} or normalized_type in {
            "authentication_error", "authorization_error", "invalid_api_key"
        }:
            raise LLMAPIAuthorizationError(detail=detail)
        if status_code == 429 or normalized_type in {
            "rate_limit_error", "rate_limit_exceeded"
        }:
            raise LLMAPIRateLimitError(detail=detail)
        if normalized_type in {
            "context_length_exceeded", "input_too_long", "max_tokens_exceeded"
        }:
            raise LLMAPITokenLimitError(detail=detail)
        if normalized_type in {"insufficient_quota", "usage_limit_exceeded"}:
            raise LLMAPIUsageLimitError(detail=detail)
        if status_code in {408, 504} or normalized_type in {"timeout", "timeout_error"}:
            raise LLMAPITimeoutError(detail=detail)
        if status_code is not None and 500 <= status_code < 600:
            raise LLMAPIServerError(detail=detail)
        raise LLMAPIClientError(detail=detail)


class _NativeRequestView(MistralSyncClient):
    """Isolated Mistral client view used for one inherited request operation."""

    _request_headers: dict[str, str]
    _request_http_error_handler: Optional[HTTPErrorHandler]
    _request_stream_error_handler: Optional[StreamErrorHandler]

    def __init__(
        self,
        *,
        transport: SyncTransport,
        headers: Mapping[str, str],
        http_error_handler: Optional[HTTPErrorHandler],
        stream_error_handler: Optional[StreamErrorHandler] = None,
    ) -> None:
        super().__init__(transport)
        self._request_headers = dict(headers)
        self._request_http_error_handler = http_error_handler
        self._request_stream_error_handler = stream_error_handler

    def _headers(self) -> dict[str, str]:
        return dict(self._request_headers)

    def _handle_http_error(self, http_error: Any) -> None:
        handler = self._request_http_error_handler
        if handler is not None:
            handler(http_error)
            return
        super()._handle_http_error(http_error)

    def _handle_stream_error(self, event: SSEEvent) -> None:
        handler = self._request_stream_error_handler
        if handler is not None:
            handler(event)
            return
        super()._handle_stream_error(event)


class _CloseableSSEIterator:
    """Add the close hook expected by Core without changing event iteration."""

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
    """Snapshot a transport and adapt only non-closeable SSE iterators."""

    _transport: SyncTransport

    def __init__(self, transport: SyncTransport) -> None:
        self._transport = transport

    def post_json(
        self,
        request: TransportRequest,
        *,
        http_error_handler: Optional[HTTPErrorHandler] = None,
    ) -> JSONResponse:
        return self._transport.post_json(
            request, http_error_handler=http_error_handler
        )

    def post_multipart(
        self,
        request: TransportRequest,
        form: MultipartForm,
        *,
        http_error_handler: Optional[HTTPErrorHandler] = None,
    ) -> JSONResponse:
        return self._transport.post_multipart(
            request, form, http_error_handler=http_error_handler
        )

    def post_sse(
        self,
        request: TransportRequest,
        *,
        http_error_handler: Optional[HTTPErrorHandler] = None,
        stream_error_handler: Optional[StreamErrorHandler] = None,
    ) -> Iterator[SSEEvent]:
        events = self._transport.post_sse(
            request,
            http_error_handler=http_error_handler,
            stream_error_handler=stream_error_handler,
        )
        if callable(getattr(events, "close", None)):
            return events
        return _CloseableSSEIterator(events)


def _model_spec(model: str) -> Optional[ModelSpec]:
    """Parse one exact Mistral record through the shared typed registry model."""
    models = MODEL_METADATA.organization_data.get("models", {})
    data = models.get(model) if isinstance(models, Mapping) else None
    if not isinstance(data, Mapping):
        return None
    currency = MODEL_METADATA.organization_data.get("currency", "USD")
    if not isinstance(currency, str):
        currency = "USD"
    return ModelSpec.from_dict(
        model,
        data,
        currency=currency,
        request_rule_registry=MODEL_METADATA.request_rule_registry,
    )


def _native_reasoning_effort(
    model_spec: Optional[ModelSpec], kwargs: Mapping[str, Any]
) -> Optional[str]:
    """Resolve or preserve only an exact effort declared for this native model."""
    if model_spec is None or not isinstance(
        model_spec.reasoning_capability, CategoricalReasoningCapability
    ):
        return None
    allowed_values = model_spec.reasoning_capability.allowed_values

    reasoning_level = kwargs.get("reasoning_level")
    if isinstance(reasoning_level, (str, int)) and not isinstance(reasoning_level, bool):
        resolution = resolve_reasoning_level(model_spec, reasoning_level)
        value = resolution.provider_value
        return value if isinstance(value, str) and value in allowed_values else None

    raw_effort = kwargs.get("reasoning_effort")
    if isinstance(raw_effort, str) and raw_effort in allowed_values:
        return raw_effort

    raw_reasoning = kwargs.get("reasoning")
    if isinstance(raw_reasoning, Mapping):
        effort = raw_reasoning.get("effort")
        if isinstance(effort, str) and effort in allowed_values:
            return effort

    return allowed_values[0] if allowed_values else None
