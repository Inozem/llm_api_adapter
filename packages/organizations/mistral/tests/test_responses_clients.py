"""Native Mistral client integration with Core's compatible sync base."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Iterator, Mapping, Optional, get_type_hints

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[2]
for source in (
    str(PACKAGE_ROOT / "src"),
    str(REPOSITORY_ROOT / "src"),
    str(REPOSITORY_ROOT),
):
    if source not in sys.path:
        sys.path.insert(0, source)

from llm_api_adapter.errors.llm_api_error import (
    LLMAPIClientError,
    LLMAPIAuthorizationError,
)
from llm_api_adapter.llms.openai_compatible.sync_client import (
    OpenAICompatibleSyncClient,
)
from llm_api_adapter.llms.openai_compatible.async_client import (
    OpenAICompatibleAsyncClient,
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
from llm_api_adapter_mistral.clients.sync_client import (
    MISTRAL_OCR_URL,
    MistralSyncClient,
    _CloseableSSEIterator,
    _NativeRequestView,
    _CloseableSyncTransport,
    _model_spec,
    _native_reasoning_effort,
)
from llm_api_adapter_mistral.clients.async_client import (
    MistralAsyncClient,
    _NativeAsyncRequestView,
    _CloseableAsyncIterator,
    _ensure_closeable_async_iterator,
)
from llm_api_adapter_mistral.registry import MODEL_METADATA


class ClosableEvents:
    """Typed SSE source that exposes inherited stream cleanup."""

    events: tuple[SSEEvent, ...]
    closed: bool
    _iterator: Iterator[SSEEvent]

    def __init__(self, events: tuple[SSEEvent, ...]) -> None:
        self.events = events
        self.closed = False
        self._iterator = iter(events)

    def __iter__(self) -> ClosableEvents:
        return self

    def __next__(self) -> SSEEvent:
        return next(self._iterator)

    def close(self) -> None:
        self.closed = True


class RecordingSyncTransport(SyncTransport):
    """Capture requests at the real Core transport interface."""

    response: Any
    events: tuple[SSEEvent, ...]
    json_calls: list[tuple[TransportRequest, Optional[HTTPErrorHandler]]]
    multipart_calls: list[
        tuple[TransportRequest, MultipartForm, Optional[HTTPErrorHandler]]
    ]
    stream_calls: list[
        tuple[
            TransportRequest,
            Optional[HTTPErrorHandler],
            Optional[StreamErrorHandler],
        ]
    ]
    event_sources: list[ClosableEvents]

    def __init__(self, response: Any, events: tuple[SSEEvent, ...] = ()) -> None:
        self.response = response
        self.events = events
        self.json_calls: list[tuple[TransportRequest, Optional[HTTPErrorHandler]]] = []
        self.multipart_calls: list[
            tuple[TransportRequest, MultipartForm, Optional[HTTPErrorHandler]]
        ] = []
        self.stream_calls: list[
            tuple[
                TransportRequest,
                Optional[HTTPErrorHandler],
                Optional[StreamErrorHandler],
            ]
        ] = []
        self.event_sources: list[ClosableEvents] = []

    def post_json(
        self,
        request: TransportRequest,
        *,
        http_error_handler: Optional[HTTPErrorHandler] = None,
    ) -> JSONResponse:
        self.json_calls.append((request, http_error_handler))
        return JSONResponse(self.response)

    def post_multipart(
        self,
        request: TransportRequest,
        form: MultipartForm,
        *,
        http_error_handler: Optional[HTTPErrorHandler] = None,
    ) -> JSONResponse:
        self.multipart_calls.append((request, form, http_error_handler))
        return JSONResponse(self.response)

    def post_sse(
        self,
        request: TransportRequest,
        *,
        http_error_handler: Optional[HTTPErrorHandler] = None,
        stream_error_handler: Optional[StreamErrorHandler] = None,
    ) -> Iterator[SSEEvent]:
        self.stream_calls.append(
            (request, http_error_handler, stream_error_handler)
        )
        source = ClosableEvents(self.events)
        self.event_sources.append(source)
        return source


def _no_http_error(_error: Any) -> None:
    return None


def _no_stream_error(_event: SSEEvent) -> None:
    return None


def _reasoning_model(
    *, hosted: bool = False, non_default: bool = False
) -> tuple[str, str, str]:
    models = MODEL_METADATA.organization_data["models"]
    for model, record in models.items():
        if not isinstance(record, Mapping):
            continue
        capability = record.get("reasoning_capability")
        values = capability.get("allowed_values") if isinstance(capability, Mapping) else None
        if isinstance(values, list) and values and isinstance(values[0], str):
            if hosted and not isinstance(record.get("provider_model"), str):
                continue
            if non_default and len(values) < 2:
                continue
            wire_model = record.get("provider_model", model)
            if isinstance(wire_model, str):
                return model, wire_model, values[-1] if non_default else values[0]
    pytest.fail("Mistral registry must contain a reasoning-enabled model")


def _ocr_model() -> str:
    operations = MODEL_METADATA.organization_data["metered_operations"]
    operation = operations["ocr"]
    model = operation.get("model") if isinstance(operation, Mapping) else None
    if isinstance(model, str):
        return model
    pytest.fail("Mistral registry must declare the OCR model")


class SyntheticHTTPResponse:
    status_code: int
    payload: Mapping[str, Any]

    def __init__(self, status_code: int, payload: Mapping[str, Any]) -> None:
        self.status_code = status_code
        self.payload = payload

    def json(self) -> Mapping[str, Any]:
        return self.payload


class SyntheticHTTPError(Exception):
    response: SyntheticHTTPResponse

    def __init__(self, status_code: int, payload: Mapping[str, Any]) -> None:
        super().__init__("synthetic HTTP failure")
        self.response = SyntheticHTTPResponse(status_code, payload)


@pytest.mark.unit
def test_client_inherits_compatible_base_and_does_not_construct_an_extra_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import llm_api_adapter.llms.openai_compatible.sync_client as base_module

    calls: list[str] = []

    def forbidden_factory(transport: object) -> SyncTransport:
        calls.append(str(transport))
        raise AssertionError("native construction must retain the injected transport")

    monkeypatch.setattr(base_module, "create_sync_transport", forbidden_factory)
    transport = RecordingSyncTransport({"id": "resp_123"})

    client = MistralSyncClient(transport)

    assert MistralSyncClient.__bases__ == (OpenAICompatibleSyncClient,)
    assert isinstance(client, OpenAICompatibleSyncClient)
    assert client.transport is transport
    assert client._sync_transport is transport
    assert calls == []
    view = client._request_view({}, _no_http_error)
    assert isinstance(view, MistralSyncClient)
    assert isinstance(view, OpenAICompatibleSyncClient)
    assert view is not client


@pytest.mark.unit
def test_legacy_chat_uses_inherited_json_runner_with_raw_payload_and_full_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = RecordingSyncTransport({"id": "resp_legacy"})
    client = MistralSyncClient(transport)
    original = OpenAICompatibleSyncClient._send_request
    inherited_calls: list[tuple[str, dict[str, Any], Optional[float]]] = []

    def trace_inherited(
        owner: OpenAICompatibleSyncClient,
        url: str,
        payload: dict[str, Any],
        timeout: Optional[float] = None,
    ) -> JSONResponse:
        assert isinstance(owner, MistralSyncClient)
        inherited_calls.append((url, payload, timeout))
        return original(owner, url, payload, timeout)

    assert MistralSyncClient._send_request is OpenAICompatibleSyncClient._send_request
    monkeypatch.setattr(OpenAICompatibleSyncClient, "_send_request", trace_inherited)
    headers = {"Authorization": "Bearer synthetic-key", "X-Test": "native"}
    payload = {"model": "unregistered-model", "input": [{"role": "user"}]}

    result = client.chat(
        endpoint="https://proxy.invalid/prefix/v1/responses",
        headers=headers,
        payload=payload,
        timeout_s=3.5,
        http_error_handler=_no_http_error,
    )

    request, used_handler = transport.json_calls[0]
    assert result == {"id": "resp_legacy"}
    assert inherited_calls == [
        ("https://proxy.invalid/prefix/v1/responses", payload, 3.5)
    ]
    assert request.url == "https://proxy.invalid/prefix/v1/responses"
    assert request.headers_dict() == headers
    assert request.payload is payload
    assert request.timeout == 3.5
    assert (
        getattr(used_handler, "__self__")._request_http_error_handler
        is _no_http_error
    )


@pytest.mark.unit
def test_legacy_stream_uses_inherited_sse_runner_and_closes_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event = SSEEvent(event="response.completed", data={"type": "response.completed"})
    transport = RecordingSyncTransport({}, (event,))
    client = MistralSyncClient(transport)
    original = OpenAICompatibleSyncClient._stream_request
    inherited_calls: list[tuple[str, dict[str, Any], Optional[float]]] = []

    def trace_inherited(
        owner: OpenAICompatibleSyncClient,
        url: str,
        payload: dict[str, Any],
        timeout: Optional[float] = None,
    ) -> Iterator[SSEEvent]:
        assert isinstance(owner, MistralSyncClient)
        inherited_calls.append((url, payload, timeout))
        return original(owner, url, payload, timeout)

    assert (
        MistralSyncClient._stream_request
        is OpenAICompatibleSyncClient._stream_request
    )
    monkeypatch.setattr(OpenAICompatibleSyncClient, "_stream_request", trace_inherited)
    payload = {"model": "unregistered-model", "input": [], "stream": True}

    events = client.stream_chat(
        endpoint="https://api.mistral.ai/v1/responses",
        headers={"Authorization": "Bearer synthetic-key"},
        payload=payload,
        timeout_s=None,
        http_error_handler=_no_http_error,
        stream_error_handler=_no_stream_error,
    )

    assert tuple(events) == (event,)
    request, used_http_handler, used_stream_handler = transport.stream_calls[0]
    assert inherited_calls == [("https://api.mistral.ai/v1/responses", payload, None)]
    assert request.url == "https://api.mistral.ai/v1/responses"
    assert request.payload is payload
    binding = getattr(used_http_handler, "__self__")
    assert binding._request_http_error_handler is _no_http_error
    assert binding._request_stream_error_handler is _no_stream_error
    assert transport.event_sources[0].closed


@pytest.mark.unit
@pytest.mark.parametrize(
    ("hosted", "explicit_reasoning"),
    ((False, True), (False, False), (True, True), (True, False)),
)
def test_inherited_complete_dispatches_responses_with_registry_policy(
    monkeypatch: pytest.MonkeyPatch,
    hosted: bool,
    explicit_reasoning: bool,
) -> None:
    model, wire_model, effort = _reasoning_model(hosted=hosted)
    transport = RecordingSyncTransport({"id": "resp_native"})
    client = MistralSyncClient(transport)
    parent_complete = OpenAICompatibleSyncClient.complete
    dispatch_calls: list[str] = []

    def trace_complete(
        owner: OpenAICompatibleSyncClient,
        selected_model: str,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        assert isinstance(owner, MistralSyncClient)
        dispatch_calls.append(selected_model)
        return parent_complete(owner, selected_model, timeout, **kwargs)

    assert MistralSyncClient.complete is OpenAICompatibleSyncClient.complete
    monkeypatch.setattr(MistralSyncClient, "complete", trace_complete)
    headers = {"Authorization": "Bearer synthetic-key"}
    endpoint = "https://proxy.invalid/prefix/v1/responses"
    complete_kwargs: dict[str, Any] = {
        "endpoint": endpoint,
        "headers": headers,
        "http_error_handler": _no_http_error,
        "messages": [{"role": "user", "content": "synthetic prompt"}],
        "max_tokens": 19,
        "temperature": 1.0,
        "top_p": 1.0,
    }
    if explicit_reasoning:
        complete_kwargs["reasoning_effort"] = effort
    result = client.complete(model, **complete_kwargs)

    request, used_handler = transport.json_calls[0]
    assert result == {"id": "resp_native"}
    assert dispatch_calls == [model]
    assert request.url == endpoint
    assert request.payload == {
        "model": wire_model,
        "input": [{"role": "user", "content": "synthetic prompt"}],
        "max_output_tokens": 19,
        "reasoning": {"effort": effort},
        "temperature": 1.0,
        "top_p": 1.0,
        "store": False,
    }
    assert (
        getattr(used_handler, "__self__")._request_http_error_handler
        is _no_http_error
    )


@pytest.mark.unit
def test_responses_preserves_supported_nested_effort_and_strips_local_fields() -> None:
    model, wire_model, effort = _reasoning_model(non_default=True)
    transport = RecordingSyncTransport({"id": "resp_reasoning"})
    client = MistralSyncClient(transport)

    assert client.complete(
        model,
        endpoint="https://api.mistral.ai/v1/responses",
        headers={"Authorization": "Bearer synthetic-key"},
        messages=[{"role": "user", "content": "synthetic prompt"}],
        max_tokens=11,
        reasoning={"effort": effort},
        capture_reasoning=True,
        conversation="synthetic-conversation",
        stream_options={"include_usage": True},
        previous_response="synthetic-prior-response",
        json_schema={"type": "object"},
        response_model="SyntheticResponseModel",
        temperature=1.0,
        top_p=1.0,
    ) == {"id": "resp_reasoning"}

    request = transport.json_calls[0][0]
    assert request.payload["model"] == wire_model
    assert request.payload["reasoning"] == {"effort": effort}
    assert request.payload["temperature"] == 1.0
    assert request.payload["top_p"] == 1.0
    assert not {
        "capture_reasoning",
        "conversation",
        "stream_options",
        "previous_response",
        "json_schema",
        "response_model",
    } & set(request.payload)
    assert request.payload["reasoning"]["effort"] == effort
    assert effort in MODEL_METADATA.organization_data["models"][model][
        "reasoning_capability"
    ]["allowed_values"]


@pytest.mark.unit
def test_responses_image_detail_does_not_mutate_caller_input() -> None:
    model = "mistral-small-2603"
    original_input = [
        {
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": "data:image/png;base64,AA=="}
            ],
        }
    ]
    expected_input = [
        {
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": "data:image/png;base64,AA=="}
            ],
        }
    ]

    payload = MistralSyncClient._prepare_responses_payload_for_model(
        MistralSyncClient(RecordingSyncTransport({})),
        model,
        {"input": original_input},
    )

    assert original_input == expected_input
    assert payload["input"] == [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,AA==",
                    "detail": "auto",
                }
            ],
        }
    ]
    assert payload["input"] is not original_input


@pytest.mark.unit
def test_inherited_stream_dispatches_responses_without_rewriting_full_url() -> None:
    assert MistralSyncClient.stream is OpenAICompatibleSyncClient.stream
    model, wire_model, effort = _reasoning_model()
    event = SSEEvent(event="response.completed", data={"type": "response.completed"})
    transport = RecordingSyncTransport({}, (event,))
    client = MistralSyncClient(transport)
    endpoint = "https://proxy.invalid/versioned/api/responses"
    payload = client.stream(
        model,
        endpoint=endpoint,
        headers={"Authorization": "Bearer synthetic-key"},
        http_error_handler=_no_http_error,
        stream_error_handler=_no_stream_error,
        messages=[{"role": "user", "content": "synthetic prompt"}],
        max_tokens=13,
        reasoning_effort=effort,
        temperature=1.0,
        top_p=1.0,
    )

    assert tuple(payload) == (event,)
    request, used_http_handler, used_stream_handler = transport.stream_calls[0]
    assert request.url == endpoint
    assert request.payload == {
        "model": wire_model,
        "input": [{"role": "user", "content": "synthetic prompt"}],
        "max_output_tokens": 13,
        "reasoning": {"effort": effort},
        "temperature": 1.0,
        "top_p": 1.0,
        "store": False,
        "stream": True,
    }
    binding = getattr(used_http_handler, "__self__")
    assert binding._request_http_error_handler is _no_http_error
    assert binding._request_stream_error_handler is _no_stream_error
    assert transport.event_sources[0].closed


@pytest.mark.unit
def test_lazy_stream_snapshots_headers_and_transport_before_iteration() -> None:
    model, _wire_model, _effort = _reasoning_model()
    event = SSEEvent(event="response.completed", data={"type": "response.completed"})
    original = RecordingSyncTransport({}, (event,))
    replacement = RecordingSyncTransport({}, ())
    client = MistralSyncClient(original)
    headers = {"Authorization": "Bearer synthetic-key", "X-Snapshot": "original"}
    stream = client.stream_chat(
        endpoint="https://api.mistral.ai/v1/responses",
        headers=headers,
        payload={"model": model, "input": []},
        timeout_s=None,
        http_error_handler=_no_http_error,
        stream_error_handler=_no_stream_error,
    )

    headers["X-Snapshot"] = "changed"
    client.transport = replacement
    assert tuple(stream) == (event,)

    request = original.stream_calls[0][0]
    assert request.headers_dict()["X-Snapshot"] == "original"
    assert replacement.stream_calls == []
    assert original.event_sources[0].closed


@pytest.mark.unit
def test_default_native_complete_and_stream_errors_map_403_to_authorization() -> None:
    model, _wire_model, _effort = _reasoning_model()

    class ForbiddenTransport(RecordingSyncTransport):
        def post_json(
            self,
            request: TransportRequest,
            *,
            http_error_handler: Optional[HTTPErrorHandler] = None,
        ) -> JSONResponse:
            if http_error_handler is None:
                raise AssertionError("inherited runner must provide its native handler")
            http_error_handler(
                SyntheticHTTPError(403, {"error": {"message": "forbidden"}})
            )
            return JSONResponse({})

    with pytest.raises(LLMAPIAuthorizationError):
        MistralSyncClient(ForbiddenTransport({})).complete(
            model,
            endpoint="https://api.mistral.ai/v1/responses",
            headers={"Authorization": "Bearer synthetic-key"},
            input=[],
        )

    failed = SSEEvent(
        event="response.failed",
        data={
            "type": "response.failed",
            "response": {
                "error": {"type": "authentication_error", "message": "forbidden"}
            },
        },
    )
    with pytest.raises(LLMAPIAuthorizationError):
        list(
            MistralSyncClient(RecordingSyncTransport({}, (failed,))).stream(
                model,
                endpoint="https://api.mistral.ai/v1/responses",
                headers={"Authorization": "Bearer synthetic-key"},
                input=[],
            )
        )


@pytest.mark.unit
def test_native_error_mapper_accepts_core_positional_argument_contract() -> None:
    with pytest.raises(LLMAPIAuthorizationError):
        MistralSyncClient._raise_mapped_error(403, None, "synthetic forbidden")


@pytest.mark.unit
def test_transport_replacement_is_used_and_handler_exception_identity_is_preserved() -> None:
    initial = RecordingSyncTransport({"unused": True})
    replacement = RecordingSyncTransport({"replaced": True})
    client = MistralSyncClient(initial)
    client.transport = replacement

    assert client.chat(
        endpoint="https://api.mistral.ai/v1/responses",
        headers={},
        payload={},
        timeout_s=None,
        http_error_handler=_no_http_error,
    ) == {"replaced": True}
    assert initial.json_calls == []
    assert len(replacement.json_calls) == 1

    caller_error = RuntimeError("synthetic caller error")

    def raise_caller_error(_error: Any) -> None:
        raise caller_error

    class ErrorTransport(RecordingSyncTransport):
        def post_json(
            self,
            request: TransportRequest,
            *,
            http_error_handler: Optional[HTTPErrorHandler] = None,
        ) -> JSONResponse:
            if http_error_handler is None:
                raise AssertionError("native handler must be supplied")
            http_error_handler(caller_error)
            return JSONResponse({})

    error_client = MistralSyncClient(ErrorTransport({}))
    with pytest.raises(RuntimeError) as caught:
        error_client.chat(
            endpoint="https://api.mistral.ai/v1/responses",
            headers={},
            payload={},
            timeout_s=None,
            http_error_handler=raise_caller_error,
        )
    assert caught.value is caller_error


@pytest.mark.unit
def test_ocr_keeps_its_existing_resource_and_non_object_error() -> None:
    transport = RecordingSyncTransport({"pages": []})
    client = MistralSyncClient(transport)

    assert client.ocr(
        headers={"Authorization": "Bearer synthetic-key"},
        payload={"model": _ocr_model()},
        timeout_s=7,
        http_error_handler=_no_http_error,
    ) == {"pages": []}
    assert transport.json_calls[0][0].url == MISTRAL_OCR_URL

    malformed = MistralSyncClient(RecordingSyncTransport(["not", "an", "object"]))
    with pytest.raises(LLMAPIClientError, match="non-object response"):
        malformed.ocr(
            headers={},
            payload={},
            timeout_s=None,
            http_error_handler=_no_http_error,
        )


@pytest.mark.unit
def test_changed_sync_client_annotations_resolve_and_legacy_callbacks_are_unchanged() -> None:
    assert MistralSyncClient.chat.__annotations__["http_error_handler"] == (
        "Callable[[Any], None]"
    )
    assert MistralSyncClient.stream_chat.__annotations__["http_error_handler"] == (
        "Callable[[Any], None]"
    )
    assert MistralSyncClient.stream_chat.__annotations__["stream_error_handler"] == (
        "Callable[[SSEEvent], None]"
    )
    assert MistralSyncClient.ocr.__annotations__["http_error_handler"] == (
        "Callable[[Any], None]"
    )
    targets = (
        MistralSyncClient.__init__,
        MistralSyncClient.transport.fget,
        MistralSyncClient.transport.fset,
        MistralSyncClient.chat,
        MistralSyncClient.ocr,
        MistralSyncClient.stream_chat,
        MistralSyncClient._should_use_responses_api,
        MistralSyncClient.responses,
        MistralSyncClient.stream_responses,
        MistralSyncClient._prepare_responses_payload_for_model,
        MistralSyncClient._post_json,
        MistralSyncClient._request_view,
        MistralSyncClient._handle_http_error,
        MistralSyncClient._handle_stream_error,
        MistralSyncClient._raise_mapped_error,
        _NativeRequestView.__init__,
        _NativeRequestView._headers,
        _NativeRequestView._handle_http_error,
        _NativeRequestView._handle_stream_error,
        _CloseableSSEIterator.__init__,
        _CloseableSSEIterator.__iter__,
        _CloseableSSEIterator.__next__,
        _CloseableSSEIterator.close,
        _CloseableSyncTransport.__init__,
        _CloseableSyncTransport.post_json,
        _CloseableSyncTransport.post_multipart,
        _CloseableSyncTransport.post_sse,
        _model_spec,
        _native_reasoning_effort,
    )
    assert all(get_type_hints(target) for target in targets)


@pytest.mark.unit
def test_async_client_inherits_common_async_engine_and_native_sync_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import inspect

    import llm_api_adapter.llms.openai_compatible.sync_client as base_module

    def forbidden_factory(_transport: str) -> SyncTransport:
        raise AssertionError("native async construction must not create sync transport")

    monkeypatch.setattr(base_module, "create_sync_transport", forbidden_factory)
    client = MistralAsyncClient()
    assert not inspect.signature(MistralAsyncClient).parameters
    assert not inspect.iscoroutinefunction(MistralAsyncClient.stream_chat)

    assert MistralAsyncClient.__bases__ == (
        OpenAICompatibleAsyncClient,
        MistralSyncClient,
    )
    assert isinstance(client, OpenAICompatibleAsyncClient)
    assert isinstance(client, MistralSyncClient)
    assert MistralAsyncClient.complete is OpenAICompatibleAsyncClient.complete
    assert MistralAsyncClient.stream is OpenAICompatibleAsyncClient.stream
    assert MistralAsyncClient._send_request is OpenAICompatibleAsyncClient._send_request
    assert MistralAsyncClient._stream_request is OpenAICompatibleAsyncClient._stream_request
    assert (
        MistralAsyncClient._prepare_responses_payload_for_model
        is MistralSyncClient._prepare_responses_payload_for_model
    )
    assert client.endpoint == "https://api.mistral.ai/v1/responses"
    assert "_sync_transport" not in vars(client)


@pytest.mark.unit
def test_inherited_async_complete_uses_native_payload_and_async_patch_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    import llm_api_adapter_mistral.clients.async_client as async_client_module

    model, wire_model, effort = _reasoning_model(hosted=True)
    calls: list[tuple[str, dict[str, Any], Mapping[str, str], Optional[float]]] = []
    common_calls: list[tuple[OpenAICompatibleAsyncClient, str]] = []

    async def fake_request(
        url: str,
        *,
        headers: Mapping[str, str],
        payload: dict[str, Any],
        timeout: Optional[float],
        http_error_handler: Callable[[Any], None],
    ) -> dict[str, Any]:
        calls.append((url, payload, headers, timeout))
        return {"id": "async_resp_native"}

    monkeypatch.setattr(async_client_module, "async_request", fake_request)
    original_send = OpenAICompatibleAsyncClient._send_request

    async def trace_send(
        owner: OpenAICompatibleAsyncClient,
        url: str,
        payload: dict[str, Any],
        timeout: Optional[float] = None,
    ) -> Any:
        assert isinstance(owner, MistralAsyncClient)
        common_calls.append((owner, url))
        return await original_send(owner, url, payload, timeout)

    monkeypatch.setattr(OpenAICompatibleAsyncClient, "_send_request", trace_send)
    client = MistralAsyncClient()
    headers = {"Authorization": "Bearer synthetic-key", "X-Test": "native"}
    endpoint = "https://proxy.invalid/versioned/v1/responses"

    result = asyncio.run(
        client.complete(
            model,
            endpoint=endpoint,
            headers=headers,
            timeout=6.25,
            messages=[{"role": "user", "content": "synthetic prompt"}],
            max_tokens=21,
            reasoning_effort=effort,
            temperature=1.0,
            top_p=1.0,
        )
    )

    assert result == {"id": "async_resp_native"}
    assert calls == [
        (
            endpoint,
            {
                "model": wire_model,
                "input": [{"role": "user", "content": "synthetic prompt"}],
                "max_output_tokens": 21,
                "reasoning": {"effort": effort},
                "temperature": 1.0,
                "top_p": 1.0,
                "store": False,
            },
            headers,
            6.25,
        )
    ]
    assert len(common_calls) == 1
    assert common_calls[0][1] == endpoint


@pytest.mark.unit
def test_async_low_level_methods_keep_raw_payload_ocr_url_and_error_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    import llm_api_adapter_mistral.clients.async_client as async_client_module

    calls: list[tuple[str, dict[str, Any], Mapping[str, str], Optional[float]]] = []

    async def fake_request(
        url: str,
        *,
        headers: Mapping[str, str],
        payload: dict[str, Any],
        timeout: Optional[float],
        http_error_handler: Callable[[Any], None],
    ) -> dict[str, Any]:
        calls.append((url, payload, headers, timeout))
        return {"ok": True}

    monkeypatch.setattr(async_client_module, "async_request", fake_request)
    client = MistralAsyncClient()
    payload = {"model": "raw-model", "input": [{"raw": True}]}
    headers = {"Authorization": "Bearer synthetic-key"}

    async def run() -> tuple[dict[str, Any], dict[str, Any]]:
        chat = await client.chat(
            endpoint="https://proxy.invalid/prefix/v1/responses",
            headers=headers,
            payload=payload,
            timeout_s=4,
            http_error_handler=_no_http_error,
        )
        ocr = await client.ocr(
            headers=headers,
            payload={"model": _ocr_model()},
            timeout_s=None,
            http_error_handler=_no_http_error,
        )
        return chat, ocr

    assert asyncio.run(run()) == ({"ok": True}, {"ok": True})
    assert calls == [
        ("https://proxy.invalid/prefix/v1/responses", payload, headers, 4),
        (MISTRAL_OCR_URL, {"model": _ocr_model()}, headers, None),
    ]

    caller_error = RuntimeError("synthetic callback failure")

    def raise_caller_error(_error: Any) -> None:
        raise caller_error

    async def failed_request(*_args: Any, **kwargs: Any) -> Any:
        handler = kwargs.get("http_error_handler")
        assert handler is raise_caller_error
        handler(caller_error)
        raise AssertionError("caller handler should raise")

    monkeypatch.setattr(async_client_module, "async_request", failed_request)
    with pytest.raises(RuntimeError) as caught:
        asyncio.run(
            client.chat(
                endpoint="https://api.mistral.ai/v1/responses",
                headers=headers,
                payload=payload,
                timeout_s=None,
                http_error_handler=raise_caller_error,
            )
        )
    assert caught.value is caller_error


@pytest.mark.unit
def test_async_stream_factory_is_lazy_and_preserves_complete_url_and_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    import llm_api_adapter_mistral.clients.async_client as async_client_module

    model, wire_model, _effort = _reasoning_model()
    event = SSEEvent(event="response.completed", data={"type": "response.completed"})
    calls: list[tuple[str, dict[str, Any], Mapping[str, str], Optional[float]]] = []
    closed: list[bool] = []
    started: list[bool] = []

    async def source() -> AsyncIterator[SSEEvent]:
        started.append(True)
        try:
            yield event
        finally:
            closed.append(True)

    def fake_stream(
        url: str,
        *,
        headers: Mapping[str, str],
        payload: dict[str, Any],
        timeout: Optional[float],
        http_error_handler: Callable[[Any], None],
        stream_error_handler: Callable[[SSEEvent], None],
    ) -> AsyncIterator[SSEEvent]:
        calls.append((url, payload, headers, timeout))
        return source()

    monkeypatch.setattr(async_client_module, "async_stream_request", fake_stream)
    client = MistralAsyncClient()
    headers = {"Authorization": "Bearer synthetic-key"}
    endpoint = "https://proxy.invalid/prefix/v1/responses"
    stream = client.stream(
        model,
        endpoint=endpoint,
        headers=headers,
        timeout=2.75,
        messages=[{"role": "user", "content": "synthetic prompt"}],
        max_tokens=12,
    )
    assert calls == []
    assert started == []

    async def consume() -> tuple[SSEEvent, ...]:
        return tuple([item async for item in stream])

    assert asyncio.run(consume()) == (event,)
    assert len(calls) == 1
    assert calls[0][0] == endpoint
    assert calls[0][2:] == (headers, 2.75)
    assert started == [True]
    assert closed == [True]
    assert calls == [
        (
            endpoint,
            {
                "model": wire_model,
                "input": [{"role": "user", "content": "synthetic prompt"}],
                "max_output_tokens": 12,
                "reasoning": {"effort": _effort},
                "store": False,
                "stream": True,
            },
            headers,
            2.75,
        )
    ]


@pytest.mark.unit
def test_async_client_changed_annotations_resolve() -> None:
    targets = (
        MistralAsyncClient.__init__,
        MistralAsyncClient.chat,
        MistralAsyncClient.ocr,
        MistralAsyncClient.stream_chat,
        MistralAsyncClient.responses,
        MistralAsyncClient.stream_responses,
        MistralAsyncClient._async_request_view,
        MistralAsyncClient._get_async_request,
        MistralAsyncClient._get_async_stream_request,
        _NativeAsyncRequestView.__init__,
        _NativeAsyncRequestView._headers,
        _NativeAsyncRequestView._handle_http_error,
        _NativeAsyncRequestView._handle_stream_error,
        _NativeAsyncRequestView._get_async_request,
        _NativeAsyncRequestView._get_async_stream_request,
        _CloseableAsyncIterator.__init__,
        _CloseableAsyncIterator.__aiter__,
        _CloseableAsyncIterator.__anext__,
        _CloseableAsyncIterator.aclose,
        _ensure_closeable_async_iterator,
    )
    assert all(get_type_hints(target) for target in targets)


@pytest.mark.unit
def test_async_stream_cancellation_closes_the_inherited_event_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    import llm_api_adapter_mistral.clients.async_client as async_client_module

    source_started = asyncio.Event()
    source_closed: list[bool] = []

    async def source() -> AsyncIterator[SSEEvent]:
        source_started.set()
        try:
            await asyncio.Event().wait()
            if False:
                yield SSEEvent(event="unused", data={})
        finally:
            source_closed.append(True)

    def fake_stream(
        _url: str,
        *,
        headers: Mapping[str, str],
        payload: dict[str, Any],
        timeout: Optional[float],
        http_error_handler: Callable[[Any], None],
        stream_error_handler: Callable[[SSEEvent], None],
    ) -> AsyncIterator[SSEEvent]:
        return source()

    monkeypatch.setattr(async_client_module, "async_stream_request", fake_stream)
    client = MistralAsyncClient()
    stream = client.stream_chat(
        endpoint="https://api.mistral.ai/v1/responses",
        headers={"Authorization": "Bearer synthetic-key"},
        payload={"model": "raw-model", "input": [], "stream": True},
        timeout_s=None,
        http_error_handler=_no_http_error,
        stream_error_handler=_no_stream_error,
    )

    async def consume_until_cancelled() -> None:
        async for _event in stream:
            pass

    async def run() -> None:
        task = asyncio.create_task(consume_until_cancelled())
        await source_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert source_closed == [True]


@pytest.mark.unit
def test_async_stream_adapts_legacy_iterator_without_aclose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    import llm_api_adapter_mistral.clients.async_client as async_client_module

    event = SSEEvent(event="response.completed", data={"type": "response.completed"})

    class BareEvents:
        _events: tuple[SSEEvent, ...]
        _index: int

        def __init__(self, events: tuple[SSEEvent, ...]) -> None:
            self._events = events
            self._index = 0

        def __aiter__(self) -> BareEvents:
            return self

        async def __anext__(self) -> SSEEvent:
            if self._index >= len(self._events):
                raise StopAsyncIteration
            event_value = self._events[self._index]
            self._index += 1
            return event_value

    def fake_stream(*_args: Any, **_kwargs: Any) -> AsyncIterator[SSEEvent]:
        return BareEvents((event,))

    monkeypatch.setattr(async_client_module, "async_stream_request", fake_stream)
    client = MistralAsyncClient()
    stream = client.stream_chat(
        endpoint="https://api.mistral.ai/v1/responses",
        headers={},
        payload={"model": "raw-model", "input": [], "stream": True},
        timeout_s=None,
        http_error_handler=_no_http_error,
        stream_error_handler=_no_stream_error,
    )

    async def consume() -> tuple[SSEEvent, ...]:
        return tuple([value async for value in stream])

    assert asyncio.run(consume()) == (event,)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("method", "expected_detail"),
    (
        ("chat", "Mistral returned a non-object response"),
        ("ocr", "Mistral OCR returned a non-object response"),
    ),
)
def test_async_legacy_json_calls_reject_non_object_responses(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    expected_detail: str,
) -> None:
    import asyncio

    import llm_api_adapter_mistral.clients.async_client as async_client_module

    async def fake_request(*_args: Any, **_kwargs: Any) -> list[str]:
        return ["not", "an", "object"]

    monkeypatch.setattr(async_client_module, "async_request", fake_request)
    client = MistralAsyncClient()

    async def call() -> dict[str, Any]:
        if method == "chat":
            return await client.chat(
                endpoint="https://api.mistral.ai/v1/responses",
                headers={},
                payload={},
                timeout_s=None,
                http_error_handler=_no_http_error,
            )
        return await client.ocr(
            headers={},
            payload={},
            timeout_s=None,
            http_error_handler=_no_http_error,
        )

    with pytest.raises(LLMAPIClientError, match=expected_detail):
        asyncio.run(call())


@pytest.mark.unit
def test_async_native_default_http_and_sse_errors_reuse_sync_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    import llm_api_adapter_mistral.clients.async_client as async_client_module

    async def forbidden_request(*_args: Any, **kwargs: Any) -> Any:
        handler = kwargs["http_error_handler"]
        handler(SyntheticHTTPError(403, {"error": {"message": "forbidden"}}))
        return {"unused": True}

    monkeypatch.setattr(async_client_module, "async_request", forbidden_request)
    client = MistralAsyncClient()
    model = _reasoning_model()[0]

    async def call_http() -> None:
        await client.complete(
            model,
            endpoint="https://api.mistral.ai/v1/responses",
            headers={"Authorization": "Bearer synthetic-key"},
            input=[],
        )

    with pytest.raises(LLMAPIAuthorizationError):
        asyncio.run(call_http())

    async def failed_events(*_args: Any, **_kwargs: Any) -> AsyncIterator[SSEEvent]:
        yield SSEEvent(
            event="response.failed",
            data={
                "type": "response.failed",
                "response": {
                    "error": {
                        "type": "authentication_error",
                        "message": "forbidden",
                    }
                },
            },
        )

    monkeypatch.setattr(async_client_module, "async_stream_request", failed_events)
    stream = client.stream(
        model,
        endpoint="https://api.mistral.ai/v1/responses",
        headers={"Authorization": "Bearer synthetic-key"},
        input=[],
    )

    async def consume() -> None:
        async for _event in stream:
            pass

    with pytest.raises(LLMAPIAuthorizationError):
        asyncio.run(consume())


@pytest.mark.unit
def test_async_stream_early_close_closes_a_real_async_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    import llm_api_adapter_mistral.clients.async_client as async_client_module

    event = SSEEvent(event="message", data={"delta": "one"})
    closed: list[bool] = []

    async def source() -> AsyncIterator[SSEEvent]:
        try:
            yield event
            await asyncio.Event().wait()
        finally:
            closed.append(True)

    def fake_stream(*_args: Any, **_kwargs: Any) -> AsyncIterator[SSEEvent]:
        return source()

    monkeypatch.setattr(async_client_module, "async_stream_request", fake_stream)
    client = MistralAsyncClient()

    async def run() -> SSEEvent:
        stream = client.stream_chat(
            endpoint="https://api.mistral.ai/v1/responses",
            headers={},
            payload={"model": "raw-model", "input": [], "stream": True},
            timeout_s=None,
            http_error_handler=_no_http_error,
            stream_error_handler=_no_stream_error,
        )
        first = await stream.__anext__()
        close = getattr(stream, "aclose", None)
        assert callable(close)
        await close()
        return first

    assert asyncio.run(run()) == event
    assert closed == [True]
