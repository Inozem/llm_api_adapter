"""Native Mistral Responses reasoning policy and opt-in capture."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator, Mapping
import ipaddress
import json
from pathlib import Path
import socket
import sys
from typing import Any
import warnings

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[2]
for source in (str(PACKAGE_ROOT / "src"), str(REPOSITORY_ROOT / "src"), str(REPOSITORY_ROOT)):
    if source not in sys.path:
        sys.path.insert(0, source)

import llm_api_adapter.adapters.base_adapter as base_adapter_module
import llm_api_adapter_mistral.clients.async_client as async_client_module
from llm_api_adapter.adapters.base_adapter import AsyncOnReasoning, OnReasoning
from llm_api_adapter.llm_registry.llm_registry import CategoricalReasoningCapability, RegistrySpec
from llm_api_adapter.llm_registry.reasoning import resolve_reasoning_level
from llm_api_adapter.llms.transports import (
    HTTPErrorHandler,
    JSONResponse,
    MultipartForm,
    SSEEvent,
    StreamErrorHandler,
    SyncTransport,
    TransportRequest,
)
from llm_api_adapter.models.responses.chat_response import ChatResponse
from llm_api_adapter.models.responses.reasoning_event import ReasoningEvent
from llm_api_adapter_mistral.adapter import MistralAdapter
from llm_api_adapter_mistral.clients.async_client import MistralAsyncClient
from llm_api_adapter_mistral.clients.sync_client import MistralSyncClient
from llm_api_adapter_mistral.plugin import PLUGIN


MODEL = "mistral-large-4"
METHODS = ("chat", "achat", "stream_chat", "astream_chat")
OPAQUE_RESPONSE_ID = "never-replay-this-response-id"
REASONING_TEXT = "Keep this thought separate."
VISIBLE_TEXT = "The answer is ready."


class ReasoningTransport(SyncTransport):
    """Capture real native requests without opening a network connection."""

    def __init__(self, response: dict[str, Any], events: tuple[SSEEvent, ...]) -> None:
        self.response = response
        self.events = events
        self.requests: list[TransportRequest] = []

    def post_json(
        self, request: TransportRequest, *, http_error_handler: HTTPErrorHandler | None = None
    ) -> JSONResponse:
        self.requests.append(request)
        return JSONResponse(self.response)

    def post_sse(
        self,
        request: TransportRequest,
        *,
        http_error_handler: HTTPErrorHandler | None = None,
        stream_error_handler: StreamErrorHandler | None = None,
    ) -> Iterator[SSEEvent]:
        self.requests.append(request)
        return iter(self.events)

    def post_multipart(
        self,
        request: TransportRequest,
        form: MultipartForm,
        *,
        http_error_handler: HTTPErrorHandler | None = None,
    ) -> JSONResponse:
        raise AssertionError("reasoning tests do not submit multipart requests")


@pytest.fixture
def mistral_registry(monkeypatch: pytest.MonkeyPatch) -> RegistrySpec:
    registry = RegistrySpec()
    assert PLUGIN.model_metadata is not None
    assert registry.register_organization_metadata(PLUGIN.model_metadata)
    monkeypatch.setattr(base_adapter_module, "LLM_REGISTRY", registry)
    return registry


@pytest.fixture(autouse=True)
def deny_external_network(monkeypatch: pytest.MonkeyPatch) -> None:
    original_getaddrinfo = socket.getaddrinfo
    original_connect = socket.socket.connect

    def is_loopback(host: Any) -> bool:
        if host in {"localhost", b"localhost"}:
            return True
        if isinstance(host, bytes):
            host = host.decode("ascii", "ignore")
        try:
            return ipaddress.ip_address(host).is_loopback
        except (ValueError, TypeError):
            return False

    def guarded_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> list[tuple[Any, ...]]:
        if not is_loopback(host):
            raise AssertionError("reasoning test attempted external DNS")
        return original_getaddrinfo(host, *args, **kwargs)

    def guarded_connect(connection: socket.socket, address: Any) -> None:
        if isinstance(address, tuple) and address and not is_loopback(address[0]):
            raise AssertionError("reasoning test attempted external network")
        original_connect(connection, address)

    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


def _completed_response(*, with_reasoning: bool = True) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {"type": "output_text", "text": VISIBLE_TEXT},
    ]
    output = [{"type": "message", "role": "assistant", "status": "completed", "content": content}]
    if with_reasoning:
        output.append({
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": REASONING_TEXT}],
            "content": [{"type": "reasoning_text", "text": "Provider reasoning content."}],
        })
    return {"object": "response", "id": "resp_reasoning", "model": MODEL, "status": "completed", "output": output}


def _reasoning_events(*, with_reasoning: bool = True) -> tuple[SSEEvent, ...]:
    events = [SSEEvent(event="response.output_text.delta", data={"delta": VISIBLE_TEXT})]
    if with_reasoning:
        events.append(SSEEvent(event="response.reasoning_summary_text.delta", data={"delta": REASONING_TEXT}))
        events.append(SSEEvent(event="response.reasoning_text.delta", data={"delta": "Provider reasoning content."}))
    events.append(SSEEvent(event="response.completed", data={"response": _completed_response(with_reasoning=with_reasoning)}))
    return tuple(events)


def _install_async(
    monkeypatch: pytest.MonkeyPatch, response: dict[str, Any], events: tuple[SSEEvent, ...]
) -> list[TransportRequest]:
    captured: list[TransportRequest] = []

    async def request(
        url: str, *, headers: Mapping[str, str] | None = None, payload: Any = None,
        timeout: float | None = None, http_error_handler: HTTPErrorHandler | None = None,
    ) -> dict[str, Any]:
        captured.append(TransportRequest(url=url, headers=headers or {}, payload=payload, timeout=timeout))
        return response

    def stream_request(
        url: str, *, headers: Mapping[str, str] | None = None, payload: Any = None,
        timeout: float | None = None, http_error_handler: HTTPErrorHandler | None = None,
        stream_error_handler: StreamErrorHandler | None = None,
    ) -> AsyncIterator[SSEEvent]:
        captured.append(TransportRequest(url=url, headers=headers or {}, payload=payload, timeout=timeout))

        async def source() -> AsyncIterator[SSEEvent]:
            for event in events:
                yield event

        return source()

    monkeypatch.setattr(async_client_module, "async_request", request)
    monkeypatch.setattr(async_client_module, "async_stream_request", stream_request)
    return captured


def _invoke(
    adapter: MistralAdapter, method: str, *, capture: bool,
    completed: list[ChatResponse], reasoning: list[ReasoningEvent],
    previous_response: ChatResponse | None = None, reasoning_level: str | int | None = None,
    on_reasoning: OnReasoning | None = None, async_on_reasoning: AsyncOnReasoning | None = None,
) -> tuple[ChatResponse, list[str]]:
    visible: list[str] = []
    sync_reasoning: OnReasoning = on_reasoning or reasoning.append

    async def collect_reasoning(event: ReasoningEvent) -> None:
        reasoning.append(event)

    async_reasoning: AsyncOnReasoning = async_on_reasoning or collect_reasoning
    if method == "chat":
        return adapter.chat(
            [{"role": "user", "content": "Answer."}], capture_reasoning=capture,
            previous_response=previous_response, reasoning_level=reasoning_level,
        ), visible
    if method == "achat":
        return asyncio.run(adapter.achat(
            [{"role": "user", "content": "Answer."}], capture_reasoning=capture,
            previous_response=previous_response, reasoning_level=reasoning_level,
        )), visible
    if method == "stream_chat":
        visible = list(adapter.stream_chat(
            [{"role": "user", "content": "Answer."}], capture_reasoning=capture,
            previous_response=previous_response, reasoning_level=reasoning_level,
            on_done=completed.append, on_reasoning=sync_reasoning,
        ))
        return completed[0], visible
    return asyncio.run(_collect_async_stream(
        adapter, capture=capture, completed=completed, previous_response=previous_response,
        reasoning_level=reasoning_level, on_reasoning=async_reasoning,
    ))


async def _collect_async_stream(
    adapter: MistralAdapter, *, capture: bool, completed: list[ChatResponse],
    previous_response: ChatResponse | None, reasoning_level: str | int | None,
    on_reasoning: AsyncOnReasoning,
) -> tuple[ChatResponse, list[str]]:
    visible = [text async for text in adapter.astream_chat(
        [{"role": "user", "content": "Answer."}], capture_reasoning=capture,
        previous_response=previous_response, reasoning_level=reasoning_level,
        on_done=completed.append, on_reasoning=on_reasoning,
    )]
    return completed[0], visible


def _assert_no_replay_controls(payload: dict[str, Any]) -> None:
    assert payload["store"] is False
    for key in ("previous_response_id", "reasoning_summary", "include", "encrypted_content"):
        assert key not in payload
    assert OPAQUE_RESPONSE_ID not in json.dumps(payload)


@pytest.mark.integration
@pytest.mark.parametrize("model", ["mistral-large-4", "mistral-small-2603", "mistral-medium-3-5", "mistral-large-2512", "glm-5.3-flash", "deepseek-flash"])
def test_registry_profiles_drive_default_and_explicit_native_effort(
    mistral_registry: RegistrySpec, monkeypatch: pytest.MonkeyPatch, model: str,
) -> None:
    response = _completed_response(with_reasoning=False)
    events = _reasoning_events(with_reasoning=False)
    adapter = MistralAdapter(api_key="synthetic-key", model=model, company="mistral")
    sync_transport = ReasoningTransport(response, events)
    adapter._sync_transport = sync_transport
    async_requests = _install_async(monkeypatch, response, events)
    assert adapter.model_spec is not None

    capability = adapter.model_spec.reasoning_capability
    expected_default = (
        capability.allowed_values[0]
        if isinstance(capability, CategoricalReasoningCapability) and capability.allowed_values
        else None
    )
    with warnings.catch_warnings(record=True) as default_warnings:
        warnings.simplefilter("always")
        result = adapter.chat([{"role": "user", "content": "Default."}])
    assert default_warnings == []
    assert result.content == VISIBLE_TEXT
    sync_payload = sync_transport.requests[-1].payload
    if expected_default is None:
        assert "reasoning" not in sync_payload
    else:
        assert sync_payload["reasoning"] == {"effort": expected_default}

    requested_level = (
        capability.allowed_values[-1]
        if isinstance(capability, CategoricalReasoningCapability) and capability.allowed_values
        else "high"
    )
    expected = resolve_reasoning_level(adapter.model_spec, requested_level)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        asyncio.run(adapter.achat([{"role": "user", "content": "Explicit."}], reasoning_level=requested_level))
    assert [str(item.message) for item in caught] == ([expected.warning] if expected.warning else [])
    async_payload = async_requests[-1].payload
    if expected.provider_value is None:
        assert "reasoning" not in async_payload
    else:
        assert async_payload["reasoning"] == {"effort": expected.provider_value}


@pytest.mark.integration
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("model", ["mistral-large-4", "glm-5.3-flash"])
def test_none_falls_back_to_minimum_with_one_registry_warning(
    mistral_registry: RegistrySpec, monkeypatch: pytest.MonkeyPatch, model: str, method: str,
) -> None:
    response = _completed_response(with_reasoning=False)
    events = _reasoning_events(with_reasoning=False)
    adapter = MistralAdapter(api_key="synthetic-key", model=model, company="mistral")
    sync_transport = ReasoningTransport(response, events)
    adapter._sync_transport = sync_transport
    async_requests = _install_async(monkeypatch, response, events)
    completion: list[ChatResponse] = []
    reasoning: list[ReasoningEvent] = []
    assert adapter.model_spec is not None
    resolution = resolve_reasoning_level(adapter.model_spec, "none")
    assert resolution.provider_value is not None
    assert resolution.warning is not None

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _invoke(
            adapter, method, capture=False, completed=completion, reasoning=reasoning,
            reasoning_level="none",
        )

    assert [str(item.message) for item in caught] == [resolution.warning]
    requests = async_requests if method in {"achat", "astream_chat"} else sync_transport.requests
    assert requests[-1].payload["reasoning"] == {"effort": resolution.provider_value}
    assert reasoning == []


@pytest.mark.integration
def test_sync_and_async_use_identical_registry_effort_payloads(
    mistral_registry: RegistrySpec, monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _completed_response(with_reasoning=False)
    events = _reasoning_events(with_reasoning=False)
    adapter = MistralAdapter(api_key="synthetic-key", model=MODEL, company="mistral")
    sync_transport = ReasoningTransport(response, events)
    adapter._sync_transport = sync_transport
    async_requests = _install_async(monkeypatch, response, events)
    assert MistralAsyncClient._prepare_responses_payload_for_model is MistralSyncClient._prepare_responses_payload_for_model

    adapter.chat([{"role": "user", "content": "Answer."}], reasoning_level="medium")
    asyncio.run(adapter.achat([{"role": "user", "content": "Answer."}], reasoning_level="medium"))

    sync_payload = sync_transport.requests[0].payload
    async_payload = async_requests[0].payload
    assert sync_payload == async_payload
    assert sync_payload["reasoning"] == {"effort": "medium"}

    assert adapter.model_spec is not None
    integer_resolution = resolve_reasoning_level(adapter.model_spec, 1234)
    adapter.chat([{"role": "user", "content": "Answer."}], reasoning_level=1234)
    asyncio.run(adapter.achat([{"role": "user", "content": "Answer."}], reasoning_level=1234))
    assert sync_transport.requests[1].payload == async_requests[1].payload
    if integer_resolution.provider_value is None:
        assert "reasoning" not in sync_transport.requests[1].payload
    else:
        assert sync_transport.requests[1].payload["reasoning"] == {"effort": integer_resolution.provider_value}


@pytest.mark.integration
@pytest.mark.parametrize("method", METHODS)
def test_native_effort_and_opt_in_reasoning_stay_separate_from_text(
    mistral_registry: RegistrySpec, monkeypatch: pytest.MonkeyPatch, method: str,
) -> None:
    response = _completed_response()
    events = _reasoning_events()
    adapter = MistralAdapter(api_key="synthetic-key", model=MODEL, company="mistral")
    sync_transport = ReasoningTransport(response, events)
    adapter._sync_transport = sync_transport
    async_requests = _install_async(monkeypatch, response, events)
    completion: list[ChatResponse] = []
    callback_events: list[ReasoningEvent] = []
    prior_reasoning_text = "must not replay prior reasoning"
    previous = ChatResponse(
        response_id=OPAQUE_RESPONSE_ID,
        reasoning_events=[ReasoningEvent(
            text=prior_reasoning_text, kind="summary", index=0, elapsed_s=0.0, delta_s=0.0,
        )],
    )

    result, visible = _invoke(
        adapter, method, capture=True, completed=completion, reasoning=callback_events,
        previous_response=previous, reasoning_level="high",
    )

    assert result.content == VISIBLE_TEXT
    assert [text for text in visible] == ([VISIBLE_TEXT] if method in {"stream_chat", "astream_chat"} else [])
    expected_events = [("summary", REASONING_TEXT), ("content", "Provider reasoning content.")]
    assert [(event.kind, event.text) for event in result.reasoning_events] == expected_events
    assert [(event.kind, event.text) for event in callback_events] == (
        expected_events if method in {"stream_chat", "astream_chat"} else []
    )
    if method in {"stream_chat", "astream_chat"}:
        assert completion == [result]
    requests = async_requests if method in {"achat", "astream_chat"} else sync_transport.requests
    payload = requests[-1].payload
    assert payload["reasoning"] == {"effort": "high"}
    _assert_no_replay_controls(payload)
    assert REASONING_TEXT not in result.content
    assert prior_reasoning_text not in json.dumps(payload)


@pytest.mark.integration
@pytest.mark.parametrize("method", METHODS)
def test_capture_does_not_synthesize_absent_provider_reasoning(
    mistral_registry: RegistrySpec, monkeypatch: pytest.MonkeyPatch, method: str,
) -> None:
    response = _completed_response(with_reasoning=False)
    events = _reasoning_events(with_reasoning=False)
    adapter = MistralAdapter(api_key="synthetic-key", model=MODEL, company="mistral")
    sync_transport = ReasoningTransport(response, events)
    adapter._sync_transport = sync_transport
    _install_async(monkeypatch, response, events)
    completion: list[ChatResponse] = []
    callback_events: list[ReasoningEvent] = []

    result, visible = _invoke(adapter, method, capture=True, completed=completion, reasoning=callback_events)

    assert result.content == VISIBLE_TEXT
    assert visible == ([VISIBLE_TEXT] if method in {"stream_chat", "astream_chat"} else [])
    assert result.reasoning_events == []
    assert callback_events == []
    if method in {"stream_chat", "astream_chat"}:
        assert completion == [result]


@pytest.mark.integration
@pytest.mark.parametrize("method", METHODS)
def test_reasoning_capture_opt_out_and_missing_reasoning_do_not_synthesize(
    mistral_registry: RegistrySpec, monkeypatch: pytest.MonkeyPatch, method: str,
) -> None:
    response = _completed_response(with_reasoning=True)
    events = _reasoning_events(with_reasoning=True)
    adapter = MistralAdapter(api_key="synthetic-key", model=MODEL, company="mistral")
    sync_transport = ReasoningTransport(response, events)
    adapter._sync_transport = sync_transport
    async_requests = _install_async(monkeypatch, response, events)
    completion: list[ChatResponse] = []
    callback_events: list[ReasoningEvent] = []

    result, visible = _invoke(adapter, method, capture=False, completed=completion, reasoning=callback_events)

    assert result.content == VISIBLE_TEXT
    assert visible == ([VISIBLE_TEXT] if method in {"stream_chat", "astream_chat"} else [])
    assert result.reasoning_events == []
    assert callback_events == []
    if method in {"stream_chat", "astream_chat"}:
        assert completion == [result]
    requests = async_requests if method in {"achat", "astream_chat"} else sync_transport.requests
    _assert_no_replay_controls(requests[-1].payload)
