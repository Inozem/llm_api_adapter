"""Lazy Kimi stream entry and native reasoning compatibility checks."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
import sys
import warnings
from typing import Any
from unittest.mock import Mock

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[2]
for source in (PACKAGE_ROOT / "src", REPOSITORY_ROOT / "src", REPOSITORY_ROOT):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

import llm_api_adapter.adapters.base_adapter as base_adapter_module
import llm_api_adapter.universal_adapter as universal_module
from llm_api_adapter.llm_registry.llm_registry import RegistrySpec
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
from llm_api_adapter.organization_registry import (
    ORGANIZATION_PLUGIN_ENTRY_POINT_GROUP,
    OrganizationPluginDiscovery,
    OrganizationPlugin,
)
from llm_api_adapter.service_provider_registry import ServiceProviderRegistry
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter
from llm_api_adapter_kimi.plugin import PLUGIN


class _RegisteredKimiEntryPoint:
    name = "kimi"
    value = "llm_api_adapter_kimi.plugin:PLUGIN"

    def load(self) -> OrganizationPlugin:
        return PLUGIN


class _ControlledTransport(SyncTransport):
    def __init__(self, events: list[SSEEvent] | None = None) -> None:
        self.events = list(events or [])
        self.json_requests: list[TransportRequest] = []
        self.stream_requests: list[TransportRequest] = []
        self.async_payloads: list[dict[str, Any]] = []
        self.async_json_payloads: list[dict[str, Any]] = []
        self.opened: list[str] = []

    def post_json(
        self,
        request: TransportRequest,
        *,
        http_error_handler: HTTPErrorHandler | None = None,
    ) -> JSONResponse:
        self.opened.append("json")
        self.json_requests.append(request)
        return JSONResponse(_completed_payload(request.payload["model"]))

    def post_multipart(
        self,
        request: TransportRequest,
        form: MultipartForm,
        *,
        http_error_handler: HTTPErrorHandler | None = None,
    ) -> JSONResponse:
        _ = (request, form, http_error_handler)
        raise AssertionError("Kimi does not use multipart requests")

    def post_sse(
        self,
        request: TransportRequest,
        *,
        http_error_handler: HTTPErrorHandler | None = None,
        stream_error_handler: StreamErrorHandler | None = None,
    ) -> Iterator[SSEEvent]:
        self.opened.append("sse")
        self.stream_requests.append(request)
        return iter(self.events)


def _registered_adapter(monkeypatch: pytest.MonkeyPatch, model: str) -> UniversalLLMAPIAdapter:
    """Use the real Kimi entry point through the public facade with isolated registries."""
    service_registry = ServiceProviderRegistry()
    model_registry = RegistrySpec()
    assert PLUGIN.model_metadata is not None
    model_registry.register_organization_metadata(PLUGIN.model_metadata)

    def entry_points(*, group: str) -> list[_RegisteredKimiEntryPoint]:
        assert group == ORGANIZATION_PLUGIN_ENTRY_POINT_GROUP
        return [_RegisteredKimiEntryPoint()]

    monkeypatch.setattr("llm_api_adapter.organization_registry.entry_points", entry_points)
    monkeypatch.setattr(universal_module, "SERVICE_PROVIDER_REGISTRY", service_registry)
    monkeypatch.setattr(universal_module, "ORGANIZATION_PLUGIN_DISCOVERY", OrganizationPluginDiscovery())
    monkeypatch.setattr(universal_module, "LLM_REGISTRY", model_registry)
    monkeypatch.setattr(base_adapter_module, "LLM_REGISTRY", model_registry)
    return UniversalLLMAPIAdapter(
        organization="kimi",
        model=model,
        api_key="kimi-test-key",
    )


def _install_async_stream(
    monkeypatch: pytest.MonkeyPatch,
    transport: _ControlledTransport,
) -> None:
    from llm_api_adapter_kimi.clients import async_client as async_client_module

    def async_request_guard(
        url: str,
        *,
        headers: dict[str, str] | None = None,
        payload: dict[str, Any] | None = None,
        timeout: float | None = None,
        http_error_handler: HTTPErrorHandler | None = None,
    ) -> dict[str, Any]:
        _ = (url, headers, payload, timeout, http_error_handler)
        transport.opened.append("unexpected-async-json")
        raise AssertionError("stream-only Kimi test must not dispatch async JSON")

    def async_stream_request(
        url: str,
        *,
        headers: dict[str, str] | None = None,
        payload: dict[str, Any] | None = None,
        timeout: float | None = None,
        http_error_handler: HTTPErrorHandler | None = None,
        stream_error_handler: StreamErrorHandler | None = None,
    ) -> AsyncIterator[SSEEvent]:
        _ = (url, headers, timeout, http_error_handler, stream_error_handler)
        transport.opened.append("async-sse")
        transport.async_payloads.append(dict(payload or {}))

        async def events() -> AsyncIterator[SSEEvent]:
            for event in transport.events:
                yield event

        return events()

    monkeypatch.setattr(async_client_module, "async_request", async_request_guard)
    monkeypatch.setattr(async_client_module, "async_stream_request", async_stream_request)


def _completed_payload(model: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-kimi-reasoning",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Visible reply",
                    "reasoning_content": "Completed reason.",
                },
                "finish_reason": "stop",
            },
        ],
    }


def _install_async_json(
    monkeypatch: pytest.MonkeyPatch,
    transport: _ControlledTransport,
) -> None:
    from llm_api_adapter_kimi.clients import async_client as async_client_module

    async def async_request(
        url: str,
        *,
        headers: dict[str, str] | None = None,
        payload: dict[str, Any] | None = None,
        timeout: float | None = None,
        http_error_handler: HTTPErrorHandler | None = None,
    ) -> dict[str, Any]:
        _ = (url, headers, timeout, http_error_handler)
        request_payload = dict(payload or {})
        transport.opened.append("async-json")
        transport.async_json_payloads.append(request_payload)
        return _completed_payload(request_payload["model"])

    monkeypatch.setattr(async_client_module, "async_request", async_request)


def _reasoning_events() -> list[SSEEvent]:
    return [
        SSEEvent(
            event=None,
            data={
                "id": "stream-kimi",
                "created": 1,
                "model": "kimi-k3",
                "choices": [
                    {
                        "index": 1,
                        "delta": {"reasoning_content": "other choice"},
                    },
                    {
                        "index": 0,
                        "delta": {
                            "reasoning_content": "Think ",
                            "content": "Visible ",
                        },
                    },
                ],
            },
        ),
        SSEEvent(
            event=None,
            data={
                "id": "stream-kimi",
                "created": 1,
                "model": "kimi-k3",
                "choices": [
                    {"index": 0, "delta": {"reasoning_content": "", "content": "reply"}},
                ],
            },
        ),
        SSEEvent(
            event=None,
            data={
                "id": "stream-kimi",
                "created": 1,
                "model": "kimi-k3",
                "choices": [
                    {"index": 0, "delta": {"reasoning_content": 42}},
                ],
            },
        ),
        SSEEvent(event=None, data="[DONE]", done=True),
    ]


@pytest.fixture
def lazy_kimi(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[UniversalLLMAPIAdapter, _ControlledTransport]:
    adapter = _registered_adapter(monkeypatch, "kimi-k3")
    transport = _ControlledTransport(_reasoning_events())
    adapter.adapter._sync_transport = transport
    _install_async_stream(monkeypatch, transport)
    return adapter, transport


@pytest.mark.unit
@pytest.mark.parametrize("path", ["sync", "async"])
def test_stream_entry_defers_preparation_validation_reasoning_and_opening(
    lazy_kimi: tuple[UniversalLLMAPIAdapter, _ControlledTransport],
    monkeypatch: pytest.MonkeyPatch,
    path: str,
) -> None:
    adapter, transport = lazy_kimi
    native = adapter.adapter
    prepare = Mock(wraps=native._prepare_chat_request)
    sampling = Mock(wraps=native._validate_sampling_parameters)
    resolve = Mock(wraps=native._resolve_reasoning_level)
    monkeypatch.setattr(native, "_prepare_chat_request", prepare)
    monkeypatch.setattr(native, "_validate_sampling_parameters", sampling)
    monkeypatch.setattr(native, "_resolve_reasoning_level", resolve)
    callbacks: list[object] = []
    warnings_seen: list[warnings.WarningMessage] = []

    def on_reasoning(event: ReasoningEvent) -> None:
        callbacks.append(event)

    def on_delta(text: str) -> None:
        callbacks.append(text)

    def on_done(response: ChatResponse) -> None:
        callbacks.append(response)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        if path == "sync":
            iterator = adapter.stream_chat(
                [{"role": "user", "content": "Think"}],
                reasoning_level="high",
                capture_reasoning=True,
                on_reasoning=on_reasoning,
                on_delta=on_delta,
                on_done=on_done,
            )
        else:
            iterator = adapter.astream_chat(
                [{"role": "user", "content": "Think"}],
                reasoning_level="high",
                capture_reasoning=True,
                on_reasoning=on_reasoning,
                on_delta=on_delta,
                on_done=on_done,
            )
        assert prepare.call_count == sampling.call_count == resolve.call_count == 0
        assert transport.opened == []
        assert callbacks == []
        assert captured == []

        if path == "sync":
            assert list(iterator) == ["Visible ", "reply"]
        else:
            async def consume() -> list[str]:
                return [text async for text in iterator]

            assert asyncio.run(consume()) == ["Visible ", "reply"]
        warnings_seen = list(captured)

    assert prepare.call_count == 1
    assert sampling.call_count == 1
    assert resolve.call_count == 1
    assert transport.opened == (["sse"] if path == "sync" else ["async-sse"])
    assert warnings_seen == []


@pytest.mark.unit
@pytest.mark.parametrize("path", ["sync", "async"])
def test_invalid_stream_inputs_fail_on_consumption_before_open_or_callbacks(
    lazy_kimi: tuple[UniversalLLMAPIAdapter, _ControlledTransport],
    path: str,
) -> None:
    adapter, transport = lazy_kimi
    callbacks: list[object] = []
    if path == "sync":
        iterator = adapter.stream_chat(
            [{"role": "user", "content": "invalid controls"}],
            temperature=-1,
            parallel_tool_calls=False,
            on_reasoning=callbacks.append,
            on_done=callbacks.append,
        )
    else:
        iterator = adapter.astream_chat(
            [{"role": "user", "content": "invalid controls"}],
            temperature=-1,
            parallel_tool_calls=False,
            on_reasoning=callbacks.append,
            on_done=callbacks.append,
        )
    assert transport.opened == []
    assert callbacks == []

    if path == "sync":
        with pytest.raises(ValueError, match="temperature"):
            next(iterator)
    else:
        async def consume_one() -> None:
            await anext(iterator)

        with pytest.raises(ValueError, match="temperature"):
            asyncio.run(consume_one())
    assert transport.opened == []
    assert transport.stream_requests == []
    assert callbacks == []


@pytest.mark.unit
@pytest.mark.parametrize("path", ["sync", "async"])
@pytest.mark.parametrize(
    ("model", "reasoning_level", "expected", "warning_text"),
    [
        ("kimi-k3", None, {}, None),
        ("kimi-k3", "high", {"reasoning_effort": "high"}, None),
        ("kimi-k3", "none", {"reasoning_effort": "low"}, "cannot disable reasoning"),
        ("kimi-k2.6", "none", {"thinking": {"type": "disabled"}}, None),
        ("kimi-k2.6", "enabled", {}, None),
        ("kimi-k2.6", "high", {}, None),
    ],
)
def test_reasoning_resolution_and_warnings_happen_on_consumption(
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    model: str,
    reasoning_level: str | None,
    expected: dict[str, Any],
    warning_text: str | None,
) -> None:
    adapter = _registered_adapter(monkeypatch, model)
    transport = _ControlledTransport(_reasoning_events())
    adapter.adapter._sync_transport = transport
    _install_async_stream(monkeypatch, transport)
    native = adapter.adapter
    prepare = Mock(wraps=native._prepare_chat_request)
    resolve = Mock(wraps=native._resolve_reasoning_level)
    monkeypatch.setattr(native, "_prepare_chat_request", prepare)
    monkeypatch.setattr(native, "_resolve_reasoning_level", resolve)
    kwargs = {} if reasoning_level is None else {"reasoning_level": reasoning_level}
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        if path == "sync":
            iterator = adapter.stream_chat([{"role": "user", "content": "Hi"}], **kwargs)
        else:
            iterator = adapter.astream_chat([{"role": "user", "content": "Hi"}], **kwargs)
        assert prepare.call_count == resolve.call_count == 0
        assert transport.opened == []
        assert captured == []
        if path == "sync":
            list(iterator)
        else:
            async def consume() -> None:
                async for _ in iterator:
                    pass

            asyncio.run(consume())

    assert prepare.call_count == 1
    assert resolve.call_count == (0 if reasoning_level is None else 1)
    assert transport.opened == (["sse"] if path == "sync" else ["async-sse"])
    payload = (
        transport.stream_requests[0].payload
        if path == "sync"
        else transport.async_payloads[0]
    )
    assert {key: payload[key] for key in ("reasoning_effort", "thinking") if key in payload} == expected
    warning_messages = [str(item.message) for item in captured]
    if warning_text is None:
        assert warning_messages == []
    else:
        assert any(warning_text in message for message in warning_messages)


@pytest.mark.unit
@pytest.mark.parametrize("path", ["sync", "async"])
def test_reasoning_events_are_separate_and_capture_is_opt_in(
    lazy_kimi: tuple[UniversalLLMAPIAdapter, _ControlledTransport],
    path: str,
) -> None:
    adapter, transport = lazy_kimi
    order: list[tuple[str, Any]] = []
    completed: list[ChatResponse] = []

    def on_reasoning(event: ReasoningEvent) -> None:
        order.append(("reasoning", event))

    def on_delta(text: str) -> None:
        order.append(("delta", text))

    def on_done(response: ChatResponse) -> None:
        completed.append(response)
        order.append(("done", response))

    if path == "sync":
        iterator = adapter.stream_chat(
            [{"role": "user", "content": "Think"}],
            capture_reasoning=True,
            on_reasoning=on_reasoning,
            on_delta=on_delta,
            on_done=on_done,
        )
        output = list(iterator)
    else:
        async def async_reasoning(event: ReasoningEvent) -> None:
            await asyncio.sleep(0)
            order.append(("reasoning", event))

        async def async_done(response: ChatResponse) -> None:
            await asyncio.sleep(0)
            completed.append(response)
            order.append(("done", response))

        async def collect() -> list[str]:
            return [
                text
                async for text in adapter.astream_chat(
                    [{"role": "user", "content": "Think"}],
                    capture_reasoning=True,
                    on_reasoning=async_reasoning,
                    on_delta=on_delta,
                    on_done=async_done,
                )
            ]

        output = asyncio.run(collect())

    assert output == ["Visible ", "reply"]
    assert [kind for kind, _ in order] == ["reasoning", "delta", "delta", "done"]
    reason_objects = [value for kind, value in order if kind == "reasoning"]
    assert len(reason_objects) == 1
    assert reason_objects[0].text == "Think "
    response = completed[0]
    assert response.reasoning_events == [reason_objects[0]]
    assert response.reasoning_events[0] is reason_objects[0]
    assert order[-1][1] is response
    assert "Think" not in "".join(output)
    assert [value for kind, value in order if kind == "delta"] == output
    assert transport.opened == (["sse"] if path == "sync" else ["async-sse"])

    transport.events = _reasoning_events()
    callbacks: list[ReasoningEvent] = []
    completed.clear()
    if path == "sync":
        output_without_capture = list(
            adapter.stream_chat(
                [{"role": "user", "content": "No capture"}],
                on_reasoning=callbacks.append,
                on_done=completed.append,
            ),
        )
    else:
        async def collect_without_capture() -> list[str]:
            return [
                text
                async for text in adapter.astream_chat(
                    [{"role": "user", "content": "No capture"}],
                    on_reasoning=callbacks.append,
                    on_done=completed.append,
                )
            ]

        output_without_capture = asyncio.run(collect_without_capture())
    assert output_without_capture == output
    assert callbacks == []
    assert completed[0].reasoning_events == []


@pytest.mark.unit
@pytest.mark.parametrize("path", ["sync", "async"])
@pytest.mark.parametrize("capture_reasoning", [False, True])
def test_completed_reasoning_capture_and_history_continuation_policy(
    lazy_kimi: tuple[UniversalLLMAPIAdapter, _ControlledTransport],
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    capture_reasoning: bool,
) -> None:
    adapter, transport = lazy_kimi
    if path == "async":
        _install_async_json(monkeypatch, transport)
    previous = ChatResponse(
        response_id="ignored-previous-response",
        reasoning_events=[
            ReasoningEvent(
                text="prior response reasoning",
                kind="content",
                index=0,
                elapsed_s=0.0,
                delta_s=0.0,
            ),
        ],
    )
    history = [
        {
            "role": "assistant",
            "content": "Earlier visible answer",
            "reasoning_content": "do not replay this",
        },
        {"role": "user", "content": "Continue"},
    ]
    if path == "sync":
        response = adapter.chat(
            history,
            previous_response=previous,
            capture_reasoning=capture_reasoning,
        )
        payload = transport.json_requests[0].payload
    else:
        async def call_achat() -> ChatResponse:
            return await adapter.achat(
                history,
                previous_response=previous,
                capture_reasoning=capture_reasoning,
            )

        response = asyncio.run(call_achat())
        payload = transport.async_json_payloads[0]

    assert response.content == "Visible reply"
    assert [event.text for event in response.reasoning_events] == (
        ["Completed reason."] if capture_reasoning else []
    )
    assert payload["messages"] == [
        {"role": "assistant", "content": "Earlier visible answer"},
        {"role": "user", "content": "Continue"},
    ]
    assert "previous_response" not in payload
    assert "previous_response_id" not in payload
    assert transport.opened == (["json"] if path == "sync" else ["async-json"])
