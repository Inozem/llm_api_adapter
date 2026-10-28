"""Contract tests for GLM Flash reasoning through the Mistral adapter."""

from __future__ import annotations

import asyncio
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

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

import llm_api_adapter.adapters.base_adapter as base_adapter_module
import llm_api_adapter.universal_adapter as universal_module
from llm_api_adapter.errors import LLMReasoningLevelError
from llm_api_adapter.llm_registry.llm_registry import RegistrySpec
from llm_api_adapter.llm_registry.reasoning import resolve_reasoning_level
from llm_api_adapter.llms.transports import JSONResponse, SSEEvent
from llm_api_adapter.models.responses.chat_response import ChatResponse
from llm_api_adapter.models.responses.reasoning_event import ReasoningEvent
from llm_api_adapter.service_provider_registry import ServiceProviderRegistry
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter
from llm_api_adapter_mistral.plugin import PLUGIN


MODEL = "glm-5.3-flash"
USER_MESSAGES = [{"role": "user", "content": "Answer briefly."}]
VISIBLE_TEXT = "Visible answer."
THINKING_TEXTS = ["First thought.", "Second thought."]


class FakeSyncTransport:
    def __init__(self, response: dict, events=()) -> None:
        self.response = response
        self.events = list(events)
        self.requests = []
        self.stream_requests = []

    def post_json(self, request, *, http_error_handler=None):
        self.requests.append(request)
        return JSONResponse(self.response)

    def post_sse(
        self,
        request,
        *,
        http_error_handler=None,
        stream_error_handler=None,
    ):
        self.stream_requests.append(request)
        return iter(self.events)


@pytest.fixture
def mistral_runtime(monkeypatch):
    model_registry = RegistrySpec()
    assert PLUGIN.model_metadata is not None
    assert model_registry.register_organization_metadata(PLUGIN.model_metadata) is True
    service_provider_registry = ServiceProviderRegistry()
    PLUGIN.register(service_provider_registry)
    monkeypatch.setattr(universal_module, "LLM_REGISTRY", model_registry)
    monkeypatch.setattr(
        universal_module,
        "SERVICE_PROVIDER_REGISTRY",
        service_provider_registry,
    )
    monkeypatch.setattr(base_adapter_module, "LLM_REGISTRY", model_registry)
    return model_registry


def _new_adapter():
    adapter = UniversalLLMAPIAdapter(
        organization="zai",
        service_provider="mistral",
        model=MODEL,
        api_key="mistral-test-key",
    )
    assert adapter.organization == "zai"
    assert adapter.service_provider == "mistral"
    assert adapter.model == MODEL
    return adapter


def _thinking_content():
    return [
        {
            "type": "thinking",
            "thinking": [
                {"type": "text", "text": THINKING_TEXTS[0]},
                {"type": "text", "text": THINKING_TEXTS[1]},
            ],
        },
        {"type": "text", "text": VISIBLE_TEXT},
    ]


def _chat_response(content):
    return {
        "id": "cmpl-glm-flash-chat",
        "model": "zai-glm-5-3-flash",
        "choices": [{"message": {"role": "assistant", "content": content}}],
    }


def _stream_event(content):
    return SSEEvent(
        event=None,
        data={
            "id": "cmpl-glm-flash-stream",
            "model": "zai-glm-5-3-flash",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": content},
                    "finish_reason": "stop",
                }
            ],
        },
    )


def _configure_async(adapter, response, event, transport):
    calls = []

    async def fake_async_request(*, endpoint, payload, **kwargs):
        calls.append(("chat", endpoint, payload))
        transport.requests.append(SimpleNamespace(payload=payload))
        return response

    def fake_async_stream_request(*, endpoint, payload, **kwargs):
        calls.append(("stream", endpoint, payload))
        transport.stream_requests.append(SimpleNamespace(payload=payload))

        async def events():
            yield event

        return events()

    adapter.adapter._async_client.chat = fake_async_request
    adapter.adapter._async_client.stream_chat = fake_async_stream_request
    return calls


def _run_mode(adapter, api_kind, monkeypatch, response, content, **kwargs):
    event = _stream_event(content)
    transport = FakeSyncTransport(response, events=[event])
    if api_kind.startswith("sync"):
        adapter.adapter._sync_transport = transport
    else:
        _configure_async(adapter, response, event, transport)

    reasoning = []
    completed = []
    if api_kind == "sync-chat":
        result = adapter.chat(USER_MESSAGES, **kwargs)
        return result, reasoning, completed, transport
    if api_kind == "async-chat":
        result = asyncio.run(adapter.achat(USER_MESSAGES, **kwargs))
        return result, reasoning, completed, transport
    if api_kind == "sync-stream":
        streamed = list(
            adapter.stream_chat(
                USER_MESSAGES,
                on_reasoning=reasoning.append,
                on_done=completed.append,
                **kwargs,
            )
        )
    else:
        async def collect():
            async def on_reasoning(event):
                reasoning.append(event)

            async def on_done(result):
                completed.append(result)

            return [
                text
                async for text in adapter.astream_chat(
                    USER_MESSAGES,
                    on_reasoning=on_reasoning,
                    on_done=on_done,
                    **kwargs,
                )
            ]

        streamed = asyncio.run(collect())
    assert completed
    if completed[0].content is None:
        assert streamed == []
    else:
        assert "".join(streamed) == completed[0].content
    return completed[0], reasoning, completed, transport


@pytest.mark.integration
@pytest.mark.parametrize(
    ("api_kind", "streaming"),
    [
        ("sync-chat", False),
        ("async-chat", False),
        ("sync-stream", True),
        ("async-stream", True),
    ],
)
@pytest.mark.parametrize("capture_reasoning", [False, True])
@pytest.mark.parametrize(
    ("content", "expected_visible", "expected_thinking"),
    [
        ("Plain string answer.", "Plain string answer.", []),
        (_thinking_content(), VISIBLE_TEXT, THINKING_TEXTS),
    ],
    ids=["plain-string", "thinking-and-text-list"],
)
def test_glm_flash_all_modes_keep_visible_text_separate_from_opt_in_reasoning(
    mistral_runtime,
    monkeypatch,
    api_kind,
    streaming,
    capture_reasoning,
    content,
    expected_visible,
    expected_thinking,
):
    adapter = _new_adapter()
    response, reasoning, completed, transport = _run_mode(
        adapter,
        api_kind,
        monkeypatch,
        _chat_response(content),
        content,
        capture_reasoning=capture_reasoning,
        reasoning_level="low",
    )

    expected_events = expected_thinking if capture_reasoning else []
    assert response.content == expected_visible
    assert [event.text for event in response.reasoning_events] == expected_events
    if streaming:
        assert [event.text for event in reasoning] == expected_events
        assert len(completed) == 1
        assert completed[0] is response
    if expected_visible is not None:
        assert all(text not in expected_visible for text in expected_events)
    request = transport.stream_requests[0] if streaming else transport.requests[0]
    assert request.payload["reasoning_effort"] == "low"


@pytest.mark.integration
@pytest.mark.parametrize(
    ("api_kind", "streaming"),
    [
        ("sync-chat", False),
        ("async-chat", False),
        ("sync-stream", True),
        ("async-stream", True),
    ],
)
@pytest.mark.parametrize("capture_reasoning", [False, True])
@pytest.mark.parametrize("content", ["Visible only.", None], ids=["plain-text", "absent"])
def test_glm_flash_missing_thinking_never_synthesizes_reasoning(
    mistral_runtime,
    monkeypatch,
    api_kind,
    streaming,
    capture_reasoning,
    content,
):
    adapter = _new_adapter()
    response, reasoning, _, _ = _run_mode(
        adapter,
        api_kind,
        monkeypatch,
        _chat_response(content),
        content,
        capture_reasoning=capture_reasoning,
    )
    assert response.reasoning_events == []
    if streaming:
        assert reasoning == []
    assert response.content == content


@pytest.mark.integration
@pytest.mark.parametrize(
    "reasoning_level",
    ["low", "high", "max", "none", "minimal", "medium", "very_high"],
    ids=["low", "high", "max", "none", "minimal", "medium", "very-high"],
)
@pytest.mark.parametrize("api_kind", ["sync-chat", "async-chat"])
def test_glm_flash_reasoning_effort_uses_shared_resolver(
    mistral_runtime,
    monkeypatch,
    reasoning_level,
    api_kind,
):
    adapter = _new_adapter()
    expected = resolve_reasoning_level(adapter.adapter.model_spec, reasoning_level)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, _, _, transport = _run_mode(
            adapter,
            api_kind,
            monkeypatch,
            _chat_response("Done."),
            "Done.",
            reasoning_level=reasoning_level,
        )
    assert [str(item.message) for item in caught] == (
        [expected.warning] if expected.warning else []
    )
    assert transport.requests[0].payload["reasoning_effort"] == expected.provider_value


@pytest.mark.integration
@pytest.mark.parametrize("api_kind", ["sync-chat", "async-chat", "sync-stream", "async-stream"])
def test_glm_flash_omitted_reasoning_effort_defaults_to_low(
    mistral_runtime,
    monkeypatch,
    api_kind,
):
    adapter = _new_adapter()
    low = resolve_reasoning_level(adapter.adapter.model_spec, "low")
    _, _, _, transport = _run_mode(
        adapter,
        api_kind,
        monkeypatch,
        _chat_response("Done."),
        "Done.",
    )
    requests = transport.stream_requests if "stream" in api_kind else transport.requests
    assert requests[0].payload["reasoning_effort"] == low.provider_value == "low"


@pytest.mark.integration
@pytest.mark.parametrize("api_kind", ["sync-chat", "async-chat", "sync-stream", "async-stream"])
@pytest.mark.parametrize("reasoning_level", ["unrecognized", True, -1])
def test_glm_flash_invalid_reasoning_controls_fail_before_provider_request(
    mistral_runtime,
    monkeypatch,
    api_kind,
    reasoning_level,
):
    adapter = _new_adapter()
    response = _chat_response("Done.")
    event = _stream_event("Done.")
    transport = FakeSyncTransport(response, events=[event])
    async_calls = _configure_async(adapter, response, event, transport)
    if api_kind.startswith("sync"):
        adapter.adapter._sync_transport = transport
    with pytest.raises(LLMReasoningLevelError):
        _run_mode(
            adapter,
            api_kind,
            monkeypatch,
            response,
            "Done.",
            reasoning_level=reasoning_level,
        )
    assert transport.requests == []
    assert transport.stream_requests == []
    assert async_calls == []


@pytest.mark.integration
@pytest.mark.parametrize("api_kind", ["sync-chat", "async-chat", "sync-stream", "async-stream"])
def test_glm_flash_ignores_previous_response_and_does_not_replay_reasoning(
    mistral_runtime,
    monkeypatch,
    api_kind,
):
    adapter = _new_adapter()
    content = _thinking_content()
    response = _chat_response(content)
    event = _stream_event(content)
    transport = FakeSyncTransport(response, events=[event])
    async_calls = _configure_async(adapter, response, event, transport)
    if api_kind.startswith("sync"):
        adapter.adapter._sync_transport = transport
    previous = ChatResponse(
        response_id="cmpl-previous-glm-flash",
        content="Earlier visible answer.",
        reasoning_events=[
            ReasoningEvent(
                text="Earlier private thought.",
                kind="content",
                index=0,
                elapsed_s=0.0,
                delta_s=0.0,
            )
        ],
    )
    kwargs = {"previous_response": previous, "capture_reasoning": True}

    if api_kind == "sync-chat":
        result = adapter.chat(USER_MESSAGES, **kwargs)
    elif api_kind == "async-chat":
        result = asyncio.run(adapter.achat(USER_MESSAGES, **kwargs))
    elif api_kind == "sync-stream":
        completed = []
        list(adapter.stream_chat(USER_MESSAGES, on_done=completed.append, **kwargs))
        result = completed[0]
    else:
        async def collect():
            completed = []

            async def on_done(response):
                completed.append(response)

            async for _ in adapter.astream_chat(
                USER_MESSAGES,
                on_done=on_done,
                **kwargs,
            ):
                pass
            return completed[0]

        result = asyncio.run(collect())

    assert result.content == VISIBLE_TEXT
    assert [event.text for event in result.reasoning_events] == THINKING_TEXTS
    if api_kind.startswith("sync"):
        request = transport.stream_requests[0] if "stream" in api_kind else transport.requests[0]
        payload = request.payload
    else:
        assert len(async_calls) == 1
        payload = async_calls[0][2]
    assert payload["messages"] == USER_MESSAGES
    serialized = repr(payload)
    assert "cmpl-previous-glm-flash" not in serialized
    assert "Earlier private thought." not in serialized
    assert "previous_response" not in serialized
    assert "reasoning_events" not in serialized
