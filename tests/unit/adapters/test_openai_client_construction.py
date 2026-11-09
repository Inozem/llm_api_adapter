from unittest.mock import AsyncMock, patch
import json
from dataclasses import is_dataclass

import pytest

from src.llm_api_adapter.adapters.openai import adapter as openai_adapter_module
from src.llm_api_adapter.adapters.openai.adapter import OpenAIAdapter
from src.llm_api_adapter.adapters.openai_compatible.adapter import (
    OpenAICompatibleAdapter,
)
from src.llm_api_adapter.adapters.openai_compatible.common.adapter import (
    _OpenAIExecutionMixin,
)
from src.llm_api_adapter.adapters.openai.payloads import (
    _OpenAIPayloadMixin,
    _apply_cache_usage as native_apply_cache_usage,
    _cache_token_counts as native_cache_token_counts,
    _token_count as native_token_count,
)
from src.llm_api_adapter.adapters.openai.streaming import _OpenAIStreamingMixin
from src.llm_api_adapter.adapters.openai_compatible.chat_completions.payloads import (
    _ChatCompletionsPayloadMixin,
)
from src.llm_api_adapter.adapters.openai_compatible.chat_completions.streaming import (
    _ChatCompletionsStreamingMixin,
)
from src.llm_api_adapter.adapters.openai_compatible.responses.payloads import (
    _ResponsesPayloadMixin,
)
from src.llm_api_adapter.adapters.openai_compatible.responses.streaming import (
    _ResponsesStreamState,
    _ResponsesStreamingMixin,
)
from src.llm_api_adapter.adapters.openai_compatible.common.usage import (
    _apply_cache_usage,
    _cache_token_counts,
    _token_count,
)
from src.llm_api_adapter.adapters.base_adapter import LLMAdapterBase
from src.llm_api_adapter.adapters.openai_adapter import (
    OpenAIAsyncClient,
    OpenAISyncClient,
)
from src.llm_api_adapter.llms.streaming import SSEEvent
from src.llm_api_adapter.models.messages.chat_message import UserMessage
from tests.fixtures.openai_compatible import text_response, text_sse_lines


def _response(model, content="complete"):
    if model == "gpt-5":
        return {
            "model": model,
            "output": [{
                "type": "message",
                "content": [{"type": "output_text", "text": content}],
            }],
        }
    return text_response("complete", model=model)


def _events(model):
    if model == "gpt-5":
        return iter([
            SSEEvent(
                event="response.output_text.delta",
                data={"type": "response.output_text.delta", "delta": "stream"},
            ),
            SSEEvent(
                event="response.completed",
                data={"type": "response.completed", "response": {
                    **_response(model, content="stream"), "status": "completed",
                }},
            ),
        ])
    events = (
        json.loads(line[6:])
        for line in text_sse_lines("stream")
        if line.startswith("data: {")
    )
    return (SSEEvent(event=None, data=event) for event in events)


class _SyncClient:
    def __init__(self, model):
        self.model = model
        self.responses_api = model == "gpt-5"
        self.completed_routes = []
        self.streamed_routes = []

    def _should_use_responses_api(self, model):
        return self.responses_api

    def complete(self, *, timeout=None, **params):
        self.completed_routes.append("responses" if self.responses_api else "chat")
        return _response(self.model)

    def stream(self, *, timeout=None, **params):
        self.streamed_routes.append("responses" if self.responses_api else "chat")
        return _events(self.model)


class _AsyncClient(_SyncClient):
    async def complete(self, *, timeout=None, **params):
        self.completed_routes.append("responses" if self.responses_api else "chat")
        return _response(self.model)

    def stream(self, *, timeout=None, **params):
        self.streamed_routes.append("responses" if self.responses_api else "chat")

        async def events():
            for event in _events(self.model):
                yield event

        return events()


class _HookedAdapter(OpenAIAdapter):
    def __init__(self, *, api_key, model, transport="requests"):
        super().__init__(api_key=api_key, model=model, transport=transport)
        self.sync_creations = []
        self.async_creations = []
        self.sync_client = _SyncClient(self.model)
        self.async_client = _AsyncClient(self.model)

    def _create_sync_client(self):
        self.sync_creations.append(True)
        return self.sync_client

    def _create_async_client(self):
        self.async_creations.append(True)
        return self.async_client


@pytest.fixture(autouse=True)
def reject_default_client_use(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError(
            "construction hook bypassed; canonical client must not run"
        )

    monkeypatch.setattr(OpenAISyncClient, "complete", fail)
    monkeypatch.setattr(OpenAISyncClient, "stream", fail)
    monkeypatch.setattr(OpenAIAsyncClient, "complete", AsyncMock(side_effect=fail))
    monkeypatch.setattr(OpenAIAsyncClient, "stream", fail)


@pytest.mark.unit
def test_sync_chat_uses_overridden_construction_hook_for_legacy_route():
    adapter = _HookedAdapter(api_key="test-key", model="gpt-4o")

    response = adapter.chat([UserMessage("hi")])

    assert response.content == "complete"
    assert adapter.sync_creations == [True]
    assert adapter.sync_client.completed_routes == ["chat"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_chat_uses_overridden_construction_hook_for_responses_route():
    adapter = _HookedAdapter(api_key="test-key", model="gpt-5")

    response = await adapter.achat([UserMessage("hi")])

    assert response.content == "complete"
    assert adapter.async_creations == [True]
    assert adapter.async_client.completed_routes == ["responses"]


@pytest.mark.unit
def test_sync_stream_uses_overridden_construction_hook_for_responses_route():
    adapter = _HookedAdapter(api_key="test-key", model="gpt-5")
    completed = []

    chunks = list(
        adapter.stream_chat(
            [UserMessage("hi")],
            on_done=lambda response: completed.append(response.content),
        )
    )

    assert chunks == ["stream"]
    assert completed == ["stream"]
    assert adapter.sync_creations == [True]
    assert adapter.sync_client.streamed_routes == ["responses"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_stream_uses_overridden_construction_hook_for_legacy_route():
    adapter = _HookedAdapter(api_key="test-key", model="gpt-4o")
    chunks = []

    async for chunk in adapter.astream_chat([UserMessage("hi")]):
        chunks.append(chunk)

    assert chunks == ["stream"]
    assert adapter.async_creations == [True]
    assert adapter.async_client.streamed_routes == ["chat"]


@pytest.mark.unit
@pytest.mark.parametrize("transport", ["requests", "httpx"])
def test_default_sync_hook_preserves_client_defaults_and_transport(transport):
    adapter = OpenAIAdapter(api_key="test-key", model="gpt-4o", transport=transport)

    client = adapter._create_sync_client()

    assert type(client) is OpenAISyncClient
    assert client.api_key == "test-key"
    assert client.endpoint == "https://api.openai.com/v1"
    assert client.transport == transport


@pytest.mark.unit
def test_default_async_hook_preserves_official_client_defaults():
    adapter = OpenAIAdapter(api_key="test-key", model="gpt-4o", transport="httpx")

    client = adapter._create_async_client()

    assert type(client) is OpenAIAsyncClient
    assert client.api_key == "test-key"
    assert client.endpoint == "https://api.openai.com/v1"
    assert client.transport == "requests"


@pytest.mark.unit
def test_default_hooks_forward_only_the_existing_constructor_keywords():
    adapter = OpenAIAdapter(api_key="test-key", model="gpt-4o", transport="httpx")

    with (
        patch.object(openai_adapter_module, "OpenAISyncClient") as sync_constructor,
        patch.object(openai_adapter_module, "OpenAIAsyncClient") as async_constructor,
    ):
        adapter._create_sync_client()
        adapter._create_async_client()

    sync_constructor.assert_called_once_with(api_key="test-key", transport="httpx")
    async_constructor.assert_called_once_with(api_key="test-key")


@pytest.mark.unit
def test_native_and_compatible_adapters_share_neutral_execution_owner():
    assert is_dataclass(OpenAIAdapter)
    assert is_dataclass(OpenAICompatibleAdapter)
    assert issubclass(OpenAIAdapter, LLMAdapterBase)
    assert issubclass(OpenAICompatibleAdapter, LLMAdapterBase)
    assert not issubclass(OpenAICompatibleAdapter, OpenAIAdapter)

    for name in ("chat", "achat", "stream_chat", "astream_chat", "_astream_chat"):
        owner_method = getattr(_OpenAIExecutionMixin, name)
        assert getattr(OpenAIAdapter, name) is owner_method
        assert getattr(OpenAICompatibleAdapter, name) is owner_method


@pytest.mark.unit
def test_native_and_compatible_adapters_share_chat_protocol_owners_and_usage_helpers():
    assert issubclass(_OpenAIPayloadMixin, _ChatCompletionsPayloadMixin)
    assert issubclass(_OpenAIStreamingMixin, _ChatCompletionsStreamingMixin)

    for name in (
        "_build_chat_completions_params",
        "_map_tools_to_openai",
        "_map_tool_choice_to_openai",
    ):
        owner_method = getattr(_ChatCompletionsPayloadMixin, name)
        assert getattr(OpenAIAdapter, name) is owner_method
        assert getattr(OpenAICompatibleAdapter, name) is owner_method

    # Compatible mode keeps its validation overrides as dynamic hooks around
    # these shared implementations.
    assert OpenAIAdapter._normalize_chat_completions_response is (
        _ChatCompletionsPayloadMixin._normalize_chat_completions_response
    )
    assert OpenAIAdapter._validate_chat_completions_stream_event is (
        _ChatCompletionsPayloadMixin._validate_chat_completions_stream_event
    )
    assert OpenAICompatibleAdapter._normalize_chat_completions_response is not (
        _ChatCompletionsPayloadMixin._normalize_chat_completions_response
    )
    assert OpenAICompatibleAdapter._validate_chat_completions_stream_event is not (
        _ChatCompletionsPayloadMixin._validate_chat_completions_stream_event
    )

    for name in (
        "_consume_chat_completions_stream_async",
        "_consume_chat_completions_stream",
        "_update_legacy_stream_metadata",
        "_consume_legacy_stream_choice",
        "_accumulate_legacy_tool_calls",
        "_build_legacy_stream_response",
    ):
        owner_method = getattr(_ChatCompletionsStreamingMixin, name)
        assert getattr(OpenAIAdapter, name) is owner_method
        assert getattr(OpenAICompatibleAdapter, name) is owner_method

    assert native_token_count is _token_count
    assert native_cache_token_counts is _cache_token_counts
    assert native_apply_cache_usage is _apply_cache_usage
    assert _OpenAIStreamingMixin._token_count is _token_count


@pytest.mark.unit
def test_native_responses_protocol_owners_keep_policy_and_eof_native():
    from src.llm_api_adapter.adapters.openai import streaming as native_streaming

    assert _OpenAIPayloadMixin.__bases__ == (
        _ChatCompletionsPayloadMixin,
        _ResponsesPayloadMixin,
    )
    assert _OpenAIStreamingMixin.__bases__ == (
        _ChatCompletionsStreamingMixin,
        _ResponsesStreamingMixin,
    )
    assert OpenAIAdapter._map_tools_to_openai_responses is (
        _ResponsesPayloadMixin._map_tools_to_openai_responses
    )
    assert OpenAIAdapter._map_tool_choice_to_openai_responses is (
        _ResponsesPayloadMixin._map_tool_choice_to_openai_responses
    )
    assert OpenAIAdapter._normalize_responses_response is (
        _ResponsesPayloadMixin._normalize_responses_response
    )
    assert OpenAIAdapter._consume_responses_stream_event is (
        _ResponsesStreamingMixin._consume_responses_stream_event
    )
    assert OpenAIAdapter._responses_input_for_request is (
        _OpenAIPayloadMixin._responses_input_for_request
    )
    assert OpenAIAdapter._resolve_openai_reasoning_effort is (
        _OpenAIPayloadMixin._resolve_openai_reasoning_effort
    )
    assert OpenAIAdapter._build_responses_stream_response.__module__.endswith(
        ".openai.adapter"
    )
    assert native_streaming._ResponsesStreamState is _ResponsesStreamState
    assert native_streaming._cache_token_counts is _cache_token_counts
    assert not hasattr(OpenAICompatibleAdapter, "_build_responses_stream_response")

    with patch.object(
        _OpenAIStreamingMixin,
        "_token_count",
        staticmethod(lambda value: 7 if value == 1 else None),
    ):
        usage = OpenAIAdapter._normalize_stream_usage(
            {"input_tokens": 1, "output_tokens": 1},
            input_field="input_tokens",
            output_field="output_tokens",
        )
    assert usage.input_tokens == 7
    assert usage.output_tokens == 7
