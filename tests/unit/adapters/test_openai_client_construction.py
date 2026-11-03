from unittest.mock import AsyncMock, patch
import json

import pytest

from src.llm_api_adapter.adapters.openai import adapter as openai_adapter_module
from src.llm_api_adapter.adapters.openai.adapter import OpenAIAdapter
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
