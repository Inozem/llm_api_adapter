import asyncio
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
from src.llm_api_adapter.adapters.openai_compatible.common.payloads import (
    _OpenAICompatiblePayloadMixin,
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
        self.last_stream = _events(self.model)
        return self.last_stream


class _AsyncClient(_SyncClient):
    async def complete(self, *, timeout=None, **params):
        self.completed_routes.append("responses" if self.responses_api else "chat")
        return _response(self.model)

    def stream(self, *, timeout=None, **params):
        self.streamed_routes.append("responses" if self.responses_api else "chat")

        async def events():
            for event in _events(self.model):
                yield event

        self.last_stream = events()
        return self.last_stream


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
    assert OpenAIAdapter.__bases__ == (OpenAICompatibleAdapter,)
    assert issubclass(OpenAIAdapter, LLMAdapterBase)
    assert issubclass(OpenAICompatibleAdapter, LLMAdapterBase)
    assert not issubclass(OpenAICompatibleAdapter, OpenAIAdapter)

    for name in ("chat", "achat", "stream_chat", "astream_chat", "_astream_chat"):
        owner_method = getattr(_OpenAIExecutionMixin, name)
        assert getattr(OpenAIAdapter, name) is owner_method
        assert getattr(OpenAICompatibleAdapter, name) is owner_method
        assert name not in OpenAIAdapter.__dict__


@pytest.mark.unit
def test_native_and_compatible_adapters_share_chat_protocol_owners_and_usage_helpers():
    assert issubclass(_OpenAICompatiblePayloadMixin, _ChatCompletionsPayloadMixin)
    assert issubclass(_OpenAIPayloadMixin, _OpenAICompatiblePayloadMixin)
    assert issubclass(_OpenAIStreamingMixin, _ChatCompletionsStreamingMixin)

    for name in (
        "_build_chat_params",
        "_build_stream_params",
        "_parse_chat_response",
    ):
        owner_method = getattr(_OpenAICompatiblePayloadMixin, name)
        assert getattr(OpenAIAdapter, name) is owner_method
        assert getattr(OpenAICompatibleAdapter, name) is owner_method

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
    assert OpenAIAdapter._token_count is _token_count
    assert OpenAIAdapter._token_count(7) == 7


@pytest.mark.unit
def test_native_responses_protocol_owners_keep_policy_and_eof_native():
    from src.llm_api_adapter.adapters.openai import streaming as native_streaming

    assert _OpenAICompatiblePayloadMixin.__bases__ == (
        _ChatCompletionsPayloadMixin,
        _ResponsesPayloadMixin,
    )
    assert _OpenAIPayloadMixin.__bases__ == (_OpenAICompatiblePayloadMixin,)
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


class _GenericHookedAdapter(OpenAICompatibleAdapter):
    def __init__(self, *, api_variant="chat_completions"):
        super().__init__(
            api_key="test-key",
            model="unregistered-model",
            base_url="https://example.test/v1",
            api_variant=api_variant,
        )
        self.sync_client = _SyncClient(self.model)
        self.async_client = _AsyncClient(self.model)
        self.sync_client.responses_api = api_variant == "responses"
        self.async_client.responses_api = api_variant == "responses"

    def _create_sync_client(self):
        return self.sync_client

    def _create_async_client(self):
        return self.async_client


class _OldConsumerObservedAdapter(_HookedAdapter):
    def __init__(self, *, api_key, model):
        super().__init__(api_key=api_key, model=model)
        self.consumer_args = []

    # Deliberately preserve the legacy strict signature: no new keyword args.
    def _consume_chat_completions_stream(
        self, events, effective_schema, response_model, on_delta, on_tool_call,
        on_done, buffer_chars, on_chunk,
    ):
        self.consumer_args.append((
            events, effective_schema, response_model, on_delta, on_tool_call,
            on_done, buffer_chars, on_chunk,
        ))
        return iter(["legacy"])

    async def _consume_chat_completions_stream_async(
        self, events, effective_schema, response_model, on_delta, on_tool_call,
        on_done, buffer_chars, on_chunk,
    ):
        self.consumer_args.append((
            events, effective_schema, response_model, on_delta, on_tool_call,
            on_done, buffer_chars, on_chunk,
        ))
        yield "legacy"


def _assert_instance_attributes_unchanged(instance, snapshot):
    """Permit mutation inside existing trackers, but no per-call state on self."""
    assert set(instance.__dict__) == set(snapshot)
    for name, value in snapshot.items():
        assert instance.__dict__[name] is value


@pytest.mark.unit
def test_default_sync_dispatch_preserves_exact_legacy_consumer_arguments():
    adapter = _OldConsumerObservedAdapter(api_key="test-key", model="gpt-4o")
    callbacks = [lambda *args: None for _ in range(4)]

    result = list(adapter.stream_chat(
        [UserMessage("hi")], on_delta=callbacks[0], on_tool_call=callbacks[1],
        on_done=callbacks[2], buffer_chars=17, on_chunk=callbacks[3],
        capture_reasoning=True, on_reasoning=lambda event: None,
    ))

    assert result == ["legacy"]
    assert len(adapter.consumer_args) == 1
    args = adapter.consumer_args[0]
    assert len(args) == 8
    assert args[0] is adapter.sync_client.last_stream
    assert args[1] is None and args[2] is None
    assert args[3] is callbacks[0] and args[4] is callbacks[1]
    assert args[5] is callbacks[2]
    assert args[6] == 17
    assert args[7] is callbacks[3]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_default_async_dispatch_preserves_exact_legacy_consumer_arguments():
    adapter = _OldConsumerObservedAdapter(api_key="test-key", model="gpt-4o")
    callbacks = [lambda *args: None for _ in range(4)]
    result = []

    async for chunk in adapter.astream_chat(
        [UserMessage("hi")], on_delta=callbacks[0], on_tool_call=callbacks[1],
        on_done=callbacks[2], buffer_chars=19, on_chunk=callbacks[3],
        capture_reasoning=True, on_reasoning=lambda event: None,
    ):
        result.append(chunk)

    assert result == ["legacy"]
    assert len(adapter.consumer_args) == 1
    args = adapter.consumer_args[0]
    assert len(args) == 8
    assert args[0] is adapter.async_client.last_stream
    assert args[1] is None and args[2] is None
    assert args[3] is callbacks[0] and args[4] is callbacks[1]
    assert args[5] is callbacks[2]
    assert args[6] == 19
    assert args[7] is callbacks[3]


@pytest.mark.unit
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("capture_reasoning", [False, True])
def test_sync_chat_dispatch_is_used_by_native_and_generic_adapters(
    native, capture_reasoning,
):
    adapter = (
        _HookedAdapter(api_key="test-key", model="gpt-4o")
        if native else _GenericHookedAdapter()
    )
    observed = []

    def dispatch(
        events, effective_schema, response_model, on_delta, on_tool_call,
        on_done, buffer_chars, on_chunk, *, capture_reasoning, on_reasoning,
    ):
        observed.append((capture_reasoning, on_reasoning))
        return iter(["routed"])

    adapter._dispatch_chat_completions_stream = dispatch
    callback = lambda event: None
    assert list(adapter.stream_chat(
        [UserMessage("hi")], capture_reasoning=capture_reasoning,
        on_reasoning=callback,
    )) == ["routed"]
    assert observed == [(capture_reasoning, callback)]


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("capture_reasoning", [False, True])
async def test_async_chat_dispatch_is_used_by_native_and_generic_adapters(
    native, capture_reasoning,
):
    adapter = (
        _HookedAdapter(api_key="test-key", model="gpt-4o")
        if native else _GenericHookedAdapter()
    )
    observed = []

    async def dispatch(
        events, effective_schema, response_model, on_delta, on_tool_call,
        on_done, buffer_chars, on_chunk, *, capture_reasoning, on_reasoning,
    ):
        observed.append((capture_reasoning, on_reasoning))
        yield "routed"

    adapter._dispatch_chat_completions_stream_async = dispatch
    callback = lambda event: None
    result = [chunk async for chunk in adapter.astream_chat(
        [UserMessage("hi")], capture_reasoning=capture_reasoning,
        on_reasoning=callback,
    )]
    assert result == ["routed"]
    assert observed == [(capture_reasoning, callback)]


@pytest.mark.unit
@pytest.mark.parametrize("native", [False, True])
def test_sync_responses_stream_keeps_direct_consumer_ownership_and_context(native):
    adapter = (
        _HookedAdapter(api_key="test-key", model="gpt-5")
        if native else _GenericHookedAdapter(api_variant="responses")
    )
    callback = lambda event: None
    observed = []

    def consume(
        events, effective_schema, response_model, on_delta, on_tool_call,
        on_done, buffer_chars, on_chunk, capture_reasoning, on_reasoning,
    ):
        observed.append((capture_reasoning, on_reasoning))
        return iter(["responses"])

    adapter._consume_responses_stream = consume
    adapter._dispatch_chat_completions_stream = lambda *args, **kwargs: pytest.fail(
        "Responses must not use Chat dispatch"
    )
    assert list(adapter.stream_chat(
        [UserMessage("hi")], capture_reasoning=True, on_reasoning=callback,
    )) == ["responses"]
    assert observed == [(True, callback)]


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True])
async def test_async_responses_stream_keeps_direct_consumer_ownership_and_context(native):
    adapter = (
        _HookedAdapter(api_key="test-key", model="gpt-5")
        if native else _GenericHookedAdapter(api_variant="responses")
    )
    callback = lambda event: None
    observed = []

    async def consume(
        events, effective_schema, response_model, on_delta, on_tool_call,
        on_done, buffer_chars, on_chunk, capture_reasoning, on_reasoning,
    ):
        observed.append((capture_reasoning, on_reasoning))
        yield "responses"

    adapter._consume_responses_stream_async = consume
    adapter._dispatch_chat_completions_stream_async = lambda *args, **kwargs: pytest.fail(
        "Responses must not use Chat dispatch"
    )
    result = [chunk async for chunk in adapter.astream_chat(
        [UserMessage("hi")], capture_reasoning=True, on_reasoning=callback,
    )]
    assert result == ["responses"]
    assert observed == [(True, callback)]


@pytest.mark.unit
def test_overlapping_sync_chat_streams_keep_capture_context_after_resumption():
    adapter = _HookedAdapter(api_key="test-key", model="gpt-4o")
    resumed = []

    def dispatch(
        events, effective_schema, response_model, on_delta, on_tool_call,
        on_done, buffer_chars, on_chunk, *, capture_reasoning, on_reasoning,
    ):
        yield "started"
        resumed.append((capture_reasoning, on_reasoning))

    adapter._dispatch_chat_completions_stream = dispatch
    adapter_attributes = dict(adapter.__dict__)
    first_callback = lambda event: None
    second_callback = lambda event: None
    first = adapter.stream_chat(
        [UserMessage("one")], capture_reasoning=True,
        on_reasoning=first_callback,
    )
    second = adapter.stream_chat(
        [UserMessage("two")], capture_reasoning=False,
        on_reasoning=second_callback,
    )

    assert next(first) == "started"
    _assert_instance_attributes_unchanged(adapter, adapter_attributes)
    assert next(second) == "started"
    _assert_instance_attributes_unchanged(adapter, adapter_attributes)
    with pytest.raises(StopIteration):
        next(first)
    _assert_instance_attributes_unchanged(adapter, adapter_attributes)
    with pytest.raises(StopIteration):
        next(second)
    _assert_instance_attributes_unchanged(adapter, adapter_attributes)
    assert resumed == [(True, first_callback), (False, second_callback)]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_overlapping_async_chat_streams_keep_capture_context_after_resumption():
    adapter = _HookedAdapter(api_key="test-key", model="gpt-4o")
    ready = [asyncio.Event(), asyncio.Event()]
    release = asyncio.Event()
    resumed = []

    dispatches = iter(range(2))

    async def dispatch(
        events, effective_schema, response_model, on_delta, on_tool_call,
        on_done, buffer_chars, on_chunk, *, capture_reasoning, on_reasoning,
    ):
        index = next(dispatches)
        yield f"started-{index}"
        ready[index].set()
        await release.wait()
        resumed.append((capture_reasoning, on_reasoning))

    callbacks = [lambda event: None, lambda event: None]
    adapter._dispatch_chat_completions_stream_async = dispatch
    adapter_attributes = dict(adapter.__dict__)
    first = adapter.astream_chat(
        [UserMessage("one")], capture_reasoning=True,
        on_reasoning=callbacks[0],
    )
    second = adapter.astream_chat(
        [UserMessage("two")], capture_reasoning=False,
        on_reasoning=callbacks[1],
    )

    async def collect(stream):
        return [chunk async for chunk in stream]

    tasks = [asyncio.create_task(collect(first)), asyncio.create_task(collect(second))]
    ready_waiters = [asyncio.create_task(event.wait()) for event in ready]
    try:
        await asyncio.wait(ready_waiters, timeout=1.0)
        _assert_instance_attributes_unchanged(adapter, adapter_attributes)
    finally:
        release.set()
        for waiter in ready_waiters:
            if not waiter.done():
                waiter.cancel()
    await asyncio.gather(*ready_waiters, return_exceptions=True)
    results = await asyncio.wait_for(asyncio.gather(*tasks), 1.0)
    _assert_instance_attributes_unchanged(adapter, adapter_attributes)
    assert all(event.is_set() for event in ready)
    assert results == [["started-0"], ["started-1"]]
    assert resumed == [(True, callbacks[0]), (False, callbacks[1])]
