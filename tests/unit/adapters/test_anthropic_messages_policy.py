"""Characterize native policy while proving shared Messages execution ownership."""

from unittest.mock import AsyncMock, patch

import pytest

from src.llm_api_adapter.adapters.anthropic.adapter import AnthropicAdapter
from src.llm_api_adapter.adapters.anthropic_compatible import payloads as shared_payloads
from src.llm_api_adapter.adapters.anthropic_compatible import streaming as shared_streaming
from src.llm_api_adapter.llms.anthropic.async_client import ClaudeAsyncClient
from src.llm_api_adapter.llms.anthropic.sync_client import ClaudeSyncClient
from src.llm_api_adapter.llms.streaming import SSEEvent
from src.llm_api_adapter.errors.llm_api_error import LLMAPIClientError
from src.llm_api_adapter.models.messages.chat_message import UserMessage


def _adapter():
    return AnthropicAdapter(api_key="test-key", model="claude-sonnet-4-5")


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["chat", "achat", "stream_chat", "astream_chat"])
def test_native_modes_dispatch_payload_through_shared_assembler(mode):
    adapter = _adapter()
    common = shared_payloads.build_messages_payload
    with patch.object(shared_payloads, "build_messages_payload", wraps=common) as assemble:
        if mode == "chat":
            with patch.object(ClaudeSyncClient, "chat_completion", return_value={"content": []}):
                adapter.chat([UserMessage("hello")], max_tokens=16)
        elif mode == "achat":
            with patch.object(ClaudeAsyncClient, "chat_completion", new_callable=AsyncMock, return_value={"content": []}):
                import asyncio
                asyncio.run(adapter.achat([UserMessage("hello")], max_tokens=16))
        elif mode == "stream_chat":
            with patch.object(ClaudeSyncClient, "stream", return_value=iter([])):
                list(adapter.stream_chat([UserMessage("hello")], max_tokens=16))
        else:
            async def run():
                async def events():
                    if False:
                        yield None
                with patch.object(ClaudeAsyncClient, "stream", return_value=events()):
                    return [item async for item in adapter.astream_chat([UserMessage("hello")], max_tokens=16)]
            import asyncio
            asyncio.run(run())
    assert assemble.call_count == 1


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["stream_chat", "astream_chat"])
def test_native_stream_modes_dispatch_shared_reconstruction_and_finalize(mode):
    adapter = _adapter()
    raw_events = [
        SSEEvent(event="content_block_start", data={"index": 0, "content_block": {"type": "text", "text": ""}}),
        SSEEvent(event="content_block_delta", data={"index": 0, "delta": {"type": "text_delta", "text": "native"}}),
    ]
    consume, finalize = shared_streaming.consume_messages_event, shared_streaming.finalize_messages_stream
    with patch.object(shared_streaming, "consume_messages_event", wraps=consume) as consume_spy, \
         patch.object(shared_streaming, "finalize_messages_stream", wraps=finalize) as finalize_spy:
        done = []
        if mode == "stream_chat":
            with patch.object(ClaudeSyncClient, "stream", return_value=iter(raw_events)):
                assert list(adapter.stream_chat([UserMessage("hello")], max_tokens=16, on_done=done.append)) == ["native"]
        else:
            async def run():
                async def events():
                    for item in raw_events:
                        yield item
                async_done = []
                async def on_done(response):
                    async_done.append(response)
                with patch.object(ClaudeAsyncClient, "stream", return_value=events()):
                    output = [item async for item in adapter.astream_chat([UserMessage("hello")], max_tokens=16, on_done=on_done)]
                return output, async_done
            import asyncio
            output, done = asyncio.run(run())
            assert output == ["native"]
    assert consume_spy.call_count == 2
    finalize_spy.assert_called_once()
    assert done[0].content == "native"


@pytest.mark.unit
def test_native_duplicate_block_start_replaces_through_shared_consumer():
    adapter = _adapter()
    events = iter([
        SSEEvent(event="content_block_start", data={"index": 0, "content_block": {"type": "text", "text": "old"}}),
        SSEEvent(event="content_block_start", data={"index": 0, "content_block": {"type": "text", "text": "new"}}),
    ])
    with patch.object(ClaudeSyncClient, "stream", return_value=events):
        done = []
        assert list(adapter.stream_chat([UserMessage("hello")], max_tokens=16, on_done=done.append)) == []
    assert done[0].content == "new"


@pytest.mark.unit
def test_shared_consumer_keeps_duplicate_start_strict_by_default():
    state = shared_streaming.MessagesBlockState()
    event = SSEEvent(event="content_block_start", data={"index": 0, "content_block": {"type": "text", "text": ""}})
    shared_streaming.consume_messages_event(state, event)
    with pytest.raises(LLMAPIClientError, match="started more than once"):
        shared_streaming.consume_messages_event(state, event)


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["stream_chat", "astream_chat"])
@pytest.mark.parametrize("capture_reasoning", [False, True])
@pytest.mark.parametrize("case", ["bool_index", "text_on_tool", "thinking_on_text", "text_on_unknown"])
def test_native_stream_preserves_historical_delta_tolerance(mode, capture_reasoning, case):
    adapter = _adapter()
    index = True if case == "bool_index" else 0
    block = {
        "bool_index": {"type": "text", "text": ""},
        "text_on_tool": {"type": "tool_use", "id": "tool-1", "name": "lookup", "input": {}},
        "thinking_on_text": {"type": "text", "text": ""},
        "text_on_unknown": {"type": "custom", "value": "kept"},
    }[case]
    delta = (
        {"type": "thinking_delta", "thinking": "private"}
        if case == "thinking_on_text"
        else {"type": "text_delta", "text": "visible"}
    )
    raw_events = [
        SSEEvent(event="content_block_start", data={"index": index, "content_block": block}),
        SSEEvent(event="content_block_delta", data={"index": index, "delta": delta}),
    ]
    reasoning = []
    done = []
    if mode == "stream_chat":
        with patch.object(ClaudeSyncClient, "stream", return_value=iter(raw_events)):
            output = list(adapter.stream_chat(
                [UserMessage("hello")], max_tokens=16, capture_reasoning=capture_reasoning,
            on_reasoning=lambda event: reasoning.append(event.text), on_done=done.append,
            ))
    else:
        async def run():
            async def events():
                for item in raw_events:
                    yield item
            async_reasoning, async_done = [], []
            async def on_reasoning(event):
                async_reasoning.append(event.text)
            async def on_done(response):
                async_done.append(response)
            with patch.object(ClaudeAsyncClient, "stream", return_value=events()):
                output = [item async for item in adapter.astream_chat(
                    [UserMessage("hello")], max_tokens=16, capture_reasoning=capture_reasoning,
                    on_reasoning=on_reasoning, on_done=on_done,
                )]
            return output, async_reasoning, async_done
        import asyncio
        output, reasoning, done = asyncio.run(run())
    expected_visible = "visible" if case != "thinking_on_text" else ""
    assert "".join(output) == expected_visible
    assert reasoning == (["private"] if capture_reasoning and case == "thinking_on_text" else [])
    assert len(done) == 1


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["stream_chat", "astream_chat"])
def test_native_duplicate_tool_start_retains_prior_json_fragments(mode):
    adapter = _adapter()
    events_list = [
        SSEEvent(event="content_block_start", data={"index": 0, "content_block": {"type": "tool_use", "id": "tool-1", "name": "lookup", "input": {}}}),
        SSEEvent(event="content_block_delta", data={"index": 0, "delta": {"type": "input_json_delta", "partial_json": '{"n":'}}),
        SSEEvent(event="content_block_start", data={"index": 0, "content_block": {"type": "tool_use", "id": "tool-1", "name": "lookup", "input": {}}}),
        SSEEvent(event="content_block_delta", data={"index": 0, "delta": {"type": "input_json_delta", "partial_json": "2}"}}),
        SSEEvent(event="content_block_stop", data={"index": 0}),
    ]
    if mode == "stream_chat":
        with patch.object(ClaudeSyncClient, "stream", return_value=iter(events_list)):
            done = []
            list(adapter.stream_chat([UserMessage("hello")], max_tokens=16, on_done=done.append))
    else:
        async def run():
            async def events():
                for item in events_list:
                    yield item
            async_done = []
            async def on_done(response):
                async_done.append(response)
            with patch.object(ClaudeAsyncClient, "stream", return_value=events()):
                [item async for item in adapter.astream_chat([UserMessage("hello")], max_tokens=16, on_done=on_done)]
            return async_done
        import asyncio
        done = asyncio.run(run())
    assert done[0].tool_calls[0].arguments == {"n": 2}


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["stream_chat", "astream_chat"])
def test_native_empty_tool_json_fragment_is_ignored(mode):
    adapter = _adapter()
    events_list = [
        SSEEvent(event="content_block_start", data={"index": 0, "content_block": {"type": "tool_use", "id": "tool-1", "name": "lookup", "input": {}}}),
        SSEEvent(event="content_block_delta", data={"index": 0, "delta": {"type": "input_json_delta", "partial_json": ""}}),
        SSEEvent(event="content_block_stop", data={"index": 0}),
    ]
    if mode == "stream_chat":
        with patch.object(ClaudeSyncClient, "stream", return_value=iter(events_list)):
            done = []
            list(adapter.stream_chat([UserMessage("hello")], max_tokens=16, on_done=done.append))
    else:
        async def run():
            async def events():
                for item in events_list:
                    yield item
            async_done = []
            async def on_done(response):
                async_done.append(response)
            with patch.object(ClaudeAsyncClient, "stream", return_value=events()):
                [item async for item in adapter.astream_chat([UserMessage("hello")], max_tokens=16, on_done=on_done)]
            return async_done
        import asyncio
        done = asyncio.run(run())
    assert done[0].tool_calls[0].arguments == {}


@pytest.mark.unit
def test_shared_consumer_rejects_provider_specific_delta_without_policy():
    state = shared_streaming.MessagesBlockState()
    shared_streaming.consume_messages_event(state, SSEEvent(
        event="content_block_start",
        data={"index": 0, "content_block": {"type": "tool_use", "id": "tool-1", "name": "lookup", "input": {}}},
    ))
    with pytest.raises(LLMAPIClientError, match="does not match its block"):
        shared_streaming.consume_messages_event(state, SSEEvent(
            event="content_block_delta", data={"index": 0, "delta": {"type": "text_delta", "text": "visible"}},
        ))


@pytest.mark.unit
def test_shared_consumer_keeps_boolean_indexes_strict_by_default():
    with pytest.raises(LLMAPIClientError, match="index is invalid"):
        shared_streaming.consume_messages_event(shared_streaming.MessagesBlockState(), SSEEvent(
            event="content_block_start",
            data={"index": True, "content_block": {"type": "text", "text": ""}},
        ))


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["stream_chat", "astream_chat"])
@pytest.mark.parametrize("malformed_type", [[], {}])
def test_native_stream_ignores_unhashable_delta_types(mode, malformed_type):
    adapter = _adapter()
    events_list = [
        SSEEvent(event="content_block_start", data={"index": 0, "content_block": {"type": "text", "text": ""}}),
        SSEEvent(event="content_block_delta", data={"index": 0, "delta": {"type": malformed_type}}),
    ]
    if mode == "stream_chat":
        with patch.object(ClaudeSyncClient, "stream", return_value=iter(events_list)):
            assert list(adapter.stream_chat([UserMessage("hello")], max_tokens=16)) == []
    else:
        async def run():
            async def events():
                for item in events_list:
                    yield item
            with patch.object(ClaudeAsyncClient, "stream", return_value=events()):
                return [item async for item in adapter.astream_chat([UserMessage("hello")], max_tokens=16)]
        import asyncio
        assert asyncio.run(run()) == []


@pytest.mark.unit
def test_shared_consumer_rejects_unhashable_delta_types_cleanly():
    state = shared_streaming.MessagesBlockState()
    shared_streaming.consume_messages_event(state, SSEEvent(
        event="content_block_start",
        data={"index": 0, "content_block": {"type": "text", "text": ""}},
    ))
    with pytest.raises(LLMAPIClientError, match="delta is malformed"):
        shared_streaming.consume_messages_event(state, SSEEvent(
            event="content_block_delta", data={"index": 0, "delta": {"type": []}},
        ))
