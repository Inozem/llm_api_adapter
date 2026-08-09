from unittest.mock import patch

import pytest

from src.llm_api_adapter.adapters.anthropic_adapter import AnthropicAdapter
from src.llm_api_adapter.errors.llm_api_error import LLMAPIError
from src.llm_api_adapter.adapters.anthropic_adapter import ClaudeSyncClient
from src.llm_api_adapter.models.messages.chat_message import Prompt, UserMessage
from src.llm_api_adapter.models.responses.chat_response import ChatResponse, Usage
from src.llm_api_adapter.llm_registry.llm_registry import Pricing
from src.llm_api_adapter.models.responses.reasoning_event import ReasoningEvent
from src.llm_api_adapter.models.tools import ToolSpec
from src.llm_api_adapter.llms.streaming import SSEEvent

@pytest.fixture
def adapter():
    return AnthropicAdapter(
        api_key="test_api_key",
        model="claude-sonnet-4-5"
    )

@pytest.mark.parametrize("temperature,width,valid", [
    (1.0, 256, True),
    (-0.1, 256, False),
    (2.1, 256, False),
    (0.0, 256, True),
])
@pytest.mark.unit
def test_temperature_validation(adapter, temperature, width, valid):
    if valid:
        result = adapter._validate_parameter("temperature", temperature, 0, 2)
        assert result == temperature
    else:
        with pytest.raises(ValueError):
            adapter._validate_parameter("temperature", temperature, 0, 2)

@pytest.mark.unit
def test_chat_handles_llmapi_error(adapter):
    messages = [Prompt("system prompt"), UserMessage("hello")]
    method = "chat_completion"
    with patch.object(
        ClaudeSyncClient, method, side_effect=LLMAPIError("API error")
    ), patch.object(adapter, "handle_error") as mock_handle_error:
        adapter.chat(messages=messages, max_tokens=2000)
        mock_handle_error.assert_called_once()

@pytest.mark.unit
def test_chat_handles_generic_exception(adapter):
    messages = [Prompt("system prompt"), UserMessage("hello")]
    method = "chat_completion"
    with patch.object(
        ClaudeSyncClient, method, side_effect=Exception("Generic error")
    ), patch.object(adapter, "handle_error") as mock_handle_error:
        adapter.chat(messages=messages, max_tokens=2000)
        mock_handle_error.assert_called_once()

@pytest.mark.unit
def test_pricing_is_applied_when_present(adapter):
    adapter.pricing = Pricing.from_dict(
        [{"up_to_prompt_tokens": None, "input_per_1m": 1_000, "output_per_1m": 2_000}],
        currency="USD",
    )
    fake_response = {"some": "anthropic response"}
    fake_chat_response = ChatResponse(content="fake")
    patch_chat_completion = patch.object(
        ClaudeSyncClient, "chat_completion", return_value=fake_response
    )
    patch_from_anthropic = patch.object(
        ChatResponse, "from_anthropic_response", return_value=fake_chat_response
    )
    patch_apply_pricing = patch.object(ChatResponse, "apply_pricing")
    with (
        patch_chat_completion,
        patch_from_anthropic as mock_from,
        patch_apply_pricing as mock_apply
    ):
        result = adapter.chat([
            UserMessage("hi")
        ], max_tokens=10)
        mock_apply.assert_called_once_with(
            price_input_per_token=adapter.pricing.tiers[0].in_per_token,
            price_output_per_token=adapter.pricing.tiers[0].out_per_token,
            currency=adapter.pricing.currency,
        )

@pytest.mark.unit
def test_normalize_reasoning_level_int_below_minimum_warns(adapter):
    adapter.is_reasoning = True
    with pytest.warns(UserWarning):
        result = adapter._normalize_reasoning_level(512)
    assert result == 1024

@pytest.mark.unit
def test_normalize_reasoning_level_bool_raises(adapter):
    adapter.is_reasoning = True
    with pytest.raises(ValueError):
        adapter._normalize_reasoning_level(True)

@pytest.mark.unit
def test_normalize_reasoning_level_unknown_str_raises(adapter):
    adapter.is_reasoning = True
    adapter.reasoning_levels = {"low": 2048}
    with pytest.raises(ValueError):
        adapter._normalize_reasoning_level("unknown-key")

@pytest.mark.unit
def test_chat_sets_thinking_when_reasoning_level_provided(adapter):
    adapter.is_reasoning = True
    adapter.reasoning_levels = {"high": 4096}
    fake_response = {"some": "anthropic response"}
    fake_chat_response = ChatResponse(content="fake")
    with patch.object(ClaudeSyncClient, "chat_completion", return_value=fake_response) as mock_chat, \
         patch.object(ChatResponse, "from_anthropic_response", return_value=fake_chat_response):
        adapter.chat([UserMessage("hi")], max_tokens=10000, reasoning_level="high")
        mock_chat.assert_called_once()
        kwargs = mock_chat.call_args.kwargs
        assert "budget_tokens" in kwargs
        assert kwargs["budget_tokens"] == 4096


@pytest.mark.unit
def test_chat_capture_reasoning_is_opt_in(adapter):
    fake_response = {
        "model": "claude-sonnet-4-5",
        "content": [
            {"type": "thinking", "thinking": "Plan", "signature": "secret"},
            {"type": "text", "text": "Answer"},
        ],
    }
    with patch.object(
        ClaudeSyncClient,
        "chat_completion",
        return_value=fake_response,
    ) as mock_chat:
        result = adapter.chat(
            [UserMessage("hi")],
            max_tokens=2048,
            reasoning_level=1024,
            capture_reasoning=True,
        )

    assert mock_chat.call_args.kwargs["capture_reasoning"] is True
    assert result.reasoning_events == [
        ReasoningEvent("Plan", "summary", 0, 0.0, 0.0),
    ]
    assert result.content == "Answer"


@pytest.mark.unit
def test_validate_reasoning_and_tokens_raises_llm_reasoning_level_error(adapter):
    from src.llm_api_adapter.errors.config_errors import LLMReasoningLevelError
    with pytest.raises(LLMReasoningLevelError):
        adapter.validate_reasoning_and_tokens(
            max_tokens=1024,
            reasoning_level="high",
            normalized_reasoning_level=2048
        )


@pytest.mark.unit
def test_chat_skips_validation_for_adaptive_thinking_model(adapter):
    adapter.is_reasoning = True
    adapter.is_adaptive_thinking = True
    adapter.reasoning_levels = {"high": 4096}
    fake_response = {"some": "anthropic response"}
    fake_chat_response = ChatResponse(content="fake")
    with patch.object(ClaudeSyncClient, "chat_completion", return_value=fake_response), \
         patch.object(ChatResponse, "from_anthropic_response", return_value=fake_chat_response):
        # max_tokens < budget_tokens — would raise LLMReasoningLevelError for a legacy model
        adapter.chat([UserMessage("hi")], max_tokens=100, reasoning_level="high")


# ---------------------------
# json_schema
# ---------------------------

@pytest.mark.unit
def test_chat_passes_output_config_when_json_schema_provided(adapter):
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    fake_response = {"some": "anthropic response"}
    fake_chat_response = ChatResponse(content='{"name": "test"}')

    with (
        patch.object(ClaudeSyncClient, "chat_completion", return_value=fake_response) as mock_chat,
        patch.object(ChatResponse, "from_anthropic_response", return_value=fake_chat_response),
    ):
        result = adapter.chat([UserMessage("hi")], max_tokens=100, json_schema=schema)

    kwargs = mock_chat.call_args.kwargs
    assert "output_config" in kwargs
    assert kwargs["output_config"]["format"]["type"] == "json_schema"
    assert "schema" in kwargs["output_config"]["format"]
    assert result.parsed_json == {"name": "test"}


@pytest.mark.unit
def test_chat_omits_output_config_when_json_schema_is_none(adapter):
    fake_response = {"some": "anthropic response"}
    fake_chat_response = ChatResponse(content="plain text")

    with (
        patch.object(ClaudeSyncClient, "chat_completion", return_value=fake_response) as mock_chat,
        patch.object(ChatResponse, "from_anthropic_response", return_value=fake_chat_response),
    ):
        adapter.chat([UserMessage("hi")], max_tokens=100)

    kwargs = mock_chat.call_args.kwargs
    assert "output_config" not in kwargs


@pytest.mark.unit
def test_stream_chat_normalizes_text_and_completed_tool_use(adapter):
    events = iter([
        SSEEvent(
            event="message_start",
            data={"type": "message_start", "message": {
                "id": "msg_123", "model": "claude-sonnet-4-5", "content": [],
                "usage": {"input_tokens": 5, "output_tokens": 0},
            }},
        ),
        SSEEvent(
            event="content_block_start",
            data={"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
        SSEEvent(
            event="content_block_delta",
            data={"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}},
        ),
        SSEEvent(
            event="content_block_start",
            data={"type": "content_block_start", "index": 1, "content_block": {
                "type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {},
            }},
        ),
        SSEEvent(
            event="content_block_delta",
            data={"type": "content_block_delta", "index": 1, "delta": {
                "type": "input_json_delta", "partial_json": "{\"city\": \"Tel",
            }},
        ),
        SSEEvent(
            event="content_block_delta",
            data={"type": "content_block_delta", "index": 1, "delta": {
                "type": "input_json_delta", "partial_json": " Aviv\"}",
            }},
        ),
        SSEEvent(event="content_block_stop", data={"type": "content_block_stop", "index": 1}),
        SSEEvent(
            event="message_delta",
            data={"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 8}},
        ),
        SSEEvent(event="message_stop", data={"type": "message_stop"}),
    ])
    done = []
    tool_calls = []
    callbacks = []
    tool = ToolSpec(name="get_weather", json_schema={"type": "object"})

    with patch.object(ClaudeSyncClient, "stream", return_value=events) as mock_stream:
        assert list(adapter.stream_chat(
            [UserMessage("hi")],
            max_tokens=64,
            tools=[tool],
            on_delta=lambda delta: callbacks.append(("delta", delta)),
            on_tool_call=lambda call: (callbacks.append(("tool", call)), tool_calls.append(call)),
            on_done=lambda response: (callbacks.append(("done", response)), done.append(response)),
        )) == ["Hello"]

    assert [kind for kind, _ in callbacks] == ["delta", "tool", "done"]
    assert mock_stream.call_args.kwargs["messages"] == [{"role": "user", "content": "hi"}]
    assert done[0].content == "Hello"
    assert done[0].usage.total_tokens == 13
    assert done[0].finish_reason == "tool_use"
    assert tool_calls[0].name == "get_weather"
    assert tool_calls[0].arguments == {"city": "Tel Aviv"}


@pytest.mark.unit
def test_stream_chat_separates_thinking_from_visible_text(adapter):
    events = iter([
        SSEEvent(
            event="message_start",
            data={"type": "message_start", "message": {
                "model": "claude-sonnet-4-5", "content": [],
            }},
        ),
        SSEEvent(
            event="content_block_start",
            data={"type": "content_block_start", "index": 0, "content_block": {
                "type": "thinking", "thinking": "",
            }},
        ),
        SSEEvent(
            event="content_block_delta",
            data={"type": "content_block_delta", "index": 0, "delta": {
                "type": "thinking_delta", "thinking": "Plan",
            }},
        ),
        SSEEvent(
            event="content_block_delta",
            data={"type": "content_block_delta", "index": 0, "delta": {
                "type": "signature_delta", "signature": "secret",
            }},
        ),
        SSEEvent(
            event="content_block_stop",
            data={"type": "content_block_stop", "index": 0},
        ),
        SSEEvent(
            event="content_block_start",
            data={"type": "content_block_start", "index": 1, "content_block": {
                "type": "text", "text": "",
            }},
        ),
        SSEEvent(
            event="content_block_delta",
            data={"type": "content_block_delta", "index": 1, "delta": {
                "type": "text_delta", "text": "Answer",
            }},
        ),
        SSEEvent(
            event="content_block_stop",
            data={"type": "content_block_stop", "index": 1},
        ),
        SSEEvent(event="message_stop", data={"type": "message_stop"}),
    ])
    reasoning = []
    visible_deltas = []
    done = []

    with patch.object(ClaudeSyncClient, "stream", return_value=events) as mock_stream:
        output = list(adapter.stream_chat(
            [UserMessage("hi")],
            max_tokens=2048,
            reasoning_level=1024,
            capture_reasoning=True,
            on_delta=visible_deltas.append,
            on_reasoning=reasoning.append,
            on_done=done.append,
        ))

    assert output == ["Answer"]
    assert visible_deltas == ["Answer"]
    assert len(reasoning) == 1
    assert reasoning[0].text == "Plan"
    assert reasoning[0].kind == "summary"
    assert reasoning[0].index == 0
    assert reasoning[0].elapsed_s >= 0.0
    assert reasoning[0].delta_s >= 0.0
    assert done[0].reasoning_events == reasoning
    assert mock_stream.call_args.kwargs["capture_reasoning"] is True


@pytest.mark.unit
def test_stream_chat_attaches_late_usage_to_final_buffered_chunk(adapter):
    events = iter([
        SSEEvent(
            event="message_start",
            data={"type": "message_start", "message": {
                "model": "claude-sonnet-4-5",
                "content": [],
                "usage": {"input_tokens": 3, "output_tokens": 0},
            }},
        ),
        SSEEvent(
            event="content_block_start",
            data={"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
        SSEEvent(
            event="content_block_delta",
            data={"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}},
        ),
        SSEEvent(
            event="message_delta",
            data={"type": "message_delta", "usage": {"output_tokens": 4}},
        ),
        SSEEvent(event="message_stop", data={"type": "message_stop"}),
    ])
    chunks = []
    done = []

    with patch.object(ClaudeSyncClient, "stream", return_value=events):
        output = list(adapter.stream_chat(
            [UserMessage("hi")],
            max_tokens=64,
            buffer_chars=10,
            on_chunk=chunks.append,
            on_done=done.append,
        ))

    assert output == ["Hello"]
    assert chunks[0].usage == Usage(input_tokens=3, output_tokens=4, total_tokens=7)
    assert chunks[0].output_tokens_delta == 4
    assert done[0].usage == Usage(input_tokens=3, output_tokens=4, total_tokens=7)


@pytest.mark.unit
def test_stream_chat_buffers_text_and_orders_chunk_callbacks(adapter):
    events = iter([
        SSEEvent(
            event="message_start",
            data={"type": "message_start", "message": {"content": []}},
        ),
        SSEEvent(
            event="content_block_start",
            data={"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
        SSEEvent(
            event="content_block_delta",
            data={"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "He"}},
        ),
        SSEEvent(
            event="content_block_delta",
            data={"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "llo"}},
        ),
        SSEEvent(
            event="content_block_delta",
            data={"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "!"}},
        ),
        SSEEvent(event="message_stop", data={"type": "message_stop"}),
    ])
    order = []
    yielded = []

    with patch.object(ClaudeSyncClient, "stream", return_value=events):
        for text in adapter.stream_chat(
            [UserMessage("hi")],
            max_tokens=64,
            buffer_chars=4,
            on_chunk=lambda chunk: order.append(("chunk", chunk.text)),
            on_delta=lambda text: order.append(("delta", text)),
            on_done=lambda response: order.append(("done", response.content)),
        ):
            yielded.append(text)
            order.append(("yield", text))

    assert yielded == ["Hell", "o!"]
    assert "".join(yielded) == "Hello!"
    assert order == [
        ("chunk", "Hell"),
        ("delta", "Hell"),
        ("yield", "Hell"),
        ("chunk", "o!"),
        ("delta", "o!"),
        ("yield", "o!"),
        ("done", "Hello!"),
    ]
