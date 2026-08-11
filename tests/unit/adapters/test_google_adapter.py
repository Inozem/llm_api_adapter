from unittest.mock import patch

import pytest

from src.llm_api_adapter.adapters.google_adapter import GoogleAdapter
from src.llm_api_adapter.errors.llm_api_error import LLMAPIError
from src.llm_api_adapter.llms.google.sync_client import GeminiSyncClient
from src.llm_api_adapter.models.messages.chat_message import Prompt, UserMessage
from src.llm_api_adapter.models.responses.chat_response import ChatResponse, Usage
from src.llm_api_adapter.llm_registry.llm_registry import Pricing
from src.llm_api_adapter.models.tools import ToolSpec
from src.llm_api_adapter.llms.streaming import SSEEvent
from src.llm_api_adapter.models.responses.reasoning_event import ReasoningEvent

@pytest.fixture
def adapter():
    return GoogleAdapter(
        api_key="test_api_key",
        model="gemini-2.5-pro"
    )

@pytest.mark.parametrize("temperature,max_tokens,top_p,valid", [
    (1.0, 256, 1.0, True),
    (-0.1, 256, 1.0, False),
    (2.1, 256, 1.0, False),
    (0.0, 256, 1.0, True),
    (1.0, 256, -0.1, False),
    (1.0, 256, 1.1, False),
])
@pytest.mark.unit
def test_parameter_validation(adapter, temperature, max_tokens, top_p, valid):
    if valid:
        temp_result = adapter._validate_parameter(
            "temperature", temperature, 0, 2
        )
        top_p_result = adapter._validate_parameter("top_p", top_p, 0, 1)
        assert temp_result == temperature
        assert top_p_result == top_p
    else:
        if temperature < 0 or temperature > 2:
            with pytest.raises(ValueError):
                adapter._validate_parameter("temperature", temperature, 0, 2)
        if top_p < 0 or top_p > 1:
            with pytest.raises(ValueError):
                adapter._validate_parameter("top_p", top_p, 0, 1)

@pytest.mark.unit
def test_chat_handles_llmapi_error(adapter):
    messages = [Prompt("system prompt"), UserMessage("hello")]
    method = "chat_completion"
    with patch.object(
        GeminiSyncClient, method, side_effect=LLMAPIError("API error")
    ), patch.object(adapter, "handle_error") as mock_handle_error:
        adapter.chat(messages)
        mock_handle_error.assert_called_once()

@pytest.mark.unit
def test_chat_handles_generic_exception(adapter):
    messages = [Prompt("system prompt"), UserMessage("hello")]
    method = "chat_completion"
    with patch.object(
        GeminiSyncClient, method, side_effect=Exception("Generic error")
    ), patch.object(adapter, "handle_error") as mock_handle_error:
        adapter.chat(messages)
        mock_handle_error.assert_called_once()

@pytest.mark.unit
def test_pricing_is_applied_when_present(adapter):
    adapter.pricing = Pricing.from_dict(
        [{"up_to_prompt_tokens": None, "input_per_1m": 1_000, "output_per_1m": 2_000}],
        currency="USD",
    )
    fake_response = {"some": "google response"}
    fake_chat_response = ChatResponse()
    patch_chat_completion = patch.object(
        GeminiSyncClient, "chat_completion", return_value=fake_response
    )
    patch_from_google_response = patch.object(
        ChatResponse, "from_google_response", return_value=fake_chat_response
    )
    patch_apply_pricing = patch.object(ChatResponse, "apply_pricing")
    with (
        patch_chat_completion as mock_client,
        patch_from_google_response as mock_from,
        patch_apply_pricing as mock_apply
    ):
        result = adapter.chat([
            UserMessage("hi")
        ], max_tokens=10)
    mock_client.assert_called_once()
    mock_from.assert_called_once_with(fake_response)
    mock_apply.assert_called_once_with(
        price_input_per_token=adapter.pricing.tiers[0].in_per_token,
        price_output_per_token=adapter.pricing.tiers[0].out_per_token,
        currency=adapter.pricing.currency,
    )
    assert result is fake_chat_response

@pytest.mark.unit
def test_chat_includes_system_instruction_in_payload(adapter):
    from src.llm_api_adapter.models.messages.chat_message import Prompt, UserMessage
    messages = [Prompt("system prompt"), UserMessage("hello")]
    fake_response = {"some": "google response"}
    with patch.object(GeminiSyncClient, "chat_completion", return_value=fake_response) as mock_client, \
         patch.object(ChatResponse, "from_google_response", return_value=ChatResponse()):
        adapter.chat(messages, max_tokens=5)
    kwargs = mock_client.call_args[1]
    assert "contents" in kwargs
    assert "system_instruction" in kwargs
    assert isinstance(kwargs["system_instruction"], dict)

@pytest.mark.unit
def test_chat_adds_thinking_config_when_reasoning_level_set(adapter):
    from src.llm_api_adapter.models.messages.chat_message import UserMessage
    fake_response = {"some": "google response"}
    with patch.object(GeminiSyncClient, "chat_completion", return_value=fake_response) as mock_client, \
         patch.object(ChatResponse, "from_google_response", return_value=ChatResponse()):
        adapter.chat([UserMessage("hi")], reasoning_level=128)
    kwargs = mock_client.call_args[1]
    assert "generationConfig" in kwargs
    gen_cfg = kwargs["generationConfig"]
    assert "thinkingConfig" in gen_cfg
    assert isinstance(gen_cfg["thinkingConfig"], dict)
    assert gen_cfg["thinkingConfig"]["thinkingBudget"] == 128
    assert gen_cfg["thinkingConfig"]["includeThoughts"] is False


@pytest.mark.unit
def test_chat_captures_google_thought_summaries_when_opted_in(adapter):
    fake_response = {
        "modelVersion": "gemini-2.5-pro",
        "candidates": [{
            "content": {
                "parts": [
                    {"text": "Plan", "thought": True},
                    {"text": "Answer"},
                ]
            }
        }],
    }

    with patch.object(
        GeminiSyncClient,
        "chat_completion",
        return_value=fake_response,
    ) as mock_client:
        response = adapter.chat(
            [UserMessage("hi")],
            capture_reasoning=True,
        )

    thinking_config = mock_client.call_args.kwargs["generationConfig"][
        "thinkingConfig"
    ]
    assert thinking_config == {"includeThoughts": True}
    assert response.content == "Answer"
    assert response.reasoning_events == [
        ReasoningEvent("Plan", "summary", 0, 0.0, 0.0),
    ]

# ---------------------------
# json_schema
# ---------------------------

@pytest.mark.unit
def test_chat_passes_json_schema_in_generation_config(adapter):
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    fake_response = {"some": "google response"}
    fake_chat_response = ChatResponse(content='{"name": "test"}')

    with (
        patch.object(GeminiSyncClient, "chat_completion", return_value=fake_response) as mock_client,
        patch.object(ChatResponse, "from_google_response", return_value=fake_chat_response),
    ):
        result = adapter.chat([UserMessage("hi")], json_schema=schema)

    kwargs = mock_client.call_args[1]
    gen_cfg = kwargs["generationConfig"]
    assert gen_cfg["responseMimeType"] == "application/json"
    assert "responseSchema" in gen_cfg
    assert result.parsed_json == {"name": "test"}


@pytest.mark.unit
def test_to_google_schema_strips_unsupported_fields(adapter):
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "additionalProperties": False,
        "$schema": "http://json-schema.org/draft-07/schema#",
    }
    result = adapter._to_google_schema(schema)
    assert "additionalProperties" not in result
    assert "$schema" not in result
    assert result["type"] == "OBJECT"
    assert result["properties"]["name"]["type"] == "STRING"


@pytest.mark.unit
def test_chat_omits_json_schema_fields_when_not_provided(adapter):
    fake_response = {"some": "google response"}
    fake_chat_response = ChatResponse(content="plain text")

    with (
        patch.object(GeminiSyncClient, "chat_completion", return_value=fake_response) as mock_client,
        patch.object(ChatResponse, "from_google_response", return_value=fake_chat_response),
    ):
        adapter.chat([UserMessage("hi")])

    kwargs = mock_client.call_args[1]
    gen_cfg = kwargs["generationConfig"]
    assert "responseMimeType" not in gen_cfg
    assert "responseSchema" not in gen_cfg


@pytest.mark.unit
def test_stream_chat_normalizes_chunks_excludes_thoughts_and_finalizes(adapter):
    events = iter([
        SSEEvent(data={
            "candidates": [{"content": {"parts": [{"text": "Hello"}]}}],
        }, event=None),
        SSEEvent(data={
            "candidates": [{"content": {"parts": [{"text": "secret", "thought": True}]}}],
        }, event=None),
        SSEEvent(data={
            "candidates": [{
                "content": {"parts": [
                    {"text": "!"},
                    {"functionCall": {"name": "get_weather", "args": {"city": "Tel Aviv"}}},
                ]},
                "finishReason": "STOP",
            }],
            "usageMetadata": {
                "promptTokenCount": 3,
                "candidatesTokenCount": 2,
                "totalTokenCount": 5,
            },
        }, event=None),
    ])
    done = []
    tool_calls = []
    callbacks = []
    tool = ToolSpec(name="get_weather", json_schema={"type": "object"})

    with patch.object(GeminiSyncClient, "stream", return_value=events) as mock_stream:
        assert list(adapter.stream_chat(
            [UserMessage("hi")],
            tools=[tool],
            on_delta=lambda delta: callbacks.append(("delta", delta)),
            on_tool_call=lambda call: (callbacks.append(("tool", call)), tool_calls.append(call)),
            on_done=lambda response: (callbacks.append(("done", response)), done.append(response)),
        )) == ["Hello", "!"]

    assert [kind for kind, _ in callbacks] == ["delta", "delta", "tool", "done"]
    assert mock_stream.call_args.kwargs["contents"] == [{"role": "user", "parts": [{"text": "hi"}]}]
    assert done[0].content == "Hello!"
    assert done[0].finish_reason == "STOP"
    assert done[0].usage.total_tokens == 5
    assert tool_calls[0].name == "get_weather"
    assert tool_calls[0].arguments == {"city": "Tel Aviv"}


@pytest.mark.unit
def test_stream_chat_captures_google_thought_summaries_separately(adapter):
    events = iter([
        SSEEvent(data={
            "candidates": [{
                "content": {"parts": [{"text": "Plan", "thought": True}]}
            }],
        }, event=None),
        SSEEvent(data={
            "candidates": [{
                "content": {"parts": [{"text": "Answer"}]}
            }],
        }, event=None),
        SSEEvent(data={
            "candidates": [{
                "content": {"parts": [{"text": "Details", "thought": True}]},
                "finishReason": "STOP",
            }],
        }, event=None),
    ])
    reasoning = []
    done = []

    with patch.object(GeminiSyncClient, "stream", return_value=events) as mock_stream:
        output = list(adapter.stream_chat(
            [UserMessage("hi")],
            capture_reasoning=True,
            on_reasoning=reasoning.append,
            on_done=done.append,
        ))

    assert output == ["Answer"]
    assert [event.text for event in reasoning] == ["Plan", "Details"]
    assert done[0].content == "Answer"
    assert done[0].reasoning_events == reasoning
    thinking_config = mock_stream.call_args.kwargs["generationConfig"][
        "thinkingConfig"
    ]
    assert thinking_config == {"includeThoughts": True}


@pytest.mark.unit
def test_stream_chat_attaches_usage_with_thought_tokens_to_buffered_chunk(adapter):
    events = iter([
        SSEEvent(data={
            "candidates": [{"content": {"parts": [{"text": "Hello"}]}}],
        }, event=None),
        SSEEvent(data={
            "candidates": [{"content": {"parts": []}, "finishReason": "STOP"}],
            "usageMetadata": {
                "promptTokenCount": 2,
                "candidatesTokenCount": 3,
                "thoughtsTokenCount": 4,
                "totalTokenCount": 9,
            },
        }, event=None),
    ])
    chunks = []
    done = []

    with patch.object(GeminiSyncClient, "stream", return_value=events):
        output = list(adapter.stream_chat(
            [UserMessage("hi")],
            buffer_chars=10,
            on_chunk=chunks.append,
            on_done=done.append,
        ))

    assert output == ["Hello"]
    assert chunks[0].usage == Usage(input_tokens=2, output_tokens=7, total_tokens=9)
    assert chunks[0].output_tokens_delta == 7
    assert done[0].usage == Usage(input_tokens=2, output_tokens=7, total_tokens=9)


@pytest.mark.unit
def test_stream_chat_buffers_text_and_orders_chunk_callbacks(adapter):
    events = iter([
        SSEEvent(data={
            "candidates": [{"content": {"parts": [{"text": "He"}]}}],
        }, event=None),
        SSEEvent(data={
            "candidates": [{"content": {"parts": [{"text": "llo"}]}}],
        }, event=None),
        SSEEvent(data={
            "candidates": [{
                "content": {"parts": [{"text": "!"}]},
                "finishReason": "STOP",
            }],
        }, event=None),
    ])
    order = []
    yielded = []

    with patch.object(GeminiSyncClient, "stream", return_value=events):
        for text in adapter.stream_chat(
            [UserMessage("hi")],
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
