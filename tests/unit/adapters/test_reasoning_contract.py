from unittest.mock import patch

import pytest

from src.llm_api_adapter.adapters.anthropic_adapter import (
    AnthropicAdapter,
    ClaudeSyncClient,
)
from src.llm_api_adapter.adapters.google_adapter import (
    GeminiSyncClient,
    GoogleAdapter,
)
from src.llm_api_adapter.adapters.openai_adapter import (
    OpenAIAdapter,
    OpenAISyncClient,
)
from src.llm_api_adapter.llms.streaming import SSEEvent
from src.llm_api_adapter.models.messages.chat_message import UserMessage


def _openai_events(*, include_reasoning=True):
    events = []
    if include_reasoning:
        events.extend([
            SSEEvent(
                event="response.reasoning_summary_text.delta",
                data={
                    "type": "response.reasoning_summary_text.delta",
                    "delta": "Plan",
                },
            ),
            SSEEvent(
                event="response.reasoning_text.delta",
                data={
                    "type": "response.reasoning_text.delta",
                    "delta": "Details",
                },
            ),
        ])
    events.extend([
        SSEEvent(
            event="response.output_text.delta",
            data={"type": "response.output_text.delta", "delta": "Answer"},
        ),
        SSEEvent(
            event="response.completed",
            data={
                "type": "response.completed",
                "response": {
                    "id": "resp_123",
                    "model": "gpt-5",
                    "output": [{
                        "type": "message",
                        "content": [{"type": "output_text", "text": "Answer"}],
                    }],
                },
            },
        ),
    ])
    return iter(events)


def _anthropic_events(*, include_reasoning=True):
    events = [
        SSEEvent(
            event="message_start",
            data={
                "type": "message_start",
                "message": {"model": "claude-sonnet-4-5", "content": []},
            },
        ),
    ]
    if include_reasoning:
        events.extend([
            SSEEvent(
                event="content_block_start",
                data={
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "thinking", "thinking": ""},
                },
            ),
            SSEEvent(
                event="content_block_delta",
                data={
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "thinking_delta", "thinking": "Plan"},
                },
            ),
            SSEEvent(
                event="content_block_stop",
                data={"type": "content_block_stop", "index": 0},
            ),
        ])
    text_index = 1 if include_reasoning else 0
    events.extend([
        SSEEvent(
            event="content_block_start",
            data={
                "type": "content_block_start",
                "index": text_index,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        SSEEvent(
            event="content_block_delta",
            data={
                "type": "content_block_delta",
                "index": text_index,
                "delta": {"type": "text_delta", "text": "Answer"},
            },
        ),
        SSEEvent(
            event="content_block_stop",
            data={"type": "content_block_stop", "index": text_index},
        ),
        SSEEvent(event="message_stop", data={"type": "message_stop"}),
    ])
    return iter(events)


def _google_events(*, include_reasoning=True):
    parts = []
    if include_reasoning:
        parts.append({"text": "Plan", "thought": True})
    parts.append({"text": "Answer"})
    return iter([
        SSEEvent(
            event=None,
            data={
                "candidates": [{
                    "content": {"parts": parts},
                    "finishReason": "STOP",
                }],
            },
        ),
    ])


PROVIDER_CASES = [
    pytest.param(
        lambda: OpenAIAdapter(api_key="test_api_key", model="gpt-5"),
        OpenAISyncClient,
        _openai_events,
        {},
        [("Plan", "summary"), ("Details", "content")],
        id="openai-responses",
    ),
    pytest.param(
        lambda: AnthropicAdapter(
            api_key="test_api_key",
            model="claude-sonnet-4-5",
        ),
        ClaudeSyncClient,
        _anthropic_events,
        {"max_tokens": 2048, "reasoning_level": 1024},
        [("Plan", "summary")],
        id="anthropic-messages",
    ),
    pytest.param(
        lambda: GoogleAdapter(api_key="test_api_key", model="gemini-2.5-pro"),
        GeminiSyncClient,
        _google_events,
        {},
        [("Plan", "summary")],
        id="google-generate-content",
    ),
]


@pytest.mark.unit
@pytest.mark.parametrize("capture_reasoning", [True, False])
@pytest.mark.parametrize(
    "adapter_factory, client_class, event_factory, stream_kwargs, expected_reasoning",
    PROVIDER_CASES,
)
def test_stream_reasoning_contract_is_provider_neutral(
    capture_reasoning,
    adapter_factory,
    client_class,
    event_factory,
    stream_kwargs,
    expected_reasoning,
):
    adapter = adapter_factory()
    reasoning = []
    deltas = []
    done = []
    callback_order = []

    with patch.object(
        client_class,
        "stream",
        return_value=event_factory(include_reasoning=True),
    ):
        output = list(
            adapter.stream_chat(
                [UserMessage("hi")],
                capture_reasoning=capture_reasoning,
                on_delta=lambda text: (
                    deltas.append(text), callback_order.append(("delta", text))
                ),
                on_reasoning=lambda event: (
                    reasoning.append(event), callback_order.append(("reasoning", event.text))
                ),
                on_done=lambda response: (
                    done.append(response), callback_order.append(("done", response.content))
                ),
                **stream_kwargs,
            )
        )

    expected_events = expected_reasoning if capture_reasoning else []
    assert output == ["Answer"]
    assert deltas == ["Answer"]
    assert [(event.text, event.kind) for event in reasoning] == expected_events
    assert [event.index for event in reasoning] == list(range(len(reasoning)))
    assert all(event.elapsed_s >= 0.0 and event.delta_s >= 0.0 for event in reasoning)
    assert callback_order[-2:] == [("delta", "Answer"), ("done", "Answer")]
    assert len(done) == 1
    assert done[0].reasoning_events == reasoning


@pytest.mark.unit
@pytest.mark.parametrize(
    "adapter_factory, client_class, event_factory, stream_kwargs, expected_reasoning",
    PROVIDER_CASES,
)
def test_stream_reasoning_contract_accepts_absent_provider_data(
    adapter_factory,
    client_class,
    event_factory,
    stream_kwargs,
    expected_reasoning,
):
    adapter = adapter_factory()
    reasoning = []
    done = []

    with patch.object(
        client_class,
        "stream",
        return_value=event_factory(include_reasoning=False),
    ):
        output = list(
            adapter.stream_chat(
                [UserMessage("hi")],
                capture_reasoning=True,
                on_reasoning=reasoning.append,
                on_done=done.append,
                **stream_kwargs,
            )
        )

    assert output == ["Answer"]
    assert reasoning == []
    assert done[0].reasoning_events == []
