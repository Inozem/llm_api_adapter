from dataclasses import FrozenInstanceError

import pytest

from src.llm_api_adapter.models import ChatResponse, ReasoningEvent


@pytest.mark.unit
def test_chat_response_reasoning_events_default_to_an_independent_empty_list():
    first = ChatResponse()
    second = ChatResponse()

    assert first.reasoning_events == []
    assert first.reasoning_events is not second.reasoning_events

    first.reasoning_events.append(
        ReasoningEvent(
            text="summary",
            kind="summary",
            index=0,
            elapsed_s=0.1,
            delta_s=0.1,
        )
    )
    assert second.reasoning_events == []


@pytest.mark.unit
def test_reasoning_event_is_immutable():
    event = ReasoningEvent(
        text="thinking",
        kind="content",
        index=1,
        elapsed_s=1.5,
        delta_s=0.5,
    )

    with pytest.raises(FrozenInstanceError):
        event.text = "changed"


@pytest.mark.unit
@pytest.mark.parametrize("kind", ["invalid", "", None])
def test_reasoning_event_rejects_unknown_kind(kind):
    with pytest.raises(ValueError, match="kind must be 'summary' or 'content'"):
        ReasoningEvent(
            text="thinking",
            kind=kind,
            index=0,
            elapsed_s=0.0,
            delta_s=0.0,
        )
