"""DeepSeek Responses SSE parsing and terminal-state validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from llm_api_adapter.adapters.base_adapter import _StreamState
from llm_api_adapter.errors.llm_api_error import LLMAPIClientError
from llm_api_adapter.llms.streaming import (
    StreamChunkBuffer,
    StreamReasoningCollector,
    StreamUsageTracker,
)
from llm_api_adapter.llms.transports import SSEEvent
from llm_api_adapter.models.responses.chat_response import ChatResponse, Usage


@dataclass
class DeepSeekResponsesStreamState(_StreamState):
    """Request-local state used to reconstruct one DeepSeek response."""

    final_response: Optional[dict[str, Any]] = None
    response_metadata: dict[str, Any] = field(default_factory=dict)
    text_parts: list[str] = field(default_factory=list)
    terminal_event: Optional[str] = None
    terminal_detail: Optional[str] = None
    usage: Optional[Usage] = None


class DeepSeekResponsesStreamParser:
    """Parse semantic Responses events without owning transport resources."""

    TERMINAL_EVENTS = {
        "response.completed",
        "response.incomplete",
        "response.failed",
    }

    @staticmethod
    def new_state(
        *,
        buffer_chars: Optional[int],
        capture_reasoning: bool = False,
    ) -> DeepSeekResponsesStreamState:
        return DeepSeekResponsesStreamState(
            chunk_buffer=StreamChunkBuffer(buffer_chars),
            usage_tracker=StreamUsageTracker(),
            reasoning_collector=(
                StreamReasoningCollector() if capture_reasoning else None
            ),
            reasoning_response=ChatResponse() if capture_reasoning else None,
        )

    @classmethod
    def consume_event(
        cls,
        event: SSEEvent,
        state: DeepSeekResponsesStreamState,
    ) -> Optional[str]:
        """Record one event and return one visible text delta, if present."""
        payload = event.data if isinstance(event.data, Mapping) else {}
        event_type = event.event or payload.get("type")
        if not isinstance(event_type, str):
            return None

        # A Responses terminal event closes the stream state.  Ignore any
        # trailing provider frames rather than allowing a later event to turn
        # an incomplete/failed stream into an apparent success.
        if state.terminal_event is not None:
            return None

        response_data = payload.get("response")
        if isinstance(response_data, Mapping):
            state.response_metadata.update(response_data)
            raw_usage = response_data.get("usage")
            usage = cls._normalize_usage(raw_usage)
            if usage is not None:
                state.usage = usage
                state.usage_tracker.record(state.chunk_buffer, usage)

        if event_type in cls.TERMINAL_EVENTS:
            cls._record_terminal(event_type, payload, state)
            return None

        if event_type == "response.output_text.delta":
            delta = payload.get("delta")
            if delta is None:
                return None
            if not isinstance(delta, str):
                raise LLMAPIClientError(
                    detail="DeepSeek Responses output_text delta must be a string",
                )
            if delta:
                state.text_parts.append(delta)
                return delta
        return None

    @classmethod
    def finalize(
        cls,
        state: DeepSeekResponsesStreamState,
        *,
        model: str,
        capture_reasoning: bool = False,
    ) -> ChatResponse:
        """Build a response only after a successful completed terminal event."""
        if state.terminal_event != "response.completed":
            detail = state.terminal_detail or (
                "DeepSeek Responses stream ended without response.completed"
            )
            raise LLMAPIClientError(detail=detail)

        response = state.final_response
        if response is None or response.get("object") != "response":
            raise LLMAPIClientError(
                detail=(
                    "DeepSeek Responses completed event must include a response "
                    "object"
                ),
            )
        if not isinstance(response.get("output"), list):
            raise LLMAPIClientError(
                detail="DeepSeek Responses API response.output must be an array",
            )
        if response.get("status") not in {None, "completed"}:
            raise LLMAPIClientError(
                detail="DeepSeek Responses completed event has a non-completed status",
            )

        completed_response = dict(response)
        completed_response.setdefault("model", model)
        completed_response.setdefault("status", "completed")
        return ChatResponse.from_openai_responses_response(
            completed_response,
            capture_reasoning=capture_reasoning,
        )

    @classmethod
    def _record_terminal(
        cls,
        event_type: str,
        payload: Mapping[str, Any],
        state: DeepSeekResponsesStreamState,
    ) -> None:
        if state.terminal_event is not None:
            return
        response_data = payload.get("response")
        if not isinstance(response_data, Mapping):
            state.terminal_event = event_type
            state.terminal_detail = (
                f"DeepSeek {event_type} event did not include a response object"
            )
            return

        state.terminal_event = event_type
        state.final_response = dict(response_data)
        if event_type == "response.incomplete":
            state.terminal_detail = (
                "DeepSeek Responses stream ended incomplete; partial output is not "
                "a completed response"
            )
        elif event_type == "response.failed":
            state.terminal_detail = cls._failed_detail(response_data)

    @staticmethod
    def _failed_detail(response_data: Mapping[str, Any]) -> str:
        error = response_data.get("error")
        if isinstance(error, Mapping):
            message = error.get("message") or error.get("code")
            if message:
                return f"DeepSeek Responses stream failed: {message}"
        return "DeepSeek Responses stream failed"

    @staticmethod
    def _normalize_usage(raw_usage: Any) -> Optional[Usage]:
        if not isinstance(raw_usage, Mapping):
            return None
        values = [
            raw_usage.get("input_tokens"),
            raw_usage.get("output_tokens"),
            raw_usage.get("total_tokens"),
        ]
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values
        ):
            return None
        return Usage(
            input_tokens=values[0],
            output_tokens=values[1],
            total_tokens=values[2],
        )


__all__ = [
    "DeepSeekResponsesStreamParser",
    "DeepSeekResponsesStreamState",
]
