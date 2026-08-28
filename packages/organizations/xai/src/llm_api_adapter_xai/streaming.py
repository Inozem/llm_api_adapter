"""Pure Responses SSE parsing for the xAI organization package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from llm_api_adapter.adapters.base_adapter import _StreamState
from llm_api_adapter.llms.streaming import StreamChunkBuffer, StreamUsageTracker
from llm_api_adapter.llms.transports import SSEEvent
from llm_api_adapter.models.responses.chat_response import ChatResponse, Usage


@dataclass
class XAIResponsesStreamState(_StreamState):
    """Accumulated state used to reconstruct an xAI Responses result."""

    final_response: Optional[dict[str, Any]] = None
    response_metadata: Dict[str, Any] = field(default_factory=dict)
    text_parts: list[str] = field(default_factory=list)
    output_items: Dict[int, dict[str, Any]] = field(default_factory=dict)
    usage: Optional[dict[str, Any]] = None
    next_output_index: int = 0


class XAIResponsesStreamParser:
    """Parse xAI Responses events without owning callbacks or generators."""

    @staticmethod
    def new_state(*, buffer_chars: Optional[int]) -> XAIResponsesStreamState:
        """Create a state that uses the shared chunk and usage primitives."""
        return XAIResponsesStreamState(
            chunk_buffer=StreamChunkBuffer(buffer_chars),
            usage_tracker=StreamUsageTracker(),
            reasoning_collector=None,
            reasoning_response=None,
        )

    @classmethod
    def consume_event(
        cls,
        event: SSEEvent,
        state: XAIResponsesStreamState,
    ) -> Optional[str]:
        """Record one event and return its visible text delta, if any."""
        payload = event.data if isinstance(event.data, Mapping) else {}
        event_type = event.event or payload.get("type")
        response_data = payload.get("response")
        if isinstance(response_data, Mapping):
            state.response_metadata.update(response_data)

        raw_usage = cls._event_usage(payload, response_data)
        if isinstance(raw_usage, Mapping):
            state.usage = dict(raw_usage)
        state.usage_tracker.record(
            state.chunk_buffer,
            cls._normalize_usage(raw_usage),
        )

        if event_type == "response.output_text.delta":
            delta = payload.get("delta")
            if isinstance(delta, str) and delta:
                state.text_parts.append(delta)
                return delta
            return None

        if event_type == "response.output_item.done":
            item = payload.get("item")
            if isinstance(item, Mapping):
                cls._record_output_item(payload, item, state)
            return None

        if event_type == "response.completed" and isinstance(response_data, Mapping):
            state.final_response = dict(response_data)
        return None

    @classmethod
    def finalize(
        cls,
        state: XAIResponsesStreamState,
        *,
        model: str,
    ) -> ChatResponse:
        """Build the shared response type from a completed or partial stream."""
        response = state.final_response or cls._build_response(state, model=model)
        return ChatResponse.from_openai_responses_response(response)

    @staticmethod
    def _event_usage(payload: Mapping[str, Any], response_data: Any) -> Any:
        if isinstance(payload.get("usage"), Mapping):
            return payload["usage"]
        if isinstance(response_data, Mapping):
            return response_data.get("usage")
        return None

    @staticmethod
    def _normalize_usage(raw_usage: Any) -> Optional[Usage]:
        if not isinstance(raw_usage, Mapping):
            return None

        input_tokens = XAIResponsesStreamParser._token_count(
            raw_usage.get("input_tokens"),
        )
        output_tokens = XAIResponsesStreamParser._token_count(
            raw_usage.get("output_tokens"),
        )
        total_tokens = XAIResponsesStreamParser._token_count(
            raw_usage.get("total_tokens"),
        )
        if input_tokens is None and output_tokens is None and total_tokens is None:
            return None
        return Usage(
            input_tokens=input_tokens or 0,
            output_tokens=output_tokens or 0,
            total_tokens=total_tokens or 0,
        )

    @staticmethod
    def _token_count(value: Any) -> Optional[int]:
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
        return None

    @staticmethod
    def _record_output_item(
        payload: Mapping[str, Any],
        item: Mapping[str, Any],
        state: XAIResponsesStreamState,
    ) -> None:
        output_index = payload.get("output_index")
        if isinstance(output_index, int) and not isinstance(output_index, bool):
            index = output_index
            state.next_output_index = max(state.next_output_index, index + 1)
        else:
            index = state.next_output_index
            state.next_output_index += 1
        state.output_items[index] = dict(item)

    @classmethod
    def _build_response(
        cls,
        state: XAIResponsesStreamState,
        *,
        model: str,
    ) -> dict[str, Any]:
        output = [state.output_items[index] for index in sorted(state.output_items)]
        if state.text_parts and not cls._has_complete_text_item(output):
            output.insert(
                0,
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "".join(state.text_parts)}
                    ],
                },
            )
        response = {
            **state.response_metadata,
            "object": state.response_metadata.get("object", "response"),
            "model": state.response_metadata.get("model", model),
            "output": output,
            "status": state.response_metadata.get("status", "completed"),
        }
        if state.usage is not None:
            response["usage"] = state.usage
        return response

    @staticmethod
    def _has_complete_text_item(output: list[dict[str, Any]]) -> bool:
        for item in output:
            if item.get("type") != "message":
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            if any(
                isinstance(part, Mapping)
                and part.get("type") in {"output_text", "text"}
                and isinstance(part.get("text"), str)
                for part in content
            ):
                return True
        return False


__all__ = ["XAIResponsesStreamParser", "XAIResponsesStreamState"]
