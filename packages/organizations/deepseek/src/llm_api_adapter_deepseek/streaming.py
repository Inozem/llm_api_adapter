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
    output_items: dict[int, dict[str, Any]] = field(default_factory=dict)
    output_item_indexes: dict[str, int] = field(default_factory=dict)
    function_argument_parts: dict[int, list[str]] = field(default_factory=dict)
    terminal_event: Optional[str] = None
    terminal_detail: Optional[str] = None
    usage: Optional[Usage] = None
    next_output_index: int = 0


class DeepSeekResponsesStreamParser:
    """Parse semantic Responses events without owning transport resources."""

    TERMINAL_EVENTS = {
        "response.completed",
        "response.incomplete",
        "response.failed",
        "response.cancelled",
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

        raw_usage = cls._event_usage(payload, response_data)
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
                cls._record_output_text(
                    payload,
                    delta,
                    state,
                    authoritative=False,
                )
                return delta
        elif event_type == "response.output_text.done":
            text = payload.get("text")
            if text is None:
                return None
            if not isinstance(text, str):
                raise LLMAPIClientError(
                    detail="DeepSeek Responses output_text done text must be a string",
                )
            current_text = "".join(state.text_parts)
            if not current_text:
                state.text_parts.append(text)
            elif current_text != text:
                state.text_parts[:] = [text]
            cls._record_output_text(
                payload,
                text,
                state,
                authoritative=True,
            )
            if not current_text:
                return text or None
        elif event_type == "response.output_item.added":
            cls._record_output_item(
                payload,
                payload.get("item"),
                state,
                authoritative=False,
            )
        elif event_type == "response.output_item.done":
            cls._record_output_item(
                payload,
                payload.get("item"),
                state,
                authoritative=True,
            )
        elif event_type in {
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
        }:
            cls._record_function_arguments(event_type, payload, state)
        elif event_type in {
            "response.reasoning_summary_text.delta",
            "response.reasoning_summary_text.done",
            "response.reasoning_text.delta",
            "response.reasoning_text.done",
        }:
            cls._record_reasoning_text(event_type, payload, state)
        return None

    @staticmethod
    def reasoning_delta(event: SSEEvent) -> Optional[tuple[str, str]]:
        """Return one visible reasoning fragment and its Core event kind."""
        payload = event.data if isinstance(event.data, Mapping) else {}
        event_type = event.event or payload.get("type")
        if event_type not in {
            "response.reasoning_summary_text.delta",
            "response.reasoning_text.delta",
        }:
            return None
        delta = payload.get("delta")
        if not isinstance(delta, str) or not delta:
            return None
        kind = (
            "summary"
            if event_type == "response.reasoning_summary_text.delta"
            else "content"
        )
        return delta, kind

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

        terminal_response = state.final_response
        if (
            not isinstance(terminal_response, Mapping)
            or terminal_response.get("object") != "response"
        ):
            raise LLMAPIClientError(
                detail=(
                    "DeepSeek Responses completed event must include a response "
                    "object"
                ),
            )
        if not isinstance(terminal_response.get("output"), list):
            raise LLMAPIClientError(
                detail="DeepSeek Responses API response.output must be an array",
            )

        response = cls._materialize_response(state, model=model)
        state.final_response = response
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
            if event_type == "response.failed":
                state.terminal_detail = cls._failed_detail({}, payload.get("error"))
            else:
                state.terminal_detail = (
                    f"DeepSeek {event_type} event did not include a response object"
                )
            return

        state.terminal_event = event_type
        state.final_response = dict(response_data)
        if event_type == "response.incomplete":
            details = response_data.get("incomplete_details")
            reason = details.get("reason") if isinstance(details, Mapping) else None
            suffix = f": {reason}" if isinstance(reason, str) and reason else ""
            state.terminal_detail = (
                "DeepSeek Responses stream ended incomplete; partial output is not "
                f"a completed response{suffix}"
            )
        elif event_type == "response.failed":
            state.terminal_detail = cls._failed_detail(
                response_data,
                payload.get("error"),
            )
        elif event_type == "response.cancelled":
            state.terminal_detail = (
                "DeepSeek Responses stream was cancelled; partial output is not a "
                "completed response"
            )

    @staticmethod
    def _event_usage(
        payload: Mapping[str, Any],
        response_data: Any,
    ) -> Any:
        if isinstance(payload.get("usage"), Mapping):
            return payload["usage"]
        if isinstance(response_data, Mapping):
            return response_data.get("usage")
        return None

    @classmethod
    def _record_output_item(
        cls,
        payload: Mapping[str, Any],
        raw_item: Any,
        state: DeepSeekResponsesStreamState,
        *,
        authoritative: bool,
    ) -> None:
        if not isinstance(raw_item, Mapping):
            return
        index = cls._output_index(payload, raw_item, state)
        item = dict(raw_item)
        item_id = item.get("id") or payload.get("item_id")
        if isinstance(item_id, str) and item_id:
            state.output_item_indexes[item_id] = index

        # ``output_item.done`` is authoritative for an item.  Replacing it
        # avoids duplicating reasoning text already received as deltas while
        # still retaining fragments when a provider sends only a partial item.
        if authoritative:
            existing = state.output_items.get(index)
            if existing and item.get("type") == "function_call":
                if not item.get("arguments"):
                    item["arguments"] = existing.get("arguments", "")
            state.output_items[index] = item
        else:
            existing = state.output_items.setdefault(index, {})
            existing.update(item)

    @classmethod
    def _record_function_arguments(
        cls,
        event_type: str,
        payload: Mapping[str, Any],
        state: DeepSeekResponsesStreamState,
    ) -> None:
        raw_delta = payload.get("delta")
        raw_arguments = payload.get("arguments")
        if event_type.endswith(".delta"):
            if raw_delta is None:
                return
            if not isinstance(raw_delta, str):
                raise LLMAPIClientError(
                    detail=(
                        "DeepSeek Responses function_call_arguments delta must be "
                        "a string"
                    ),
                )
            text = raw_delta
        else:
            if raw_arguments is None:
                return
            if not isinstance(raw_arguments, str):
                raise LLMAPIClientError(
                    detail=(
                        "DeepSeek Responses function_call_arguments must be a "
                        "string"
                    ),
                )
            text = raw_arguments

        index = cls._output_index(payload, {}, state)
        item = state.output_items.setdefault(index, {"type": "function_call"})
        item.setdefault("type", "function_call")
        parts = state.function_argument_parts.setdefault(index, [])
        if event_type.endswith(".done"):
            parts[:] = [text]
        else:
            parts.append(text)
        item["arguments"] = "".join(parts)

        item_id = payload.get("item_id")
        if isinstance(item_id, str) and item_id:
            state.output_item_indexes[item_id] = index

    @classmethod
    def _record_output_text(
        cls,
        payload: Mapping[str, Any],
        text: str,
        state: DeepSeekResponsesStreamState,
        *,
        authoritative: bool,
    ) -> None:
        raw_index = payload.get("output_index")
        if not isinstance(raw_index, int) or isinstance(raw_index, bool):
            return
        state.next_output_index = max(state.next_output_index, raw_index + 1)
        item = state.output_items.setdefault(raw_index, {"type": "message"})
        if item.get("type") != "message":
            return
        content = item.setdefault("content", [])
        if not isinstance(content, list):
            content = []
            item["content"] = content
        text_parts = [
            part
            for part in content
            if isinstance(part, dict) and part.get("type") == "output_text"
        ]
        if authoritative and text_parts:
            text_parts[0]["text"] = text
            return
        if text_parts:
            current = text_parts[0].get("text")
            if isinstance(current, str):
                text_parts[0]["text"] = current + text
                return
        content.append({"type": "output_text", "text": text})

    @classmethod
    def _record_reasoning_text(
        cls,
        event_type: str,
        payload: Mapping[str, Any],
        state: DeepSeekResponsesStreamState,
    ) -> None:
        text = payload.get("delta")
        if event_type.endswith(".done"):
            text = payload.get("text", payload.get("delta"))
        if text is None:
            return
        if not isinstance(text, str):
            raise LLMAPIClientError(
                detail="DeepSeek Responses reasoning text must be a string",
            )
        if not text:
            return
        index = cls._output_index(payload, {}, state)
        item = state.output_items.setdefault(index, {"type": "reasoning"})
        item.setdefault("type", "reasoning")
        kind = (
            "summary"
            if "summary" in event_type
            else "content"
        )
        key = "summary" if kind == "summary" else "content"
        part_type = "summary_text" if kind == "summary" else "reasoning_text"
        parts = item.setdefault(key, [])
        if not isinstance(parts, list):
            parts = []
            item[key] = parts
        if event_type.endswith(".done") and parts:
            # A done event contains the authoritative aggregate text.
            parts[:] = [{"type": part_type, "text": text}]
        else:
            parts.append({"type": part_type, "text": text})

    @classmethod
    def _output_index(
        cls,
        payload: Mapping[str, Any],
        item: Mapping[str, Any],
        state: DeepSeekResponsesStreamState,
    ) -> int:
        raw_index = payload.get("output_index")
        if not isinstance(raw_index, int) or isinstance(raw_index, bool):
            raw_index = item.get("output_index")
        if isinstance(raw_index, int) and not isinstance(raw_index, bool):
            state.next_output_index = max(state.next_output_index, raw_index + 1)
            return raw_index
        item_id = payload.get("item_id") or item.get("id")
        if isinstance(item_id, str) and item_id in state.output_item_indexes:
            return state.output_item_indexes[item_id]
        index = state.next_output_index
        state.next_output_index += 1
        return index

    @classmethod
    def _materialize_response(
        cls,
        state: DeepSeekResponsesStreamState,
        *,
        model: str,
    ) -> Optional[dict[str, Any]]:
        response_source = (
            state.final_response
            if state.final_response is not None
            else state.response_metadata
        )
        response = dict(response_source)
        if not response and not state.output_items and not state.text_parts:
            return None
        raw_output = response.get("output")
        output = (
            [dict(item) for item in raw_output if isinstance(item, Mapping)]
            if isinstance(raw_output, list)
            else []
        )

        # Merge incremental items into a terminal response by output index.  A
        # terminal item wins for opaque fields, while accumulated arguments and
        # reasoning text fill gaps in truncated terminal payloads.
        for index in sorted(state.output_items):
            item = state.output_items[index]
            if (
                index < len(output)
                and isinstance(output[index], Mapping)
                and output[index].get("type") == item.get("type")
            ):
                merged = dict(item)
                merged.update(output[index])
                if item.get("type") == "function_call" and not merged.get("arguments"):
                    merged["arguments"] = item.get("arguments", "")
                output[index] = merged
            else:
                output.append(dict(item))

        if state.text_parts and not cls._has_complete_text_item(output):
            output.insert(
                0,
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "".join(state.text_parts)}
                    ],
                },
            )
        response.update(
            {
                "object": response.get("object", "response"),
                "model": response.get("model", model),
                "output": output,
                "status": response.get("status", "completed"),
            }
        )
        if state.usage is not None:
            response["usage"] = {
                "input_tokens": state.usage.input_tokens,
                "output_tokens": state.usage.output_tokens,
                "total_tokens": state.usage.total_tokens,
            }
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

    @staticmethod
    def _failed_detail(
        response_data: Mapping[str, Any],
        raw_event_error: Any = None,
    ) -> str:
        error = response_data.get("error")
        if not isinstance(error, Mapping):
            error = raw_event_error
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
