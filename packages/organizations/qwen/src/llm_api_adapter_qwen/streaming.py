"""Qwen Messages SSE normalization without transport or callback ownership."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from llm_api_adapter.adapters.anthropic_compatible.streaming import (
    MessagesBlockState,
    consume_messages_event,
    finalize_messages_stream,
)
from llm_api_adapter.adapters.anthropic_compatible.usage import (
    merge_messages_usage_snapshots,
)
from llm_api_adapter.adapters.base_adapter import _StreamState
from llm_api_adapter.errors.llm_api_error import (
    InvalidToolArgumentsError,
    LLMAPIClientError,
)
from llm_api_adapter.llms.streaming import (
    StreamChunkBuffer,
    StreamReasoningCollector,
    StreamUsageTracker,
)
from llm_api_adapter.llms.transports import SSEEvent
from llm_api_adapter.models.responses.chat_response import ChatResponse, Usage

from .usage import normalize_qwen_accounted_usage


@dataclass
class QwenMessagesStreamState(_StreamState):
    """State needed to reconstruct one Messages response from SSE events."""

    message_data: dict[str, Any] = field(default_factory=dict)
    content_blocks: dict[int, dict[str, Any]] = field(default_factory=dict)
    input_json_fragments: dict[int, list[str]] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)
    message_delta: dict[str, Any] = field(default_factory=dict)
    policy_accounting: bool = False
    pdf_requested: bool = False
    _messages_state: MessagesBlockState = field(
        default_factory=MessagesBlockState,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        self._messages_state.message = self.message_data
        self._messages_state.blocks = self.content_blocks
        self._messages_state.argument_fragments = self.input_json_fragments
        self._messages_state.usage = self.usage
        self._messages_state.message_delta = self.message_delta


class QwenMessagesStreamParser:
    """Accumulate the documented Anthropic-compatible Qwen SSE event trace."""

    @staticmethod
    def new_state(
        *,
        buffer_chars: Optional[int],
        capture_reasoning: bool,
        policy_accounting: bool = False,
    ) -> QwenMessagesStreamState:
        """Create fresh shared lifecycle state for one Qwen stream."""
        return QwenMessagesStreamState(
            message_data={"content": []},
            chunk_buffer=StreamChunkBuffer(buffer_chars),
            usage_tracker=StreamUsageTracker(),
            reasoning_collector=(
                StreamReasoningCollector() if capture_reasoning else None
            ),
            reasoning_response=ChatResponse() if capture_reasoning else None,
            policy_accounting=policy_accounting,
        )

    @classmethod
    def consume_event(
        cls,
        event: SSEEvent,
        state: QwenMessagesStreamState,
    ) -> tuple[Optional[str], Optional[str]]:
        """Record one event and return visible and thinking deltas separately."""
        if not isinstance(event, SSEEvent):
            return None, None
        cls._sync_compat_state(state)
        previous_usage = dict(state.usage)
        adapted = cls._adapt_event(
            event, preserve_malformed_usage=state.policy_accounting
        )
        try:
            visible, thinking = consume_messages_event(
                state._messages_state,
                adapted,
                event_policy=cls._event_is_usable,
                block_policy=cls._prepare_block,
                delta_policy=cls._delta_is_usable,
                index_policy=cls._index_policy,
            )
        except InvalidToolArgumentsError as error:
            detail = error.detail or "invalid tool arguments"
            index = (
                event.data.get("index")
                if isinstance(event.data, Mapping)
                else None
            )
            block = (
                state.content_blocks.get(index)
                if isinstance(index, int) and not isinstance(index, bool)
                else None
            )
            tool_name = block.get("name") if isinstance(block, Mapping) else None
            if "must be a JSON object" in detail:
                detail = (
                    "Qwen tool input must decode to an object "
                    f"for tool={tool_name!r}"
                )
            else:
                cause = error.__cause__
                parse_detail = str(cause) if cause is not None else detail
                detail = (
                    "Qwen tool input JSON parse failed "
                    f"for tool={tool_name!r}: {parse_detail}"
                )
            raise InvalidToolArgumentsError(detail=detail) from error
        state.message_data = state._messages_state.message
        state.content_blocks = state._messages_state.blocks
        state.input_json_fragments = state._messages_state.argument_fragments
        merged_usage = (
            dict(state._messages_state.usage)
            if isinstance(state._messages_state.usage, Mapping)
            else previous_usage
        )
        if state.policy_accounting and state._messages_state.usage_snapshots:
            merged_usage = merge_messages_usage_snapshots(
                state._messages_state.usage_snapshots
            ) or {}
        state.usage.clear()
        state.usage.update(merged_usage)
        state._messages_state.usage = state.usage
        state.message_delta = state._messages_state.message_delta
        normalized = cls._normalize_usage(
            state.usage, policy_accounting=state.policy_accounting
        )
        if state.policy_accounting and state._messages_state.usage_snapshots and normalized is None:
            state.chunk_buffer.update_metadata(usage=Usage(None, None, None, None, None))
        else:
            state.usage_tracker.record(state.chunk_buffer, normalized)
        return visible or None, thinking or None

    @staticmethod
    def finalize(
        state: QwenMessagesStreamState,
        *,
        model: str,
    ) -> dict[str, Any]:
        """Rebuild an Anthropic-compatible terminal message response."""
        QwenMessagesStreamParser._sync_compat_state(state)
        response = finalize_messages_stream(
            state._messages_state,
            validate_completion=lambda _state: None,
        )
        response["model"] = response.get("model") or model
        if state.pdf_requested:
            response["_qwen_pdf_requested"] = True
        if state.policy_accounting and state._messages_state.usage_snapshots:
            merged = merge_messages_usage_snapshots(
                state._messages_state.usage_snapshots
            )
            if merged is not None:
                response["usage"] = merged
            else:
                response.pop("usage", None)
        elif state.usage:
            response["usage"] = dict(state.usage)
        elif "usage" in state.message_data:
            response["usage"] = state.message_data["usage"]
        return response

    @staticmethod
    def _sync_compat_state(state: QwenMessagesStreamState) -> None:
        """Keep the historical public state attributes writable by callers."""
        state._messages_state.message = state.message_data
        state._messages_state.blocks = state.content_blocks
        state._messages_state.argument_fragments = state.input_json_fragments
        state._messages_state.usage = state.usage
        state._messages_state.message_delta = state.message_delta

    @staticmethod
    def _adapt_event(event: Any, *, preserve_malformed_usage: bool = False) -> Any:
        """Drop malformed optional usage snapshots as the native parser did."""
        if not isinstance(event.data, Mapping):
            return event
        payload = dict(event.data)
        kind = event.event or payload.get("type")
        if (kind == "message_delta" and not preserve_malformed_usage
                and not isinstance(payload.get("usage"), Mapping)):
            payload.pop("usage", None)
        if kind == "message_delta" and not isinstance(payload.get("delta"), Mapping):
            payload["delta"] = {}
        return SSEEvent(event=event.event or None, data=payload, done=event.done)

    @staticmethod
    def _event_is_usable(event: Any) -> bool:
        payload = event.data
        if not isinstance(payload, Mapping):
            return False
        kind = event.event or payload.get("type")
        if not isinstance(kind, str):
            return False
        if kind == "message_start":
            return isinstance(payload.get("message"), Mapping)
        if kind in {
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
        }:
            index = payload.get("index")
            if isinstance(index, bool) or not isinstance(index, int):
                return False
            if kind == "content_block_start":
                return isinstance(payload.get("content_block"), Mapping)
            if kind == "content_block_delta":
                return isinstance(payload.get("delta"), Mapping)
        if kind == "message_delta":
            return True
        return True

    @staticmethod
    def _prepare_block(index: int, block: dict[str, Any]) -> dict[str, Any]:
        _ = index
        if block.get("type") == "text" and not isinstance(
            block.get("text"), str
        ):
            block["text"] = ""
        elif block.get("type") == "thinking" and not isinstance(
            block.get("thinking"), str
        ):
            block["thinking"] = ""
        elif block.get("type") == "tool_use":
            if not isinstance(block.get("input"), Mapping):
                block["input"] = {}
            else:
                block["input"] = dict(block["input"])
        return block

    @staticmethod
    def _delta_is_usable(block: dict[str, Any], delta: Mapping[str, Any]) -> bool:
        kind = delta.get("type")
        if kind == "text_delta":
            return block.get("type") == "text" and isinstance(delta.get("text"), str)
        if kind == "thinking_delta":
            return block.get("type") == "thinking" and isinstance(
                delta.get("thinking"), str
            )
        if kind == "input_json_delta":
            value = delta.get("partial_json")
            return (
                block.get("type") == "tool_use"
                and isinstance(value, str)
                and bool(value)
            )
        return False

    @staticmethod
    def _index_policy(index: int, kind: str, state: MessagesBlockState) -> bool | str:
        if kind == "content_block_start":
            return "replace" if index in state.blocks else True
        return index in state.blocks

    @staticmethod
    def _normalize_usage(raw_usage: Mapping[str, Any], *, policy_accounting: bool = False) -> Optional[Usage]:
        if policy_accounting:
            return normalize_qwen_accounted_usage(raw_usage)
        input_tokens = QwenMessagesStreamParser._token_count(
            raw_usage.get("input_tokens"),
        )
        output_tokens = QwenMessagesStreamParser._token_count(
            raw_usage.get("output_tokens"),
        )
        cache_read_tokens = None
        if "cache_read_input_tokens" in raw_usage:
            cache_read_tokens = QwenMessagesStreamParser._token_count(
                raw_usage.get("cache_read_input_tokens"),
            )
            if cache_read_tokens is None:
                raise LLMAPIClientError(
                    detail=(
                        "Qwen cache_read_input_tokens must be a "
                        "non-negative integer"
                    ),
                )
        if input_tokens is not None and cache_read_tokens is not None:
            input_tokens += cache_read_tokens
        if (
            input_tokens is None
            and output_tokens is None
            and cache_read_tokens is None
        ):
            return None
        return Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=(
                input_tokens + output_tokens
                if input_tokens is not None and output_tokens is not None
                else None
            ),
            cached_tokens=cache_read_tokens,
        )

    @staticmethod
    def _token_count(value: Any) -> Optional[int]:
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
        return None


__all__ = ["QwenMessagesStreamParser", "QwenMessagesStreamState"]
