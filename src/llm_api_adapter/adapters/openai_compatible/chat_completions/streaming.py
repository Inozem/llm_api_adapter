"""Shared Chat Completions stream consumers."""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, Iterator, List, Mapping, Optional

from ...base_adapter import (
    AsyncOnChunk,
    AsyncOnDelta,
    AsyncOnDone,
    AsyncOnToolCall,
    OnChunk,
    OnDelta,
    OnDone,
    OnToolCall,
)
from ....llms.streaming import StreamChunkBuffer, StreamUsageTracker
from ....models.responses.chat_response import ChatResponse


class _ChatCompletionsStreamingMixin:
    """Consume Chat Completions events through the common stream lifecycle."""

    async def _consume_chat_completions_stream_async(
        self,
        events: AsyncIterator[Any],
        effective_schema: Optional[dict],
        response_model: Optional[Any],
        on_delta: Optional[AsyncOnDelta],
        on_tool_call: Optional[AsyncOnToolCall],
        on_done: Optional[AsyncOnDone],
        buffer_chars: Optional[int],
        on_chunk: Optional[AsyncOnChunk],
    ) -> AsyncIterator[str]:
        text_parts: List[str] = []
        refusal_parts: List[str] = []
        tool_calls: Dict[int, Dict[str, Any]] = {}
        legacy_response: Dict[str, Any] = {"model": self.model, "choices": []}
        finish_reason: Optional[str] = None
        chunk_buffer = StreamChunkBuffer(buffer_chars)
        usage_tracker = StreamUsageTracker()
        async for event in self._aiter_provider_stream_events(events):
            self._validate_chat_completions_stream_event(event)
            payload = event.data if isinstance(event.data, Mapping) else {}
            self._update_legacy_stream_metadata(payload, legacy_response)
            usage_tracker.record(
                chunk_buffer,
                self._normalize_stream_usage(
                    payload.get("usage"),
                    input_field="prompt_tokens",
                    output_field="completion_tokens",
                ),
            )
            choices = payload.get("choices")
            if not isinstance(choices, list):
                continue
            for choice in choices:
                text, choice_finish_reason = self._consume_legacy_stream_choice(
                    choice,
                    text_parts,
                    refusal_parts,
                    tool_calls,
                )
                if text is not None:
                    async for emitted_text in self._emit_async_stream_chunks(
                        chunk_buffer.add(text),
                        on_chunk,
                        on_delta,
                    ):
                        yield emitted_text
                if choice_finish_reason is not None:
                    finish_reason = choice_finish_reason
        chat_response = self._build_legacy_stream_response(
            legacy_response,
            text_parts,
            refusal_parts,
            tool_calls,
            finish_reason,
        )
        chat_response = self._finalize_stream_response(
            chat_response,
            effective_schema=effective_schema,
            response_model=response_model,
        )
        async for text in self._complete_async_stream(
            chat_response,
            chunk_buffer,
            on_chunk,
            on_delta,
            on_tool_call,
            on_done,
        ):
            yield text

    def _consume_chat_completions_stream(
        self,
        events: Iterator[Any],
        effective_schema: Optional[dict],
        response_model: Optional[Any],
        on_delta: Optional[OnDelta],
        on_tool_call: Optional[OnToolCall],
        on_done: Optional[OnDone],
        buffer_chars: Optional[int],
        on_chunk: Optional[OnChunk],
    ) -> Iterator[str]:
        text_parts: List[str] = []
        refusal_parts: List[str] = []
        tool_calls: Dict[int, Dict[str, Any]] = {}
        legacy_response: Dict[str, Any] = {"model": self.model, "choices": []}
        finish_reason: Optional[str] = None
        chunk_buffer = StreamChunkBuffer(buffer_chars)
        usage_tracker = StreamUsageTracker()
        for event in self._iter_provider_stream_events(events):
            self._validate_chat_completions_stream_event(event)
            payload = event.data if isinstance(event.data, Mapping) else {}
            self._update_legacy_stream_metadata(payload, legacy_response)
            usage_tracker.record(
                chunk_buffer,
                self._normalize_stream_usage(
                    payload.get("usage"),
                    input_field="prompt_tokens",
                    output_field="completion_tokens",
                ),
            )
            choices = payload.get("choices")
            if not isinstance(choices, list):
                continue
            for choice in choices:
                text, choice_finish_reason = self._consume_legacy_stream_choice(
                    choice,
                    text_parts,
                    refusal_parts,
                    tool_calls,
                )
                if text is not None:
                    yield from self._emit_stream_chunks(
                        chunk_buffer.add(text),
                        on_chunk,
                        on_delta,
                    )
                if choice_finish_reason is not None:
                    finish_reason = choice_finish_reason
        chat_response = self._build_legacy_stream_response(
            legacy_response,
            text_parts,
            refusal_parts,
            tool_calls,
            finish_reason,
        )
        chat_response = self._finalize_stream_response(
            chat_response,
            effective_schema=effective_schema,
            response_model=response_model,
        )
        yield from self._complete_stream(
            chat_response,
            chunk_buffer,
            on_chunk,
            on_delta,
            on_tool_call,
            on_done,
        )

    @staticmethod
    def _update_legacy_stream_metadata(
        payload: Mapping[str, Any],
        legacy_response: Dict[str, Any],
    ) -> None:
        for field in ("id", "model", "created", "usage"):
            if field in payload:
                legacy_response[field] = payload[field]

    def _consume_legacy_stream_choice(
        self,
        choice: Any,
        text_parts: List[str],
        refusal_parts: List[str],
        tool_calls: Dict[int, Dict[str, Any]],
    ) -> tuple[Optional[str], Optional[str]]:
        if not isinstance(choice, Mapping) or choice.get("index", 0) != 0:
            return None, None

        delta = choice.get("delta") or {}
        if not isinstance(delta, Mapping):
            delta = {}
        content = delta.get("content")
        text = content if isinstance(content, str) and content else None
        if text is not None:
            text_parts.append(text)

        refusal = delta.get("refusal")
        if isinstance(refusal, str) and refusal:
            refusal_parts.append(refusal)

        raw_tool_calls = delta.get("tool_calls")
        if isinstance(raw_tool_calls, list):
            self._accumulate_legacy_tool_calls(raw_tool_calls, tool_calls)

        finish_reason = choice.get("finish_reason")
        return text, finish_reason

    @staticmethod
    def _accumulate_legacy_tool_calls(
        raw_tool_calls: List[Any],
        tool_calls: Dict[int, Dict[str, Any]],
    ) -> None:
        for raw_tool_call in raw_tool_calls:
            if not isinstance(raw_tool_call, Mapping):
                continue
            index = raw_tool_call.get("index")
            if not isinstance(index, int):
                index = len(tool_calls)
            tool_call = tool_calls.setdefault(index, {"function": {"arguments": ""}})
            for field in ("id", "type"):
                if raw_tool_call.get(field) is not None:
                    tool_call[field] = raw_tool_call[field]
            function = raw_tool_call.get("function") or {}
            if not isinstance(function, Mapping):
                continue
            target = tool_call["function"]
            if function.get("name") is not None:
                target["name"] = function["name"]
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                target["arguments"] = f"{target.get('arguments', '')}{arguments}"
            elif isinstance(arguments, dict):
                target["arguments"] = arguments

    def _build_legacy_stream_response(
        self,
        legacy_response: Dict[str, Any],
        text_parts: List[str],
        refusal_parts: List[str],
        tool_calls: Dict[int, Dict[str, Any]],
        finish_reason: Optional[str],
    ) -> ChatResponse:
        message: Dict[str, Any] = {"content": "".join(text_parts) or None}
        if refusal_parts:
            message["refusal"] = "".join(refusal_parts)
        if tool_calls:
            message["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]
        legacy_response["choices"] = [{"message": message, "finish_reason": finish_reason}]
        return self._normalize_chat_completions_response(legacy_response)
