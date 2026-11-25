"""Invocation-local Kimi reasoning observation and source ownership."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from llm_api_adapter.adapters.base_adapter import AsyncOnReasoning, LLMAdapterBase, OnReasoning
from llm_api_adapter.llms.streaming import StreamReasoningCollector
from llm_api_adapter.llms.transports import SSEEvent
from llm_api_adapter.models.responses.chat_response import ChatResponse


@dataclass
class _KimiStreamInvocation:
    """Request-local reasoning policy carried by a shallow adapter view."""

    capture_reasoning: bool
    collector: StreamReasoningCollector | None
    response: ChatResponse | None
    on_reasoning: OnReasoning | None = None
    on_async_reasoning: AsyncOnReasoning | None = None
    provider_events: Iterator[SSEEvent] | None = None
    async_provider_events: AsyncIterator[SSEEvent] | None = None
    async_consumer: AsyncIterator[str] | None = None


def observe_sync_events(
    adapter: LLMAdapterBase,
    inherited: Iterator[SSEEvent],
    invocation: _KimiStreamInvocation,
) -> Iterator[SSEEvent]:
    """Observe reasoning outside provider translation and own only its wrapper."""
    try:
        for event in inherited:
            _record_sync_reasoning(adapter, event, invocation)
            yield event
    except BaseException:
        close_sync_layers(inherited, suppress_errors=True)
        raise
    else:
        close_sync_layers(inherited, suppress_errors=False)


async def observe_async_events(
    adapter: LLMAdapterBase,
    inherited: AsyncIterator[SSEEvent],
    invocation: _KimiStreamInvocation,
) -> AsyncIterator[SSEEvent]:
    """Async observer; the inherited async execution owns raw provider events."""
    async with adapter._closing_async_source(inherited):
        async for event in inherited:
            await _record_async_reasoning(adapter, event, invocation)
            yield event


def close_sync_consumer(
    source: Iterator[SSEEvent],
    consumer: Iterator[str],
    invocation: _KimiStreamInvocation | None,
) -> Iterator[str]:
    """Delegate assembly to Core and close every owned sync layer."""
    try:
        yield from consumer
    except BaseException:
        close_sync_layers(
            consumer,
            invocation.provider_events if invocation is not None else None,
            source,
            suppress_errors=True,
        )
        raise
    else:
        close_sync_layers(
            consumer,
            invocation.provider_events if invocation is not None else None,
            source,
            suppress_errors=False,
        )


def close_sync_layers(
    *sources: Iterator[Any] | None,
    suppress_errors: bool,
) -> None:
    first_error: BaseException | None = None
    for source in sources:
        if source is None:
            continue
        close = getattr(source, "close", None)
        if not callable(close):
            continue
        try:
            close()
        except BaseException as error:
            if first_error is None:
                first_error = error
    if first_error is not None and not suppress_errors:
        raise first_error


async def close_async_layers(
    adapter: LLMAdapterBase,
    *sources: AsyncIterator[Any] | None,
    suppress_errors: bool,
) -> None:
    first_error: BaseException | None = None
    for source in sources:
        if source is None:
            continue
        try:
            async with adapter._closing_async_source(source):
                pass
        except BaseException as error:
            if first_error is None:
                first_error = error
    if first_error is not None and not suppress_errors:
        raise first_error


def _record_sync_reasoning(
    adapter: LLMAdapterBase,
    event: SSEEvent,
    invocation: _KimiStreamInvocation,
) -> None:
    if not invocation.capture_reasoning or invocation.collector is None or invocation.response is None:
        return
    for text in _reasoning_texts(event):
        adapter._record_reasoning_event(
            invocation.response,
            invocation.collector,
            text,
            capture_reasoning=True,
            on_reasoning=invocation.on_reasoning,
        )


async def _record_async_reasoning(
    adapter: LLMAdapterBase,
    event: SSEEvent,
    invocation: _KimiStreamInvocation,
) -> None:
    if not invocation.capture_reasoning or invocation.collector is None or invocation.response is None:
        return
    for text in _reasoning_texts(event):
        await adapter._record_async_reasoning_event(
            invocation.response,
            invocation.collector,
            text,
            capture_reasoning=True,
            on_reasoning=invocation.on_async_reasoning,
        )


def _reasoning_texts(event: SSEEvent) -> tuple[str, ...]:
    payload = event.data
    choices = payload.get("choices") if isinstance(payload, Mapping) else None
    if not isinstance(choices, list):
        return ()
    texts: list[str] = []
    for choice in choices:
        if not isinstance(choice, Mapping) or choice.get("index", 0) != 0:
            continue
        delta = choice.get("delta")
        text = delta.get("reasoning_content") if isinstance(delta, Mapping) else None
        if isinstance(text, str) and text:
            texts.append(text)
    return tuple(texts)


__all__ = [
    "_KimiStreamInvocation",
    "close_async_layers",
    "close_sync_consumer",
    "observe_async_events",
    "observe_sync_events",
]
