"""Tests for internal transport-neutral boundaries."""

from __future__ import annotations

from typing import Any, AsyncIterator, Iterator, Optional

import pytest

from src.llm_api_adapter.llms.transports import (
    AsyncTransport,
    HTTPErrorHandler,
    SSEEvent,
    SSEFrameDecoder,
    StreamErrorHandler,
    SyncTransport,
    TransportRequest,
)


class StubSyncTransport(SyncTransport):
    def post_json(
        self,
        request: TransportRequest,
        *,
        http_error_handler: Optional[HTTPErrorHandler] = None,
    ) -> Any:
        _ = http_error_handler
        return request.payload

    def post_sse(
        self,
        request: TransportRequest,
        *,
        http_error_handler: Optional[HTTPErrorHandler] = None,
        stream_error_handler: Optional[StreamErrorHandler] = None,
    ) -> Iterator[SSEEvent]:
        _ = request, http_error_handler, stream_error_handler
        yield SSEEvent(event=None, data={"transport": "sync"})


class StubAsyncTransport(AsyncTransport):
    async def post_json(
        self,
        request: TransportRequest,
        *,
        http_error_handler: Optional[HTTPErrorHandler] = None,
    ) -> Any:
        _ = http_error_handler
        return request.payload

    async def post_sse(
        self,
        request: TransportRequest,
        *,
        http_error_handler: Optional[HTTPErrorHandler] = None,
        stream_error_handler: Optional[StreamErrorHandler] = None,
    ) -> AsyncIterator[SSEEvent]:
        _ = request, http_error_handler, stream_error_handler
        yield SSEEvent(event=None, data={"transport": "async"})


@pytest.mark.unit
def test_transport_request_owns_an_immutable_headers_snapshot():
    source_headers = {"Authorization": "Bearer initial"}
    request = TransportRequest(
        url="https://example.test/messages",
        headers=source_headers,
        payload={"message": "Hello"},
        timeout=2.5,
    )
    source_headers["Authorization"] = "Bearer changed"

    assert request.headers == {"Authorization": "Bearer initial"}
    assert request.headers_dict() == {"Authorization": "Bearer initial"}
    with pytest.raises(TypeError):
        request.headers["Authorization"] = "Bearer mutated"  # type: ignore[index]


@pytest.mark.unit
def test_sync_transport_contract_carries_json_and_sse_operations():
    transport = StubSyncTransport()
    request = TransportRequest(url="https://example.test", payload={"ok": True})

    assert transport.post_json(request) == {"ok": True}
    assert list(transport.post_sse(request)) == [
        SSEEvent(event=None, data={"transport": "sync"})
    ]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_async_transport_contract_carries_json_and_sse_operations():
    transport = StubAsyncTransport()
    request = TransportRequest(url="https://example.test", payload={"ok": True})

    assert await transport.post_json(request) == {"ok": True}
    assert [event async for event in transport.post_sse(request)] == [
        SSEEvent(event=None, data={"transport": "async"})
    ]


@pytest.mark.unit
def test_shared_sse_frame_decoder_handles_multiline_events_and_eof():
    decoder = SSEFrameDecoder()

    assert decoder.feed(b"event: message") is None
    assert decoder.feed(b'data: {"text":') is None
    assert decoder.feed(b'data: "hello"}') is None
    assert decoder.feed(b"") == SSEEvent(
        event="message",
        data={"text": "hello"},
    )

    assert decoder.feed('data: {"final": true}') is None
    assert decoder.finish() == SSEEvent(event=None, data={"final": True})
    assert decoder.finish() is None
