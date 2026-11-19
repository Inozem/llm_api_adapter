"""Asynchronous HTTP client for Model Studio's Messages-compatible API."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from llm_api_adapter.errors.llm_api_error import LLMAPIClientError
from llm_api_adapter.llms.async_streaming import async_request, async_stream_request
from llm_api_adapter.llms.anthropic_compatible.async_client import (
    _AsyncMessagesExecutionMixin,
)
from llm_api_adapter.llms.transports import SSEEvent, TransportRequest

from .sync_client import QwenMessagesSyncClient, messages_url


class _QwenAsyncExecutionBridge(_AsyncMessagesExecutionMixin):
    """Bind one Qwen operation to the neutral shared execution methods."""

    def __init__(
        self,
        client_class: type,
        stream_source: AsyncIterator[SSEEvent] | None = None,
    ) -> None:
        self._handle_http_error = client_class._handle_http_error
        self._handle_stream_error = client_class._handle_stream_error
        self._stream_source = stream_source

    async def _get_async_request(self, *args: Any, **kwargs: Any) -> Any:
        # Keep Qwen's historical module-level monkeypatch surface.
        return await async_request(*args, **kwargs)

    def _get_async_stream_request(
        self, *_args: Any, **_kwargs: Any
    ) -> AsyncIterator[SSEEvent]:
        assert self._stream_source is not None
        return self._stream_source


class _PreparedAsyncStreamIterator:
    """Delegate iteration to shared execution and close eager sources pre-start."""

    def __init__(
        self,
        execution: AsyncIterator[SSEEvent],
        events: AsyncIterator[SSEEvent],
    ) -> None:
        self._execution = execution
        self._events = events
        self._started = False

    def __aiter__(self) -> _PreparedAsyncStreamIterator:
        return self

    async def __anext__(self) -> SSEEvent:
        self._started = True
        return await anext(self._execution)

    async def aclose(self) -> None:
        if self._started:
            close = getattr(self._execution, "aclose", None)
        else:
            close = getattr(self._events, "aclose", None)
        if close is not None:
            await close()


class QwenMessagesAsyncClient:
    """Submit asynchronous Qwen Messages JSON and SSE requests."""

    async def chat(
        self,
        *,
        api_key: str,
        workspace_id: object,
        payload: dict[str, Any],
        timeout_s: float | None,
    ) -> dict[str, Any]:
        """Create one non-streaming Qwen Messages response."""
        bridge = _QwenAsyncExecutionBridge(QwenMessagesSyncClient)
        response_data = await bridge._execute_prepared_request(
            TransportRequest(
                url=messages_url(workspace_id),
                headers=self._headers(api_key),
                payload=payload,
                timeout=timeout_s,
            )
        )
        if not isinstance(response_data, dict):
            raise LLMAPIClientError(
                detail="Qwen Messages returned a non-object response",
            )
        return response_data

    def stream(
        self,
        *,
        api_key: str,
        workspace_id: object,
        payload: dict[str, Any],
        timeout_s: float | None,
    ) -> AsyncIterator[SSEEvent]:
        """Open one eager native SSE source, then use shared async execution."""
        stream_payload = dict(payload)
        stream_payload["stream"] = True
        request = TransportRequest(
            url=messages_url(workspace_id),
            headers=self._headers(api_key),
            payload=stream_payload,
            timeout=timeout_s,
        )
        # Preserve Qwen's historical eager helper call and native patch point.
        events = async_stream_request(
            request.url,
            headers=request.headers_dict(),
            payload=request.payload,
            timeout=request.timeout,
            http_error_handler=QwenMessagesSyncClient._handle_http_error,
            stream_error_handler=QwenMessagesSyncClient._handle_stream_error,
        )
        bridge = _QwenAsyncExecutionBridge(QwenMessagesSyncClient, events)
        execution = bridge._execute_prepared_stream(request)
        return _PreparedAsyncStreamIterator(execution, events)

    @staticmethod
    def _headers(api_key: str) -> dict[str, str]:
        return {
            "x-api-key": api_key,
            "Content-Type": "application/json",
        }


__all__ = ["QwenMessagesAsyncClient"]
