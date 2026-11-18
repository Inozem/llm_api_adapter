"""Asynchronous execution for generic Anthropic Messages connections."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from ...errors.llm_api_error import LLMAPIClientError, LLMAPIError
from ..async_streaming import async_request, async_stream_request
from ..transports import SSEEvent, TransportRequest, validate_sync_transport
from .sync_client import AnthropicCompatibleSyncClient, _SyncMessagesExecutionMixin


class _AsyncMessagesExecutionMixin(_SyncMessagesExecutionMixin):
    """Async prepared-request engine, usable without generic client fields."""

    async def _get_async_request(self, *args: Any, **kwargs: Any) -> Any:
        return await async_request(*args, **kwargs)

    def _get_async_stream_request(self, *args: Any, **kwargs: Any) -> AsyncIterator[SSEEvent]:
        return async_stream_request(*args, **kwargs)

    async def _execute_prepared_request(self, request: TransportRequest) -> Any:
        try:
            return await self._get_async_request(
                request.url,
                headers=request.headers_dict(),
                payload=request.payload,
                timeout=request.timeout,
                http_error_handler=self._handle_http_error,
            )
        except (LLMAPIError, json.JSONDecodeError) as exc:
            self._raise_execution_error(exc)

    def _execute_prepared_stream(self, request: TransportRequest) -> AsyncIterator[SSEEvent]:
        async def iterate() -> AsyncIterator[SSEEvent]:
            events = None
            primary_error: BaseException | None = None
            try:
                events = self._get_async_stream_request(
                    request.url,
                    headers=request.headers_dict(),
                    payload=request.payload,
                    timeout=request.timeout,
                    http_error_handler=self._handle_http_error,
                    stream_error_handler=self._handle_stream_error,
                )
                async for event in events:
                    yield event
            except BaseException as exc:
                primary_error = exc
                if isinstance(exc, LLMAPIError):
                    self._raise_execution_error(exc)
                raise
            finally:
                if events is not None:
                    try:
                        close = getattr(events, "aclose", None)
                        if close is not None:
                            await close()
                    except BaseException as close_error:
                        if primary_error is None:
                            if isinstance(close_error, LLMAPIError):
                                self._raise_execution_error(close_error)
                            raise

        return iterate()

    async def _send_request(self, url: str, payload: dict, timeout_s: float | None = None):
        return await self._execute_prepared_request(
            TransportRequest(url=url, headers=self._headers(), payload=dict(payload), timeout=timeout_s)
        )

    def _stream_request(self, url: str, payload: dict, timeout_s: float | None = None):
        return self._execute_prepared_stream(
            TransportRequest(url=url, headers=self._headers(), payload=dict(payload), timeout=timeout_s)
        )


class AnthropicCompatibleAsyncClient(_AsyncMessagesExecutionMixin, AnthropicCompatibleSyncClient):
    """Use an explicit Messages API root with true asynchronous requests."""

    def __init__(
        self, api_key: str, endpoint: str, transport: str = "requests"
    ) -> None:
        self._initialize_compatible(
            api_key,
            endpoint,
            validate_sync_transport(transport),
            initialize_sync_transport=False,
        )

    def _raise_execution_error(self, error: BaseException) -> None:
        if isinstance(error, json.JSONDecodeError):
            raise LLMAPIClientError(detail="Malformed JSON response.") from None
        assert isinstance(error, LLMAPIError)
        self._raise_safe_execution_error(error)

    async def chat_completion(self, model: str, timeout_s: float | None = None, **wire_payload: Any):
        url, payload = self._request_data(model, wire_payload, stream=False)
        return await self._send_request(url, payload, timeout_s)

    def stream(self, model: str, timeout_s: float | None = None, **wire_payload: Any):
        url, payload = self._request_data(model, wire_payload, stream=True)
        return self._stream_request(url, payload, timeout_s)


__all__ = ["AnthropicCompatibleAsyncClient"]
