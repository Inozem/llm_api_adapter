"""Asynchronous client for DeepSeek's official Responses API."""

from __future__ import annotations

from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any, AsyncIterator, Mapping

from llm_api_adapter.errors.llm_api_error import LLMAPIClientError, LLMAPIError
from llm_api_adapter.llms.async_streaming import (
    async_multipart_request,
    async_request,
    async_stream_request,
)
from llm_api_adapter.llms.openai_compatible.async_client import (
    OpenAICompatibleAsyncClient,
)
from llm_api_adapter.llms.transports import SSEEvent

from .sync_client import (
    DEEPSEEK_FILES_URL,
    DEEPSEEK_RESPONSES_URL,
    DeepSeekResponsesSyncClient,
    _FILE_UPLOAD_FAILURE,
    _FILE_UPLOAD_TIMEOUT,
    _file_id_from_upload,
    _pdf_upload_form,
)


@dataclass(repr=False, init=False)
class DeepSeekResponsesAsyncClient(OpenAICompatibleAsyncClient):
    """Use Core's async engines with native DeepSeek request policy."""

    api_key: str
    endpoint: str = DEEPSEEK_RESPONSES_URL
    __match_args__ = ("api_key", "endpoint")

    def __init__(
        self,
        api_key: str,
        endpoint: str = DEEPSEEK_RESPONSES_URL,
    ) -> None:
        """Retain the native constructor without creating a sync transport."""
        self.api_key = api_key
        self.endpoint = endpoint
        self._api_variant_override = "responses"
        self._diagnostic_formatter = None

    def __repr__(self) -> str:
        """Retain the opaque native representation without credential data."""
        return object.__repr__(self)

    async def create(
        self,
        *,
        model: str,
        timeout: float | None = None,
        **parameters: Any,
    ) -> dict[str, Any]:
        """Create a Responses result through Core's inherited JSON engine."""
        result = await self._send_request(
            self.endpoint,
            {"model": model, **parameters},
            timeout,
        )
        if not isinstance(result, dict):
            raise LLMAPIClientError(
                detail="DeepSeek Responses API returned a non-object response",
            )
        return result

    async def _upload_pdf(
        self,
        content: bytes,
        timeout: float | None = None,
    ) -> str:
        """Upload one PDF asynchronously and return its private receipt."""
        try:
            payload = await async_multipart_request(
                DEEPSEEK_FILES_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                form=_pdf_upload_form(content),
                timeout=_FILE_UPLOAD_TIMEOUT if timeout is None else timeout,
                http_error_handler=self._handle_file_upload_http_error,
            )
        except LLMAPIError as error:
            raise type(error)(detail=_FILE_UPLOAD_FAILURE) from None
        except (JSONDecodeError, UnicodeDecodeError):
            raise LLMAPIClientError(detail=_FILE_UPLOAD_FAILURE) from None
        return _file_id_from_upload(payload)

    @staticmethod
    def _handle_file_upload_http_error(error: Any) -> None:
        DeepSeekResponsesSyncClient._handle_file_upload_http_error(error)

    def stream(
        self,
        *,
        model: str,
        timeout: float | None = None,
        **parameters: Any,
    ) -> AsyncIterator[SSEEvent]:
        """Bind native request data before returning Core's lazy stream."""
        endpoint = self.endpoint
        api_key = self.api_key
        payload = {"model": model, **parameters, "stream": True}
        headers = self._headers()
        http_error_handler = self._handle_http_error
        stream_error_handler = self._handle_stream_error
        events = async_stream_request(
            endpoint,
            headers=headers,
            payload=payload,
            timeout=timeout,
            http_error_handler=http_error_handler,
            stream_error_handler=stream_error_handler,
        )
        source = _BoundAsyncEventSource(events)
        view = _NativeAsyncRequestView(
            api_key=api_key,
            endpoint=endpoint,
            headers=headers,
            events=source,
        )
        return _CoreAsyncStreamIterator(
            view._stream_request(endpoint, payload, timeout),
            source,
        )

    async def _get_async_request(self, *args: Any, **kwargs: Any) -> Any:
        """Resolve the native module JSON patch point for each invocation."""
        return await async_request(*args, **kwargs)

    @staticmethod
    def _handle_http_error(error: Any) -> None:
        DeepSeekResponsesSyncClient._handle_http_error(error)

    @staticmethod
    def _handle_stream_error(event: SSEEvent) -> None:
        DeepSeekResponsesSyncClient._handle_stream_error(event)


class _NativeAsyncRequestView(DeepSeekResponsesAsyncClient):
    """Bind one native async stream to Core without shared client state."""

    def __init__(
        self,
        *,
        api_key: str,
        endpoint: str,
        headers: Mapping[str, str],
        events: AsyncIterator[SSEEvent],
    ) -> None:
        super().__init__(api_key, endpoint)
        self._headers_snapshot = dict(headers)
        self._events = events

    def _headers(self) -> dict[str, str]:
        return dict(self._headers_snapshot)

    def _get_async_stream_request(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> AsyncIterator[SSEEvent]:
        del args, kwargs
        return self._events

    @staticmethod
    def _is_failed_stream_event(event: SSEEvent) -> bool:
        """Leave DeepSeek semantic failure events to native response parsing."""
        del event
        return False


class _BoundAsyncEventSource:
    """Bind one native async source and own its idempotent cleanup."""

    def __init__(self, events: AsyncIterator[SSEEvent]) -> None:
        self._events = events
        self._closed = False
        self._iteration_failed = False

    def __aiter__(self) -> "_BoundAsyncEventSource":
        return self

    async def __anext__(self) -> SSEEvent:
        try:
            return await self._events.__anext__()
        except StopAsyncIteration:
            raise
        except BaseException:
            self._iteration_failed = True
            raise

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self._events, "aclose", None)
        if not callable(close):
            return
        try:
            await close()
        except BaseException:
            if self._iteration_failed:
                return
            raise


class _CoreAsyncStreamIterator:
    """Expose Core's iterator while owning its source from return onward."""

    def __init__(
        self,
        events: AsyncIterator[SSEEvent],
        source: _BoundAsyncEventSource,
    ) -> None:
        self._events = events
        self._source = source

    def __aiter__(self) -> "_CoreAsyncStreamIterator":
        return self

    async def __anext__(self) -> SSEEvent:
        return await self._events.__anext__()

    async def aclose(self) -> None:
        await self._events.aclose()
        await self._source.aclose()


DeepSeekResponsesAsyncClient.__dataclass_fields__ = {
    name: DeepSeekResponsesAsyncClient.__dataclass_fields__[name]
    for name in ("api_key", "endpoint")
}


DeepSeekAsyncClient = DeepSeekResponsesAsyncClient


__all__ = ["DeepSeekResponsesAsyncClient", "DeepSeekAsyncClient"]
