"""Asynchronous transport calls for Mistral's official API."""

from __future__ import annotations

from typing import Any, AsyncIterator, Callable, Mapping, Optional

from llm_api_adapter.errors.llm_api_error import LLMAPIClientError
from llm_api_adapter.llms.async_streaming import async_request, async_stream_request
from llm_api_adapter.llms.transports import SSEEvent

from .sync_client import MISTRAL_OCR_URL


class MistralAsyncClient:
    """Submit direct asynchronous Mistral JSON and SSE requests."""

    async def chat(
        self,
        *,
        endpoint: str,
        headers: Mapping[str, str],
        payload: dict[str, Any],
        timeout_s: Optional[float],
        http_error_handler: Callable[[Any], None],
    ) -> dict[str, Any]:
        return await self._request(
            endpoint,
            headers,
            payload,
            timeout_s,
            http_error_handler,
            "Mistral returned a non-object response",
        )

    async def ocr(
        self,
        *,
        headers: Mapping[str, str],
        payload: dict[str, Any],
        timeout_s: Optional[float],
        http_error_handler: Callable[[Any], None],
    ) -> dict[str, Any]:
        return await self._request(
            MISTRAL_OCR_URL,
            headers,
            payload,
            timeout_s,
            http_error_handler,
            "Mistral OCR returned a non-object response",
        )

    def stream_chat(
        self,
        *,
        endpoint: str,
        headers: Mapping[str, str],
        payload: dict[str, Any],
        timeout_s: Optional[float],
        http_error_handler: Callable[[Any], None],
        stream_error_handler: Callable[[SSEEvent], None],
    ) -> AsyncIterator[SSEEvent]:
        return async_stream_request(
            endpoint,
            headers=dict(headers),
            payload=payload,
            timeout=timeout_s,
            http_error_handler=http_error_handler,
            stream_error_handler=stream_error_handler,
        )

    async def _request(
        self,
        url: str,
        headers: Mapping[str, str],
        payload: dict[str, Any],
        timeout_s: Optional[float],
        http_error_handler: Callable[[Any], None],
        malformed_detail: str,
    ) -> dict[str, Any]:
        response_data = await async_request(
            url,
            headers=dict(headers),
            payload=payload,
            timeout=timeout_s,
            http_error_handler=http_error_handler,
        )
        if not isinstance(response_data, dict):
            raise LLMAPIClientError(detail=malformed_detail)
        return response_data
