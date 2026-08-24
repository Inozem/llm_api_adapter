"""Synchronous transport calls for Mistral's official API."""

from __future__ import annotations

from typing import Any, Callable, Iterator, Mapping, Optional

from llm_api_adapter.errors.llm_api_error import LLMAPIClientError
from llm_api_adapter.llms.transports import (
    JSONResponse,
    SSEEvent,
    SyncTransport,
    TransportRequest,
)

MISTRAL_CHAT_COMPLETIONS_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_OCR_URL = "https://api.mistral.ai/v1/ocr"


class MistralSyncClient:
    """Submit direct Mistral JSON and SSE requests through a core transport."""

    def __init__(self, transport: SyncTransport) -> None:
        self.transport = transport

    def chat(
        self,
        *,
        endpoint: str,
        headers: Mapping[str, str],
        payload: dict[str, Any],
        timeout_s: Optional[float],
        http_error_handler: Callable[[Any], None],
    ) -> dict[str, Any]:
        return self._post_json(
            endpoint,
            headers,
            payload,
            timeout_s,
            http_error_handler,
            "Mistral returned a non-object response",
        )

    def ocr(
        self,
        *,
        headers: Mapping[str, str],
        payload: dict[str, Any],
        timeout_s: Optional[float],
        http_error_handler: Callable[[Any], None],
    ) -> dict[str, Any]:
        return self._post_json(
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
    ) -> Iterator[SSEEvent]:
        return self.transport.post_sse(
            TransportRequest(
                url=endpoint,
                headers=dict(headers),
                payload=payload,
                timeout=timeout_s,
            ),
            http_error_handler=http_error_handler,
            stream_error_handler=stream_error_handler,
        )

    def _post_json(
        self,
        url: str,
        headers: Mapping[str, str],
        payload: dict[str, Any],
        timeout_s: Optional[float],
        http_error_handler: Callable[[Any], None],
        malformed_detail: str,
    ) -> dict[str, Any]:
        response: JSONResponse = self.transport.post_json(
            TransportRequest(
                url=url,
                headers=dict(headers),
                payload=payload,
                timeout=timeout_s,
            ),
            http_error_handler=http_error_handler,
        )
        response_data = response.json()
        if not isinstance(response_data, dict):
            raise LLMAPIClientError(detail=malformed_detail)
        return response_data
