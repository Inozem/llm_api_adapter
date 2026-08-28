"""Asynchronous client for xAI's official Responses API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Mapping

from llm_api_adapter.errors.llm_api_error import LLMAPIClientError
from llm_api_adapter.llms.async_streaming import async_request, async_stream_request
from llm_api_adapter.llms.transports import SSEEvent

from .sync_client import XAIResponsesSyncClient, _XAI_RESPONSES_URL


@dataclass(repr=False)
class XAIResponsesAsyncClient:
    """Send asynchronous xAI Responses API requests through the core helper."""

    api_key: str
    endpoint: str = _XAI_RESPONSES_URL

    async def create(
        self,
        *,
        model: str,
        timeout: float | None = None,
        **parameters: Any,
    ) -> dict[str, Any]:
        """Create one Responses API response and validate its JSON envelope."""
        payload = await async_request(
            self.endpoint,
            headers=self._headers(),
            payload={"model": model, **parameters},
            timeout=timeout,
            http_error_handler=self._handle_http_error,
        )
        if not isinstance(payload, dict):
            raise LLMAPIClientError(
                detail="xAI Responses API returned a non-object response",
            )
        return payload

    def stream(
        self,
        *,
        model: str,
        timeout: float | None = None,
        **parameters: Any,
    ) -> AsyncIterator[SSEEvent]:
        """Stream one Responses API response as provider-neutral SSE events."""
        return async_stream_request(
            self.endpoint,
            headers=self._headers(),
            payload={"model": model, **parameters, "stream": True},
            timeout=timeout,
            http_error_handler=self._handle_http_error,
            stream_error_handler=self._handle_stream_error,
        )

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _handle_http_error(error: Any) -> None:
        response = getattr(error, "response", None)
        status_code = getattr(response, "status_code", None)
        payload: Mapping[str, Any] = {}
        if response is not None:
            try:
                candidate = response.json()
                if isinstance(candidate, Mapping):
                    payload = candidate
            except Exception:
                pass
        error_data = payload.get("error", payload)
        if not isinstance(error_data, Mapping):
            error_data = {}
        error_type = error_data.get("type") or error_data.get("code")
        detail = error_data.get("message") or str(error)
        XAIResponsesSyncClient._raise_mapped_error(
            status_code=status_code if isinstance(status_code, int) else None,
            error_type=str(error_type) if error_type else None,
            detail=str(detail),
        )

    @staticmethod
    def _handle_stream_error(event: SSEEvent) -> None:
        payload = event.data if isinstance(event.data, Mapping) else {}
        error_data = payload.get("error", payload)
        if not isinstance(error_data, Mapping):
            error_data = {}
        error_type = error_data.get("type") or error_data.get("code")
        detail = error_data.get("message") or "xAI Responses stream failed"
        XAIResponsesSyncClient._raise_mapped_error(
            status_code=None,
            error_type=str(error_type) if error_type else None,
            detail=str(detail),
        )


__all__ = ["XAIResponsesAsyncClient"]
