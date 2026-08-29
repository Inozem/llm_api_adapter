"""Tests for asynchronous multipart transport behavior."""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest

from src.llm_api_adapter.errors.llm_api_error import (
    LLMAPIClientError,
    LLMAPIRateLimitError,
    LLMAPITimeoutError,
)
from src.llm_api_adapter.llms.async_streaming import async_multipart_request
from src.llm_api_adapter.llms.transports import MultipartFile, MultipartForm


class FakeAsyncResponse:
    def __init__(self, *, body: dict | None = None, status_code: int = 200) -> None:
        self.body = body or {}
        self.status_code = status_code
        self.request = httpx.Request("POST", "https://example.test/files")
        self.aclose_calls = 0
        self.aread_calls = 0

    def json(self) -> dict:
        return self.body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=self.request,
                response=self,
            )

    async def aread(self) -> bytes:
        self.aread_calls += 1
        return b""

    async def aclose(self) -> None:
        self.aclose_calls += 1


class FakeAsyncClient:
    def __init__(
        self,
        *,
        response: FakeAsyncResponse | None = None,
        post_error: Exception | None = None,
    ) -> None:
        self.response = response
        self.post_error = post_error
        self.post_calls: list[tuple[str, dict]] = []
        self.aclose_calls = 0

    async def post(self, url: str, **kwargs):
        self.post_calls.append((url, kwargs))
        if self.post_error is not None:
            raise self.post_error
        return self.response

    async def aclose(self) -> None:
        self.aclose_calls += 1


def _form() -> MultipartForm:
    return MultipartForm(
        fields=(("purpose", "documents"),),
        files=(
            MultipartFile(
                "file",
                "report.pdf",
                b"%PDF-test",
                "application/pdf",
            ),
        ),
    )


@pytest.mark.asyncio
@pytest.mark.unit
async def test_async_multipart_posts_form_and_closes_resources():
    response = FakeAsyncResponse(body={"id": "file_123"})
    client = FakeAsyncClient(response=response)

    with patch.object(httpx, "AsyncClient", return_value=client):
        result = await async_multipart_request(
            "https://example.test/files",
            headers={
                "Authorization": "Bearer test",
                "Content-Type": "application/json",
            },
            form=_form(),
            timeout=3.0,
        )

    assert result == {"id": "file_123"}
    assert len(client.post_calls) == 1
    url, kwargs = client.post_calls[0]
    assert url == "https://example.test/files"
    assert kwargs["headers"]["authorization"] == "Bearer test"
    assert kwargs["headers"]["content-type"].startswith("multipart/form-data;")
    assert b'name="purpose"' in kwargs["content"]
    assert b"documents" in kwargs["content"]
    assert b'name="file"; filename="report.pdf"' in kwargs["content"]
    assert b"%PDF-test" in kwargs["content"]
    assert kwargs["timeout"] == 3.0
    assert response.aclose_calls == 1
    assert client.aclose_calls == 1


@pytest.mark.asyncio
@pytest.mark.unit
async def test_async_multipart_uses_bytes_compatible_with_an_async_client():
    received: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        received.append(request)
        return httpx.Response(200, json={"id": "file_123"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with patch.object(httpx, "AsyncClient", return_value=client):
        result = await async_multipart_request(
            "https://example.test/files",
            headers={"Authorization": "Bearer test"},
            form=_form(),
        )

    assert result == {"id": "file_123"}
    assert len(received) == 1
    assert received[0].headers["content-type"].startswith("multipart/form-data;")
    assert b'name="purpose"' in received[0].content
    assert b'name="file"; filename="report.pdf"' in received[0].content


@pytest.mark.asyncio
@pytest.mark.unit
async def test_async_multipart_preserves_provider_http_error_mapping_and_cleanup():
    response = FakeAsyncResponse(status_code=429)
    client = FakeAsyncClient(response=response)
    observed = []

    def provider_handler(error):
        observed.append(error.response)
        raise LLMAPIRateLimitError(detail="provider mapping")

    with patch.object(httpx, "AsyncClient", return_value=client):
        with pytest.raises(LLMAPIRateLimitError, match="provider mapping"):
            await async_multipart_request(
                "https://example.test/files",
                form=_form(),
                http_error_handler=provider_handler,
            )

    assert observed == [response]
    assert response.aread_calls == 1
    assert response.aclose_calls == 1
    assert client.aclose_calls == 1


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize(
    ("post_error", "expected_error"),
    (
        (httpx.TimeoutException("timed out"), LLMAPITimeoutError),
        (httpx.RequestError("connection failed"), LLMAPIClientError),
    ),
)
async def test_async_multipart_maps_request_failures_and_closes_client(
    post_error,
    expected_error,
):
    client = FakeAsyncClient(post_error=post_error)

    with patch.object(httpx, "AsyncClient", return_value=client):
        with pytest.raises(expected_error):
            await async_multipart_request(
                "https://example.test/files",
                form=_form(),
            )

    assert client.aclose_calls == 1
