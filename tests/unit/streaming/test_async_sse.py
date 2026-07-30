import asyncio
import builtins
from unittest.mock import patch

import httpx
import pytest

from src.llm_api_adapter.errors.llm_api_error import (
    LLMAPIClientError,
    LLMAPIAuthorizationError,
    LLMAPIServerError,
    LLMAPITimeoutError,
)
from src.llm_api_adapter.llms.async_streaming import (
    SSEEvent,
    async_request,
    async_stream_request,
)


class FakeResponse:
    def __init__(self, lines=None, status_code=200, json_data=None):
        self.lines = lines or []
        self.status_code = status_code
        self.json_data = json_data
        self.request = httpx.Request("POST", "https://example.test")
        self.closed = False
        self.close_calls = 0

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=self.request,
                response=self,
            )

    def json(self):
        return self.json_data

    async def aiter_lines(self):
        for line in self.lines:
            yield line

    async def aclose(self):
        self.close_calls += 1
        self.closed = True


class BlockingResponse(FakeResponse):
    def __init__(self):
        super().__init__()
        self.started = asyncio.Event()

    async def aiter_lines(self):
        self.started.set()
        await asyncio.Event().wait()
        yield ""


class FakeAsyncClient:
    def __init__(self, response=None, post_error=None):
        self.response = response
        self.post_error = post_error
        self.post_calls = []
        self.build_request_calls = []
        self.send_calls = []
        self.closed = False

    async def post(self, url, **kwargs):
        self.post_calls.append((url, kwargs))
        if self.post_error is not None:
            raise self.post_error
        return self.response

    def build_request(self, method, url, **kwargs):
        self.build_request_calls.append((method, url, kwargs))
        return httpx.Request(
            method,
            url,
            headers=kwargs.get("headers"),
            json=kwargs.get("json"),
        )

    async def send(self, request, *, stream=False):
        self.send_calls.append((request, stream))
        return self.response

    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
@pytest.mark.unit
async def test_async_import_error_explains_optional_install():
    real_import = builtins.__import__

    def missing_httpx(name, *args, **kwargs):
        if name == "httpx":
            raise ImportError("httpx is missing")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=missing_httpx):
        with pytest.raises(ImportError, match=r"llm-api-adapter\[async\]"):
            await async_request("https://example.test")


@pytest.mark.asyncio
@pytest.mark.unit
async def test_async_request_returns_json_and_closes_resources():
    response = FakeResponse(json_data={"answer": "ok"})
    client = FakeAsyncClient(response=response)

    with patch.object(httpx, "AsyncClient", return_value=client):
        result = await async_request(
            "https://example.test/json",
            headers={"Authorization": "Bearer test"},
            payload={"prompt": "hello"},
            timeout=3.0,
        )

    assert result == {"answer": "ok"}
    assert response.closed is True
    assert client.closed is True
    assert client.post_calls == [
        (
            "https://example.test/json",
            {
                "headers": {"Authorization": "Bearer test"},
                "json": {"prompt": "hello"},
                "timeout": 3.0,
            },
        )
    ]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_async_request_maps_http_status_and_closes_resources():
    response = FakeResponse(status_code=401)
    client = FakeAsyncClient(response=response)

    with patch.object(httpx, "AsyncClient", return_value=client):
        with pytest.raises(LLMAPIAuthorizationError):
            await async_request("https://example.test/json")

    assert response.closed is True
    assert client.closed is True


@pytest.mark.asyncio
@pytest.mark.unit
async def test_async_request_maps_timeout_and_closes_client():
    client = FakeAsyncClient(post_error=httpx.TimeoutException("timeout"))

    with patch.object(httpx, "AsyncClient", return_value=client):
        with pytest.raises(LLMAPITimeoutError):
            await async_request("https://example.test/json")

    assert client.closed is True


@pytest.mark.asyncio
@pytest.mark.unit
async def test_async_stream_request_parses_sse_and_closes_resources():
    response = FakeResponse(
        [
            b"event: message",
            b'data: {"text":',
            b'data: "hello"}',
            b"",
            b"data: [DONE]",
            b"",
        ]
    )
    client = FakeAsyncClient(response=response)

    with patch.object(httpx, "AsyncClient", return_value=client):
        events = [
            event
            async for event in async_stream_request(
                "https://example.test/stream",
                headers={"Authorization": "Bearer test"},
                payload={"prompt": "hello"},
                timeout=3.0,
            )
        ]

    assert events == [SSEEvent(event="message", data={"text": "hello"})]
    assert response.closed is True
    assert client.closed is True
    assert client.send_calls[0][1] is True
    assert client.build_request_calls[0][2]["timeout"] == 3.0


@pytest.mark.asyncio
@pytest.mark.unit
async def test_async_stream_request_maps_http_status():
    response = FakeResponse(status_code=500)
    client = FakeAsyncClient(response=response)

    with patch.object(httpx, "AsyncClient", return_value=client):
        with pytest.raises(LLMAPIServerError):
            [
                event
                async for event in async_stream_request("https://example.test/stream")
            ]

    assert response.closed is True
    assert client.closed is True


@pytest.mark.asyncio
@pytest.mark.unit
async def test_async_stream_request_maps_malformed_sse_and_closes_response():
    response = FakeResponse([b"data: not-json", b""])
    client = FakeAsyncClient(response=response)

    with patch.object(httpx, "AsyncClient", return_value=client):
        with pytest.raises(LLMAPIClientError, match="Malformed SSE JSON data"):
            [
                event
                async for event in async_stream_request("https://example.test/stream")
            ]

    assert response.closed is True
    assert client.closed is True


@pytest.mark.asyncio
@pytest.mark.unit
async def test_async_stream_request_closes_resources_on_early_aclose():
    response = FakeResponse(
        [b'data: {"text":"first"}', b"", b'data: {"text":"second"}', b""]
    )
    client = FakeAsyncClient(response=response)

    with patch.object(httpx, "AsyncClient", return_value=client):
        events = async_stream_request("https://example.test/stream")
        assert await events.__anext__() == SSEEvent(
            event=None, data={"text": "first"}
        )
        await events.aclose()

    assert response.closed is True
    assert client.closed is True


@pytest.mark.asyncio
@pytest.mark.unit
async def test_async_stream_request_closes_resources_on_cancellation():
    response = BlockingResponse()
    client = FakeAsyncClient(response=response)

    async def consume():
        async for _ in async_stream_request("https://example.test/stream"):
            pass

    with patch.object(httpx, "AsyncClient", return_value=client):
        task = asyncio.create_task(consume())
        await response.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert response.closed is True
    assert client.closed is True
