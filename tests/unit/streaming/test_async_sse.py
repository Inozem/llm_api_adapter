import asyncio
import builtins
import json
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
from src.llm_api_adapter.llms.transports import _async_cleanup_preserving_primary
from src.llm_api_adapter.adapters.google.adapter import GoogleAdapter
from src.llm_api_adapter.models.messages.chat_message import UserMessage


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
async def test_async_http_error_read_cancellation_survives_both_cleanup_failures():
    response = FakeResponse(status_code=503)
    client = FakeAsyncClient(response=response)
    started = asyncio.Event()
    close_attempts = []

    async def blocked_read():
        started.set()
        await asyncio.Event().wait()

    async def fail_response_close():
        close_attempts.append("response")
        raise RuntimeError("response cleanup")

    async def fail_client_close():
        close_attempts.append("client")
        raise RuntimeError("client cleanup")

    response.aread = blocked_read
    response.aclose = fail_response_close
    client.aclose = fail_client_close

    task = None
    cancelled = False
    with patch.object(httpx, "AsyncClient", return_value=client):
        try:
            task = asyncio.create_task(async_request("https://example.test/json"))
            await asyncio.wait_for(started.wait(), timeout=2)
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=2)
            except asyncio.CancelledError:
                cancelled = True
        finally:
            if task is not None and not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=2)
                except BaseException:
                    pass

    assert cancelled
    assert close_attempts == ["response", "client"]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_async_cleanup_only_failure_attempts_both_response_and_client():
    response = FakeResponse(json_data={"answer": "ok"})
    client = FakeAsyncClient(response=response)
    close_attempts = []

    async def fail_response_close():
        close_attempts.append("response")
        raise RuntimeError("first cleanup")

    async def fail_client_close():
        close_attempts.append("client")
        raise ValueError("second cleanup")

    response.aclose = fail_response_close
    client.aclose = fail_client_close
    with patch.object(httpx, "AsyncClient", return_value=client):
        with pytest.raises(RuntimeError, match="first cleanup"):
            await async_request("https://example.test/json")

    assert close_attempts == ["response", "client"]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_async_cleanup_guard_skips_unacquired_resource_closer():
    close_attempts = []

    async def close_acquired_resource():
        close_attempts.append("acquired")

    try:
        raise ValueError("caller-side exception is already handled")
    except ValueError:
        async with _async_cleanup_preserving_primary(
            lambda: None,
            close_acquired_resource,
        ):
            pass

    assert close_attempts == ["acquired"]


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
async def test_async_malformed_sse_error_survives_response_close_failure():
    response = FakeResponse([b"data: not-json", b""])
    client = FakeAsyncClient(response=response)
    close_attempts = []

    async def fail_response_close():
        close_attempts.append("response")
        raise RuntimeError("response cleanup")

    response.aclose = fail_response_close
    with patch.object(httpx, "AsyncClient", return_value=client):
        with pytest.raises(LLMAPIClientError, match="Malformed SSE JSON data"):
            [
                event
                async for event in async_stream_request(
                    "https://example.test/stream"
                )
            ]

    assert close_attempts == ["response"]
    assert client.closed is True


@pytest.mark.asyncio
@pytest.mark.unit
async def test_async_stream_body_cancellation_survives_response_and_client_close_failures():
    started, release = asyncio.Event(), asyncio.Event()
    close_attempts = []

    class PausedBodyResponse(FakeResponse):
        async def aiter_lines(self):
            started.set()
            await release.wait()
            yield b'data: {"message":"too late"}'

    response = PausedBodyResponse()
    client = FakeAsyncClient(response=response)

    async def fail_response_close():
        close_attempts.append("response")
        raise RuntimeError("response cleanup")

    async def fail_client_close():
        close_attempts.append("client")
        raise RuntimeError("client cleanup")

    response.aclose = fail_response_close
    client.aclose = fail_client_close
    stream = async_stream_request("https://example.test/stream")
    pending = None
    cancelled = False
    try:
        with patch.object(httpx, "AsyncClient", return_value=client):
            pending = asyncio.create_task(stream.__anext__())
            await asyncio.wait_for(started.wait(), timeout=2)
            pending.cancel()
            try:
                await asyncio.wait_for(pending, timeout=2)
            except asyncio.CancelledError:
                cancelled = True
    finally:
        release.set()
        if pending is not None and not pending.done():
            pending.cancel()
            try:
                await asyncio.wait_for(pending, timeout=2)
            except BaseException:
                pass
        try:
            await asyncio.wait_for(stream.aclose(), timeout=2)
        except BaseException:
            pass

    assert cancelled
    assert close_attempts == ["response", "client"]


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


@pytest.mark.asyncio
@pytest.mark.unit
async def test_google_achat_uses_httpx_async_request_and_normalizes_final_usage():
    response = FakeResponse(
        json_data={
            "modelVersion": "gemini-3.1-pro-preview",
            "candidates": [{
                "content": {"parts": [{"text": "Hello"}]},
                "finishReason": "STOP",
            }],
            "usageMetadata": {
                "promptTokenCount": 11,
                "candidatesTokenCount": 3,
                "thoughtsTokenCount": 2,
                "totalTokenCount": 25,
            },
        }
    )
    client = FakeAsyncClient(response=response)

    with patch.object(httpx, "AsyncClient", return_value=client):
        result = await GoogleAdapter(
            api_key="test_api_key",
            model="gemini-3.1-pro-preview",
        ).achat([UserMessage("hi")])

    assert result.content == "Hello"
    assert result.usage.input_tokens == 11
    assert result.usage.output_tokens == 5
    assert result.usage.total_tokens == 25
    assert result.cost_input == pytest.approx(22e-6)
    assert result.cost_output == pytest.approx(60e-6)
    assert result.cost_total == pytest.approx(82e-6)
    assert len(client.post_calls) == 1
    assert client.post_calls[0][0].endswith("/models/gemini-3.1-pro-preview:generateContent")
    assert response.closed is True
    assert client.closed is True


@pytest.mark.asyncio
@pytest.mark.unit
async def test_google_astream_chat_uses_httpx_and_retains_usage_until_completion():
    first = {
        "candidates": [{"content": {"parts": [{"text": "Hi"}]}}],
        "usageMetadata": {
            "promptTokenCount": 11,
            "candidatesTokenCount": 3,
            "thoughtsTokenCount": 2,
            "totalTokenCount": 25,
            "cachedContentTokenCount": 4,
        },
    }
    final = {
        "candidates": [{"content": {"parts": []}, "finishReason": "STOP"}],
        "usageMetadata": {
            "promptTokenCount": 12,
            "candidatesTokenCount": 4,
            "totalTokenCount": 30,
            "cachedContentTokenCount": None,
        },
    }
    response = FakeResponse(
        [
            f"data: {json.dumps(first)}".encode(),
            b"",
            f"data: {json.dumps(final)}".encode(),
            b"",
        ]
    )
    client = FakeAsyncClient(response=response)
    done = []

    with patch.object(httpx, "AsyncClient", return_value=client):
        output = [
            text
            async for text in GoogleAdapter(
                api_key="test_api_key",
                model="gemini-3.1-pro-preview",
            ).astream_chat(
                [UserMessage("hi")],
                buffer_chars=2,
                on_done=done.append,
            )
        ]

    assert output == ["Hi"]
    assert len(done) == 1
    assert done[0].usage.input_tokens == 12
    assert done[0].usage.output_tokens == 6
    assert done[0].usage.total_tokens == 30
    assert done[0].usage.cached_tokens is None
    assert done[0].cost_input is None
    assert done[0].cost_output == pytest.approx(72e-6)
    assert done[0].cost_total is None
    assert client.send_calls[0][1] is True
    assert client.send_calls[0][0].url.path.endswith(
        "/models/gemini-3.1-pro-preview:streamGenerateContent"
    )
    assert response.closed is True
    assert client.closed is True


@pytest.mark.asyncio
@pytest.mark.unit
async def test_google_astream_chat_cancellation_skips_on_done_and_closes_resources():
    response = BlockingResponse()
    client = FakeAsyncClient(response=response)
    adapter = GoogleAdapter(api_key="test_api_key", model="gemini-3.1-pro-preview")
    done = []

    async def consume():
        async for _ in adapter.astream_chat(
            [UserMessage("hi")],
            on_done=done.append,
        ):
            pass

    with patch.object(httpx, "AsyncClient", return_value=client):
        task = asyncio.create_task(consume())
        await response.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert done == []
    assert response.closed is True
    assert client.closed is True


@pytest.mark.asyncio
@pytest.mark.unit
async def test_google_astream_chat_http_failure_skips_on_done_and_closes_resources():
    response = FakeResponse(status_code=500)
    client = FakeAsyncClient(response=response)
    adapter = GoogleAdapter(api_key="test_api_key", model="gemini-3.1-pro-preview")
    done = []

    with patch.object(httpx, "AsyncClient", return_value=client):
        with pytest.raises(LLMAPIServerError):
            [
                text
                async for text in adapter.astream_chat(
                    [UserMessage("hi")],
                    on_done=done.append,
                )
            ]

    assert done == []
    assert response.closed is True
    assert client.closed is True
