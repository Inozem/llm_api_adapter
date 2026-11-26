"""Private DeepSeek Files uploads use Core multipart transports safely."""

from __future__ import annotations

import asyncio
from json import JSONDecodeError
from types import SimpleNamespace
from typing import Any

import pytest
import requests

from llm_api_adapter.errors.llm_api_error import (
    LLMAPIAuthorizationError,
    LLMAPIClientError,
    LLMAPIServerError,
)
from llm_api_adapter.llms import async_streaming, httpx_transport
from llm_api_adapter.llms.transports import JSONResponse
from llm_api_adapter_deepseek.clients.async_client import (
    DeepSeekResponsesAsyncClient,
)
from llm_api_adapter_deepseek.clients.sync_client import (
    DEEPSEEK_FILES_URL,
    DeepSeekResponsesSyncClient,
)


PDF = b"%PDF-1.7\nprivate-pdf-bytes"


@pytest.mark.unit
@pytest.mark.parametrize("transport", ["requests", "httpx"])
def test_sync_upload_uses_core_multipart_transport(
    monkeypatch: pytest.MonkeyPatch,
    transport: str,
) -> None:
    calls: list[dict[str, Any]] = []

    class Response:
        closed = False

        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, str]:
            return {"id": "file_receipt_private"}

        def close(self) -> None:
            self.closed = True

    response = Response()

    if transport == "requests":
        monkeypatch.setattr(
            requests,
            "post",
            lambda url, **kwargs: calls.append({"url": url, **kwargs}) or response,
        )
    else:
        pytest.importorskip("httpx")

        class Client:
            def post(self, url: str, **kwargs: Any) -> Response:
                calls.append({"url": url, **kwargs})
                return response

            def close(self) -> None:
                pass

        monkeypatch.setattr(
            httpx_transport,
            "_require_httpx",
            lambda: SimpleNamespace(Client=Client),
        )

    client = DeepSeekResponsesSyncClient("native-secret", transport=transport)
    assert client._upload_pdf(PDF) == "file_receipt_private"

    call = calls[0]
    assert call["url"] == DEEPSEEK_FILES_URL
    assert call["headers"] == {"Authorization": "Bearer native-secret"}
    assert call["timeout"] == 300.0
    assert call["data"] == [
        ("purpose", "rich_layout_recognition"),
        ("expires_after[anchor]", "created_at"),
        ("expires_after[seconds]", "3600"),
    ]
    assert call["files"] == [
        ("file", ("document.pdf", PDF, "application/pdf")),
    ]
    assert response.closed


@pytest.mark.unit
def test_sync_upload_honors_timeout_and_rejects_bad_receipts() -> None:
    class Transport:
        def __init__(self, payload: Any) -> None:
            self.payload = payload
            self.request = None
            self.form = None

        def post_multipart(self, request, form, *, http_error_handler=None):
            self.request = request
            self.form = form
            return JSONResponse(self.payload)

    transport = Transport({"id": "file_ok"})
    client = DeepSeekResponsesSyncClient("key")
    client._sync_transport = transport
    assert client._upload_pdf(PDF, timeout=17.5) == "file_ok"
    assert transport.request.timeout == 17.5

    for payload in (None, [], {}, {"id": ""}, {"id": "  "}, {"id": 12}):
        client._sync_transport = Transport(payload)
        with pytest.raises(LLMAPIClientError) as caught:
            client._upload_pdf(PDF)
        assert "file upload failed" in str(caught.value).lower()
        assert "file_ok" not in str(caught.value)


@pytest.mark.unit
def test_sync_upload_redacts_transport_error_details() -> None:
    class Transport:
        def post_multipart(self, *args, **kwargs):
            raise LLMAPIServerError(detail=f"{PDF!r} file_receipt_private")

    client = DeepSeekResponsesSyncClient("key")
    client._sync_transport = Transport()
    with pytest.raises(LLMAPIServerError) as caught:
        client._upload_pdf(PDF)
    assert str(caught.value) == "Server error occurred. Detail: DeepSeek file upload failed"
    assert "private-pdf-bytes" not in str(caught.value)
    assert "file_receipt_private" not in str(caught.value)


@pytest.mark.unit
def test_sync_upload_preserves_unexpected_errors_and_sanitizes_bad_json() -> None:
    failure = RuntimeError("unexpected local failure")

    class FailingTransport:
        def post_multipart(self, *args, **kwargs):
            raise failure

    client = DeepSeekResponsesSyncClient("key")
    client._sync_transport = FailingTransport()
    with pytest.raises(RuntimeError) as caught_failure:
        client._upload_pdf(PDF)
    assert caught_failure.value is failure

    class InvalidResponse:
        def __init__(self, error: Exception) -> None:
            self.error = error

        def json(self) -> Any:
            raise self.error

    for decode_error in (
        JSONDecodeError("bad JSON", "synthetic-file-id", 0),
        UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"),
    ):
        class ResponseTransport:
            def post_multipart(self, *args, **kwargs):
                return InvalidResponse(decode_error)

        client._sync_transport = ResponseTransport()
        with pytest.raises(LLMAPIClientError) as caught_json:
            client._upload_pdf(PDF)
        assert caught_json.value.detail == "DeepSeek file upload failed"
        assert "synthetic-file-id" not in str(caught_json.value)


@pytest.mark.unit
def test_upload_http_error_keeps_native_category_and_redacts_provider_detail() -> None:
    class Response:
        status_code = 401

        @staticmethod
        def json() -> dict[str, Any]:
            return {
                "error": {
                    "type": "invalid_api_key",
                    "message": f"rejected {PDF!r} file_receipt_private",
                },
            }

    error = SimpleNamespace(response=Response())
    with pytest.raises(LLMAPIAuthorizationError) as caught:
        DeepSeekResponsesSyncClient._handle_file_upload_http_error(error)
    assert caught.value.detail == "DeepSeek file upload failed"
    assert "file_receipt_private" not in str(caught.value)
    assert "private-pdf-bytes" not in str(caught.value)


@pytest.mark.unit
def test_async_upload_uses_core_multipart_helper_and_is_cancellable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    httpx = pytest.importorskip("httpx")
    calls: list[dict[str, Any]] = []

    class Response:
        closed = False

        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, str]:
            return {"id": "async_receipt_private"}

        async def aclose(self) -> None:
            self.closed = True

    response = Response()

    class AsyncClient:
        async def post(self, url: str, **kwargs: Any) -> Response:
            calls.append({"url": url, **kwargs})
            return response

        async def aclose(self) -> None:
            pass

    monkeypatch.setattr(
        async_streaming,
        "_require_httpx",
        lambda: SimpleNamespace(
            AsyncClient=AsyncClient,
            Request=httpx.Request,
            TimeoutException=httpx.TimeoutException,
            HTTPStatusError=httpx.HTTPStatusError,
            RequestError=httpx.RequestError,
        ),
    )
    client = DeepSeekResponsesAsyncClient("native-secret")

    async def run_upload() -> str:
        return await client._upload_pdf(PDF)

    assert asyncio.run(run_upload()) == "async_receipt_private"
    call = calls[0]
    assert call["url"] == DEEPSEEK_FILES_URL
    assert call["headers"]["authorization"] == "Bearer native-secret"
    assert call["timeout"] == 300.0
    body = call["content"]
    assert body.index(b"rich_layout_recognition") < body.index(b"expires_after[anchor]")
    assert body.index(b"expires_after[seconds]") < body.index(b"name=\"file\"")
    assert body.index(b"rich_layout_recognition") < body.index(b"name=\"file\"")
    assert b'filename="document.pdf"' in body
    assert b"application/pdf" in body
    assert PDF in body
    assert response.closed

    async def run_explicit_timeout() -> str:
        return await client._upload_pdf(PDF, timeout=23)

    assert asyncio.run(run_explicit_timeout()) == "async_receipt_private"
    assert calls[1]["timeout"] == 23
    assert calls[1]["headers"]["authorization"] == "Bearer native-secret"

    started = asyncio.Event()
    closed = []

    class BlockingAsyncClient:
        async def post(self, url: str, **kwargs: Any) -> Any:
            started.set()
            await asyncio.Event().wait()

        async def aclose(self) -> None:
            closed.append(True)

    monkeypatch.setattr(
        async_streaming,
        "_require_httpx",
        lambda: SimpleNamespace(
            AsyncClient=BlockingAsyncClient,
            Request=httpx.Request,
            TimeoutException=httpx.TimeoutException,
            HTTPStatusError=httpx.HTTPStatusError,
            RequestError=httpx.RequestError,
        ),
    )

    async def cancel_upload() -> None:
        task = asyncio.create_task(client._upload_pdf(PDF, timeout=23))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_upload())
    assert closed == [True]


@pytest.mark.unit
def test_async_upload_rejects_invalid_receipts_and_redacts_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run_case(payload: Any) -> None:
        async def helper(*args: Any, **kwargs: Any) -> Any:
            return payload

        monkeypatch.setattr(
            "llm_api_adapter_deepseek.clients.async_client.async_multipart_request",
            helper,
        )
        with pytest.raises(LLMAPIClientError) as caught:
            await DeepSeekResponsesAsyncClient("key")._upload_pdf(PDF)
        assert "file upload failed" in str(caught.value).lower()
        assert "async_receipt_private" not in str(caught.value)

    async def run_error() -> None:
        async def helper(*args: Any, **kwargs: Any) -> Any:
            raise LLMAPIServerError(detail=f"{PDF!r} async_receipt_private")

        monkeypatch.setattr(
            "llm_api_adapter_deepseek.clients.async_client.async_multipart_request",
            helper,
        )
        with pytest.raises(LLMAPIServerError) as caught:
            await DeepSeekResponsesAsyncClient("key")._upload_pdf(PDF)
        assert "private-pdf-bytes" not in str(caught.value)
        assert "async_receipt_private" not in str(caught.value)

    async def run_all() -> None:
        for payload in (None, [], {}, {"id": ""}, {"id": "  "}, {"id": 12}):
            await run_case(payload)
        await run_error()

    asyncio.run(run_all())


@pytest.mark.unit
def test_async_upload_preserves_optional_httpx_import_error_and_programming_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    programming_error = RuntimeError("unexpected local failure")

    def missing_httpx() -> Any:
        raise ImportError(async_streaming.ASYNC_DEPENDENCY_MESSAGE)

    monkeypatch.setattr(async_streaming, "_require_httpx", missing_httpx)
    with pytest.raises(ImportError, match="llm-api-adapter\\[async\\]"):
        asyncio.run(DeepSeekResponsesAsyncClient("key")._upload_pdf(PDF))

    async def helper(*args: Any, **kwargs: Any) -> Any:
        raise programming_error

    monkeypatch.setattr(
        "llm_api_adapter_deepseek.clients.async_client.async_multipart_request",
        helper,
    )
    with pytest.raises(RuntimeError) as caught:
        asyncio.run(DeepSeekResponsesAsyncClient("key")._upload_pdf(PDF))
    assert caught.value is programming_error


@pytest.mark.unit
@pytest.mark.parametrize(
    "failure",
    [
        JSONDecodeError("invalid payload", "secret-id private-pdf", 0),
        UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"),
    ],
)
def test_async_upload_sanitizes_malformed_json_only(
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
) -> None:
    async def helper(*args: Any, **kwargs: Any) -> Any:
        raise failure

    monkeypatch.setattr(
        "llm_api_adapter_deepseek.clients.async_client.async_multipart_request",
        helper,
    )
    with pytest.raises(LLMAPIClientError) as caught:
        asyncio.run(DeepSeekResponsesAsyncClient("key")._upload_pdf(PDF))
    assert caught.value.detail == "DeepSeek file upload failed"
    assert "secret-id" not in str(caught.value)
    assert "private-pdf" not in str(caught.value)


@pytest.mark.unit
def test_requests_multipart_http_failure_redacts_body_from_logs_and_sdk_error(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    marker = "private-pdf-marker"
    file_id = "synthetic-file-id"

    class Response:
        status_code = 400
        text = f"provider echoed {marker} and {file_id}"
        closed = False

        def raise_for_status(self) -> None:
            raise requests.HTTPError(
                f"400 Client Error: bad request for url: {DEEPSEEK_FILES_URL}",
                response=self,
            )

        def json(self) -> dict[str, Any]:
            return {
                "error": {
                    "type": "invalid_request_error",
                    "message": self.text,
                },
            }

        def close(self) -> None:
            self.closed = True

    response = Response()
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: response)
    client = DeepSeekResponsesSyncClient("native-secret")

    with pytest.raises(LLMAPIClientError) as caught:
        client._upload_pdf(marker.encode())

    assert caught.value.detail == "DeepSeek file upload failed"
    assert marker not in str(caught.value)
    assert file_id not in str(caught.value)
    assert marker not in caplog.text
    assert file_id not in caplog.text
    assert response.closed
