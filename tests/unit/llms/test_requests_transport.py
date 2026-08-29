"""Tests for the default synchronous ``requests`` transport."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
import requests

from src.llm_api_adapter.errors.llm_api_error import (
    LLMAPIRateLimitError,
    LLMAPITimeoutError,
)
from src.llm_api_adapter.llms.anthropic.sync_client import ClaudeSyncClient
from src.llm_api_adapter.llms.google.sync_client import GeminiSyncClient
from src.llm_api_adapter.llms.openai.sync_client import OpenAISyncClient
from src.llm_api_adapter.llms.requests_transport import RequestsSyncTransport
from src.llm_api_adapter.llms.transports import (
    MultipartFile,
    MultipartForm,
    SSEEvent,
    TransportRequest,
    multipart_headers,
)


class FakeResponse:
    def __init__(
        self,
        *,
        body: dict | None = None,
        lines: list[str] | None = None,
        status_code: int = 200,
    ) -> None:
        self.body = body or {}
        self.lines = lines or []
        self.status_code = status_code
        self.close = Mock()

    def json(self) -> dict:
        return self.body

    def iter_lines(self, decode_unicode: bool = True):
        _ = decode_unicode
        return iter(self.lines)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(response=self)


@pytest.mark.unit
@pytest.mark.parametrize(
    "client_class",
    (OpenAISyncClient, ClaudeSyncClient, GeminiSyncClient),
)
def test_builtin_sync_clients_default_to_requests_transport(client_class):
    client = client_class(api_key="test-key")

    assert isinstance(client._sync_transport, RequestsSyncTransport)


@pytest.mark.unit
def test_requests_transport_posts_json_and_closes_response():
    response = FakeResponse(body={"answer": "ok"})
    request = TransportRequest(
        url="https://example.test/messages",
        headers={"Authorization": "Bearer test"},
        payload={"message": "Hello"},
        timeout=3.0,
    )

    with patch(
        "src.llm_api_adapter.llms.requests_transport.requests.post",
        return_value=response,
    ) as mock_post:
        assert RequestsSyncTransport().post_json(request).json() == {"answer": "ok"}

    mock_post.assert_called_once_with(
        "https://example.test/messages",
        headers={"Authorization": "Bearer test"},
        json={"message": "Hello"},
        timeout=3.0,
    )
    response.close.assert_called_once_with()


@pytest.mark.unit
def test_requests_transport_posts_multipart_and_closes_response():
    response = FakeResponse(body={"id": "file_123"})
    request = TransportRequest(
        url="https://example.test/files",
        headers={
            "Authorization": "Bearer test",
            "Content-Type": "application/json",
        },
        timeout=3.0,
    )
    form = MultipartForm(
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

    with patch(
        "src.llm_api_adapter.llms.requests_transport.requests.post",
        return_value=response,
    ) as mock_post:
        assert RequestsSyncTransport().post_multipart(request, form).json() == {
            "id": "file_123"
        }

    mock_post.assert_called_once_with(
        "https://example.test/files",
        headers={"Authorization": "Bearer test"},
        data=[("purpose", "documents")],
        files=[("file", ("report.pdf", b"%PDF-test", "application/pdf"))],
        timeout=3.0,
    )
    response.close.assert_called_once_with()


@pytest.mark.unit
def test_multipart_form_produces_a_valid_requests_body():
    form = MultipartForm(
        fields=(("purpose", "documents"),),
        files=(MultipartFile("file", "note.txt", b"hello", "text/plain"),),
    )
    prepared = requests.Request(
        "POST",
        "https://example.test/files",
        headers=multipart_headers(
            {"Authorization": "Bearer test", "Content-Type": "application/json"}
        ),
        data=form.fields_list(),
        files=form.files_list(),
    ).prepare()

    assert prepared.headers["Content-Type"].startswith("multipart/form-data; boundary=")
    assert b'name="purpose"' in prepared.body
    assert b'documents' in prepared.body
    assert b'filename="note.txt"' in prepared.body
    assert b"hello" in prepared.body


@pytest.mark.unit
def test_requests_transport_delegates_http_error_to_provider_handler():
    response = FakeResponse(status_code=429)
    request = TransportRequest(url="https://example.test/messages")
    observed = []

    def provider_handler(error):
        observed.append(error.response)
        raise LLMAPIRateLimitError(detail="provider mapping")

    with patch(
        "src.llm_api_adapter.llms.requests_transport.requests.post",
        return_value=response,
    ), pytest.raises(LLMAPIRateLimitError, match="provider mapping"):
        RequestsSyncTransport().post_json(
            request,
            http_error_handler=provider_handler,
        )

    assert observed == [response]
    response.close.assert_called_once_with()


@pytest.mark.unit
def test_requests_transport_delegates_multipart_http_error_to_provider_handler():
    response = FakeResponse(status_code=429)
    request = TransportRequest(url="https://example.test/files")
    form = MultipartForm(files=(MultipartFile("file", "note.txt", b"hello"),))
    observed = []

    def provider_handler(error):
        observed.append(error.response)
        raise LLMAPIRateLimitError(detail="provider mapping")

    with patch(
        "src.llm_api_adapter.llms.requests_transport.requests.post",
        return_value=response,
    ), pytest.raises(LLMAPIRateLimitError, match="provider mapping"):
        RequestsSyncTransport().post_multipart(
            request,
            form,
            http_error_handler=provider_handler,
        )

    assert observed == [response]
    response.close.assert_called_once_with()


@pytest.mark.unit
def test_requests_transport_maps_multipart_timeouts():
    request = TransportRequest(url="https://example.test/files")
    form = MultipartForm(files=(MultipartFile("file", "note.txt", b"hello"),))

    with patch(
        "src.llm_api_adapter.llms.requests_transport.requests.post",
        side_effect=requests.exceptions.Timeout("timed out"),
    ), pytest.raises(LLMAPITimeoutError):
        RequestsSyncTransport().post_multipart(request, form)


@pytest.mark.unit
def test_requests_transport_stops_sse_and_closes_response_on_early_close():
    response = FakeResponse(
        lines=[
            'data: {"sequence": 1}',
            "",
            'data: {"sequence": 2}',
            "",
        ]
    )
    request = TransportRequest(url="https://example.test/stream")

    with patch(
        "src.llm_api_adapter.llms.requests_transport.requests.post",
        return_value=response,
    ) as mock_post:
        events = RequestsSyncTransport().post_sse(request)
        assert next(events) == SSEEvent(event=None, data={"sequence": 1})
        events.close()

    assert mock_post.call_args.kwargs["stream"] is True
    response.close.assert_called_once_with()
