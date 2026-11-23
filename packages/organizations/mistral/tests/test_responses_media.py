"""Native Mistral image payload behavior over the real Responses adapter."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator, Mapping
import ipaddress
from pathlib import Path
import socket
import sys
from typing import Any

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[2]
for source in (str(PACKAGE_ROOT / "src"), str(REPOSITORY_ROOT / "src"), str(REPOSITORY_ROOT)):
    if source not in sys.path:
        sys.path.insert(0, source)

import llm_api_adapter.adapters.base_adapter as base_adapter_module
import llm_api_adapter_mistral.clients.async_client as async_client_module
from llm_api_adapter.llm_registry.llm_registry import RegistrySpec
from llm_api_adapter.llms.transports import (
    HTTPErrorHandler,
    JSONResponse,
    MultipartForm,
    SSEEvent,
    StreamErrorHandler,
    SyncTransport,
    TransportRequest,
)
from llm_api_adapter.models.messages.chat_message import AIMessage, Messages, UserMessage
from llm_api_adapter.models.messages.file_parts import DocumentPart, ImagePart
from llm_api_adapter_mistral.adapter import MistralAdapter
from llm_api_adapter_mistral.plugin import PLUGIN


MODELS = (
    "mistral-large-4",
    "mistral-small-2603",
    "mistral-medium-3-5",
    "mistral-large-2512",
    "glm-5.3-flash",
    "deepseek-flash",
)
METHODS = ("chat", "achat", "stream_chat", "astream_chat")
IMAGE_BYTES = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de"
    "0000000c49444154789c63f8cfc0000003010100c9fe92ef0000000049454e44ae426082"
)
IMAGE_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNgYPgPAAEDAQAIicLsAAAAAElFTkSuQmCC"
)
IMAGE_BYTES_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)


class MediaTransport(SyncTransport):
    """Capture typed native requests and return controlled Responses results."""

    def __init__(self, model: str) -> None:
        self.requests: list[TransportRequest] = []
        self.response = _completed_response(model)
        self.events = (
            SSEEvent(event="response.output_text.delta", data={"delta": "image received"}),
            SSEEvent(event="response.completed", data={"response": self.response}),
        )

    def post_json(
        self, request: TransportRequest, *, http_error_handler: HTTPErrorHandler | None = None
    ) -> JSONResponse:
        self.requests.append(request)
        return JSONResponse(self.response)

    def post_sse(
        self,
        request: TransportRequest,
        *,
        http_error_handler: HTTPErrorHandler | None = None,
        stream_error_handler: StreamErrorHandler | None = None,
    ) -> Iterator[SSEEvent]:
        self.requests.append(request)
        return iter(self.events)

    def post_multipart(
        self, request: TransportRequest, form: MultipartForm,
        *, http_error_handler: HTTPErrorHandler | None = None,
    ) -> JSONResponse:
        raise AssertionError("Responses images use JSON, not multipart")


@pytest.fixture(autouse=True)
def deny_external_network(monkeypatch: pytest.MonkeyPatch) -> None:
    original_getaddrinfo = socket.getaddrinfo
    original_connect = socket.socket.connect

    def is_loopback(host: Any) -> bool:
        if host in {"localhost", b"localhost"}:
            return True
        if isinstance(host, bytes):
            host = host.decode("ascii", "ignore")
        try:
            return ipaddress.ip_address(host).is_loopback
        except (ValueError, TypeError):
            return False

    def guarded_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> list[tuple[Any, ...]]:
        if not is_loopback(host):
            raise AssertionError("media test attempted external DNS")
        return original_getaddrinfo(host, *args, **kwargs)

    def guarded_connect(connection: socket.socket, address: Any) -> None:
        if isinstance(address, tuple) and address and not is_loopback(address[0]):
            raise AssertionError("media test attempted external network")
        original_connect(connection, address)

    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


@pytest.fixture
def mistral_registry(monkeypatch: pytest.MonkeyPatch) -> RegistrySpec:
    registry = RegistrySpec()
    assert PLUGIN.model_metadata is not None
    assert registry.register_organization_metadata(PLUGIN.model_metadata)
    monkeypatch.setattr(base_adapter_module, "LLM_REGISTRY", registry)
    return registry


def _completed_response(model: str) -> dict[str, Any]:
    return {
        "object": "response", "id": "resp_media", "model": model,
        "status": "completed",
        "output": [{"type": "message", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": "image received"}]}],
    }


def _image_message() -> UserMessage:
    return UserMessage(
        "Inspect in order",
        files=[
            ImagePart(url="https://images.example.test/first.png"),
            ImagePart(data=IMAGE_BYTES, media_type="image/png"),
            ImagePart(url=IMAGE_DATA_URL),
        ],
    )


def _expected_images() -> list[dict[str, str]]:
    return [
        {"type": "input_image", "image_url": "https://images.example.test/first.png", "detail": "auto"},
        {"type": "input_image", "image_url": IMAGE_BYTES_DATA_URL, "detail": "auto"},
        {"type": "input_image", "image_url": IMAGE_DATA_URL, "detail": "auto"},
    ]


def _install_async(monkeypatch: pytest.MonkeyPatch, transport: MediaTransport) -> list[TransportRequest]:
    requests: list[TransportRequest] = []

    async def request(
        url: str, *, headers: Mapping[str, str] | None = None, payload: Any = None,
        timeout: float | None = None, http_error_handler: HTTPErrorHandler | None = None,
    ) -> dict[str, Any]:
        requests.append(TransportRequest(url=url, headers=headers or {}, payload=payload, timeout=timeout))
        return transport.response

    def stream_request(
        url: str, *, headers: Mapping[str, str] | None = None, payload: Any = None,
        timeout: float | None = None, http_error_handler: HTTPErrorHandler | None = None,
        stream_error_handler: StreamErrorHandler | None = None,
    ) -> AsyncIterator[SSEEvent]:
        requests.append(TransportRequest(url=url, headers=headers or {}, payload=payload, timeout=timeout))

        async def source() -> AsyncIterator[SSEEvent]:
            for event in transport.events:
                yield event

        return source()

    monkeypatch.setattr(async_client_module, "async_request", request)
    monkeypatch.setattr(async_client_module, "async_stream_request", stream_request)
    return requests


@pytest.mark.unit
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("method", METHODS)
def test_native_responses_preserves_image_sources_and_adds_default_detail(
    mistral_registry: RegistrySpec,
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    method: str,
) -> None:
    message = _image_message()
    caller_messages = Messages([message])
    before_snapshot = [
        (part.url, part.data, part.media_type) for part in message.files or []
    ]
    common_before = caller_messages.to_openai_responses_input()
    assert all("detail" not in part for part in common_before[0]["content"][1:])

    adapter = MistralAdapter(api_key="synthetic-key", model=model, company="mistral")
    transport = MediaTransport(model)
    adapter._sync_transport = transport
    async_requests = _install_async(monkeypatch, transport)

    if method == "chat":
        assert adapter.chat(caller_messages).content == "image received"
    elif method == "achat":
        assert asyncio.run(adapter.achat(caller_messages)).content == "image received"
    elif method == "stream_chat":
        assert "".join(adapter.stream_chat(caller_messages)) == "image received"
    else:
        async def collect() -> str:
            return "".join([chunk async for chunk in adapter.astream_chat(caller_messages)])

        assert asyncio.run(collect()) == "image received"

    is_async = method in {"achat", "astream_chat"}
    requests = async_requests if is_async else transport.requests
    assert len(requests) == 1
    assert requests[0].url == "https://api.mistral.ai/v1/responses"
    assert adapter.model_spec is not None
    assert requests[0].payload["model"] == (adapter.model_spec.provider_model or model)
    assert (len(transport.requests), len(async_requests)) == ((0, 1) if is_async else (1, 0))
    wire_content = requests[0].payload["input"][0]["content"]
    assert wire_content == [
        {"type": "input_text", "text": "Inspect in order"},
        *_expected_images(),
    ]
    assert [(part.url, part.data, part.media_type) for part in message.files or []] == before_snapshot
    common_after = caller_messages.to_openai_responses_input()
    assert common_after == common_before
    assert all("detail" not in part for part in common_after[0]["content"][1:])


@pytest.mark.unit
def test_native_payload_preserves_explicit_image_detail_without_mutating_raw_input(
    mistral_registry: RegistrySpec,
) -> None:
    from llm_api_adapter_mistral.clients.sync_client import MistralSyncClient

    client = MistralSyncClient(MediaTransport("mistral-large-4"))
    caller_input: list[dict[str, Any]] = [{
        "role": "user",
        "content": [{"type": "input_image", "image_url": "https://images.example.test/a.png", "detail": "high"}],
    }]
    snapshot = [{"role": "user", "content": [dict(caller_input[0]["content"][0])]}]
    payload = client._prepare_responses_payload_for_model("mistral-large-4", {"input": caller_input})
    assert payload["input"][0]["content"][0]["detail"] == "high"
    assert caller_input == snapshot


@pytest.mark.unit
def test_native_responses_keeps_ocr_page_order_and_resets_document_indexes_per_message(
    mistral_registry: RegistrySpec,
) -> None:
    model = MODELS[0]
    response = _completed_response(model)
    ocr_results = [
        {"pages": [{"markdown": "# First"}, {"markdown": "## Second"}], "usage_info": {"pages_processed": 2}},
        {"pages": [{"markdown": "# Third"}], "usage_info": {"pages_processed": 1}},
        {"pages": [{"markdown": "# Fourth"}], "usage_info": {"pages_processed": 1}},
    ]

    class OCRMediaTransport(MediaTransport):
        def post_json(
            self,
            request: TransportRequest,
            *,
            http_error_handler: HTTPErrorHandler | None = None,
        ) -> JSONResponse:
            self.requests.append(request)
            if request.url.endswith("/ocr"):
                return JSONResponse(ocr_results.pop(0))
            return JSONResponse(response)

    messages = Messages([
        UserMessage("First turn", files=[
            DocumentPart(url="https://docs.example.test/one.pdf"),
            DocumentPart(data=b"%PDF two", media_type="application/pdf"),
            ImagePart(data=IMAGE_BYTES, media_type="image/png"),
        ]),
        AIMessage("Earlier answer"),
        UserMessage("Second turn", files=[
            DocumentPart(url="data:application/pdf;base64,cGRm")
        ]),
    ])
    adapter = MistralAdapter(api_key="synthetic-key", model=model, company="mistral")
    transport = OCRMediaTransport(model)
    adapter._sync_transport = transport

    assert adapter.chat(messages).content == "image received"

    assert [request.url for request in transport.requests] == [
        "https://api.mistral.ai/v1/ocr",
        "https://api.mistral.ai/v1/ocr",
        "https://api.mistral.ai/v1/ocr",
        "https://api.mistral.ai/v1/responses",
    ]
    assert [request.payload["model"] for request in transport.requests[:3]] == [
        "mistral-ocr-4-1",
    ] * 3
    assert transport.requests[0].payload["document"]["document_url"] == "https://docs.example.test/one.pdf"
    assert transport.requests[1].payload["document"]["document_url"] == (
        "data:application/pdf;base64,JVBERiB0d28="
    )
    assert transport.requests[2].payload["document"]["document_url"] == (
        "data:application/pdf;base64,cGRm"
    )
    sent = transport.requests[3].payload["input"]
    assert sent[0]["content"][0]["text"] == (
        "First turn\n\n<document index=\"1\">\n# First\n\n---\n\n## Second\n</document>\n\n"
        "<document index=\"2\">\n# Third\n</document>"
    )
    assert sent[0]["content"][1]["image_url"] == IMAGE_BYTES_DATA_URL
    assert sent[1] == {"role": "assistant", "content": "Earlier answer"}
    assert sent[2]["content"] == (
        "Second turn\n\n<document index=\"1\">\n# Fourth\n</document>"
    )
