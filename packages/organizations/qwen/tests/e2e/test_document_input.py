"""Qwen-only E2E checks that do not fit the portable Core feature profile."""

from importlib.metadata import PackageNotFoundError, version
from unittest.mock import patch

import pytest

from llm_api_adapter.models.messages.file_parts import DocumentPart
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter
from tests.e2e import harness


def _assert_document_parts_are_rejected_before_transport(
    *,
    adapter: UniversalLLMAPIAdapter,
    transport_call_count,
) -> None:
    """Assert Qwen rejects URL and byte PDFs before Messages transport."""
    documents = (
        DocumentPart(url="https://example.com/report.pdf"),
        DocumentPart(data=b"%PDF-qwen", media_type="application/pdf"),
    )
    baseline_calls = transport_call_count()
    operations = {
        "chat": lambda message: adapter.chat(
            [message],
            max_tokens=64,
            workspace_id="frankfurt-workspace",
        ),
        "stream_chat": lambda message: adapter.stream_chat(
            [message],
            max_tokens=64,
            workspace_id="frankfurt-workspace",
        ),
    }

    for operation_name, operation in operations.items():
        for document in documents:
            message = harness.make_document_message("Summarize this PDF.", document)
            with pytest.raises(
                ValueError,
                match="DocumentPart; PDF and OCR are unavailable",
            ):
                operation(message)
            assert transport_call_count() == baseline_calls, operation_name


@pytest.mark.e2e
@pytest.mark.e2e_qwen
def test_qwen_document_parts_are_rejected_before_messages_transport():
    """Keep rejected PDFs local while exercising the installed plugin facade."""
    try:
        version("llm-api-adapter-qwen")
    except PackageNotFoundError:
        pytest.skip("llm-api-adapter-qwen is not installed")

    from llm_api_adapter_qwen.clients.sync_client import QwenMessagesSyncClient

    adapter = UniversalLLMAPIAdapter(
        organization="qwen",
        model="qwen3.8-max",
        api_key="qwen-e2e-non-live-key",
    )
    with (
        patch.object(QwenMessagesSyncClient, "chat") as chat,
        patch.object(QwenMessagesSyncClient, "stream") as stream,
    ):
        _assert_document_parts_are_rejected_before_transport(
            adapter=adapter,
            transport_call_count=lambda: chat.call_count + stream.call_count,
        )
