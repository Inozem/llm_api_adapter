"""Installed-distribution checks for DeepSeek document inputs rejected locally."""

from importlib.metadata import PackageNotFoundError, version
from unittest.mock import patch

import pytest

from llm_api_adapter.models.messages.chat_message import UserMessage
from llm_api_adapter.models.messages.file_parts import DocumentPart
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter


def _assert_unsupported_document_inputs_do_not_start_responses(
    *,
    adapter: UniversalLLMAPIAdapter,
    client_call_count,
) -> None:
    """Keep both PDF forms local after the organization package is installed."""
    unsupported_parts = (
        DocumentPart(url="https://example.com/deepseek.pdf"),
        DocumentPart(data=b"%PDF-deepseek", media_type="application/pdf"),
    )
    baseline_calls = client_call_count()
    operations = {
        "chat": lambda message: adapter.chat([message], max_tokens=64),
        "stream_chat": lambda message: list(
            adapter.stream_chat([message], max_tokens=64)
        ),
    }

    for operation_name, operation in operations.items():
        for file_part in unsupported_parts:
            message = UserMessage("Read this document.", files=[file_part])
            with pytest.raises(ValueError, match="DocumentPart|document|file"):
                operation(message)
            assert client_call_count() == baseline_calls, operation_name


@pytest.mark.e2e
@pytest.mark.e2e_deepseek
def test_deepseek_file_contract_rejects_documents_before_transport():
    """Verify the installed plugin rejects PDFs without contacting DeepSeek."""
    try:
        version("llm-api-adapter-deepseek")
    except PackageNotFoundError:
        pytest.skip("llm-api-adapter-deepseek is not installed")

    from llm_api_adapter_deepseek.clients.sync_client import (
        DeepSeekResponsesSyncClient,
    )

    adapter = UniversalLLMAPIAdapter(
        organization="deepseek",
        model="deepseek-flash",
        api_key="deepseek-e2e-non-live-key",
    )
    with (
        patch.object(DeepSeekResponsesSyncClient, "create") as create,
        patch.object(DeepSeekResponsesSyncClient, "stream") as stream,
    ):
        _assert_unsupported_document_inputs_do_not_start_responses(
            adapter=adapter,
            client_call_count=lambda: create.call_count + stream.call_count,
        )
