"""Credential-free live-lane boundary checks for the installed Z.ai package."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from unittest.mock import patch

import pytest

from llm_api_adapter.models.messages.chat_message import UserMessage
from llm_api_adapter.models.messages.file_parts import DocumentPart
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter


_MODEL = "glm-5.3-flash"


class _ResponseModel:
    """Response-model stand-in for the unsupported structured-output boundary."""


def _require_installed_package() -> None:
    try:
        version("llm-api-adapter-zai")
    except PackageNotFoundError:
        pytest.skip("llm-api-adapter-zai is not installed")


@pytest.mark.e2e
@pytest.mark.e2e_zai
@pytest.mark.parametrize(
    "request_kwargs",
    [
        {"json_schema": {"type": "object", "properties": {}}},
        {"response_model": _ResponseModel},
    ],
    ids=["json-schema", "response-model"],
)
def test_zai_rejects_structured_output_before_provider_transport(request_kwargs):
    """The unsupported portable output forms must stay a local boundary."""
    _require_installed_package()
    adapter = UniversalLLMAPIAdapter(
        organization="zai",
        model=_MODEL,
        api_key="zai-boundary-test-key",
    )

    from llm_api_adapter_zai.clients.sync_client import ZaiSyncClient

    with patch.object(ZaiSyncClient, "chat") as chat:
        with pytest.raises(NotImplementedError, match="structured output"):
            adapter.chat([UserMessage("Return JSON")], **request_kwargs)
        chat.assert_not_called()


@pytest.mark.e2e
@pytest.mark.e2e_zai
@pytest.mark.parametrize(
    "document",
    [
        DocumentPart(url="https://example.com/zai-e2e.pdf"),
        DocumentPart(data=b"%PDF-zai-e2e", media_type="application/pdf"),
    ],
    ids=["document-url", "document-bytes"],
)
def test_zai_rejects_document_forms_before_provider_transport(document):
    """Both withheld PDF forms must remain local until the live document gate."""
    _require_installed_package()
    adapter = UniversalLLMAPIAdapter(
        organization="zai",
        model=_MODEL,
        api_key="zai-boundary-test-key",
    )

    from llm_api_adapter_zai.clients.sync_client import ZaiSyncClient

    message = UserMessage("Summarize this document", files=[document])
    with (
        patch.object(ZaiSyncClient, "chat") as chat,
        patch.object(ZaiSyncClient, "stream") as stream,
    ):
        with pytest.raises(ValueError, match="DocumentPart|document"):
            adapter.chat([message])
        with pytest.raises(ValueError, match="DocumentPart|document"):
            list(adapter.stream_chat([message]))
        chat.assert_not_called()
        stream.assert_not_called()
