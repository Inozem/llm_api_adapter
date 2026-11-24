"""Installed-distribution checks for Kimi inputs intentionally rejected locally."""

from importlib.metadata import PackageNotFoundError, version
from collections.abc import Callable
from unittest.mock import patch

import pytest

from llm_api_adapter.models.messages.chat_message import UserMessage
from llm_api_adapter.models.messages.file_parts import DocumentPart, ImagePart
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter


def _assert_unsupported_file_inputs_do_not_start_chat_completions(
    *,
    adapter: UniversalLLMAPIAdapter,
    transport_call_count: Callable[[], int],
) -> None:
    """Keep the public-URL and PDF contract local after plugin installation."""
    unsupported_parts = (
        ImagePart(url="https://example.com/kimi.png"),
        DocumentPart(url="https://example.com/kimi.pdf"),
        DocumentPart(data=b"%PDF-kimi", media_type="application/pdf"),
    )
    baseline_calls = transport_call_count()
    for file_part in unsupported_parts:
        message = UserMessage("Read this file.", files=[file_part])
        with pytest.raises(ValueError):
            adapter.chat([message], max_tokens=64)
        assert transport_call_count() == baseline_calls, "chat"

        with pytest.raises(ValueError):
            stream = adapter.stream_chat([message], max_tokens=64)
            next(stream)
        assert transport_call_count() == baseline_calls, "stream_chat"


@pytest.mark.e2e
@pytest.mark.e2e_kimi
def test_kimi_file_contract_rejects_unsupported_parts_before_transport():
    """Verify the TestPyPI-installed plugin preserves the shared file boundary."""
    try:
        version("llm-api-adapter-kimi")
    except PackageNotFoundError:
        pytest.skip("llm-api-adapter-kimi is not installed")

    adapter = UniversalLLMAPIAdapter(
        organization="kimi",
        model="kimi-k3",
        api_key="kimi-e2e-non-live-key",
    )
    transport = adapter._sync_transport
    with (
        patch.object(
            transport,
            "post_json",
            side_effect=AssertionError("unexpected JSON request"),
        ) as post_json,
        patch.object(
            transport,
            "post_sse",
            side_effect=AssertionError("unexpected SSE request"),
        ) as post_sse,
    ):
        _assert_unsupported_file_inputs_do_not_start_chat_completions(
            adapter=adapter,
            transport_call_count=lambda: post_json.call_count + post_sse.call_count,
        )
