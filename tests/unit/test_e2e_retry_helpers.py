from unittest.mock import AsyncMock, MagicMock
from types import SimpleNamespace

import pytest

from llm_api_adapter.errors import LLMAPITimeoutError
from tests.e2e import conftest as e2e_conftest
from tests.e2e import harness as e2e_harness


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_e2e_retry_helper_retries_timeout(monkeypatch):
    expected_response = SimpleNamespace(finish_reason=None)
    adapter = AsyncMock()
    adapter.achat.side_effect = [LLMAPITimeoutError(), expected_response]
    sleep = AsyncMock()
    monkeypatch.setattr(e2e_harness.asyncio, "sleep", sleep)

    retry = e2e_harness.async_chat_with_transient_retry

    assert await retry(adapter, request="value") is expected_response
    assert adapter.achat.await_args_list[0].kwargs == {"request": "value"}
    assert adapter.achat.await_count == 2
    sleep.assert_awaited_once_with(2)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_profiled_e2e_adapter_forwards_operation_kwargs_to_every_operation():
    adapter = MagicMock()
    adapter.achat = AsyncMock(return_value="async response")

    async def astream_chat(**kwargs):
        yield kwargs

    adapter.astream_chat = astream_chat
    profiled = e2e_harness.ProfiledE2EAdapter(
        adapter,
        {"workspace_id": "frankfurt-workspace"},
    )

    profiled.chat(messages="sync")
    profiled.stream_chat(messages="stream")
    assert await profiled.achat(messages="async") == "async response"
    assert [item async for item in profiled.astream_chat(messages="async-stream")] == [
        {"messages": "async-stream", "workspace_id": "frankfurt-workspace"}
    ]

    assert adapter.chat.call_args.kwargs == {
        "messages": "sync",
        "workspace_id": "frankfurt-workspace",
    }
    assert adapter.stream_chat.call_args.kwargs == {
        "messages": "stream",
        "workspace_id": "frankfurt-workspace",
    }
    assert adapter.achat.await_args.kwargs == {
        "messages": "async",
        "workspace_id": "frankfurt-workspace",
    }


@pytest.mark.unit
def test_qwen_e2e_profile_selects_only_portable_document_free_features():
    profile = e2e_conftest._QWEN_E2E_PROFILE

    assert profile.distribution == "llm-api-adapter-qwen"
    assert profile.operation_kwargs_env == (("workspace_id", "QWEN_WORKSPACE_ID"),)
    assert e2e_conftest._profile_supports_features(profile, frozenset({"image_input"}))
    assert not e2e_conftest._profile_supports_features(
        profile,
        frozenset({"document_input"}),
    )
