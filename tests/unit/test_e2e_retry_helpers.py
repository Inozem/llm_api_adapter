from unittest.mock import AsyncMock, MagicMock
from types import SimpleNamespace

import httpx
import pytest
import requests

from llm_api_adapter.errors import LLMAPIClientError, LLMAPITimeoutError
from tests.e2e import conftest as e2e_conftest
from tests.e2e import harness as e2e_harness


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error_type", "cause"),
    [
        (LLMAPITimeoutError, None),
        (LLMAPIClientError, requests.exceptions.ConnectionError("unreachable")),
        (LLMAPIClientError, httpx.ConnectError("unreachable")),
    ],
)
async def test_async_e2e_retry_helper_retries_transient_errors(
    monkeypatch, error_type, cause
):
    expected_response = SimpleNamespace(finish_reason=None)
    error = error_type()
    error.__cause__ = cause
    adapter = AsyncMock()
    adapter.achat.side_effect = [error, expected_response]
    sleep = AsyncMock()
    monkeypatch.setattr(e2e_harness.asyncio, "sleep", sleep)

    retry = e2e_harness.async_chat_with_transient_retry

    assert await retry(adapter, request="value") is expected_response
    assert adapter.achat.await_args_list[0].kwargs == {"request": "value"}
    assert adapter.achat.await_count == 2
    sleep.assert_awaited_once_with(2)


@pytest.mark.unit
@pytest.mark.parametrize("connection_failures", [0, 1, 4])
def test_e2e_timeout_check_uses_shared_retry_without_retrying_expected_timeout(
    monkeypatch, connection_failures
):
    connection_error = LLMAPIClientError()
    connection_error.__cause__ = requests.exceptions.ConnectionError("unreachable")
    timeout_error = LLMAPITimeoutError()
    adapter = MagicMock()
    adapter.chat.side_effect = [connection_error] * connection_failures + [timeout_error]
    sleep = MagicMock()
    monkeypatch.setattr(e2e_harness.time, "sleep", sleep)
    expected_error = connection_error if connection_failures == 4 else timeout_error

    with pytest.raises(type(expected_error)) as raised:
        e2e_harness.chat_with_transient_retry(
            adapter, expected_error=LLMAPITimeoutError, request="value"
        )

    assert raised.value is expected_error
    expected_calls = min(connection_failures + 1, 4)
    assert adapter.chat.call_count == expected_calls
    assert all(call.kwargs == {"request": "value"} for call in adapter.chat.call_args_list)
    assert [call.args[0] for call in sleep.call_args_list] == [2, 4, 8][:expected_calls - 1]


@pytest.mark.unit
@pytest.mark.parametrize("cause", [None, requests.exceptions.HTTPError("bad request")])
def test_e2e_retry_helper_propagates_other_client_errors(monkeypatch, cause):
    error = LLMAPIClientError()
    error.__cause__ = cause
    adapter = MagicMock()
    adapter.chat.side_effect = error
    sleep = MagicMock()
    monkeypatch.setattr(e2e_harness.time, "sleep", sleep)

    with pytest.raises(LLMAPIClientError) as raised:
        e2e_harness.chat_with_transient_retry(adapter, request="value")

    assert raised.value is error
    adapter.chat.assert_called_once_with(request="value")
    sleep.assert_not_called()


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
def test_qwen_e2e_profile_keeps_lane_install_and_operation_settings():
    profile = e2e_conftest._QWEN_E2E_PROFILE

    assert profile.name == "qwen"
    assert profile.organization_names == ("qwen",)
    assert profile.distribution == "llm-api-adapter-qwen"
    assert profile.operation_kwargs_env == (("workspace_id", "QWEN_WORKSPACE_ID"),)
    assert not hasattr(profile, "supported_features")


@pytest.mark.unit
def test_kimi_e2e_profile_keeps_lane_install_and_key_settings():
    profile = e2e_conftest._KIMI_E2E_PROFILE

    assert profile.name == "kimi"
    assert profile.organization_names == ("kimi",)
    assert profile.distribution == "llm-api-adapter-kimi"
    assert profile.operation_kwargs_env == ()
    assert profile.api_key_is_required
    assert profile.missing_api_key_is_usage_error
    assert not hasattr(profile, "supported_features")
