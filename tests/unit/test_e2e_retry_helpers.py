from unittest.mock import AsyncMock
from types import SimpleNamespace

import pytest

from llm_api_adapter.errors import LLMAPITimeoutError
from tests.e2e import conftest as e2e_conftest


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_e2e_retry_helper_retries_timeout(monkeypatch):
    expected_response = SimpleNamespace(finish_reason=None)
    adapter = AsyncMock()
    adapter.achat.side_effect = [LLMAPITimeoutError(), expected_response]
    sleep = AsyncMock()
    monkeypatch.setattr(e2e_conftest.asyncio, "sleep", sleep)

    retry = e2e_conftest.async_chat_with_retry.__wrapped__()

    assert await retry(adapter, request="value") is expected_response
    assert adapter.achat.await_args_list[0].kwargs == {"request": "value"}
    assert adapter.achat.await_count == 2
    sleep.assert_awaited_once_with(2)
