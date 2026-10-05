"""Verify timeout normalization through the facade without external requests."""

from unittest.mock import AsyncMock, Mock

import httpx
import pytest
import requests

from llm_api_adapter.errors import LLMAPITimeoutError
from llm_api_adapter.llm_registry.llm_registry import LLM_REGISTRY
from llm_api_adapter.models.messages.chat_message import UserMessage
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("organization", ["openai", "anthropic", "google"])
@pytest.mark.parametrize("operation", ["chat", "chat_httpx", "achat"])
async def test_facade_normalizes_transport_timeout(monkeypatch, organization, operation):
    if operation == "chat":
        raw_timeout = requests.exceptions.ReadTimeout("read timed out")
        post = Mock(side_effect=raw_timeout)
        monkeypatch.setattr(requests, "post", post)
    else:
        raw_timeout = httpx.ReadTimeout("read timed out")
        post = (AsyncMock if operation == "achat" else Mock)(side_effect=raw_timeout)
        monkeypatch.setattr(
            httpx.AsyncClient if operation == "achat" else httpx.Client, "post", post
        )

    model = next(iter(LLM_REGISTRY.organizations[organization].models))
    adapter = UniversalLLMAPIAdapter(
        organization=organization,
        model=model,
        api_key="test-key",
        transport="httpx" if operation == "chat_httpx" else "requests",
    )
    kwargs = dict(messages=[UserMessage("Say OK")], max_tokens=512, timeout_s=0.2)

    with pytest.raises(LLMAPITimeoutError) as raised:
        if operation == "achat":
            await adapter.achat(**kwargs)
        else:
            adapter.chat(**kwargs)

    assert raised.value.__cause__ is raw_timeout
    assert post.call_count == 1
    assert post.call_args.kwargs["timeout"] == 0.2
