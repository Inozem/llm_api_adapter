"""Characterize model-specific request rules before moving them to the registry."""

from __future__ import annotations

import warnings

import pytest

from src.llm_api_adapter.llms.anthropic.async_client import ClaudeAsyncClient
from src.llm_api_adapter.llms.anthropic.sync_client import ClaudeSyncClient
from src.llm_api_adapter.llms.google.async_client import GeminiAsyncClient
from src.llm_api_adapter.llms.google.sync_client import GeminiSyncClient
from src.llm_api_adapter.llms.openai.async_client import OpenAIAsyncClient
from src.llm_api_adapter.llms.openai.sync_client import OpenAISyncClient


OPENAI_CLIENTS = (OpenAISyncClient, OpenAIAsyncClient)
ANTHROPIC_CLIENTS = (ClaudeSyncClient, ClaudeAsyncClient)
GOOGLE_CLIENTS = (GeminiSyncClient, GeminiAsyncClient)


def _captured_warnings():
    return warnings.catch_warnings(record=True)


@pytest.mark.unit
@pytest.mark.parametrize("client_class", OPENAI_CLIENTS, ids=("sync", "async"))
def test_openai_selects_the_api_variant_for_each_model_family(client_class):
    client = client_class(api_key="test_api_key")

    assert client._should_use_responses_api("gpt-5") is True
    assert client._should_use_responses_api("gpt-4o") is False


@pytest.mark.unit
@pytest.mark.parametrize("client_class", OPENAI_CLIENTS, ids=("sync", "async"))
def test_openai_snapshot_uses_its_registered_base_api_variant(client_class):
    client = client_class(api_key="test_api_key")

    assert client._should_use_responses_api("gpt-5-2025-08-07") is True


@pytest.mark.unit
@pytest.mark.parametrize("client_class", OPENAI_CLIENTS, ids=("sync", "async"))
def test_openai_does_not_assume_an_api_variant_for_unregistered_models(client_class):
    client = client_class(api_key="test_api_key")

    assert client._should_use_responses_api("gpt-5-unregistered") is False


@pytest.mark.unit
@pytest.mark.parametrize("client_class", OPENAI_CLIENTS, ids=("sync", "async"))
def test_openai_renames_max_tokens_for_chat_completions_models(client_class):
    client = client_class(api_key="test_api_key")

    payload = client._prepare_chat_payload_for_model(
        "gpt-4.1-mini",
        {"messages": [], "max_tokens": 256},
    )

    assert payload["max_completion_tokens"] == 256
    assert "max_tokens" not in payload


@pytest.mark.unit
@pytest.mark.parametrize("client_class", OPENAI_CLIENTS, ids=("sync", "async"))
def test_openai_snapshot_uses_its_registered_base_payload_rules(client_class):
    client = client_class(api_key="test_api_key")

    payload = client._prepare_chat_payload_for_model(
        "gpt-4.1-2025-04-14",
        {"messages": [], "max_tokens": 256},
    )

    assert payload["max_completion_tokens"] == 256
    assert "max_tokens" not in payload


@pytest.mark.unit
@pytest.mark.parametrize("client_class", OPENAI_CLIENTS, ids=("sync", "async"))
def test_openai_converts_the_legacy_gpt5_reasoning_default(client_class):
    client = client_class(api_key="test_api_key")

    with _captured_warnings() as caught:
        warnings.simplefilter("always")
        payload = client._prepare_responses_payload_for_model(
            "gpt-5",
            {"reasoning_effort": "none"},
        )

    assert payload["reasoning"] == {"effort": "minimal"}
    assert caught == []


@pytest.mark.unit
@pytest.mark.parametrize("client_class", OPENAI_CLIENTS, ids=("sync", "async"))
def test_openai_preserves_native_none_when_registry_allows_it(client_class):
    client = client_class(api_key="test_api_key")

    payload = client._prepare_responses_payload_for_model(
        "gpt-5.6-sol",
        {"reasoning_effort": "none"},
    )

    assert payload["reasoning"] == {"effort": "none"}


@pytest.mark.unit
@pytest.mark.parametrize("client_class", OPENAI_CLIENTS, ids=("sync", "async"))
def test_openai_silently_omits_default_top_p_for_gpt5(client_class):
    client = client_class(api_key="test_api_key")

    with _captured_warnings() as caught:
        warnings.simplefilter("always")
        payload = client._prepare_responses_payload_for_model(
            "gpt-5",
            {"top_p": 1.0},
        )

    assert "top_p" not in payload
    assert caught == []


@pytest.mark.unit
@pytest.mark.parametrize("client_class", OPENAI_CLIENTS, ids=("sync", "async"))
def test_openai_warns_once_when_omitting_non_default_top_p_for_gpt5(client_class):
    client = client_class(api_key="test_api_key")

    with _captured_warnings() as caught:
        warnings.simplefilter("always")
        payload = client._prepare_responses_payload_for_model(
            "gpt-5",
            {"top_p": 0.2},
        )

    assert "top_p" not in payload
    assert len(caught) == 1
    assert caught[0].category is UserWarning
    assert "top_p" in str(caught[0].message)
    assert "gpt-5" in str(caught[0].message)


@pytest.mark.unit
@pytest.mark.parametrize("client_class", OPENAI_CLIENTS, ids=("sync", "async"))
def test_openai_silently_omits_default_temperature_for_gpt5_nano(client_class):
    client = client_class(api_key="test_api_key")

    with _captured_warnings() as caught:
        warnings.simplefilter("always")
        payload = client._prepare_responses_payload_for_model(
            "gpt-5-nano",
            {"temperature": 1.0},
        )

    assert "temperature" not in payload
    assert caught == []


@pytest.mark.unit
@pytest.mark.parametrize("client_class", OPENAI_CLIENTS, ids=("sync", "async"))
def test_openai_warns_once_when_omitting_non_default_temperature_for_gpt5_nano(
    client_class,
):
    client = client_class(api_key="test_api_key")

    with _captured_warnings() as caught:
        warnings.simplefilter("always")
        payload = client._prepare_responses_payload_for_model(
            "gpt-5-nano",
            {"temperature": 0.2},
        )

    assert "temperature" not in payload
    assert len(caught) == 1
    assert caught[0].category is UserWarning
    assert "temperature" in str(caught[0].message)
    assert "gpt-5-nano" in str(caught[0].message)


@pytest.mark.unit
@pytest.mark.parametrize("client_class", ANTHROPIC_CLIENTS, ids=("sync", "async"))
def test_anthropic_silently_omits_default_top_p_for_claude_4_5(client_class):
    client = client_class(api_key="test_api_key")

    with _captured_warnings() as caught:
        warnings.simplefilter("always")
        payload = client._prepare_chat_payload_for_model(
            "claude-sonnet-4-5",
            {"messages": [], "top_p": 1.0},
        )

    assert "top_p" not in payload
    assert caught == []


@pytest.mark.unit
@pytest.mark.parametrize("client_class", ANTHROPIC_CLIENTS, ids=("sync", "async"))
def test_anthropic_snapshot_uses_its_registered_base_payload_rules(client_class):
    client = client_class(api_key="test_api_key")

    with _captured_warnings() as caught:
        warnings.simplefilter("always")
        payload = client._prepare_chat_payload_for_model(
            "claude-sonnet-4-5-20250929",
            {"messages": [], "top_p": 1.0},
        )

    assert "top_p" not in payload
    assert caught == []


@pytest.mark.unit
@pytest.mark.parametrize("client_class", ANTHROPIC_CLIENTS, ids=("sync", "async"))
def test_anthropic_warns_once_when_omitting_non_default_top_p_for_claude_4_5(
    client_class,
):
    client = client_class(api_key="test_api_key")

    with _captured_warnings() as caught:
        warnings.simplefilter("always")
        payload = client._prepare_chat_payload_for_model(
            "claude-sonnet-4-5",
            {"messages": [], "top_p": 0.2},
        )

    assert "top_p" not in payload
    assert len(caught) == 1
    assert caught[0].category is UserWarning
    assert "top_p" in str(caught[0].message)
    assert "claude-sonnet-4-5" in str(caught[0].message)


@pytest.mark.unit
@pytest.mark.parametrize("client_class", GOOGLE_CLIENTS, ids=("sync", "async"))
def test_google_silently_omits_default_max_output_tokens_for_gemini_2_5(
    client_class,
):
    client = client_class(api_key="test_api_key")

    with _captured_warnings() as caught:
        warnings.simplefilter("always")
        payload = client._prepare_chat_payload_for_model(
            "gemini-2.5-flash",
            {"generationConfig": {"maxOutputTokens": None}},
        )

    assert "maxOutputTokens" not in payload["generationConfig"]
    assert caught == []


@pytest.mark.unit
@pytest.mark.parametrize("client_class", GOOGLE_CLIENTS, ids=("sync", "async"))
def test_google_warns_once_when_omitting_non_default_max_output_tokens_for_gemini_2_5(
    client_class,
):
    client = client_class(api_key="test_api_key")

    with _captured_warnings() as caught:
        warnings.simplefilter("always")
        payload = client._prepare_chat_payload_for_model(
            "gemini-2.5-flash",
            {"generationConfig": {"maxOutputTokens": 256}},
        )

    assert "maxOutputTokens" not in payload["generationConfig"]
    assert len(caught) == 1
    assert caught[0].category is UserWarning
    assert "maxOutputTokens" in str(caught[0].message)
    assert "gemini-2.5-flash" in str(caught[0].message)
