from dataclasses import fields
from inspect import signature
from unittest.mock import Mock, patch

import pytest

from src.llm_api_adapter.llms.openai.async_client import OpenAIAsyncClient
from src.llm_api_adapter.llms.openai.sync_client import OpenAISyncClient


CLIENTS = (OpenAISyncClient, OpenAIAsyncClient)


@pytest.mark.unit
@pytest.mark.parametrize("client_class", CLIENTS, ids=("sync", "async"))
def test_policy_is_private_and_does_not_change_constructor_repr_or_equality(
    client_class,
):
    parameters = tuple(signature(client_class).parameters)
    assert parameters == ("api_key", "endpoint", "transport")

    left = client_class(api_key="test_api_key")
    right = client_class(api_key="test_api_key")
    assert left == right
    assert "official_behavior" not in repr(left)
    assert not next(
        item for item in fields(client_class) if item.name == "_official_behavior_enabled"
    ).init
    assert not next(
        item for item in fields(client_class) if item.name == "_official_behavior_enabled"
    ).repr
    assert not next(
        item for item in fields(client_class) if item.name == "_official_behavior_enabled"
    ).compare

    left._set_official_behavior_enabled(False)
    assert left == right


@pytest.mark.unit
@pytest.mark.parametrize("client_class", CLIENTS, ids=("sync", "async"))
def test_disabled_policy_skips_all_official_registry_behavior(client_class):
    client = client_class(api_key="test_api_key")
    client._set_official_behavior_enabled(False)
    kwargs = {"messages": [], "max_tokens": 7, "temperature": 0.2}

    with (
        patch("src.llm_api_adapter.llms.openai.sync_client.model_api_variant") as variant,
        patch("src.llm_api_adapter.llms.openai.sync_client.model_spec_for") as spec,
        patch(
            "src.llm_api_adapter.llms.openai.sync_client.apply_model_request_rules"
        ) as rules,
    ):
        assert client._should_use_responses_api("gpt-5") is False
        assert client._should_use_responses_api("arbitrary-model") is False
        payload = client._prepare_chat_payload_for_model("gpt-5", kwargs)

    assert payload == {
        "model": "gpt-5",
        "messages": [],
        "max_completion_tokens": 7,
        "temperature": 0.2,
    }
    assert payload is not kwargs
    assert kwargs == {"messages": [], "max_tokens": 7, "temperature": 0.2}
    assert client._prepare_chat_payload_for_model(
        "arbitrary-model", kwargs
    )["model"] == "arbitrary-model"
    variant.assert_not_called()
    spec.assert_not_called()
    rules.assert_not_called()


@pytest.mark.unit
def test_policy_is_instance_local():
    disabled = OpenAISyncClient(api_key="test_api_key")
    unchanged = OpenAISyncClient(api_key="test_api_key")
    disabled._set_official_behavior_enabled(False)

    assert disabled._should_use_responses_api("gpt-5") is False
    assert unchanged._should_use_responses_api("gpt-5") is True


@pytest.mark.unit
@pytest.mark.parametrize("client_class", CLIENTS, ids=("sync", "async"))
def test_api_variant_override_is_private_and_does_not_change_client_identity(client_class):
    chat_client = client_class(api_key="test_api_key")
    responses_client = client_class(api_key="test_api_key")
    native_client = client_class(api_key="test_api_key")
    route_field = next(
        item for item in fields(client_class) if item.name == "_api_variant_override"
    )

    assert not route_field.init
    assert not route_field.repr
    assert not route_field.compare
    assert chat_client == responses_client == native_client
    assert "responses" not in repr(chat_client)

    chat_client._set_official_behavior_enabled(False)
    chat_client._set_api_variant("chat_completions")
    assert chat_client._should_use_responses_api("vendor/arbitrary-model") is False

    responses_client._set_official_behavior_enabled(False)
    responses_client._set_api_variant("responses")
    assert responses_client._should_use_responses_api("vendor/arbitrary-model") is True
    assert chat_client._should_use_responses_api("vendor/arbitrary-model") is False
    assert chat_client == responses_client == native_client
    assert "responses" not in repr(responses_client)

    assert native_client._should_use_responses_api("gpt-5") is True
    assert native_client._should_use_responses_api("gpt-4.1-mini") is False
    assert responses_client._should_use_responses_api("vendor/arbitrary-model") is True
    assert chat_client._should_use_responses_api("vendor/arbitrary-model") is False


@pytest.mark.unit
def test_native_policy_still_selects_registered_api_independent_of_override():
    client = OpenAISyncClient(api_key="test_api_key")
    client._set_api_variant("chat_completions")

    assert client._should_use_responses_api("gpt-5") is True
    assert client._should_use_responses_api("gpt-4.1-mini") is False


@pytest.mark.unit
@pytest.mark.parametrize("model", ("gpt-5", "arbitrary-model"))
def test_disabled_sync_complete_and_stream_dispatch_to_chat(model):
    client = OpenAISyncClient(api_key="test_api_key", endpoint="https://local.test/v1")
    client._set_official_behavior_enabled(False)
    complete_dispatch = Mock(return_value=Mock(json=lambda: {"ok": True}))
    stream_dispatch = Mock(return_value=iter(()))
    client._send_request = complete_dispatch
    client._stream_request = stream_dispatch

    assert client.complete(model, messages=[], max_tokens=8) == {"ok": True}
    assert list(client.stream(model, messages=[], max_tokens=8)) == []

    expected = {
        "model": model,
        "messages": [],
        "max_completion_tokens": 8,
    }
    complete_dispatch.assert_called_once_with(
        "https://local.test/v1/chat/completions", expected, None
    )
    stream_dispatch.assert_called_once_with(
        "https://local.test/v1/chat/completions",
        {**expected, "stream_options": {"include_usage": True}, "stream": True},
        None,
    )


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("model", ("gpt-5", "arbitrary-model"))
async def test_disabled_async_complete_and_stream_dispatch_to_chat(model):
    client = OpenAIAsyncClient(api_key="test_api_key", endpoint="https://local.test/v1")
    client._set_official_behavior_enabled(False)
    complete_dispatch = Mock(return_value={"ok": True})
    stream_payloads = []

    async def send(url, payload, timeout):
        complete_dispatch(url, payload, timeout)
        return {"ok": True}

    async def stream(url, payload, timeout):
        stream_payloads.append((url, payload, timeout))
        if False:
            yield None

    client._send_request = send
    client._stream_request = stream

    assert await client.complete(model, messages=[], max_tokens=8) == {"ok": True}
    assert [event async for event in client.stream(model, messages=[], max_tokens=8)] == []

    expected = {"model": model, "messages": [], "max_completion_tokens": 8}
    complete_dispatch.assert_called_once_with(
        "https://local.test/v1/chat/completions", expected, None
    )
    assert stream_payloads == [
        (
            "https://local.test/v1/chat/completions",
            {**expected, "stream_options": {"include_usage": True}, "stream": True},
            None,
        )
    ]


@pytest.mark.unit
def test_disabled_policy_keeps_explicit_sync_responses_methods_available():
    client = OpenAISyncClient(api_key="test_api_key")
    client._set_official_behavior_enabled(False)
    send = Mock(return_value=Mock(json=lambda: {"ok": True}))
    stream = Mock(return_value=iter(()))
    client._send_request = send
    client._stream_request = stream

    assert client.responses("gpt-5", messages=[], max_tokens=8) == {"ok": True}
    assert list(client.stream_responses("gpt-5", messages=[], max_tokens=8)) == []

    send.assert_called_once_with(
        "https://api.openai.com/v1/responses",
        {"model": "gpt-5", "input": [], "max_output_tokens": 8, "store": False},
        None,
    )
    stream.assert_called_once_with(
        "https://api.openai.com/v1/responses",
        {
            "model": "gpt-5",
            "input": [],
            "max_output_tokens": 8,
            "store": False,
            "stream": True,
        },
        None,
    )


@pytest.mark.asyncio
@pytest.mark.unit
async def test_disabled_policy_keeps_explicit_async_responses_method_available():
    client = OpenAIAsyncClient(api_key="test_api_key")
    client._set_official_behavior_enabled(False)
    calls = []

    async def send(url, payload, timeout):
        calls.append((url, payload, timeout))
        return {"ok": True}

    async def stream(url, payload, timeout):
        calls.append((url, payload, timeout))
        if False:
            yield None

    client._send_request = send
    client._stream_request = stream

    assert await client.responses("gpt-5", messages=[], max_tokens=8) == {"ok": True}
    assert [event async for event in client.stream_responses("gpt-5", messages=[], max_tokens=8)] == []

    assert calls == [
        (
            "https://api.openai.com/v1/responses",
            {"model": "gpt-5", "input": [], "max_output_tokens": 8, "store": False},
            None,
        ),
        (
            "https://api.openai.com/v1/responses",
            {
                "model": "gpt-5",
                "input": [],
                "max_output_tokens": 8,
                "store": False,
                "stream": True,
            },
            None,
        ),
    ]
