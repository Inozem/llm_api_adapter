import inspect
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from src.llm_api_adapter.errors.llm_api_error import LLMAPIClientError
from src.llm_api_adapter.llms.openai.async_client import OpenAIAsyncClient
from src.llm_api_adapter.llms.openai.sync_client import OpenAISyncClient
from src.llm_api_adapter.llms.openai_compatible.async_client import OpenAICompatibleAsyncClient
from src.llm_api_adapter.llms.openai_compatible.sync_client import OpenAICompatibleSyncClient


@pytest.mark.unit
@pytest.mark.parametrize("api_variant", (None, "chat_completions", "responses"))
@pytest.mark.parametrize("operation", ("complete", "stream"))
def test_generic_sync_dispatch_uses_connection_selection_without_native_policy(
    monkeypatch, api_variant, operation
):
    _guard_native_policy(monkeypatch)
    client = OpenAICompatibleSyncClient(
        api_key="synthetic-key", endpoint="https://host.invalid/proxy/v1"
    )
    client._set_api_variant(api_variant)
    observed = []

    def send(url, payload, timeout):
        observed.append((url, payload, timeout))
        return Mock(json=lambda: {"ok": True})

    def stream(url, payload, timeout):
        observed.append((url, payload, timeout))
        return iter(("event",))

    client._send_request = send
    client._stream_request = stream
    kwargs = {"messages": [], "max_tokens": 8, "temperature": 1.0}
    if operation == "complete":
        assert client.complete("unregistered-model", **kwargs) == {"ok": True}
    else:
        assert list(client.stream("unregistered-model", **kwargs)) == ["event"]

    selected = "responses" if api_variant == "responses" else "chat/completions"
    if api_variant == "responses":
        payload = {
            "model": "unregistered-model",
            "input": [],
            "max_output_tokens": 8,
            "store": False,
        }
    else:
        payload = {
            "model": "unregistered-model",
            "messages": [],
            "max_completion_tokens": 8,
        }
    if operation == "stream":
        payload["stream"] = True
        if api_variant != "responses":
            payload["stream_options"] = {"include_usage": True}
    assert observed == [
        (
            f"https://host.invalid/proxy/v1/{selected}",
            payload,
            None,
        )
    ]


@pytest.mark.unit
@pytest.mark.parametrize("api_variant", (None, "chat_completions", "responses"))
@pytest.mark.parametrize("operation", ("complete", "stream"))
@pytest.mark.asyncio
async def test_generic_async_dispatch_uses_connection_selection_without_native_policy(
    monkeypatch, api_variant, operation
):
    _guard_native_policy(monkeypatch)
    client = OpenAICompatibleAsyncClient(
        api_key="synthetic-key", endpoint="https://host.invalid/proxy/v1"
    )
    client._set_api_variant(api_variant)
    observed = []

    async def send(url, payload, timeout):
        observed.append((url, payload, timeout))
        return {"ok": True}

    async def stream(url, payload, timeout):
        observed.append((url, payload, timeout))
        yield "event"

    client._send_request = send
    client._stream_request = stream
    kwargs = {"messages": [], "max_tokens": 8, "temperature": 1.0}
    if operation == "complete":
        assert await client.complete("unregistered-model", **kwargs) == {"ok": True}
    else:
        assert [event async for event in client.stream("unregistered-model", **kwargs)] == ["event"]

    selected = "responses" if api_variant == "responses" else "chat/completions"
    if api_variant == "responses":
        payload = {
            "model": "unregistered-model",
            "input": [],
            "max_output_tokens": 8,
            "store": False,
        }
    else:
        payload = {
            "model": "unregistered-model",
            "messages": [],
            "max_completion_tokens": 8,
        }
    if operation == "stream":
        payload["stream"] = True
        if api_variant != "responses":
            payload["stream_options"] = {"include_usage": True}
    assert observed == [
        (
            f"https://host.invalid/proxy/v1/{selected}",
            payload,
            None,
        )
    ]


def _guard_native_policy(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("generic clients must not call native model policy")

    for name in ("model_api_variant", "model_spec_for", "apply_model_request_rules"):
        monkeypatch.setattr(
            f"src.llm_api_adapter.llms.openai.sync_client.{name}", forbidden
        )


@pytest.mark.unit
def test_native_sync_global_patch_points_still_control_shared_execution(monkeypatch):
    transport = object()
    factory = Mock(return_value=transport)
    monkeypatch.setattr("src.llm_api_adapter.llms.openai.sync_client.create_sync_transport", factory)
    variant = Mock(return_value="chat_completions")
    rules = Mock(return_value=({"model": "gpt-4o", "messages": []}, []))
    monkeypatch.setattr("src.llm_api_adapter.llms.openai.sync_client.model_api_variant", variant)
    monkeypatch.setattr("src.llm_api_adapter.llms.openai.sync_client.apply_model_request_rules", rules)
    client = OpenAISyncClient(api_key="synthetic-key")
    async_client = OpenAIAsyncClient(api_key="synthetic-key")
    assert client._sync_transport is transport
    assert async_client._sync_transport is transport
    assert factory.call_count == 2
    client._send_request = Mock(return_value=Mock(json=lambda: {"ok": True}))
    assert client.complete("gpt-4o", messages=[]) == {"ok": True}
    variant.assert_called_once_with("openai", "gpt-4o")
    rules.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_native_async_module_patch_points_still_control_shared_execution(monkeypatch):
    variant = Mock(return_value="chat_completions")
    rules = Mock(return_value=({"model": "gpt-4o", "messages": []}, []))
    request = AsyncMock(return_value={"ok": True})
    monkeypatch.setattr("src.llm_api_adapter.llms.openai.sync_client.model_api_variant", variant)
    monkeypatch.setattr("src.llm_api_adapter.llms.openai.sync_client.apply_model_request_rules", rules)
    monkeypatch.setattr("src.llm_api_adapter.llms.openai.async_client.async_request", request)
    client = OpenAIAsyncClient(api_key="synthetic-key")
    assert await client.complete("gpt-4o", messages=[]) == {"ok": True}
    variant.assert_called_once_with("openai", "gpt-4o")
    rules.assert_called_once()
    request.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.parametrize("client_class", (OpenAISyncClient, OpenAIAsyncClient))
def test_native_http_parse_fallback_warning_maps_expected_error(monkeypatch, client_class):
    client = client_class(api_key="synthetic-key")
    response = Mock(status_code=400)
    response.json.side_effect = ValueError("malformed")
    error = Mock(response=response)
    warning = Mock()
    monkeypatch.setattr(
        "src.llm_api_adapter.llms.openai.sync_client.logger",
        SimpleNamespace(warning=warning),
    )

    with pytest.raises(LLMAPIClientError, match="Mock"):
        client._handle_http_error(error)
    warning.assert_called_once()
    assert warning.call_args.args[0] == "Failed to parse error response: %s"
    assert str(warning.call_args.args[1]) == "malformed"


@pytest.mark.unit
def test_native_model_spec_patch_point_remains_live(monkeypatch):
    import src.llm_api_adapter.llms.openai.sync_client as native_sync

    original = native_sync.model_spec_for
    lookup = Mock(wraps=original)
    monkeypatch.setattr(native_sync, "model_spec_for", lookup)
    assert OpenAISyncClient._fallback_reasoning_effort_for_none("gpt-6-astra") == "low"
    lookup.assert_called_once_with("openai", "gpt-6-astra")


@pytest.mark.unit
def test_native_signatures_defaults_positional_calls_and_aliases():
    for client_class in (OpenAISyncClient, OpenAIAsyncClient):
        parameters = inspect.signature(client_class).parameters
        assert list(parameters) == ["api_key", "endpoint", "transport"]
        assert parameters["api_key"].annotation is str
        assert parameters["endpoint"].annotation is str
        assert parameters["transport"].annotation is str
        assert parameters["endpoint"].default == "https://api.openai.com/v1"
        assert parameters["transport"].default == "requests"
    assert OpenAISyncClient("key", "https://custom.invalid/v1", "requests").endpoint == "https://custom.invalid/v1"
    positional_async = OpenAIAsyncClient("key", "https://custom.invalid/v1", "requests")
    assert positional_async.endpoint == "https://custom.invalid/v1"
    assert positional_async.transport == "requests"
    native_fields = [item.name for item in fields(OpenAISyncClient)]
    assert native_fields.count("_api_variant_override") == 1
    assert "_api_variant" not in native_fields
    assert issubclass(OpenAIAsyncClient, OpenAISyncClient)

    import src.llm_api_adapter.llms.openai.sync_client as native_sync
    import src.llm_api_adapter.llms.openai_compatible.sync_client as shared_sync

    assert native_sync._prepare_compatible_chat_payload is shared_sync._prepare_compatible_chat_payload
    assert native_sync._prepare_compatible_responses_payload is shared_sync._prepare_compatible_responses_payload
    assert native_sync._omit_numeric_sampling_defaults is shared_sync._omit_numeric_sampling_defaults
