"""Explicit checks for the installed Core and Mistral wheel artifacts.

This module intentionally does not use pytest's conventional test filename so
it is not collected during ordinary editable-source test runs. Copy it outside
the checkout to exercise freshly installed wheels.
"""

from __future__ import annotations

import asyncio
from importlib import metadata, resources
import json
from pathlib import Path

import pytest

import llm_api_adapter
import llm_api_adapter_mistral
import llm_api_adapter_mistral.adapter as mistral_adapter_module
import llm_api_adapter_mistral.clients.async_client as mistral_async_client_module
from llm_api_adapter_mistral.plugin import PLUGIN
from llm_api_adapter.adapters import base_adapter as base_adapter_module
from llm_api_adapter.llm_registry.llm_registry import RegistrySpec, resolve_model_spec
import llm_api_adapter.organization_registry as organization_registry_module
from llm_api_adapter.llms.transports import JSONResponse, SSEEvent
from llm_api_adapter.organization_registry import OrganizationPluginDiscovery
from llm_api_adapter.service_provider_registry import ServiceProviderRegistry
import llm_api_adapter.universal_adapter as universal_adapter_module
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter


MISTRAL_ENDPOINT = "https://api.mistral.ai/v1/chat/completions"
API_KEY = "mistral-test-key"
HOSTED_MODEL = "glm-5.3-flash"
HOSTED_WIRE_MODEL = "zai-glm-5-3-flash"
NATIVE_MODELS = (
    "mistral-large-4",
    "mistral-small-2603",
    "mistral-medium-3-5",
    "mistral-large-2512",
)
ALL_MODELS = (*NATIVE_MODELS, HOSTED_MODEL)


def _distribution_module_origin(distribution_name: str, module) -> Path:
    distribution = metadata.distribution(distribution_name)
    origin = Path(module.__file__).resolve()
    installed_files = distribution.files
    assert installed_files is not None
    resolved_files = {
        Path(distribution.locate_file(item)).resolve() for item in installed_files
    }
    assert origin in resolved_files, f"{origin} is not owned by {distribution_name}"
    return origin


def _assert_installed_mistral_artifacts(*, require_zai_absent: bool) -> None:
    core_origin = _distribution_module_origin("llm-api-adapter", llm_api_adapter)
    mistral_origin = _distribution_module_origin(
        "llm-api-adapter-mistral", llm_api_adapter_mistral
    )
    assert metadata.version("llm-api-adapter") == "0.9.10"
    assert metadata.version("llm-api-adapter-mistral") == "0.1.5"
    assert "site-packages" in str(core_origin).lower()
    assert "site-packages" in str(mistral_origin).lower()

    catalogue = resources.files("llm_api_adapter_mistral").joinpath(
        "registry", "organizations", "mistral.json"
    )
    payload = json.loads(catalogue.read_text(encoding="utf-8"))
    models = payload.get("models", {})
    assert set(NATIVE_MODELS).issubset(models)
    hosted = models[HOSTED_MODEL]
    assert hosted["provider_model"] == HOSTED_WIRE_MODEL
    assert PLUGIN.model_metadata is not None
    plugin_models = PLUGIN.model_metadata.organization_data["models"]
    assert set(NATIVE_MODELS).issubset(plugin_models)
    assert plugin_models[HOSTED_MODEL]["provider_model"] == HOSTED_WIRE_MODEL

    if require_zai_absent:
        with pytest.raises(metadata.PackageNotFoundError):
            metadata.distribution("llm-api-adapter-zai")


@pytest.fixture(autouse=True)
def installed_artifacts_are_isolated(request):
    """Require each selected Mistral-only check to prove wheel provenance."""
    _assert_installed_mistral_artifacts(
        require_zai_absent=request.node.originalname
        != "test_joint_installation_keeps_routes_independent"
    )


class RecordingSyncTransport:
    def __init__(self, *, streaming: bool) -> None:
        self.requests = []
        self.stream_requests = []
        self.streaming = streaming

    def post_json(self, request, *, http_error_handler=None):
        self.requests.append(request)
        return JSONResponse(
            {
                "id": "wheel-sync-response",
                "model": request.payload["model"],
                "choices": [
                    {"message": {"role": "assistant", "content": "controlled"}}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
        )

    def post_sse(
        self,
        request,
        *,
        http_error_handler=None,
        stream_error_handler=None,
    ):
        self.stream_requests.append(request)
        model = request.payload["model"]
        return iter(
            [
                SSEEvent(
                    event=None,
                    data={
                        "id": "wheel-sync-stream",
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": "controlled stream"},
                                "finish_reason": "stop",
                            }
                        ],
                    },
                )
            ]
        )


def _adapter(model: str, *, transport: str = "requests"):
    if model == HOSTED_MODEL:
        return UniversalLLMAPIAdapter(
            organization="zai",
            service_provider="mistral",
            model=model,
            api_key=API_KEY,
            transport=transport,
        )
    return UniversalLLMAPIAdapter(
        organization="mistral",
        model=model,
        api_key=API_KEY,
        transport=transport,
    )


def _expected_wire_model(model: str) -> str:
    return HOSTED_WIRE_MODEL if model == HOSTED_MODEL else model


def _assert_hosted_flash_profile(client) -> None:
    assert isinstance(client.adapter, mistral_adapter_module.MistralAdapter)
    assert client.adapter.service_provider == "mistral"
    profile = client.adapter.model_spec
    assert profile is not None
    assert profile.name == HOSTED_MODEL
    assert profile.limits.context_window_tokens == 1_048_576
    assert profile.limits.max_output_tokens == 128_000
    assert profile.provider_model == HOSTED_WIRE_MODEL
    assert profile.pricing_tiers.currency == "USD"
    assert len(profile.pricing_tiers.tiers) == 1
    tier = profile.pricing_tiers.tiers[0]
    assert tier.in_per_token == pytest.approx(0.15 / 1_000_000)
    assert tier.cache_read_in_per_token == pytest.approx(0.03 / 1_000_000)
    assert tier.out_per_token == pytest.approx(0.50 / 1_000_000)


def _assert_request(request, model: str) -> None:
    assert request.url == MISTRAL_ENDPOINT
    assert request.headers_dict()["Authorization"] == f"Bearer {API_KEY}"
    assert request.payload["model"] == _expected_wire_model(model)


def test_mistral_only_installation_excludes_zai():
    with pytest.raises(metadata.PackageNotFoundError):
        metadata.distribution("llm-api-adapter-zai")


@pytest.mark.parametrize("model", ALL_MODELS)
@pytest.mark.parametrize("transport", ("requests", "httpx"))
@pytest.mark.parametrize("operation", ("chat", "stream_chat"))
def test_installed_sync_routes_for_all_models_and_transports(
    model: str, transport: str, operation: str
):
    client = _adapter(model, transport=transport)
    if model == HOSTED_MODEL:
        _assert_hosted_flash_profile(client)
    fake_transport = RecordingSyncTransport(streaming=operation == "stream_chat")
    client.adapter._sync_transport = fake_transport
    messages = [{"role": "user", "content": "controlled input"}]

    if operation == "chat":
        response = client.chat(messages)
        assert response.content == "controlled"
        request = fake_transport.requests[0]
    else:
        assert list(client.stream_chat(messages)) == ["controlled stream"]
        request = fake_transport.stream_requests[0]

    _assert_request(request, model)
    assert client.adapter.service_provider == "mistral"
    assert client.adapter.model == model


@pytest.mark.parametrize("model", ALL_MODELS)
@pytest.mark.parametrize("operation", ("achat", "astream_chat"))
def test_installed_async_routes_for_all_models(
    model: str, operation: str, monkeypatch
):
    client = _adapter(model)
    if model == HOSTED_MODEL:
        _assert_hosted_flash_profile(client)
    request_calls = []
    stream_calls = []

    async def controlled_request(url, **kwargs):
        request_calls.append((url, kwargs))
        return {
            "id": "wheel-async-response",
            "model": kwargs["payload"]["model"],
            "choices": [
                {"message": {"role": "assistant", "content": "controlled async"}}
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    def controlled_stream(url, **kwargs):
        stream_calls.append((url, kwargs))

        async def events():
            yield SSEEvent(
                event=None,
                data={
                    "id": "wheel-async-stream",
                    "model": kwargs["payload"]["model"],
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "controlled async stream"},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )

        return events()

    monkeypatch.setattr(
        mistral_async_client_module, "async_request", controlled_request
    )
    monkeypatch.setattr(
        mistral_async_client_module, "async_stream_request", controlled_stream
    )

    async def run_operation():
        messages = [{"role": "user", "content": "controlled input"}]
        if operation == "achat":
            result = await client.achat(messages)
            assert result.content == "controlled async"
        else:
            result = [text async for text in client.astream_chat(messages)]
            assert result == ["controlled async stream"]

    asyncio.run(run_operation())
    calls = request_calls if operation == "achat" else stream_calls
    assert len(calls) == 1
    url, kwargs = calls[0]
    assert url == MISTRAL_ENDPOINT
    assert kwargs["headers"]["Authorization"] == f"Bearer {API_KEY}"
    assert kwargs["payload"]["model"] == _expected_wire_model(model)
    assert client.adapter.service_provider == "mistral"
    assert client.adapter.model == model


@pytest.mark.parametrize("order", (("mistral", "zai"), ("zai", "mistral")))
def test_joint_installation_keeps_routes_independent(monkeypatch, order):
    zai_distribution = metadata.distribution("llm-api-adapter-zai")
    zai_package = __import__("llm_api_adapter_zai")
    zai_origin = Path(zai_package.__file__).resolve()
    zai_files = zai_distribution.files
    assert zai_files is not None
    assert zai_origin in {
        Path(zai_distribution.locate_file(item)).resolve() for item in zai_files
    }
    from llm_api_adapter_zai.adapter import ZaiAdapter
    from llm_api_adapter_zai.plugin import PLUGIN as ZAI_PLUGIN

    class PluginEntryPoint:
        def __init__(self, name, plugin):
            self.name = name
            self.value = f"llm_api_adapter_{name}.plugin:PLUGIN"
            self.plugin = plugin

        def load(self):
            return self.plugin

    plugins = {"mistral": PLUGIN, "zai": ZAI_PLUGIN}
    entries = {name: PluginEntryPoint(name, plugin) for name, plugin in plugins.items()}
    active_order = list(order)

    def fake_entry_points(*, group):
        assert group == organization_registry_module.ORGANIZATION_PLUGIN_ENTRY_POINT_GROUP
        return tuple(entries[name] for name in active_order)

    registry = RegistrySpec()
    providers = ServiceProviderRegistry(
        {"openai": universal_adapter_module.OpenAIAdapter}
    )
    discovery = OrganizationPluginDiscovery()
    monkeypatch.setattr(organization_registry_module, "entry_points", fake_entry_points)
    monkeypatch.setattr(universal_adapter_module, "SERVICE_PROVIDER_REGISTRY", providers)
    monkeypatch.setattr(universal_adapter_module, "ORGANIZATION_PLUGIN_DISCOVERY", discovery)
    monkeypatch.setattr(universal_adapter_module, "LLM_REGISTRY", registry)
    monkeypatch.setattr(base_adapter_module, "LLM_REGISTRY", registry)

    discovery.discover(providers, model_registry=registry)
    active_order.reverse()
    discovery.discover(providers, model_registry=registry)

    mistral_spec = resolve_model_spec(registry, "mistral", HOSTED_MODEL)
    zai_spec = resolve_model_spec(registry, "zai", HOSTED_MODEL)
    assert mistral_spec is registry.organizations["mistral"].models[HOSTED_MODEL]
    assert zai_spec is registry.organizations["zai"].models[HOSTED_MODEL]
    assert mistral_spec is not zai_spec
    assert mistral_spec.provider_model == HOSTED_WIRE_MODEL
    assert zai_spec.provider_model is None
    assert mistral_spec.limits.context_window_tokens == 1_048_576
    assert mistral_spec.limits.max_output_tokens == 128_000
    assert mistral_spec.limits != zai_spec.limits
    assert mistral_spec.pricing_tiers != zai_spec.pricing_tiers

    mistral_hosted = UniversalLLMAPIAdapter(
        organization="zai",
        model=HOSTED_MODEL,
        service_provider="mistral",
        api_key=API_KEY,
    )
    zai_direct = UniversalLLMAPIAdapter(
        organization="zai",
        model=HOSTED_MODEL,
        service_provider="zai",
        api_key="zai-test-key",
    )
    assert isinstance(mistral_hosted.adapter, mistral_adapter_module.MistralAdapter)
    assert isinstance(zai_direct.adapter, ZaiAdapter)
    assert mistral_hosted.adapter.model_spec is mistral_spec
    assert mistral_hosted.adapter.model_spec is not zai_direct.adapter.model_spec
    assert zai_direct.adapter.model_spec is zai_spec
    assert mistral_hosted.adapter.service_provider == "mistral"
    assert zai_direct.adapter.service_provider == "zai"
    assert mistral_hosted.adapter.endpoint == MISTRAL_ENDPOINT
    assert zai_direct.adapter.endpoint == "https://api.z.ai/api/paas/v4/chat/completions"
    _assert_hosted_flash_profile(mistral_hosted)

    mistral_transport = RecordingSyncTransport(streaming=False)
    zai_transport = RecordingSyncTransport(streaming=False)
    mistral_hosted.adapter._sync_transport = mistral_transport
    zai_direct.adapter._sync_transport = zai_transport
    messages = [{"role": "user", "content": "controlled input"}]
    assert mistral_hosted.chat(messages).content == "controlled"
    assert zai_direct.chat(messages).content == "controlled"
    mistral_request = mistral_transport.requests[0]
    zai_request = zai_transport.requests[0]
    _assert_request(mistral_request, HOSTED_MODEL)
    assert mistral_request.headers_dict()["Authorization"] == f"Bearer {API_KEY}"
    assert zai_request.url == zai_direct.adapter.endpoint
    assert zai_request.headers_dict()["Authorization"] == "Bearer zai-test-key"
    assert zai_request.payload["model"] == HOSTED_MODEL
