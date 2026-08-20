from dataclasses import dataclass

import pytest

import src.llm_api_adapter.universal_adapter as universal_module
from src.llm_api_adapter.adapters.anthropic_adapter import AnthropicAdapter
from src.llm_api_adapter.adapters.google_adapter import GoogleAdapter
from src.llm_api_adapter.adapters.openai_adapter import OpenAIAdapter
from src.llm_api_adapter.provider_registry import ServiceProviderRegistry
from src.llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter

@pytest.mark.unit
def test_selects_adapter_and_delegates(monkeypatch):
    @dataclass
    class FakeAdapter(universal_module.LLMAdapterBase):
        company: str = "anthropic"
        model: str = ""
        api_key: str = ""

        def chat(self, *args, **kwargs):
            return {"response": "ok"}

        def stream_chat(self, *args, **kwargs):
            yield "streamed"

        def greet(self, name: str) -> str:
            return f"hello {name} from {self.company}"

    monkeypatch.setattr(
        universal_module,
        "SERVICE_PROVIDER_REGISTRY",
        ServiceProviderRegistry({"anthropic": FakeAdapter}),
    )
    ua = UniversalLLMAPIAdapter(
        organization="anthropic", model="claude-sonnet-4-5", api_key="sk-test"
    )
    assert isinstance(ua.adapter, FakeAdapter)
    assert ua.adapter.model == "claude-sonnet-4-5"
    assert ua.adapter.api_key == "sk-test"
    assert ua.greet("Alice") == "hello Alice from anthropic"
    assert list(ua.stream_chat()) == ["streamed"]

@pytest.mark.unit
def test_unknown_organization_raises(monkeypatch):
    monkeypatch.setattr(
        universal_module,
        "SERVICE_PROVIDER_REGISTRY",
        ServiceProviderRegistry(),
    )
    with pytest.raises(ValueError, match="Unsupported organization: UnknownCorp"):
        UniversalLLMAPIAdapter(
            organization="UnknownCorp", model="test-model", api_key="k"
        )


@pytest.mark.unit
def test_explicit_unknown_service_provider_raises(monkeypatch):
    monkeypatch.setattr(
        universal_module,
        "SERVICE_PROVIDER_REGISTRY",
        ServiceProviderRegistry(),
    )

    with pytest.raises(ValueError, match="Unsupported service provider: openrouter"):
        UniversalLLMAPIAdapter(
            organization="mistral",
            service_provider="openrouter",
            model="test-model",
            api_key="k",
        )

@pytest.mark.unit
def test_invalid_inputs_raise_value_error():
    with pytest.raises(ValueError, match="Invalid organization"):
        UniversalLLMAPIAdapter(organization="", model="m", api_key="k")
    with pytest.raises(ValueError, match="Invalid model"):
        UniversalLLMAPIAdapter(organization="Anthropic", model="", api_key="k")
    with pytest.raises(ValueError, match="Invalid API key"):
        UniversalLLMAPIAdapter(organization="Anthropic", model="m", api_key="")
    with pytest.raises(ValueError, match="Invalid service provider"):
        UniversalLLMAPIAdapter(
            organization="Anthropic",
            service_provider="",
            model="m",
            api_key="k",
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("organization", "model", "adapter_class"),
    [
        ("anthropic", "claude-sonnet-4-5", AnthropicAdapter),
        ("openai", "gpt-5", OpenAIAdapter),
        ("google", "gemini-2.5-flash", GoogleAdapter),
    ],
)
def test_explicit_builtin_registry_selects_each_provider(
    monkeypatch,
    organization,
    model,
    adapter_class,
):
    monkeypatch.setattr(
        universal_module.LLMAdapterBase,
        "__subclasses__",
        classmethod(lambda cls: []),
        raising=False,
    )

    adapter = UniversalLLMAPIAdapter(
        organization=organization,
        model=model,
        api_key="test-key",
    )

    assert isinstance(adapter.adapter, adapter_class)
    assert adapter.transport == adapter.adapter.transport == "requests"
    assert adapter.service_provider == adapter.adapter.service_provider == organization


@pytest.mark.unit
def test_transport_selection_is_validated_and_forwarded_to_provider_adapter():
    adapter = UniversalLLMAPIAdapter(
        organization="openai",
        model="gpt-4o",
        api_key="test-key",
        transport="httpx",
    )

    assert adapter.transport == "httpx"
    assert adapter.adapter.transport == "httpx"

    with pytest.raises(ValueError, match="requests.*httpx"):
        UniversalLLMAPIAdapter(
            organization="openai",
            model="gpt-4o",
            api_key="test-key",
            transport="urllib3",
        )


@pytest.mark.unit
def test_service_provider_selects_the_adapter_and_organization_is_forwarded(
    monkeypatch,
):
    @dataclass
    class HostedMistralAdapter:
        company: str
        model: str
        api_key: str
        transport: str
        service_provider: str

        def chat(self, *args, **kwargs):
            return {"response": "ok"}

        def stream_chat(self, *args, **kwargs):
            yield "streamed"

    monkeypatch.setattr(
        universal_module,
        "SERVICE_PROVIDER_REGISTRY",
        ServiceProviderRegistry({"openrouter": HostedMistralAdapter}),
    )

    adapter = UniversalLLMAPIAdapter(
        organization="mistral",
        service_provider="openrouter",
        model="mistral-large",
        api_key="test-key",
    )

    assert isinstance(adapter.adapter, HostedMistralAdapter)
    assert adapter.adapter.company == "mistral"
    assert adapter.adapter.service_provider == "openrouter"

@pytest.mark.unit
def test_getattr_missing_raises_attribute_error(monkeypatch):
    @dataclass
    class FakeAdapter(universal_module.LLMAdapterBase):
        company: str = "anthropic"
        model: str = ""
        api_key: str = ""

        def chat(self, *args, **kwargs):
            return {"response": "ok"}

        def stream_chat(self, *args, **kwargs):
            yield "streamed"
        
    monkeypatch.setattr(
        universal_module,
        "SERVICE_PROVIDER_REGISTRY",
        ServiceProviderRegistry({"anthropic": FakeAdapter}),
    )
    ua = UniversalLLMAPIAdapter(
        organization="anthropic", model="claude-sonnet-4-5", api_key="k"
    )
    with pytest.raises(AttributeError):
        ua.nonexistent_method()
