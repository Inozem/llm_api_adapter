from dataclasses import dataclass
import inspect
import socket

import httpx
import pytest
import requests

import src.llm_api_adapter.universal_adapter as universal_module
from src.llm_api_adapter.adapters.anthropic_adapter import AnthropicAdapter
from src.llm_api_adapter.adapters.google_adapter import GoogleAdapter
from src.llm_api_adapter.adapters.openai_adapter import OpenAIAdapter
from src.llm_api_adapter.service_provider_registry import ServiceProviderRegistry
from src.llm_api_adapter.errors.config_errors import (
    LLMConfigError,
    OrganizationNotInstalledError,
)
from src.llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter


SYNTHETIC_API_KEY_MARKER = "synthetic-api-key-marker"


@pytest.fixture
def openai_compatible_no_io(monkeypatch):
    """Guard configuration-only tests against discovery, DNS, and HTTP I/O."""
    calls = {"discovery": [], "dns": [], "http": []}

    def unexpected(boundary):
        def fail(*args, **kwargs):
            calls[boundary].append((args, kwargs))
            if boundary == "discovery":
                # Record and stop discovery so environment plugins cannot affect a test.
                return None
            raise AssertionError(f"Unexpected {boundary} during configuration")

        return fail

    monkeypatch.setattr(
        universal_module.ORGANIZATION_PLUGIN_DISCOVERY,
        "discover",
        unexpected("discovery"),
    )
    monkeypatch.setattr(socket, "getaddrinfo", unexpected("dns"))
    monkeypatch.setattr(requests.sessions.Session, "send", unexpected("http"))
    monkeypatch.setattr(httpx.Client, "send", unexpected("http"))
    monkeypatch.setattr(httpx.AsyncClient, "send", unexpected("http"))
    yield calls
    assert calls == {"discovery": [], "dns": [], "http": []}


@pytest.fixture
def openai_compatible_no_adapter_selection(monkeypatch):
    """Ensure selector conflicts fail before choosing any adapter factory."""
    calls = []

    def unexpected(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("Adapter selection must not run for a configuration conflict")

    monkeypatch.setattr(UniversalLLMAPIAdapter, "_select_adapter", unexpected)
    return calls


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
    adapter = UniversalLLMAPIAdapter(
        organization="anthropic",
        model="claude-sonnet-4-5",
        api_key="sk-test",
    )

    assert isinstance(adapter.adapter, FakeAdapter)
    assert adapter.adapter.model == "claude-sonnet-4-5"
    assert adapter.adapter.api_key == "sk-test"
    assert adapter.greet("Alice") == "hello Alice from anthropic"
    assert list(adapter.stream_chat()) == ["streamed"]


@pytest.mark.unit
def test_unknown_organization_raises(monkeypatch):
    monkeypatch.setattr(
        universal_module,
        "SERVICE_PROVIDER_REGISTRY",
        ServiceProviderRegistry(),
    )

    with pytest.raises(ValueError, match="Unsupported organization: UnknownCorp"):
        UniversalLLMAPIAdapter(
            organization="UnknownCorp",
            model="test-model",
            api_key="k",
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
        ("google", "gemini-3.8-flash", GoogleAdapter),
    ],
)
def test_builtin_organization_registry_selects_each_adapter(
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
    assert adapter.adapter.company == organization
    assert adapter.service_provider == adapter.adapter.service_provider == organization


@pytest.mark.unit
def test_transport_selection_is_validated_and_forwarded_to_organization_adapter():
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
def test_zai_organization_can_route_glm_through_discovered_mistral(monkeypatch):
    @dataclass
    class MistralAdapter:
        company: str
        model: str
        api_key: str
        transport: str
        service_provider: str

        def chat(self, *args, **kwargs):
            return {"response": "ok"}

        def stream_chat(self, *args, **kwargs):
            yield "streamed"

    discoveries = []

    def discover(registry, model_registry=None):
        discoveries.append(registry)
        registry.register("mistral", MistralAdapter)

    monkeypatch.setattr(
        universal_module,
        "SERVICE_PROVIDER_REGISTRY",
        ServiceProviderRegistry(),
    )
    monkeypatch.setattr(
        universal_module,
        "ORGANIZATION_PLUGIN_DISCOVERY",
        type("FakeDiscovery", (), {"discover": staticmethod(discover)})(),
    )

    adapter = UniversalLLMAPIAdapter(
        organization="zai",
        service_provider="mistral",
        model="glm-5.3-flash",
        api_key="mistral-test-key",
    )

    assert len(discoveries) == 1
    assert isinstance(adapter.adapter, MistralAdapter)
    assert adapter.adapter.company == "zai"
    assert adapter.adapter.service_provider == "mistral"
    assert adapter.adapter.model == "glm-5.3-flash"
    assert adapter.adapter.api_key == "mistral-test-key"
    assert adapter.chat() == {"response": "ok"}


@pytest.mark.unit
def test_missing_mistral_provider_is_distinct_from_missing_zai_package(
    monkeypatch,
):
    monkeypatch.setattr(
        universal_module,
        "SERVICE_PROVIDER_REGISTRY",
        ServiceProviderRegistry(),
    )
    monkeypatch.setattr(
        universal_module,
        "ORGANIZATION_PLUGIN_DISCOVERY",
        type("EmptyDiscovery", (), {"discover": lambda self, *args, **kwargs: None})(),
    )

    with pytest.raises(
        ValueError,
        match="Unsupported service provider: mistral",
    ):
        UniversalLLMAPIAdapter(
            organization="zai",
            service_provider="mistral",
            model="glm-5.3-flash",
            api_key="mistral-test-key",
        )

    with pytest.raises(OrganizationNotInstalledError) as error:
        UniversalLLMAPIAdapter(
            organization="zai",
            model="glm-5.3-flash",
            api_key="zai-test-key",
        )

    assert error.value.distribution == "llm-api-adapter-zai"


@pytest.mark.unit
def test_omitted_provider_keeps_zai_direct_default_when_mistral_is_discoverable(
    monkeypatch,
):
    @dataclass
    class MistralAdapter:
        company: str
        model: str
        api_key: str
        transport: str
        service_provider: str

        def chat(self, *args, **kwargs):
            return {"response": "unexpected"}

        def stream_chat(self, *args, **kwargs):
            yield "unexpected"

    created = []

    def discover(registry, model_registry=None):
        registry.register("mistral", MistralAdapter)

    monkeypatch.setattr(
        universal_module,
        "SERVICE_PROVIDER_REGISTRY",
        ServiceProviderRegistry(),
    )
    monkeypatch.setattr(
        universal_module,
        "ORGANIZATION_PLUGIN_DISCOVERY",
        type("FakeDiscovery", (), {"discover": staticmethod(discover)})(),
    )
    original_init = MistralAdapter.__init__

    def track_init(self, *args, **kwargs):
        created.append(self)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(MistralAdapter, "__init__", track_init)

    with pytest.raises(OrganizationNotInstalledError) as error:
        UniversalLLMAPIAdapter(
            organization="zai",
            model="glm-5.3-flash",
            api_key="zai-test-key",
        )

    assert error.value.distribution == "llm-api-adapter-zai"
    assert created == []


@pytest.mark.unit
def test_mistral_request_failure_does_not_fall_back_to_zai(monkeypatch):
    zai_requests = []

    @dataclass
    class MistralAdapter:
        company: str
        model: str
        api_key: str
        transport: str
        service_provider: str

        def chat(self, *args, **kwargs):
            raise RuntimeError("Mistral request failed")

        def stream_chat(self, *args, **kwargs):
            yield "unused"

    @dataclass
    class ZaiAdapter:
        company: str
        model: str
        api_key: str
        transport: str
        service_provider: str

        def chat(self, *args, **kwargs):
            zai_requests.append((args, kwargs))
            return {"response": "unexpected fallback"}

        def stream_chat(self, *args, **kwargs):
            yield "unused"

    monkeypatch.setattr(
        universal_module,
        "SERVICE_PROVIDER_REGISTRY",
        ServiceProviderRegistry({"mistral": MistralAdapter, "zai": ZaiAdapter}),
    )

    adapter = UniversalLLMAPIAdapter(
        organization="zai",
        service_provider="mistral",
        model="glm-5.3-flash",
        api_key="mistral-test-key",
    )

    with pytest.raises(RuntimeError, match="Mistral request failed"):
        adapter.chat("hello")

    assert zai_requests == []


@pytest.mark.unit
def test_public_constructor_signature_remains_compatible():
    parameters = inspect.signature(UniversalLLMAPIAdapter).parameters

    assert list(parameters) == [
        "organization",
        "model",
        "api_key",
        "transport",
        "service_provider",
        "base_url",
        "api_variant",
    ]
    assert [parameter.default for parameter in parameters.values()] == [
        inspect.Parameter.empty,
        inspect.Parameter.empty,
        inspect.Parameter.empty,
        "requests",
        None,
        None,
        None,
    ]


@pytest.mark.unit
def test_caller_pricing_native_facade_accessors_preserve_registry_isolation():
    omitted = UniversalLLMAPIAdapter(
        organization="openai", model="gpt-5", api_key="test-key"
    )
    separate = UniversalLLMAPIAdapter(
        organization="openai",
        model="gpt-5",
        api_key="test-key",
    )
    registered = omitted.adapter.model_spec.pricing_tiers

    assert omitted.adapter.pricing is not None
    assert separate.adapter.pricing is not None
    assert omitted.pricing is omitted.adapter.pricing
    assert separate.pricing is separate.adapter.pricing
    assert omitted.pricing is not separate.pricing
    before = registered.tiers

    omitted.pricing.set_in_per_1m(77.0)
    omitted.pricing.set_currency("JPY")

    assert omitted.pricing.currency == "JPY"
    assert omitted.pricing.tiers[0].in_per_token == pytest.approx(77 / 1_000_000)
    assert separate.pricing.tiers == before
    assert registered.tiers == before


@pytest.mark.unit
def test_caller_pricing_compatible_six_position_call_remains_valid(openai_compatible_no_io):
    facade = UniversalLLMAPIAdapter(
        "openai_compatible",
        "vendor/six-position-model",
        SYNTHETIC_API_KEY_MARKER,
        "requests",
        None,
        "https://example.invalid/v1",
    )

    assert facade.adapter.model == "vendor/six-position-model"
    assert facade.adapter.model_spec is None
    assert facade.adapter.pricing is None


@pytest.mark.unit
@pytest.mark.parametrize("base_url_option", [{}, {"base_url": None}])
def test_openai_compatible_legacy_positional_and_factory_contract(
    monkeypatch,
    base_url_option,
    openai_compatible_no_io,
):
    class LegacyFactory:
        def __init__(
            self,
            *,
            company,
            model,
            api_key,
            transport,
            service_provider,
        ):
            self.company = company
            self.model = model
            self.api_key = api_key
            self.transport = transport
            self.service_provider = service_provider

    monkeypatch.setattr(
        universal_module,
        "SERVICE_PROVIDER_REGISTRY",
        ServiceProviderRegistry(
            {"openai": OpenAIAdapter, "legacy-provider": LegacyFactory}
        ),
    )

    # The original five positional parameters retain their established meanings.
    native = UniversalLLMAPIAdapter("openai", "gpt-4o", "test-key", "requests", None)
    assert native.organization == "openai"
    assert native.model == "gpt-4o"
    assert native.api_key == "test-key"
    assert native.transport == "requests"
    assert native.service_provider == "openai"

    # None is still a legacy option and must not reach a factory without base_url.
    legacy = UniversalLLMAPIAdapter(
        "external-org",
        "vendor-model",
        "test-key",
        "requests",
        "legacy-provider",
        **base_url_option,
    )
    assert isinstance(legacy.adapter, LegacyFactory)
    assert legacy.adapter.company == "external-org"
    assert legacy.adapter.model == "vendor-model"
    assert legacy.adapter.api_key == "test-key"
    assert legacy.adapter.transport == "requests"
    assert legacy.adapter.service_provider == "legacy-provider"


@pytest.mark.unit
@pytest.mark.parametrize("base_url_option", [{}, {"base_url": None}])
def test_openai_compatible_requires_an_api_root(
    base_url_option,
    openai_compatible_no_io,
):
    with pytest.raises(LLMConfigError) as error:
        UniversalLLMAPIAdapter(
            organization="openai_compatible",
            model="unregistered-model",
            api_key=SYNTHETIC_API_KEY_MARKER,
            **base_url_option,
        )

    diagnostic = str(error.value)
    assert SYNTHETIC_API_KEY_MARKER not in diagnostic
    assert any(
        term in diagnostic.lower()
        for term in ("base_url", "api root", "endpoint", "url", "address")
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "base_url",
    [
        "",
        "   ",
        False,
        0,
        123,
        "localhost:8000/v1",
        "//example.invalid/v1",
        "/v1",
        "ftp://example.invalid/v1",
        "http:///v1",
        "https://",
        "https:// example.invalid/v1",
        " https://example.invalid/v1",
        "https://example.invalid/v1 ",
        "https://example.invalid:bad/v1",
        "https://:8080/v1",
        "https://example.invalid:-1/v1",
        "https://example.invalid:65536/v1",
        "https://[2001:db8::1/v1",
        "https://[2001:db8::zz]:8080/v1",
        "https://user:synthetic-userinfo-marker@example.invalid/v1",
        "https://example.invalid/v1?token=synthetic-query-marker",
        "https://example.invalid/v1?",
        "https://example.invalid/v1#fragment",
        "https://example.invalid/v1#",
        "https://example.invalid/v1/chat/completions",
        "https://example.invalid/v1/chat/completions/",
        "https://example.invalid/gateway/chat/completions",
    ],
)
def test_openai_compatible_rejects_invalid_api_roots(
    base_url,
    openai_compatible_no_io,
):
    with pytest.raises(LLMConfigError) as error:
        UniversalLLMAPIAdapter(
            organization="openai_compatible",
            model="unregistered-model",
            api_key=SYNTHETIC_API_KEY_MARKER,
            base_url=base_url,
        )

    diagnostic = str(error.value)
    assert SYNTHETIC_API_KEY_MARKER not in diagnostic
    assert "synthetic-userinfo-marker" not in diagnostic
    assert "synthetic-query-marker" not in diagnostic
    assert any(
        term in diagnostic.lower()
        for term in ("base_url", "api root", "endpoint", "url", "address")
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "base_url",
    [
        "http://localhost:11434/v1",
        "http://localhost:11434/v1/",
        "https://example.invalid/gateway/tenant/v1/",
        "https://example.invalid/api",
        "https://[2001:db8::1]:9443/proxy/v1/",
    ],
)
@pytest.mark.parametrize(
    "service_provider_option",
    [{}, {"service_provider": "openai_compatible"}],
)
def test_openai_compatible_accepts_api_roots_and_unregistered_models(
    base_url,
    service_provider_option,
    openai_compatible_no_io,
):
    model = "vendor/unregistered-model"
    adapter = UniversalLLMAPIAdapter(
        organization="openai_compatible",
        model=model,
        api_key=SYNTHETIC_API_KEY_MARKER,
        base_url=base_url,
        **service_provider_option,
    )

    assert adapter.organization == "openai_compatible"
    assert adapter.service_provider == "openai_compatible"
    assert adapter.model == model
    assert adapter.adapter.model == model
    assert adapter.adapter.company == "openai_compatible"
    assert adapter.adapter.service_provider == "openai_compatible"
    assert adapter.adapter.api_key == SYNTHETIC_API_KEY_MARKER


@pytest.mark.unit
@pytest.mark.parametrize("base_url", ["", False, 0, "https://custom.invalid/v1"])
@pytest.mark.parametrize(
    "organization",
    ["openai", "anthropic", "google", "mistral", "UnknownCorp"],
)
def test_openai_compatible_rejects_base_url_for_other_organizations_before_selection(
    organization,
    base_url,
    openai_compatible_no_io,
    openai_compatible_no_adapter_selection,
):
    with pytest.raises(LLMConfigError) as error:
        UniversalLLMAPIAdapter(
            organization=organization,
            model="valid-model",
            api_key=SYNTHETIC_API_KEY_MARKER,
            base_url=base_url,
        )

    diagnostic = str(error.value)
    assert SYNTHETIC_API_KEY_MARKER not in diagnostic
    assert any(
        term in diagnostic.lower()
        for term in ("base_url", "api root", "endpoint", "url", "address")
    )

    assert openai_compatible_no_adapter_selection == []


@pytest.mark.unit
@pytest.mark.parametrize("service_provider", ["openai", "openrouter"])
def test_openai_compatible_rejects_conflicting_service_provider_before_selection(
    service_provider,
    openai_compatible_no_io,
    openai_compatible_no_adapter_selection,
):
    with pytest.raises(LLMConfigError) as error:
        UniversalLLMAPIAdapter(
            organization="openai_compatible",
            model="unregistered-model",
            api_key=SYNTHETIC_API_KEY_MARKER,
            service_provider=service_provider,
            base_url="https://example.invalid/v1",
        )

    diagnostic = str(error.value)
    assert SYNTHETIC_API_KEY_MARKER not in diagnostic
    assert any(
        term in diagnostic.lower()
        for term in ("service_provider", "provider", "openai_compatible")
    )

    assert openai_compatible_no_adapter_selection == []


@pytest.mark.unit
@pytest.mark.parametrize(
    ("organization", "service_provider", "base_url"),
    [
        ("openai", None, "https://custom.invalid/v1"),
        ("mistral", "openrouter", False),
        ("mistral", "external-hosted", 0),
    ],
)
def test_noncompatible_root_is_rejected_before_native_or_hosted_factory(
    monkeypatch,
    organization,
    service_provider,
    base_url,
    openai_compatible_no_io,
    openai_compatible_no_adapter_selection,
):
    factory_calls = []

    class HostedFactory:
        def __init__(self, **kwargs):
            factory_calls.append(kwargs)

    monkeypatch.setattr(
        universal_module,
        "SERVICE_PROVIDER_REGISTRY",
        ServiceProviderRegistry({
            "openrouter": HostedFactory,
            "external-hosted": HostedFactory,
        }),
    )

    kwargs = {} if service_provider is None else {"service_provider": service_provider}
    with pytest.raises(LLMConfigError):
        UniversalLLMAPIAdapter(
            organization=organization,
            model="gpt-5" if organization == "openai" else "mistral-large",
            api_key=SYNTHETIC_API_KEY_MARKER,
            base_url=base_url,
            **kwargs,
        )

    assert factory_calls == []
    assert openai_compatible_no_adapter_selection == []


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
    adapter = UniversalLLMAPIAdapter(
        organization="anthropic",
        model="claude-sonnet-4-5",
        api_key="k",
    )
    with pytest.raises(AttributeError):
        adapter.nonexistent_method()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("api_variant_option", "expected_variant"),
    [
        ({}, "chat_completions"),
        ({"api_variant": None}, "chat_completions"),
        ({"api_variant": "chat_completions"}, "chat_completions"),
        ({"api_variant": "responses"}, "responses"),
    ],
)
def test_facade_api_variant_defaults_and_passes_resolved_value_to_generic_factory(
    monkeypatch, openai_compatible_no_io, api_variant_option, expected_variant
):
    """Connection contract:

    organization — “Exact `openai_compatible` selector; other organizations retain native selection.”
    model — “Existing nonempty exact string, no registry alias/snapshot inference.”
    key — “Existing explicit nonempty credential; never persisted or disclosed.”
    transport — “Existing requests default, explicit optional sync HTTPX, optional async.”
    provider — “Omitted means compatible mode's own key; conflicting value fails.”
    root — “Facade default None; valid HTTP(S) API root required only in compatible mode.”
    API — “Facade default None; compatible effective default `chat_completions`, otherwise exact `chat_completions` or `responses`. Non-None is forbidden for other organizations.”
    """
    factory_calls = []

    class GenericFactory:
        def __init__(self, **kwargs):
            factory_calls.append(kwargs)

    monkeypatch.setattr(
        universal_module,
        "SERVICE_PROVIDER_REGISTRY",
        ServiceProviderRegistry({"openai_compatible": GenericFactory}),
    )
    facade = UniversalLLMAPIAdapter(
        organization="openai_compatible",
        model="vendor/exact-model-name",
        api_key=SYNTHETIC_API_KEY_MARKER,
        base_url="https://example.invalid/gateway/v1///",
        **api_variant_option,
    )

    assert facade.model == "vendor/exact-model-name"
    assert facade.service_provider == "openai_compatible"
    assert factory_calls == [{
        "company": "openai_compatible",
        "model": "vendor/exact-model-name",
        "api_key": SYNTHETIC_API_KEY_MARKER,
        "transport": "requests",
        "service_provider": "openai_compatible",
        "base_url": "https://example.invalid/gateway/v1",
        "api_variant": expected_variant,
    }]


@pytest.mark.unit
@pytest.mark.parametrize(
    "api_variant",
    [
        "",
        " ",
        "chat_completions ",
        " Chat_Completions",
        "Chat_Completions",
        "RESPONSES",
        "responses ",
        "completion",
        False,
        True,
        0,
        1,
        [],
        {},
    ],
)
def test_facade_rejects_invalid_api_variant_before_discovery_or_io(
    api_variant,
    openai_compatible_no_io,
    openai_compatible_no_adapter_selection,
):
    with pytest.raises(LLMConfigError):
        UniversalLLMAPIAdapter(
            organization="openai_compatible",
            model="vendor/exact-model",
            api_key=SYNTHETIC_API_KEY_MARKER,
            base_url="https://example.invalid/v1",
            api_variant=api_variant,
        )
    assert openai_compatible_no_adapter_selection == []


@pytest.mark.unit
@pytest.mark.parametrize(
    ("organization", "service_provider"),
    [("openai", None), ("mistral", "external-hosted")],
)
@pytest.mark.parametrize(
    "api_variant",
    ["chat_completions", "responses", "invalid", "", False, 0],
)
def test_noncompatible_api_variant_rejected_before_native_or_hosted_factory(
    monkeypatch,
    openai_compatible_no_io,
    openai_compatible_no_adapter_selection,
    organization,
    service_provider,
    api_variant,
):
    factory_calls = []

    class HostedFactory:
        def __init__(self, **kwargs):
            factory_calls.append(kwargs)

    monkeypatch.setattr(
        universal_module,
        "SERVICE_PROVIDER_REGISTRY",
        ServiceProviderRegistry({"external-hosted": HostedFactory}),
    )
    options = {} if service_provider is None else {"service_provider": service_provider}
    with pytest.raises(LLMConfigError):
        UniversalLLMAPIAdapter(
            organization=organization,
            model="gpt-5" if organization == "openai" else "vendor/model",
            api_key=SYNTHETIC_API_KEY_MARKER,
            api_variant=api_variant,
            **options,
        )

    assert factory_calls == []
    assert openai_compatible_no_adapter_selection == []


@pytest.mark.unit
@pytest.mark.parametrize(
    "api_variant_option",
    [{}, {"api_variant": None}],
)
def test_legacy_factory_kwargs_omit_api_variant_for_omitted_and_none(
    monkeypatch, openai_compatible_no_io, api_variant_option
):
    factory_calls = []

    class LegacyFactory:
        def __init__(self, **kwargs):
            factory_calls.append(kwargs)

    monkeypatch.setattr(
        universal_module,
        "SERVICE_PROVIDER_REGISTRY",
        ServiceProviderRegistry({"legacy-provider": LegacyFactory}),
    )
    facade = UniversalLLMAPIAdapter(
        "external-org",
        "vendor/model",
        SYNTHETIC_API_KEY_MARKER,
        "requests",
        "legacy-provider",
        **api_variant_option,
    )

    assert facade.adapter is not None
    assert factory_calls == [{
        "company": "external-org",
        "model": "vendor/model",
        "api_key": SYNTHETIC_API_KEY_MARKER,
        "transport": "requests",
        "service_provider": "legacy-provider",
    }]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("organization", "provider_option", "expected_company", "expected_provider"),
    [
        ("openai", {}, "openai", "openai"),
        (
            "mistral",
            {"service_provider": "external-hosted"},
            "mistral",
            "external-hosted",
        ),
    ],
)
@pytest.mark.parametrize("api_variant_option", [{}, {"api_variant": None}])
def test_native_and_hosted_factory_kwargs_omit_api_variant_and_root(
    monkeypatch,
    openai_compatible_no_io,
    organization,
    provider_option,
    expected_company,
    expected_provider,
    api_variant_option,
):
    factory_calls = []

    class FactorySpy:
        def __init__(self, **kwargs):
            factory_calls.append(kwargs)

    monkeypatch.setattr(
        universal_module,
        "SERVICE_PROVIDER_REGISTRY",
        ServiceProviderRegistry({
            "openai": FactorySpy,
            "external-hosted": FactorySpy,
        }),
    )
    facade = UniversalLLMAPIAdapter(
        organization,
        "gpt-5" if organization == "openai" else "vendor/model",
        SYNTHETIC_API_KEY_MARKER,
        **provider_option,
        **api_variant_option,
    )

    assert facade.adapter is not None
    assert factory_calls == [{
        "company": expected_company,
        "model": "gpt-5" if organization == "openai" else "vendor/model",
        "api_key": SYNTHETIC_API_KEY_MARKER,
        "transport": "requests",
        "service_provider": expected_provider,
    }]


@pytest.mark.unit
@pytest.mark.parametrize(
    "api_variant_option",
    [{}, {"api_variant": None}, {"api_variant": "chat_completions"},
     {"api_variant": "responses"}],
)
@pytest.mark.parametrize("resource_suffix", ["/chat/completions", "/responses"])
def test_facade_rejects_both_full_resource_roots_for_every_api_variant(
    openai_compatible_no_io,
    openai_compatible_no_adapter_selection,
    api_variant_option,
    resource_suffix,
):
    with pytest.raises(LLMConfigError):
        UniversalLLMAPIAdapter(
            organization="openai_compatible",
            model="vendor/model",
            api_key=SYNTHETIC_API_KEY_MARKER,
            base_url=f"https://example.invalid/gateway/v1{resource_suffix}/",
            **api_variant_option,
        )
    assert openai_compatible_no_adapter_selection == []


@pytest.mark.unit
def test_compatible_seventh_positional_argument_selects_responses_before_io(
    openai_compatible_no_io,
):
    facade = UniversalLLMAPIAdapter(
        "openai_compatible",
        "vendor/seventh-position-model",
        SYNTHETIC_API_KEY_MARKER,
        "requests",
        None,
        "https://example.invalid/gateway/v1/",
        "responses",
    )

    assert facade.model == "vendor/seventh-position-model"
    assert facade.adapter.model == "vendor/seventh-position-model"
    assert facade.base_url == "https://example.invalid/gateway/v1"
    assert facade.adapter.api_variant == "responses"
