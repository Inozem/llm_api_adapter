from dataclasses import dataclass

import pytest

import src.llm_api_adapter.universal_adapter as universal_module
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

        def greet(self, name: str) -> str:
            return f"hello {name} from {self.company}"

        def _normalize_reasoning_level(self, reasoning_level):
            return reasoning_level

    monkeypatch.setattr(
        universal_module.LLMAdapterBase,
        "__subclasses__",
        classmethod(lambda cls: [FakeAdapter]),
        raising=False,
    )
    ua = UniversalLLMAPIAdapter(
        organization="anthropic", model="claude-sonnet-4-5", api_key="sk-test"
    )
    assert isinstance(ua.adapter, FakeAdapter)
    assert ua.adapter.model == "claude-sonnet-4-5"
    assert ua.adapter.api_key == "sk-test"
    assert ua.greet("Alice") == "hello Alice from anthropic"

@pytest.mark.unit
def test_unsupported_organization_raises(monkeypatch):
    @dataclass
    class OtherAdapter(universal_module.LLMAdapterBase):
        company: str = "OpenAI"
        model: str = ""
        api_key: str = ""

        def chat(self, *args, **kwargs):
            return {"response": "openai"}

    monkeypatch.setattr(
        universal_module.LLMAdapterBase,
        "__subclasses__",
        classmethod(lambda cls: [OtherAdapter]),
        raising=False,
    )
    with pytest.raises(ValueError, match="Unsupported organization: UnknownCorp"):
        UniversalLLMAPIAdapter(
            organization="UnknownCorp", model="gpt", api_key="k"
        )

@pytest.mark.unit
def test_invalid_inputs_raise_value_error():
    with pytest.raises(ValueError, match="Invalid organization"):
        UniversalLLMAPIAdapter(organization="", model="m", api_key="k")
    with pytest.raises(ValueError, match="Invalid model"):
        UniversalLLMAPIAdapter(organization="Anthropic", model="", api_key="k")
    with pytest.raises(ValueError, match="Invalid API key"):
        UniversalLLMAPIAdapter(organization="Anthropic", model="m", api_key="")

@pytest.mark.unit
def test_getattr_missing_raises_attribute_error(monkeypatch):
    @dataclass
    class FakeAdapter(universal_module.LLMAdapterBase):
        company: str = "anthropic"
        model: str = ""
        api_key: str = ""

        def chat(self, *args, **kwargs):
            return {"response": "ok"}
        
        def _normalize_reasoning_level(self, reasoning_level):
            return reasoning_level

    monkeypatch.setattr(
        universal_module.LLMAdapterBase,
        "__subclasses__",
        classmethod(lambda cls: [FakeAdapter]),
        raising=False,
    )
    ua = UniversalLLMAPIAdapter(
        organization="anthropic", model="claude-sonnet-4-5", api_key="k"
    )
    with pytest.raises(AttributeError):
        ua.nonexistent_method()
