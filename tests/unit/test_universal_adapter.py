from dataclasses import dataclass
import pytest

import src.llm_api_adapter.universal_adapter as universal_module
from src.llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter

def test_selects_adapter_and_delegates(monkeypatch):
    @dataclass
    class FakeAdapter(universal_module.LLMAdapterBase):
        company: str = "Anthropic"
        model: str = ""
        api_key: str = ""

        def chat(self, *args, **kwargs):
            return {"response": "ok"}

        def greet(self, name: str) -> str:
            return f"hello {name} from {self.company}"

    monkeypatch.setattr(
        universal_module.LLMAdapterBase,
        "__subclasses__",
        classmethod(lambda cls: [FakeAdapter]),
        raising=False,
    )
    ua = UniversalLLMAPIAdapter(organization="Anthropic", model="claude-2",
                                api_key="sk-test")
    assert isinstance(ua.adapter, FakeAdapter)
    assert ua.adapter.model == "claude-2"
    assert ua.adapter.api_key == "sk-test"
    assert ua.greet("Alice") == "hello Alice from Anthropic"


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
        UniversalLLMAPIAdapter(organization="UnknownCorp", model="gpt", api_key="k")

def test_invalid_inputs_raise_value_error():
    with pytest.raises(ValueError, match="Invalid organization"):
        UniversalLLMAPIAdapter(organization="", model="m", api_key="k")

    with pytest.raises(ValueError, match="Invalid model"):
        UniversalLLMAPIAdapter(organization="Anthropic", model="", api_key="k")

    with pytest.raises(ValueError, match="Invalid API key"):
        UniversalLLMAPIAdapter(organization="Anthropic", model="m", api_key="")

def test_getattr_missing_raises_attribute_error(monkeypatch):
    @dataclass
    class FakeAdapter(universal_module.LLMAdapterBase):
        company: str = "Anthropic"
        model: str = ""
        api_key: str = ""

        def chat(self, *args, **kwargs):
            return {"response": "ok"}

    monkeypatch.setattr(
        universal_module.LLMAdapterBase,
        "__subclasses__",
        classmethod(lambda cls: [FakeAdapter]),
        raising=False,
    )
    ua = UniversalLLMAPIAdapter(organization="Anthropic", model="m", api_key="k")
    with pytest.raises(AttributeError):
        ua.nonexistent_method()
