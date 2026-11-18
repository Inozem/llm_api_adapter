"""Focused ownership tests for protocol-neutral compatible connections."""

import ast
from pathlib import Path

import pytest

from src.llm_api_adapter.adapters import compatible_connection
from src.llm_api_adapter.adapters.openai_compatible import adapter as openai_adapter
from src.llm_api_adapter.errors.config_errors import LLMConfigError
from src.llm_api_adapter.llm_registry.llm_registry import Pricing, PricingTier


@pytest.mark.unit
def test_neutral_root_validation_uses_only_caller_selected_resource_suffixes():
    root = "https://example.invalid/gateway/v1/messages/"
    assert compatible_connection._validate_api_root(root) == root.rstrip("/")
    with pytest.raises(LLMConfigError, match="API root, not a resource URL"):
        compatible_connection._validate_api_root(
            root, resource_paths=("/messages",)
        )

    for resource_path in ("/chat/completions", "/responses", "/messages"):
        with pytest.raises(LLMConfigError, match="API root, not a resource URL"):
            compatible_connection._validate_api_root(
                f"https://example.invalid/v1{resource_path}/",
                resource_paths=(resource_path,),
            )


@pytest.mark.unit
@pytest.mark.parametrize("resource", ("/chat/completions", "/responses"))
def test_openai_wrapper_keeps_existing_resource_rules(resource):
    with pytest.raises(LLMConfigError, match="API root, not a resource URL"):
        openai_adapter._validate_api_root(f"https://example.invalid/v1{resource}/")

    # Messages is not an OpenAI-compatible endpoint suffix policy.
    assert openai_adapter._validate_api_root(
        "https://example.invalid/v1/messages/"
    ) == "https://example.invalid/v1/messages"
    assert openai_adapter._validate_compatible_connection(
        "https://example.invalid/v1/messages/", None
    ) == ("https://example.invalid/v1/messages", "chat_completions")


@pytest.mark.unit
def test_openai_module_aliases_and_local_patch_bindings_remain(monkeypatch):
    assert openai_adapter._redact_compatible_diagnostic is (
        compatible_connection._redact_compatible_diagnostic
    )
    assert openai_adapter._validate_compatible_pricing is (
        compatible_connection._validate_compatible_pricing
    )
    assert openai_adapter._validate_ip_literal is compatible_connection._validate_ip_literal

    roots = []
    monkeypatch.setattr(
        openai_adapter,
        "_validate_api_root",
        lambda value: roots.append(value) or "patched-root",
    )
    assert openai_adapter._validate_compatible_connection("ignored", None) == (
        "patched-root",
        "chat_completions",
    )
    assert roots == ["ignored"]

    pricing = Pricing((PricingTier(None, 0, 0),))
    validations = []
    monkeypatch.setattr(
        openai_adapter,
        "_validate_compatible_pricing",
        lambda value: validations.append(value),
    )
    instance = openai_adapter.OpenAICompatibleAdapter(
        api_key="synthetic-key",
        model="vendor/model",
        base_url="https://example.invalid/v1",
        pricing=pricing,
    )
    assert validations == [pricing]
    assert instance.pricing == pricing
    assert instance.pricing is not pricing

    monkeypatch.setattr(
        openai_adapter,
        "_redact_compatible_diagnostic",
        lambda text, key: f"patched:{key}:{text}",
    )
    assert instance._format_compatible_diagnostic("message") == (
        "patched:synthetic-key:message"
    )


@pytest.mark.unit
def test_neutral_module_imports_no_protocol_or_native_adapter_policy():
    source_path = Path(compatible_connection.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported_names.append(node.module or "")

    assert not any(
        "openai" in name.lower()
        or "anthropic" in name.lower()
        or "registry_lookup" in name.lower()
        or "organization_registry" in name.lower()
        for name in imported_names
    )
