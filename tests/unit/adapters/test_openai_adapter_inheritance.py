"""Characterization for the independent OpenAI protocol adapters."""

from __future__ import annotations

import inspect
import os
from pathlib import Path
import re
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest

from src.llm_api_adapter.adapters.openai.adapter import OpenAIAdapter
from src.llm_api_adapter.adapters.openai_compatible.adapter import (
    OpenAICompatibleAdapter,
)


@pytest.mark.unit
def test_native_and_compatible_constructor_signatures_and_positional_calls():
    native_signature = inspect.signature(OpenAIAdapter)
    compatible_signature = inspect.signature(OpenAICompatibleAdapter)

    native_expected = [
        ("api_key", "str", inspect.Parameter.empty),
        ("model", "str", inspect.Parameter.empty),
        ("company", "str", "openai"),
        ("transport", "str", "requests"),
        ("pricing", "Optional[Pricing]", None),
        ("is_reasoning", "bool", False),
        ("is_adaptive_thinking", "bool", False),
        ("service_provider", "Optional[str]", None),
    ]
    compatible_expected = [
        *native_expected[:2],
        ("company", "str", "openai_compatible"),
        *native_expected[3:],
        ("base_url", "str", ""),
        ("api_variant", "str | None", None),
    ]

    def actual_parameters(signature):
        return [
            (parameter.name, parameter.annotation, parameter.default)
            for parameter in signature.parameters.values()
        ]

    assert actual_parameters(native_signature) == native_expected
    assert native_signature.return_annotation is None
    assert actual_parameters(compatible_signature) == compatible_expected
    assert compatible_signature.return_annotation is None
    assert all(
        parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        for signature in (native_signature, compatible_signature)
        for parameter in signature.parameters.values()
    )

    native = OpenAIAdapter(
        "key", "gpt-4o", "openai", "requests", None, False, False, None
    )
    compatible = OpenAICompatibleAdapter(
        "key",
        "unregistered-model",
        "openai_compatible",
        "requests",
        None,
        False,
        False,
        None,
        "https://example.test/v1",
        "responses",
    )
    assert native.company == "openai"
    assert compatible.company == "openai_compatible"
    assert compatible.base_url == "https://example.test/v1"
    assert compatible.api_variant == "responses"


@pytest.mark.unit
@pytest.mark.parametrize("option", ("base_url", "api_variant"))
def test_native_adapter_rejects_compatible_only_constructor_keywords(option):
    with pytest.raises(
        TypeError,
        match=re.escape(f"unexpected keyword argument '{option}'"),
    ):
        OpenAIAdapter(api_key="key", model="gpt-4o", **{option: "value"})


@pytest.mark.unit
def test_compatible_reasoning_hook_rejects_non_categorical_provider_value(monkeypatch):
    adapter = OpenAICompatibleAdapter(
        api_key="key",
        model="unregistered-model",
        base_url="https://example.test/v1",
    )
    monkeypatch.setattr(
        adapter,
        "_resolve_reasoning_level",
        lambda _: SimpleNamespace(provider_value=7),
    )

    with pytest.raises(
        TypeError,
        match="OpenAI reasoning resolution must produce a categorical value",
    ):
        adapter._resolve_protocol_reasoning_effort("high")


@pytest.mark.unit
@pytest.mark.parametrize("namespace", ("llm_api_adapter", "src.llm_api_adapter"))
@pytest.mark.parametrize("order", ("generic-first", "native-first"))
def test_native_and_compatible_modules_import_in_either_order(namespace, order):
    module_names = {
        "generic": f"{namespace}.adapters.openai_compatible.adapter",
        "native": f"{namespace}.adapters.openai.adapter",
    }
    imports = (
        ("generic", "native")
        if order == "generic-first"
        else ("native", "generic")
    )
    script = "\n".join(
        f"importlib.import_module({module_names[name]!r})" for name in imports
    )
    root = Path(__file__).resolve().parents[3]
    if namespace == "llm_api_adapter":
        script = "import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path.cwd() / 'src'))\n" + script
    result = subprocess.run(
        [sys.executable, "-c", "import importlib\n" + script],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        env=os.environ.copy(),
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.unit
@pytest.mark.parametrize("namespace", ("llm_api_adapter", "src.llm_api_adapter"))
def test_generic_import_does_not_initialize_native_policy_or_optional_packages(
    namespace,
):
    script = textwrap.dedent(
        """
        import importlib
        import sys

        native_roots = (
            'llm_api_adapter.adapters.openai',
            'src.llm_api_adapter.adapters.openai',
            'llm_api_adapter.llms.openai',
            'src.llm_api_adapter.llms.openai',
        )
        optional_roots = (
            'httpx',
            'anyio',
            'openai',
            'llm_api_adapter_mistral',
            'llm_api_adapter_xai',
            'llm_api_adapter_qwen',
            'llm_api_adapter_kimi',
            'llm_api_adapter_deepseek',
            'llm_api_adapter_zai',
        )

        forbidden_roots = native_roots + optional_roots

        class GuardFinder:
            def find_spec(self, fullname, path=None, target=None):
                if any(
                    fullname == root or fullname.startswith(root + '.')
                    for root in forbidden_roots
                ):
                    raise AssertionError('forbidden import: ' + fullname)
                return None

        sys.meta_path.insert(0, GuardFinder())
        importlib.import_module('NAMESPACE.adapters.openai_compatible.adapter')
        assert not any(
            name == root or name.startswith(root + '.')
            for name in sys.modules
            for root in forbidden_roots
        )
        """
    ).replace("NAMESPACE", namespace)
    root = Path(__file__).resolve().parents[3]
    if namespace == "llm_api_adapter":
        script = (
            "import sys\nfrom pathlib import Path\n"
            "sys.path.insert(0, str(Path.cwd() / 'src'))\n" + script
        )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        env=os.environ.copy(),
    )
    assert result.returncode == 0, result.stderr
