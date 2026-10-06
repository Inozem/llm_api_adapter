"""Characterize the capability catalogue before implementing it."""

from dataclasses import replace
from importlib import import_module

import pytest


_MODULE = "src.llm_api_adapter.llm_registry.model_capabilities"


@pytest.fixture(scope="module")
def capability_module():
    try:
        return import_module(_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name == _MODULE:
            pytest.fail(
                "T003 must provide llm_registry/model_capabilities.py",
                pytrace=False,
            )
        raise


@pytest.mark.unit
def test_catalogue_has_unique_valid_identifiers(capability_module):
    catalogue = capability_module.CAPABILITY_CATALOGUE
    ids = [entry.id for entry in catalogue]

    assert ids
    assert all(isinstance(identifier, str) and identifier for identifier in ids)
    assert len(ids) == len(set(ids)), "duplicate capability identifiers"
    assert all("::" not in identifier for identifier in ids)
    capability_module.validate_capability_catalogue(catalogue)


@pytest.mark.unit
def test_only_model_dependent_capabilities_can_be_profile_decisions(capability_module):
    scopes = {entry.id: entry.scope for entry in capability_module.CAPABILITY_CATALOGUE}

    assert set(scopes.values()) == {"model-dependent", "always-on"}
    for identifier in (
        "facade_discovery",
        "message_normalization",
        "transport_parity",
        "error_normalization",
        "request_rule_fidelity",
        "pricing_correctness",
        "missing_usage_honesty",
    ):
        assert scopes[identifier] == "always-on"


@pytest.mark.unit
@pytest.mark.parametrize(
    "variants",
    [
        ("sync_chat", "async_chat", "sync_streaming", "async_streaming"),
        ("tool_choice_auto", "tool_choice_none", "tool_choice_any", "tool_choice_named"),
        ("image_url", "image_bytes", "image_data_url", "pdf_url", "pdf_bytes"),
        ("reasoning_control", "reasoning_events"),
        ("refusal_outcome", "incomplete_outcome"),
    ],
)
def test_meaningful_variants_have_distinct_identifiers(capability_module, variants):
    model_ids = {
        entry.id
        for entry in capability_module.CAPABILITY_CATALOGUE
        if entry.scope == "model-dependent"
    }

    assert set(variants) <= model_ids
    assert len(variants) == len(set(variants))


@pytest.mark.unit
def test_catalogue_rejects_duplicate_and_unknown_identifiers(capability_module):
    catalogue = capability_module.CAPABILITY_CATALOGUE
    first = catalogue[0]

    with pytest.raises(ValueError, match="(?i)duplicate"):
        capability_module.validate_capability_catalogue((*catalogue, first))

    with pytest.raises(ValueError, match="(?i)missing"):
        capability_module.validate_capability_catalogue(catalogue[1:])

    unknown = replace(first, id="unregistered_provider_shortcut")
    with pytest.raises(ValueError, match="(?i)unknown"):
        capability_module.validate_capability_catalogue((unknown, *catalogue[1:]))


@pytest.mark.unit
def test_catalogue_rejects_invalid_scope(capability_module):
    catalogue = capability_module.CAPABILITY_CATALOGUE
    first = catalogue[0]

    with pytest.raises(ValueError, match="(?i)scope"):
        capability_module.validate_capability_catalogue(
            (replace(first, scope="optional"), *catalogue[1:])
        )
