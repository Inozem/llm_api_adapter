"""Static E2E evidence routes for model capability profiles.

Pytest node IDs stay here so runtime registry metadata contains only stable
capability and behavior identifiers. A route may be shared by multiple
capabilities when the same test proves both contracts.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import json
from pathlib import Path
import re

from llm_api_adapter.llm_registry.model_capabilities import (
    ALWAYS_ON_CAPABILITY_IDS,
    CAPABILITY_CATALOGUE,
    MODEL_DEPENDENT_CAPABILITY_IDS,
    ModelCapability,
)


@dataclass(frozen=True)
class CapabilityScenario:
    """One baseline-positive or unconditional scenario route."""

    capability_id: str
    node_id: str


@dataclass(frozen=True)
class ExceptionScenario:
    """One replacement route for a non-pass capability behavior pair."""

    capability_id: str
    behavior_id: str
    node_id: str


@dataclass(frozen=True)
class PassSupplement:
    """One additive route scoped to an exact organization, model, and capability."""

    organization: str
    model: str
    capability_id: str
    node_id: str


@dataclass(frozen=True)
class DeclaredException:
    """A capability exception read from one first-party model profile."""

    organization: str
    model: str
    capability_id: str
    behavior_id: str


@dataclass(frozen=True)
class ScenarioCatalogue:
    """Positive, exception, package-supplement, and always-on test routes."""

    positive: tuple[CapabilityScenario, ...]
    exceptions: tuple[ExceptionScenario, ...]
    supplements: tuple[PassSupplement, ...]
    always_on: tuple[CapabilityScenario, ...]


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_CORE_CATALOGUES = (
    REPOSITORY_ROOT / "src" / "llm_api_adapter" / "llm_registry" / "organizations"
)
_PACKAGE_CATALOGUES = REPOSITORY_ROOT / "packages" / "organizations"
_BEHAVIOR_ID_PATTERN = re.compile(r"[a-z][a-z0-9_]*\Z")
_STANDARD_REASONING_CHAT_SCENARIO = (
    "tests/e2e/test_llm_adapter_chat.py::"
    "test_chat_with_reasoning_level_returns_valid_contract"
)
# The shared tool loop uses one viable mode per model. It does not establish
# separate E2E evidence for every tool-choice mode or provider-side state.
_NO_SEPARATE_SHARED_E2E_SCENARIO_IDS = frozenset(
    {
        "tool_choice_auto",
        "tool_choice_none",
        "tool_choice_any",
        "tool_choice_named",
        "provider_continuation",
    }
)
E2E_SCENARIO_CAPABILITIES = tuple(
    capability
    for capability in CAPABILITY_CATALOGUE
    if capability.id not in _NO_SEPARATE_SHARED_E2E_SCENARIO_IDS
)


SCENARIO_CATALOGUE = ScenarioCatalogue(
    positive=(
        CapabilityScenario(
            "sync_chat",
            "tests/e2e/test_llm_adapter_chat.py::test_chat_accepts_basic_params_and_returns_contract",
        ),
        CapabilityScenario(
            "async_chat",
            "tests/e2e/test_async.py::test_async_chat_returns_structured_response_and_pricing",
        ),
        CapabilityScenario(
            "sync_streaming",
            "tests/e2e/test_streaming.py::test_stream_chat_returns_text_and_finalized_response",
        ),
        CapabilityScenario(
            "async_streaming",
            "tests/e2e/test_async.py::test_async_streaming_preserves_callbacks_and_final_response",
        ),
        CapabilityScenario(
            "application_tools",
            "tests/e2e/test_tools_auto_loop.py::test_basic_tool_loop_with_previous_response",
        ),
        CapabilityScenario(
            "structured_output_schema",
            "tests/e2e/test_json_schema.py::test_json_schema_returns_structured_output_for_every_configured_model",
        ),
        CapabilityScenario(
            "structured_output_model",
            "tests/e2e/test_json_schema.py::test_pydantic_response_model_returns_structured_output",
        ),
        CapabilityScenario(
            "image_url",
            "tests/e2e/test_vision.py::test_vision_url_returns_text",
        ),
        CapabilityScenario(
            "image_bytes",
            "tests/e2e/test_vision.py::test_vision_bytes_returns_non_empty_response",
        ),
        CapabilityScenario(
            "image_data_url",
            "tests/e2e/test_vision.py::test_vision_data_url_returns_text",
        ),
        CapabilityScenario(
            "pdf_url",
            "tests/e2e/test_file_uploads.py::test_document_url_returns_non_empty_response",
        ),
        CapabilityScenario(
            "pdf_bytes",
            "tests/e2e/test_file_uploads.py::test_document_bytes_returns_non_empty_response",
        ),
        CapabilityScenario(
            "reasoning_control",
            _STANDARD_REASONING_CHAT_SCENARIO,
        ),
        CapabilityScenario(
            "reasoning_events",
            "tests/e2e/test_async.py::test_async_reasoning_events_are_normalized",
        ),
        CapabilityScenario(
            "usage_reporting",
            "tests/e2e/test_llm_adapter_chat.py::test_chat_accepts_basic_params_and_returns_contract",
        ),
        CapabilityScenario(
            "refusal_outcome",
            "tests/e2e/test_async.py::test_refusal_outcome_is_normalized",
        ),
        CapabilityScenario(
            "incomplete_outcome",
            "tests/e2e/test_async.py::test_incomplete_outcome_is_normalized",
        ),
    ),
    exceptions=(
        ExceptionScenario(
            "image_url",
            "rejected_before_transport",
            "tests/e2e/test_vision.py::test_vision_url_declared_exception",
        ),
        ExceptionScenario(
            "pdf_bytes",
            "rejected_before_transport",
            "tests/e2e/test_file_uploads.py::test_document_bytes_declared_exception",
        ),
        ExceptionScenario(
            "pdf_url",
            "rejected_before_transport",
            "tests/e2e/test_file_uploads.py::test_document_url_declared_exception",
        ),
        # Every model uses the same public chat contract, including models with
        # different reasoning policies recorded in their capability profiles.
        ExceptionScenario(
            "reasoning_control",
            "cannot_disable_thinking",
            _STANDARD_REASONING_CHAT_SCENARIO,
        ),
        ExceptionScenario(
            "reasoning_control",
            "none_falls_back_to_low",
            _STANDARD_REASONING_CHAT_SCENARIO,
        ),
        ExceptionScenario(
            "reasoning_control",
            "none_to_low_xhigh_to_high",
            _STANDARD_REASONING_CHAT_SCENARIO,
        ),
        ExceptionScenario(
            "reasoning_control",
            "reasoning_unsupported",
            _STANDARD_REASONING_CHAT_SCENARIO,
        ),
        ExceptionScenario(
            "structured_output_model",
            "rejected_before_transport",
            "tests/e2e/test_json_schema.py::test_response_model_declared_exception",
        ),
        ExceptionScenario(
            "structured_output_schema",
            "rejected_before_transport",
            "tests/e2e/test_json_schema.py::test_json_schema_declared_exception",
        ),
    ),
    supplements=(
        PassSupplement(
            "mistral",
            "mistral-small-2603",
            "pdf_url",
            "tests/e2e/test_mistral_ocr_costs.py::test_mistral_pdf_url_ocr_exposes_cost_breakdown",
        ),
        PassSupplement(
            "mistral",
            "mistral-small-2603",
            "pdf_bytes",
            "tests/e2e/test_mistral_ocr_costs.py::test_mistral_pdf_ocr_exposes_cost_breakdown",
        ),
        PassSupplement(
            "mistral",
            "mistral-medium-3-5",
            "pdf_url",
            "tests/e2e/test_mistral_ocr_costs.py::test_mistral_pdf_url_ocr_exposes_cost_breakdown",
        ),
        PassSupplement(
            "mistral",
            "mistral-medium-3-5",
            "pdf_bytes",
            "tests/e2e/test_mistral_ocr_costs.py::test_mistral_pdf_ocr_exposes_cost_breakdown",
        ),
        PassSupplement(
            "mistral",
            "mistral-large-2512",
            "pdf_url",
            "tests/e2e/test_mistral_ocr_costs.py::test_mistral_pdf_url_ocr_exposes_cost_breakdown",
        ),
        PassSupplement(
            "mistral",
            "mistral-large-2512",
            "pdf_bytes",
            "tests/e2e/test_mistral_ocr_costs.py::test_mistral_pdf_ocr_exposes_cost_breakdown",
        ),
        PassSupplement(
            "xai",
            "grok-4.7",
            "pdf_url",
            "tests/e2e/test_file_uploads.py::test_xai_pdf_url_uses_attachment_search",
        ),
        PassSupplement(
            "xai",
            "grok-4.7",
            "pdf_bytes",
            "tests/e2e/test_file_uploads.py::test_xai_pdf_bytes_uses_attachment_search",
        ),
        PassSupplement(
            "xai",
            "grok-4.6",
            "pdf_url",
            "tests/e2e/test_file_uploads.py::test_xai_pdf_url_uses_attachment_search",
        ),
        PassSupplement(
            "xai",
            "grok-4.6",
            "pdf_bytes",
            "tests/e2e/test_file_uploads.py::test_xai_pdf_bytes_uses_attachment_search",
        ),
        PassSupplement(
            "xai",
            "grok-4.5",
            "pdf_url",
            "tests/e2e/test_file_uploads.py::test_xai_pdf_url_uses_attachment_search",
        ),
        PassSupplement(
            "xai",
            "grok-4.5",
            "pdf_bytes",
            "tests/e2e/test_file_uploads.py::test_xai_pdf_bytes_uses_attachment_search",
        ),
    ),
    always_on=(
        CapabilityScenario(
            "facade_discovery",
            "tests/unit/test_organization_profile_compatibility.py::test_profile_parsing_preserves_registry_resolution_plugin_discovery_and_facade",
        ),
        CapabilityScenario(
            "message_normalization",
            "tests/unit/models/messages/test_chat_message.py::test_messages_normalize_openai_style_content_list",
        ),
        CapabilityScenario(
            "response_normalization",
            "tests/unit/conformance/test_facade_contract.py::test_facade_chat_normalizes_messages_response_usage_and_pricing",
        ),
        CapabilityScenario(
            "transport_parity",
            "tests/e2e/test_sync_httpx.py::test_sync_httpx_chat_returns_contract_for_latest_provider_models",
        ),
        CapabilityScenario(
            "stream_cleanup",
            "tests/unit/adapters/test_async_lifecycle.py::test_async_stream_close_before_completion_skips_tool_and_done",
        ),
        CapabilityScenario(
            "tool_validation",
            "tests/unit/conformance/test_facade_contract.py::test_facade_preserves_tool_structured_output_file_and_reasoning_contracts",
        ),
        CapabilityScenario(
            "schema_validation",
            "tests/unit/conformance/test_facade_contract.py::test_facade_rejects_nonportable_core_schema_before_sending_request",
        ),
        CapabilityScenario(
            "error_normalization",
            "tests/e2e/test_errors.py::test_chat_timeout_error",
        ),
        CapabilityScenario(
            "registry_exactness",
            "tests/unit/llm_registry/test_model_profile_inventory.py::test_every_first_party_model_has_a_valid_explicit_exception_profile",
        ),
        CapabilityScenario(
            "request_rule_fidelity",
            "tests/unit/adapters/test_request_rule_conformance.py::test_adapter_accepts_each_registered_tool_choice_mode",
        ),
        CapabilityScenario(
            "pricing_correctness",
            "tests/unit/adapters/test_pricing_lifecycle.py::test_tiered_pricing_matches_sync_chat_and_stream",
        ),
        CapabilityScenario(
            "missing_usage_honesty",
            "tests/unit/adapters/test_pricing_lifecycle.py::test_multi_tier_chat_without_provider_usage_leaves_costs_unset",
        ),
    ),
)


def first_party_declared_exceptions() -> tuple[DeclaredException, ...]:
    """Read model exception IDs from the nine checked-in first-party catalogues."""
    paths = sorted(_CORE_CATALOGUES.glob("*.json")) + sorted(
        _PACKAGE_CATALOGUES.glob("*/src/*/registry/organizations/*.json")
    )
    if len(paths) != 9:
        raise ValueError(
            f"expected nine first-party model catalogues, found {len(paths)}"
        )

    declared = []
    organizations = set()
    for path in paths:
        organization = path.stem
        if organization in organizations:
            raise ValueError(f"duplicate first-party catalogue: {organization}")
        organizations.add(organization)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read first-party catalogue {path}") from exc
        models = data.get("models")
        if not isinstance(models, dict):
            raise ValueError(f"{organization} catalogue must contain a models object")
        for model, profile in models.items():
            if not isinstance(profile, dict):
                raise ValueError(f"{organization}/{model} profile must be an object")
            exceptions = profile.get("capability_exceptions")
            if not isinstance(exceptions, list):
                raise ValueError(
                    f"{organization}/{model} must declare capability_exceptions"
                )
            for exception in exceptions:
                if not isinstance(exception, dict):
                    raise ValueError(
                        f"{organization}/{model} has a malformed capability exception"
                    )
                capability_id = exception.get("capability_id")
                behavior_id = exception.get("behavior_id")
                if not isinstance(capability_id, str) or not isinstance(
                    behavior_id, str
                ):
                    raise ValueError(
                        f"{organization}/{model} exception requires capability_id "
                        "and behavior_id"
                    )
                if not _BEHAVIOR_ID_PATTERN.fullmatch(behavior_id):
                    raise ValueError(
                        f"{organization}/{model} has an invalid behavior_id: "
                        f"{behavior_id!r}"
                    )
                declared.append(
                    DeclaredException(
                        organization=organization,
                        model=model,
                        capability_id=capability_id,
                        behavior_id=behavior_id,
                    )
                )
    return tuple(declared)


def _validate_node_id(node_id: str, *, route: str) -> None:
    if (
        not isinstance(node_id, str)
        or node_id.count("::") != 1
        or not node_id.startswith(("tests/", "packages/organizations/"))
        or "\\" in node_id
        or any(character.isspace() for character in node_id)
    ):
        raise ValueError(f"{route} must contain a valid pytest node ID")
    test_path, _, test_name = node_id.partition("::")
    if not test_path or not test_name:
        raise ValueError(f"{route} must contain a valid pytest node ID")


def _unique_routes(
    records: Iterable,
    key,
    *,
    route_name: str,
    record_type: type,
) -> dict:
    by_key = {}
    for record in records:
        if not isinstance(record, record_type):
            raise ValueError(f"{route_name} evidence has an invalid record")
        try:
            record_key = key(record)
        except (AttributeError, TypeError) as exc:
            raise ValueError(f"{route_name} evidence has an invalid key") from exc
        try:
            duplicate = record_key in by_key
        except TypeError as exc:
            raise ValueError(f"{route_name} evidence has an invalid key") from exc
        if duplicate:
            raise ValueError(f"duplicate {route_name} evidence: {record_key!r}")
        by_key[record_key] = record
    return by_key


def validate_scenario_catalogue(
    catalogue: ScenarioCatalogue = SCENARIO_CATALOGUE,
    *,
    capabilities: Iterable[ModelCapability] | None = None,
    declared_exceptions: Iterable[DeclaredException] | None = None,
) -> None:
    """Reject missing, duplicate, mis-scoped, or unbacked scenario evidence."""
    if not isinstance(catalogue, ScenarioCatalogue):
        raise ValueError("scenario evidence must be a ScenarioCatalogue")
    use_canonical_scopes = capabilities is None
    if capabilities is None:
        capabilities = E2E_SCENARIO_CAPABILITIES

    capability_scopes = {}
    for capability in capabilities:
        if not isinstance(capability, ModelCapability):
            raise ValueError("capability catalogue entries must be ModelCapability values")
        if not isinstance(capability.id, str) or capability.id not in (
            MODEL_DEPENDENT_CAPABILITY_IDS | ALWAYS_ON_CAPABILITY_IDS
        ):
            raise ValueError(f"unknown capability catalogue ID: {capability.id!r}")
        if capability.id in capability_scopes:
            raise ValueError(f"duplicate capability catalogue entry: {capability.id!r}")
        if capability.scope not in ("model-dependent", "always-on"):
            raise ValueError(f"invalid capability scope: {capability.scope!r}")
        capability_scopes[capability.id] = capability.scope

    model_dependent = {
        capability_id
        for capability_id, scope in capability_scopes.items()
        if scope == "model-dependent"
    }
    always_on_ids = {
        capability_id
        for capability_id, scope in capability_scopes.items()
        if scope == "always-on"
    }
    if use_canonical_scopes and (
        model_dependent
        != MODEL_DEPENDENT_CAPABILITY_IDS - _NO_SEPARATE_SHARED_E2E_SCENARIO_IDS
        or always_on_ids != ALWAYS_ON_CAPABILITY_IDS
    ):
        raise ValueError("scenario capability scopes differ from the canonical catalogue")

    positive = _unique_routes(
        catalogue.positive,
        lambda route: route.capability_id,
        route_name="baseline-positive",
        record_type=CapabilityScenario,
    )
    always_on = _unique_routes(
        catalogue.always_on,
        lambda route: route.capability_id,
        route_name="always-on",
        record_type=CapabilityScenario,
    )
    for capability_id, route in positive.items():
        if capability_id not in capability_scopes:
            raise ValueError(f"unknown positive capability route: {capability_id!r}")
        if capability_scopes[capability_id] != "model-dependent":
            raise ValueError(
                f"always-on capability {capability_id!r} cannot have a model route"
            )
        _validate_node_id(route.node_id, route=f"positive {capability_id}")
    for capability_id, route in always_on.items():
        if capability_id not in capability_scopes:
            raise ValueError(f"unknown always-on capability route: {capability_id!r}")
        if capability_scopes[capability_id] != "always-on":
            raise ValueError(
                f"model-dependent capability {capability_id!r} cannot have an "
                "unconditional route"
            )
        _validate_node_id(route.node_id, route=f"always-on {capability_id}")

    missing_positive = model_dependent - positive.keys()
    if missing_positive:
        raise ValueError(
            "missing baseline-positive scenarios: "
            + ", ".join(sorted(missing_positive))
        )
    missing_always_on = always_on_ids - always_on.keys()
    if missing_always_on:
        raise ValueError(
            "missing unconditional scenarios: " + ", ".join(sorted(missing_always_on))
        )

    exception_routes = _unique_routes(
        catalogue.exceptions,
        lambda route: (route.capability_id, route.behavior_id),
        route_name="exception",
        record_type=ExceptionScenario,
    )
    for (capability_id, behavior_id), route in exception_routes.items():
        if capability_id not in capability_scopes:
            raise ValueError(f"unknown exception capability: {capability_id!r}")
        if capability_scopes[capability_id] != "model-dependent":
            raise ValueError(
                f"always-on capability {capability_id!r} cannot declare an exception"
            )
        if (
            not isinstance(behavior_id, str)
            or not _BEHAVIOR_ID_PATTERN.fullmatch(behavior_id)
            or behavior_id == "pass"
        ):
            raise ValueError(
                f"exception route {capability_id!r} requires a valid non-pass behavior_id"
            )
        _validate_node_id(
            route.node_id,
            route=f"exception {capability_id}/{behavior_id}",
        )

    supplements = _unique_routes(
        catalogue.supplements,
        lambda route: (route.organization, route.model, route.capability_id),
        route_name="pass supplement",
        record_type=PassSupplement,
    )
    for key, route in supplements.items():
        organization, model, capability_id = key
        if not all(isinstance(value, str) and value for value in key):
            raise ValueError(f"pass supplement requires organization, model, and capability: {key!r}")
        if capability_id not in model_dependent:
            raise ValueError(f"pass supplement has non-model capability: {key!r}")
        _validate_node_id(route.node_id, route=f"pass supplement {organization}/{model}/{capability_id}")

    profile_inventory_is_default = declared_exceptions is None
    declared = tuple(
        first_party_declared_exceptions()
        if profile_inventory_is_default
        else declared_exceptions
    )
    if any(not isinstance(entry, DeclaredException) for entry in declared):
        raise ValueError("declared exception inventory has an invalid record")
    if len(set(declared)) != len(declared):
        raise ValueError("duplicate declared model exception profile entry")
    expected_exception_pairs = {
        (entry.capability_id, entry.behavior_id)
        for entry in declared
        if entry.capability_id in model_dependent and entry.behavior_id != "pass"
    }
    expected_pass_supplements = {
        (entry.organization, entry.model, entry.capability_id)
        for entry in declared
        if entry.capability_id in model_dependent and entry.behavior_id == "pass"
    }
    missing_exception_routes = expected_exception_pairs - exception_routes.keys()
    if missing_exception_routes:
        missing = ", ".join(
            f"{capability_id}/{behavior_id}"
            for capability_id, behavior_id in sorted(missing_exception_routes)
        )
        raise ValueError(f"missing exception evidence routes: {missing}")
    missing_supplements = expected_pass_supplements - supplements.keys()
    if missing_supplements:
        missing = ", ".join(
            f"{organization}/{model}/{capability_id}"
            for organization, model, capability_id in sorted(missing_supplements)
        )
        raise ValueError(f"missing exact-model pass supplements: {missing}")

    if profile_inventory_is_default:
        orphan_exception_routes = exception_routes.keys() - expected_exception_pairs
        if orphan_exception_routes:
            orphaned = ", ".join(
                f"{capability_id}/{behavior_id}"
                for capability_id, behavior_id in sorted(orphan_exception_routes)
            )
            raise ValueError(f"exception evidence has no declared model route: {orphaned}")
        orphan_supplements = supplements.keys() - expected_pass_supplements
        if orphan_supplements:
            orphaned = ", ".join(
                f"{organization}/{model}/{capability_id}"
                for organization, model, capability_id in sorted(orphan_supplements)
            )
            raise ValueError(f"pass evidence has no declared model exception: {orphaned}")


validate_scenario_catalogue()


__all__ = [
    "CapabilityScenario",
    "DeclaredException",
    "E2E_SCENARIO_CAPABILITIES",
    "ExceptionScenario",
    "PassSupplement",
    "SCENARIO_CATALOGUE",
    "ScenarioCatalogue",
    "first_party_declared_exceptions",
    "validate_scenario_catalogue",
]
