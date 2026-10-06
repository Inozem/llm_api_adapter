"""Static pytest routes for the capabilities exercised by shared E2E tests."""

from __future__ import annotations

import json
from pathlib import Path

from llm_api_adapter.llm_registry.llm_registry import ModelSpec
from llm_api_adapter.llm_registry.model_capabilities import CAPABILITY_CATALOGUE


def _routes(*entries):
    """Build a route map while rejecting duplicate keys and invalid node IDs."""
    routes = {}
    for key, node_id in entries:
        if key in routes:
            raise ValueError(f"duplicate scenario route: {key!r}")
        if (
            not isinstance(node_id, str)
            or node_id.count("::") != 1
            or not node_id.startswith(("tests/", "packages/organizations/"))
            or "\\" in node_id
            or any(character.isspace() for character in node_id)
        ):
            raise ValueError(f"invalid pytest node ID for {key!r}: {node_id!r}")
        routes[key] = node_id
    return routes


_REASONING_CHAT = (
    "tests/e2e/test_llm_adapter_chat.py::"
    "test_chat_with_reasoning_level_returns_valid_contract"
)

BASELINE_SCENARIOS = _routes(
    (
        "sync_chat",
        "tests/e2e/test_llm_adapter_chat.py::test_chat_accepts_basic_params_and_returns_contract",
    ),
    (
        "async_chat",
        "tests/e2e/test_async.py::test_async_chat_returns_structured_response_and_pricing",
    ),
    (
        "sync_streaming",
        "tests/e2e/test_streaming.py::test_stream_chat_returns_text_and_finalized_response",
    ),
    (
        "async_streaming",
        "tests/e2e/test_async.py::test_async_streaming_preserves_callbacks_and_final_response",
    ),
    (
        "application_tools",
        "tests/e2e/test_tools_auto_loop.py::test_basic_tool_loop_with_previous_response",
    ),
    (
        "structured_output_schema",
        "tests/e2e/test_json_schema.py::test_json_schema_returns_structured_output_for_every_configured_model",
    ),
    (
        "image_bytes",
        "tests/e2e/test_vision.py::test_vision_bytes_returns_non_empty_response",
    ),
    (
        "pdf_bytes",
        "tests/e2e/test_file_uploads.py::test_document_bytes_returns_non_empty_response",
    ),
    ("reasoning_control", _REASONING_CHAT),
    (
        "usage_reporting",
        "tests/e2e/test_llm_adapter_chat.py::test_chat_accepts_basic_params_and_returns_contract",
    ),
)

# Keys are (capability_id, behavior_id, organization). ``None`` is a shared route.
EXCEPTION_SCENARIOS = _routes(
    (
        ("pdf_bytes", "rejected_before_transport", "zai"),
        "packages/organizations/zai/tests/e2e/test_capability_boundaries.py::test_zai_rejects_document_forms_before_provider_transport",
    ),
    (
        ("pdf_bytes", "rejected_before_transport", "kimi"),
        "packages/organizations/kimi/tests/e2e/test_file_contract.py::test_kimi_file_contract_rejects_unsupported_parts_before_transport",
    ),
    (
        ("pdf_bytes", "rejected_before_transport", "deepseek"),
        "packages/organizations/deepseek/tests/e2e/test_file_contract.py::test_deepseek_file_contract_rejects_documents_before_transport",
    ),
    (
        ("pdf_bytes", "rejected_before_transport", "qwen"),
        "packages/organizations/qwen/tests/e2e/test_document_input.py::test_qwen_document_parts_are_rejected_before_messages_transport",
    ),
    (("reasoning_control", "none_falls_back_to_minimum", None), _REASONING_CHAT),
    (
        ("reasoning_control", "minimum_fallback_with_effort_alias", None),
        _REASONING_CHAT,
    ),
    (("reasoning_control", "ignored", None), _REASONING_CHAT),
    (
        ("structured_output_schema", "rejected_before_transport", "zai"),
        "packages/organizations/zai/tests/e2e/test_capability_boundaries.py::test_zai_rejects_structured_output_before_provider_transport",
    ),
)

ALWAYS_ON_SCENARIOS = _routes(
    (
        "facade_discovery",
        "tests/unit/test_organization_profile_compatibility.py::test_profile_parsing_preserves_registry_resolution_plugin_discovery_and_facade",
    ),
    (
        "message_normalization",
        "tests/unit/models/messages/test_chat_message.py::test_messages_normalize_openai_style_content_list",
    ),
    (
        "response_normalization",
        "tests/unit/conformance/test_facade_contract.py::test_facade_chat_normalizes_messages_response_usage_and_pricing",
    ),
    (
        "transport_parity",
        "tests/e2e/test_sync_httpx.py::test_sync_httpx_chat_returns_contract_for_latest_provider_models",
    ),
    (
        "stream_cleanup",
        "tests/unit/adapters/test_async_lifecycle.py::test_async_stream_close_before_completion_skips_tool_and_done",
    ),
    (
        "tool_validation",
        "tests/unit/conformance/test_facade_contract.py::test_facade_preserves_tool_structured_output_file_and_reasoning_contracts",
    ),
    (
        "schema_validation",
        "tests/unit/conformance/test_facade_contract.py::test_facade_rejects_nonportable_core_schema_before_sending_request",
    ),
    ("error_normalization", "tests/e2e/test_async.py::test_async_errors_are_normalized"),
    (
        "registry_exactness",
        "tests/unit/llm_registry/test_model_profile_inventory.py::test_every_first_party_model_has_a_valid_explicit_exception_profile",
    ),
    (
        "request_rule_fidelity",
        "tests/unit/adapters/test_request_rule_conformance.py::test_adapter_accepts_each_registered_tool_choice_mode",
    ),
    (
        "pricing_correctness",
        "tests/unit/adapters/test_pricing_lifecycle.py::test_tiered_pricing_matches_sync_chat_and_stream",
    ),
    (
        "missing_usage_honesty",
        "tests/unit/adapters/test_pricing_lifecycle.py::test_multi_tier_chat_without_provider_usage_leaves_costs_unset",
    ),
)

_CAPABILITY_SCOPES = {item.id: item.scope for item in CAPABILITY_CATALOGUE}
if any(_CAPABILITY_SCOPES.get(key) != "model-dependent" for key in BASELINE_SCENARIOS):
    raise ValueError("baseline routes must name model-dependent capabilities")
if any(_CAPABILITY_SCOPES.get(key) != "always-on" for key in ALWAYS_ON_SCENARIOS):
    raise ValueError("unconditional routes must name always-on capabilities")
if any(key[0] not in BASELINE_SCENARIOS or key[1] == "pass" for key in EXCEPTION_SCENARIOS):
    raise ValueError("exception routes must replace a baseline with a non-pass behavior")

_E2E_CAPABILITY_IDS = BASELINE_SCENARIOS.keys() | ALWAYS_ON_SCENARIOS.keys()
E2E_SCENARIO_CAPABILITIES = tuple(
    capability
    for capability in CAPABILITY_CATALOGUE
    if capability.id in _E2E_CAPABILITY_IDS
)
ALL_SCENARIO_NODE_IDS = frozenset(
    (*BASELINE_SCENARIOS.values(), *EXCEPTION_SCENARIOS.values(), *ALWAYS_ON_SCENARIOS.values())
)


def first_party_model_profiles() -> tuple[tuple[str, ModelSpec], ...]:
    """Load the exception profiles used by deterministic routing tests."""
    root = Path(__file__).resolve().parents[1]
    paths = sorted(
        (root / "src/llm_api_adapter/llm_registry/organizations").glob("*.json")
    ) + sorted(
        (root / "packages/organizations").glob(
            "*/src/*/registry/organizations/*.json"
        )
    )
    profiles = []
    for path in paths:
        catalogue = json.loads(path.read_text(encoding="utf-8"))
        for model_name, model_data in catalogue["models"].items():
            selector_data = {
                "limits": {"context_window_tokens": 1, "max_output_tokens": 1},
                "pricing_tiers": [
                    {"up_to_prompt_tokens": None, "input_per_1m": 1, "output_per_1m": 1}
                ],
                "capability_exceptions": model_data["capability_exceptions"],
            }
            profiles.append(
                (
                    path.stem,
                    ModelSpec.from_dict(
                        model_name,
                        selector_data,
                        currency=catalogue.get("currency", "USD"),
                    ),
                )
            )
    return tuple(profiles)


__all__ = [
    "ALL_SCENARIO_NODE_IDS",
    "ALWAYS_ON_SCENARIOS",
    "BASELINE_SCENARIOS",
    "E2E_SCENARIO_CAPABILITIES",
    "EXCEPTION_SCENARIOS",
    "first_party_model_profiles",
]
