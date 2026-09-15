"""Closed, package-local request-rule schema for Kimi Chat Completions."""

from __future__ import annotations

from llm_api_adapter.llm_registry.request_rules import (
    RequestRuleRegistry,
    SamplingRequestRuleRegistry,
)


class KimiRequestRuleRegistry(SamplingRequestRuleRegistry):
    """Allow only documented Kimi request normalizations from registry data."""

    organization_name = "kimi"
    supported_handlers = SamplingRequestRuleRegistry.supported_handlers | frozenset(
        {
            RequestRuleRegistry.RENAME_PARAMETER,
            RequestRuleRegistry.RESTRICT_TOOL_CHOICE,
        },
    )
    supported_parameter_renames = frozenset(
        {("max_tokens", "max_completion_tokens")},
    )
    droppable_parameter_defaults = {
        "temperature": 1.0,
        "top_p": 1.0,
    }
    supported_tool_choice_modes = frozenset({"auto", "none", "any", "tool"})


KIMI_REQUEST_RULE_REGISTRY = KimiRequestRuleRegistry()


__all__ = ["KIMI_REQUEST_RULE_REGISTRY", "KimiRequestRuleRegistry"]
