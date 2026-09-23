"""Closed registry schema for Z.ai's documented request restrictions."""

from __future__ import annotations

from llm_api_adapter.llm_registry.request_rules import RequestRuleRegistry


class ZaiRequestRuleRegistry(RequestRuleRegistry):
    """Allow only the Z.ai request constraints represented in package metadata."""

    organization_name = "zai"
    supported_handlers = frozenset({RequestRuleRegistry.RESTRICT_TOOL_CHOICE})
    supported_tool_choice_modes = frozenset({"auto"})


ZAI_REQUEST_RULE_REGISTRY = ZaiRequestRuleRegistry()


__all__ = ["ZAI_REQUEST_RULE_REGISTRY", "ZaiRequestRuleRegistry"]
