from dataclasses import dataclass, field
import logging

# Kept as a module attribute for existing white-box patch points.
import requests

from ...llm_registry.llm_registry import CategoricalReasoningCapability
from ..request_rules import apply_model_request_rules, model_api_variant, model_spec_for
from ..transports import SyncTransport, create_sync_transport
from ..openai_compatible.sync_client import (
    OpenAICompatibleSyncClient,
    _omit_numeric_sampling_defaults,
    _prepare_compatible_chat_payload,
    _prepare_compatible_responses_payload,
)

logger = logging.getLogger(__name__)


@dataclass
class OpenAISyncClient(OpenAICompatibleSyncClient):
    endpoint: str = "https://api.openai.com/v1"
    _official_behavior_enabled: bool = field(default=True, init=False, repr=False, compare=False)

    def _create_sync_transport(self, transport: str) -> SyncTransport:
        """Create a sync transport through the established native patch point."""
        return create_sync_transport(transport)

    def __repr__(self) -> str:
        masked = f"{self.api_key[:8]}...{self.api_key[-4:]}" if len(self.api_key) > 12 else "***"
        result = f"OpenAISyncClient(api_key='{masked}', endpoint='{self.endpoint}')"
        return self._format_diagnostic(result)

    def _warn_http_parse_fallback(self, error: Exception) -> None:
        logger.warning("Failed to parse error response: %s", self._format_diagnostic(error))

    def _should_use_responses_api(self, model: str) -> bool:
        if not self._official_behavior_enabled:
            return self._api_variant_override == "responses"
        return model_api_variant("openai", model) == "responses"

    def _prepare_chat_payload_for_model(
        self, model: str, kwargs: dict, *, streaming: bool = False
    ) -> dict:
        if not self._official_behavior_enabled:
            return _prepare_compatible_chat_payload(model, kwargs, streaming=streaming)
        payload, _ = apply_model_request_rules("openai", model, {"model": model, **kwargs})
        return payload

    def _set_official_behavior_enabled(self, enabled: bool) -> None:
        self._official_behavior_enabled = enabled

    def _prepare_responses_payload_for_model(self, model: str, kwargs: dict) -> dict:
        if not self._official_behavior_enabled:
            return _prepare_compatible_responses_payload(model, kwargs)
        payload = {"model": model, **dict(kwargs)}
        if "messages" in payload:
            payload["input"] = payload.pop("messages")
        if "max_tokens" in payload:
            payload["max_output_tokens"] = payload.pop("max_tokens")
        reasoning_effort = payload.pop("reasoning_effort", None)
        capture_reasoning = payload.pop("capture_reasoning", False)
        reasoning: dict = {}
        if reasoning_effort is not None:
            fallback_effort = self._fallback_reasoning_effort_for_none(model)
            if reasoning_effort == "none" and fallback_effort is not None:
                reasoning_effort = fallback_effort
            reasoning["effort"] = reasoning_effort
        if capture_reasoning:
            reasoning["summary"] = "auto"
        if reasoning:
            payload["reasoning"] = reasoning
        payload, _ = apply_model_request_rules("openai", model, payload)
        return payload

    @staticmethod
    def _fallback_reasoning_effort_for_none(model: str) -> str | None:
        """Return the lowest native effort when a model cannot disable reasoning."""
        model_spec = model_spec_for("openai", model)
        capability = model_spec.reasoning_capability if model_spec else None
        if (
            isinstance(capability, CategoricalReasoningCapability)
            and "none" not in capability.allowed_values
        ):
            return capability.allowed_values[0]
        return None


__all__ = ["OpenAISyncClient"]
