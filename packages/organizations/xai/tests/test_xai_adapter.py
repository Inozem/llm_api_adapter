"""Deterministic contract tests for xAI's initial Responses API adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CORE_SOURCE = PACKAGE_ROOT.parents[2] / "src"
PACKAGE_SOURCE = PACKAGE_ROOT / "src"
for source in (str(PACKAGE_SOURCE), str(CORE_SOURCE)):
    if source not in sys.path:
        sys.path.insert(0, source)

import llm_api_adapter.adapters.base_adapter as base_adapter_module
import llm_api_adapter.universal_adapter as universal_module
from llm_api_adapter.errors.llm_api_error import (
    LLMAPIAuthorizationError,
    LLMAPIClientError,
    LLMAPIRateLimitError,
    LLMAPIServerError,
    LLMAPITimeoutError,
)
from llm_api_adapter.llm_registry.llm_registry import OrganizationSpec, RegistrySpec
from llm_api_adapter.llms.transports import (
    JSONResponse,
    MultipartForm,
    SSEEvent,
    SyncTransport,
    TransportRequest,
)
from llm_api_adapter.models.messages.chat_message import Prompt, UserMessage
from llm_api_adapter.service_provider_registry import ServiceProviderRegistry
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter

from llm_api_adapter_xai.adapter import XAIAdapter
from llm_api_adapter_xai.clients.sync_client import XAIResponsesSyncClient
from llm_api_adapter_xai.plugin import PLUGIN, register
from llm_api_adapter_xai.registry import MODEL_METADATA


@dataclass
class FakeSyncTransport(SyncTransport):
    payload: Any
    requests: list[TransportRequest] = field(default_factory=list)

    def post_json(
        self,
        request: TransportRequest,
        *,
        http_error_handler=None,
    ) -> JSONResponse:
        self.requests.append(request)
        return JSONResponse(self.payload)

    def post_multipart(
        self,
        request: TransportRequest,
        form: MultipartForm,
        *,
        http_error_handler=None,
    ) -> JSONResponse:
        raise AssertionError("xAI text chat must not upload multipart data")

    def post_sse(
        self,
        request: TransportRequest,
        *,
        http_error_handler=None,
        stream_error_handler=None,
    ) -> Iterator[SSEEvent]:
        raise AssertionError("xAI streaming is not part of this implementation")
        yield SSEEvent(event=None)


def _response(*, model: str) -> dict[str, Any]:
    return {
        "object": "response",
        "id": f"resp-{model}",
        "model": model,
        "created_at": 1774274151,
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "Hello from xAI."}],
            },
        ],
        "usage": {
            "input_tokens": 20,
            "output_tokens": 5,
            "total_tokens": 25,
        },
    }


@pytest.fixture
def xai_runtime(monkeypatch):
    model_registry = RegistrySpec()
    assert PLUGIN.model_metadata is not None
    assert model_registry.register_organization_metadata(PLUGIN.model_metadata) is True

    service_provider_registry = ServiceProviderRegistry()
    PLUGIN.register(service_provider_registry)
    monkeypatch.setattr(universal_module, "LLM_REGISTRY", model_registry)
    monkeypatch.setattr(
        universal_module,
        "SERVICE_PROVIDER_REGISTRY",
        service_provider_registry,
    )
    monkeypatch.setattr(base_adapter_module, "LLM_REGISTRY", model_registry)
    return model_registry


@pytest.mark.unit
def test_plugin_registers_the_xai_service_provider():
    service_providers = ServiceProviderRegistry()

    register(service_providers)

    assert PLUGIN.model_metadata is MODEL_METADATA
    assert service_providers.get("xai") is XAIAdapter


@pytest.mark.unit
def test_model_metadata_validates_all_fixed_model_ids():
    organization = OrganizationSpec.from_dict("xai", MODEL_METADATA.organization_data)

    assert set(organization.models) == {
        "grok-4.3",
        "grok-4.5",
        "grok-4.6",
        "grok-build-0.1",
    }
    pricing = organization.models["grok-4.6"].pricing_tiers
    assert pricing.tier_for_prompt_tokens(199999).in_per_token == pytest.approx(
        0.000002
    )
    assert pricing.tier_for_prompt_tokens(200000).out_per_token == pytest.approx(
        0.000012
    )
    assert organization.models["grok-4.3"].reasoning_capability.allowed_values == (
        "none",
        "low",
        "medium",
        "high",
    )
    assert organization.models["grok-build-0.1"].reasoning_capability is None


@pytest.mark.unit
@pytest.mark.parametrize(
    ("model", "expected_total_cost"),
    [
        ("grok-4.3", 0.0000375),
        ("grok-4.5", 0.00007),
        ("grok-4.6", 0.00007),
        ("grok-build-0.1", 0.00003),
    ],
)
def test_universal_chat_maps_text_to_responses_api_and_normalizes_output(
    xai_runtime,
    model: str,
    expected_total_cost: float,
):
    adapter = UniversalLLMAPIAdapter(
        organization="xai",
        api_key="test-key",
        model=model,
    )
    assert isinstance(adapter.adapter, XAIAdapter)
    transport = FakeSyncTransport(_response(model=model))
    adapter.adapter._client._sync_transport = transport

    response = adapter.chat(
        messages=[Prompt("Be concise."), UserMessage("Hello")],
        max_tokens=12,
        temperature=0.5,
        top_p=0.8,
        timeout_s=3.0,
    )

    assert transport.requests == [
        TransportRequest(
            url="https://api.x.ai/v1/responses",
            headers={
                "Authorization": "Bearer test-key",
                "Content-Type": "application/json",
            },
            payload={
                "model": model,
                "input": [{"role": "user", "content": "Hello"}],
                "max_output_tokens": 12,
                "temperature": 0.5,
                "top_p": 0.8,
                "instructions": "Be concise.",
            },
            timeout=3.0,
        ),
    ]
    assert response.content == "Hello from xAI."
    assert response.response_id == f"resp-{model}"
    assert response.usage is not None
    assert response.usage.total_tokens == 25
    assert response.cost_total == pytest.approx(expected_total_cost)


@pytest.mark.unit
def test_chat_rejects_options_reserved_for_later_xai_capabilities(xai_runtime):
    adapter = XAIAdapter(api_key="test-key", model="grok-4.6")

    with pytest.raises(ValueError, match="previous_response"):
        adapter.chat(messages=[UserMessage("Hello")], previous_response=object())

    with pytest.raises(ValueError, match="reasoning_level"):
        adapter.chat(messages=[UserMessage("Hello")], reasoning_level="high")


@pytest.mark.unit
def test_chat_rejects_malformed_responses_payload(xai_runtime):
    adapter = XAIAdapter(api_key="test-key", model="grok-4.6")
    adapter._client._sync_transport = FakeSyncTransport({"object": "response"})

    with pytest.raises(LLMAPIClientError, match="response.output"):
        adapter.chat(messages=[UserMessage("Hello")])


@pytest.mark.unit
@pytest.mark.parametrize(
    ("status_code", "error_type", "error_class"),
    [
        (401, None, LLMAPIAuthorizationError),
        (429, None, LLMAPIRateLimitError),
        (504, None, LLMAPITimeoutError),
        (500, None, LLMAPIServerError),
        (400, "validation_error", LLMAPIClientError),
    ],
)
def test_client_maps_xai_errors(
    status_code: int,
    error_type: str | None,
    error_class: type[Exception],
):
    with pytest.raises(error_class):
        XAIResponsesSyncClient._raise_mapped_error(
            status_code=status_code,
            error_type=error_type,
            detail="test failure",
        )
