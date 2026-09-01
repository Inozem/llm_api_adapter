"""Portable structured-output conformance through the public facade.

The matrix deliberately uses the same ``UniversalLLMAPIAdapter`` entry point
for built-in organizations and installed organization plugins.  HTTP/SSE
clients are replaced at their network boundary, so this suite records native
payloads without credentials or live API calls.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Final
from unittest.mock import AsyncMock, Mock, patch

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
for source in (
    REPOSITORY_ROOT / "src",
    REPOSITORY_ROOT / "packages" / "organizations" / "mistral" / "src",
    REPOSITORY_ROOT / "packages" / "organizations" / "xai" / "src",
):
    source_path = str(source)
    if source_path not in sys.path:
        sys.path.insert(0, source_path)


import llm_api_adapter.adapters.base_adapter as base_adapter_module
import llm_api_adapter.universal_adapter as universal_adapter_module
from llm_api_adapter.adapters.anthropic_adapter import AnthropicAdapter
from llm_api_adapter.adapters.google_adapter import GoogleAdapter
from llm_api_adapter.adapters.openai_adapter import OpenAIAdapter
from llm_api_adapter.errors.llm_api_error import JSONSchemaError
from llm_api_adapter.llm_registry.llm_registry import RegistrySpec
from llm_api_adapter.llms.anthropic.async_client import ClaudeAsyncClient
from llm_api_adapter.llms.anthropic.sync_client import ClaudeSyncClient
from llm_api_adapter.llms.google.async_client import GeminiAsyncClient
from llm_api_adapter.llms.google.sync_client import GeminiSyncClient
from llm_api_adapter.llms.openai.async_client import OpenAIAsyncClient
from llm_api_adapter.llms.openai.sync_client import OpenAISyncClient
from llm_api_adapter.llms.transports import SSEEvent
from llm_api_adapter.models.messages.chat_message import UserMessage
from llm_api_adapter.models.messages.file_parts import DocumentPart
from llm_api_adapter.service_provider_registry import ServiceProviderRegistry
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter
from llm_api_adapter_mistral.adapter import MistralAdapter
from llm_api_adapter_mistral.plugin import PLUGIN as MISTRAL_PLUGIN
from llm_api_adapter_xai.adapter import XAIAdapter
from llm_api_adapter_xai.plugin import PLUGIN as XAI_PLUGIN
from tests.fixtures.structured_output import (
    FLAT_OBJECT_SCHEMA,
    NESTED_PYDANTIC_RESPONSE_JSON,
    NestedPydanticResponse,
    PORTABLE_PROFILE_SCHEMAS,
)


@dataclass(frozen=True)
class _JSONResponse:
    data: dict[str, Any]

    def json(self) -> dict[str, Any]:
        return self.data


@dataclass(frozen=True)
class PortableProfileCase:
    organization: str
    model: str
    max_tokens: int | None
    sync_client_class: type[Any] | None
    async_client_class: type[Any] | None


def _openai_response(content: str) -> dict[str, Any]:
    return {
        "id": "resp_123",
        "model": "gpt-5-nano",
        "status": "completed",
        "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
        "output": [{
            "type": "message",
            "content": [{"type": "output_text", "text": content}],
        }],
    }


def _anthropic_response(content: str) -> dict[str, Any]:
    return {
        "id": "msg_123",
        "model": "claude-sonnet-4-5",
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": content}],
        "usage": {"input_tokens": 2, "output_tokens": 3},
    }


def _google_response(content: str) -> dict[str, Any]:
    return {
        "modelVersion": "gemini-2.5-flash",
        "candidates": [{
            "content": {"parts": [{"text": content}]},
            "finishReason": "STOP",
        }],
        "usageMetadata": {
            "promptTokenCount": 2,
            "candidatesTokenCount": 3,
            "totalTokenCount": 5,
        },
    }


def _mistral_response(content: str) -> dict[str, Any]:
    return {
        "model": "mistral-small-2603",
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
    }


def _xai_response(content: str) -> dict[str, Any]:
    return {
        "object": "response",
        "id": "resp_xai_123",
        "model": "grok-4.6",
        "created_at": 1_774_274_151,
        "status": "completed",
        "output": [{
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": content}],
        }],
        "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
    }


CASES: Final = (
    pytest.param(
        PortableProfileCase(
            "openai", "gpt-5-nano", 64, OpenAISyncClient, OpenAIAsyncClient,
        ),
        id="openai",
    ),
    pytest.param(
        PortableProfileCase(
            "anthropic", "claude-sonnet-4-5", 64,
            ClaudeSyncClient, ClaudeAsyncClient,
        ),
        id="anthropic",
    ),
    pytest.param(
        PortableProfileCase(
            "google", "gemini-2.5-flash", None,
            GeminiSyncClient, GeminiAsyncClient,
        ),
        id="google",
    ),
    pytest.param(
        PortableProfileCase("mistral", "mistral-small-2603", 64, None, None),
        id="mistral",
    ),
    pytest.param(
        PortableProfileCase("xai", "grok-4.6", 64, None, None),
        id="xai",
    ),
)


# This matrix is intentionally exact at organization, model, and endpoint
# granularity.  It is the only source of skips in the terminal-outcome tests:
# Mistral Chat Completions exposes no documented distinct refusal terminal
# signal, while every other case has a response shape that carries one.
_TERMINAL_OUTCOME_CAPABILITIES: Final = {
    ("openai", "gpt-5-nano", "responses"): frozenset({"refusal", "incomplete"}),
    ("anthropic", "claude-sonnet-4-5", "messages"): frozenset({"refusal", "incomplete"}),
    ("google", "gemini-2.5-flash", "generate_content"): frozenset({"refusal", "incomplete"}),
    ("mistral", "mistral-small-2603", "chat_completions"): frozenset({"incomplete"}),
    ("xai", "grok-4.6", "responses"): frozenset({"refusal", "incomplete"}),
}

_ENDPOINTS: Final = {
    "openai": "responses",
    "anthropic": "messages",
    "google": "generate_content",
    "mistral": "chat_completions",
    "xai": "responses",
}


@pytest.fixture
def installed_organization_plugins(monkeypatch):
    """Register the two installed organization plugins beside Core adapters."""
    registry = RegistrySpec()
    service_providers = ServiceProviderRegistry({
        AnthropicAdapter.company: AnthropicAdapter,
        OpenAIAdapter.company: OpenAIAdapter,
        GoogleAdapter.company: GoogleAdapter,
    })
    for plugin in (MISTRAL_PLUGIN, XAI_PLUGIN):
        assert plugin.model_metadata is not None
        assert registry.register_organization_metadata(plugin.model_metadata) is True
        plugin.register(service_providers)

    monkeypatch.setattr(base_adapter_module, "LLM_REGISTRY", registry)
    monkeypatch.setattr(universal_adapter_module, "LLM_REGISTRY", registry)
    monkeypatch.setattr(
        universal_adapter_module,
        "SERVICE_PROVIDER_REGISTRY",
        service_providers,
    )


def _facade(case: PortableProfileCase) -> UniversalLLMAPIAdapter:
    return UniversalLLMAPIAdapter(
        organization=case.organization,
        model=case.model,
        api_key="test-api-key",
    )


def _chat_kwargs(case: PortableProfileCase) -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": "Return JSON."}],
        "max_tokens": case.max_tokens,
    }


def _response(case: PortableProfileCase, content: str) -> dict[str, Any]:
    responses = {
        "openai": _openai_response,
        "anthropic": _anthropic_response,
        "google": _google_response,
        "mistral": _mistral_response,
        "xai": _xai_response,
    }
    return responses[case.organization](content)


def _sync_payload_mock(
    case: PortableProfileCase,
    facade: UniversalLLMAPIAdapter,
    response: dict[str, Any],
) -> tuple[Mock, Any]:
    """Install one sync network mock and return a payload reader."""
    if case.sync_client_class is not None:
        transport = Mock(return_value=_JSONResponse(response))
        return transport, patch.object(
            case.sync_client_class,
            "_send_request",
            new=transport,
        )
    if case.organization == "mistral":
        transport = Mock(return_value=response)
        return transport, patch.object(
            MistralAdapter,
            "_post_payload",
            new=transport,
        )
    transport = Mock(return_value=response)
    facade.adapter._client.create = transport
    return transport, None


def _sync_payload(case: PortableProfileCase, transport: Mock) -> dict[str, Any]:
    if case.sync_client_class is not None:
        return transport.call_args.args[1]
    if case.organization == "mistral":
        return transport.call_args.args[0]
    return {
        "model": transport.call_args.kwargs["model"],
        **transport.call_args.kwargs,
    }


def _async_payload_mock(
    case: PortableProfileCase,
    facade: UniversalLLMAPIAdapter,
    response: dict[str, Any],
) -> tuple[AsyncMock, Any]:
    if case.async_client_class is not None:
        transport = AsyncMock(return_value=response)
        return transport, patch.object(
            case.async_client_class,
            "_send_request",
            new=transport,
        )
    transport = AsyncMock(return_value=response)
    if case.organization == "mistral":
        facade.adapter._apost_payload = transport
    else:
        facade.adapter._async_client.create = transport
    return transport, None


def _async_payload(case: PortableProfileCase, transport: AsyncMock) -> dict[str, Any]:
    if case.async_client_class is not None:
        return transport.call_args.args[1]
    if case.organization == "mistral":
        return transport.call_args.args[0]
    return {
        "model": transport.call_args.kwargs["model"],
        **transport.call_args.kwargs,
    }


def _schema_from_payload(
    organization: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    if organization in {"openai", "xai"}:
        return payload["text"]["format"]["schema"]
    if organization == "anthropic":
        return payload["output_config"]["format"]["schema"]
    if organization == "google":
        return payload["generationConfig"]["responseJsonSchema"]
    if organization == "mistral":
        return payload["response_format"]["json_schema"]["schema"]
    raise AssertionError(f"No schema path for {organization!r}")


def _google_wire_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Google responseJsonSchema uses the portable JSON Schema vocabulary."""
    return deepcopy(schema)


def _assert_native_format(
    organization: str,
    payload: dict[str, Any],
    expected_schema: dict[str, Any],
) -> None:
    assert _schema_from_payload(organization, payload) == expected_schema
    if organization in {"openai", "xai"}:
        fmt = payload["text"]["format"]
        assert fmt["type"] == "json_schema"
        assert fmt["name"] == "response"
        assert fmt["strict"] is True
    elif organization == "anthropic":
        assert payload["output_config"]["format"]["type"] == "json_schema"
    elif organization == "google":
        assert payload["generationConfig"]["responseMimeType"] == "application/json"
    else:
        fmt = payload["response_format"]
        assert fmt["type"] == "json_schema"
        assert fmt["json_schema"]["name"] == "response"
        assert fmt["json_schema"]["strict"] is True


def _contains_reference(node: Any) -> bool:
    if isinstance(node, dict):
        return "$ref" in node or any(_contains_reference(value) for value in node.values())
    if isinstance(node, list):
        return any(_contains_reference(value) for value in node)
    return False


@pytest.mark.unit
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("fixture_name", sorted(PORTABLE_PROFILE_SCHEMAS))
def test_portable_profile_raw_schema_matrix_uses_exact_native_payloads(
    case,
    fixture_name,
    installed_organization_plugins,
):
    """Every portable fixture reaches every organization unchanged in meaning."""
    source_schema = deepcopy(PORTABLE_PROFILE_SCHEMAS[fixture_name])
    facade = _facade(case)
    transport, patcher = _sync_payload_mock(
        case,
        facade,
        _response(case, '{"answer": "ok"}'),
    )

    if patcher is None:
        response = facade.chat(**_chat_kwargs(case), json_schema=source_schema)
    else:
        with patcher:
            response = facade.chat(**_chat_kwargs(case), json_schema=source_schema)

    expected_schema = (
        _google_wire_schema(source_schema)
        if case.organization == "google"
        else source_schema
    )
    assert response.parsed_json == {"answer": "ok"}
    _assert_native_format(
        case.organization,
        _sync_payload(case, transport),
        expected_schema,
    )


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("case", CASES)
async def test_portable_profile_response_model_has_sync_async_payload_parity(
    case,
    installed_organization_plugins,
):
    """Pydantic extraction and schema normalization agree in both API modes."""
    sync_facade = _facade(case)
    sync_transport, sync_patcher = _sync_payload_mock(
        case,
        sync_facade,
        _response(case, NESTED_PYDANTIC_RESPONSE_JSON),
    )
    if sync_patcher is None:
        sync_response = sync_facade.chat(
            **_chat_kwargs(case),
            response_model=NestedPydanticResponse,
        )
    else:
        with sync_patcher:
            sync_response = sync_facade.chat(
                **_chat_kwargs(case),
                response_model=NestedPydanticResponse,
            )

    async_facade = _facade(case)
    async_transport, async_patcher = _async_payload_mock(
        case,
        async_facade,
        _response(case, NESTED_PYDANTIC_RESPONSE_JSON),
    )
    if async_patcher is None:
        async_response = await async_facade.achat(
            **_chat_kwargs(case),
            response_model=NestedPydanticResponse,
        )
    else:
        with async_patcher:
            async_response = await async_facade.achat(
                **_chat_kwargs(case),
                response_model=NestedPydanticResponse,
            )

    expected_model = NestedPydanticResponse(contact={"name": "Ada"})
    assert sync_response.parsed_model == expected_model
    assert async_response.parsed_model == expected_model
    expected_schema = _schema_from_payload(
        case.organization,
        _sync_payload(case, sync_transport),
    )
    assert _schema_from_payload(
        case.organization,
        _async_payload(case, async_transport),
    ) == expected_schema
    assert "$defs" not in expected_schema
    assert not _contains_reference(expected_schema)


def _stream_events(case: PortableProfileCase, content: str) -> list[SSEEvent]:
    if case.organization == "openai":
        return [
            SSEEvent(event="response.output_text.delta", data={"delta": content}),
            SSEEvent(
                event="response.completed",
                data={"response": _openai_response(content)},
            ),
        ]
    if case.organization == "anthropic":
        return [
            SSEEvent(
                event="message_start",
                data={"message": {
                    "id": "msg_123",
                    "model": case.model,
                    "content": [],
                    "usage": {"input_tokens": 2, "output_tokens": 0},
                }},
            ),
            SSEEvent(
                event="content_block_start",
                data={"index": 0, "content_block": {"type": "text", "text": ""}},
            ),
            SSEEvent(
                event="content_block_delta",
                data={"index": 0, "delta": {"type": "text_delta", "text": content}},
            ),
            SSEEvent(
                event="message_delta",
                data={"delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 3}},
            ),
            SSEEvent(event="message_stop", data={}),
        ]
    if case.organization == "google":
        return [SSEEvent(
            event=None,
            data=_google_response(content),
        )]
    if case.organization == "mistral":
        return [SSEEvent(
            event=None,
            data={"choices": [{
                "index": 0,
                "delta": {"content": content},
                "finish_reason": "stop",
            }]},
        )]
    return [
        SSEEvent(event="response.output_text.delta", data={"delta": content}),
        SSEEvent(
            event="response.completed",
            data={"response": _xai_response(content)},
        ),
    ]


def _install_sync_stream(
    case: PortableProfileCase,
    facade: UniversalLLMAPIAdapter,
    events: list[SSEEvent],
) -> Any:
    stream = Mock(return_value=iter(events))
    if case.sync_client_class is not None:
        return patch.object(case.sync_client_class, "stream", new=stream)
    if case.organization == "mistral":
        facade.adapter._stream_payload = stream
    else:
        facade.adapter._client.stream = stream
    return None


async def _as_async_iter(events: list[SSEEvent]) -> AsyncIterator[SSEEvent]:
    for event in events:
        yield event


def _mistral_metered_messages() -> list[UserMessage]:
    return [
        UserMessage(
            "Summarize this document.",
            files=[DocumentPart(url="https://example.com/document.pdf")],
        )
    ]


def _mistral_ocr_response(pages_processed: int | None) -> dict[str, Any]:
    response: dict[str, Any] = {
        "model": "mistral-ocr-4-1",
        "pages": [{"markdown": "# Document"}],
    }
    if pages_processed is not None:
        response["usage_info"] = {"pages_processed": pages_processed}
    return response


def _mistral_metered_chat_response() -> dict[str, Any]:
    return {
        "model": "mistral-small-2603",
        "choices": [{"message": {"content": "Summary."}}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
    }


def _mistral_metered_stream_events() -> list[SSEEvent]:
    return [
        SSEEvent(
            event=None,
            data={
                "model": "mistral-small-2603",
                "choices": [{
                    "index": 0,
                    "delta": {"content": "Summary."},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 3,
                    "total_tokens": 5,
                },
            },
        )
    ]


def _assert_mistral_metered_ocr_cost(
    response: Any,
    *,
    quantity: float,
    total_available: bool = True,
) -> None:
    assert response.currency == "USD"
    assert response.cost_input == pytest.approx(0.0000003)
    assert response.cost_output == pytest.approx(0.0000018)
    assert response.cost_breakdown is not None
    assert [
        (
            item.operation,
            item.model,
            item.unit,
            item.quantity,
            item.rate,
            item.currency,
            item.cost,
        )
        for item in response.cost_breakdown
    ] == [
        ("ocr", "mistral-ocr-4-1", "page", quantity, 0.004, "USD", quantity * 0.004)
    ]
    if total_available:
        assert response.cost_total == pytest.approx(
            0.0000003 + 0.0000018 + quantity * 0.004
        )
    else:
        assert response.cost_total is None


def _install_async_stream(
    case: PortableProfileCase,
    facade: UniversalLLMAPIAdapter,
    events: list[SSEEvent],
) -> Any:
    stream = Mock(return_value=_as_async_iter(events))
    if case.async_client_class is not None:
        return patch.object(case.async_client_class, "stream", new=stream)
    if case.organization == "mistral":
        facade.adapter._astream_payload = stream
    else:
        facade.adapter._async_client.stream = stream
    return None


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("case", CASES)
async def test_portable_profile_streams_finalize_through_on_done_in_both_modes(
    case,
    installed_organization_plugins,
):
    content = '{"answer": "ok"}'
    sync_done: list[Any] = []
    sync_facade = _facade(case)
    sync_patcher = _install_sync_stream(case, sync_facade, _stream_events(case, content))
    if sync_patcher is None:
        sync_output = list(sync_facade.stream_chat(
            **_chat_kwargs(case),
            json_schema=FLAT_OBJECT_SCHEMA,
            on_done=sync_done.append,
        ))
    else:
        with sync_patcher:
            sync_output = list(sync_facade.stream_chat(
                **_chat_kwargs(case),
                json_schema=FLAT_OBJECT_SCHEMA,
                on_done=sync_done.append,
            ))

    async_done: list[Any] = []
    async_facade = _facade(case)
    async_patcher = _install_async_stream(
        case,
        async_facade,
        _stream_events(case, content),
    )
    if async_patcher is None:
        async_output = [
            text async for text in async_facade.astream_chat(
                **_chat_kwargs(case),
                json_schema=FLAT_OBJECT_SCHEMA,
                on_done=async_done.append,
            )
        ]
    else:
        with async_patcher:
            async_output = [
                text async for text in async_facade.astream_chat(
                    **_chat_kwargs(case),
                    json_schema=FLAT_OBJECT_SCHEMA,
                    on_done=async_done.append,
                )
            ]

    assert "".join(sync_output) == content
    assert "".join(async_output) == content
    assert sync_done[0].parsed_json == {"answer": "ok"}
    assert async_done[0].parsed_json == {"answer": "ok"}


@pytest.mark.asyncio
@pytest.mark.unit
async def test_mistral_metered_ocr_pricing_matches_across_facade_modes(
    installed_organization_plugins,
):
    sync_facade = UniversalLLMAPIAdapter(
        organization="mistral",
        model="mistral-small-2603",
        api_key="test-api-key",
    )
    sync_facade.adapter._post_ocr_payload = Mock(
        return_value=_mistral_ocr_response(2)
    )
    sync_facade.adapter._post_payload = Mock(
        return_value=_mistral_metered_chat_response()
    )
    sync_response = sync_facade.chat(_mistral_metered_messages())

    async_facade = UniversalLLMAPIAdapter(
        organization="mistral",
        model="mistral-small-2603",
        api_key="test-api-key",
    )
    async_facade.adapter._apost_ocr_payload = AsyncMock(
        return_value=_mistral_ocr_response(2)
    )
    async_facade.adapter._apost_payload = AsyncMock(
        return_value=_mistral_metered_chat_response()
    )
    async_response = await async_facade.achat(_mistral_metered_messages())

    sync_stream_facade = UniversalLLMAPIAdapter(
        organization="mistral",
        model="mistral-small-2603",
        api_key="test-api-key",
    )
    sync_stream_facade.adapter._post_ocr_payload = Mock(
        return_value=_mistral_ocr_response(2)
    )
    sync_stream_facade.adapter._stream_payload = Mock(
        return_value=iter(_mistral_metered_stream_events())
    )
    sync_completed: list[Any] = []
    assert list(
        sync_stream_facade.stream_chat(
            _mistral_metered_messages(),
            on_done=sync_completed.append,
        )
    ) == ["Summary."]

    async_stream_facade = UniversalLLMAPIAdapter(
        organization="mistral",
        model="mistral-small-2603",
        api_key="test-api-key",
    )
    async_stream_facade.adapter._apost_ocr_payload = AsyncMock(
        return_value=_mistral_ocr_response(2)
    )
    async_stream_facade.adapter._astream_payload = Mock(
        return_value=_as_async_iter(_mistral_metered_stream_events())
    )
    async_completed: list[Any] = []
    assert [
        text
        async for text in async_stream_facade.astream_chat(
            _mistral_metered_messages(),
            on_done=async_completed.append,
        )
    ] == ["Summary."]

    for response in (
        sync_response,
        async_response,
        sync_completed[0],
        async_completed[0],
    ):
        _assert_mistral_metered_ocr_cost(response, quantity=2.0)


@pytest.mark.unit
def test_mistral_zero_page_ocr_cost_is_a_known_zero_line_item(
    installed_organization_plugins,
):
    facade = UniversalLLMAPIAdapter(
        organization="mistral",
        model="mistral-small-2603",
        api_key="test-api-key",
    )
    facade.adapter._post_ocr_payload = Mock(
        return_value=_mistral_ocr_response(0)
    )
    facade.adapter._post_payload = Mock(
        return_value=_mistral_metered_chat_response()
    )

    response = facade.chat(_mistral_metered_messages())

    _assert_mistral_metered_ocr_cost(response, quantity=0.0)


@pytest.mark.unit
def test_mistral_incomplete_ocr_usage_preserves_known_cost_components(
    installed_organization_plugins,
):
    facade = UniversalLLMAPIAdapter(
        organization="mistral",
        model="mistral-small-2603",
        api_key="test-api-key",
    )
    facade.adapter._post_ocr_payload = Mock(
        side_effect=[
            _mistral_ocr_response(1),
            _mistral_ocr_response(None),
        ]
    )
    facade.adapter._post_payload = Mock(
        return_value=_mistral_metered_chat_response()
    )

    response = facade.chat(
        [
            UserMessage(
                "Summarize these documents.",
                files=[
                    DocumentPart(url="https://example.com/known.pdf"),
                    DocumentPart(url="https://example.com/unknown.pdf"),
                ],
            )
        ]
    )

    _assert_mistral_metered_ocr_cost(
        response,
        quantity=1.0,
        total_available=False,
    )


def _terminal_response(
    case: PortableProfileCase,
    outcome: str,
) -> dict[str, Any]:
    if outcome == "valid":
        return _response(case, '{"answer": "ok"}')
    if outcome == "invalid_json":
        return _response(case, '{"answer": ')
    if outcome == "pydantic_validation_error":
        return _response(case, '{"contact": {}}')

    if outcome == "refusal":
        if case.organization == "openai":
            return {
                "id": "resp_refusal",
                "model": case.model,
                "status": "completed",
                "output": [{"type": "message", "content": [{
                    "type": "refusal", "refusal": "I can't help with that.",
                }]}],
            }
        if case.organization == "anthropic":
            return {
                "id": "msg_refusal",
                "model": case.model,
                "stop_reason": "refusal",
                "stop_details": {"reason": "safety"},
                "content": [],
                "usage": {"input_tokens": 2, "output_tokens": 0},
            }
        if case.organization == "google":
            return {
                "modelVersion": case.model,
                "candidates": [{
                    "content": {"parts": []},
                    "finishReason": "SAFETY",
                    "finishMessage": "Blocked by safety policy.",
                }],
            }
        return {
            **_xai_response("unused"),
            "output": [{"type": "message", "content": [{
                "type": "refusal", "refusal": "I can't help with that.",
            }]}],
        }

    if case.organization == "openai":
        return {
            "id": "resp_incomplete",
            "model": case.model,
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [],
        }
    if case.organization == "anthropic":
        return {
            "id": "msg_incomplete",
            "model": case.model,
            "stop_reason": "max_tokens",
            "content": [{"type": "text", "text": '{"answer": '}],
            "usage": {"input_tokens": 2, "output_tokens": 3},
        }
    if case.organization == "google":
        return {
            "modelVersion": case.model,
            "candidates": [{
                "content": {"parts": [{"text": '{"answer": '}]},
                "finishReason": "MAX_TOKENS",
            }],
        }
    if case.organization == "mistral":
        return {
            **_mistral_response('{"answer": '),
            "choices": [{
                "message": {"content": '{"answer": '},
                "finish_reason": "length",
            }],
        }
    return {
        **_xai_response('{"answer": '),
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
    }


def _terminal_kwargs(case: PortableProfileCase, outcome: str) -> dict[str, Any]:
    kwargs = _chat_kwargs(case)
    if outcome == "pydantic_validation_error":
        kwargs["response_model"] = NestedPydanticResponse
    else:
        kwargs["json_schema"] = FLAT_OBJECT_SCHEMA
    return kwargs


def _assert_terminal_outcome(outcome: str, response: Any) -> None:
    if outcome == "valid":
        assert response.parsed_json == {"answer": "ok"}
        return
    if outcome == "refusal":
        assert response.refusal
        assert response.incomplete_reason is None
    elif outcome == "incomplete":
        assert response.incomplete_reason
        assert response.refusal is None
    else:
        raise AssertionError(f"Unexpected terminal outcome: {outcome}")
    assert response.parsed_json is None
    assert response.parsed_model is None


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize(
    "outcome",
    ("valid", "refusal", "incomplete", "invalid_json", "pydantic_validation_error"),
)
async def test_portable_profile_terminal_outcomes_keep_sync_async_contracts(
    case,
    outcome,
    installed_organization_plugins,
):
    capabilities = _TERMINAL_OUTCOME_CAPABILITIES[(
        case.organization,
        case.model,
        _ENDPOINTS[case.organization],
    )]
    if outcome in {"refusal", "incomplete"} and outcome not in capabilities:
        pytest.skip(
            f"{case.organization}/{case.model}/{_ENDPOINTS[case.organization]} "
            f"does not expose a distinct {outcome} terminal signal",
        )

    sync_facade = _facade(case)
    _, sync_patcher = _sync_payload_mock(
        case,
        sync_facade,
        _terminal_response(case, outcome),
    )
    kwargs = _terminal_kwargs(case, outcome)
    if outcome in {"invalid_json", "pydantic_validation_error"}:
        if sync_patcher is None:
            with pytest.raises(JSONSchemaError):
                sync_facade.chat(**kwargs)
        else:
            with sync_patcher, pytest.raises(JSONSchemaError):
                sync_facade.chat(**kwargs)
    elif sync_patcher is None:
        _assert_terminal_outcome(outcome, sync_facade.chat(**kwargs))
    else:
        with sync_patcher:
            _assert_terminal_outcome(outcome, sync_facade.chat(**kwargs))

    async_facade = _facade(case)
    _, async_patcher = _async_payload_mock(
        case,
        async_facade,
        _terminal_response(case, outcome),
    )
    if outcome in {"invalid_json", "pydantic_validation_error"}:
        if async_patcher is None:
            with pytest.raises(JSONSchemaError):
                await async_facade.achat(**kwargs)
        else:
            with async_patcher, pytest.raises(JSONSchemaError):
                await async_facade.achat(**kwargs)
        return
    if async_patcher is None:
        async_response = await async_facade.achat(**kwargs)
    else:
        with async_patcher:
            async_response = await async_facade.achat(**kwargs)
    _assert_terminal_outcome(outcome, async_response)
