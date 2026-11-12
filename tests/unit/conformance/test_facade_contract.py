"""Provider-neutral conformance checks through the public facade.

These tests deliberately exercise only ``UniversalLLMAPIAdapter``.  Provider
clients are mocked at their network boundary so the suite freezes the common
contract without credentials or live API calls.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import socket
import sys
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any, AsyncIterator, Callable
from unittest.mock import AsyncMock, Mock, patch

import pytest
import requests_mock
from pydantic import BaseModel, ConfigDict, field_validator

from src.llm_api_adapter.errors.llm_api_error import (
    JSONSchemaError,
    LLMAPIRateLimitError,
)
from src.llm_api_adapter.adapters.openai_compatible.common.adapter import _OpenAIExecutionMixin
from src.llm_api_adapter.llms.anthropic.async_client import ClaudeAsyncClient
from src.llm_api_adapter.llms.anthropic.sync_client import ClaudeSyncClient
from src.llm_api_adapter.llms.google.async_client import GeminiAsyncClient
from src.llm_api_adapter.llms.google.sync_client import GeminiSyncClient
from src.llm_api_adapter.llms.openai.async_client import OpenAIAsyncClient
from src.llm_api_adapter.llms.openai.sync_client import OpenAISyncClient
from src.llm_api_adapter.llms.openai_compatible.async_client import OpenAICompatibleAsyncClient
from src.llm_api_adapter.llms.openai_compatible.sync_client import OpenAICompatibleSyncClient
from src.llm_api_adapter.llms.streaming import SSEEvent
from src.llm_api_adapter.llms.transports import SSEFrameDecoder
from src.llm_api_adapter.models.messages.chat_message import AIMessage, ToolMessage, UserMessage
from src.llm_api_adapter.models.messages.file_parts import ImagePart
from src.llm_api_adapter.models.responses.chat_response import ChatResponse
from src.llm_api_adapter.models.responses.chat_response import Usage
from src.llm_api_adapter.models.tools import ToolCall, ToolSpec
from src.llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter
from src.llm_api_adapter.llm_registry.llm_registry import Pricing
from tests.fixtures.structured_output import (
    FLAT_OBJECT_SCHEMA,
    NESTED_PYDANTIC_RESPONSE_JSON,
    NestedPydanticResponse,
    PORTABLE_PROFILE_SCHEMAS,
    STRUCTURED_OUTPUT_OUTCOMES,
)
from tests.fixtures.openai_compatible import (
    FIXTURE_MODEL,
    chat_completion_response,
    event_lines,
    incomplete_response,
    incomplete_sse_lines,
    premature_eof_sse_lines,
    refusal_response,
    refusal_sse_lines,
    responses_incomplete_response,
    responses_refusal_response,
    responses_response,
    responses_text_response,
    responses_text_sse_lines,
    responses_tool_call_response,
    responses_tool_call_sse_lines,
    structured_result_sse_lines,
    text_response,
    text_sse_lines,
    tool_call_response,
    usage_payload,
)


@dataclass(frozen=True)
class _JSONResponse:
    data: dict[str, Any]

    def json(self) -> dict[str, Any]:
        return self.data


@dataclass(frozen=True)
class OrganizationConformanceCase:
    organization: str
    model: str
    max_tokens: int | None
    connection_kwargs: tuple[tuple[str, Any], ...]
    wire_format: str
    sync_client_class: type[Any]
    async_client_class: type[Any]
    message_key: str
    structured_schema_path: tuple[str, ...]
    response_factory: Callable[..., dict[str, Any]]
    stream_events_factory: Callable[[], list[SSEEvent]]
    expected_usage: Usage
    expected_currency: str | None
    expected_cost_total_available: bool
    expected_reasoning_texts: tuple[str, ...]
    expected_refusal: str
    expected_incomplete_reason: str
    expected_native_profile: bool = True


def _compatible_usage() -> dict[str, Any]:
    return usage_payload(
        2,
        3,
        5,
        prompt_tokens_details={"cached_tokens": 0},
    )


def _compatible_response(
    content: str = "ok",
    *,
    include_tool: bool = False,
    include_reasoning: bool = False,
) -> dict[str, Any]:
    # Chat Completions fixtures intentionally contain no vendor reasoning field.
    del include_reasoning
    if include_tool:
        return tool_call_response(
            name="get_weather",
            arguments_json='{"city": "Tel Aviv"}',
            call_id="call_123",
            usage=_compatible_usage(),
        )
    return text_response(content, usage=_compatible_usage())


def _decode_compatible_sse(lines: list[str]) -> list[SSEEvent]:
    decoder = SSEFrameDecoder()
    events = [event for line in lines if (event := decoder.feed(line)) is not None]
    final_event = decoder.finish()
    if final_event is not None:
        events.append(final_event)
    return events


def _compatible_stream_events() -> list[SSEEvent]:
    return _decode_compatible_sse(
        text_sse_lines(
            "Hello!",
            fragments=("He", "llo!"),
            usage=_compatible_usage(),
        ),
    )


def _compatible_responses_shared_response(
    content: str = "ok",
    *,
    include_tool: bool = False,
    include_reasoning: bool = False,
) -> dict[str, Any]:
    response = deepcopy(_openai_response(
        content,
        include_tool=include_tool,
        include_reasoning=include_reasoning,
    ))
    response["model"] = FIXTURE_MODEL
    return response


def _compatible_responses_shared_stream_events() -> list[SSEEvent]:
    events = deepcopy(_openai_stream_events())
    events[-1].data["response"]["model"] = FIXTURE_MODEL
    return events


def _openai_response(
    content: str = "ok",
    *,
    include_tool: bool = False,
    include_reasoning: bool = False,
) -> dict[str, Any]:
    output: list[dict[str, Any]] = []
    if include_reasoning:
        output.append({
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "Plan"}],
        })
    output.append({
        "type": "message",
        "content": [{"type": "output_text", "text": content}],
    })
    if include_tool:
        output.append({
            "type": "function_call",
            "call_id": "call_123",
            "name": "get_weather",
            "arguments": '{"city": "Tel Aviv"}',
        })
    return {
        "id": "resp_123",
        "model": "gpt-5-nano",
        "status": "completed",
        "usage": {
            "input_tokens": 2,
            "output_tokens": 3,
            "total_tokens": 5,
            "input_tokens_details": {"cached_tokens": 0},
        },
        "output": output,
    }


def _anthropic_response(
    content: str = "ok",
    *,
    include_tool: bool = False,
    include_reasoning: bool = False,
) -> dict[str, Any]:
    blocks: list[dict[str, Any]] = []
    if include_reasoning:
        blocks.append({"type": "thinking", "thinking": "Plan"})
    blocks.append({"type": "text", "text": content})
    if include_tool:
        blocks.append({
            "type": "tool_use",
            "id": "toolu_123",
            "name": "get_weather",
            "input": {"city": "Tel Aviv"},
        })
    return {
        "id": "msg_123",
        "model": "claude-sonnet-4-5",
        "stop_reason": "tool_use" if include_tool else "end_turn",
        "content": blocks,
        "usage": {"input_tokens": 2, "output_tokens": 3},
    }


def _google_response(
    content: str = "ok",
    *,
    include_tool: bool = False,
    include_reasoning: bool = False,
) -> dict[str, Any]:
    parts: list[dict[str, Any]] = []
    if include_reasoning:
        parts.append({"text": "Plan", "thought": True})
    parts.append({"text": content})
    if include_tool:
        parts.append({
            "functionCall": {
                "name": "get_weather",
                "args": {"city": "Tel Aviv"},
            },
        })
    return {
        "modelVersion": "gemini-3.8-flash",
        "candidates": [{
            "content": {"parts": parts},
            "finishReason": "STOP",
        }],
        "usageMetadata": {
            "promptTokenCount": 2,
            "candidatesTokenCount": 3,
            "thoughtsTokenCount": 0,
            "cachedContentTokenCount": 0,
            "totalTokenCount": 5,
        },
    }


def _openai_stream_events() -> list[SSEEvent]:
    return [
        SSEEvent(
            event="response.output_text.delta",
            data={"type": "response.output_text.delta", "delta": "He"},
        ),
        SSEEvent(
            event="response.output_text.delta",
            data={"type": "response.output_text.delta", "delta": "llo!"},
        ),
        SSEEvent(
            event="response.completed",
            data={
                "type": "response.completed",
                "response": _openai_response("Hello!"),
            },
        ),
    ]


def _anthropic_stream_events() -> list[SSEEvent]:
    return [
        SSEEvent(
            event="message_start",
            data={"type": "message_start", "message": {
                "id": "msg_123",
                "model": "claude-sonnet-4-5",
                "content": [],
                "usage": {"input_tokens": 2, "output_tokens": 0},
            }},
        ),
        SSEEvent(
            event="content_block_start",
            data={
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        SSEEvent(
            event="content_block_delta",
            data={
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "He"},
            },
        ),
        SSEEvent(
            event="content_block_delta",
            data={
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "llo!"},
            },
        ),
        SSEEvent(
            event="message_delta",
            data={
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 3},
            },
        ),
        SSEEvent(event="message_stop", data={"type": "message_stop"}),
    ]


def _google_stream_events() -> list[SSEEvent]:
    return [
        SSEEvent(
            event=None,
            data={"candidates": [{"content": {"parts": [{"text": "He"}]}}]},
        ),
        SSEEvent(
            event=None,
            data={
                "candidates": [{
                    "content": {"parts": [{"text": "llo!"}]},
                    "finishReason": "STOP",
                }],
                "usageMetadata": {
                    "promptTokenCount": 2,
                    "candidatesTokenCount": 3,
                    "thoughtsTokenCount": 0,
                    "cachedContentTokenCount": 0,
                    "totalTokenCount": 5,
                },
            },
        ),
    ]


CASES = (
    pytest.param(
        OrganizationConformanceCase(
            organization="openai",
            model="gpt-5-nano",
            max_tokens=64,
            connection_kwargs=(),
            wire_format="openai_responses",
            sync_client_class=OpenAISyncClient,
            async_client_class=OpenAIAsyncClient,
            message_key="input",
            structured_schema_path=("text", "format", "schema"),
            response_factory=_openai_response,
            stream_events_factory=_openai_stream_events,
            expected_usage=Usage(
                input_tokens=2,
                output_tokens=3,
                total_tokens=5,
                cached_tokens=0,
                cache_write_tokens=None,
            ),
            expected_currency="USD",
            expected_cost_total_available=True,
            expected_reasoning_texts=("Plan",),
            expected_refusal="I can't help with that.",
            expected_incomplete_reason="max_output_tokens",
        ),
        id="openai",
    ),
    pytest.param(
        OrganizationConformanceCase(
            organization="anthropic",
            model="claude-sonnet-4-5",
            max_tokens=64,
            connection_kwargs=(),
            wire_format="anthropic_messages",
            sync_client_class=ClaudeSyncClient,
            async_client_class=ClaudeAsyncClient,
            message_key="messages",
            structured_schema_path=("output_config", "format", "schema"),
            response_factory=_anthropic_response,
            stream_events_factory=_anthropic_stream_events,
            expected_usage=Usage(
                input_tokens=2,
                output_tokens=3,
                total_tokens=5,
                cached_tokens=None,
                cache_write_tokens=None,
            ),
            expected_currency="USD",
            expected_cost_total_available=True,
            expected_reasoning_texts=("Plan",),
            expected_refusal="safety",
            expected_incomplete_reason="max_tokens",
        ),
        id="anthropic",
    ),
    pytest.param(
        OrganizationConformanceCase(
            organization="google",
            model="gemini-3.8-flash",
            max_tokens=None,
            connection_kwargs=(),
            wire_format="google_generate_content",
            sync_client_class=GeminiSyncClient,
            async_client_class=GeminiAsyncClient,
            message_key="contents",
            structured_schema_path=("generationConfig", "responseJsonSchema"),
            response_factory=_google_response,
            stream_events_factory=_google_stream_events,
            expected_usage=Usage(
                input_tokens=2,
                output_tokens=3,
                total_tokens=5,
                cached_tokens=0,
                cache_write_tokens=None,
            ),
            expected_currency="USD",
            expected_cost_total_available=True,
            expected_reasoning_texts=("Plan",),
            expected_refusal="Blocked by safety policy.",
            expected_incomplete_reason="MAX_TOKENS",
        ),
        id="google",
    ),
    pytest.param(
        OrganizationConformanceCase(
            organization="openai_compatible",
            model=FIXTURE_MODEL,
            max_tokens=64,
            connection_kwargs=(("base_url", "https://example.invalid/proxy/v1/"),),
            wire_format="openai_chat_completions",
            sync_client_class=OpenAICompatibleSyncClient,
            async_client_class=OpenAICompatibleAsyncClient,
            message_key="messages",
            structured_schema_path=("response_format", "json_schema", "schema"),
            response_factory=_compatible_response,
            stream_events_factory=_compatible_stream_events,
            expected_usage=Usage(
                input_tokens=2,
                output_tokens=3,
                total_tokens=5,
                cached_tokens=0,
                cache_write_tokens=None,
            ),
            expected_currency=None,
            expected_cost_total_available=False,
            expected_reasoning_texts=(),
            expected_refusal="I can't help with that.",
            expected_incomplete_reason="length",
            expected_native_profile=False,
        ),
        id="openai_compatible",
    ),
    pytest.param(
        OrganizationConformanceCase(
            organization="openai_compatible",
            model=FIXTURE_MODEL,
            max_tokens=64,
            connection_kwargs=(
                ("base_url", "https://example.invalid/proxy/v1/"),
                ("api_variant", "responses"),
            ),
            wire_format="openai_responses",
            sync_client_class=OpenAICompatibleSyncClient,
            async_client_class=OpenAICompatibleAsyncClient,
            message_key="input",
            structured_schema_path=("text", "format", "schema"),
            response_factory=_compatible_responses_shared_response,
            stream_events_factory=_compatible_responses_shared_stream_events,
            expected_usage=Usage(
                input_tokens=2,
                output_tokens=3,
                total_tokens=5,
                cached_tokens=0,
                cache_write_tokens=None,
            ),
            expected_currency=None,
            expected_cost_total_available=False,
            expected_reasoning_texts=("Plan",),
            expected_refusal="Fixture refusal",
            expected_incomplete_reason="max_output_tokens",
            expected_native_profile=False,
        ),
        id="openai_compatible_responses",
    ),
)


class _StructuredAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str


def _facade(case: OrganizationConformanceCase) -> UniversalLLMAPIAdapter:
    return UniversalLLMAPIAdapter(
        organization=case.organization,
        model=case.model,
        api_key="test-api-key",
        **dict(case.connection_kwargs),
    )


def _chat_kwargs(case: OrganizationConformanceCase) -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": case.max_tokens,
    }


def _structured_schema_from_payload(
    case: OrganizationConformanceCase,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Return the schema at the exact path declared by this wire fixture."""
    node: Any = payload
    for component in case.structured_schema_path:
        assert isinstance(node, dict) and component in node, (
            f"Missing structured-schema path component {component!r} for "
            f"{case.wire_format!r}"
        )
        node = node[component]
    assert isinstance(node, dict), (
        f"Structured-schema path for {case.wire_format!r} did not resolve to an object"
    )
    return node


def _contains_reference(node: Any) -> bool:
    if isinstance(node, dict):
        return "$ref" in node or any(_contains_reference(value) for value in node.values())
    if isinstance(node, list):
        return any(_contains_reference(value) for value in node)
    return False


def _google_wire_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Google responseJsonSchema uses the portable JSON Schema vocabulary."""
    return deepcopy(schema)


def _assert_usage_and_pricing(
    response: Any,
    case: OrganizationConformanceCase,
) -> None:
    assert response.usage == case.expected_usage
    if case.expected_cost_total_available:
        assert response.currency == case.expected_currency
        assert response.cost_total is not None
    else:
        assert case.expected_currency is None
        assert response.currency is None
        assert response.cost_input is None
        assert response.cost_output is None
        assert response.cost_total is None
        assert response.cost_breakdown is None


def _assert_case_profile(
    facade: UniversalLLMAPIAdapter,
    case: OrganizationConformanceCase,
) -> None:
    assert (facade.adapter.model_spec is not None) is case.expected_native_profile


async def _as_async_iter(events: list[SSEEvent]) -> AsyncIterator[SSEEvent]:
    for event in events:
        yield event


class _CountedAsyncSource:
    """Provider-shaped source whose owner close can fail independently."""

    def __init__(self, events=(), *, next_error=None, close_error=None):
        self._events = iter(events)
        self.next_error = next_error
        self.close_error = close_error
        self.close_count = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.next_error is not None:
            error, self.next_error = self.next_error, None
            raise error
        try:
            return next(self._events)
        except StopIteration:
            raise StopAsyncIteration from None

    async def aclose(self):
        self.close_count += 1
        if self.close_error is not None:
            raise self.close_error


_TERMINAL_OUTCOME_FIXTURES = tuple(
    outcome
    for outcome in STRUCTURED_OUTPUT_OUTCOMES
    if outcome.name != "unsupported_schema"
)


def _terminal_outcome_response(
    case: OrganizationConformanceCase,
    outcome_name: str,
    content: str | None,
) -> dict[str, Any]:
    """Return the native terminal shape for one structured-output outcome."""
    if case.wire_format == "openai_responses" and case.organization == "openai_compatible":
        if outcome_name == "refusal":
            return responses_refusal_response(
                case.expected_refusal,
                model=case.model,
                usage={
                    "input_tokens": 2,
                    "output_tokens": 0,
                    "total_tokens": 2,
                    "input_tokens_details": {"cached_tokens": 0},
                },
            )
        if outcome_name == "incomplete_result":
            return responses_incomplete_response(
                model=case.model,
                usage={
                    "input_tokens": 2,
                    "output_tokens": 3,
                    "total_tokens": 5,
                    "input_tokens_details": {"cached_tokens": 0},
                },
            )
        return case.response_factory(content or "")

    if case.wire_format == "openai_responses":
        if outcome_name == "refusal":
            return {
                "id": "resp_refusal",
                "model": case.model,
                "status": "completed",
                "output": [{
                    "type": "message",
                    "content": [{
                        "type": "refusal",
                        "refusal": "I can't help with that.",
                    }],
                }],
            }
        if outcome_name == "incomplete_result":
            return {
                "id": "resp_incomplete",
                "model": case.model,
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "output": [],
            }
        return _openai_response(content or "")

    if case.wire_format == "anthropic_messages":
        if outcome_name == "refusal":
            return {
                "id": "msg_refusal",
                "model": case.model,
                "stop_reason": "refusal",
                "stop_details": {"reason": "safety"},
                "content": [],
                "usage": {"input_tokens": 2, "output_tokens": 0},
            }
        if outcome_name == "incomplete_result":
            return {
                "id": "msg_incomplete",
                "model": case.model,
                "stop_reason": "max_tokens",
                "content": [{"type": "text", "text": '{"answer": '}],
                "usage": {"input_tokens": 2, "output_tokens": 3},
            }
        return _anthropic_response(content or "")

    if case.wire_format == "google_generate_content":
        if outcome_name == "refusal":
            return {
                "modelVersion": case.model,
                "candidates": [{
                    "content": {"parts": []},
                    "finishReason": "SAFETY",
                    "finishMessage": "Blocked by safety policy.",
                }],
            }
        if outcome_name == "incomplete_result":
            return {
                "modelVersion": case.model,
                "candidates": [{
                    "content": {"parts": [{"text": '{"answer": '}]},
                    "finishReason": "MAX_TOKENS",
                }],
            }
        return _google_response(content or "")

    if case.wire_format == "openai_chat_completions":
        if outcome_name == "refusal":
            return refusal_response(
                case.expected_refusal,
                usage=_compatible_usage(),
                model=case.model,
            )
        if outcome_name == "incomplete_result":
            return incomplete_response(
                '{"answer": ',
                usage=_compatible_usage(),
                model=case.model,
            )
        return _compatible_response(content or "")

    raise AssertionError(f"No terminal-outcome fixture for {case.wire_format!r}")


def _terminal_outcome_stream_events(
    case: OrganizationConformanceCase,
    outcome_name: str,
    content: str | None,
) -> list[SSEEvent]:
    response = _terminal_outcome_response(case, outcome_name, content)
    if case.wire_format == "openai_responses":
        return [
            SSEEvent(
                event=(
                    "response.incomplete"
                    if response["status"] == "incomplete"
                    else "response.completed"
                ),
                data={"response": response},
            ),
        ]

    if case.wire_format == "anthropic_messages":
        events = [
            SSEEvent(
                event="message_start",
                data={
                    "message": {
                        "id": response["id"],
                        "model": response["model"],
                        "content": [],
                        "usage": {"input_tokens": 2, "output_tokens": 0},
                    },
                },
            ),
        ]
        if response["content"]:
            events.extend([
                SSEEvent(
                    event="content_block_start",
                    data={
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    },
                ),
                SSEEvent(
                    event="content_block_delta",
                    data={
                        "index": 0,
                        "delta": {
                            "type": "text_delta",
                            "text": response["content"][0]["text"],
                        },
                    },
                ),
                SSEEvent(event="content_block_stop", data={"index": 0}),
            ])
        delta: dict[str, Any] = {"stop_reason": response["stop_reason"]}
        if "stop_details" in response:
            delta["stop_details"] = response["stop_details"]
        events.extend([
            SSEEvent(
                event="message_delta",
                data={"delta": delta, "usage": {"output_tokens": 3}},
            ),
            SSEEvent(event="message_stop", data={}),
        ])
        return events

    if case.wire_format == "google_generate_content":
        return [SSEEvent(event=None, data=response)]

    if case.wire_format == "openai_chat_completions":
        if outcome_name == "refusal":
            return _decode_compatible_sse(refusal_sse_lines(case.expected_refusal))
        if outcome_name == "incomplete_result":
            return _decode_compatible_sse(incomplete_sse_lines('{"answer": '))
        return _decode_compatible_sse(
            structured_result_sse_lines(
                content or "",
                usage=_compatible_usage(),
            ),
        )

    raise AssertionError(f"No terminal stream fixture for {case.wire_format!r}")


def _structured_outcome_kwargs(
    case: OrganizationConformanceCase,
    outcome_name: str,
) -> dict[str, Any]:
    kwargs = _chat_kwargs(case)
    if outcome_name == "pydantic_validation_error":
        kwargs["response_model"] = NestedPydanticResponse
    elif outcome_name in {"refusal", "incomplete_result"}:
        kwargs["response_model"] = _StructuredAnswer
    else:
        kwargs["json_schema"] = FLAT_OBJECT_SCHEMA
    return kwargs


def _assert_structured_terminal_outcome(
    response: Any,
    case: OrganizationConformanceCase,
    outcome_name: str,
) -> None:
    if outcome_name == "valid_structured_result":
        assert response.parsed_json == {"answer": "ok"}
        return
    if outcome_name == "refusal":
        assert response.refusal == case.expected_refusal
        assert response.incomplete_reason is None
    elif outcome_name == "incomplete_result":
        assert response.incomplete_reason == case.expected_incomplete_reason
        assert response.refusal is None
    else:
        raise AssertionError(f"Unexpected terminal outcome {outcome_name!r}")
    assert response.parsed_json is None
    assert response.parsed_model is None


@pytest.mark.unit
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("outcome", _TERMINAL_OUTCOME_FIXTURES)
def test_facade_structured_terminal_outcomes_are_consistent_in_sync_chat(
    case,
    outcome,
):
    transport = Mock(
        return_value=_JSONResponse(
            _terminal_outcome_response(case, outcome.name, outcome.content),
        ),
    )
    kwargs = _structured_outcome_kwargs(case, outcome.name)

    with patch.object(case.sync_client_class, "_send_request", new=transport):
        if outcome.name in {"invalid_json", "pydantic_validation_error"}:
            with pytest.raises(JSONSchemaError):
                _facade(case).chat(**kwargs)
            return
        response = _facade(case).chat(**kwargs)

    _assert_structured_terminal_outcome(response, case, outcome.name)


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("outcome", _TERMINAL_OUTCOME_FIXTURES)
async def test_facade_structured_terminal_outcomes_are_consistent_in_async_chat(
    case,
    outcome,
):
    transport = AsyncMock(
        return_value=_terminal_outcome_response(case, outcome.name, outcome.content),
    )
    kwargs = _structured_outcome_kwargs(case, outcome.name)

    with patch.object(case.async_client_class, "_send_request", new=transport):
        if outcome.name in {"invalid_json", "pydantic_validation_error"}:
            with pytest.raises(JSONSchemaError):
                await _facade(case).achat(**kwargs)
            return
        response = await _facade(case).achat(**kwargs)

    _assert_structured_terminal_outcome(response, case, outcome.name)


@pytest.mark.unit
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("outcome", _TERMINAL_OUTCOME_FIXTURES)
def test_facade_structured_terminal_outcomes_are_consistent_in_sync_streams(
    case,
    outcome,
):
    events = _terminal_outcome_stream_events(case, outcome.name, outcome.content)
    completed = []
    kwargs = _structured_outcome_kwargs(case, outcome.name)
    kwargs["on_done"] = completed.append

    with patch.object(case.sync_client_class, "stream", return_value=iter(events)):
        if outcome.name in {"invalid_json", "pydantic_validation_error"}:
            with pytest.raises(JSONSchemaError):
                list(_facade(case).stream_chat(**kwargs))
            return
        list(_facade(case).stream_chat(**kwargs))

    assert len(completed) == 1
    _assert_structured_terminal_outcome(completed[0], case, outcome.name)


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("outcome", _TERMINAL_OUTCOME_FIXTURES)
async def test_facade_structured_terminal_outcomes_are_consistent_in_async_streams(
    case,
    outcome,
):
    events = _terminal_outcome_stream_events(case, outcome.name, outcome.content)
    completed = []
    kwargs = _structured_outcome_kwargs(case, outcome.name)
    kwargs["on_done"] = completed.append

    with patch.object(
        case.async_client_class,
        "stream",
        return_value=_as_async_iter(events),
    ):
        if outcome.name in {"invalid_json", "pydantic_validation_error"}:
            with pytest.raises(JSONSchemaError):
                async for _ in _facade(case).astream_chat(**kwargs):
                    pass
            return
        async for _ in _facade(case).astream_chat(**kwargs):
            pass

    assert len(completed) == 1
    _assert_structured_terminal_outcome(completed[0], case, outcome.name)


@pytest.mark.unit
@pytest.mark.parametrize("case", CASES)
def test_facade_chat_normalizes_messages_response_usage_and_pricing(case):
    transport = Mock(return_value=_JSONResponse(case.response_factory()))
    facade = _facade(case)

    with patch.object(case.sync_client_class, "_send_request", new=transport):
        response = facade.chat(**_chat_kwargs(case))

    payload = transport.call_args.args[1]
    assert payload["model"] == case.model
    assert payload[case.message_key]
    assert facade.adapter.model == case.model
    assert facade.adapter.api_key == "test-api-key"
    _assert_case_profile(facade, case)
    if case.organization == "openai_compatible":
        assert facade.base_url == "https://example.invalid/proxy/v1"
        assert facade.adapter.base_url == facade.base_url
    assert response.content == "ok"
    _assert_usage_and_pricing(response, case)


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("case", CASES)
async def test_facade_achat_matches_sync_response_contract(case):
    transport = AsyncMock(return_value=case.response_factory())

    with patch.object(case.async_client_class, "_send_request", new=transport):
        response = await _facade(case).achat(**_chat_kwargs(case))

    payload = transport.await_args.args[1]
    assert payload["model"] == case.model
    assert payload[case.message_key]
    assert response.content == "ok"
    _assert_usage_and_pricing(response, case)


@pytest.mark.unit
@pytest.mark.parametrize("case", CASES)
def test_facade_preserves_tool_structured_output_file_and_reasoning_contracts(case):
    tool = ToolSpec(name="get_weather", json_schema={"type": "object"})
    image = ImagePart(data=b"image-bytes", media_type="image/png")

    tool_transport = Mock(
        return_value=_JSONResponse(case.response_factory(include_tool=True)),
    )
    with patch.object(case.sync_client_class, "_send_request", new=tool_transport):
        tool_response = _facade(case).chat(
            messages=[UserMessage("What is the weather?", files=[image])],
            max_tokens=case.max_tokens,
            tools=[tool],
        )

    payload = tool_transport.call_args.args[1]
    assert "aW1hZ2UtYnl0ZXM=" in str(payload)
    assert tool_response.tool_calls[0].name == "get_weather"
    assert tool_response.tool_calls[0].arguments == {"city": "Tel Aviv"}

    structured_transport = Mock(
        return_value=_JSONResponse(case.response_factory('{"answer": "ok"}')),
    )
    with patch.object(case.sync_client_class, "_send_request", new=structured_transport):
        structured_response = _facade(case).chat(
            **_chat_kwargs(case),
            response_model=_StructuredAnswer,
        )

    assert structured_response.parsed_json == {"answer": "ok"}
    assert structured_response.parsed_model == _StructuredAnswer(answer="ok")

    reasoning_transport = Mock(
        return_value=_JSONResponse(case.response_factory(include_reasoning=True)),
    )
    with patch.object(case.sync_client_class, "_send_request", new=reasoning_transport):
        reasoning_response = _facade(case).chat(
            **_chat_kwargs(case),
            capture_reasoning=True,
        )

    assert [event.text for event in reasoning_response.reasoning_events] == (
        list(case.expected_reasoning_texts)
    )


@pytest.mark.unit
def test_portable_profile_fixture_covers_the_v092_acceptance_vocabulary():
    assert set(PORTABLE_PROFILE_SCHEMAS) == {
        "flat_object",
        "nullable_required_field",
        "enum",
        "array",
        "inline_nested_object",
    }
    assert {fixture.name for fixture in STRUCTURED_OUTPUT_OUTCOMES} == {
        "valid_structured_result",
        "refusal",
        "incomplete_result",
        "invalid_json",
        "pydantic_validation_error",
        "unsupported_schema",
    }


@pytest.mark.unit
@pytest.mark.parametrize("case", CASES)
def test_facade_preserves_flat_portable_schema_baseline(case):
    transport = Mock(
        return_value=_JSONResponse(case.response_factory('{"answer": "ok"}')),
    )

    with patch.object(case.sync_client_class, "_send_request", new=transport):
        response = _facade(case).chat(
            **_chat_kwargs(case),
            json_schema=FLAT_OBJECT_SCHEMA,
        )

    schema = _structured_schema_from_payload(case, transport.call_args.args[1])
    assert response.parsed_json == {"answer": "ok"}
    assert schema["properties"]["answer"]["type"] in {"string", "STRING"}
    assert schema["additionalProperties"] is False


@pytest.mark.unit
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("fixture_name", sorted(PORTABLE_PROFILE_SCHEMAS))
def test_core_portable_profile_has_exact_provider_payloads(case, fixture_name):
    source_schema = PORTABLE_PROFILE_SCHEMAS[fixture_name]
    transport = Mock(
        return_value=_JSONResponse(case.response_factory('{"answer": "ok"}')),
    )

    with patch.object(case.sync_client_class, "_send_request", new=transport):
        _facade(case).chat(
            **_chat_kwargs(case),
            json_schema=source_schema,
        )

    schema = _structured_schema_from_payload(case, transport.call_args.args[1])
    expected = (
        _google_wire_schema(source_schema)
        if case.wire_format == "google_generate_content"
        else source_schema
    )
    assert schema == expected


@pytest.mark.unit
@pytest.mark.parametrize("case", CASES)
def test_facade_rejects_nonportable_core_schema_before_sending_request(case):
    transport = Mock()
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": [],
        "additionalProperties": False,
    }

    with (
        patch.object(case.sync_client_class, "_send_request", new=transport),
        pytest.raises(JSONSchemaError, match=r"#/required"),
    ):
        _facade(case).chat(**_chat_kwargs(case), json_schema=schema)

    transport.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize("case", CASES)
def test_nested_pydantic_schema_is_portable_across_builtin_adapters(case):
    transport = Mock(
        return_value=_JSONResponse(case.response_factory(NESTED_PYDANTIC_RESPONSE_JSON)),
    )

    with patch.object(case.sync_client_class, "_send_request", new=transport):
        response = _facade(case).chat(
            **_chat_kwargs(case),
            response_model=NestedPydanticResponse,
        )

    schema = _structured_schema_from_payload(case, transport.call_args.args[1])
    assert response.parsed_model == NestedPydanticResponse(
        contact={"name": "Ada"},
    )
    assert "$defs" not in schema
    assert not _contains_reference(schema)
    assert schema["properties"]["contact"]["type"] in {"object", "OBJECT"}


@pytest.mark.unit
@pytest.mark.parametrize("case", CASES)
def test_facade_rejects_external_schema_references_before_sending_request(case):
    transport = Mock()
    schema = {
        "type": "object",
        "properties": {"contact": {"$ref": "https://example.com/contact.json"}},
    }

    with (
        patch.object(case.sync_client_class, "_send_request", new=transport),
        pytest.raises(JSONSchemaError, match=r"#?/properties/contact"),
    ):
        _facade(case).chat(**_chat_kwargs(case), json_schema=schema)

    transport.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize("case", CASES)
def test_facade_re_raises_normalized_provider_errors(case):
    transport = Mock(side_effect=LLMAPIRateLimitError())

    with (
        patch.object(case.sync_client_class, "_send_request", new=transport),
        pytest.raises(LLMAPIRateLimitError),
    ):
        _facade(case).chat(**_chat_kwargs(case))


@pytest.mark.unit
@pytest.mark.parametrize("case", CASES)
def test_facade_stream_finalizes_response_and_preserves_callback_order(case):
    order: list[tuple[str, str]] = []

    with patch.object(
        case.sync_client_class,
        "stream",
        return_value=iter(case.stream_events_factory()),
    ):
        output = []
        for text in _facade(case).stream_chat(
            **_chat_kwargs(case),
            buffer_chars=4,
            on_chunk=lambda chunk: order.append(("chunk", chunk.text)),
            on_delta=lambda text: order.append(("delta", text)),
            on_done=lambda response: order.append(("done", response.content)),
        ):
            output.append(text)
            order.append(("yield", text))

    assert output == ["Hell", "o!"]
    assert order == [
        ("chunk", "Hell"),
        ("delta", "Hell"),
        ("yield", "Hell"),
        ("chunk", "o!"),
        ("delta", "o!"),
        ("yield", "o!"),
        ("done", "Hello!"),
    ]


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("case", CASES)
async def test_facade_astream_matches_stream_contract(case):
    order: list[tuple[str, str]] = []

    async def on_chunk(chunk):
        order.append(("chunk", chunk.text))

    async def on_delta(text):
        order.append(("delta", text))

    async def on_done(response):
        order.append(("done", response.content))

    with patch.object(
        case.async_client_class,
        "stream",
        return_value=_as_async_iter(case.stream_events_factory()),
    ):
        output = []
        async for text in _facade(case).astream_chat(
            **_chat_kwargs(case),
            buffer_chars=4,
            on_chunk=on_chunk,
            on_delta=on_delta,
            on_done=on_done,
        ):
            output.append(text)
            order.append(("yield", text))

    assert output == ["Hell", "o!"]
    assert order[-1] == ("done", "Hello!")
    assert order[:6] == [
        ("chunk", "Hell"),
        ("delta", "Hell"),
        ("yield", "Hell"),
        ("chunk", "o!"),
        ("delta", "o!"),
        ("yield", "o!"),
    ]


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("case", CASES)
async def test_facade_streams_skip_finalization_when_cancelled(case):
    sync_closed: list[bool] = []

    def sync_events():
        try:
            yield from case.stream_events_factory()
        finally:
            sync_closed.append(True)

    sync_done: list[Any] = []
    with patch.object(case.sync_client_class, "stream", return_value=sync_events()):
        stream = _facade(case).stream_chat(
            **_chat_kwargs(case),
            on_done=sync_done.append,
        )
        next(stream)
        stream.close()

    assert sync_closed == [True]
    assert sync_done == []

    async def async_events():
        for event in case.stream_events_factory():
            yield event

    async_done: list[Any] = []
    with patch.object(case.async_client_class, "stream", return_value=async_events()):
        stream = _facade(case).astream_chat(
            **_chat_kwargs(case),
            on_done=async_done.append,
        )
        await anext(stream)
        await stream.aclose()

    assert async_done == []


@pytest.mark.unit
@pytest.mark.parametrize("case", CASES)
def test_facade_buffered_sync_early_close_releases_source_without_success(case):
    closed: list[bool] = []
    done: list[Any] = []

    def source():
        try:
            yield from case.stream_events_factory()
        finally:
            closed.append(True)

    raw_source = source()
    try:
        with patch.object(case.sync_client_class, "stream", return_value=raw_source):
            stream = _facade(case).stream_chat(
                **_chat_kwargs(case), buffer_chars=4, on_done=done.append
            )
            assert next(stream) == "Hell"
            stream.close()

        assert closed == [True]
        assert done == []
    finally:
        raw_source.close()


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("case", CASES)
async def test_facade_buffered_async_early_close_releases_source_without_success(case):
    closed: list[bool] = []
    done: list[Any] = []

    async def source():
        try:
            for event in case.stream_events_factory():
                yield event
        finally:
            closed.append(True)

    raw_source = source()
    try:
        with patch.object(case.async_client_class, "stream", return_value=raw_source):
            stream = _facade(case).astream_chat(
                **_chat_kwargs(case), buffer_chars=4, on_done=done.append
            )
            assert await anext(stream) == "Hell"
            await stream.aclose()

        assert closed == [True]
        assert done == []
    finally:
        await raw_source.aclose()


@pytest.mark.unit
@pytest.mark.parametrize("case", CASES)
def test_facade_sync_callback_failure_remains_caller_owned(case):
    failure = RuntimeError("caller callback failure", case.organization)
    closed: list[bool] = []
    done: list[Any] = []

    def source():
        try:
            yield from case.stream_events_factory()
        finally:
            closed.append(True)

    raw_source = source()

    def fail(_text):
        raise failure

    with patch.object(case.sync_client_class, "stream", return_value=raw_source):
        try:
            with pytest.raises(RuntimeError) as raised:
                next(
                    _facade(case).stream_chat(
                        **_chat_kwargs(case), on_delta=fail, on_done=done.append
                    )
                )

            assert raised.value is failure
            assert raised.value.args == ("caller callback failure", case.organization)
            assert closed == [True]
            assert done == []
        finally:
            raw_source.close()


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("case", CASES)
async def test_facade_async_awaitable_callbacks_are_awaited_in_order(case):
    order: list[str] = []

    def on_chunk(chunk):
        async def record_chunk():
            order.append(f"chunk:{chunk.text}")

        return record_chunk()

    def on_delta(text):
        async def record_delta():
            order.append(f"delta:{text}")

        return record_delta()

    def on_done(response):
        async def record_done():
            order.append(f"done:{response.content}")

        return record_done()

    with patch.object(
        case.async_client_class,
        "stream",
        return_value=_as_async_iter(case.stream_events_factory()),
    ):
        output = [
            text
            async for text in _facade(case).astream_chat(
                **_chat_kwargs(case),
                buffer_chars=4,
                on_chunk=on_chunk,
                on_delta=on_delta,
                on_done=on_done,
            )
        ]

    assert output == ["Hell", "o!"]
    assert order == [
        "chunk:Hell", "delta:Hell", "chunk:o!", "delta:o!", "done:Hello!"
    ]


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("case", CASES)
async def test_facade_async_callback_failure_remains_caller_owned(case):
    failure = RuntimeError("async caller callback failure", case.organization)
    closed: list[bool] = []
    done: list[Any] = []

    async def source():
        try:
            for event in case.stream_events_factory():
                yield event
        finally:
            closed.append(True)

    raw_source = source()

    def fail(_text):
        async def raise_failure():
            raise failure

        return raise_failure()

    try:
        with patch.object(case.async_client_class, "stream", return_value=raw_source):
            with pytest.raises(RuntimeError) as raised:
                async for _text in _facade(case).astream_chat(
                    **_chat_kwargs(case), on_delta=fail, on_done=done.append
                ):
                    pass

        assert raised.value is failure
        assert raised.value.args == ("async caller callback failure", case.organization)
        assert closed == [True]
        assert done == []
    finally:
        await raw_source.aclose()


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("case", CASES)
async def test_facade_async_task_cancellation_releases_suspended_source(case):
    suspended = asyncio.Event()
    release = asyncio.Event()
    closed: list[bool] = []
    done: list[Any] = []
    visible: list[str] = []

    async def source():
        try:
            events = case.stream_events_factory()
            for event in events[:-1]:
                yield event
            suspended.set()
            await release.wait()
            yield events[-1]
        finally:
            closed.append(True)

    raw_source = source()
    task = None

    async def consume():
        async for _text in _facade(case).astream_chat(
            **_chat_kwargs(case), buffer_chars=64, on_done=done.append
        ):
            visible.append(_text)

    try:
        with patch.object(case.async_client_class, "stream", return_value=raw_source):
            task = asyncio.create_task(consume())
            await asyncio.wait_for(suspended.wait(), timeout=2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=2)

        assert closed == [True]
        assert visible == []
        assert done == []
    finally:
        if task is not None:
            if not task.done():
                task.cancel()
            try:
                await asyncio.wait_for(task, timeout=2)
            except asyncio.CancelledError:
                pass
        release.set()
        await raw_source.aclose()


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize(
    "await_callback",
    (False, True),
    ids=("sync-callback", "awaitable-callback"),
)
async def test_facade_async_callback_failure_survives_source_close_failure(
    case, await_callback
):
    failure = RuntimeError("caller failure", case.organization)
    close_failure = RuntimeError("close failure", case.organization)
    source = _CountedAsyncSource(
        case.stream_events_factory(), close_error=close_failure
    )
    done: list[Any] = []

    def fail(_text):
        if await_callback:
            async def raise_failure():
                raise failure

            return raise_failure()
        raise failure

    with patch.object(case.async_client_class, "stream", return_value=source):
        with pytest.raises(RuntimeError) as raised:
            async for _text in _facade(case).astream_chat(
                **_chat_kwargs(case), on_delta=fail, on_done=done.append
            ):
                pass

    assert raised.value is failure
    assert raised.value.args == ("caller failure", case.organization)
    assert source.close_count == 1
    assert done == []


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("case", CASES)
async def test_facade_async_provider_failure_survives_source_close_failure(case):
    failure = LLMAPIRateLimitError(
        message="Rate limited during streaming.",
        detail=f"provider stream for {case.organization}",
    )
    close_failure = RuntimeError("close failure", case.organization)
    source = _CountedAsyncSource(next_error=failure, close_error=close_failure)
    done: list[Any] = []

    with patch.object(case.async_client_class, "stream", return_value=source):
        with pytest.raises(LLMAPIRateLimitError) as raised:
            async for _text in _facade(case).astream_chat(
                **_chat_kwargs(case), on_done=done.append
            ):
                pass

    assert raised.value is failure
    assert raised.value.args == (
        "Rate limited during streaming. Detail: provider stream for "
        f"{case.organization}",
    )
    assert raised.value.detail == f"provider stream for {case.organization}"
    assert source.close_count == 1
    assert done == []


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("case", CASES)
async def test_facade_async_cancellation_survives_source_close_failure(case):
    callback_started = asyncio.Event()
    release_callback = asyncio.Event()
    close_failure = RuntimeError("close failure", case.organization)
    source = _CountedAsyncSource(
        case.stream_events_factory(), close_error=close_failure
    )
    done: list[Any] = []

    async def on_delta(_text):
        callback_started.set()
        await release_callback.wait()

    async def consume():
        async for _text in _facade(case).astream_chat(
            **_chat_kwargs(case), on_delta=on_delta, on_done=done.append
        ):
            pass

    with patch.object(case.async_client_class, "stream", return_value=source):
        task = asyncio.create_task(consume())
        try:
            await asyncio.wait_for(callback_started.wait(), timeout=2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=2)
        finally:
            release_callback.set()
            if not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task

    assert source.close_count == 1
    assert done == []


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("case", CASES)
async def test_facade_async_early_close_survives_source_close_failure(case):
    close_failure = RuntimeError("close failure", case.organization)
    source = _CountedAsyncSource(
        case.stream_events_factory(), close_error=close_failure
    )
    done: list[Any] = []

    stream = None
    try:
        with patch.object(case.async_client_class, "stream", return_value=source):
            stream = _facade(case).astream_chat(
                **_chat_kwargs(case), buffer_chars=1, on_done=done.append
            )
            assert await stream.__anext__()
            await stream.aclose()
    finally:
        if stream is not None:
            await stream.aclose()
        await asyncio.get_running_loop().shutdown_asyncgens()

    assert source.close_count == 1
    assert done == []


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("case", CASES)
async def test_facade_async_normal_exhaustion_propagates_source_close_failure(case):
    close_failure = RuntimeError("close failure", case.organization)
    source = _CountedAsyncSource(
        case.stream_events_factory(), close_error=close_failure
    )

    with patch.object(case.async_client_class, "stream", return_value=source):
        with pytest.raises(RuntimeError) as raised:
            async for _text in _facade(case).astream_chat(
                **_chat_kwargs(case)
            ):
                pass

    assert raised.value is close_failure
    assert source.close_count == 1


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("case", CASES)
async def test_facade_async_close_failure_is_not_hidden_in_caller_except(case):
    close_failure = RuntimeError("close failure", case.organization)
    source = _CountedAsyncSource(
        case.stream_events_factory(), close_error=close_failure
    )

    with patch.object(case.async_client_class, "stream", return_value=source):
        with pytest.raises(RuntimeError) as raised:
            try:
                raise LookupError("unrelated caller exception")
            except LookupError:
                async for _text in _facade(case).astream_chat(
                    **_chat_kwargs(case)
                ):
                    pass

    assert raised.value is close_failure
    assert source.close_count == 1


@pytest.mark.unit
def test_facade_rejects_invalid_constructor_values_before_provider_selection():
    with pytest.raises(ValueError, match="Invalid organization"):
        UniversalLLMAPIAdapter(organization="", model="model", api_key="key")
    with pytest.raises(ValueError, match="Invalid model"):
        UniversalLLMAPIAdapter(organization="openai", model="", api_key="key")
    with pytest.raises(ValueError, match="Invalid API key"):
        UniversalLLMAPIAdapter(organization="openai", model="model", api_key="")


@pytest.mark.unit
def test_flash_facade_selection_uses_mistral_provider_without_zai_catalogue():
    """Keep optional Flash coverage local so Core-only runs remain available."""
    plugin_module = pytest.importorskip("llm_api_adapter_mistral.plugin")
    import llm_api_adapter.adapters.base_adapter as base_adapter_module
    import llm_api_adapter.universal_adapter as universal_adapter_module
    from llm_api_adapter.llm_registry.llm_registry import RegistrySpec
    from llm_api_adapter.service_provider_registry import ServiceProviderRegistry

    plugin = plugin_module.PLUGIN
    assert plugin.model_metadata is not None
    registry = RegistrySpec()
    assert registry.register_organization_metadata(plugin.model_metadata) is True
    service_providers = ServiceProviderRegistry({})
    plugin.register(service_providers)
    with patch.object(base_adapter_module, "LLM_REGISTRY", registry), patch.object(
        universal_adapter_module, "LLM_REGISTRY", registry
    ), patch.object(
        universal_adapter_module,
        "SERVICE_PROVIDER_REGISTRY",
        service_providers,
    ):
        facade = universal_adapter_module.UniversalLLMAPIAdapter(
            organization="zai",
            service_provider="mistral",
            model="glm-5.3-flash",
            api_key="test-api-key",
        )
        wire_model = facade.adapter.model_spec.provider_model
        response_payload = {
            "model": wire_model,
            "choices": [{"message": {"content": '{"answer": "ok"}'}}],
        }
        refusal_payload = {
            "model": wire_model,
            "choices": [{"message": {
                "content": None,
                "refusal": "I can't help with that.",
            }}],
        }
        transport = Mock(side_effect=[response_payload, refusal_payload])
        facade.adapter._post_payload = transport

        response = facade.chat(
            [{"role": "user", "content": "Return JSON."}],
            json_schema=FLAT_OBJECT_SCHEMA,
        )
        refusal = facade.chat(
            [{"role": "user", "content": "Return JSON."}],
            response_model=_StructuredAnswer,
        )

    assert wire_model
    assert transport.call_args.args[0]["model"] == wire_model
    assert response.parsed_json == {"answer": "ok"}
    assert refusal.refusal == "I can't help with that."
    assert refusal.parsed_json is None
    assert refusal.parsed_model is None
    assert refusal.incomplete_reason is None


def _expected_shared_tool_choice(case, choice):
    if case.wire_format == "openai_responses":
        return {
            "auto": "auto",
            "none": "none",
            "any": "required",
            "get_weather": {"type": "function", "name": "get_weather"},
        }[choice]
    if case.wire_format == "openai_chat_completions":
        return {
            "auto": "auto",
            "none": "none",
            "any": "required",
            "get_weather": {
                "type": "function",
                "function": {"name": "get_weather"},
            },
        }[choice]
    if case.wire_format == "anthropic_messages":
        return {
            "auto": {"type": "auto"},
            "none": {"type": "none"},
            "any": {"type": "any"},
            "get_weather": {"type": "tool", "name": "get_weather"},
        }[choice]
    if case.wire_format == "google_generate_content":
        mode = {"auto": "AUTO", "none": "NONE", "any": "ANY", "get_weather": "ANY"}[choice]
        config = {"mode": mode}
        if choice == "get_weather":
            config["allowedFunctionNames"] = ["get_weather"]
        return {"functionCallingConfig": config}
    raise AssertionError(f"Unknown wire format: {case.wire_format!r}")


def _shared_test_tool():
    return ToolSpec(
        name="get_weather",
        json_schema={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    )


def _assert_shared_tool_declaration(payload, case, tool):
    tools = payload["tools"]
    assert len(tools) == 1
    if case.wire_format == "openai_responses":
        declaration = tools[0]
        assert declaration["name"] == tool.name
        assert declaration["parameters"] == tool.json_schema
    elif case.wire_format == "openai_chat_completions":
        declaration = tools[0]["function"]
        assert declaration["name"] == tool.name
        assert declaration["parameters"] == tool.json_schema
    elif case.wire_format == "anthropic_messages":
        declaration = tools[0]
        assert declaration["name"] == tool.name
        assert declaration["input_schema"] == tool.json_schema
    elif case.wire_format == "google_generate_content":
        declarations = tools[0]["functionDeclarations"]
        assert len(declarations) == 1
        declaration = declarations[0]
        assert declaration["name"] == tool.name
        assert declaration["parametersJsonSchema"] == tool.json_schema
    else:
        raise AssertionError(f"Unknown wire format: {case.wire_format!r}")


def _assert_shared_tool_call_ids(response, case):
    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert call.name == "get_weather"
    assert call.arguments == {"city": "Tel Aviv"}
    expected_id = {
        "openai_responses": "call_123",
        "openai_chat_completions": "call_123",
        "anthropic_messages": "toolu_123",
        "google_generate_content": "get_weather",
    }.get(case.wire_format)
    assert expected_id is not None, f"Unknown wire format: {case.wire_format!r}"
    assert call.call_id == expected_id


@pytest.mark.unit
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("choice", ("auto", "none", "any", "get_weather"))
def test_facade_normalizes_tool_choices_and_completed_calls(case, choice):
    tool = _shared_test_tool()
    response = (
        case.response_factory(include_tool=True)
        if choice != "none"
        else case.response_factory()
    )
    transport = Mock(return_value=_JSONResponse(response))
    with patch.object(case.sync_client_class, "_send_request", new=transport):
        result = _facade(case).chat(
            **_chat_kwargs(case), tools=[tool], tool_choice=choice
        )

    payload = transport.call_args.args[1]
    choice_key = "toolConfig" if case.wire_format == "google_generate_content" else "tool_choice"
    assert payload[choice_key] == _expected_shared_tool_choice(case, choice)
    _assert_shared_tool_declaration(payload, case, tool)
    transport.assert_called_once()
    if choice == "none":
        assert not result.tool_calls
    else:
        _assert_shared_tool_call_ids(result, case)


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("choice", ("auto", "none", "any", "get_weather"))
async def test_facade_achat_normalizes_tool_choices_and_completed_calls(case, choice):
    tool = _shared_test_tool()
    response = (
        case.response_factory(include_tool=True)
        if choice != "none"
        else case.response_factory()
    )
    transport = AsyncMock(return_value=response)
    with patch.object(case.async_client_class, "_send_request", new=transport):
        result = await _facade(case).achat(
            **_chat_kwargs(case), tools=[tool], tool_choice=choice
        )

    payload = transport.await_args.args[1]
    choice_key = "toolConfig" if case.wire_format == "google_generate_content" else "tool_choice"
    assert payload[choice_key] == _expected_shared_tool_choice(case, choice)
    _assert_shared_tool_declaration(payload, case, tool)
    transport.assert_awaited_once()
    if choice == "none":
        assert not result.tool_calls
    else:
        _assert_shared_tool_call_ids(result, case)


def _shared_tool_history(case):
    return [
        UserMessage("What is the weather?"),
        AIMessage(
            content="",
            tool_calls=[ToolCall(
                name="get_weather", arguments={"city": "Haifa"},
                call_id=(
                    "toolu_history"
                    if case.wire_format == "anthropic_messages"
                    else "call_history"
                ),
            )],
        ),
        ToolMessage(
            content='{"temperature": 18}',
            tool_call_id=(
                "get_weather"
                if case.wire_format == "google_generate_content"
                else (
                    "toolu_history"
                    if case.wire_format == "anthropic_messages"
                    else "call_history"
                )
            ),
        ),
    ]


def _assert_shared_tool_history_payload(case, payload):
    payload_key = case.message_key
    serialized = payload[payload_key]
    call_id = "toolu_history" if case.wire_format == "anthropic_messages" else "call_history"
    expected_arguments = {"city": "Haifa"}
    if case.wire_format == "openai_responses":
        assert serialized[-1] == {
            "type": "function_call_output",
            "call_id": "call_history",
            "output": '{"temperature": 18}',
        }
        assistant_calls = [
            item for item in serialized[:-1]
            if isinstance(item, dict) and item.get("type") == "function_call"
        ]
        assert len(assistant_calls) == 1
        assistant_call = assistant_calls[0]
        assert assistant_call["call_id"] == "call_history"
        assert assistant_call["name"] == "get_weather"
        assert json.loads(assistant_call["arguments"]) == expected_arguments
    elif case.wire_format == "openai_chat_completions":
        assistant = serialized[-2]
        assert assistant["role"] == "assistant"
        assert assistant["content"] == ""
        assert len(assistant["tool_calls"]) == 1
        call = assistant["tool_calls"][0]
        assert call["id"] == "call_history"
        assert call["type"] == "function"
        assert call["function"]["name"] == "get_weather"
        assert json.loads(call["function"]["arguments"]) == expected_arguments
        assert serialized[-1] == {
            "role": "tool", "tool_call_id": "call_history",
            "content": '{"temperature": 18}',
        }
    elif case.wire_format == "anthropic_messages":
        assert serialized[-2] == {
            "role": "assistant",
            "content": [{
                "type": "tool_use", "id": call_id, "name": "get_weather",
                "input": expected_arguments,
            }],
        }
        assert serialized[-1] == {
            "role": "user",
            "content": [{
                "type": "tool_result", "tool_use_id": call_id,
                "content": '{"temperature": 18}',
            }],
        }
    elif case.wire_format == "google_generate_content":
        assert serialized[-2] == {
            "role": "model",
            "parts": [{"functionCall": {
                "name": "get_weather", "args": expected_arguments,
            }}],
        }
        assert serialized[-1] == {
            "role": "user",
            "parts": [{"functionResponse": {
                "name": "get_weather", "response": {"temperature": 18},
            }}],
        }
    else:
        raise AssertionError(f"Unknown wire format: {case.wire_format!r}")


@pytest.mark.unit
@pytest.mark.parametrize("case", CASES)
def test_facade_preserves_caller_tool_result_history_without_execution(case):
    tool = _shared_test_tool()
    history = _shared_tool_history(case)
    original_history = deepcopy(history)
    transport = Mock(return_value=_JSONResponse(case.response_factory()))
    with patch.object(case.sync_client_class, "_send_request", new=transport):
        result = _facade(case).chat(
            messages=history, tools=[tool], max_tokens=case.max_tokens
        )

    payload = transport.call_args.args[1]
    assert history == original_history
    transport.assert_called_once()
    _assert_shared_tool_history_payload(case, payload)
    assert result.content == "ok"


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("case", CASES)
async def test_facade_achat_preserves_caller_tool_result_history(case):
    tool = _shared_test_tool()
    history = _shared_tool_history(case)
    original_history = deepcopy(history)
    transport = AsyncMock(return_value=case.response_factory())
    with patch.object(case.async_client_class, "_send_request", new=transport):
        result = await _facade(case).achat(
            messages=history, tools=[tool], max_tokens=case.max_tokens
        )

    payload = transport.await_args.args[1]
    assert history == original_history
    transport.assert_awaited_once()
    _assert_shared_tool_history_payload(case, payload)
    assert result.content == "ok"


@pytest.mark.unit
@pytest.mark.parametrize("case", CASES)
def test_facade_stream_chat_preserves_caller_tool_result_history(case):
    tool = _shared_test_tool()
    history = _shared_tool_history(case)
    original_history = deepcopy(history)
    transport = Mock(return_value=iter(case.stream_events_factory()))
    completed = []
    with patch.object(case.sync_client_class, "stream", new=transport):
        list(_facade(case).stream_chat(
            messages=history, tools=[tool], max_tokens=case.max_tokens,
            on_done=completed.append,
        ))

    assert history == original_history
    transport.assert_called_once()
    _assert_shared_tool_history_payload(case, transport.call_args.kwargs)
    assert len(completed) == 1


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("case", CASES)
async def test_facade_astream_chat_preserves_caller_tool_result_history(case):
    tool = _shared_test_tool()
    history = _shared_tool_history(case)
    original_history = deepcopy(history)

    async def events():
        for event in case.stream_events_factory():
            yield event

    transport = Mock(return_value=events())
    completed = []
    with patch.object(case.async_client_class, "stream", new=transport):
        async for _text in _facade(case).astream_chat(
            messages=history, tools=[tool], max_tokens=case.max_tokens,
            on_done=completed.append,
        ):
            pass

    assert history == original_history
    transport.assert_called_once()
    _assert_shared_tool_history_payload(case, transport.call_args.kwargs)
    assert len(completed) == 1


def _openai_responses_continuation_case():
    return next(
        case.values[0] for case in CASES
        if case.values[0].organization == "openai"
    )


def _assert_openai_responses_continuation_payload(payload):
    assert payload["previous_response_id"] == "resp_previous"
    assert payload["input"] == [
        {"role": "user", "content": "What is the weather?"},
        {
            "type": "function_call_output",
            "call_id": "call_history",
            "output": '{"temperature": 18}',
        },
    ]
    assert not any(
        isinstance(item, dict) and item.get("type") == "function_call"
        for item in payload["input"]
    )


@pytest.mark.unit
@pytest.mark.parametrize("method", ("chat", "stream_chat"))
def test_openai_responses_continuation_uses_server_tool_call_context(method):
    case = _openai_responses_continuation_case()
    history = _shared_tool_history(case)
    original_history = deepcopy(history)
    previous = ChatResponse(content="", response_id="resp_previous")
    tool = _shared_test_tool()
    if method == "chat":
        transport = Mock(return_value=_JSONResponse(case.response_factory()))
        with patch.object(case.sync_client_class, "_send_request", new=transport):
            _facade(case).chat(
                messages=history, tools=[tool], previous_response=previous,
                max_tokens=case.max_tokens,
            )
        payload = transport.call_args.args[1]
    else:
        transport = Mock(return_value=iter(case.stream_events_factory()))
        with patch.object(case.sync_client_class, "stream", new=transport):
            list(_facade(case).stream_chat(
                messages=history, tools=[tool], previous_response=previous,
                max_tokens=case.max_tokens,
            ))
        payload = transport.call_args.kwargs

    assert history == original_history
    transport.assert_called_once()
    _assert_openai_responses_continuation_payload(payload)


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("method", ("achat", "astream_chat"))
async def test_openai_responses_async_continuation_uses_server_tool_call_context(method):
    case = _openai_responses_continuation_case()
    history = _shared_tool_history(case)
    original_history = deepcopy(history)
    previous = ChatResponse(content="", response_id="resp_previous")
    tool = _shared_test_tool()
    if method == "achat":
        transport = AsyncMock(return_value=case.response_factory())
        with patch.object(case.async_client_class, "_send_request", new=transport):
            await _facade(case).achat(
                messages=history, tools=[tool], previous_response=previous,
                max_tokens=case.max_tokens,
            )
        payload = transport.await_args.args[1]
        transport.assert_awaited_once()
    else:
        async def events():
            for event in case.stream_events_factory():
                yield event

        transport = Mock(return_value=events())
        with patch.object(case.async_client_class, "stream", new=transport):
            async for _text in _facade(case).astream_chat(
                messages=history, tools=[tool], previous_response=previous,
                max_tokens=case.max_tokens,
            ):
                pass
        payload = transport.call_args.kwargs
        transport.assert_called_once()

    assert history == original_history
    _assert_openai_responses_continuation_payload(payload)


def _assert_compatible_history_without_continuation(payload, case):
    _assert_shared_tool_history_payload(case, payload)
    serialized = payload[case.message_key]
    assert len(serialized) == 3
    assert serialized[0] == {"role": "user", "content": "What is the weather?"}
    assert "previous_response_id" not in payload
    assert "previous_response" not in payload
    serialized_payload = repr(payload)
    for marker in (
        "opaque-replay-sentinel",
        "resp_unrelated_previous",
        "unrelated prior content",
        "call_unrelated_previous",
    ):
        assert marker not in serialized_payload


@pytest.mark.unit
@pytest.mark.parametrize("method", ("chat", "stream_chat"))
def test_compatible_continuation_keeps_caller_history_without_replay(method):
    case = _COMPATIBLE_CASE
    history = _shared_tool_history(case)
    original_history = deepcopy(history)
    previous = ChatResponse(
        response_id="resp_unrelated_previous",
        content="unrelated prior content",
        tool_calls=[ToolCall(
            name="unrelated_tool", arguments={"note": "opaque-replay-sentinel"},
            call_id="call_unrelated_previous",
        )],
        provider_data={"replay": "opaque-replay-sentinel"},
    )
    original_previous = deepcopy(previous)
    tool = _shared_test_tool()
    facade = _facade(case)
    if method == "chat":
        transport = Mock(return_value=_JSONResponse(case.response_factory()))
        with patch.object(case.sync_client_class, "_send_request", new=transport):
            facade.chat(
                messages=history, tools=[tool], max_tokens=case.max_tokens,
                previous_response=previous,
            )
        payload = transport.call_args.args[1]
    else:
        transport = Mock(return_value=iter(case.stream_events_factory()))
        with patch.object(case.sync_client_class, "stream", new=transport):
            list(facade.stream_chat(
                messages=history, tools=[tool], max_tokens=case.max_tokens,
                previous_response=previous,
            ))
        payload = transport.call_args.kwargs

    assert history == original_history
    assert previous == original_previous
    transport.assert_called_once()
    _assert_compatible_history_without_continuation(payload, case)
    assert facade.adapter.model_spec is None
    assert facade.adapter.pricing is None


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("method", ("achat", "astream_chat"))
async def test_compatible_async_continuation_keeps_caller_history_without_replay(method):
    case = _COMPATIBLE_CASE
    history = _shared_tool_history(case)
    original_history = deepcopy(history)
    previous = ChatResponse(
        response_id="resp_unrelated_previous",
        content="unrelated prior content",
        tool_calls=[ToolCall(
            name="unrelated_tool", arguments={"note": "opaque-replay-sentinel"},
            call_id="call_unrelated_previous",
        )],
        provider_data={"replay": "opaque-replay-sentinel"},
    )
    original_previous = deepcopy(previous)
    tool = _shared_test_tool()
    facade = _facade(case)
    if method == "achat":
        transport = AsyncMock(return_value=case.response_factory())
        with patch.object(case.async_client_class, "_send_request", new=transport):
            await facade.achat(
                messages=history, tools=[tool], max_tokens=case.max_tokens,
                previous_response=previous,
            )
        payload = transport.await_args.args[1]
        transport.assert_awaited_once()
    else:
        async def events():
            for event in case.stream_events_factory():
                yield event

        transport = Mock(return_value=events())
        with patch.object(case.async_client_class, "stream", new=transport):
            async for _text in facade.astream_chat(
                messages=history, tools=[tool], max_tokens=case.max_tokens,
                previous_response=previous,
            ):
                pass
        payload = transport.call_args.kwargs
        transport.assert_called_once()

    assert history == original_history
    assert previous == original_previous
    _assert_compatible_history_without_continuation(payload, case)
    assert facade.adapter.model_spec is None
    assert facade.adapter.pricing is None


# Compatible structured-output coverage keeps request ownership, response parsing,
# and terminal handling on the same four public facade methods.
_COMPATIBLE_CASE_MATCHES = tuple(
    case.values[0]
    for case in CASES
    if (
        case.values[0].organization == "openai_compatible"
        and case.values[0].wire_format == "openai_chat_completions"
    )
)
assert len(_COMPATIBLE_CASE_MATCHES) == 1
_COMPATIBLE_CASE = _COMPATIBLE_CASE_MATCHES[0]
_COMPATIBLE_SCHEMA_RESULTS = (
    ("flat_object", {"answer": "ready"}),
    ("nullable_required_field", {"nickname": None}),
    ("enum", {"priority": "high"}),
    ("array", {"tags": ["sdk", "portable"]}),
    ("inline_nested_object", {"contact": {"name": "Ada"}}),
)


class _AdaOnlyNestedResponse(NestedPydanticResponse):
    @field_validator("contact")
    @classmethod
    def contact_must_be_ada(cls, contact):
        if contact.name != "Ada":
            raise ValueError("contact name must be Ada")
        return contact


@pytest.mark.unit
@pytest.mark.parametrize(("fixture_name", "expected"), _COMPATIBLE_SCHEMA_RESULTS)
def test_compatible_chat_parses_each_portable_schema_result(fixture_name, expected):
    case = _COMPATIBLE_CASE
    source_schema = PORTABLE_PROFILE_SCHEMAS[fixture_name]
    original_schema = deepcopy(source_schema)
    content = json.dumps(expected)
    transport = Mock(return_value=_JSONResponse(_compatible_response(content)))

    with patch.object(case.sync_client_class, "_send_request", new=transport):
        response = _facade(case).chat(
            **_chat_kwargs(case), json_schema=source_schema
        )

    schema = _structured_schema_from_payload(case, transport.call_args.args[1])
    assert response.parsed_json == expected
    assert response.parsed_model is None
    assert schema == original_schema
    assert source_schema == original_schema
    transport.assert_called_once()


@pytest.mark.unit
@pytest.mark.parametrize(("fixture_name", "expected"), _COMPATIBLE_SCHEMA_RESULTS)
def test_compatible_stream_chat_parses_each_portable_schema_result(fixture_name, expected):
    case = _COMPATIBLE_CASE
    source_schema = PORTABLE_PROFILE_SCHEMAS[fixture_name]
    original_schema = deepcopy(source_schema)
    content = json.dumps(expected)
    events = _decode_compatible_sse(
        structured_result_sse_lines(content, usage=_compatible_usage())
    )
    completed = []

    with patch.object(case.sync_client_class, "stream", return_value=iter(events)) as transport:
        chunks = list(_facade(case).stream_chat(
            **_chat_kwargs(case), json_schema=source_schema, on_done=completed.append
        ))

    schema = _structured_schema_from_payload(case, transport.call_args.kwargs)
    assert "".join(chunks) == content
    assert len(completed) == 1
    assert completed[0].parsed_json == expected
    assert completed[0].parsed_model is None
    assert schema == original_schema
    assert source_schema == original_schema
    transport.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize(("fixture_name", "expected"), _COMPATIBLE_SCHEMA_RESULTS)
async def test_compatible_achat_parses_each_portable_schema_result(fixture_name, expected):
    case = _COMPATIBLE_CASE
    source_schema = PORTABLE_PROFILE_SCHEMAS[fixture_name]
    original_schema = deepcopy(source_schema)
    content = json.dumps(expected)
    transport = AsyncMock(return_value=_compatible_response(content))

    with patch.object(case.async_client_class, "_send_request", new=transport):
        response = await _facade(case).achat(
            **_chat_kwargs(case), json_schema=source_schema
        )

    schema = _structured_schema_from_payload(case, transport.await_args.args[1])
    assert response.parsed_json == expected
    assert response.parsed_model is None
    assert schema == original_schema
    assert source_schema == original_schema
    transport.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize(("fixture_name", "expected"), _COMPATIBLE_SCHEMA_RESULTS)
async def test_compatible_astream_parses_each_portable_schema_result(fixture_name, expected):
    case = _COMPATIBLE_CASE
    source_schema = PORTABLE_PROFILE_SCHEMAS[fixture_name]
    original_schema = deepcopy(source_schema)
    content = json.dumps(expected)
    events = _decode_compatible_sse(
        structured_result_sse_lines(content, usage=_compatible_usage())
    )
    completed = []

    with patch.object(
        case.async_client_class, "stream", return_value=_as_async_iter(events)
    ) as transport:
        chunks = [
            text async for text in _facade(case).astream_chat(
                **_chat_kwargs(case), json_schema=source_schema, on_done=completed.append
            )
        ]

    schema = _structured_schema_from_payload(case, transport.call_args.kwargs)
    assert "".join(chunks) == content
    assert len(completed) == 1
    assert completed[0].parsed_json == expected
    assert completed[0].parsed_model is None
    assert schema == original_schema
    assert source_schema == original_schema
    transport.assert_called_once()
    transport.assert_called_once()


@pytest.mark.unit
@pytest.mark.parametrize("invalid", (False, True), ids=("valid", "custom-validator-failure"))
def test_compatible_chat_validates_the_original_response_model(invalid):
    case = _COMPATIBLE_CASE
    content = '{"contact":{"name":"Grace"}}' if invalid else NESTED_PYDANTIC_RESPONSE_JSON
    transport = Mock(return_value=_JSONResponse(_compatible_response(content)))

    with patch.object(case.sync_client_class, "_send_request", new=transport):
        if invalid:
            with pytest.raises(JSONSchemaError):
                _facade(case).chat(
                    **_chat_kwargs(case), response_model=_AdaOnlyNestedResponse
                )
        else:
            response = _facade(case).chat(
                **_chat_kwargs(case), response_model=_AdaOnlyNestedResponse
            )

    schema = _structured_schema_from_payload(case, transport.call_args.args[1])
    assert "$defs" not in schema
    assert not _contains_reference(schema)
    transport.assert_called_once()
    if invalid:
        return
    assert response.parsed_json == json.loads(NESTED_PYDANTIC_RESPONSE_JSON)
    assert type(response.parsed_model) is _AdaOnlyNestedResponse
    assert response.parsed_model.contact.name == "Ada"


@pytest.mark.unit
@pytest.mark.parametrize("invalid", (False, True), ids=("valid", "custom-validator-failure"))
def test_compatible_stream_chat_validates_the_original_response_model(invalid):
    case = _COMPATIBLE_CASE
    content = '{"contact":{"name":"Grace"}}' if invalid else NESTED_PYDANTIC_RESPONSE_JSON
    events = _decode_compatible_sse(
        structured_result_sse_lines(content, usage=_compatible_usage())
    )
    completed = []
    visible = []

    with patch.object(case.sync_client_class, "stream", return_value=iter(events)) as transport:
        stream = _facade(case).stream_chat(
            **_chat_kwargs(case), response_model=_AdaOnlyNestedResponse,
            buffer_chars=64, on_done=completed.append,
        )
        if invalid:
            with pytest.raises(JSONSchemaError):
                visible.extend(stream)
        else:
            visible.extend(stream)

    schema = _structured_schema_from_payload(case, transport.call_args.kwargs)
    assert "$defs" not in schema
    assert not _contains_reference(schema)
    if invalid:
        assert visible == []
        assert completed == []
    else:
        assert len(completed) == 1
        assert completed[0].parsed_json == json.loads(NESTED_PYDANTIC_RESPONSE_JSON)
        assert type(completed[0].parsed_model) is _AdaOnlyNestedResponse
        assert completed[0].parsed_model.contact.name == "Ada"
    transport.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("invalid", (False, True), ids=("valid", "custom-validator-failure"))
async def test_compatible_achat_validates_the_original_response_model(invalid):
    case = _COMPATIBLE_CASE
    content = '{"contact":{"name":"Grace"}}' if invalid else NESTED_PYDANTIC_RESPONSE_JSON
    transport = AsyncMock(return_value=_compatible_response(content))

    with patch.object(case.async_client_class, "_send_request", new=transport):
        if invalid:
            with pytest.raises(JSONSchemaError):
                await _facade(case).achat(
                    **_chat_kwargs(case), response_model=_AdaOnlyNestedResponse
                )
        else:
            response = await _facade(case).achat(
                **_chat_kwargs(case), response_model=_AdaOnlyNestedResponse
            )

    schema = _structured_schema_from_payload(case, transport.await_args.args[1])
    assert "$defs" not in schema
    assert not _contains_reference(schema)
    transport.assert_awaited_once()
    if invalid:
        return
    assert response.parsed_json == json.loads(NESTED_PYDANTIC_RESPONSE_JSON)
    assert type(response.parsed_model) is _AdaOnlyNestedResponse
    assert response.parsed_model.contact.name == "Ada"


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("invalid", (False, True), ids=("valid", "custom-validator-failure"))
async def test_compatible_astream_validates_the_original_response_model(invalid):
    case = _COMPATIBLE_CASE
    content = '{"contact":{"name":"Grace"}}' if invalid else NESTED_PYDANTIC_RESPONSE_JSON
    events = _decode_compatible_sse(
        structured_result_sse_lines(content, usage=_compatible_usage())
    )
    completed = []
    visible = []

    with patch.object(
        case.async_client_class, "stream", return_value=_as_async_iter(events)
    ) as transport:
        stream = _facade(case).astream_chat(
            **_chat_kwargs(case), response_model=_AdaOnlyNestedResponse,
            buffer_chars=64, on_done=completed.append,
        )
        if invalid:
            with pytest.raises(JSONSchemaError):
                async for text in stream:
                    visible.append(text)
        else:
            async for text in stream:
                visible.append(text)

    schema = _structured_schema_from_payload(case, transport.call_args.kwargs)
    assert "$defs" not in schema
    assert not _contains_reference(schema)
    if invalid:
        assert visible == []
        assert completed == []
    else:
        assert len(completed) == 1
        assert completed[0].parsed_json == json.loads(NESTED_PYDANTIC_RESPONSE_JSON)
        assert type(completed[0].parsed_model) is _AdaOnlyNestedResponse
        assert completed[0].parsed_model.contact.name == "Ada"
    transport.assert_called_once()


_COMPATIBLE_INVALID_REQUESTS = (
    "invalid_schema", "external_reference", "schema_model_conflict",
)


def _compatible_invalid_request(name):
    if name == "invalid_schema":
        schema = deepcopy(FLAT_OBJECT_SCHEMA)
        schema["required"] = []
        return {"json_schema": schema}
    if name == "external_reference":
        schema = deepcopy(PORTABLE_PROFILE_SCHEMAS["inline_nested_object"])
        schema["properties"]["contact"] = {"$ref": "https://example.com/contact.json"}
        return {"json_schema": schema}
    if name == "schema_model_conflict":
        return {
            "json_schema": deepcopy(FLAT_OBJECT_SCHEMA),
            "response_model": NestedPydanticResponse,
        }
    raise AssertionError(f"Unknown invalid structured request: {name!r}")


@pytest.mark.unit
@pytest.mark.parametrize("invalid_request", _COMPATIBLE_INVALID_REQUESTS)
@pytest.mark.parametrize("streaming", (False, True), ids=("chat", "stream"))
def test_compatible_sync_rejects_invalid_structured_requests_before_sending(
    invalid_request, streaming
):
    case = _COMPATIBLE_CASE
    send = Mock()
    with (
        patch.object(case.sync_client_class, "stream" if streaming else "_send_request", new=send),
        pytest.raises(JSONSchemaError),
    ):
        if streaming:
            list(_facade(case).stream_chat(
                **_chat_kwargs(case), **_compatible_invalid_request(invalid_request)
            ))
        else:
            _facade(case).chat(
                **_chat_kwargs(case), **_compatible_invalid_request(invalid_request)
            )
    send.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("invalid_request", _COMPATIBLE_INVALID_REQUESTS)
@pytest.mark.parametrize("streaming", (False, True), ids=("achat", "astream"))
async def test_compatible_async_rejects_invalid_structured_requests_before_sending(
    invalid_request, streaming
):
    case = _COMPATIBLE_CASE
    send = Mock()
    with (
        patch.object(case.async_client_class, "stream" if streaming else "_send_request", new=send),
        pytest.raises(JSONSchemaError),
    ):
        if streaming:
            async for _text in _facade(case).astream_chat(
                **_chat_kwargs(case), **_compatible_invalid_request(invalid_request)
            ):
                pass
        else:
            await _facade(case).achat(
                **_chat_kwargs(case), **_compatible_invalid_request(invalid_request)
            )
    send.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize("outcome", ("invalid_json", "pydantic_validation_error"))
def test_compatible_sync_invalid_stream_results_do_not_flush_or_complete(outcome):
    case = _COMPATIBLE_CASE
    fixture = next(item for item in STRUCTURED_OUTPUT_OUTCOMES if item.name == outcome)
    kwargs = _structured_outcome_kwargs(case, outcome)
    completed = []
    kwargs.update(buffer_chars=64, on_done=completed.append)
    events = _terminal_outcome_stream_events(case, outcome, fixture.content)
    visible = []

    with patch.object(case.sync_client_class, "stream", return_value=iter(events)) as transport:
        with pytest.raises(JSONSchemaError):
            visible.extend(_facade(case).stream_chat(**kwargs))

    assert visible == []
    assert completed == []
    transport.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("outcome", ("invalid_json", "pydantic_validation_error"))
async def test_compatible_async_stream_invalid_results_do_not_flush_or_complete(outcome):
    case = _COMPATIBLE_CASE
    fixture = next(item for item in STRUCTURED_OUTPUT_OUTCOMES if item.name == outcome)
    kwargs = _structured_outcome_kwargs(case, outcome)
    completed = []
    kwargs.update(buffer_chars=64, on_done=completed.append)
    events = _terminal_outcome_stream_events(case, outcome, fixture.content)
    visible = []

    with patch.object(
        case.async_client_class, "stream", return_value=_as_async_iter(events)
    ) as transport:
        with pytest.raises(JSONSchemaError):
            async for text in _facade(case).astream_chat(**kwargs):
                visible.append(text)

    assert visible == []
    assert completed == []
    transport.assert_called_once()


@pytest.mark.unit
@pytest.mark.parametrize("outcome", ("refusal", "incomplete_result"))
@pytest.mark.parametrize("request_kind", ("raw_schema", "original_model"))
def test_compatible_chat_terminal_outcome_skips_structured_parsing(outcome, request_kind):
    case = _COMPATIBLE_CASE
    content = '{"contact":{"name":"Grace"}}' if request_kind == "original_model" else '{"answer":"ok"}'
    message = {"role": "assistant", "content": content}
    if outcome == "refusal":
        message["refusal"] = case.expected_refusal
        payload = chat_completion_response(message=message, usage=_compatible_usage(), model=case.model)
    else:
        payload = incomplete_response(content, usage=_compatible_usage(), model=case.model)
    transport = Mock(return_value=_JSONResponse(payload))
    request = (
        {"json_schema": FLAT_OBJECT_SCHEMA}
        if request_kind == "raw_schema"
        else {"response_model": _AdaOnlyNestedResponse}
    )

    with patch.object(case.sync_client_class, "_send_request", new=transport):
        response = _facade(case).chat(**_chat_kwargs(case), **request)

    _assert_structured_terminal_outcome(response, case, outcome)
    transport.assert_called_once()


@pytest.mark.unit
@pytest.mark.parametrize("outcome", ("refusal", "incomplete_result"))
@pytest.mark.parametrize("request_kind", ("raw_schema", "original_model"))
def test_compatible_stream_terminal_outcome_skips_structured_parsing(outcome, request_kind):
    case = _COMPATIBLE_CASE
    content = '{"contact":{"name":"Grace"}}' if request_kind == "original_model" else '{"answer":"ok"}'
    if outcome == "refusal":
        lines = premature_eof_sse_lines(content)
        lines.extend(refusal_sse_lines(case.expected_refusal))
    else:
        lines = incomplete_sse_lines(content)
    events = _decode_compatible_sse(lines)
    request = (
        {"json_schema": FLAT_OBJECT_SCHEMA}
        if request_kind == "raw_schema"
        else {"response_model": _AdaOnlyNestedResponse}
    )
    completed = []

    with patch.object(case.sync_client_class, "stream", return_value=iter(events)) as transport:
        visible = list(_facade(case).stream_chat(
            **_chat_kwargs(case), **request, on_done=completed.append
        ))

    assert json.loads("".join(visible)) == json.loads(content)
    assert len(completed) == 1
    _assert_structured_terminal_outcome(completed[0], case, outcome)
    transport.assert_called_once()


def _caller_pricing_facade(case):
    pricing = Pricing.from_dict(
        [{
            "up_to_prompt_tokens": None,
            "input_per_1m": 10,
            "output_per_1m": 20,
        }],
        currency="EUR",
    )
    facade = UniversalLLMAPIAdapter(
        organization=case.organization,
        model=case.model,
        api_key="test-api-key",
        **dict(case.connection_kwargs),
    )
    facade.adapter.pricing = pricing
    return facade, pricing


@pytest.mark.unit
@pytest.mark.parametrize("method", ("chat", "stream_chat"))
def test_caller_pricing_compatible_sync_facade_finalizes_costs(method):
    case = _COMPATIBLE_CASE
    facade, caller_pricing = _caller_pricing_facade(case)
    completed = []
    if method == "chat":
        transport = Mock(return_value=_JSONResponse(case.response_factory()))
        with patch.object(case.sync_client_class, "_send_request", new=transport):
            response = facade.chat(**_chat_kwargs(case))
    else:
        transport = Mock(return_value=iter(case.stream_events_factory()))
        with patch.object(case.sync_client_class, "stream", new=transport):
            list(facade.stream_chat(
                **_chat_kwargs(case),
                buffer_chars=64,
                on_done=completed.append,
            ))
        assert len(completed) == 1
        response = completed[0]

    assert response.cost_input == pytest.approx(2 * 10 / 1_000_000)
    assert response.cost_output == pytest.approx(3 * 20 / 1_000_000)
    assert response.cost_total == pytest.approx(80 / 1_000_000)
    assert response.currency == "EUR"
    assert response.cost_breakdown is None
    assert facade.adapter.model_spec is None
    assert facade.adapter.pricing is not None
    assert facade.pricing is facade.adapter.pricing
    assert caller_pricing.currency == "EUR"
    transport.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("method", ("achat", "astream_chat"))
async def test_caller_pricing_compatible_async_facade_finalizes_costs(method):
    case = _COMPATIBLE_CASE
    facade, _caller_pricing = _caller_pricing_facade(case)
    completed = []
    if method == "achat":
        transport = AsyncMock(return_value=case.response_factory())
        with patch.object(case.async_client_class, "_send_request", new=transport):
            response = await facade.achat(**_chat_kwargs(case))
    else:
        transport = Mock(return_value=_as_async_iter(case.stream_events_factory()))
        with patch.object(case.async_client_class, "stream", new=transport):
            async for _text in facade.astream_chat(
                **_chat_kwargs(case), buffer_chars=64, on_done=completed.append
            ):
                pass
        assert len(completed) == 1
        response = completed[0]

    assert response.cost_input == pytest.approx(2 * 10 / 1_000_000)
    assert response.cost_output == pytest.approx(3 * 20 / 1_000_000)
    assert response.cost_total == pytest.approx(80 / 1_000_000)
    assert response.currency == "EUR"
    assert response.cost_breakdown is None
    assert facade.adapter.model_spec is None
    assert facade.adapter.pricing is not None
    transport.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("outcome", ("refusal", "incomplete_result"))
@pytest.mark.parametrize("request_kind", ("raw_schema", "original_model"))
async def test_compatible_achat_terminal_outcome_skips_structured_parsing(
    outcome, request_kind
):
    case = _COMPATIBLE_CASE
    content = '{"contact":{"name":"Grace"}}' if request_kind == "original_model" else '{"answer":"ok"}'
    message = {"role": "assistant", "content": content}
    if outcome == "refusal":
        message["refusal"] = case.expected_refusal
        payload = chat_completion_response(message=message, usage=_compatible_usage(), model=case.model)
    else:
        payload = incomplete_response(content, usage=_compatible_usage(), model=case.model)
    transport = AsyncMock(return_value=payload)
    request = (
        {"json_schema": FLAT_OBJECT_SCHEMA}
        if request_kind == "raw_schema"
        else {"response_model": _AdaOnlyNestedResponse}
    )

    with patch.object(case.async_client_class, "_send_request", new=transport):
        response = await _facade(case).achat(**_chat_kwargs(case), **request)

    _assert_structured_terminal_outcome(response, case, outcome)
    transport.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("outcome", ("refusal", "incomplete_result"))
@pytest.mark.parametrize("request_kind", ("raw_schema", "original_model"))
async def test_compatible_astream_terminal_outcome_skips_structured_parsing(
    outcome, request_kind
):
    case = _COMPATIBLE_CASE
    content = '{"contact":{"name":"Grace"}}' if request_kind == "original_model" else '{"answer":"ok"}'
    if outcome == "refusal":
        lines = premature_eof_sse_lines(content)
        lines.extend(refusal_sse_lines(case.expected_refusal))
    else:
        lines = incomplete_sse_lines(content)
    events = _decode_compatible_sse(lines)
    request = (
        {"json_schema": FLAT_OBJECT_SCHEMA}
        if request_kind == "raw_schema"
        else {"response_model": _AdaOnlyNestedResponse}
    )
    completed = []

    with patch.object(
        case.async_client_class, "stream", return_value=_as_async_iter(events)
    ) as transport:
        visible = [
            text async for text in _facade(case).astream_chat(
                **_chat_kwargs(case), **request, on_done=completed.append
            )
        ]

    assert json.loads("".join(visible)) == json.loads(content)
    assert len(completed) == 1
    _assert_structured_terminal_outcome(completed[0], case, outcome)
    transport.assert_called_once()


def _compatible_responses_case():
    """Return a bounded Responses case without changing the shared CASES table."""
    return replace(
        _COMPATIBLE_CASE,
        model="vendor/conformance-responses-tools",
        connection_kwargs=(
            ("base_url", "https://example.invalid/proxy/v1/"),
            ("api_variant", "responses"),
        ),
        wire_format="openai_responses",
        message_key="input",
        structured_schema_path=("text", "format", "schema"),
        response_factory=_compatible_responses_response,
        stream_events_factory=_compatible_responses_stream_events,
    )


def _compatible_responses_response(
    content="ok", *, include_tool=False, include_reasoning=False,
):
    del include_reasoning
    if include_tool:
        return responses_tool_call_response(
            name="get_weather", arguments_json='{"city": "Tel Aviv"}',
            call_id="call_123", model="vendor/conformance-responses-tools",
        )
    return responses_text_response(
        content, model="vendor/conformance-responses-tools"
    )


def _compatible_responses_stream_events():
    return _decode_compatible_sse(
        responses_text_sse_lines("ok", model="vendor/conformance-responses-tools")
    )


def _responses_tool_stream_events():
    return _decode_compatible_sse(
        responses_tool_call_sse_lines(
            name="get_weather",
            arguments_json='{"city": "Tel Aviv"}',
            argument_fragments=("{\"city\": ", "\"Tel Aviv\"}"),
            call_id="call_123",
            model="vendor/conformance-responses-tools",
        )
    )


def _responses_stream_request_probe(client_class, events, *, asynchronous=False):
    """Patch only transport streaming, preserving the client's wire formatting."""
    observed = []

    def stream_request(_client, url, payload, timeout=None):
        observed.append((url, deepcopy(payload), timeout))
        return _as_async_iter(events) if asynchronous else iter(events)

    return patch.object(client_class, "_stream_request", new=stream_request), observed


def _assert_responses_stream_request(observed, case):
    assert len(observed) == 1
    url, payload, _timeout = observed[0]
    root = dict(case.connection_kwargs)["base_url"].rstrip("/")
    assert url == f"{root}/responses"
    assert payload["model"] == case.model
    assert payload["store"] is False
    assert payload["stream"] is True
    return payload


def _assert_responses_ordinary_request(url, payload, case):
    root = dict(case.connection_kwargs)["base_url"].rstrip("/")
    assert url == f"{root}/responses"
    assert payload["model"] == case.model
    assert payload["store"] is False
    assert not payload.get("stream", False)
    return payload


@pytest.mark.unit
@pytest.mark.parametrize("method", ("chat", "achat", "stream_chat", "astream_chat"))
@pytest.mark.parametrize("choice", ("auto", "none", "any", "get_weather"))
def test_compatible_responses_tool_choices_and_completed_calls(method, choice):
    case = _compatible_responses_case()
    tool = _shared_test_tool()
    response = (
        _compatible_responses_response(include_tool=True)
        if choice != "none" else _compatible_responses_response()
    )
    completed = []
    if method == "chat":
        transport = Mock(return_value=_JSONResponse(response))
        with patch.object(case.sync_client_class, "_send_request", new=transport):
            result = _facade(case).chat(
                **_chat_kwargs(case), tools=[tool], tool_choice=choice
            )
        payload = transport.call_args.args[1]
        transport.assert_called_once()
    elif method == "stream_chat":
        events = (
            _responses_tool_stream_events() if choice != "none"
            else _compatible_responses_stream_events()
        )
        context, observed = _responses_stream_request_probe(case.sync_client_class, events)
        with context:
            list(_facade(case).stream_chat(
                **_chat_kwargs(case), tools=[tool], tool_choice=choice,
                on_done=completed.append,
            ))
        payload = _assert_responses_stream_request(observed, case)
        result = completed[0]
    elif method == "achat":
        transport = AsyncMock(return_value=response)
        with patch.object(case.async_client_class, "_send_request", new=transport):
            result = asyncio.run(_facade(case).achat(
                **_chat_kwargs(case), tools=[tool], tool_choice=choice
            ))
        payload = transport.await_args.args[1]
        transport.assert_awaited_once()
    else:
        events = (
            _responses_tool_stream_events() if choice != "none"
            else _compatible_responses_stream_events()
        )
        context, observed = _responses_stream_request_probe(
            case.async_client_class, events, asynchronous=True
        )
        async def run():
            async for _text in _facade(case).astream_chat(
                **_chat_kwargs(case), tools=[tool], tool_choice=choice,
                on_done=completed.append,
            ):
                pass
        with context:
            asyncio.run(run())
        payload = _assert_responses_stream_request(observed, case)
        result = completed[0]

    assert payload["tool_choice"] == _expected_shared_tool_choice(case, choice)
    _assert_shared_tool_declaration(payload, case, tool)
    if method in ("chat", "achat"):
        url = transport.call_args.args[0] if method == "chat" else transport.await_args.args[0]
        _assert_responses_ordinary_request(url, payload, case)
    if choice == "none":
        assert not result.tool_calls
    else:
        _assert_shared_tool_call_ids(result, case)


def _assert_responses_history_has_no_replay(payload, case):
    _assert_shared_tool_history_payload(case, payload)
    serialized = payload[case.message_key]
    assert len(serialized) == 3
    assert serialized[0] == {"role": "user", "content": "What is the weather?"}
    assert serialized[1]["type"] == "function_call"
    assert serialized[2] == {
        "type": "function_call_output",
        "call_id": "call_history",
        "output": '{"temperature": 18}',
    }
    assert not {"previous_response_id", "conversation", "previous_response"} & payload.keys()
    for marker in ("opaque-replay-sentinel", "resp_unrelated_previous", "unrelated prior content"):
        assert marker not in repr(payload)


@pytest.mark.unit
@pytest.mark.parametrize("method", ("chat", "achat", "stream_chat", "astream_chat"))
def test_compatible_responses_uses_full_caller_tool_history_with_previous_id(method):
    case = _compatible_responses_case()
    history = _shared_tool_history(case)
    original_history = deepcopy(history)
    previous = ChatResponse(
        response_id="resp_unrelated_previous", content="unrelated prior content",
        tool_calls=[ToolCall(name="unrelated", arguments={"x": 1}, call_id="call_extra")],
        provider_data={"replay": "opaque-replay-sentinel"},
    )
    original_previous = deepcopy(previous)
    if method == "chat":
        transport = Mock(return_value=_JSONResponse(_compatible_responses_response()))
        with patch.object(case.sync_client_class, "_send_request", new=transport):
            _facade(case).chat(
                messages=history, tools=[_shared_test_tool()], previous_response=previous,
            )
        payload = transport.call_args.args[1]
        transport.assert_called_once()
    elif method == "stream_chat":
        context, observed = _responses_stream_request_probe(
            case.sync_client_class, _compatible_responses_stream_events()
        )
        with context:
            list(_facade(case).stream_chat(
                messages=history, tools=[_shared_test_tool()], previous_response=previous,
            ))
        payload = _assert_responses_stream_request(observed, case)
    elif method == "achat":
        transport = AsyncMock(return_value=_compatible_responses_response())
        with patch.object(case.async_client_class, "_send_request", new=transport):
            asyncio.run(_facade(case).achat(
                messages=history, tools=[_shared_test_tool()], previous_response=previous,
            ))
        payload = transport.await_args.args[1]
        transport.assert_awaited_once()
    else:
        context, observed = _responses_stream_request_probe(
            case.async_client_class, _compatible_responses_stream_events(), asynchronous=True
        )
        async def run():
            async for _text in _facade(case).astream_chat(
                messages=history, tools=[_shared_test_tool()], previous_response=previous,
            ):
                pass
        with context:
            asyncio.run(run())
        payload = _assert_responses_stream_request(observed, case)

    assert history == original_history
    assert previous == original_previous
    _assert_responses_history_has_no_replay(payload, case)
    if method in ("chat", "achat"):
        url = transport.call_args.args[0] if method == "chat" else transport.await_args.args[0]
        _assert_responses_ordinary_request(url, payload, case)


def _responses_structured_result(content):
    return responses_response(
        output=[{
            "id": "msg-structured",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": content, "annotations": []}],
        }],
        model="vendor/conformance-responses-tools",
    )


def _responses_structured_events(content):
    return _decode_compatible_sse(
        responses_text_sse_lines(content, model="vendor/conformance-responses-tools")
    )


@pytest.mark.unit
@pytest.mark.parametrize("method", ("chat", "achat", "stream_chat", "astream_chat"))
@pytest.mark.parametrize(("fixture_name", "expected"), _COMPATIBLE_SCHEMA_RESULTS)
def test_compatible_responses_parses_portable_nested_schemas(method, fixture_name, expected):
    case = _compatible_responses_case()
    source_schema = deepcopy(PORTABLE_PROFILE_SCHEMAS[fixture_name])
    original_schema = deepcopy(source_schema)
    content = json.dumps(expected)
    if method == "chat":
        transport = Mock(return_value=_JSONResponse(_responses_structured_result(content)))
        with patch.object(case.sync_client_class, "_send_request", new=transport):
            response = _facade(case).chat(**_chat_kwargs(case), json_schema=source_schema)
        payload = transport.call_args.args[1]
        _assert_responses_ordinary_request(transport.call_args.args[0], payload, case)
        transport.assert_called_once()
    elif method == "stream_chat":
        completed = []
        context, observed = _responses_stream_request_probe(
            case.sync_client_class, _responses_structured_events(content)
        )
        with context:
            list(_facade(case).stream_chat(
                **_chat_kwargs(case), json_schema=source_schema, on_done=completed.append,
            ))
        response, payload = completed[0], _assert_responses_stream_request(observed, case)
    elif method == "achat":
        transport = AsyncMock(return_value=_responses_structured_result(content))
        with patch.object(case.async_client_class, "_send_request", new=transport):
            response = asyncio.run(_facade(case).achat(
                **_chat_kwargs(case), json_schema=source_schema,
            ))
        payload = transport.await_args.args[1]
        _assert_responses_ordinary_request(transport.await_args.args[0], payload, case)
        transport.assert_awaited_once()
    else:
        completed = []
        context, observed = _responses_stream_request_probe(
            case.async_client_class, _responses_structured_events(content), asynchronous=True
        )
        async def run():
            async for _text in _facade(case).astream_chat(
                **_chat_kwargs(case), json_schema=source_schema, on_done=completed.append,
            ):
                pass
        with context:
            asyncio.run(run())
        response, payload = completed[0], _assert_responses_stream_request(observed, case)

    assert response.parsed_json == expected
    assert response.parsed_model is None
    assert _structured_schema_from_payload(case, payload) == original_schema
    assert source_schema == original_schema


@pytest.mark.unit
@pytest.mark.parametrize("method", ("chat", "achat", "stream_chat", "astream_chat"))
@pytest.mark.parametrize("invalid", (False, True), ids=("valid", "custom-validator-failure"))
def test_compatible_responses_validates_original_response_model(method, invalid):
    case = _compatible_responses_case()
    content = '{"contact":{"name":"Grace"}}' if invalid else NESTED_PYDANTIC_RESPONSE_JSON
    response = _responses_structured_result(content)
    events = _responses_structured_events(content)
    if method == "chat":
        transport = Mock(return_value=_JSONResponse(response))
        completed, visible, deltas, chunks, tool_callbacks = [], [], [], [], []
        with patch.object(case.sync_client_class, "_send_request", new=transport):
            if invalid:
                with pytest.raises(JSONSchemaError):
                    _facade(case).chat(**_chat_kwargs(case), response_model=_AdaOnlyNestedResponse)
            else:
                result = _facade(case).chat(
                    **_chat_kwargs(case), response_model=_AdaOnlyNestedResponse,
                )
        request_payload = transport.call_args.args[1]
        _assert_responses_ordinary_request(transport.call_args.args[0], request_payload, case)
        transport.assert_called_once()
    elif method == "stream_chat":
        completed, visible, deltas, chunks, tool_callbacks = [], [], [], [], []
        context, observed = _responses_stream_request_probe(case.sync_client_class, events)
        with context:
            stream = _facade(case).stream_chat(
                **_chat_kwargs(case), response_model=_AdaOnlyNestedResponse,
                buffer_chars=64, on_delta=deltas.append, on_chunk=chunks.append,
                on_tool_call=tool_callbacks.append, on_done=completed.append,
            )
            if invalid:
                with pytest.raises(JSONSchemaError):
                    visible.extend(stream)
            else:
                visible.extend(stream)
        request_payload = _assert_responses_stream_request(observed, case)
        if not invalid:
            result = completed[0]
    elif method == "achat":
        transport = AsyncMock(return_value=response)
        completed, visible, deltas, chunks, tool_callbacks = [], [], [], [], []
        with patch.object(case.async_client_class, "_send_request", new=transport):
            if invalid:
                with pytest.raises(JSONSchemaError):
                    asyncio.run(_facade(case).achat(
                        **_chat_kwargs(case), response_model=_AdaOnlyNestedResponse,
                    ))
            else:
                result = asyncio.run(_facade(case).achat(
                    **_chat_kwargs(case), response_model=_AdaOnlyNestedResponse,
                ))
        request_payload = transport.await_args.args[1]
        _assert_responses_ordinary_request(transport.await_args.args[0], request_payload, case)
        transport.assert_awaited_once()
    else:
        completed, visible, deltas, chunks, tool_callbacks = [], [], [], [], []
        context, observed = _responses_stream_request_probe(
            case.async_client_class, events, asynchronous=True
        )
        async def run():
            stream = _facade(case).astream_chat(
                **_chat_kwargs(case), response_model=_AdaOnlyNestedResponse,
                buffer_chars=64, on_delta=deltas.append, on_chunk=chunks.append,
                on_tool_call=tool_callbacks.append, on_done=completed.append,
            )
            if invalid:
                with pytest.raises(JSONSchemaError):
                    async for text in stream:
                        visible.append(text)
            else:
                async for text in stream:
                    visible.append(text)
        with context:
            asyncio.run(run())
        request_payload = _assert_responses_stream_request(observed, case)
        if not invalid:
            result = completed[0]

    schema = _structured_schema_from_payload(case, request_payload)
    assert "$defs" not in schema
    assert not _contains_reference(schema)
    if invalid:
        if method in ("stream_chat", "astream_chat"):
            assert visible == []
            assert deltas == []
            assert chunks == []
            assert tool_callbacks == []
            assert completed == []
        return
    assert result.parsed_json == json.loads(NESTED_PYDANTIC_RESPONSE_JSON)
    assert type(result.parsed_model) is _AdaOnlyNestedResponse
    assert result.parsed_model.contact.name == "Ada"


@pytest.mark.unit
@pytest.mark.parametrize("method", ("chat", "achat", "stream_chat", "astream_chat"))
@pytest.mark.parametrize("outcome", ("invalid-json", "invalid-model", "refusal", "incomplete"))
def test_compatible_responses_invalid_and_terminal_structured_results(method, outcome):
    case = _compatible_responses_case()
    if outcome == "invalid-json":
        content = '{"contact":'
        response = _responses_structured_result(content)
        events = _responses_structured_events(content)
        request = {"json_schema": PORTABLE_PROFILE_SCHEMAS["inline_nested_object"]}
    elif outcome == "invalid-model":
        content = '{"contact": {}}'
        response = _responses_structured_result(content)
        events = _responses_structured_events(content)
        request = {"response_model": NestedPydanticResponse}
    elif outcome == "refusal":
        response = responses_refusal_response(model=case.model)
        events = _decode_compatible_sse(
            event_lines({"type": "response.completed", "response": response})
        )
        request = {"response_model": NestedPydanticResponse}
    else:
        response = responses_incomplete_response(model=case.model)
        events = _decode_compatible_sse(
            event_lines({"type": "response.incomplete", "response": response})
        )
        request = {"response_model": NestedPydanticResponse}

    if method == "chat":
        transport = Mock(return_value=_JSONResponse(response))
        with patch.object(case.sync_client_class, "_send_request", new=transport):
            if outcome.startswith("invalid-"):
                with pytest.raises(JSONSchemaError):
                    _facade(case).chat(**_chat_kwargs(case), **request)
                result = None
            else:
                result = _facade(case).chat(**_chat_kwargs(case), **request)
        _assert_responses_ordinary_request(
            transport.call_args.args[0], transport.call_args.args[1], case
        )
        transport.assert_called_once()
    elif method == "stream_chat":
        completed, visible, deltas, chunks, tool_callbacks = [], [], [], [], []
        context, observed = _responses_stream_request_probe(case.sync_client_class, events)
        with context:
            stream = _facade(case).stream_chat(
                **_chat_kwargs(case), **request, buffer_chars=64,
                on_delta=deltas.append, on_chunk=chunks.append,
                on_tool_call=tool_callbacks.append, on_done=completed.append,
            )
            if outcome.startswith("invalid-"):
                with pytest.raises(JSONSchemaError):
                    visible.extend(stream)
                result = None
            else:
                visible.extend(stream)
                result = completed[0]
        _assert_responses_stream_request(observed, case)
    elif method == "achat":
        transport = AsyncMock(return_value=response)
        with patch.object(case.async_client_class, "_send_request", new=transport):
            if outcome.startswith("invalid-"):
                with pytest.raises(JSONSchemaError):
                    asyncio.run(_facade(case).achat(**_chat_kwargs(case), **request))
                result = None
            else:
                result = asyncio.run(_facade(case).achat(**_chat_kwargs(case), **request))
        _assert_responses_ordinary_request(
            transport.await_args.args[0], transport.await_args.args[1], case
        )
        transport.assert_awaited_once()
    else:
        completed, visible, deltas, chunks, tool_callbacks = [], [], [], [], []
        context, observed = _responses_stream_request_probe(
            case.async_client_class, events, asynchronous=True
        )
        async def run():
            stream = _facade(case).astream_chat(
                **_chat_kwargs(case), **request, buffer_chars=64,
                on_delta=deltas.append, on_chunk=chunks.append,
                on_tool_call=tool_callbacks.append, on_done=completed.append,
            )
            if outcome.startswith("invalid-"):
                with pytest.raises(JSONSchemaError):
                    async for text in stream:
                        visible.append(text)
                return None
            async for text in stream:
                visible.append(text)
            return completed[0]
        with context:
            result = asyncio.run(run())
        _assert_responses_stream_request(observed, case)

    if outcome.startswith("invalid-"):
        if method in ("stream_chat", "astream_chat"):
            assert visible == []
            assert deltas == []
            assert chunks == []
            assert tool_callbacks == []
            assert completed == []
        return
    assert result.parsed_json is None
    assert result.parsed_model is None
    if outcome == "refusal":
        assert result.refusal == "Fixture refusal"
        assert result.incomplete_reason is None
    else:
        assert result.incomplete_reason == "max_output_tokens"
        assert result.refusal is None


@pytest.mark.unit
@pytest.mark.parametrize("method", ("chat", "achat", "stream_chat", "astream_chat"))
@pytest.mark.parametrize("invalid_request", _COMPATIBLE_INVALID_REQUESTS)
def test_compatible_responses_invalid_portable_requests_fail_before_send(method, invalid_request):
    case = _compatible_responses_case()
    request = _compatible_invalid_request(invalid_request)
    original_request = deepcopy(request)
    send = Mock()
    if method == "chat":
        with patch.object(case.sync_client_class, "_send_request", new=send):
            with pytest.raises(JSONSchemaError):
                _facade(case).chat(**_chat_kwargs(case), **request)
    elif method == "stream_chat":
        with patch.object(case.sync_client_class, "_stream_request", new=send):
            with pytest.raises(JSONSchemaError):
                list(_facade(case).stream_chat(**_chat_kwargs(case), **request))
    elif method == "achat":
        with patch.object(case.async_client_class, "_send_request", new=send):
            with pytest.raises(JSONSchemaError):
                asyncio.run(_facade(case).achat(**_chat_kwargs(case), **request))
    else:
        with patch.object(case.async_client_class, "_stream_request", new=send):
            with pytest.raises(JSONSchemaError):
                async def run():
                    async for _text in _facade(case).astream_chat(
                        **_chat_kwargs(case), **request,
                    ):
                        pass
                asyncio.run(run())
    send.assert_not_called()
    assert request == original_request


def _facade_with_sync_transport(case, transport):
    return UniversalLLMAPIAdapter(
        organization=case.organization,
        model=case.model,
        api_key="test-api-key",
        transport=transport,
        **dict(case.connection_kwargs),
    )


def _execution_cases():
    return (CASES[0].values[0], CASES[3].values[0], CASES[4].values[0])


@pytest.mark.unit
@pytest.mark.parametrize(
    "case", _execution_cases(),
    ids=("native-openai", "compatible-chat", "compatible-responses"),
)
@pytest.mark.parametrize("method", ("chat", "achat", "stream_chat", "astream_chat"))
def test_neutral_execution_mixin_dispatches_each_facade_method(case, method, monkeypatch):
    """Exercise the original shared entry points and protocol hooks through the facade."""
    facade = _facade(case)
    adapter = facade.adapter
    assert isinstance(adapter, _OpenAIExecutionMixin)
    entry_calls = []
    entry_name = method
    original_entry = getattr(_OpenAIExecutionMixin, entry_name)

    if entry_name == "astream_chat":
        def instrument_entry(self, *args, **kwargs):
            entry_calls.append(entry_name)
            return original_entry(self, *args, **kwargs)
    elif entry_name == "achat":
        async def instrument_entry(self, *args, **kwargs):
            entry_calls.append(entry_name)
            return await original_entry(self, *args, **kwargs)
    else:
        def instrument_entry(self, *args, **kwargs):
            entry_calls.append(entry_name)
            return original_entry(self, *args, **kwargs)

    monkeypatch.setattr(_OpenAIExecutionMixin, entry_name, instrument_entry)
    if method == "astream_chat":
        original_async_dispatch = _OpenAIExecutionMixin._astream_chat

        async def instrument_async_dispatch(self, *args, **kwargs):
            entry_calls.append("_astream_chat")
            async for text in original_async_dispatch(self, *args, **kwargs):
                yield text

        monkeypatch.setattr(_OpenAIExecutionMixin, "_astream_chat", instrument_async_dispatch)

    hook_name = "_build_chat_params" if method in ("chat", "achat") else "_build_stream_params"
    hook = Mock(wraps=getattr(adapter, hook_name))
    setattr(adapter, hook_name, hook)
    observed = []

    def sync_send(client, url, payload, timeout=None):
        observed.append((url, deepcopy(payload), client))
        return _JSONResponse(case.response_factory())

    async def async_send(client, url, payload, timeout=None):
        observed.append((url, deepcopy(payload), client))
        return case.response_factory()

    def sync_stream(client, url, payload, timeout=None):
        observed.append((url, deepcopy(payload), client))
        return iter(case.stream_events_factory())

    def async_stream(client, url, payload, timeout=None):
        observed.append((url, deepcopy(payload), client))
        return _as_async_iter(case.stream_events_factory())

    if method == "chat":
        with patch.object(case.sync_client_class, "_send_request", new=sync_send):
            result = facade.chat(**_chat_kwargs(case))
    elif method == "achat":
        with patch.object(case.async_client_class, "_send_request", new=async_send):
            result = asyncio.run(facade.achat(**_chat_kwargs(case)))
    elif method == "stream_chat":
        with patch.object(case.sync_client_class, "_stream_request", new=sync_stream):
            completed = []
            list(facade.stream_chat(**_chat_kwargs(case), on_done=completed.append))
            result = completed[0]
    else:
        with patch.object(case.async_client_class, "_stream_request", new=async_stream):
            async def run():
                completed = []
                async for _text in facade.astream_chat(
                    **_chat_kwargs(case), on_done=completed.append,
                ):
                    pass
                return completed[0]
            result = asyncio.run(run())

    assert entry_calls == (["astream_chat", "_astream_chat"] if method == "astream_chat" else [method])
    assert hook.call_count == 1
    assert len(observed) == 1
    url, payload, _client = observed[0]
    expected_resource = "responses" if case.wire_format == "openai_responses" else "chat/completions"
    assert url.endswith("/" + expected_resource)
    assert payload["model"] == case.model
    assert payload[case.message_key]
    if case.organization == "openai":
        assert "api_variant" not in dict(case.connection_kwargs)
        assert payload["input"]
    assert result.content == ("Hello!" if method in ("stream_chat", "astream_chat") else "ok")


_COMPATIBLE_EXECUTION_LANES = (
    ("chat", "requests"),
    ("chat", "httpx"),
    ("stream_chat", "requests"),
    ("stream_chat", "httpx"),
    ("achat", "httpx"),
    ("astream_chat", "httpx"),
)


def _events_as_sse_body(events: list[SSEEvent]) -> str:
    lines = []
    for event in events:
        if event.done:
            lines.append("data: [DONE]\r\n\r\n")
            continue
        if event.event is not None:
            lines.append(f"event: {event.event}\r\n")
        lines.append(f"data: {json.dumps(event.data, separators=(',', ':'))}\r\n\r\n")
    return "".join(lines)


@pytest.mark.unit
@pytest.mark.parametrize("case", (CASES[3].values[0], CASES[4].values[0]), ids=("chat", "responses"))
@pytest.mark.parametrize("lane", _COMPATIBLE_EXECUTION_LANES, ids=lambda lane: f"{lane[0]}-{lane[1]}")
@pytest.mark.parametrize("caller_pricing", (False, True), ids=("unpriced", "caller-priced"))
def test_compatible_api_transport_and_caller_pricing_matrix(case, lane, caller_pricing):
    method, transport_name = lane
    if caller_pricing:
        facade, pricing = _caller_pricing_facade(case)
        facade.adapter.transport = transport_name
    else:
        facade = _facade_with_sync_transport(case, transport_name)

    messages = [UserMessage("Hello")]
    original_messages = deepcopy(messages)
    observed = []
    completed = []
    callback_order = []
    chunks = []
    visible = []
    url = f"{dict(case.connection_kwargs)['base_url'].rstrip('/')}/"
    url += "responses" if case.wire_format == "openai_responses" else "chat/completions"
    is_stream = method in ("stream_chat", "astream_chat")
    response_payload = case.response_factory()
    stream_body = _events_as_sse_body(case.stream_events_factory())

    def record_chunk(chunk):
        chunks.append(chunk)
        callback_order.append("chunk")

    def record_done(response):
        completed.append(response)
        callback_order.append("done")

    def invoke():
        if method == "chat":
            return facade.chat(messages=messages, max_tokens=case.max_tokens)
        if method == "achat":
            return asyncio.run(facade.achat(messages=messages, max_tokens=case.max_tokens))
        if method == "stream_chat":
            visible.extend(facade.stream_chat(
                messages=messages, max_tokens=case.max_tokens, buffer_chars=64,
                on_chunk=record_chunk, on_done=record_done,
            ))
        else:
            async def run():
                async for text in facade.astream_chat(
                    messages=messages, max_tokens=case.max_tokens, buffer_chars=64,
                    on_chunk=record_chunk, on_done=record_done,
                ):
                    visible.append(text)
            asyncio.run(run())
        return completed[0]

    if transport_name == "requests":
        with requests_mock.Mocker(real_http=False) as http:
            if is_stream:
                http.post(url, text=stream_body, headers={"Content-Type": "text/event-stream"})
            else:
                http.post(url, json=response_payload)
            result = invoke()
            assert http.call_count == 1
            request = http.request_history[0]
            payload = request.json()
            observed.append(request)
    else:
        httpx = importlib.import_module("httpx")

        def handler(request):
            observed.append(request)
            if is_stream:
                return httpx.Response(
                    200, text=stream_body, headers={"Content-Type": "text/event-stream"},
                )
            return httpx.Response(200, json=response_payload)

        mock_transport = httpx.MockTransport(handler)
        if method in ("chat", "stream_chat"):
            client = httpx.Client(transport=mock_transport)
            with patch.object(httpx, "Client", return_value=client):
                result = invoke()
        else:
            client = httpx.AsyncClient(transport=mock_transport)
            with patch.object(httpx, "AsyncClient", return_value=client):
                result = invoke()
        assert len(observed) == 1
        request = observed[0]
        payload = json.loads(request.content)

    assert messages == original_messages
    assert len(observed) == 1
    assert request.method == "POST"
    assert str(request.url) == url
    assert request.headers["Authorization"] == "Bearer test-api-key"
    assert payload["model"] == case.model
    assert payload[case.message_key] == [{"role": "user", "content": "Hello"}]
    if case.wire_format == "openai_responses":
        assert payload["max_output_tokens"] == case.max_tokens
        assert payload["store"] is False
        assert "max_tokens" not in payload
        assert "max_completion_tokens" not in payload
        assert "stream_options" not in payload
        if is_stream:
            assert payload["stream"] is True
        else:
            assert not payload.get("stream", False)
    else:
        assert payload["max_completion_tokens"] == case.max_tokens
        assert "max_tokens" not in payload
        if is_stream:
            assert payload["stream"] is True
            assert payload["stream_options"] == {"include_usage": True}
        else:
            assert not payload.get("stream", False)
            assert "stream_options" not in payload
    assert result.usage == case.expected_usage
    assert result.content == ("Hello!" if is_stream else "ok")
    if is_stream:
        assert "".join(visible) == "Hello!"
    assert facade.adapter.model_spec is None
    if caller_pricing:
        assert result.cost_input == pytest.approx(20 / 1_000_000)
        assert result.cost_output == pytest.approx(60 / 1_000_000)
        assert result.cost_total == pytest.approx(80 / 1_000_000)
        assert result.currency == "EUR"
        assert result.cost_breakdown is None
        assert facade.adapter.pricing is pricing
    else:
        _assert_usage_and_pricing(result, case)
    if is_stream:
        assert len(completed) == 1
        assert completed[0] is result
        assert callback_order == ["chunk", "done"]
        assert "".join(chunk.text for chunk in chunks) == "Hello!"
        assert chunks[-1].usage == case.expected_usage


@pytest.mark.unit
@pytest.mark.parametrize("method", ("chat", "stream_chat"))
def test_compatible_responses_requests_transport_works_without_httpx(method, monkeypatch):
    case = CASES[4].values[0]
    facade = _facade_with_sync_transport(case, "requests")
    url = f"{dict(case.connection_kwargs)['base_url'].rstrip('/')}/responses"
    body = (
        json.dumps(case.response_factory())
        if method == "chat"
        else _events_as_sse_body(case.stream_events_factory())
    )
    forbidden = []
    try:
        httpx = importlib.import_module("httpx")
    except ImportError:
        httpx = None

    def forbid_httpx(*args, **kwargs):
        forbidden.append("httpx construction")
        raise AssertionError("Requests transport attempted to construct HTTPX")

    def forbid_dns(*args, **kwargs):
        forbidden.append("DNS")
        raise AssertionError("Mocked requests transport attempted DNS")

    if httpx is not None:
        monkeypatch.setattr(httpx, "Client", forbid_httpx)
        monkeypatch.setattr(httpx, "AsyncClient", forbid_httpx)
    monkeypatch.setattr(socket, "getaddrinfo", forbid_dns)
    with requests_mock.Mocker(real_http=False) as http, patch.dict(sys.modules, {"httpx": None}):
        http.post(
            url,
            text=body,
            headers={
                "Content-Type": "application/json" if method == "chat" else "text/event-stream",
            },
        )
        if method == "chat":
            result = facade.chat(**_chat_kwargs(case))
        else:
            completed = []
            list(facade.stream_chat(**_chat_kwargs(case), on_done=completed.append))
            result = completed[0]
        assert http.call_count == 1
        assert http.request_history[0].url == url
    assert result.content == ("ok" if method == "chat" else "Hello!")
    assert forbidden == []


@pytest.mark.unit
@pytest.mark.parametrize(
    ("method", "extra"),
    (
        ("chat", "httpx"),
        ("stream_chat", "httpx"),
        ("achat", "async"),
        ("astream_chat", "async"),
    ),
)
def test_compatible_responses_optional_httpx_failures_precede_io(method, extra, monkeypatch):
    case = CASES[4].values[0]
    facade = _facade_with_sync_transport(case, "httpx")
    url = f"{dict(case.connection_kwargs)['base_url'].rstrip('/')}/responses"
    attempts = []

    def forbid_io(name):
        def forbidden(*args, **kwargs):
            attempts.append(name)
            raise AssertionError(f"Unexpected outbound attempt through {name}")
        return forbidden

    try:
        httpx = importlib.import_module("httpx")
    except ImportError:
        httpx = None
    if httpx is not None:
        monkeypatch.setattr(httpx.Client, "send", forbid_io("httpx sync send"))
        monkeypatch.setattr(httpx.AsyncClient, "send", forbid_io("httpx async send"))
        monkeypatch.setattr(httpx, "Client", forbid_io("httpx sync constructor"))
        monkeypatch.setattr(httpx, "AsyncClient", forbid_io("httpx async constructor"))
    monkeypatch.setattr(socket, "getaddrinfo", forbid_io("DNS"))

    with requests_mock.Mocker(real_http=False) as http, patch.dict(sys.modules, {"httpx": None}):
        http.post(url, status_code=599)
        with pytest.raises(ImportError) as error:
            if method == "chat":
                facade.chat(**_chat_kwargs(case))
            elif method == "stream_chat":
                list(facade.stream_chat(**_chat_kwargs(case)))
            elif method == "achat":
                asyncio.run(facade.achat(**_chat_kwargs(case)))
            else:
                async def run():
                    async for _text in facade.astream_chat(**_chat_kwargs(case)):
                        pass
                asyncio.run(run())
        assert http.call_count == 0
    assert f"llm-api-adapter[{extra}]" in str(error.value)
    assert attempts == []
