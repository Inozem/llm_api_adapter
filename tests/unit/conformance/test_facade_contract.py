"""Provider-neutral conformance checks through the public facade.

These tests deliberately exercise only ``UniversalLLMAPIAdapter``.  Provider
clients are mocked at their network boundary so the suite freezes the common
contract without credentials or live API calls.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable
from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import BaseModel, ConfigDict

from src.llm_api_adapter.errors.llm_api_error import (
    JSONSchemaError,
    LLMAPIRateLimitError,
)
from src.llm_api_adapter.llms.anthropic.async_client import ClaudeAsyncClient
from src.llm_api_adapter.llms.anthropic.sync_client import ClaudeSyncClient
from src.llm_api_adapter.llms.google.async_client import GeminiAsyncClient
from src.llm_api_adapter.llms.google.sync_client import GeminiSyncClient
from src.llm_api_adapter.llms.openai.async_client import OpenAIAsyncClient
from src.llm_api_adapter.llms.openai.sync_client import OpenAISyncClient
from src.llm_api_adapter.llms.streaming import SSEEvent
from src.llm_api_adapter.llms.transports import SSEFrameDecoder
from src.llm_api_adapter.models.messages.chat_message import UserMessage
from src.llm_api_adapter.models.messages.file_parts import ImagePart
from src.llm_api_adapter.models.responses.chat_response import Usage
from src.llm_api_adapter.models.tools import ToolSpec
from src.llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter
from tests.fixtures.structured_output import (
    FLAT_OBJECT_SCHEMA,
    NESTED_PYDANTIC_RESPONSE_JSON,
    NestedPydanticResponse,
    PORTABLE_PROFILE_SCHEMAS,
    STRUCTURED_OUTPUT_OUTCOMES,
)
from tests.fixtures.openai_compatible import (
    FIXTURE_MODEL,
    incomplete_response,
    incomplete_sse_lines,
    refusal_response,
    refusal_sse_lines,
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
        "modelVersion": "gemini-2.5-flash",
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
            model="gemini-2.5-flash",
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
            sync_client_class=OpenAISyncClient,
            async_client_class=OpenAIAsyncClient,
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
