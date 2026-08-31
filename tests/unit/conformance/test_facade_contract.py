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
    sync_client_class: type[Any]
    async_client_class: type[Any]
    message_key: str
    response_factory: Callable[..., dict[str, Any]]
    stream_events_factory: Callable[[], list[SSEEvent]]


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
        "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
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
            sync_client_class=OpenAISyncClient,
            async_client_class=OpenAIAsyncClient,
            message_key="input",
            response_factory=_openai_response,
            stream_events_factory=_openai_stream_events,
        ),
        id="openai",
    ),
    pytest.param(
        OrganizationConformanceCase(
            organization="anthropic",
            model="claude-sonnet-4-5",
            max_tokens=64,
            sync_client_class=ClaudeSyncClient,
            async_client_class=ClaudeAsyncClient,
            message_key="messages",
            response_factory=_anthropic_response,
            stream_events_factory=_anthropic_stream_events,
        ),
        id="anthropic",
    ),
    pytest.param(
        OrganizationConformanceCase(
            organization="google",
            model="gemini-2.5-flash",
            max_tokens=None,
            sync_client_class=GeminiSyncClient,
            async_client_class=GeminiAsyncClient,
            message_key="contents",
            response_factory=_google_response,
            stream_events_factory=_google_stream_events,
        ),
        id="google",
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
    """Return the effective schema from each built-in organization's payload."""
    if case.organization == "openai":
        return payload["text"]["format"]["schema"]
    if case.organization == "anthropic":
        return payload["output_config"]["format"]["schema"]
    if case.organization == "google":
        return payload["generationConfig"]["responseSchema"]
    raise AssertionError(f"No structured-output payload path for {case.organization!r}")


def _contains_reference(node: Any) -> bool:
    if isinstance(node, dict):
        return "$ref" in node or any(_contains_reference(value) for value in node.values())
    if isinstance(node, list):
        return any(_contains_reference(value) for value in node)
    return False


def _google_wire_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Apply only Google's type-value casing to a portable source schema."""
    converted = deepcopy(schema)

    def convert(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                convert(item)
            return
        if not isinstance(node, dict):
            return
        schema_type = node.get("type")
        if isinstance(schema_type, str):
            node["type"] = schema_type.upper()
        elif isinstance(schema_type, list):
            node["type"] = [value.upper() for value in schema_type]
        for value in node.values():
            convert(value)

    convert(converted)
    return converted


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
    if case.organization == "openai":
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

    if case.organization == "anthropic":
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

    if case.organization == "google":
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

    raise AssertionError(f"No terminal-outcome fixture for {case.organization!r}")


def _terminal_outcome_stream_events(
    case: OrganizationConformanceCase,
    outcome_name: str,
    content: str | None,
) -> list[SSEEvent]:
    response = _terminal_outcome_response(case, outcome_name, content)
    if case.organization == "openai":
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

    if case.organization == "anthropic":
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

    if case.organization == "google":
        return [SSEEvent(event=None, data=response)]

    raise AssertionError(f"No terminal stream fixture for {case.organization!r}")


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
        expected_refusals = {
            "openai": "I can't help with that.",
            "anthropic": "safety",
            "google": "Blocked by safety policy.",
        }
        assert response.refusal == expected_refusals[case.organization]
        assert response.incomplete_reason is None
    elif outcome_name == "incomplete_result":
        expected_reasons = {
            "openai": "max_output_tokens",
            "anthropic": "max_tokens",
            "google": "MAX_TOKENS",
        }
        assert response.incomplete_reason == expected_reasons[case.organization]
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

    with patch.object(case.sync_client_class, "_send_request", new=transport):
        response = _facade(case).chat(**_chat_kwargs(case))

    payload = transport.call_args.args[1]
    assert payload["model"] == case.model
    assert payload[case.message_key]
    assert response.content == "ok"
    assert response.usage == Usage(input_tokens=2, output_tokens=3, total_tokens=5)
    assert response.currency == "USD"
    assert response.cost_total is not None


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
    assert response.usage == Usage(input_tokens=2, output_tokens=3, total_tokens=5)
    assert response.cost_total is not None


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

    assert [event.text for event in reasoning_response.reasoning_events] == ["Plan"]


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
        if case.organization == "google"
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
