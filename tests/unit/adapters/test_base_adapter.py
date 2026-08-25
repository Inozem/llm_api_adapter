from types import SimpleNamespace
from inspect import signature
from unittest.mock import patch
import warnings

import pytest
from pydantic import BaseModel

from src.llm_api_adapter.adapters.base_adapter import LLMAdapterBase
from src.llm_api_adapter.adapters import base_adapter as base_module
from src.llm_api_adapter.errors.llm_api_error import (
    InvalidToolSchemaError,
    JSONSchemaError,
    LLMAPIError,
    ToolChoiceError,
)
from src.llm_api_adapter.models.messages.chat_message import (
    Messages,
    Prompt,
    UserMessage,
)
from src.llm_api_adapter.models.responses.chat_response import ChatResponse, Usage
from src.llm_api_adapter.models.responses.reasoning_event import ReasoningEvent
from src.llm_api_adapter.llm_registry.llm_registry import Pricing
from src.llm_api_adapter.llms.streaming import (
    StreamChunkBuffer,
    StreamReasoningCollector,
)
from src.llm_api_adapter.models.tools import ToolCall, ToolSpec


class DummyClient:
    def chat_completion(self, messages, **kwargs):
        return {"choices": [{"message": {"content": "dummy response"}}]}


class _TestAdapter(LLMAdapterBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = DummyClient()

    def chat(self, messages, **kwargs):
        try:
            raw_response = self.client.chat_completion(messages, **kwargs)
            try:
                return ChatResponse(**raw_response)
            except Exception:
                return raw_response
        except LLMAPIError as e:
            return self.handle_error(e)
        except Exception as e:
            return self.handle_error(e)

    def stream_chat(self, *args, **kwargs):
        raise NotImplementedError


@pytest.fixture
def adapter():
    adapter_instance = _TestAdapter(
        company="openai",
        api_key="dummy_key",
        model="gpt-5",
    )
    adapter_instance.client = DummyClient()
    return adapter_instance


def make_tool(name="weather", schema=None, description="desc"):
    if schema is None:
        schema = {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
            },
        }
    return ToolSpec(name=name, description=description, json_schema=schema)


@pytest.mark.unit
def test_snapshot_model_uses_its_registered_base_metadata_without_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        adapter = _TestAdapter(
            company="openai",
            api_key="dummy_key",
            model="gpt-5-2025-08-07",
        )

    assert caught == []
    assert adapter.model == "gpt-5-2025-08-07"
    assert adapter.model_spec is not None
    assert adapter.model_spec.name == "gpt-5"
    assert adapter.pricing is not None
    assert adapter.is_reasoning is True


@pytest.mark.parametrize(
    "temperature,max_tokens,top_p,valid",
    [
        (1.0, 256, 1.0, True),
        (-0.1, 256, 1.0, False),
        (2.1, 256, 1.0, False),
        (0.0, 256, 1.0, True),
        (1.0, 256, -0.1, False),
        (1.0, 256, 1.1, False),
    ],
)
@pytest.mark.unit
def test_parameter_validation(adapter, temperature, max_tokens, top_p, valid):
    if valid:
        temp_result = adapter._validate_parameter("temperature", temperature, 0, 2)
        top_p_result = adapter._validate_parameter("top_p", top_p, 0, 1)
        assert temp_result == temperature
        assert top_p_result == top_p
    else:
        if temperature < 0 or temperature > 2:
            with pytest.raises(ValueError):
                adapter._validate_parameter("temperature", temperature, 0, 2)
        if top_p < 0 or top_p > 1:
            with pytest.raises(ValueError):
                adapter._validate_parameter("top_p", top_p, 0, 1)


@pytest.mark.unit
def test_chat_handles_llmapi_error(adapter):
    messages = [Prompt("system prompt"), UserMessage("hello")]
    with patch.object(
        DummyClient, "chat_completion", side_effect=LLMAPIError("API error")
    ), patch.object(adapter, "handle_error") as mock_handle_error:
        adapter.chat(messages)
        mock_handle_error.assert_called_once()


@pytest.mark.unit
def test_chat_handles_generic_exception(adapter):
    messages = [Prompt("system prompt"), UserMessage("hello")]
    with patch.object(
        DummyClient, "chat_completion", side_effect=Exception("Generic error")
    ), patch.object(adapter, "handle_error") as mock_handle_error:
        adapter.chat(messages)
        mock_handle_error.assert_called_once()


@pytest.mark.unit
def test_init_with_empty_api_key_raises():
    with pytest.raises(ValueError):
        _TestAdapter(company="any", api_key="", model="m1")


@pytest.mark.unit
def test_unverified_model_warns_and_leaves_pricing_none(monkeypatch):
    monkeypatch.setattr(
        base_module,
        "LLM_REGISTRY",
        SimpleNamespace(organizations={}),
        raising=False,
    )
    with pytest.warns(UserWarning):
        a = _TestAdapter(company="missing", api_key="k", model="unknown-model")
    assert a.pricing is None


@pytest.mark.unit
def test_pricing_copied_from_registry(monkeypatch):
    base_pricing = Pricing.from_dict(
        [
            {
                "up_to_prompt_tokens": None,
                "input_per_1m": 1.0,
                "output_per_1m": 2.0,
            }
        ],
        currency="USD",
    )
    organization = SimpleNamespace(
        models={
            "m-pro": SimpleNamespace(
                pricing_tiers=base_pricing,
                is_reasoning=False,
            )
        }
    )
    monkeypatch.setattr(
        base_module,
        "LLM_REGISTRY",
        SimpleNamespace(organizations={"acme": organization}),
        raising=False,
    )
    adapter_instance = _TestAdapter(company="acme", api_key="k", model="m-pro")
    assert adapter_instance.pricing == base_pricing
    assert adapter_instance.pricing is not base_pricing


@pytest.mark.unit
@pytest.mark.parametrize(
    ("input_tokens", "expected_cost_input", "expected_cost_output"),
    [
        (200, 0.0002, 0.000402),
        (201, 0.000603, 0.000804),
    ],
)
def test_finalize_chat_response_resolves_pricing_tier_from_usage(
    adapter,
    input_tokens,
    expected_cost_input,
    expected_cost_output,
):
    adapter.pricing = Pricing.from_dict(
        [
            {
                "up_to_prompt_tokens": 200,
                "input_per_1m": 1,
                "output_per_1m": 2,
            },
            {
                "up_to_prompt_tokens": None,
                "input_per_1m": 3,
                "output_per_1m": 4,
            },
        ],
        currency="USD",
    )
    response = ChatResponse(usage=Usage(input_tokens=input_tokens, output_tokens=201))

    adapter._finalize_chat_response(
        response,
        effective_schema=None,
        response_model=None,
    )

    assert response.cost_input == pytest.approx(expected_cost_input)
    assert response.cost_output == pytest.approx(expected_cost_output)
    assert response.cost_total == pytest.approx(
        expected_cost_input + expected_cost_output
    )


@pytest.mark.unit
def test_post_init_sets_reasoning_flag_from_registry(monkeypatch):
    organization = SimpleNamespace(
        models={"m-reason": SimpleNamespace(pricing=None, is_reasoning=True, is_adaptive_thinking=False)}
    )
    monkeypatch.setattr(
        base_module,
        "LLM_REGISTRY",
        SimpleNamespace(organizations={"acme": organization}),
        raising=False,
    )
    adapter_instance = _TestAdapter(company="acme", api_key="k", model="m-reason")
    assert adapter_instance.is_reasoning is True


@pytest.mark.unit
def test_post_init_sets_adaptive_thinking_flag_from_registry(monkeypatch):
    organization = SimpleNamespace(
        models={"m-adaptive": SimpleNamespace(pricing=None, is_reasoning=True, is_adaptive_thinking=True)}
    )
    monkeypatch.setattr(
        base_module,
        "LLM_REGISTRY",
        SimpleNamespace(organizations={"acme": organization}),
        raising=False,
    )
    adapter_instance = _TestAdapter(company="acme", api_key="k", model="m-adaptive")
    assert adapter_instance.is_adaptive_thinking is True


@pytest.mark.unit
def test_normalize_messages_accepts_list_and_messages(adapter):
    raw_messages = [UserMessage("hi")]
    normalized = adapter._normalize_messages(raw_messages)
    assert isinstance(normalized, Messages)

    same = adapter._normalize_messages(normalized)
    assert same is normalized

    with pytest.raises(TypeError):
        adapter._normalize_messages(123)


@pytest.mark.unit
def test_prepare_chat_request_normalizes_provider_neutral_context(adapter):
    tool = make_tool()

    context = adapter._prepare_chat_request(
        [UserMessage("hi")],
        [tool],
        "auto",
        None,
        None,
    )

    assert isinstance(context.normalized_messages, Messages)
    assert context.normalized_messages.items[0] == UserMessage("hi")
    assert context.effective_schema is None
    assert context.normalized_tool_choice == "auto"


@pytest.mark.unit
def test_generate_chat_answer_emits_deprecation_warning(adapter):
    with pytest.warns(DeprecationWarning):
        adapter.generate_chat_answer(messages=[UserMessage("hi")])


@pytest.mark.unit
def test_handle_error_reraises_and_logs(adapter, caplog):
    with pytest.raises(ValueError, match="boom"):
        try:
            raise ValueError("boom")
        except ValueError as err:
            adapter.handle_error(err, "some message")

    assert adapter.company in caplog.text
    assert adapter.model in caplog.text
    assert "some message" in caplog.text


@pytest.mark.unit
def test_base_abstract_chat_method_raises_not_implemented(adapter):
    with pytest.raises(NotImplementedError):
        LLMAdapterBase.chat(adapter, messages=[UserMessage("hello")])


@pytest.mark.unit
def test_base_stream_chat_contract_raises_not_implemented(adapter):
    with pytest.raises(NotImplementedError):
        LLMAdapterBase.stream_chat(
            adapter,
            messages=[UserMessage("hello")],
            on_delta=lambda delta: None,
        )


@pytest.mark.unit
def test_reasoning_contract_is_exposed_by_all_provider_adapters():
    from src.llm_api_adapter.adapters.anthropic_adapter import AnthropicAdapter
    from src.llm_api_adapter.adapters.google_adapter import GoogleAdapter
    from src.llm_api_adapter.adapters.openai_adapter import OpenAIAdapter

    for adapter_class in (OpenAIAdapter, AnthropicAdapter, GoogleAdapter):
        chat_params = signature(adapter_class.chat).parameters
        stream_params = signature(adapter_class.stream_chat).parameters
        assert chat_params["capture_reasoning"].default is False
        assert stream_params["capture_reasoning"].default is False
        assert stream_params["on_reasoning"].default is None


@pytest.mark.unit
def test_record_reasoning_event_updates_response_before_callback(adapter):
    response = ChatResponse()
    collector = StreamReasoningCollector(clock=iter([0.0, 0.25]).__next__)
    observed = []

    def on_reasoning(event):
        observed.append((event, list(response.reasoning_events)))

    event = adapter._record_reasoning_event(
        response,
        collector,
        "visible summary",
        capture_reasoning=True,
        on_reasoning=on_reasoning,
    )

    expected = ReasoningEvent("visible summary", "summary", 0, 0.25, 0.25)
    assert event == expected
    assert response.reasoning_events == [expected]
    assert collector.snapshot() == [expected]
    assert observed == [(expected, [expected])]

    completed = []
    adapter._invoke_stream_completion_callbacks(response, None, completed.append)
    assert completed == [response]
    assert completed[0].reasoning_events == [expected]


@pytest.mark.unit
def test_complete_stream_flushes_chunks_before_completion_callbacks(adapter):
    response = ChatResponse(
        content="Hello",
        tool_calls=[ToolCall(name="weather", arguments={"city": "Tel Aviv"})],
    )
    chunk_buffer = StreamChunkBuffer(buffer_chars=10)
    list(chunk_buffer.add("Hello"))
    events = []

    output = list(
        adapter._complete_stream(
            response,
            chunk_buffer,
            on_chunk=lambda chunk: events.append(("chunk", chunk.text)),
            on_delta=lambda text: events.append(("delta", text)),
            on_tool_call=lambda call: events.append(("tool", call.name)),
            on_done=lambda completed: events.append(("done", completed.content)),
        )
    )

    assert output == ["Hello"]
    assert events == [
        ("chunk", "Hello"),
        ("delta", "Hello"),
        ("tool", "weather"),
        ("done", "Hello"),
    ]


@pytest.mark.unit
def test_finalize_stream_response_attaches_reasoning_and_structured_output(adapter):
    class StreamAnswer(BaseModel):
        answer: str

    response = ChatResponse(content='{"answer": "Hello"}')
    collector = StreamReasoningCollector(clock=iter([0.0, 0.25]).__next__)
    event = collector.add("Plan")

    finalized = adapter._finalize_stream_response(
        response,
        reasoning_collector=collector,
        effective_schema={"type": "object"},
        response_model=StreamAnswer,
    )

    assert finalized is response
    assert finalized.reasoning_events == [event]
    assert finalized.parsed_json == {"answer": "Hello"}
    assert finalized.parsed_model == StreamAnswer(answer="Hello")


@pytest.mark.unit
def test_record_reasoning_event_is_opt_in_and_does_not_call_callback(adapter):
    response = ChatResponse()
    collector = StreamReasoningCollector(clock=iter([0.0, 0.25]).__next__)
    callbacks = []

    assert adapter._record_reasoning_event(
        response,
        collector,
        "not captured",
        capture_reasoning=False,
        on_reasoning=callbacks.append,
    ) is None

    assert response.reasoning_events == []
    assert collector.snapshot() == []
    assert callbacks == []


@pytest.mark.unit
def test_reasoning_callback_exception_preserves_existing_callback_semantics(adapter):
    response = ChatResponse()
    collector = StreamReasoningCollector(clock=iter([0.0, 0.25]).__next__)

    with pytest.raises(RuntimeError, match="callback failed"):
        adapter._record_reasoning_event(
            response,
            collector,
            "captured before failure",
            capture_reasoning=True,
            on_reasoning=lambda event: (_ for _ in ()).throw(
                RuntimeError("callback failed")
            ),
        )

    assert len(response.reasoning_events) == 1
    assert response.reasoning_events[0].text == "captured before failure"


# ---------------------------
# _validate_tools
# ---------------------------

@pytest.mark.unit
def test_validate_tools_accepts_none(adapter):
    adapter._validate_tools(None)


@pytest.mark.unit
def test_validate_tools_rejects_non_list(adapter):
    with pytest.raises(InvalidToolSchemaError, match="tools must be a list\\[ToolSpec\\] or None"):
        adapter._validate_tools("not-a-list")


@pytest.mark.unit
def test_validate_tools_rejects_non_toolspec_items(adapter):
    with pytest.raises(InvalidToolSchemaError, match="tools must contain ToolSpec items only"):
        adapter._validate_tools([SimpleNamespace(name="x", json_schema={})])


@pytest.mark.unit
def test_validate_tools_rejects_empty_name(adapter):
    tool = make_tool("ok_name")
    object.__setattr__(tool, "name", "")
    with pytest.raises(
        InvalidToolSchemaError,
        match="ToolSpec.name must be a non-empty string",
    ):
        adapter._validate_tools([tool])


@pytest.mark.unit
def test_validate_tools_rejects_non_string_name(adapter):
    tool = make_tool("ok_name")
    object.__setattr__(tool, "name", 123)
    with pytest.raises(
        InvalidToolSchemaError,
        match="ToolSpec.name must be a non-empty string",
    ):
        adapter._validate_tools([tool])


@pytest.mark.unit
def test_validate_tools_rejects_invalid_name_format(adapter):
    tool = make_tool("bad name")
    with pytest.raises(InvalidToolSchemaError, match="Invalid tool name"):
        adapter._validate_tools([tool])


@pytest.mark.unit
def test_validate_tools_rejects_duplicate_names(adapter):
    tools = [make_tool("weather"), make_tool("weather")]
    with pytest.raises(InvalidToolSchemaError, match="Duplicate tool name"):
        adapter._validate_tools(tools)


@pytest.mark.unit
def test_validate_tools_rejects_non_dict_json_schema(adapter):
    tool = make_tool("weather")
    object.__setattr__(tool, "json_schema", "not-a-dict")
    with pytest.raises(
        InvalidToolSchemaError,
        match="json_schema must be a dict",
    ):
        adapter._validate_tools([tool])


@pytest.mark.unit
def test_validate_tools_accepts_valid_tools(adapter):
    tools = [make_tool("weather"), make_tool("search")]
    adapter._validate_tools(tools)


# ---------------------------
# _normalize_tool_choice
# ---------------------------

@pytest.mark.unit
def test_normalize_tool_choice_none_returns_none(adapter):
    assert adapter._normalize_tool_choice(None, None) is None


@pytest.mark.unit
@pytest.mark.parametrize("value", ["auto", "none"])
def test_normalize_tool_choice_str_auto_none(adapter, value):
    tools = [make_tool("weather")]
    assert adapter._normalize_tool_choice(value, tools) == value


@pytest.mark.unit
def test_normalize_tool_choice_str_required_raises(adapter):
    with pytest.raises(ToolChoiceError, match="tool_choice='required' is not supported; use 'any'"):
        adapter._normalize_tool_choice("required", [make_tool("weather")])


@pytest.mark.unit
def test_normalize_tool_choice_str_any_requires_tools(adapter):
    with pytest.raises(ToolChoiceError, match="tool_choice='any' requires tools to be provided"):
        adapter._normalize_tool_choice("any", None)


@pytest.mark.unit
def test_normalize_tool_choice_str_any_ok(adapter):
    assert adapter._normalize_tool_choice("any", [make_tool("weather")]) == "any"


@pytest.mark.unit
def test_normalize_tool_choice_str_named_tool_ok(adapter):
    tools = [make_tool("weather"), make_tool("search")]
    assert adapter._normalize_tool_choice("weather", tools) == "weather"


@pytest.mark.unit
def test_normalize_tool_choice_str_named_tool_requires_tools(adapter):
    with pytest.raises(ToolChoiceError, match="tool_choice references a tool but tools=None"):
        adapter._normalize_tool_choice("weather", None)


@pytest.mark.unit
def test_normalize_tool_choice_str_unknown_tool_raises(adapter):
    with pytest.raises(ToolChoiceError, match="Unknown tool_choice string"):
        adapter._normalize_tool_choice("missing_tool", [make_tool("weather")])


@pytest.mark.unit
def test_normalize_tool_choice_invalid_type_raises(adapter):
    with pytest.raises(ToolChoiceError, match="Invalid tool_choice type: int"):
        adapter._normalize_tool_choice(123, [make_tool("weather")])


@pytest.mark.unit
def test_get_tool_names_returns_set(adapter):
    tools = [make_tool("weather"), make_tool("search")]
    assert adapter._get_tool_names(tools) == {"weather", "search"}
    assert adapter._get_tool_names(None) == set()


@pytest.mark.unit
def test_normalize_tool_choice_dict_required_raises(adapter):
    with pytest.raises(ToolChoiceError, match="tool_choice='required' is not supported; use 'any'"):
        adapter._normalize_tool_choice({"type": "required"}, [make_tool("weather")])


@pytest.mark.unit
@pytest.mark.parametrize("value", ["auto", "none"])
def test_normalize_tool_choice_dict_auto_none(adapter, value):
    assert (
        adapter._normalize_tool_choice({"type": value}, [make_tool("weather")]) == value
    )


@pytest.mark.unit
def test_normalize_tool_choice_dict_any_requires_tools(adapter):
    with pytest.raises(ToolChoiceError, match="tool_choice.type='any' requires tools to be provided"):
        adapter._normalize_tool_choice({"type": "any"}, None)


@pytest.mark.unit
def test_normalize_tool_choice_dict_any_ok(adapter):
    assert adapter._normalize_tool_choice({"type": "any"}, [make_tool("weather")]) == "any"


@pytest.mark.unit
def test_normalize_tool_choice_dict_tool_with_name_ok(adapter):
    tools = [make_tool("weather"), make_tool("search")]
    assert (
        adapter._normalize_tool_choice({"type": "tool", "name": "search"}, tools)
        == "search"
    )


@pytest.mark.unit
def test_normalize_tool_choice_dict_tool_requires_non_empty_name(adapter):
    with pytest.raises(ToolChoiceError, match="tool_choice.type='tool' requires non-empty name"):
        adapter._normalize_tool_choice({"type": "tool", "name": ""}, [make_tool("weather")])


@pytest.mark.unit
def test_normalize_tool_choice_dict_tool_unknown_name_raises(adapter):
    with pytest.raises(ToolChoiceError, match="tool_choice references unknown tool"):
        adapter._normalize_tool_choice(
            {"type": "tool", "name": "missing"},
            [make_tool("weather")],
        )


@pytest.mark.unit
def test_normalize_tool_choice_dict_name_fallback_ok(adapter):
    tools = [make_tool("weather"), make_tool("search")]
    assert adapter._normalize_tool_choice({"name": "weather"}, tools) == "weather"


@pytest.mark.unit
def test_normalize_tool_choice_dict_invalid_dict_raises(adapter):
    with pytest.raises(ToolChoiceError, match="Invalid tool_choice dict"):
        adapter._normalize_tool_choice({"foo": "bar"}, [make_tool("weather")])


# ---------------------------
# direct helper coverage
# ---------------------------

@pytest.mark.unit
def test_normalize_named_tool_choice_requires_name_string(adapter):
    with pytest.raises(ToolChoiceError, match="tool_choice.type='tool' requires non-empty name"):
        adapter._normalize_named_tool_choice(None, [make_tool("weather")], {"weather"})


@pytest.mark.unit
def test_normalize_named_tool_choice_requires_tools(adapter):
    with pytest.raises(ToolChoiceError, match="tool_choice references a tool but tools=None"):
        adapter._normalize_named_tool_choice("weather", None, {"weather"})


@pytest.mark.unit
def test_normalize_named_tool_choice_unknown_tool(adapter):
    with pytest.raises(ToolChoiceError, match="tool_choice references unknown tool"):
        adapter._normalize_named_tool_choice(
            "search",
            [make_tool("weather")],
            {"weather"},
        )


@pytest.mark.unit
def test_normalize_named_tool_choice_ok(adapter):
    assert (
        adapter._normalize_named_tool_choice(
            "weather",
            [make_tool("weather")],
            {"weather"},
        )
        == "weather"
    )


@pytest.mark.unit
def test_ensure_tools_provided_raises(adapter):
    with pytest.raises(ToolChoiceError, match="custom detail"):
        adapter._ensure_tools_provided(None, detail="custom detail")


@pytest.mark.unit
def test_ensure_tools_provided_accepts_non_empty_tools(adapter):
    adapter._ensure_tools_provided([make_tool("weather")], detail="unused")


@pytest.mark.unit
def test_raise_required_tool_choice_error(adapter):
    with pytest.raises(ToolChoiceError, match="tool_choice='required' is not supported; use 'any'"):
        adapter._raise_required_tool_choice_error()


# ---------------------------
# _resolve_json_schema
# ---------------------------

class _Person(BaseModel):
    name: str
    age: int


class _NotPydantic:
    pass


@pytest.mark.unit
def test_resolve_json_schema_all_none_returns_none(adapter):
    assert adapter._resolve_json_schema(None, None, None) is None


@pytest.mark.unit
def test_resolve_json_schema_dict_returns_dict(adapter):
    schema = {"type": "object"}
    assert adapter._resolve_json_schema(schema, None, None) == schema


@pytest.mark.unit
def test_resolve_json_schema_response_model_returns_schema(adapter):
    result = adapter._resolve_json_schema(None, _Person, None)
    assert result == _Person.model_json_schema()


@pytest.mark.unit
def test_resolve_json_schema_rejects_non_dict_json_schema(adapter):
    with pytest.raises(JSONSchemaError, match="json_schema must be a dict"):
        adapter._resolve_json_schema("not-a-dict", None, None)


@pytest.mark.unit
def test_resolve_json_schema_rejects_both_json_schema_and_response_model(adapter):
    with pytest.raises(JSONSchemaError, match="json_schema and response_model cannot be used together"):
        adapter._resolve_json_schema({"type": "object"}, _Person, None)


@pytest.mark.unit
def test_resolve_json_schema_rejects_response_model_with_tools(adapter):
    with pytest.raises(JSONSchemaError, match="response_model and tools cannot be used together"):
        adapter._resolve_json_schema(None, _Person, [make_tool("weather")])


@pytest.mark.unit
def test_resolve_json_schema_rejects_json_schema_with_tools(adapter):
    with pytest.raises(JSONSchemaError, match="json_schema and tools cannot be used together"):
        adapter._resolve_json_schema({"type": "object"}, None, [make_tool("weather")])


@pytest.mark.unit
def test_resolve_json_schema_rejects_non_pydantic_class(adapter):
    with pytest.raises(JSONSchemaError, match="Pydantic BaseModel"):
        adapter._resolve_json_schema(None, _NotPydantic, None)


# ---------------------------
# _strip_json_fences
# ---------------------------

@pytest.mark.unit
def test_strip_json_fences_plain_json_unchanged(adapter):
    assert adapter._strip_json_fences('{"a": 1}') == '{"a": 1}'


@pytest.mark.unit
def test_strip_json_fences_removes_json_fence(adapter):
    assert adapter._strip_json_fences('```json\n{"a": 1}\n```') == '{"a": 1}'


@pytest.mark.unit
def test_strip_json_fences_removes_plain_fence(adapter):
    assert adapter._strip_json_fences('```\n{"a": 1}\n```') == '{"a": 1}'


@pytest.mark.unit
def test_strip_json_fences_extracts_object_from_preamble(adapter):
    assert adapter._strip_json_fences('Here you go:\n{"a": 1}\nDone.') == '{"a": 1}'


@pytest.mark.unit
def test_strip_json_fences_extracts_array(adapter):
    assert adapter._strip_json_fences('Result: [1, 2, 3]') == '[1, 2, 3]'


# ---------------------------
# _parse_json_response
# ---------------------------

@pytest.mark.unit
def test_parse_json_response_returns_none_when_schema_none(adapter):
    assert adapter._parse_json_response('{"name": "test"}', None) is None


@pytest.mark.unit
def test_parse_json_response_returns_none_when_content_none(adapter):
    assert adapter._parse_json_response(None, {"type": "object"}) is None


@pytest.mark.unit
def test_parse_json_response_returns_parsed_dict(adapter):
    result = adapter._parse_json_response('{"name": "test"}', {"type": "object"})
    assert result == {"name": "test"}


@pytest.mark.unit
def test_parse_json_response_strips_json_fence(adapter):
    result = adapter._parse_json_response('```json\n{"name": "test"}\n```', {"type": "object"})
    assert result == {"name": "test"}


@pytest.mark.unit
def test_parse_json_response_strips_preamble(adapter):
    result = adapter._parse_json_response('Sure! {"name": "test"} done.', {"type": "object"})
    assert result == {"name": "test"}


@pytest.mark.unit
def test_parse_json_response_raises_on_invalid_json(adapter):
    with pytest.raises(JSONSchemaError, match="Model response is not valid JSON"):
        adapter._parse_json_response("not json at all !!!", {"type": "object"})


# ---------------------------
# _parse_response_model
# ---------------------------

@pytest.mark.unit
def test_parse_response_model_returns_none_when_model_none(adapter):
    assert adapter._parse_response_model({"name": "Alice", "age": 30}, None) is None


@pytest.mark.unit
def test_parse_response_model_returns_none_when_json_none(adapter):
    assert adapter._parse_response_model(None, _Person) is None


@pytest.mark.unit
def test_parse_response_model_returns_validated_instance(adapter):
    result = adapter._parse_response_model({"name": "Alice", "age": 30}, _Person)
    assert isinstance(result, _Person)
    assert result.name == "Alice"
    assert result.age == 30


@pytest.mark.unit
def test_parse_response_model_raises_on_invalid_data(adapter):
    with pytest.raises(JSONSchemaError, match="Pydantic validation"):
        adapter._parse_response_model({"name": 123, "age": "not-an-int"}, _Person)
