from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.llm_api_adapter.adapters.base_adapter import LLMAdapterBase
from src.llm_api_adapter.adapters import base_adapter as base_module
from src.llm_api_adapter.errors.llm_api_error import (
    InvalidToolSchemaError,
    LLMAPIError,
    ToolChoiceError,
)
from src.llm_api_adapter.models.messages.chat_message import (
    Messages,
    Prompt,
    UserMessage,
)
from src.llm_api_adapter.models.responses.chat_response import ChatResponse
from src.llm_api_adapter.models.tools import ToolSpec


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

    def _normalize_reasoning_level(self, reasoning_level):
        return reasoning_level


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
        SimpleNamespace(providers={}),
        raising=False,
    )
    with pytest.warns(UserWarning):
        a = _TestAdapter(company="missing", api_key="k", model="unknown-model")
    assert a.pricing is None


@pytest.mark.unit
def test_pricing_copied_from_registry(monkeypatch):
    base_pricing = {"cost_per_token": 0.001}
    provider = SimpleNamespace(
        models={"m-pro": SimpleNamespace(pricing=base_pricing, is_reasoning=False)}
    )
    monkeypatch.setattr(
        base_module,
        "LLM_REGISTRY",
        SimpleNamespace(providers={"acme": provider}),
        raising=False,
    )
    adapter_instance = _TestAdapter(company="acme", api_key="k", model="m-pro")
    assert adapter_instance.pricing == base_pricing
    assert adapter_instance.pricing is not base_pricing


@pytest.mark.unit
def test_post_init_sets_reasoning_flag_from_registry(monkeypatch):
    provider = SimpleNamespace(
        models={"m-reason": SimpleNamespace(pricing=None, is_reasoning=True)}
    )
    monkeypatch.setattr(
        base_module,
        "LLM_REGISTRY",
        SimpleNamespace(providers={"acme": provider}),
        raising=False,
    )
    adapter_instance = _TestAdapter(company="acme", api_key="k", model="m-reason")
    assert adapter_instance.is_reasoning is True


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
        LLMAdapterBase.chat(adapter)


@pytest.mark.unit
def test_base_abstract_normalize_reasoning_level_raises_not_implemented(adapter):
    with pytest.raises(NotImplementedError):
        LLMAdapterBase._normalize_reasoning_level(adapter, "low")


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
