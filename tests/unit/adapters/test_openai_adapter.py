from unittest.mock import patch

import pytest

from src.llm_api_adapter.adapters.openai_adapter import OpenAIAdapter
from src.llm_api_adapter.errors.llm_api_error import LLMAPIError
from src.llm_api_adapter.llms.openai.sync_client import OpenAISyncClient
from src.llm_api_adapter.models.messages.chat_message import Prompt, UserMessage
from src.llm_api_adapter.models.responses.chat_response import ChatResponse
from src.llm_api_adapter.models.tools import ToolSpec


@pytest.fixture
def adapter():
    return OpenAIAdapter(
        api_key="test_api_key",
        model="gpt-5",
    )


@pytest.fixture
def legacy_adapter():
    return OpenAIAdapter(
        api_key="test_api_key",
        model="gpt-4o",
    )


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
        temp_result = adapter._validate_parameter(
            "temperature", temperature, 0, 2
        )
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
        OpenAISyncClient,
        "complete",
        side_effect=LLMAPIError("API error"),
    ), patch.object(adapter, "handle_error") as mock_handle_error:
        adapter.chat(messages)
        mock_handle_error.assert_called_once()


@pytest.mark.unit
def test_chat_handles_generic_exception(adapter):
    messages = [Prompt("system prompt"), UserMessage("hello")]
    with patch.object(
        OpenAISyncClient,
        "complete",
        side_effect=Exception("Generic error"),
    ), patch.object(adapter, "handle_error") as mock_handle_error:
        adapter.chat(messages)
        mock_handle_error.assert_called_once()


@pytest.mark.unit
def test_pricing_is_applied_when_present_for_responses_api(adapter):
    adapter.pricing = type(
        "P",
        (),
        {
            "in_per_token": 0.001,
            "out_per_token": 0.002,
            "currency": "USD",
        },
    )()
    fake_response = {"some": "openai response"}
    fake_chat_response = ChatResponse()

    with (
        patch.object(OpenAISyncClient, "complete", return_value=fake_response) as mock_client,
        patch.object(
            ChatResponse,
            "from_openai_responses_response",
            return_value=fake_chat_response,
        ) as mock_from,
        patch.object(ChatResponse, "apply_pricing") as mock_apply,
    ):
        result = adapter.chat([UserMessage("hi")], max_tokens=10)

    mock_client.assert_called_once()
    mock_from.assert_called_once_with(fake_response)
    mock_apply.assert_called_once_with(
        price_input_per_token=adapter.pricing.in_per_token,
        price_output_per_token=adapter.pricing.out_per_token,
        currency=adapter.pricing.currency,
    )
    assert result is fake_chat_response


@pytest.mark.unit
def test_pricing_is_applied_when_present_for_legacy_api(legacy_adapter):
    legacy_adapter.pricing = type(
        "P",
        (),
        {
            "in_per_token": 0.001,
            "out_per_token": 0.002,
            "currency": "USD",
        },
    )()
    fake_response = {"some": "openai response"}
    fake_chat_response = ChatResponse()

    with (
        patch.object(OpenAISyncClient, "complete", return_value=fake_response) as mock_client,
        patch.object(
            ChatResponse,
            "from_openai_response",
            return_value=fake_chat_response,
        ) as mock_from,
        patch.object(ChatResponse, "apply_pricing") as mock_apply,
    ):
        result = legacy_adapter.chat([UserMessage("hi")], max_tokens=10)

    mock_client.assert_called_once()
    mock_from.assert_called_once_with(fake_response)
    mock_apply.assert_called_once_with(
        price_input_per_token=legacy_adapter.pricing.in_per_token,
        price_output_per_token=legacy_adapter.pricing.out_per_token,
        currency=legacy_adapter.pricing.currency,
    )
    assert result is fake_chat_response


@pytest.mark.unit
def test_chat_responses_api_builds_expected_params(adapter):
    fake_response = {"id": "resp_123"}
    fake_chat_response = ChatResponse(response_id="prev_123")
    expected_previous_response = ChatResponse(response_id="prev_123")

    with (
        patch.object(OpenAISyncClient, "complete", return_value=fake_response) as mock_complete,
        patch.object(
            ChatResponse,
            "from_openai_responses_response",
            return_value=fake_chat_response,
        ) as mock_from,
    ):
        result = adapter.chat(
            [Prompt("system prompt"), UserMessage("hello")],
            max_tokens=42,
            temperature=0.7,
            top_p=0.9,
            previous_response=expected_previous_response,
        )

    assert result is fake_chat_response
    mock_from.assert_called_once_with(fake_response)

    _, kwargs = mock_complete.call_args
    assert kwargs["model"] == "gpt-5"
    assert kwargs["max_tokens"] == 42
    assert kwargs["temperature"] == 0.7
    assert kwargs["top_p"] == 0.9
    assert kwargs["previous_response_id"] == "prev_123"
    assert "input" in kwargs
    assert "instructions" in kwargs
    assert "messages" not in kwargs
    assert "parallel_tool_calls" not in kwargs


@pytest.mark.unit
def test_chat_legacy_api_builds_expected_params(legacy_adapter):
    fake_response = {"id": "chatcmpl_123"}
    fake_chat_response = ChatResponse()

    with (
        patch.object(OpenAISyncClient, "complete", return_value=fake_response) as mock_complete,
        patch.object(
            ChatResponse,
            "from_openai_response",
            return_value=fake_chat_response,
        ) as mock_from,
    ):
        result = legacy_adapter.chat(
            [Prompt("system prompt"), UserMessage("hello")],
            max_tokens=33,
            temperature=0.5,
            top_p=0.8,
            parallel_tool_calls=True,
        )

    assert result is fake_chat_response
    mock_from.assert_called_once_with(fake_response)

    _, kwargs = mock_complete.call_args
    assert kwargs["model"] == "gpt-4o"
    assert kwargs["max_tokens"] == 33
    assert kwargs["temperature"] == 0.5
    assert kwargs["top_p"] == 0.8
    assert kwargs["parallel_tool_calls"] is True
    assert "messages" in kwargs
    assert "input" not in kwargs
    assert "instructions" not in kwargs
    assert "previous_response_id" not in kwargs


@pytest.mark.unit
def test_chat_responses_api_omits_instructions_when_none(adapter):
    fake_response = {"id": "resp_123"}
    fake_chat_response = ChatResponse()

    with (
        patch.object(
            adapter,
            "_normalize_messages",
        ) as mock_normalize_messages,
        patch.object(OpenAISyncClient, "complete", return_value=fake_response) as mock_complete,
        patch.object(
            ChatResponse,
            "from_openai_responses_response",
            return_value=fake_chat_response,
        ),
    ):
        normalized_messages = mock_normalize_messages.return_value
        normalized_messages.to_openai_responses_input.return_value = [{"role": "user", "content": "hi"}]
        normalized_messages.to_openai_responses_instructions.return_value = None

        adapter.chat([UserMessage("hi")])

    _, kwargs = mock_complete.call_args
    assert "instructions" not in kwargs


@pytest.mark.unit
def test_chat_without_pricing_does_not_call_apply_pricing(adapter):
    adapter.pricing = None
    fake_response = {"id": "resp_123"}
    fake_chat_response = ChatResponse()

    with (
        patch.object(OpenAISyncClient, "complete", return_value=fake_response),
        patch.object(
            ChatResponse,
            "from_openai_responses_response",
            return_value=fake_chat_response,
        ),
        patch.object(ChatResponse, "apply_pricing") as mock_apply,
    ):
        result = adapter.chat([UserMessage("hi")])

    assert result is fake_chat_response
    mock_apply.assert_not_called()


@pytest.mark.unit
def test_chat_calls_validate_tools_and_normalize_tool_choice(adapter):
    fake_response = {"id": "resp_123"}
    fake_chat_response = ChatResponse()
    tools = [
        ToolSpec(
            name="get_weather",
            description="Get weather",
            json_schema={"type": "object", "properties": {}},
        )
    ]

    with (
        patch.object(adapter, "_validate_tools") as mock_validate_tools,
        patch.object(adapter, "_normalize_tool_choice", return_value="auto") as mock_normalize_tool_choice,
        patch.object(OpenAISyncClient, "complete", return_value=fake_response),
        patch.object(
            ChatResponse,
            "from_openai_responses_response",
            return_value=fake_chat_response,
        ),
    ):
        adapter.chat(
            [UserMessage("hi")],
            tools=tools,
            tool_choice="auto",
        )

    mock_validate_tools.assert_called_once_with(tools)
    mock_normalize_tool_choice.assert_called_once_with("auto", tools)


@pytest.mark.unit
def test_chat_passes_responses_tools_and_tool_choice(adapter):
    fake_response = {"id": "resp_123"}
    fake_chat_response = ChatResponse()
    tools = [
        ToolSpec(
            name="get_weather",
            description="Get weather",
            json_schema={"type": "object", "properties": {}},
        )
    ]

    with (
        patch.object(OpenAISyncClient, "complete", return_value=fake_response) as mock_complete,
        patch.object(
            ChatResponse,
            "from_openai_responses_response",
            return_value=fake_chat_response,
        ),
    ):
        adapter.chat(
            [UserMessage("hi")],
            tools=tools,
            tool_choice="get_weather",
        )

    _, kwargs = mock_complete.call_args
    assert kwargs["tools"] == [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object", "properties": {}},
        }
    ]
    assert kwargs["tool_choice"] == {
        "type": "function",
        "name": "get_weather",
    }


@pytest.mark.unit
def test_chat_passes_legacy_tools_and_tool_choice(legacy_adapter):
    fake_response = {"id": "chatcmpl_123"}
    fake_chat_response = ChatResponse()
    tools = [
        ToolSpec(
            name="get_weather",
            description="Get weather",
            json_schema={"type": "object", "properties": {}},
        )
    ]

    with (
        patch.object(OpenAISyncClient, "complete", return_value=fake_response) as mock_complete,
        patch.object(
            ChatResponse,
            "from_openai_response",
            return_value=fake_chat_response,
        ),
    ):
        legacy_adapter.chat(
            [UserMessage("hi")],
            tools=tools,
            tool_choice="get_weather",
        )

    _, kwargs = mock_complete.call_args
    assert kwargs["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    assert kwargs["tool_choice"] == {
        "type": "function",
        "function": {"name": "get_weather"},
    }


@pytest.mark.unit
def test_normalize_reasoning_level_disabled_warns_and_returns_none(adapter):
    adapter.is_reasoning = False
    with pytest.warns(UserWarning) as record:
        res = adapter._normalize_reasoning_level("low")
    assert res is None
    assert any("does not support reasoning" in str(w.message) for w in record)


@pytest.mark.unit
def test_normalize_reasoning_level_enabled_none_returns_none_string(adapter):
    adapter.is_reasoning = True
    res = adapter._normalize_reasoning_level(None)
    assert res == "none"


@pytest.mark.unit
def test_normalize_reasoning_level_disabled_none_returns_none(adapter):
    adapter.is_reasoning = False
    res = adapter._normalize_reasoning_level(None)
    assert res is None


@pytest.mark.unit
def test_normalize_reasoning_level_bool_raises(adapter):
    adapter.is_reasoning = True
    with pytest.raises(ValueError, match="bool is not accepted"):
        adapter._normalize_reasoning_level(True)


@pytest.mark.unit
def test_normalize_reasoning_level_valid_string_key_returns_key(adapter):
    adapter.is_reasoning = True
    adapter.reasoning_levels = {"low": 1, "medium": 5, "high": 10}
    assert adapter._normalize_reasoning_level("medium") == "medium"


@pytest.mark.unit
def test_normalize_reasoning_level_invalid_string_key_raises(adapter):
    adapter.is_reasoning = True
    adapter.reasoning_levels = {"low": 1, "medium": 5}
    with pytest.raises(ValueError) as exc:
        adapter._normalize_reasoning_level("unknown")
    assert "Unknown reasoning level key" in str(exc.value)
    assert "low" in str(exc.value)
    assert "medium" in str(exc.value)


@pytest.mark.unit
def test_normalize_reasoning_level_int_maps_to_correct_key(adapter):
    adapter.is_reasoning = True
    adapter.reasoning_levels = {"low": 1, "medium": 5, "high": 10}
    assert adapter._normalize_reasoning_level(0) == "low"
    assert adapter._normalize_reasoning_level(3) == "medium"
    assert adapter._normalize_reasoning_level(100) == "high"


@pytest.mark.unit
def test_normalize_reasoning_level_invalid_type_raises(adapter):
    adapter.is_reasoning = True
    with pytest.raises(ValueError, match="expected int or str"):
        adapter._normalize_reasoning_level(1.5)


@pytest.mark.unit
def test_map_tools_to_openai_returns_none_for_empty(adapter):
    assert adapter._map_tools_to_openai(None) is None
    assert adapter._map_tools_to_openai([]) is None


@pytest.mark.unit
def test_map_tools_to_openai_maps_description(adapter):
    tools = [
        ToolSpec(
            name="get_weather",
            description="Get weather",
            json_schema={"type": "object", "properties": {"city": {"type": "string"}}},
        )
    ]

    result = adapter._map_tools_to_openai(tools)

    assert result == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        }
    ]


@pytest.mark.unit
def test_map_tools_to_openai_omits_description_when_missing(adapter):
    tools = [
        ToolSpec(
            name="get_weather",
            description=None,
            json_schema={"type": "object", "properties": {}},
        )
    ]

    result = adapter._map_tools_to_openai(tools)

    assert result == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


@pytest.mark.unit
def test_map_tool_choice_to_openai(adapter):
    assert adapter._map_tool_choice_to_openai(None) is None
    assert adapter._map_tool_choice_to_openai("auto") == "auto"
    assert adapter._map_tool_choice_to_openai("none") == "none"
    assert adapter._map_tool_choice_to_openai("any") == "required"
    assert adapter._map_tool_choice_to_openai("get_weather") == {
        "type": "function",
        "function": {"name": "get_weather"},
    }


@pytest.mark.unit
def test_map_tools_to_openai_responses_returns_none_for_empty(adapter):
    assert adapter._map_tools_to_openai_responses(None) is None
    assert adapter._map_tools_to_openai_responses([]) is None


@pytest.mark.unit
def test_map_tools_to_openai_responses_maps_tool(adapter):
    tools = [
        ToolSpec(
            name="get_weather",
            description="Get weather",
            json_schema={"type": "object", "properties": {"city": {"type": "string"}}},
        )
    ]

    result = adapter._map_tools_to_openai_responses(tools)

    assert result == [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        }
    ]


@pytest.mark.unit
def test_map_tool_choice_to_openai_responses(adapter):
    assert adapter._map_tool_choice_to_openai_responses(None) is None
    assert adapter._map_tool_choice_to_openai_responses("auto") == "auto"
    assert adapter._map_tool_choice_to_openai_responses("none") == "none"
    assert adapter._map_tool_choice_to_openai_responses("any") == "required"
    assert adapter._map_tool_choice_to_openai_responses("get_weather") == {
        "type": "function",
        "name": "get_weather",
    }


@pytest.mark.unit
def test_map_tool_choice_to_openai_responses_passthrough_non_string(adapter):
    tool_choice = {"type": "function", "name": "get_weather"}
    assert adapter._map_tool_choice_to_openai_responses(tool_choice) is tool_choice
