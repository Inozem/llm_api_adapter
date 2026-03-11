import pytest

from src.llm_api_adapter.errors.llm_api_error import (
    InvalidToolArgumentsError,
    LLMAPIError,
)
from src.llm_api_adapter.models.responses.chat_response import ChatResponse, Usage
from src.llm_api_adapter.models.tools import ToolCall


@pytest.mark.unit
def test_from_openai_response():
    api_response = {
        "model": "gpt-4",
        "id": "response123",
        "created": 1234567890,
        "usage": {
            "prompt_tokens": 30,
            "completion_tokens": 70,
            "total_tokens": 100,
        },
        "choices": [
            {
                "message": {"content": "Hello from OpenAI!"},
                "finish_reason": "stop",
            }
        ],
    }
    response = ChatResponse.from_openai_response(api_response)
    assert response.model == "gpt-4"
    assert response.response_id == "response123"
    assert response.timestamp == 1234567890
    assert response.usage.input_tokens == 30
    assert response.usage.output_tokens == 70
    assert response.usage.total_tokens == 100
    assert response.content == "Hello from OpenAI!"
    assert response.tool_calls is None
    assert response.finish_reason == "stop"


@pytest.mark.unit
def test_from_openai_response_parses_tool_calls_json_string():
    api_response = {
        "model": "gpt-4",
        "id": "resp_1",
        "created": 111,
        "usage": {},
        "choices": [
            {
                "message": {
                    "content": "tool time",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "weather",
                                "arguments": '{"city":"Tel Aviv"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }

    response = ChatResponse.from_openai_response(api_response)
    assert response.content == "tool time"
    assert response.tool_calls == [
        ToolCall(
            name="weather",
            arguments={"city": "Tel Aviv"},
            call_id="call_1",
        )
    ]
    assert response.finish_reason == "tool_calls"


@pytest.mark.unit
def test_from_openai_response_parses_tool_calls_dict_arguments():
    api_response = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": "weather",
                                "arguments": {"city": "Haifa"},
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }

    response = ChatResponse.from_openai_response(api_response)
    assert response.tool_calls == [
        ToolCall(name="weather", arguments={"city": "Haifa"}, call_id="call_1")
    ]


@pytest.mark.unit
def test_from_openai_response_parses_tool_calls_non_str_non_dict_arguments_as_empty_dict():
    api_response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": "weather",
                                "arguments": 123,
                            },
                        }
                    ]
                }
            }
        ]
    }

    response = ChatResponse.from_openai_response(api_response)
    assert response.tool_calls == [
        ToolCall(name="weather", arguments={}, call_id="call_1")
    ]


@pytest.mark.unit
def test_from_openai_response_invalid_tool_arguments_raise():
    api_response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": "weather",
                                "arguments": "{bad json}",
                            },
                        }
                    ]
                }
            }
        ]
    }

    with pytest.raises(
        InvalidToolArgumentsError,
        match="OpenAI tool arguments JSON parse failed",
    ):
        ChatResponse.from_openai_response(api_response)


@pytest.mark.unit
def test_from_openai_response_legacy_function_call_parses_json_string():
    api_response = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "function_call": {
                        "name": "weather",
                        "arguments": '{"city":"Jerusalem"}',
                    },
                },
                "finish_reason": "function_call",
            }
        ]
    }

    response = ChatResponse.from_openai_response(api_response)
    assert response.tool_calls == [
        ToolCall(
            name="weather",
            arguments={"city": "Jerusalem"},
            call_id=None,
        )
    ]
    assert response.finish_reason == "function_call"


@pytest.mark.unit
def test_from_openai_response_legacy_function_call_parses_dict_arguments():
    api_response = {
        "choices": [
            {
                "message": {
                    "function_call": {
                        "name": "weather",
                        "arguments": {"city": "Eilat"},
                    }
                }
            }
        ]
    }

    response = ChatResponse.from_openai_response(api_response)
    assert response.tool_calls == [
        ToolCall(name="weather", arguments={"city": "Eilat"}, call_id=None)
    ]


@pytest.mark.unit
def test_from_openai_response_legacy_function_call_non_str_non_dict_arguments_become_empty_dict():
    api_response = {
        "choices": [
            {
                "message": {
                    "function_call": {
                        "name": "weather",
                        "arguments": 42,
                    }
                }
            }
        ]
    }

    response = ChatResponse.from_openai_response(api_response)
    assert response.tool_calls == [
        ToolCall(name="weather", arguments={}, call_id=None)
    ]


@pytest.mark.unit
def test_from_openai_response_legacy_function_call_invalid_json_raises():
    api_response = {
        "choices": [
            {
                "message": {
                    "function_call": {
                        "name": "weather",
                        "arguments": "{bad json}",
                    }
                }
            }
        ]
    }

    with pytest.raises(
        InvalidToolArgumentsError,
        match="OpenAI function_call arguments JSON parse failed",
    ):
        ChatResponse.from_openai_response(api_response)


@pytest.mark.unit
def test_from_openai_response_warns_on_empty_content_and_no_tool_calls():
    api_response = {
        "choices": [
            {
                "message": {"content": "   "},
                "finish_reason": "length",
            }
        ]
    }

    with pytest.warns(UserWarning, match="OpenAI returned empty content"):
        response = ChatResponse.from_openai_response(api_response)

    assert response.tool_calls is None
    assert response.content == "   "


@pytest.mark.unit
def test_from_openai_response_handles_missing_choices():
    response = ChatResponse.from_openai_response({"usage": {}})
    assert response.usage.input_tokens == 0
    assert response.usage.output_tokens == 0
    assert response.usage.total_tokens == 0
    assert response.content is None
    assert response.tool_calls is None
    assert response.finish_reason is None


@pytest.mark.unit
def test_from_openai_responses_response_parses_text_and_tool_calls():
    api_response = {
        "model": "gpt-5",
        "id": "resp_123",
        "created_at": 999,
        "status": "completed",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
        },
        "output": [
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "Hello"},
                    {"type": "text", "text": "World"},
                ],
            },
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "weather",
                "arguments": '{"city":"Tel Aviv"}',
            },
        ],
    }

    response = ChatResponse.from_openai_responses_response(api_response)
    assert response.model == "gpt-5"
    assert response.response_id == "resp_123"
    assert response.timestamp == 999
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 20
    assert response.usage.total_tokens == 30
    assert response.content == "Hello\nWorld"
    assert response.tool_calls == [
        ToolCall(name="weather", arguments={"city": "Tel Aviv"}, call_id="call_1")
    ]
    assert response.finish_reason == "completed"


@pytest.mark.unit
def test_from_openai_responses_response_uses_id_when_call_id_missing():
    api_response = {
        "output": [
            {
                "type": "tool_call",
                "id": "item_1",
                "name": "weather",
                "arguments": {"city": "Ashdod"},
            }
        ]
    }

    response = ChatResponse.from_openai_responses_response(api_response)
    assert response.tool_calls == [
        ToolCall(name="weather", arguments={"city": "Ashdod"}, call_id="item_1")
    ]


@pytest.mark.unit
def test_from_openai_responses_response_non_list_output_becomes_empty_and_warns():
    api_response = {
        "output": {"not": "a list"},
    }

    with pytest.warns(
        UserWarning,
        match="OpenAI Responses API returned empty content and no tool calls",
    ):
        response = ChatResponse.from_openai_responses_response(api_response)

    assert response.content is None
    assert response.tool_calls is None


@pytest.mark.unit
def test_from_openai_responses_response_skips_non_dict_items():
    api_response = {
        "output": [
            "bad-item",
            123,
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "ok"}],
            },
        ]
    }

    response = ChatResponse.from_openai_responses_response(api_response)
    assert response.content == "ok"
    assert response.tool_calls is None


@pytest.mark.unit
def test_from_openai_responses_response_skips_non_list_message_content():
    api_response = {
        "output": [
            {
                "type": "message",
                "content": "bad-content",
            }
        ]
    }

    with pytest.warns(
        UserWarning,
        match="OpenAI Responses API returned empty content and no tool calls",
    ):
        response = ChatResponse.from_openai_responses_response(api_response)

    assert response.content is None


@pytest.mark.unit
def test_from_openai_responses_response_skips_non_dict_content_items():
    api_response = {
        "output": [
            {
                "type": "message",
                "content": [
                    "bad",
                    {"type": "output_text", "text": "ok"},
                ],
            }
        ]
    }

    response = ChatResponse.from_openai_responses_response(api_response)
    assert response.content == "ok"


@pytest.mark.unit
def test_from_openai_responses_response_ignores_blank_text_chunks():
    api_response = {
        "output": [
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "   "},
                    {"type": "text", "text": "hello"},
                ],
            }
        ]
    }

    response = ChatResponse.from_openai_responses_response(api_response)
    assert response.content == "hello"


@pytest.mark.unit
def test_from_openai_responses_response_tool_arguments_non_str_non_dict_become_empty_dict():
    api_response = {
        "output": [
            {
                "type": "function_call",
                "name": "weather",
                "arguments": 123,
            }
        ]
    }

    response = ChatResponse.from_openai_responses_response(api_response)
    assert response.tool_calls == [
        ToolCall(name="weather", arguments={}, call_id=None)
    ]


@pytest.mark.unit
def test_from_openai_responses_response_invalid_tool_arguments_raise():
    api_response = {
        "output": [
            {
                "type": "function_call",
                "name": "weather",
                "arguments": "{bad json}",
            }
        ]
    }

    with pytest.raises(
        InvalidToolArgumentsError,
        match="OpenAI responses tool arguments JSON parse failed",
    ):
        ChatResponse.from_openai_responses_response(api_response)


@pytest.mark.unit
def test_from_anthropic_response():
    api_response = {
        "usage": {"input_tokens": 30, "output_tokens": 70},
        "model": "claude-2",
        "id": "anthropic123",
        "content": [{"type": "text", "text": "Hello from Anthropic!"}],
        "stop_reason": "end",
    }
    response = ChatResponse.from_anthropic_response(api_response)
    assert response.model == "claude-2"
    assert response.response_id == "anthropic123"
    assert response.usage.input_tokens == 30
    assert response.usage.output_tokens == 70
    assert response.usage.total_tokens == 100
    assert response.content == "Hello from Anthropic!"
    assert response.tool_calls is None
    assert response.finish_reason == "end"


@pytest.mark.unit
def test_from_anthropic_response_parses_tool_use_blocks():
    api_response = {
        "usage": {"input_tokens": 5, "output_tokens": 7},
        "model": "claude-3",
        "id": "a1",
        "content": [
            {"type": "text", "text": "first text"},
            {
                "type": "tool_use",
                "id": "tool_1",
                "name": "weather",
                "input": {"city": "TLV"},
            },
            {"type": "text", "text": "second text"},
        ],
        "stop_reason": "tool_use",
    }

    response = ChatResponse.from_anthropic_response(api_response)
    assert response.content == "first text"
    assert response.tool_calls == [
        ToolCall(name="weather", arguments={"city": "TLV"}, call_id="tool_1")
    ]
    assert response.finish_reason == "tool_use"


@pytest.mark.unit
def test_from_anthropic_response_invalid_tool_input_raises():
    api_response = {
        "content": [
            {
                "type": "tool_use",
                "id": "tool_1",
                "name": "weather",
                "input": "bad",
            }
        ]
    }

    with pytest.raises(
        InvalidToolArgumentsError,
        match="Anthropic tool input must be dict",
    ):
        ChatResponse.from_anthropic_response(api_response)


@pytest.mark.unit
def test_from_google_response():
    api_response = {
        "candidates": [
            {
                "content": {"parts": [{"text": "Hello from Google!"}]},
                "finishReason": "length",
            }
        ],
        "usageMetadata": {"totalTokenCount": 42},
    }
    response = ChatResponse.from_google_response(api_response)
    assert response.usage.total_tokens == 42
    assert response.content == "Hello from Google!"
    assert response.finish_reason == "length"


@pytest.mark.unit
def test_from_google_response_parses_usage_and_function_call():
    api_response = {
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 20,
            "thoughtsTokenCount": 5,
            "totalTokenCount": 35,
        },
        "candidates": [
            {
                "finishReason": "STOP",
                "content": {
                    "parts": [
                        {"text": "Hello from Google!"},
                        {
                            "functionCall": {
                                "name": "weather",
                                "args": {"city": "Tel Aviv"},
                            }
                        },
                    ]
                },
            }
        ],
    }

    response = ChatResponse.from_google_response(api_response)
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 25
    assert response.usage.total_tokens == 35
    assert response.content == "Hello from Google!"
    assert response.tool_calls == [
        ToolCall(name="weather", arguments={"city": "Tel Aviv"}, call_id="weather")
    ]
    assert response.finish_reason == "STOP"


@pytest.mark.unit
def test_from_google_response_uses_function_call_alias():
    api_response = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "function_call": {
                                "name": "weather",
                                "arguments": {"city": "Beer Sheva"},
                            }
                        }
                    ]
                }
            }
        ]
    }

    response = ChatResponse.from_google_response(api_response)
    assert response.tool_calls == [
        ToolCall(
            name="weather",
            arguments={"city": "Beer Sheva"},
            call_id="weather",
        )
    ]


@pytest.mark.unit
def test_from_google_response_none_args_become_empty_dict():
    api_response = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "functionCall": {
                                "name": "weather",
                                "args": None,
                            }
                        }
                    ]
                }
            }
        ]
    }

    response = ChatResponse.from_google_response(api_response)
    assert response.tool_calls == [
        ToolCall(name="weather", arguments={}, call_id="weather")
    ]


@pytest.mark.unit
def test_from_google_response_malformed_parts_raises():
    api_response = {
        "candidates": [
            {
                "content": {
                    "parts": "not-a-list",
                }
            }
        ]
    }

    with pytest.raises(LLMAPIError, match="Google API returned malformed response"):
        ChatResponse.from_google_response(api_response)


@pytest.mark.unit
def test_from_google_response_skips_non_dict_parts_and_uses_first_text_only():
    api_response = {
        "candidates": [
            {
                "finishReason": 123,
                "content": {
                    "parts": [
                        "bad",
                        {"text": "first"},
                        {"text": "second"},
                    ]
                },
            }
        ]
    }

    response = ChatResponse.from_google_response(api_response)
    assert response.content == "first"
    assert response.finish_reason == "123"


@pytest.mark.unit
def test_from_google_response_function_call_requires_non_empty_name():
    api_response = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "functionCall": {
                                "name": "",
                                "args": {},
                            }
                        }
                    ]
                }
            }
        ]
    }

    with pytest.raises(
        InvalidToolArgumentsError,
        match="Google functionCall.name must be non-empty str",
    ):
        ChatResponse.from_google_response(api_response)


@pytest.mark.unit
def test_from_google_response_function_call_requires_dict_args():
    api_response = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "functionCall": {
                                "name": "weather",
                                "args": "bad",
                            }
                        }
                    ]
                }
            }
        ]
    }

    with pytest.raises(
        InvalidToolArgumentsError,
        match="Google functionCall args must be dict",
    ):
        ChatResponse.from_google_response(api_response)


@pytest.mark.unit
def test_apply_pricing_sets_costs():
    response = ChatResponse(
        usage=Usage(input_tokens=100, output_tokens=50, total_tokens=150)
    )

    response.apply_pricing(
        price_input_per_token=0.001,
        price_output_per_token=0.002,
        currency="USD",
    )

    assert response.currency == "USD"
    assert response.cost_input == 0.1
    assert response.cost_output == 0.1
    assert response.cost_total == 0.2


@pytest.mark.unit
def test_apply_pricing_without_usage_does_nothing():
    response = ChatResponse()
    response.apply_pricing(
        price_input_per_token=0.001,
        price_output_per_token=0.002,
        currency="EUR",
    )

    assert response.currency is None
    assert response.cost_input is None
    assert response.cost_output is None
    assert response.cost_total is None
