import json

import httpx
import pytest
import respx
import requests_mock

from src.llm_api_adapter.models.messages.chat_message import (
    AIMessage,
    Prompt,
    ToolMessage,
    UserMessage,
)
from src.llm_api_adapter.models.tools import ToolSpec
from src.llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter


TOOLS = [
    ToolSpec(
        name="get_weather",
        description="Get current weather for a city",
        json_schema={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    )
]

MULTI_TOOLS = TOOLS + [
    ToolSpec(
        name="get_time",
        description="Get the local time for a city",
        json_schema={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    )
]


def _multi_tool_messages():
    return [
        Prompt("Use the available tools to answer the user's question."),
        UserMessage("What's the weather and local time in Tel Aviv today?"),
    ]


def _append_multi_tool_results(messages, response):
    assert response.tool_calls is not None
    messages.append(AIMessage(content="", tool_calls=response.tool_calls))
    results = {
        "get_weather": {"city": "Tel Aviv", "temperature": 22, "unit": "C"},
        "get_time": {"city": "Tel Aviv", "time": "14:30", "timezone": "Asia/Jerusalem"},
    }
    for tool_call in response.tool_calls:
        messages.append(
            ToolMessage(
                tool_call_id=tool_call.call_id,
                content=json.dumps(results[tool_call.name], ensure_ascii=False),
            )
        )


def _assert_multi_tool_calls(response, expected):
    assert response.tool_calls is not None
    assert [
        (tool_call.name, tool_call.arguments, tool_call.call_id)
        for tool_call in response.tool_calls
    ] == expected


@pytest.fixture
def openai_tools_mock():
    with requests_mock.Mocker() as mock:
        mock.post(
            "https://api.openai.com/v1/responses",
            [
                {
                    "json": {
                        "id": "resp_1",
                        "model": "gpt-5.2",
                        "status": "completed",
                        "output": [
                            {
                                "type": "function_call",
                                "name": "get_weather",
                                "arguments": "{\"city\":\"Tel Aviv\"}",
                                "call_id": "call_123",
                            }
                        ],
                        "usage": {
                            "input_tokens": 10,
                            "output_tokens": 5,
                            "total_tokens": 15,
                        },
                    }
                },
                {
                    "json": {
                        "id": "resp_2",
                        "model": "gpt-5.2",
                        "status": "completed",
                        "output": [
                            {
                                "type": "message",
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "output_text",
                                        "text": "The weather in Tel Aviv is 22 C.",
                                    }
                                ],
                            }
                        ],
                        "usage": {
                            "input_tokens": 12,
                            "output_tokens": 8,
                            "total_tokens": 20,
                        },
                    }
                },
            ],
        )
        yield mock


@pytest.fixture
def anthropic_tools_mock():
    with requests_mock.Mocker() as mock:
        mock.post(
            "https://api.anthropic.com/v1/messages",
            [
                {
                    "json": {
                        "id": "msg_1",
                        "model": "claude-sonnet-4-5",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_123",
                                "name": "get_weather",
                                "input": {"city": "Tel Aviv"},
                            }
                        ],
                        "usage": {"input_tokens": 5, "output_tokens": 7},
                        "stop_reason": "tool_use",
                    }
                },
                {
                    "json": {
                        "id": "msg_2",
                        "model": "claude-sonnet-4-5",
                        "content": [
                            {
                                "type": "text",
                                "text": "The weather in Tel Aviv is 22 C.",
                            }
                        ],
                        "usage": {"input_tokens": 6, "output_tokens": 8},
                        "stop_reason": "end_turn",
                    }
                },
            ],
        )
        yield mock


@pytest.fixture
def google_tools_mock():
    with requests_mock.Mocker() as mock:
        mock.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent",
            [
                {
                    "json": {
                        "candidates": [
                            {
                                "content": {
                                    "parts": [
                                        {
                                            "functionCall": {
                                                "name": "get_weather",
                                                "args": {"city": "Tel Aviv"},
                                            }
                                        }
                                    ]
                                },
                                "finishReason": "STOP",
                            }
                        ],
                        "usageMetadata": {"totalTokenCount": 42},
                    }
                },
                {
                    "json": {
                        "candidates": [
                            {
                                "content": {
                                    "parts": [
                                        {
                                            "text": "The weather in Tel Aviv is 22 C."
                                        }
                                    ]
                                },
                                "finishReason": "STOP",
                            }
                        ],
                        "usageMetadata": {"totalTokenCount": 50},
                    }
                },
            ],
        )
        yield mock


@pytest.fixture
def openai_multi_tools_mock():
    with requests_mock.Mocker() as mock:
        mock.post(
            "https://api.openai.com/v1/responses",
            [
                {
                    "json": {
                        "id": "resp_multi_1",
                        "model": "gpt-5.2",
                        "status": "completed",
                        "output": [
                            {
                                "type": "function_call",
                                "name": "get_weather",
                                "arguments": '{"city":"Tel Aviv"}',
                                "call_id": "call_weather",
                            },
                            {
                                "type": "function_call",
                                "name": "get_time",
                                "arguments": '{"city":"Tel Aviv"}',
                                "call_id": "call_time",
                            },
                        ],
                        "usage": {
                            "input_tokens": 12,
                            "output_tokens": 8,
                            "total_tokens": 20,
                        },
                    }
                },
                {
                    "json": {
                        "id": "resp_multi_2",
                        "model": "gpt-5.2",
                        "status": "completed",
                        "output": [
                            {
                                "type": "message",
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "output_text",
                                        "text": "Tel Aviv: 22 C, 14:30 local time.",
                                    }
                                ],
                            }
                        ],
                        "usage": {
                            "input_tokens": 20,
                            "output_tokens": 9,
                            "total_tokens": 29,
                        },
                    }
                },
            ],
        )
        yield mock


@pytest.fixture
def anthropic_multi_tools_mock():
    with requests_mock.Mocker() as mock:
        mock.post(
            "https://api.anthropic.com/v1/messages",
            [
                {
                    "json": {
                        "id": "msg_multi_1",
                        "model": "claude-sonnet-4-5",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_weather",
                                "name": "get_weather",
                                "input": {"city": "Tel Aviv"},
                            },
                            {
                                "type": "tool_use",
                                "id": "toolu_time",
                                "name": "get_time",
                                "input": {"city": "Tel Aviv"},
                            },
                        ],
                        "usage": {"input_tokens": 8, "output_tokens": 10},
                        "stop_reason": "tool_use",
                    }
                },
                {
                    "json": {
                        "id": "msg_multi_2",
                        "model": "claude-sonnet-4-5",
                        "content": [
                            {
                                "type": "text",
                                "text": "Tel Aviv: 22 C, 14:30 local time.",
                            }
                        ],
                        "usage": {"input_tokens": 18, "output_tokens": 9},
                        "stop_reason": "end_turn",
                    }
                },
            ],
        )
        yield mock


@pytest.fixture
def google_multi_tools_mock():
    with requests_mock.Mocker() as mock:
        mock.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent",
            [
                {
                    "json": {
                        "candidates": [
                            {
                                "content": {
                                    "parts": [
                                        {
                                            "functionCall": {
                                                "name": "get_weather",
                                                "args": {"city": "Tel Aviv"},
                                            }
                                        },
                                        {
                                            "functionCall": {
                                                "name": "get_time",
                                                "args": {"city": "Tel Aviv"},
                                            }
                                        },
                                    ]
                                },
                                "finishReason": "STOP",
                            }
                        ],
                        "usageMetadata": {"totalTokenCount": 20},
                    }
                },
                {
                    "json": {
                        "candidates": [
                            {
                                "content": {
                                    "parts": [
                                        {
                                            "text": "Tel Aviv: 22 C, 14:30 local time."
                                        }
                                    ]
                                },
                                "finishReason": "STOP",
                            }
                        ],
                        "usageMetadata": {"totalTokenCount": 29},
                    }
                },
            ],
        )
        yield mock


@pytest.mark.integration
def test_openai_gpt5_tool_call_parsing_and_request_mapping(openai_tools_mock):
    adapter = UniversalLLMAPIAdapter(
        organization="openai",
        model="gpt-5.2",
        api_key="dummy_key",
    )
    messages = [
        Prompt("You are a helpful assistant. If user asks about weather, call get_weather."),
        UserMessage("What's the weather in Tel Aviv today?"),
    ]

    response = adapter.chat(
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        max_tokens=128,
    )

    assert response.content is None
    assert response.finish_reason == "completed"
    assert response.response_id == "resp_1"
    assert response.tool_calls is not None
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "get_weather"
    assert response.tool_calls[0].arguments == {"city": "Tel Aviv"}
    assert response.tool_calls[0].call_id == "call_123"

    first_request = openai_tools_mock.request_history[0]
    payload = first_request.json()

    assert payload["model"] == "gpt-5.2"
    assert payload["input"] == [{"role": "user", "content": "What's the weather in Tel Aviv today?"}]
    assert payload["instructions"] == "You are a helpful assistant. If user asks about weather, call get_weather."
    assert payload["tool_choice"] == "auto"
    assert payload["max_output_tokens"] == 128

    assert payload["tools"] == [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
        }
    ]


@pytest.mark.integration
def test_openai_gpt5_tool_loop_with_previous_response(openai_tools_mock):
    adapter = UniversalLLMAPIAdapter(
        organization="openai",
        model="gpt-5.2",
        api_key="dummy_key",
    )
    messages = [
        Prompt("You are a helpful assistant. If user asks about weather, call get_weather."),
        UserMessage("What's the weather in Tel Aviv today?"),
    ]

    first = adapter.chat(
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        max_tokens=128,
    )

    assert first.tool_calls is not None
    assert first.response_id == "resp_1"

    messages.append(AIMessage(content="", tool_calls=first.tool_calls))
    messages.append(
        ToolMessage(
            tool_call_id=first.tool_calls[0].call_id,
            content=json.dumps(
                {"city": "Tel Aviv", "temperature": 22, "unit": "C"},
                ensure_ascii=False,
            ),
        )
    )

    final = adapter.chat(
        messages=messages,
        max_tokens=256,
        previous_response=first,
    )

    assert final.content == "The weather in Tel Aviv is 22 C."
    assert final.tool_calls is None
    assert final.response_id == "resp_2"

    second_request = openai_tools_mock.request_history[1]
    payload = second_request.json()

    assert payload["previous_response_id"] == "resp_1"
    assert payload["max_output_tokens"] == 256
    assert payload["input"] == [
        {
            "role": "user",
            "content": "What's the weather in Tel Aviv today?",
        },
        {
            "type": "function_call_output",
            "call_id": "call_123",
            "output": json.dumps(
                {"city": "Tel Aviv", "temperature": 22, "unit": "C"},
                ensure_ascii=False,
            ),
        },
    ]


@pytest.mark.integration
def test_anthropic_tool_call_parsing_and_request_mapping(anthropic_tools_mock):
    adapter = UniversalLLMAPIAdapter(
        organization="anthropic",
        model="claude-sonnet-4-5",
        api_key="dummy_key",
    )
    messages = [
        Prompt("You are a helpful assistant. If user asks about weather, call get_weather."),
        UserMessage("What's the weather in Tel Aviv today?"),
    ]

    response = adapter.chat(
        messages=messages,
        max_tokens=256,
        tools=TOOLS,
        tool_choice="auto",
    )

    assert response.content is None
    assert response.finish_reason == "tool_use"
    assert response.response_id == "msg_1"
    assert response.tool_calls is not None
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "get_weather"
    assert response.tool_calls[0].arguments == {"city": "Tel Aviv"}
    assert response.tool_calls[0].call_id == "toolu_123"

    first_request = anthropic_tools_mock.request_history[0]
    payload = first_request.json()

    assert payload["model"] == "claude-sonnet-4-5"
    assert payload["system"] == "You are a helpful assistant. If user asks about weather, call get_weather."
    assert payload["tool_choice"] == {"type": "auto"}
    assert payload["tools"] == [
        {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
        }
    ]


@pytest.mark.integration
def test_anthropic_tool_loop_message_protocol(anthropic_tools_mock):
    adapter = UniversalLLMAPIAdapter(
        organization="anthropic",
        model="claude-sonnet-4-5",
        api_key="dummy_key",
    )
    messages = [
        Prompt("You are a helpful assistant. If user asks about weather, call get_weather."),
        UserMessage("What's the weather in Tel Aviv today?"),
    ]

    first = adapter.chat(
        messages=messages,
        max_tokens=256,
        tools=TOOLS,
        tool_choice="auto",
    )

    assert first.tool_calls is not None
    messages.append(AIMessage(content="", tool_calls=first.tool_calls))
    messages.append(
        ToolMessage(
            tool_call_id=first.tool_calls[0].call_id,
            content=json.dumps(
                {"city": "Tel Aviv", "temperature": 22, "unit": "C"},
                ensure_ascii=False,
            ),
        )
    )

    final = adapter.chat(
        messages=messages,
        max_tokens=256,
    )

    assert final.content == "The weather in Tel Aviv is 22 C."

    second_request = anthropic_tools_mock.request_history[1]
    payload = second_request.json()
    request_messages = payload["messages"]

    assert request_messages[1] == {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_123",
                "name": "get_weather",
                "input": {"city": "Tel Aviv"},
            }
        ],
    }
    assert request_messages[2] == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_123",
                "content": json.dumps(
                    {"city": "Tel Aviv", "temperature": 22, "unit": "C"},
                    ensure_ascii=False,
                ),
            }
        ],
    }


@pytest.mark.integration
def test_google_tool_call_and_tool_result_protocol(google_tools_mock):
    adapter = UniversalLLMAPIAdapter(
        organization="google",
        model="gemini-2.5-pro",
        api_key="dummy_key",
    )
    messages = [
        Prompt("You are a helpful assistant. If user asks about weather, call get_weather."),
        UserMessage("What's the weather in Tel Aviv today?"),
    ]

    first = adapter.chat(
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        max_tokens=128,
    )

    assert first.tool_calls is not None
    assert len(first.tool_calls) == 1
    assert first.tool_calls[0].name == "get_weather"
    assert first.tool_calls[0].arguments == {"city": "Tel Aviv"}
    assert first.tool_calls[0].call_id == "get_weather"

    first_request = google_tools_mock.request_history[0]
    first_payload = first_request.json()

    assert first_payload["system_instruction"] == {
        "parts": [{"text": "You are a helpful assistant. If user asks about weather, call get_weather."}]
    }
    assert first_payload["toolConfig"] == {
        "functionCallingConfig": {"mode": "AUTO"}
    }
    assert first_payload["tools"] == [
        {
            "functionDeclarations": [
                {
                    "name": "get_weather",
                    "description": "Get current weather for a city",
                    "parametersJsonSchema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                        "additionalProperties": False,
                    },
                }
            ]
        }
    ]

    messages.append(AIMessage(content="", tool_calls=first.tool_calls))
    messages.append(
        ToolMessage(
            tool_call_id=first.tool_calls[0].call_id,
            content=json.dumps(
                {"city": "Tel Aviv", "temperature": 22, "unit": "C"},
                ensure_ascii=False,
            ),
        )
    )

    final = adapter.chat(messages=messages, max_tokens=256)

    assert final.content == "The weather in Tel Aviv is 22 C."

    second_request = google_tools_mock.request_history[1]
    second_payload = second_request.json()
    contents = second_payload["contents"]

    assert contents[1] == {
        "role": "model",
        "parts": [
            {
                "functionCall": {
                    "name": "get_weather",
                    "args": {"city": "Tel Aviv"},
                }
            }
        ],
    }
    assert contents[2] == {
        "role": "user",
        "parts": [
            {
                "functionResponse": {
                    "name": "get_weather",
                    "response": {
                        "city": "Tel Aviv",
                        "temperature": 22,
                        "unit": "C",
                    },
                }
            }
        ],
    }


@pytest.mark.integration
def test_openai_gpt5_multi_tool_round_trip(openai_multi_tools_mock):
    adapter = UniversalLLMAPIAdapter(
        organization="openai",
        model="gpt-5.2",
        api_key="dummy_key",
    )
    messages = _multi_tool_messages()

    first = adapter.chat(
        messages=messages,
        tools=MULTI_TOOLS,
        tool_choice="auto",
        max_tokens=128,
    )

    _assert_multi_tool_calls(
        first,
        [
            ("get_weather", {"city": "Tel Aviv"}, "call_weather"),
            ("get_time", {"city": "Tel Aviv"}, "call_time"),
        ],
    )
    assert first.response_id == "resp_multi_1"

    _append_multi_tool_results(messages, first)
    final = adapter.chat(messages=messages, max_tokens=256, previous_response=first)

    assert final.content == "Tel Aviv: 22 C, 14:30 local time."
    assert final.tool_calls is None
    assert final.response_id == "resp_multi_2"

    assert len(openai_multi_tools_mock.request_history) == 2
    second_payload = openai_multi_tools_mock.request_history[1].json()
    assert second_payload["previous_response_id"] == "resp_multi_1"
    assert second_payload["input"] == [
        {"role": "user", "content": "What's the weather and local time in Tel Aviv today?"},
        {
            "type": "function_call_output",
            "call_id": "call_weather",
            "output": json.dumps(
                {"city": "Tel Aviv", "temperature": 22, "unit": "C"},
                ensure_ascii=False,
            ),
        },
        {
            "type": "function_call_output",
            "call_id": "call_time",
            "output": json.dumps(
                {"city": "Tel Aviv", "time": "14:30", "timezone": "Asia/Jerusalem"},
                ensure_ascii=False,
            ),
        },
    ]


@pytest.mark.integration
def test_anthropic_multi_tool_round_trip(anthropic_multi_tools_mock):
    adapter = UniversalLLMAPIAdapter(
        organization="anthropic",
        model="claude-sonnet-4-5",
        api_key="dummy_key",
    )
    messages = _multi_tool_messages()

    first = adapter.chat(
        messages=messages,
        max_tokens=256,
        tools=MULTI_TOOLS,
        tool_choice="auto",
    )

    _assert_multi_tool_calls(
        first,
        [
            ("get_weather", {"city": "Tel Aviv"}, "toolu_weather"),
            ("get_time", {"city": "Tel Aviv"}, "toolu_time"),
        ],
    )
    assert first.response_id == "msg_multi_1"

    _append_multi_tool_results(messages, first)
    final = adapter.chat(messages=messages, max_tokens=256)

    assert final.content == "Tel Aviv: 22 C, 14:30 local time."
    assert final.tool_calls is None
    assert final.response_id == "msg_multi_2"

    assert len(anthropic_multi_tools_mock.request_history) == 2
    second_payload = anthropic_multi_tools_mock.request_history[1].json()
    assert second_payload["messages"] == [
        {"role": "user", "content": "What's the weather and local time in Tel Aviv today?"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_weather",
                    "name": "get_weather",
                    "input": {"city": "Tel Aviv"},
                },
                {
                    "type": "tool_use",
                    "id": "toolu_time",
                    "name": "get_time",
                    "input": {"city": "Tel Aviv"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_weather",
                    "content": json.dumps(
                        {"city": "Tel Aviv", "temperature": 22, "unit": "C"},
                        ensure_ascii=False,
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_time",
                    "content": json.dumps(
                        {"city": "Tel Aviv", "time": "14:30", "timezone": "Asia/Jerusalem"},
                        ensure_ascii=False,
                    ),
                }
            ],
        },
    ]


@pytest.mark.integration
def test_google_multi_tool_round_trip(google_multi_tools_mock):
    adapter = UniversalLLMAPIAdapter(
        organization="google",
        model="gemini-2.5-pro",
        api_key="dummy_key",
    )
    messages = _multi_tool_messages()

    first = adapter.chat(
        messages=messages,
        tools=MULTI_TOOLS,
        tool_choice="auto",
        max_tokens=128,
    )

    _assert_multi_tool_calls(
        first,
        [
            ("get_weather", {"city": "Tel Aviv"}, "get_weather"),
            ("get_time", {"city": "Tel Aviv"}, "get_time"),
        ],
    )

    _append_multi_tool_results(messages, first)
    final = adapter.chat(messages=messages, max_tokens=256)

    assert final.content == "Tel Aviv: 22 C, 14:30 local time."
    assert final.tool_calls is None

    assert len(google_multi_tools_mock.request_history) == 2
    second_payload = google_multi_tools_mock.request_history[1].json()
    assert second_payload["contents"] == [
        {
            "role": "user",
            "parts": [
                {"text": "What's the weather and local time in Tel Aviv today?"}
            ],
        },
        {
            "role": "model",
            "parts": [
                {"functionCall": {"name": "get_weather", "args": {"city": "Tel Aviv"}}},
                {"functionCall": {"name": "get_time", "args": {"city": "Tel Aviv"}}},
            ],
        },
        {
            "role": "user",
            "parts": [
                {
                    "functionResponse": {
                        "name": "get_weather",
                        "response": {
                            "city": "Tel Aviv",
                            "temperature": 22,
                            "unit": "C",
                        },
                    }
                }
            ],
        },
        {
            "role": "user",
            "parts": [
                {
                    "functionResponse": {
                        "name": "get_time",
                        "response": {
                            "city": "Tel Aviv",
                            "time": "14:30",
                            "timezone": "Asia/Jerusalem",
                        },
                    }
                }
            ],
        },
    ]


@pytest.mark.asyncio
@pytest.mark.integration
async def test_openai_async_gpt5_multi_tool_round_trip():
    responses = [
        {
            "id": "async_resp_multi_1",
            "model": "gpt-5.2",
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "name": "get_weather",
                    "arguments": '{"city":"Tel Aviv"}',
                    "call_id": "async_call_weather",
                },
                {
                    "type": "function_call",
                    "name": "get_time",
                    "arguments": '{"city":"Tel Aviv"}',
                    "call_id": "async_call_time",
                },
            ],
        },
        {
            "id": "async_resp_multi_2",
            "model": "gpt-5.2",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Tel Aviv: 22 C, 14:30 local time.",
                        }
                    ],
                }
            ],
        },
    ]
    with respx.mock(assert_all_called=False) as router:
        route = router.post("https://api.openai.com/v1/responses")
        route.side_effect = lambda request: httpx.Response(200, json=responses.pop(0))

        adapter = UniversalLLMAPIAdapter(
            organization="openai",
            model="gpt-5.2",
            api_key="dummy_key",
        )
        messages = _multi_tool_messages()
        first = await adapter.achat(
            messages=messages,
            tools=MULTI_TOOLS,
            tool_choice="auto",
            max_tokens=128,
        )

        _assert_multi_tool_calls(
            first,
            [
                ("get_weather", {"city": "Tel Aviv"}, "async_call_weather"),
                ("get_time", {"city": "Tel Aviv"}, "async_call_time"),
            ],
        )
        _append_multi_tool_results(messages, first)
        final = await adapter.achat(
            messages=messages,
            max_tokens=256,
            previous_response=first,
        )

        assert final.content == "Tel Aviv: 22 C, 14:30 local time."
        assert final.tool_calls is None
        assert final.response_id == "async_resp_multi_2"
        assert route.call_count == 2
        second_payload = json.loads(route.calls[1].request.content)
        assert [item["call_id"] for item in second_payload["input"][1:]] == [
            "async_call_weather",
            "async_call_time",
        ]


@pytest.mark.asyncio
@pytest.mark.integration
async def test_anthropic_async_multi_tool_round_trip():
    responses = [
        {
            "id": "async_msg_multi_1",
            "model": "claude-sonnet-4-5",
            "content": [
                {
                    "type": "tool_use",
                    "id": "async_toolu_weather",
                    "name": "get_weather",
                    "input": {"city": "Tel Aviv"},
                },
                {
                    "type": "tool_use",
                    "id": "async_toolu_time",
                    "name": "get_time",
                    "input": {"city": "Tel Aviv"},
                },
            ],
            "usage": {"input_tokens": 8, "output_tokens": 10},
            "stop_reason": "tool_use",
        },
        {
            "id": "async_msg_multi_2",
            "model": "claude-sonnet-4-5",
            "content": [
                {"type": "text", "text": "Tel Aviv: 22 C, 14:30 local time."}
            ],
            "usage": {"input_tokens": 18, "output_tokens": 9},
            "stop_reason": "end_turn",
        },
    ]
    with respx.mock(assert_all_called=False) as router:
        route = router.post("https://api.anthropic.com/v1/messages")
        route.side_effect = lambda request: httpx.Response(200, json=responses.pop(0))

        adapter = UniversalLLMAPIAdapter(
            organization="anthropic",
            model="claude-sonnet-4-5",
            api_key="dummy_key",
        )
        messages = _multi_tool_messages()
        first = await adapter.achat(
            messages=messages,
            max_tokens=256,
            tools=MULTI_TOOLS,
            tool_choice="auto",
        )

        _assert_multi_tool_calls(
            first,
            [
                ("get_weather", {"city": "Tel Aviv"}, "async_toolu_weather"),
                ("get_time", {"city": "Tel Aviv"}, "async_toolu_time"),
            ],
        )
        _append_multi_tool_results(messages, first)
        final = await adapter.achat(messages=messages, max_tokens=256)

        assert final.content == "Tel Aviv: 22 C, 14:30 local time."
        assert final.tool_calls is None
        assert final.response_id == "async_msg_multi_2"
        assert route.call_count == 2
        second_payload = json.loads(route.calls[1].request.content)
        assistant_blocks = second_payload["messages"][1]["content"]
        result_blocks = [item["content"][0] for item in second_payload["messages"][2:]]
        assert [block["id"] for block in assistant_blocks] == [
            "async_toolu_weather",
            "async_toolu_time",
        ]
        assert [block["tool_use_id"] for block in result_blocks] == [
            "async_toolu_weather",
            "async_toolu_time",
        ]


@pytest.mark.asyncio
@pytest.mark.integration
async def test_google_async_multi_tool_round_trip():
    responses = [
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"functionCall": {"name": "get_weather", "args": {"city": "Tel Aviv"}}},
                            {"functionCall": {"name": "get_time", "args": {"city": "Tel Aviv"}}},
                        ]
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"totalTokenCount": 20},
        },
        {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Tel Aviv: 22 C, 14:30 local time."}]
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"totalTokenCount": 29},
        },
    ]
    with respx.mock(assert_all_called=False) as router:
        route = router.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent"
        )
        route.side_effect = lambda request: httpx.Response(200, json=responses.pop(0))

        adapter = UniversalLLMAPIAdapter(
            organization="google",
            model="gemini-2.5-pro",
            api_key="dummy_key",
        )
        messages = _multi_tool_messages()
        first = await adapter.achat(
            messages=messages,
            tools=MULTI_TOOLS,
            tool_choice="auto",
            max_tokens=128,
        )

        _assert_multi_tool_calls(
            first,
            [
                ("get_weather", {"city": "Tel Aviv"}, "get_weather"),
                ("get_time", {"city": "Tel Aviv"}, "get_time"),
            ],
        )
        _append_multi_tool_results(messages, first)
        final = await adapter.achat(messages=messages, max_tokens=256)

        assert final.content == "Tel Aviv: 22 C, 14:30 local time."
        assert final.tool_calls is None
        assert route.call_count == 2
        second_payload = json.loads(route.calls[1].request.content)
        model_parts = second_payload["contents"][1]["parts"]
        response_parts = second_payload["contents"][2:]
        assert [part["functionCall"]["name"] for part in model_parts] == [
            "get_weather",
            "get_time",
        ]
        assert [part["functionResponse"]["name"] for item in response_parts for part in item["parts"]] == [
            "get_weather",
            "get_time",
        ]
