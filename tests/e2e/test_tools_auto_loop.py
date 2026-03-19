import json

import pytest

from llm_api_adapter.models.messages.chat_message import (
    UserMessage,
    AIMessage,
    ToolMessage,
)
from llm_api_adapter.models.tools import ToolSpec
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter


def run_tool(name, args):
    if name == "get_secret_word_score":
        token = args["token"]
        scores = {
            "strawberry_v2": 73,
            "banana_v1": 41,
            "orange_v3": 58,
        }
        if token not in scores:
            raise ValueError(f"Unknown token {token}")

        return {
            "token": token,
            "score": scores[token],
        }

    raise ValueError(f"Unknown tool {name}")


@pytest.mark.e2e
def test_basic_auto_tool_loop_with_previous_response(providers, subtests):
    tools = [
        ToolSpec(
            name="get_secret_word_score",
            description="Return the hidden score for a token. Use this tool when the user asks for a token score.",
            json_schema={
                "type": "object",
                "properties": {
                    "token": {"type": "string"},
                },
                "required": ["token"],
                "additionalProperties": False,
            },
        )
    ]

    for p in providers:
        for model in p["models"]:
            with subtests.test(provider=p["name"], model=model):
                adapter = UniversalLLMAPIAdapter(
                    organization=p["name"],
                    model=model,
                    api_key=p["api_key"],
                )

                messages = [
                    UserMessage(
                        'What is the secret score for token "strawberry_v2"? '
                        "Use the available tool if needed."
                    )
                ]

                first = adapter.chat(
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    max_tokens=512,
                    timeout_s=60,
                )

                assert first.tool_calls, (
                    f"{p['name']} / {model}: expected at least one tool_call. "
                    f"Content was: {first.content!r}"
                )

                messages.append(
                    AIMessage(content=first.content or "", tool_calls=first.tool_calls)
                )

                for tc in first.tool_calls:
                    assert tc.name == "get_secret_word_score"
                    assert isinstance(tc.arguments, dict)
                    assert tc.arguments["token"] == "strawberry_v2"

                    result = run_tool(tc.name, tc.arguments)

                    messages.append(
                        ToolMessage(
                            tool_call_id=tc.call_id,
                            content=json.dumps(result),
                        )
                    )

                final = adapter.chat(
                    messages=messages,
                    previous_response=first,
                    max_tokens=512,
                    timeout_s=60,
                )

                assert isinstance(final.content, str)
                assert final.content.strip() != ""
                assert not final.tool_calls, (
                    f"{p['name']} / {model}: expected final natural-language answer, "
                    f"got tool_calls: {final.tool_calls!r}"
                )
                assert "73" in final.content, (
                    f"{p['name']} / {model}: expected final answer to mention 73, "
                    f"got: {final.content!r}"
                )
