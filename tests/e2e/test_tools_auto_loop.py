import json

import pytest

from llm_api_adapter.models.messages.chat_message import (
    UserMessage,
    AIMessage,
    ToolMessage,
)
from llm_api_adapter.models.tools import ToolSpec
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter


FRUIT_POPULARITY = {
    "strawberry": 73,
    "banana": 41,
    "orange": 58,
}


def run_tool(name, args):
    if name == "get_fruit_popularity":
        fruit = args["fruit"]
        if fruit not in FRUIT_POPULARITY:
            raise ValueError(f"Unknown fruit {fruit}")

        return {
            "fruit": fruit,
            "popularity": FRUIT_POPULARITY[fruit],
        }

    raise ValueError(f"Unknown tool {name}")


@pytest.mark.e2e
def test_basic_auto_tool_loop_with_previous_response(subtests, iter_organization_models, chat_with_retry):
    tools = [
        ToolSpec(
            name="get_fruit_popularity",
            description="Return the popularity rating for a fruit.",
            json_schema={
                "type": "object",
                "properties": {
                    "fruit": {
                        "type": "string",
                        "enum": list(FRUIT_POPULARITY),
                        "description": "The fruit whose popularity is requested.",
                    },
                },
                "required": ["fruit"],
                "additionalProperties": False,
            },
        )
    ]

    for p, model in iter_organization_models():
        with subtests.test(provider=p["name"], model=model):
            adapter = UniversalLLMAPIAdapter(
                organization=p["name"],
                model=model,
                api_key=p["api_key"],
            )

            messages = [
                UserMessage(
                    "What is the popularity of the fruit banana? "
                    "Use the available tool to look it up."
                )
            ]

            first = chat_with_retry(
                adapter,
                messages=messages,
                tools=tools,
                tool_choice="get_fruit_popularity",
                max_tokens=512,
                timeout_s=60,
            )

            assert first.finish_reason != "refusal", (
                f"{p['name']} / {model}: model refused the request (safety classifier)"
            )

            assert first.tool_calls, (
                f"{p['name']} / {model}: expected at least one tool_call. "
                f"Content was: {first.content!r}. "
                f"Raw tool_calls: {first.tool_calls!r}"
            )

            messages.append(
                AIMessage(content=first.content or "", tool_calls=first.tool_calls)
            )

            for tc in first.tool_calls:
                assert tc.name == "get_fruit_popularity"
                assert isinstance(tc.arguments, dict)
                fruit = tc.arguments["fruit"]
                assert fruit in FRUIT_POPULARITY

                result = run_tool(tc.name, tc.arguments)
                assert result["popularity"] == FRUIT_POPULARITY[fruit]

                messages.append(
                    ToolMessage(
                        tool_call_id=tc.call_id,
                        content=json.dumps(result),
                    )
                )

            final = chat_with_retry(
                adapter,
                messages=messages,
                max_tokens=512,
                timeout_s=60,
                previous_response=first,
            )

            assert isinstance(final.content, str)
            assert final.content.strip() != ""
            assert not final.tool_calls, (
                f"{p['name']} / {model}: expected final natural-language answer, "
                f"got tool_calls: {final.tool_calls!r}"
            )
