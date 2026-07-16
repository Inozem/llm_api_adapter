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
    if name == "get_fruit_popularity":
        fruit = args["fruit"]
        popularity = {
            "strawberry": 73,
            "banana": 41,
            "orange": 58,
        }
        if fruit not in popularity:
            raise ValueError(f"Unknown fruit {fruit}")

        return {
            "fruit": fruit,
            "popularity": popularity[fruit],
        }

    raise ValueError(f"Unknown tool {name}")


@pytest.mark.e2e
def test_basic_auto_tool_loop_with_previous_response(subtests, iter_provider_models, chat_with_retry):
    tools = [
        ToolSpec(
            name="get_fruit_popularity",
            description="Return the popularity rating for a fruit.",
            json_schema={
                "type": "object",
                "properties": {
                    "fruit": {"type": "string"},
                },
                "required": ["fruit"],
                "additionalProperties": False,
            },
        )
    ]

    for p, model in iter_provider_models():
        with subtests.test(provider=p["name"], model=model):
            adapter = UniversalLLMAPIAdapter(
                organization=p["name"],
                model=model,
                api_key=p["api_key"],
            )

            messages = [
                UserMessage(
                    'What is the popularity of the fruit "strawberry"? '
                    "Use the available tool if needed."
                )
            ]

            first = chat_with_retry(
                adapter,
                messages=messages,
                tools=tools,
                tool_choice="any",
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
                assert tc.arguments["fruit"] == "strawberry"

                result = run_tool(tc.name, tc.arguments)

                messages.append(
                    ToolMessage(
                        tool_call_id=tc.call_id,
                        content=json.dumps(result),
                    )
                )

            final = chat_with_retry(
                adapter,
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
