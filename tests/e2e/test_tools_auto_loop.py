import json

import pytest

from llm_api_adapter.models.messages.chat_message import (
    UserMessage,
    AIMessage,
    ToolMessage,
)
from llm_api_adapter.models.tools import ToolSpec
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter


KUDIBLOID_COUNTS = {7: 479}


def run_tool(name, args):
    if name == "lookup_kudibloids":
        brankiches = args["brankiches"]
        if brankiches not in KUDIBLOID_COUNTS:
            raise ValueError(f"Unknown brankich count {brankiches}")

        return {
            "brankiches": brankiches,
            "kudibloids": KUDIBLOID_COUNTS[brankiches],
        }

    raise ValueError(f"Unknown tool {name}")


@pytest.mark.e2e
def test_basic_tool_loop_with_previous_response(
    subtests,
    iter_organization_models,
    chat_with_retry,
    tool_choice_for_model,
):
    tools = [
        ToolSpec(
            name="lookup_kudibloids",
            description=(
                "Return the authoritative kudibloid count for a number of "
                "brankiches. This tool is the only source for these values."
            ),
            json_schema={
                "type": "object",
                "properties": {
                    "brankiches": {
                        "type": "integer",
                        "enum": list(KUDIBLOID_COUNTS),
                        "description": "The number of brankiches to look up.",
                    },
                },
                "required": ["brankiches"],
                "additionalProperties": False,
            },
        )
    ]

    for p, model in iter_organization_models():
        tool_choice = tool_choice_for_model(p["name"], model, tools[0].name)
        with subtests.test(
            provider=p["name"],
            model=model,
            tool_choice=tool_choice,
        ):
            adapter = UniversalLLMAPIAdapter(
                organization=p["name"],
                model=model,
                api_key=p["api_key"],
            )

            messages = [
                UserMessage(
                    "Retrieve the kudibloid count for 7 brankiches. The count is "
                    "not available in this prompt: call lookup_kudibloids to "
                    "obtain it. After the tool returns, answer with its "
                    "kudibloids value; do not guess."
                )
            ]

            first = chat_with_retry(
                adapter,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
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
                assert tc.name == "lookup_kudibloids"
                assert isinstance(tc.arguments, dict)
                brankiches = tc.arguments["brankiches"]
                assert brankiches in KUDIBLOID_COUNTS

                result = run_tool(tc.name, tc.arguments)
                assert result["kudibloids"] == KUDIBLOID_COUNTS[brankiches]

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
            assert str(KUDIBLOID_COUNTS[7]) in final.content
            assert not final.tool_calls, (
                f"{p['name']} / {model}: expected final natural-language answer, "
                f"got tool_calls: {final.tool_calls!r}"
            )
