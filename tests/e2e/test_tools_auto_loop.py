import json

import pytest

from llm_api_adapter.models.messages.chat_message import (
    UserMessage,
    AIMessage,
    ToolMessage,
)
from llm_api_adapter.models.tools import ToolSpec
from tests.e2e.conftest import e2e_model_case_parameters


KUDIBLOID_COUNTS = {7: 479}
_LOOKUP_TOOL_NAME = "lookup_kudibloids"
_TOOLS = [
    ToolSpec(
        name=_LOOKUP_TOOL_NAME,
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
_TOOL_PROMPT = (
    "Retrieve the kudibloid count for 7 brankiches. The count is not available "
    "in this prompt: call lookup_kudibloids to obtain it. After the tool returns, "
    "answer with its kudibloids value; do not guess."
)


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
@pytest.mark.e2e_capability("application_tools")
@pytest.mark.parametrize("e2e_model_case", e2e_model_case_parameters())
def test_basic_tool_loop_with_previous_response(
    e2e_model_case,
    e2e_model_organization,
    chat_with_retry,
    tool_choice_for_model,
    e2e_adapter,
):
    model = e2e_model_case.model_spec
    assert model is not None
    organization = e2e_model_organization
    tool_choice = tool_choice_for_model(
        organization["name"],
        model.name,
        _LOOKUP_TOOL_NAME,
    )
    adapter = e2e_adapter(organization, model.name)
    messages = [UserMessage(_TOOL_PROMPT)]

    first = chat_with_retry(
        adapter,
        messages=messages,
        tools=_TOOLS,
        tool_choice=tool_choice,
        max_tokens=512,
        timeout_s=60,
    )

    assert first.finish_reason != "refusal", "model refused the tool request"
    assert first.tool_calls, (
        f"Expected a tool_call. Content was: {first.content!r}. "
        f"Raw tool_calls: {first.tool_calls!r}"
    )

    messages.append(AIMessage(content=first.content or "", tool_calls=first.tool_calls))
    for tool_call in first.tool_calls:
        assert tool_call.name == _LOOKUP_TOOL_NAME
        assert isinstance(tool_call.arguments, dict)
        brankiches = tool_call.arguments["brankiches"]
        assert brankiches in KUDIBLOID_COUNTS
        result = run_tool(tool_call.name, tool_call.arguments)
        assert result["kudibloids"] == KUDIBLOID_COUNTS[brankiches]
        messages.append(
            ToolMessage(
                tool_call_id=tool_call.call_id,
                content=json.dumps(result),
            )
        )

    final = chat_with_retry(
        adapter,
        messages=messages,
        tools=_TOOLS,
        max_tokens=512,
        timeout_s=60,
        previous_response=first,
    )
    assert isinstance(final.content, str) and final.content.strip()
    assert str(KUDIBLOID_COUNTS[7]) in final.content
    assert not final.tool_calls
