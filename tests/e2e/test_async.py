import json

import pytest

from llm_api_adapter.errors.llm_api_error import (
    LLMAPIAuthorizationError,
    LLMAPITimeoutError,
)
from llm_api_adapter.models.messages.chat_message import (
    AIMessage,
    ToolMessage,
    UserMessage,
)
from llm_api_adapter.models.messages.file_parts import DocumentPart, ImagePart
from llm_api_adapter.models.tools import ToolSpec
from tests.e2e import harness
from tests.e2e.conftest import e2e_model_case_parameters


pytest.importorskip("httpx")


SIMPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
    },
    "required": ["name", "age"],
    "additionalProperties": False,
}

KUDIBLOID_COUNTS = {7: 479}

KUDIBLOID_TOOL = ToolSpec(
    name="lookup_kudibloids",
    description=(
        "Return the authoritative kudibloid count for a number of brankiches. "
        "This tool is the only source for these values."
    ),
    json_schema={
        "type": "object",
        "properties": {
            "brankiches": {
                "type": "integer",
                "enum": list(KUDIBLOID_COUNTS),
            },
        },
        "required": ["brankiches"],
        "additionalProperties": False,
    },
)


def _assert_usage_and_pricing(response):
    assert response.usage is not None
    assert response.usage.input_tokens >= 0
    assert response.usage.output_tokens >= 0
    assert response.usage.total_tokens >= response.usage.input_tokens
    assert response.currency
    assert response.cost_total is not None and response.cost_total >= 0
    if response.cost_input is None or response.cost_output is None:
        assert response.cost_input is response.cost_output is None
        return
    assert response.cost_input >= 0
    assert response.cost_output >= 0


@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.e2e_capability("async_chat")
@pytest.mark.parametrize("e2e_model_case", e2e_model_case_parameters())
async def test_async_chat_returns_structured_response_and_pricing(
    e2e_model_case,
    e2e_model_organization,
    async_chat_with_retry,
    e2e_adapter,
):
    model = e2e_model_case.model_spec
    assert model is not None
    if not e2e_model_organization["api_key"]:
        pytest.skip("No organization API key is configured")

    response = await async_chat_with_retry(
        e2e_adapter(e2e_model_organization, model.name),
        messages=[UserMessage('Return JSON with name="Alice" and age=30.')],
        max_tokens=1024,
        json_schema=SIMPLE_SCHEMA,
        timeout_s=60,
    )

    assert response.refusal is None
    assert response.incomplete_reason is None
    assert isinstance(response.content, str)
    assert response.parsed_json == {"name": "Alice", "age": 30}
    _assert_usage_and_pricing(response)


@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.e2e_capability("async_streaming")
@pytest.mark.parametrize("e2e_model_case", e2e_model_case_parameters())
async def test_async_streaming_preserves_callbacks_and_final_response(
    e2e_model_case,
    e2e_model_organization,
    async_stream_with_retry,
    e2e_adapter,
):
    model = e2e_model_case.model_spec
    assert model is not None
    if not e2e_model_organization["api_key"]:
        pytest.skip("No organization API key is configured")

    chunks = []
    deltas = []
    completed = []

    async def on_chunk(chunk):
        chunks.append(chunk)

    async def on_delta(text):
        deltas.append(text)

    async def on_done(response):
        completed.append(response)

    def reset_observers():
        chunks.clear()
        deltas.clear()
        completed.clear()

    text_chunks = await async_stream_with_retry(
        e2e_adapter(e2e_model_organization, model.name),
        messages=[UserMessage("Reply with exactly: OK")],
        max_tokens=1024,
        timeout_s=60,
        buffer_chars=8,
        on_chunk=on_chunk,
        on_delta=on_delta,
        on_done=on_done,
        on_retry=reset_observers,
    )

    streamed_text = "".join(text_chunks)
    assert streamed_text.strip()
    assert [chunk.text for chunk in chunks] == text_chunks
    assert deltas == text_chunks
    assert all(len(chunk.text) <= 8 for chunk in chunks)
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))
    assert len(completed) == 1
    assert completed[0].content == streamed_text
    assert completed[0].finish_reason


@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.e2e_capability("application_tools")
@pytest.mark.parametrize("e2e_model_case", e2e_model_case_parameters())
async def test_async_tools_round_trip_with_previous_response(
    e2e_model_case,
    e2e_model_organization,
    async_chat_with_retry,
    tool_choice_for_model,
    e2e_adapter,
):
    model = e2e_model_case.model_spec
    assert model is not None
    organization = e2e_model_organization
    if not organization["api_key"]:
        pytest.skip("No organization API key is configured")

    tool_choice = tool_choice_for_model(
        organization["name"],
        model.name,
        KUDIBLOID_TOOL.name,
    )
    adapter = e2e_adapter(organization, model.name)
    messages = [
        UserMessage(
            "Retrieve the kudibloid count for 7 brankiches. The count is "
            "not available in this prompt: call lookup_kudibloids to "
            "obtain it. After the tool returns, answer with its "
            "kudibloids value; do not guess."
        )
    ]
    first = await async_chat_with_retry(
        adapter,
        messages=messages,
        tools=[KUDIBLOID_TOOL],
        tool_choice=tool_choice,
        max_tokens=512,
        timeout_s=60,
    )

    assert first.tool_calls
    messages.append(AIMessage(content=first.content or "", tool_calls=first.tool_calls))
    for tool_call in first.tool_calls:
        assert tool_call.name == KUDIBLOID_TOOL.name
        brankiches = tool_call.arguments["brankiches"]
        assert brankiches in KUDIBLOID_COUNTS
        messages.append(
            ToolMessage(
                tool_call_id=tool_call.call_id,
                content=json.dumps(
                    {
                        "brankiches": brankiches,
                        "kudibloids": KUDIBLOID_COUNTS[brankiches],
                    }
                ),
            )
        )

    final = await async_chat_with_retry(
        adapter,
        messages=messages,
        max_tokens=512,
        timeout_s=60,
        previous_response=first,
    )
    assert final.content and final.content.strip()
    assert str(KUDIBLOID_COUNTS[7]) in final.content
    assert not final.tool_calls


@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.e2e_capability("image_bytes")
@pytest.mark.parametrize("e2e_model_case", e2e_model_case_parameters())
async def test_async_image_input_returns_text(
    e2e_model_case,
    e2e_model_organization,
    vision_image_bytes,
    async_chat_with_retry,
    e2e_adapter,
):
    model = e2e_model_case.model_spec
    assert model is not None
    if not e2e_model_organization["api_key"]:
        pytest.skip("No organization API key is configured")

    image_response = await async_chat_with_retry(
        e2e_adapter(e2e_model_organization, model.name),
        messages=[
            UserMessage(
                "Describe this image in one short sentence.",
                files=[ImagePart(data=vision_image_bytes, media_type="image/png")],
            )
        ],
        max_tokens=512,
        timeout_s=60,
    )
    assert image_response.content and image_response.content.strip()


@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.e2e_capability("pdf_bytes")
@pytest.mark.parametrize("e2e_model_case", e2e_model_case_parameters())
async def test_async_document_input_returns_text(
    e2e_model_case,
    e2e_model_organization,
    pdf_bytes,
    async_chat_with_retry,
    e2e_adapter,
):
    model = e2e_model_case.model_spec
    assert model is not None
    if not e2e_model_organization["api_key"]:
        pytest.skip("No organization API key is configured")

    document_response = await async_chat_with_retry(
        e2e_adapter(e2e_model_organization, model.name),
        messages=[
            harness.make_document_message(
                "Summarize this document in one sentence.",
                DocumentPart(data=pdf_bytes, media_type="application/pdf"),
            )
        ],
        max_tokens=512,
        timeout_s=60,
    )
    assert document_response.content and document_response.content.strip()


@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.e2e_capability("error_normalization")
@pytest.mark.parametrize("e2e_model_case", e2e_model_case_parameters())
async def test_async_errors_are_normalized(
    e2e_model_case,
    e2e_model_organization,
    e2e_adapter,
):
    model = e2e_model_case.model_spec
    assert model is not None
    organization = e2e_model_organization

    with pytest.raises(LLMAPIAuthorizationError):
        await e2e_adapter(
            {**organization, "api_key": "NON_VALID_KEY"}, model.name
        ).achat(
            messages=[UserMessage("Say OK")],
            max_tokens=32,
            timeout_s=10,
        )

    if not organization["api_key"]:
        pytest.skip("No organization API key is configured for timeout checks")

    with pytest.raises(LLMAPITimeoutError):
        await e2e_adapter(organization, model.name).achat(
            messages=[UserMessage("Say OK")],
            max_tokens=32,
            timeout_s=0.1,
        )
