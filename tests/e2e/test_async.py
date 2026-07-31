import json

import pytest

from llm_api_adapter.errors import JSONSchemaError
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
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter


pytest.importorskip("httpx")


SIMPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
    },
    "required": ["name", "age"],
}

FRUIT_POPULARITY = {
    "strawberry": 73,
    "banana": 41,
    "orange": 58,
}

FRUIT_TOOL = ToolSpec(
    name="get_fruit_popularity",
    description="Return the popularity rating for a fruit.",
    json_schema={
        "type": "object",
        "properties": {
            "fruit": {
                "type": "string",
                "enum": list(FRUIT_POPULARITY),
            },
        },
        "required": ["fruit"],
        "additionalProperties": False,
    },
)


def _adapter(provider, model):
    return UniversalLLMAPIAdapter(
        organization=provider["name"],
        model=model,
        api_key=provider["api_key"],
    )


def _assert_usage_and_pricing(response):
    assert response.usage is not None
    assert response.usage.input_tokens >= 0
    assert response.usage.output_tokens >= 0
    assert response.usage.total_tokens >= response.usage.input_tokens
    assert response.currency
    assert response.cost_input >= 0
    assert response.cost_output >= 0
    assert response.cost_total >= 0


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_async_chat_returns_structured_response_and_pricing(
    subtests,
    configured_async_e2e_models,
    async_chat_with_retry,
):
    if not configured_async_e2e_models:
        pytest.skip("No provider API keys are configured")

    for provider, model in configured_async_e2e_models:
        with subtests.test(provider=provider["name"], model=model):
            try:
                response = await async_chat_with_retry(
                    _adapter(provider, model),
                    messages=[
                        UserMessage(
                            'Return JSON with name="Alice" and age=30.'
                        )
                    ],
                    max_tokens=256,
                    json_schema=SIMPLE_SCHEMA,
                    timeout_s=60,
                )
            except JSONSchemaError as exc:
                pytest.skip(f"Model returned non-JSON in schema mode: {exc}")

            assert isinstance(response.content, str)
            assert response.parsed_json == {"name": "Alice", "age": 30}
            _assert_usage_and_pricing(response)


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_async_streaming_preserves_callbacks_and_final_response(
    subtests,
    configured_async_e2e_models,
    async_stream_with_retry,
):
    if not configured_async_e2e_models:
        pytest.skip("No provider API keys are configured")

    for provider, model in configured_async_e2e_models:
        with subtests.test(provider=provider["name"], model=model):
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
                _adapter(provider, model),
                messages=[UserMessage("Reply with exactly: OK")],
                max_tokens=256,
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
async def test_async_tools_round_trip_with_previous_response(
    subtests,
    configured_async_e2e_models,
    async_chat_with_retry,
):
    if not configured_async_e2e_models:
        pytest.skip("No provider API keys are configured")

    for provider, model in configured_async_e2e_models:
        with subtests.test(provider=provider["name"], model=model):
            adapter = _adapter(provider, model)
            messages = [
                UserMessage(
                    "What is the popularity of banana? "
                    "Use the available tool to look it up."
                )
            ]
            first = await async_chat_with_retry(
                adapter,
                messages=messages,
                tools=[FRUIT_TOOL],
                tool_choice="get_fruit_popularity",
                max_tokens=256,
                timeout_s=60,
            )

            assert first.tool_calls
            messages.append(AIMessage(content=first.content or "", tool_calls=first.tool_calls))
            for tool_call in first.tool_calls:
                assert tool_call.name == FRUIT_TOOL.name
                fruit = tool_call.arguments["fruit"]
                assert fruit in FRUIT_POPULARITY
                messages.append(
                    ToolMessage(
                        tool_call_id=tool_call.call_id,
                        content=json.dumps(
                            {
                                "fruit": fruit,
                                "popularity": FRUIT_POPULARITY[fruit],
                            }
                        ),
                    )
                )

            final = await async_chat_with_retry(
                adapter,
                messages=messages,
                previous_response=first,
                max_tokens=256,
                timeout_s=60,
            )
            assert final.content and final.content.strip()
            assert not final.tool_calls


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_async_image_and_document_inputs_return_text(
    subtests,
    configured_async_e2e_models,
    vision_image_bytes,
    pdf_bytes,
    async_chat_with_retry,
):
    if not configured_async_e2e_models:
        pytest.skip("No provider API keys are configured")

    for provider, model in configured_async_e2e_models:
        with subtests.test(provider=provider["name"], model=model, input="image"):
            image_response = await async_chat_with_retry(
                _adapter(provider, model),
                messages=[
                    UserMessage(
                        "Describe this image in one short sentence.",
                        files=[ImagePart(data=vision_image_bytes, media_type="image/png")],
                    )
                ],
                max_tokens=128,
                timeout_s=60,
            )
            assert image_response.content and image_response.content.strip()

        if provider["name"] not in {"anthropic", "google"}:
            continue

        with subtests.test(provider=provider["name"], model=model, input="pdf"):
            document_response = await async_chat_with_retry(
                _adapter(provider, model),
                messages=[
                    UserMessage(
                        "Summarize this document in one sentence.",
                        files=[
                            DocumentPart(
                                data=pdf_bytes,
                                media_type="application/pdf",
                            )
                        ],
                    )
                ],
                max_tokens=128,
                timeout_s=60,
            )
            assert document_response.content and document_response.content.strip()


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_async_errors_are_normalized(
    subtests,
    async_e2e_models,
    configured_async_e2e_models,
):
    for provider, model in async_e2e_models:
        with subtests.test(provider=provider["name"], model=model, error="auth"):
            with pytest.raises(LLMAPIAuthorizationError):
                await _adapter(
                    {**provider, "api_key": "NON_VALID_KEY"}, model
                ).achat(
                    messages=[UserMessage("Say OK")],
                    max_tokens=32,
                    timeout_s=10,
                )

    if not configured_async_e2e_models:
        pytest.skip("No provider API keys are configured for timeout checks")

    for provider, model in configured_async_e2e_models:
        with subtests.test(provider=provider["name"], model=model, error="timeout"):
            with pytest.raises(LLMAPITimeoutError):
                await _adapter(provider, model).achat(
                    messages=[UserMessage("Say OK")],
                    max_tokens=32,
                    timeout_s=0.1,
                )
