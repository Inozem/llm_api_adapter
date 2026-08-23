"""Opt-in paid end-to-end checks for the Mistral provider package."""

from __future__ import annotations

import asyncio
from importlib.metadata import PackageNotFoundError, version
import os

import pytest

from llm_api_adapter.models.tools import ToolSpec
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter


pytestmark = [pytest.mark.e2e, pytest.mark.e2e_mistral]


@pytest.fixture(scope="session")
def mistral_api_key() -> str:
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        pytest.skip("MISTRAL_API_KEY is not configured")
    try:
        version("llm-api-adapter-mistral")
    except PackageNotFoundError:
        pytest.skip("llm-api-adapter-mistral is not installed")
    return api_key


@pytest.fixture
def mistral_adapter(mistral_api_key: str) -> UniversalLLMAPIAdapter:
    """Construct Mistral through the core's public provider discovery path."""
    return UniversalLLMAPIAdapter(
        organization="mistral",
        model=os.getenv("MISTRAL_E2E_MODEL", "mistral-small-2603"),
        api_key=mistral_api_key,
    )


def _assert_completed_response(response) -> None:
    assert response.content and response.content.strip()
    assert response.usage is not None
    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens > 0
    assert response.cost_total is not None
    assert response.cost_total > 0


def test_mistral_chat_json_schema_and_tools(mistral_adapter):
    json_response = mistral_adapter.chat(
        [{"role": "user", "content": "Reply with a short greeting."}],
        max_tokens=128,
        json_schema={
            "type": "object",
            "properties": {"greeting": {"type": "string"}},
            "required": ["greeting"],
        },
        timeout_s=60,
    )

    assert json_response.parsed_json is not None
    assert isinstance(json_response.parsed_json["greeting"], str)
    _assert_completed_response(json_response)

    tool = ToolSpec(
        name="get_fruit_popularity",
        description="Return a popularity score for a fruit.",
        json_schema={
            "type": "object",
            "properties": {"fruit": {"type": "string"}},
            "required": ["fruit"],
        },
    )
    tool_response = mistral_adapter.chat(
        [
            {
                "role": "user",
                "content": (
                    "Call get_fruit_popularity for banana. "
                    "Do not answer in prose."
                ),
            }
        ],
        max_tokens=128,
        tools=[tool],
        tool_choice="get_fruit_popularity",
        timeout_s=60,
    )

    assert tool_response.tool_calls is not None
    assert len(tool_response.tool_calls) == 1
    assert tool_response.tool_calls[0].name == "get_fruit_popularity"
    assert tool_response.tool_calls[0].arguments["fruit"].lower() == "banana"
    assert tool_response.usage is not None


def test_mistral_sync_stream_reports_text_usage_and_callbacks(mistral_adapter):
    deltas = []
    chunks = []
    completed = []

    visible_text = list(
        mistral_adapter.stream_chat(
            [{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=128,
            timeout_s=60,
            buffer_chars=2,
            on_delta=deltas.append,
            on_chunk=chunks.append,
            on_done=completed.append,
        )
    )

    assert visible_text
    assert deltas == visible_text
    assert [chunk.text for chunk in chunks] == visible_text
    assert len(completed) == 1
    assert completed[0].content == "".join(visible_text)
    _assert_completed_response(completed[0])


def test_mistral_async_chat_and_stream(mistral_adapter):
    async def run() -> tuple[object, list[str], list[str], list[object]]:
        chat_response = await mistral_adapter.achat(
            [{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=128,
            timeout_s=60,
        )
        deltas: list[str] = []
        completed: list[object] = []

        async def on_delta(text: str) -> None:
            deltas.append(text)

        async def on_done(response) -> None:
            completed.append(response)

        visible_text = [
            text
            async for text in mistral_adapter.astream_chat(
                [{"role": "user", "content": "Reply with exactly: OK"}],
                max_tokens=128,
                timeout_s=60,
                buffer_chars=2,
                on_delta=on_delta,
                on_done=on_done,
            )
        ]
        return chat_response, visible_text, deltas, completed

    chat_response, visible_text, deltas, completed = asyncio.run(run())

    _assert_completed_response(chat_response)
    assert visible_text
    assert deltas == visible_text
    assert len(completed) == 1
    assert completed[0].content == "".join(visible_text)
    _assert_completed_response(completed[0])
