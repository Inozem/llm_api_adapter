"""Bounded live checks for Z.ai's declared GLM Flash contract."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from llm_api_adapter.models.messages.chat_message import (
    AIMessage,
    ToolMessage,
    UserMessage,
)
from llm_api_adapter.models.messages.file_parts import ImagePart
from llm_api_adapter.models.tools.tool_spec import ToolSpec
from tests.e2e import conftest as core_e2e
from tests.e2e import harness as e2e_harness


_MODEL = "glm-5.3-flash"
_MAX_TOKENS = 256
_REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
_IMAGE_BYTES = (_REPOSITORY_ROOT / "tests" / "fixtures" / "test_image.png").read_bytes()
_IMAGE_DATA_URL = (
    "data:image/png;base64," + base64.b64encode(_IMAGE_BYTES).decode("ascii")
)
_WEATHER_TOOL = ToolSpec(
    name="get_weather",
    description="Return a fixed weather result for a city.",
    json_schema={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
        "additionalProperties": False,
    },
)


def _zai_organization():
    profile = core_e2e.get_e2e_organization_profile("zai")
    organizations = core_e2e.resolve_e2e_organizations(profile)
    assert len(organizations) == 1
    organization = organizations[0]
    assert organization["models"] == [_MODEL]
    return organization


@pytest.mark.e2e
@pytest.mark.e2e_zai
def test_zai_flash_live_reasoning_uses_core_level_resolution():
    """Exercise a canonical Core level and keep reasoning out of visible text."""
    organization = _zai_organization()
    response = e2e_harness.chat_with_transient_retry(
        e2e_harness.create_e2e_adapter(organization, _MODEL),
        messages=[UserMessage("Reply with exactly: OK")],
        max_tokens=_MAX_TOKENS,
        reasoning_level="medium",
        capture_reasoning=True,
        timeout_s=60,
    )

    assert response.content and response.content.strip() == "OK"
    assert response.reasoning_events
    assert all(event.text not in response.content for event in response.reasoning_events)
    assert response.usage is not None


@pytest.mark.e2e
@pytest.mark.e2e_zai
def test_zai_flash_live_function_tools_use_auto_and_tool_history():
    """Exercise an automatic function call and a normalized tool-result turn."""
    organization = _zai_organization()
    adapter = e2e_harness.create_e2e_adapter(organization, _MODEL)
    messages = [
        UserMessage(
            "Call get_weather for Paris. Do not answer until the tool result "
            "has been returned."
        )
    ]
    first = e2e_harness.chat_with_transient_retry(
        adapter,
        messages=messages,
        tools=[_WEATHER_TOOL],
        tool_choice="auto",
        max_tokens=_MAX_TOKENS,
        timeout_s=60,
    )

    assert first.tool_calls
    messages.append(AIMessage(content=first.content or "", tool_calls=first.tool_calls))
    for tool_call in first.tool_calls:
        assert tool_call.name == _WEATHER_TOOL.name
        assert isinstance(tool_call.arguments, dict)
        messages.append(
            ToolMessage(
                tool_call_id=tool_call.call_id,
                content=json.dumps({"city": "Paris", "temperature_c": 18}),
            )
        )

    final = e2e_harness.chat_with_transient_retry(
        adapter,
        messages=messages,
        tools=[_WEATHER_TOOL],
        max_tokens=_MAX_TOKENS,
        timeout_s=60,
    )
    assert final.content and final.content.strip()
    assert not final.tool_calls


@pytest.mark.e2e
@pytest.mark.e2e_zai
@pytest.mark.parametrize(
    ("image_name", "image"),
    [
        (
            "url",
            ImagePart(url="https://httpbin.org/image/png", media_type="image/png"),
        ),
        ("data_url", ImagePart(url=_IMAGE_DATA_URL)),
    ],
)
def test_zai_flash_live_image_url_and_data_url(image_name, image):
    """Exercise both documented image URL forms through the public facade."""
    organization = _zai_organization()
    response = e2e_harness.chat_with_transient_retry(
        e2e_harness.create_e2e_adapter(organization, _MODEL),
        messages=[UserMessage(f"Describe this {image_name} image briefly.", files=[image])],
        max_tokens=_MAX_TOKENS,
        timeout_s=60,
    )

    assert response.content and response.content.strip()


@pytest.mark.e2e
@pytest.mark.e2e_zai
def test_zai_flash_live_usage_and_cache_pricing_are_reported_when_valid():
    """Check standard pricing and validate any provider-reported cache split."""
    organization = _zai_organization()
    adapter = e2e_harness.create_e2e_adapter(organization, _MODEL)
    prompt = "Return exactly OK after reading this stable cache prefix. " + (
        "This sentence is intentionally repeated for cache accounting. " * 600
    )

    responses = [
        e2e_harness.chat_with_transient_retry(
            adapter,
            messages=[UserMessage(prompt)],
            max_tokens=_MAX_TOKENS,
            timeout_s=60,
        )
        for _ in range(2)
    ]

    for response in responses:
        assert response.content and response.content.strip() == "OK"
        assert response.usage is not None
        assert response.usage.total_tokens == (
            response.usage.input_tokens + response.usage.output_tokens
        )
        assert response.currency == "USD"
        assert response.cost_input is not None
        assert response.cost_output is not None
        assert response.cost_total == pytest.approx(
            response.cost_input + response.cost_output,
        )
        cached_tokens = getattr(response.usage, "cached_tokens", None)
        if cached_tokens is not None:
            assert 0 <= cached_tokens <= response.usage.input_tokens
