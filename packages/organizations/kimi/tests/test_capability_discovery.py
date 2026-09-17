"""Package-local contract checks for the Kimi discovery record.

The tests keep the pre-implementation admission matrix closed: a later change
cannot silently add a candidate, capability, evidence source, or unsupported
file path without changing the reviewed discovery record.
"""

from __future__ import annotations

import pytest


from packages.organizations.kimi.tests.fixtures.kimi_capability_discovery import (
    CANDIDATE_MODELS,
    KIMI_CAPABILITY_DISCOVERY,
    KIMI_CHAT_COMPLETION_RESPONSE,
    KIMI_FILE_EXTRACTION_RESPONSE,
    KIMI_IMAGE_DATA_URI_MESSAGE,
    KIMI_PUBLIC_IMAGE_URL,
    KIMI_STREAM_EVENTS,
    KIMI_STRUCTURED_OUTPUT_RESPONSE,
    KIMI_TOOL_CALL_RESPONSE,
    KIMI_TOOL_RESULT_MESSAGE,
    MATRIX_CAPABILITIES,
)


_FIXTURE_NAMES = {
    "KIMI_CHAT_COMPLETION_RESPONSE",
    "KIMI_TOOL_CALL_RESPONSE",
    "KIMI_STRUCTURED_OUTPUT_RESPONSE",
    "KIMI_STREAM_EVENTS",
    "KIMI_IMAGE_DATA_URI_MESSAGE",
    "KIMI_PUBLIC_IMAGE_URL",
    "KIMI_FILE_EXTRACTION_RESPONSE",
}


@pytest.fixture(scope="module")
def kimi_discovery_record() -> dict:
    return KIMI_CAPABILITY_DISCOVERY


@pytest.mark.unit
def test_discovery_matrix_is_closed_to_the_supported_candidates(
    kimi_discovery_record,
):
    assert kimi_discovery_record["candidate_models"] == CANDIDATE_MODELS
    assert tuple(kimi_discovery_record["models"]) == CANDIDATE_MODELS
    assert kimi_discovery_record["admission_policy"]["initial_public_models"] == ()
    assert kimi_discovery_record["admission_policy"]["initial_status"] == "blocked"


@pytest.mark.unit
def test_every_candidate_has_complete_reviewable_capability_evidence(
    kimi_discovery_record,
):
    sources = kimi_discovery_record["official_sources"]

    for model_name, model in kimi_discovery_record["models"].items():
        assert model_name in CANDIDATE_MODELS
        assert model["context_window_tokens"] > 0
        assert set(model["pricing_per_1m_usd"]) == {"cache_hit_input", "cache_miss_input", "output"}
        assert all(model["pricing_per_1m_usd"][key] > 0 for key in (
            "cache_hit_input", "cache_miss_input", "output",
        ))
        assert set(model["capabilities"]) == set(MATRIX_CAPABILITIES)

        for status, official_sources, fixture_name in model["capabilities"].values():
            assert status in {
                "pending_conformance",
                "pending_manual_e2e",
                "excluded",
            }
            assert official_sources
            assert all(source in sources for source in official_sources)
            assert fixture_name in _FIXTURE_NAMES


@pytest.mark.unit
def test_discovery_does_not_infer_unsupported_public_image_or_document_urls(
    kimi_discovery_record,
):
    for model in kimi_discovery_record["models"].values():
        assert model["capabilities"]["image_url"][0] == "excluded"
        assert model["capabilities"]["pdf_url"][0] == "excluded"
        assert model["capabilities"]["pdf_bytes"][0] == "pending_manual_e2e"


@pytest.mark.unit
def test_sanitized_fixtures_preserve_the_documented_kimi_protocol_boundaries():
    choice = KIMI_CHAT_COMPLETION_RESPONSE["choices"][0]
    assert choice["message"]["content"]
    assert choice["message"]["reasoning_content"]
    assert KIMI_CHAT_COMPLETION_RESPONSE["usage"]["cached_tokens"] > 0

    tool_call = KIMI_TOOL_CALL_RESPONSE["choices"][0]["message"]["tool_calls"][0]
    assert tool_call["id"] == KIMI_TOOL_RESULT_MESSAGE["tool_call_id"]
    assert tool_call["function"]["name"] == "get_weather"

    assert KIMI_STRUCTURED_OUTPUT_RESPONSE["choices"][0]["message"]["content"] == (
        '{"answer":"ok"}'
    )
    assert "reasoning_content" in KIMI_STRUCTURED_OUTPUT_RESPONSE["choices"][0]["message"]

    assert KIMI_IMAGE_DATA_URI_MESSAGE["content"][0]["image_url"]["url"].startswith(
        "data:image/"
    )
    assert KIMI_PUBLIC_IMAGE_URL.startswith("https://")
    assert KIMI_FILE_EXTRACTION_RESPONSE["purpose"] == "file-extract"
    assert KIMI_FILE_EXTRACTION_RESPONSE["status"] == "ready"

    final_event = KIMI_STREAM_EVENTS[-2]["data"]
    assert final_event["choices"][0]["finish_reason"] == "stop"
    assert final_event["usage"]["cached_tokens"] == 12
    assert KIMI_STREAM_EVENTS[-1]["data"] == "[DONE]"
