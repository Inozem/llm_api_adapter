"""Focused live checks for the Mistral-hosted GLM Flash route."""

from math import isclose

import pytest

from llm_api_adapter.llm_registry.llm_registry import LLM_REGISTRY
from llm_api_adapter.models.messages.chat_message import UserMessage
from llm_api_adapter.models.messages.file_parts import DocumentPart
from tests.e2e import harness
from tests.e2e.conftest import e2e_model_case_parameters


@pytest.mark.e2e
@pytest.mark.e2e_capability("reasoning_control")
@pytest.mark.parametrize(
    "e2e_model_case",
    e2e_model_case_parameters("zai", service_provider="mistral"),
)
@pytest.mark.parametrize(
    ("reasoning_level", "capture_reasoning", "pdf_accounting"),
    (
        pytest.param("low", False, False, id="low-uncaptured"),
        pytest.param("low", True, False, id="low-captured"),
        pytest.param("high", False, False, id="high"),
        pytest.param("max", False, False, id="max"),
        pytest.param("none", False, False, id="none-fallback"),
        pytest.param(
            None,
            False,
            True,
            marks=pytest.mark.e2e_capability("pdf_bytes"),
            id="pdf-ocr-accounting",
        ),
    ),
)
def test_flash_reasoning_control(
    request,
    e2e_model_case,
    e2e_model_organization,
    chat_with_retry,
    e2e_adapter,
    reasoning_level,
    capture_reasoning,
    pdf_accounting,
):
    model = e2e_model_case.model_spec
    assert model is not None
    assert e2e_model_case.organization == "zai"
    assert e2e_model_case.service_provider == "mistral"

    adapter = e2e_adapter(
        e2e_model_organization,
        model.name,
        organization_name=e2e_model_case.organization,
        service_provider=e2e_model_case.service_provider,
    )
    assert adapter.organization == "zai"
    assert adapter.service_provider == "mistral"
    assert adapter.model == "glm-5.3-flash"
    assert adapter.model_spec.provider_model == "zai-glm-5-3-flash"

    if pdf_accounting:
        pdf_bytes = request.getfixturevalue("pdf_bytes")
        response = chat_with_retry(
            adapter,
            messages=[
                harness.make_document_message(
                    "Summarize this document in one sentence.",
                    DocumentPart(data=pdf_bytes, media_type="application/pdf"),
                )
            ],
            max_tokens=256,
            timeout_s=60,
        )
        _assert_pdf_ocr_accounting(response, adapter.model_spec)
        return

    assert model.reasoning_capability.allowed_values == ("low", "high", "max")
    if reasoning_level == "none":
        with pytest.warns(UserWarning, match="cannot disable reasoning"):
            response = chat_with_retry(
                adapter,
                messages=[UserMessage("Reply with OK.")],
                max_tokens=256,
                reasoning_level=reasoning_level,
                capture_reasoning=capture_reasoning,
                timeout_s=60,
            )
    else:
        response = chat_with_retry(
            adapter,
            messages=[UserMessage("Reply with OK.")],
            max_tokens=256,
            reasoning_level=reasoning_level,
            capture_reasoning=capture_reasoning,
            timeout_s=60,
        )

    assert type(response.content) is str
    assert bool(response.content)
    assert type(response.finish_reason) is str
    assert bool(response.finish_reason)
    if capture_reasoning:
        assert all(isinstance(event.text, str) for event in response.reasoning_events)
    else:
        assert len(response.reasoning_events) == 0
    _assert_profile_token_pricing(response, adapter.model_spec)


def _assert_pdf_ocr_accounting(response, model_spec):
    _assert_profile_token_pricing(response, model_spec)
    meter = LLM_REGISTRY.organizations["mistral"].metered_operations["ocr"]
    ocr_items = [
        item for item in response.cost_breakdown or () if item.operation == "ocr"
    ]
    if ocr_items:
        assert all(item.model == meter.model for item in ocr_items)
        assert all(item.unit == meter.unit for item in ocr_items)
        assert all(item.currency == meter.currency for item in ocr_items)
        assert all(item.quantity >= 0 for item in ocr_items)
        assert all(
            isclose(item.rate, meter.rate, rel_tol=0, abs_tol=1e-12)
            for item in ocr_items
        )
        assert all(
            isclose(item.cost, item.quantity * meter.rate, rel_tol=0, abs_tol=1e-12)
            for item in ocr_items
        )
    else:
        # No page count means the incurred OCR operation cannot be priced.
        assert response.cost_total is None

    if response.cost_total is not None:
        assert all(item.currency == response.currency for item in ocr_items)
        assert response.cost_input is not None
        assert response.cost_output is not None
        assert isclose(
            response.cost_total,
            response.cost_input
            + response.cost_output
            + sum(item.cost for item in response.cost_breakdown or ()),
            rel_tol=0,
            abs_tol=1e-9,
        )


def _assert_profile_token_pricing(response, model_spec):
    """Check only token components that provider-reported usage can price."""
    harness.assert_usage_and_pricing(response)
    assert response.currency == model_spec.pricing_tiers.currency
    usage = response.usage
    assert usage is not None

    if usage.input_tokens is None:
        assert response.cost_input is None
    elif usage.cached_tokens is None:
        # Inclusive input count alone cannot distinguish standard from cache reads.
        assert response.cost_input is None
        assert response.cost_total is None
    else:
        tier = model_spec.pricing_tiers.tier_for_prompt_tokens(usage.input_tokens)
        cache_reads = usage.cached_tokens
        cache_writes = usage.cache_write_tokens
        cache_rates_available = (
            (cache_reads == 0 or tier.cache_read_in_per_token is not None)
            and not (cache_writes is None and tier.cache_write_in_per_token is not None)
            and (cache_writes in (None, 0) or tier.cache_write_in_per_token is not None)
        )
        cache_counts_valid = (
            cache_reads >= 0
            and (cache_writes is None or cache_writes >= 0)
            and cache_reads + (cache_writes or 0) <= usage.input_tokens
        )
        if not cache_rates_available or not cache_counts_valid:
            assert response.cost_input is None
            assert response.cost_total is None
        else:
            expected_input = (
                (usage.input_tokens - cache_reads - (cache_writes or 0))
                * tier.in_per_token
                + cache_reads * (tier.cache_read_in_per_token or 0)
                + (cache_writes or 0) * (tier.cache_write_in_per_token or 0)
            )
            assert response.cost_input is not None
            assert isclose(response.cost_input, expected_input, rel_tol=0, abs_tol=1e-9)

    if usage.output_tokens is None or usage.input_tokens is None:
        assert response.cost_output is None
    else:
        tier = model_spec.pricing_tiers.tier_for_prompt_tokens(usage.input_tokens)
        expected_output = usage.output_tokens * tier.out_per_token
        assert response.cost_output is not None
        assert isclose(response.cost_output, expected_output, rel_tol=0, abs_tol=1e-9)

    if (
        usage.input_tokens is None
        or usage.output_tokens is None
        or usage.cached_tokens is None
        or (
            usage.cache_write_tokens is not None
            and usage.cache_write_tokens > 0
            and model_spec.pricing_tiers.tier_for_prompt_tokens(
                usage.input_tokens or 0
            ).cache_write_in_per_token is None
        )
        or (
            usage.input_tokens is not None
            and usage.cached_tokens is not None
            and usage.cached_tokens + (usage.cache_write_tokens or 0)
            > usage.input_tokens
        )
    ):
        assert response.cost_total is None
    if response.cost_input is None:
        assert response.cost_total is None
