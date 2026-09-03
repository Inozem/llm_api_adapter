import pytest

from llm_api_adapter.llm_registry.llm_registry import LLM_REGISTRY
from llm_api_adapter.models.messages.chat_message import UserMessage
from llm_api_adapter.models.messages.file_parts import DocumentPart
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter


@pytest.mark.e2e
def test_mistral_pdf_ocr_exposes_cost_breakdown(
    e2e_organization_profile,
    organizations,
    pdf_bytes,
    chat_with_retry,
):
    if e2e_organization_profile.name != "mistral":
        pytest.skip("Mistral-specific OCR cost contract")

    organization = organizations[0]
    meter = LLM_REGISTRY.organizations["mistral"].metered_operations["ocr"]
    adapter = UniversalLLMAPIAdapter(
        organization="mistral",
        model=organization["latest_model"],
        api_key=organization["api_key"],
    )

    response = chat_with_retry(
        adapter,
        messages=[
            UserMessage(
                "Summarize this document in one sentence.",
                files=[DocumentPart(data=pdf_bytes, media_type="application/pdf")],
            )
        ],
        max_tokens=150,
    )

    assert response.cost_input is not None
    assert response.cost_output is not None
    assert response.cost_breakdown is not None
    ocr_line_items = [
        item for item in response.cost_breakdown if item.operation == "ocr"
    ]
    assert len(ocr_line_items) == 1

    ocr_cost = ocr_line_items[0]
    assert ocr_cost.model == meter.model
    assert ocr_cost.unit == meter.unit
    assert ocr_cost.quantity > 0
    assert ocr_cost.rate == meter.rate
    assert ocr_cost.currency == meter.currency
    assert ocr_cost.cost == pytest.approx(ocr_cost.quantity * ocr_cost.rate)

    assert response.cost_total == pytest.approx(
        response.cost_input + response.cost_output + ocr_cost.cost
    )
