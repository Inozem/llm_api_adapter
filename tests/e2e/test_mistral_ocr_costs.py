import pytest

from llm_api_adapter.llm_registry.llm_registry import LLM_REGISTRY
from llm_api_adapter.models.messages.file_parts import DocumentPart
from tests.e2e import harness
from tests.e2e.conftest import e2e_model_case_parameters


@pytest.mark.e2e
@pytest.mark.e2e_capability("pdf_bytes")
@pytest.mark.parametrize("e2e_model_case", e2e_model_case_parameters("mistral"))
def test_mistral_pdf_ocr_exposes_cost_breakdown(
    e2e_model_case,
    e2e_model_organization,
    pdf_bytes,
    chat_with_retry,
    e2e_adapter,
):
    model = e2e_model_case.model_spec
    assert model is not None
    meter = LLM_REGISTRY.organizations["mistral"].metered_operations["ocr"]
    adapter = e2e_adapter(e2e_model_organization, model.name)

    response = chat_with_retry(
        adapter,
        messages=[
            harness.make_document_message(
                "Summarize this document in one sentence.",
                DocumentPart(data=pdf_bytes, media_type="application/pdf"),
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
