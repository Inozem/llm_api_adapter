import pytest

from llm_api_adapter.models.messages.file_parts import DocumentPart
from tests.e2e import harness
from tests.e2e.conftest import e2e_model_case_parameters


_PROMPT = "Summarize this document in one sentence."


@pytest.mark.e2e
@pytest.mark.e2e_capability("pdf_bytes")
@pytest.mark.parametrize("e2e_model_case", e2e_model_case_parameters())
def test_document_bytes_returns_non_empty_response(
    e2e_model_case,
    e2e_model_organization,
    pdf_bytes,
    chat_with_retry,
    e2e_adapter,
):
    model = e2e_model_case.model_spec
    assert model is not None
    if not e2e_model_organization["api_key"]:
        pytest.skip("No organization API key is configured")

    adapter = e2e_adapter(e2e_model_organization, model.name)
    msg = harness.make_document_message(
        _PROMPT,
        DocumentPart(data=pdf_bytes, media_type="application/pdf"),
    )
    resp = chat_with_retry(adapter, messages=[msg], max_tokens=150)
    assert isinstance(resp.content, str)
    assert len(resp.content) > 0
