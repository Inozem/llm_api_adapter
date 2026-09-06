import pytest

from llm_api_adapter.models.messages.file_parts import DocumentPart
from tests.e2e import harness


_PROMPT = "Summarize this document in one sentence."


@pytest.mark.e2e
@pytest.mark.e2e_feature("document_input")
def test_document_bytes_returns_non_empty_response(
    subtests,
    iter_organization_models,
    pdf_bytes,
    chat_with_retry,
    e2e_adapter,
):
    for p, model in iter_organization_models():
        if not p["api_key"]:
            continue

        with subtests.test(provider=p["name"], model=model):
            adapter = e2e_adapter(p, model)
            msg = harness.make_document_message(
                _PROMPT,
                DocumentPart(data=pdf_bytes, media_type="application/pdf"),
            )
            resp = chat_with_retry(adapter, messages=[msg], max_tokens=150)
            assert isinstance(resp.content, str)
            assert len(resp.content) > 0
