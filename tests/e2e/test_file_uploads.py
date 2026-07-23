import pytest

from llm_api_adapter.models.messages.chat_message import UserMessage
from llm_api_adapter.models.messages.file_parts import DocumentPart
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter


_PROMPT = "Summarize this document in one sentence."


@pytest.mark.e2e
def test_document_bytes_returns_non_empty_response(
    subtests,
    iter_provider_models,
    pdf_bytes,
    chat_with_retry,
):
    for p, model in iter_provider_models():
        if p["name"] not in {"anthropic", "google"}:
            continue
        if not p["api_key"]:
            continue

        with subtests.test(provider=p["name"], model=model):
            adapter = UniversalLLMAPIAdapter(
                organization=p["name"],
                model=model,
                api_key=p["api_key"],
            )
            msg = UserMessage(
                _PROMPT,
                files=[DocumentPart(data=pdf_bytes, media_type="application/pdf")],
            )
            resp = chat_with_retry(adapter, messages=[msg], max_tokens=150)
            assert isinstance(resp.content, str)
            assert len(resp.content) > 0
