import pytest

from llm_api_adapter.models.messages.chat_message import UserMessage
from llm_api_adapter.models.messages.file_parts import ImagePart
from tests.e2e.conftest import e2e_model_case_parameters


_PROMPT = "What do you see in this image? One sentence."


@pytest.mark.e2e
@pytest.mark.e2e_capability("image_bytes")
@pytest.mark.parametrize("e2e_model_case", e2e_model_case_parameters())
def test_vision_bytes_returns_non_empty_response(
    e2e_model_case,
    e2e_model_organization,
    vision_image_bytes,
    chat_with_retry,
    e2e_adapter,
):
    model = e2e_model_case.model_spec
    assert model is not None
    adapter = e2e_adapter(e2e_model_organization, model.name)
    msg = UserMessage(
        _PROMPT,
        files=[ImagePart(data=vision_image_bytes, media_type="image/png")],
    )
    resp = chat_with_retry(
        adapter,
        messages=[msg],
        max_tokens=1200,
        reasoning_level="none",
    )
    assert isinstance(resp.content, str)
    assert len(resp.content) > 0
