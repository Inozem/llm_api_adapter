import pytest

from llm_api_adapter.models.messages.chat_message import UserMessage
from llm_api_adapter.models.messages.file_parts import ImagePart


_PROMPT = "What do you see in this image? One sentence."


@pytest.mark.e2e
@pytest.mark.e2e_feature("image_input")
def test_vision_bytes_returns_non_empty_response(
    subtests,
    iter_organization_models,
    vision_image_bytes,
    chat_with_retry,
    e2e_adapter,
):
    for p, model in iter_organization_models():
        with subtests.test(provider=p["name"], model=model):
            adapter = e2e_adapter(p, model)
            msg = UserMessage(_PROMPT, files=[ImagePart(data=vision_image_bytes, media_type="image/png")])
            resp = chat_with_retry(
                adapter,
                messages=[msg],
                max_tokens=150,
                reasoning_level="none",
            )
            assert isinstance(resp.content, str)
            assert len(resp.content) > 0
