import time

import pytest

from llm_api_adapter.models.messages.chat_message import UserMessage
from llm_api_adapter.models.messages.file_parts import ImagePart
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter

_PROMPT = "What do you see in this image? One sentence."


@pytest.mark.e2e
def test_vision_bytes_returns_non_empty_response(providers, vision_image_bytes, subtests):
    for p in providers:
        for model in p["models"]:
            with subtests.test(provider=p["name"], model=model):
                time.sleep(3)
                adapter = UniversalLLMAPIAdapter(
                    organization=p["name"], model=model, api_key=p["api_key"]
                )
                msg = UserMessage(_PROMPT, files=[ImagePart(data=vision_image_bytes, media_type="image/png")])
                resp = adapter.chat(messages=[msg], max_tokens=150)
                assert isinstance(resp.content, str)
                assert len(resp.content) > 0
