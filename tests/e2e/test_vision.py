import time
from itertools import zip_longest

import pytest

from llm_api_adapter.models.messages.chat_message import UserMessage
from llm_api_adapter.models.messages.file_parts import ImagePart
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter

_PROMPT = "What do you see in this image? One sentence."


@pytest.mark.e2e
def test_vision_bytes_returns_non_empty_response(providers, vision_image_bytes, subtests):
    groups = list(zip_longest(*[p["models"] for p in providers]))
    for i, group in enumerate(groups):
        if i > 0:
            time.sleep(1)
        for p, model in zip(providers, group):
            if model is None:
                continue
            with subtests.test(provider=p["name"], model=model):
                adapter = UniversalLLMAPIAdapter(
                    organization=p["name"], model=model, api_key=p["api_key"]
                )
                msg = UserMessage(_PROMPT, files=[ImagePart(data=vision_image_bytes, media_type="image/png")])
                resp = adapter.chat(messages=[msg], max_tokens=150)
                assert isinstance(resp.content, str)
                assert len(resp.content) > 0
