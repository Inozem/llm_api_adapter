from math import isclose
import time

import pytest

from llm_api_adapter.models.messages.chat_message import UserMessage     
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter


@pytest.mark.e2e
def test_chat_accepts_basic_params_and_returns_contract(providers, subtests):
    for p in providers:
        for model in p["models"]:
            with subtests.test(provider=p["name"], model=model):
                time.sleep(3)

                adapter = UniversalLLMAPIAdapter(
                    organization=p["name"],
                    model=model,
                    api_key=p["api_key"],
                )
                resp = adapter.chat(
                    messages=[UserMessage("Say 'OK'.")],
                    max_tokens=1026,
                    temperature=1.0,
                    top_p=1.0,
                    timeout_s=60
                )

                assert isinstance(resp.content, str)
                assert isinstance(resp.finish_reason, str)

                assert resp.usage is not None
                assert resp.usage.input_tokens >= 0
                assert resp.usage.output_tokens >= 0
                assert resp.usage.total_tokens >= resp.usage.input_tokens

                assert resp.currency
                assert resp.cost_input >= 0
                assert resp.cost_output >= 0
                assert resp.cost_total >= 0
                assert abs(resp.cost_total - (resp.cost_input + resp.cost_output)) < 1e-9


@pytest.mark.e2e
def test_chat_with_reasoning_level_returns_valid_contract(providers, subtests):
    for p in providers:
        for model in p["models"]:
            with subtests.test(provider=p["name"], model=model):
                time.sleep(3)

                adapter = UniversalLLMAPIAdapter(
                    organization=p["name"],
                    model=model,
                    api_key=p["api_key"],
                )
                resp = adapter.chat(
                    messages=[{"role": "user", "content": "Say 'OK'."}],
                    max_tokens=1026,
                    reasoning_level=1024,
                    timeout_s=60
                )

                assert isinstance(resp.content, str)
                assert isinstance(resp.finish_reason, str) and resp.finish_reason

                assert resp.usage is not None
                assert resp.usage.input_tokens >= 0
                assert resp.usage.output_tokens >= 0
                assert resp.usage.total_tokens >= 0
                assert resp.usage.total_tokens >= resp.usage.input_tokens + resp.usage.output_tokens

                assert isinstance(resp.currency, str) and resp.currency
                assert resp.cost_input is not None and resp.cost_input >= 0
                assert resp.cost_output is not None and resp.cost_output >= 0
                assert resp.cost_total is not None and resp.cost_total >= 0
                assert isclose(resp.cost_total, resp.cost_input + resp.cost_output, rel_tol=0, abs_tol=1e-9)
