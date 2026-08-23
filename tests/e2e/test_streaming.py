import pytest

from src.llm_api_adapter.models.messages.chat_message import UserMessage
from src.llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter


pytestmark = pytest.mark.e2e_builtin


@pytest.mark.e2e
def test_stream_chat_returns_text_and_finalized_response(
    subtests,
    iter_provider_models,
    stream_with_retry,
):
    configured_models = 0

    for provider, model in iter_provider_models():
        if not provider["api_key"]:
            continue
        configured_models += 1

        with subtests.test(provider=provider["name"], model=model):
            adapter = UniversalLLMAPIAdapter(
                organization=provider["name"],
                model=model,
                api_key=provider["api_key"],
            )
            completed_responses = []
            observed_chunks = []

            def reset_observers():
                completed_responses.clear()
                observed_chunks.clear()

            text_chunks = stream_with_retry(
                adapter,
                messages=[UserMessage("Reply with exactly: OK")],
                max_tokens=1026,
                timeout_s=60,
                buffer_chars=8,
                on_chunk=observed_chunks.append,
                on_done=completed_responses.append,
                on_retry=reset_observers,
            )

            streamed_text = "".join(text_chunks)
            assert streamed_text.strip()
            assert [chunk.text for chunk in observed_chunks] == text_chunks
            assert all(len(chunk.text) <= 8 for chunk in observed_chunks)
            assert [chunk.index for chunk in observed_chunks] == list(
                range(len(observed_chunks))
            )
            assert [chunk.elapsed_s for chunk in observed_chunks] == sorted(
                chunk.elapsed_s for chunk in observed_chunks
            )
            assert all(chunk.delta_s >= 0 for chunk in observed_chunks)
            assert len(completed_responses) == 1

            response = completed_responses[0]
            assert isinstance(response.model, str) and response.model
            assert response.content == streamed_text
            if response.usage is not None:
                assert response.usage.input_tokens >= 0
                assert response.usage.output_tokens >= 0
                assert response.usage.total_tokens >= response.usage.input_tokens
            for chunk in observed_chunks:
                if chunk.usage is not None:
                    assert chunk.usage.input_tokens >= 0
                    assert chunk.usage.output_tokens >= 0
                    assert chunk.usage.total_tokens >= chunk.usage.input_tokens
                if chunk.output_tokens_delta is not None:
                    assert chunk.output_tokens_delta >= 0
            assert isinstance(response.finish_reason, str) and response.finish_reason

    if configured_models == 0:
        pytest.skip("No provider API keys are configured")
