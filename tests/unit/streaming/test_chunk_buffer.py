from dataclasses import FrozenInstanceError

import pytest

from src.llm_api_adapter.models import StreamChunk
from src.llm_api_adapter.models.responses.chat_response import Usage
from src.llm_api_adapter.llms.streaming import StreamChunkBuffer


class FakeClock:
    def __init__(self, *values):
        self._values = iter(values)

    def __call__(self):
        return next(self._values)


@pytest.mark.unit
def test_stream_chunk_is_exported_from_models_package():
    assert StreamChunk.__name__ == "StreamChunk"


@pytest.mark.unit
def test_buffer_default_passthrough_preserves_text_and_reports_timing():
    buffer = StreamChunkBuffer(clock=FakeClock(10.0, 10.2, 10.7))

    chunks = [*buffer.add("hello"), *buffer.add(" world")]

    assert [chunk.text for chunk in chunks] == ["hello", " world"]
    assert [chunk.index for chunk in chunks] == [0, 1]
    assert [chunk.elapsed_s for chunk in chunks] == pytest.approx([0.2, 0.7])
    assert [chunk.delta_s for chunk in chunks] == pytest.approx([0.2, 0.5])


@pytest.mark.unit
def test_buffer_coalesces_splits_and_flushes_remaining_text():
    buffer = StreamChunkBuffer(buffer_chars=4, clock=FakeClock(0.0, 1.0, 2.0))

    chunks = [*buffer.add("ab"), *buffer.add("cdefghi"), *buffer.flush()]

    assert [chunk.text for chunk in chunks] == ["abcd", "efgh", "i"]
    assert [chunk.index for chunk in chunks] == [0, 1, 2]
    assert [chunk.elapsed_s for chunk in chunks] == [1.0, 1.0, 2.0]
    assert [chunk.delta_s for chunk in chunks] == [1.0, 0.0, 1.0]


@pytest.mark.unit
def test_buffer_preserves_usage_as_a_snapshot_and_consumes_token_delta():
    usage = Usage(input_tokens=3, output_tokens=5, total_tokens=8)
    buffer = StreamChunkBuffer(clock=FakeClock(0.0, 0.1, 0.2))

    first_chunk = next(buffer.add("first", usage=usage, output_tokens_delta=5))
    usage.output_tokens = 99
    second_chunk = next(buffer.add("second"))

    assert first_chunk.usage == Usage(3, 5, 8)
    assert second_chunk.usage == Usage(3, 5, 8)
    assert first_chunk.usage is not second_chunk.usage
    assert first_chunk.output_tokens_delta == 5
    assert second_chunk.output_tokens_delta is None
    with pytest.raises(FrozenInstanceError):
        first_chunk.index = 1


@pytest.mark.unit
@pytest.mark.parametrize("buffer_chars", [0, -1, True, 1.5, "4"])
def test_buffer_rejects_invalid_buffer_chars(buffer_chars):
    with pytest.raises(ValueError, match="buffer_chars must be None or a positive integer"):
        StreamChunkBuffer(buffer_chars)


@pytest.mark.unit
def test_buffer_updates_metadata_without_text_until_a_chunk_is_emitted():
    usage = Usage(input_tokens=2, output_tokens=1, total_tokens=3)
    buffer = StreamChunkBuffer(clock=FakeClock(0.0, 0.4))

    buffer.update_metadata(usage=usage, output_tokens_delta=1)
    chunk = next(buffer.add("text"))

    assert chunk.usage == usage
    assert chunk.output_tokens_delta == 1
