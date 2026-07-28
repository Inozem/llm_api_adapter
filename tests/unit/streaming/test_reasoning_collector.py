import pytest

from src.llm_api_adapter.llms.streaming import StreamReasoningCollector
from src.llm_api_adapter.models import ReasoningEvent


class FakeClock:
    def __init__(self, *values):
        self._values = iter(values)

    def __call__(self):
        return next(self._values)


@pytest.mark.unit
def test_collector_assigns_sequential_indices_and_timing():
    collector = StreamReasoningCollector(clock=FakeClock(10.0, 10.2, 10.7))

    first = collector.add("first summary")
    second = collector.add("second thought", kind="content")

    assert first.text == "first summary"
    assert first.kind == "summary"
    assert first.index == 0
    assert first.elapsed_s == pytest.approx(0.2)
    assert first.delta_s == pytest.approx(0.2)
    assert second.text == "second thought"
    assert second.kind == "content"
    assert second.index == 1
    assert second.elapsed_s == pytest.approx(0.7)
    assert second.delta_s == pytest.approx(0.5)


@pytest.mark.unit
def test_collector_snapshot_isolated_from_internal_state():
    collector = StreamReasoningCollector(clock=FakeClock(0.0, 0.1))
    event = collector.add("summary")

    snapshot = collector.snapshot()
    snapshot.clear()

    assert collector.snapshot() == [event]
    assert collector.snapshot() is not snapshot


@pytest.mark.unit
def test_collector_skips_empty_fragments_without_consuming_clock():
    collector = StreamReasoningCollector(clock=FakeClock(0.0, 0.1))

    assert collector.add("") is None
    assert collector.snapshot() == []
    assert collector.add("summary").elapsed_s == pytest.approx(0.1)


@pytest.mark.unit
def test_collector_rejects_non_string_fragments():
    collector = StreamReasoningCollector(clock=FakeClock(0.0))

    with pytest.raises(TypeError, match="text must be a string"):
        collector.add(None)
