"""Immutable provider-neutral reasoning observability events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ReasoningEventKind = Literal["summary", "content"]


@dataclass(frozen=True)
class ReasoningEvent:
    """A normalized provider-emitted reasoning fragment.

    The event is deliberately immutable so callbacks and final responses can
    safely share the same value without exposing collector state.
    """

    text: str
    kind: ReasoningEventKind
    index: int
    elapsed_s: float
    delta_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        if self.kind not in ("summary", "content"):
            raise ValueError("kind must be 'summary' or 'content'")


__all__ = ["ReasoningEvent", "ReasoningEventKind"]
