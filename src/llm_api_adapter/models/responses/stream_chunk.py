"""Public value object for visible streaming chunks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .chat_response import Usage


@dataclass(frozen=True)
class StreamChunk:
    """A visible text chunk with local stream-observability metadata.

    ``usage`` is copied when a chunk is created.  A chunk therefore never
    exposes a mutable usage object held by an adapter's internal accumulator.
    """

    text: str
    index: int
    elapsed_s: float
    delta_s: float
    usage: Optional[Usage] = None
    output_tokens_delta: Optional[int] = None

    def __post_init__(self) -> None:
        if self.usage is not None:
            object.__setattr__(
                self,
                "usage",
                Usage(
                    input_tokens=self.usage.input_tokens,
                    output_tokens=self.usage.output_tokens,
                    total_tokens=self.usage.total_tokens,
                ),
            )
