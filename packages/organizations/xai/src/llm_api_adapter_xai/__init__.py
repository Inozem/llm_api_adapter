"""Official xAI Responses API support for :mod:`llm_api_adapter`."""

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from .adapter import XAIAdapter


def __getattr__(name: str) -> Any:
    """Load the adapter only when the public class is requested."""
    if name == "XAIAdapter":
        from .adapter import XAIAdapter

        return XAIAdapter
    raise AttributeError(name)


__all__ = ["XAIAdapter"]
