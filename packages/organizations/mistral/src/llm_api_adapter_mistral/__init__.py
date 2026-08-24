"""Official direct-API support for Mistral models."""

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from .adapter import MistralAdapter


def __getattr__(name: str) -> Any:
    """Load the adapter only when the public class is requested."""
    if name == "MistralAdapter":
        from .adapter import MistralAdapter

        return MistralAdapter
    raise AttributeError(name)

__all__ = ["MistralAdapter"]
