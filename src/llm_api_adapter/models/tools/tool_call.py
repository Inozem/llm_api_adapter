from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True, slots=True)
class ToolCall:
    """
    Normalized tool call emitted by the model.

    arguments must be a parsed dict (provider parsing converts JSON -> dict).
    """
    name: str
    arguments: Dict[str, Any]
    call_id: Optional[str] = None
