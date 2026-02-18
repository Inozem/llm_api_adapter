from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """
    Provider-agnostic tool/function specification.

    json_schema: JSON Schema object (dict). Validation is performed in adapters.
    """
    name: str
    json_schema: Dict[str, Any]
    description: Optional[str] = None
