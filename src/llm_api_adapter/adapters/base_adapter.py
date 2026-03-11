from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
import logging
import re
from typing import Any, Dict, List, Optional
import warnings

from ..errors.llm_api_error import InvalidToolSchemaError, ToolChoiceError
from ..llm_registry.llm_registry import Pricing, LLM_REGISTRY
from ..models.messages.chat_message import Messages
from ..models.responses.chat_response import ChatResponse
from ..models.tools import ToolSpec

logger = logging.getLogger(__name__)

REASONING_LEVELS_DEFAULT = {
    "none": 0,
    "low": 100,
    "medium": 1000,
    "high": 10000,
}

TOOL_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


@dataclass
class LLMAdapterBase(ABC):
    api_key: str
    model: str
    company: str
    pricing: Optional[Pricing] = None
    is_reasoning: bool = False
    reasoning_levels: Dict[str, int] = field(
        default_factory=lambda: REASONING_LEVELS_DEFAULT.copy()
    )

    def __post_init__(self):
        if not self.api_key:
            error_message = "api_key must be a non-empty string"
            logger.error(error_message)
            raise ValueError(error_message)
        provider = LLM_REGISTRY.providers.get(self.company)
        model_spec = provider.models.get(self.model) if provider else None
        if not model_spec:
            warnings.warn(
                (
                    f"Model '{self.model}' is not verified for the {self.company} adapter. "
                    f"Continuing with the selected adapter."
                ),
                UserWarning,
            )
            logger.warning(f"Unverified model used: {self.model}")
            self.pricing = None
        else:
            base_pricing = getattr(model_spec, "pricing", None)
            self.pricing = deepcopy(base_pricing) if base_pricing else None
            self.is_reasoning = getattr(model_spec, "is_reasoning", False)

    @abstractmethod
    def chat(self, **kwargs) -> ChatResponse:
        """
        Generates a response based on the provided conversation.
        """
        raise NotImplementedError

    @abstractmethod
    def _normalize_reasoning_level(self, level: str | int) -> int | str:
        """
        Convert a user-provided reasoning level ('none', 'low', 'medium', 'high' or numeric)
        into the provider-specific format.
        Models without reasoning support should ignore
        """
        raise NotImplementedError

    def _validate_parameter(
        self, name: str, value: float, min_value: float, max_value: float
    ) -> float:
        if not (min_value <= value <= max_value):
            error_message = (
                f"{name} must be between {min_value} and {max_value}, got {value}"
            )
            logger.error(error_message)
            raise ValueError(error_message)
        return value

    def _normalize_messages(self, messages: Any) -> Messages:
        if isinstance(messages, Messages):
            return messages
        if isinstance(messages, list):
            return Messages(messages)
        raise TypeError("messages must be a list or Messages instance")

    def _validate_tools(self, tools: Optional[List[ToolSpec]]) -> None:
        """
        Contract-level validation only.
        Provider-specific constraints are handled in provider adapters.
        """
        if tools is None:
            return
        if not isinstance(tools, list):
            raise InvalidToolSchemaError(detail="tools must be a list[ToolSpec] or None")
        seen: set[str] = set()
        for t in tools:
            if not isinstance(t, ToolSpec):
                raise InvalidToolSchemaError(
                    detail="tools must contain ToolSpec items only"
                )
            if not t.name or not isinstance(t.name, str):
                raise InvalidToolSchemaError(
                    detail="ToolSpec.name must be a non-empty string"
                )
            if not TOOL_NAME_RE.match(t.name):
                raise InvalidToolSchemaError(
                    detail=(
                        f"Invalid tool name {t.name!r}. "
                        "Must match ^[a-zA-Z0-9_-]{1,64}$"
                    )
                )
            if t.name in seen:
                raise InvalidToolSchemaError(detail=f"Duplicate tool name: {t.name!r}")
            seen.add(t.name)
            if not isinstance(t.json_schema, dict):
                raise InvalidToolSchemaError(
                    detail=f"Tool {t.name!r}: json_schema must be a dict"
                )

    def _normalize_tool_choice(
        self,
        tool_choice: Any,
        tools: Optional[List[ToolSpec]],
    ) -> Optional[str]:
        if tool_choice is None:
            return None
        tool_names = self._get_tool_names(tools)
        if isinstance(tool_choice, str):
            return self._normalize_tool_choice_from_str(tool_choice, tools, tool_names)
        if isinstance(tool_choice, dict):
            return self._normalize_tool_choice_from_dict(tool_choice, tools, tool_names)
        raise ToolChoiceError(
            detail=f"Invalid tool_choice type: {type(tool_choice).__name__}"
        )

    def _get_tool_names(self, tools: Optional[List[ToolSpec]]) -> set[str]:
        return {tool.name for tool in tools or []}

    def _normalize_tool_choice_from_str(
        self,
        tool_choice: str,
        tools: Optional[List[ToolSpec]],
        tool_names: set[str],
    ) -> str:
        if tool_choice == "required":
            self._raise_required_tool_choice_error()
        if tool_choice in ("auto", "none"):
            return tool_choice
        if tool_choice == "any":
            self._ensure_tools_provided(tools, detail="tool_choice='any' requires tools to be provided")
            return "any"
        self._ensure_tools_provided(
            tools,
            detail="tool_choice references a tool but tools=None",
        )
        if tool_choice in tool_names:
            return tool_choice
        raise ToolChoiceError(detail=f"Unknown tool_choice string: {tool_choice!r}")

    def _normalize_tool_choice_from_dict(
        self,
        tool_choice: Dict[str, Any],
        tools: Optional[List[ToolSpec]],
        tool_names: set[str],
    ) -> str:
        tc_type = tool_choice.get("type")
        tc_name = tool_choice.get("name")
        if tc_type == "required":
            self._raise_required_tool_choice_error()
        if tc_type in ("auto", "none"):
            return tc_type
        if tc_type == "any":
            self._ensure_tools_provided(
                tools,
                detail=f"tool_choice.type={tc_type!r} requires tools to be provided",
            )
            return "any"
        if tc_type == "tool":
            return self._normalize_named_tool_choice(tc_name, tools, tool_names)
        if isinstance(tc_name, str):
            return self._normalize_named_tool_choice(tc_name, tools, tool_names)
        raise ToolChoiceError(detail=f"Invalid tool_choice dict: {tool_choice!r}")

    def _normalize_named_tool_choice(
        self,
        tool_name: Any,
        tools: Optional[List[ToolSpec]],
        tool_names: set[str],
    ) -> str:
        if not isinstance(tool_name, str) or not tool_name:
            raise ToolChoiceError(
                detail="tool_choice.type='tool' requires non-empty name"
            )
        self._ensure_tools_provided(
            tools,
            detail="tool_choice references a tool but tools=None",
        )
        if tool_name not in tool_names:
            raise ToolChoiceError(
                detail=f"tool_choice references unknown tool: {tool_name!r}"
            )
        return tool_name

    def _ensure_tools_provided(
        self,
        tools: Optional[List[ToolSpec]],
        *,
        detail: str,
    ) -> None:
        if not tools:
            raise ToolChoiceError(detail=detail)

    def _raise_required_tool_choice_error(self) -> None:
        raise ToolChoiceError(
            detail="tool_choice='required' is not supported; use 'any'"
        )

    def handle_error(self, error: Exception, error_message: Optional[str] = None):
        err_msg = (
            f"Error with the provider '{self.company}' "
            f"the model '{self.model}': {error_message}. "
        )
        logger.error(err_msg)
        raise

    # ---------------- LEGACY ---------------- #

    def generate_chat_answer(self, **kwargs) -> ChatResponse:
        """Deprecated: use .chat() instead."""
        warnings.warn(
            "'generate_chat_answer' is deprecated, use 'chat' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.chat(**kwargs)
