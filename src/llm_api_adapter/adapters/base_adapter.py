from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
import json
import logging
import re
from typing import Any, Dict, List, Optional
import warnings

from ..errors.llm_api_error import InvalidToolSchemaError, JSONSchemaError, ToolChoiceError
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
    is_adaptive_thinking: bool = False
    reasoning_levels: Dict[str, int] = field(
        default_factory=lambda: REASONING_LEVELS_DEFAULT.copy()
    )

    def __repr__(self) -> str:
        masked = f"{self.api_key[:8]}...{self.api_key[-4:]}" if len(self.api_key) > 12 else "***"
        return f"{self.__class__.__name__}(company='{self.company}', model='{self.model}', api_key='{masked}')"

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
            self.is_adaptive_thinking = getattr(model_spec, "is_adaptive_thinking", False)

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

    def _enforce_strict_schema(self, schema: dict) -> dict:
        """Recursively add additionalProperties: false to all object types — required by OpenAI and Anthropic strict mode."""
        schema = dict(schema)
        if schema.get("type") == "object":
            schema["additionalProperties"] = False
            if "properties" in schema:
                schema["properties"] = {
                    k: self._enforce_strict_schema(v) if isinstance(v, dict) else v
                    for k, v in schema["properties"].items()
                }
        if "items" in schema and isinstance(schema["items"], dict):
            schema["items"] = self._enforce_strict_schema(schema["items"])
        return schema


    def _resolve_json_schema(
        self,
        json_schema: Optional[dict],
        response_model: Optional[Any],
        tools: Optional[List[ToolSpec]],
    ) -> Optional[dict]:
        if json_schema is not None and response_model is not None:
            raise JSONSchemaError(detail="json_schema and response_model cannot be used together")
        if response_model is not None and tools is not None:
            raise JSONSchemaError(detail="response_model and tools cannot be used together")
        if json_schema is not None and tools is not None:
            raise JSONSchemaError(detail="json_schema and tools cannot be used together")
        if response_model is not None:
            try:
                return response_model.model_json_schema()
            except AttributeError:
                try:
                    import pydantic  # noqa
                except ImportError:
                    raise JSONSchemaError(
                        detail="pydantic is required for response_model; install it with: pip install pydantic"
                    )
                raise JSONSchemaError(detail="response_model must be a Pydantic BaseModel subclass")
        if json_schema is not None and not isinstance(json_schema, dict):
            raise JSONSchemaError(detail="json_schema must be a dict")
        return json_schema

    def _strip_json_fences(self, content: str) -> str:
        text = content.strip()
        fence_match = re.match(r'^```(?:json)?\s*([\s\S]*?)\s*```$', text, re.IGNORECASE)
        if fence_match:
            return fence_match.group(1).strip()
        block_match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', text)
        if block_match:
            return block_match.group(1)
        return text

    def _parse_json_response(
        self,
        content: Optional[str],
        json_schema: Optional[dict],
    ) -> Optional[dict]:
        if json_schema is None or content is None:
            return None
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            try:
                return json.loads(self._strip_json_fences(content))
            except json.JSONDecodeError as e:
                raise JSONSchemaError(detail=f"Model response is not valid JSON: {e}")

    def _parse_response_model(
        self,
        parsed_json: Optional[dict],
        response_model: Optional[Any],
    ) -> Optional[Any]:
        if response_model is None or parsed_json is None:
            return None
        try:
            return response_model.model_validate(parsed_json)
        except Exception:
            try:
                return response_model.model_validate(parsed_json, strict=False)
            except Exception as e:
                raise JSONSchemaError(detail=f"Response failed Pydantic validation: {e}")

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
