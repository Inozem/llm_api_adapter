"""Provider-neutral assembly helpers for the Anthropic Messages wire format."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Mapping

from ...models.messages.chat_message import Messages
from ...models.responses.chat_response import ChatResponse
from ...models.tools.tool_spec import ToolSpec
from ..structured_output import validate_core_portable_schema


def build_messages_payload(
    *, model: str, messages: Messages, max_tokens: int,
    tools: list[ToolSpec] | None = None, tool_choice: str | None = None,
    parallel_tool_calls: bool | None = None,
    response_schema: dict[str, Any] | None = None,
    temperature: float | None = 1.0, top_p: float | None = 1.0,
) -> dict[str, Any]:
    """Assemble already validated Messages context without applying model policy."""
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    system, converted = messages.to_anthropic()
    payload: dict[str, Any] = {
        "model": model, "messages": deepcopy(converted), "max_tokens": max_tokens
    }
    if system is not None:
        payload["system"] = system
    if tools:
        payload["tools"] = [
            {"name": tool.name, **({"description": tool.description} if tool.description else {}),
             "input_schema": deepcopy(tool.json_schema)}
            for tool in tools
        ]
    choice = normalize_tool_choice(tool_choice)
    if choice is not None:
        payload["tool_choice"] = choice
    if parallel_tool_calls is not None:
        payload["disable_parallel_tool_use"] = not parallel_tool_calls
    if response_schema is not None:
        payload["output_config"] = {"format": {
            "type": "json_schema",
            "schema": validate_core_portable_schema(
                deepcopy(response_schema), provider="anthropic"
            ),
        }}
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    return payload


def normalize_tool_choice(value: str | None) -> dict[str, str] | None:
    if value is None:
        return None
    if value in {"auto", "none", "any"}:
        return {"type": value}
    return {"type": "tool", "name": value}


def build_generic_messages_payload(**request: Any) -> dict[str, Any]:
    """Apply generic sampling defaults, then use the common wire assembler."""
    request = dict(request)
    for field_name in ("temperature", "top_p"):
        value = request.setdefault(field_name, 1.0)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value == 1.0
        ):
            request[field_name] = None
    return build_messages_payload(**request)


def concatenate_text_blocks(blocks: list[dict[str, Any]]) -> str | None:
    texts = [
        block.get("text")
        for block in blocks
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "".join(text for text in texts if isinstance(text, str)) if texts else None


def first_text_block(blocks: list[dict[str, Any]]) -> str | None:
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            return block.get("text") if isinstance(block.get("text"), str) else None
    return None


def normalize_messages_response(
    response: Mapping[str, Any], *, capture_reasoning: bool = False,
    text_normalizer: Callable[[list[dict[str, Any]]], str | None] = concatenate_text_blocks,
    usage_normalizer: Callable[[Mapping[str, Any]], Any] | None = None,
) -> ChatResponse:
    """Normalize via the shared response factory with an explicit text policy."""
    safe_response = deepcopy(dict(response))
    usage = (
        usage_normalizer(deepcopy(safe_response))
        if usage_normalizer is not None
        else None
    )
    safe_response.pop("usage", None)
    blocks = safe_response.get("content")
    if isinstance(blocks, list):
        selected = text_normalizer(blocks)
        first_index = next(
            (index for index, block in enumerate(blocks)
             if isinstance(block, dict) and block.get("type") == "text"),
            None,
        )
        if first_index is not None:
            blocks[first_index]["text"] = selected
            for index, block in enumerate(blocks):
                if index != first_index and isinstance(block, dict) and block.get("type") == "text":
                    block["text"] = ""
    result = ChatResponse.from_anthropic_response(
        safe_response, capture_reasoning=capture_reasoning
    )
    result.usage = usage
    return result


__all__ = ["build_generic_messages_payload", "build_messages_payload", "concatenate_text_blocks", "first_text_block",
           "normalize_messages_response", "normalize_tool_choice"]
