from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Dict, List, Optional, Tuple, Type

from ..tools import ToolCall


@dataclass
class Message:
    content: str
    role: str = field(init=False)

    def to_openai(self) -> Dict[str, Any]:
        return {"role": self.role, "content": self.content}

    def to_anthropic(self) -> Dict[str, Any]:
        return {"role": self.role, "content": self.content}

    def to_google(self) -> Dict[str, Any]:
        return {"role": self.role, "parts": [{"text": self.content}]}


@dataclass
class Prompt(Message):
    role: str = field(default="system", init=False)

    def to_anthropic(self) -> str:
        return self.content

    def to_google(self) -> str:
        return self.content


@dataclass
class UserMessage(Message):
    role: str = field(default="user", init=False)

    def to_google(self) -> Dict[str, Any]:
        return {"role": "user", "parts": [{"text": self.content}]}


@dataclass
class AIMessage(Message):
    """
    Assistant message.

    - OpenAI tool loop: may include tool_calls (assistant turn that requested tools).
    - Anthropic tool loop: tool_calls -> content blocks type="tool_use".
    - Google tool loop: tool_calls -> parts[].functionCall.
    """
    role: str = field(default="assistant", init=False)
    tool_calls: Optional[List[ToolCall]] = None

    def to_openai(self) -> Dict[str, Any]:
        msg: Dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.call_id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in self.tool_calls
            ]
        return msg

    def to_anthropic(self) -> Dict[str, Any]:
        if not self.tool_calls:
            return {"role": "assistant", "content": self.content}
        blocks: List[Dict[str, Any]] = []
        if self.content:
            blocks.append({"type": "text", "text": self.content})
        for tc in self.tool_calls:
            if not tc.call_id:
                raise ValueError(
                    "Anthropic tool_use requires a non-empty call_id (tool_use id)."
                )
            blocks.append(
                {
                    "type": "tool_use",
                    "id": tc.call_id,
                    "name": tc.name,
                    "input": tc.arguments or {},
                }
            )
        return {"role": "assistant", "content": blocks}

    def to_google(self) -> Dict[str, Any]:
        parts: List[Dict[str, Any]] = []
        if self.content:
            parts.append({"text": self.content})
        if self.tool_calls:
            for tc in self.tool_calls:
                parts.append(
                    {
                        "functionCall": {
                            "name": tc.name,
                            "args": tc.arguments or {},
                        }
                    }
                )
        if not parts:
            parts = [{"text": ""}]
        return {"role": "model", "parts": parts}


@dataclass
class ToolMessage(Message):
    """
    Tool result message.

    - OpenAI format: role="tool" + tool_call_id
    - Anthropic format: user turn with tool_result block referencing tool_use_id
    - Google format: user turn with functionResponse part (name + response object)
      NOTE: Gemini doesn't have call_id; for Google tool loop we treat tool_call_id as function name.
    """
    tool_call_id: str
    role: str = field(default="tool", init=False)

    def to_openai(self) -> Dict[str, Any]:
        return {"role": "tool", "tool_call_id": self.tool_call_id, "content": self.content}

    def to_anthropic(self) -> Dict[str, Any]:
        if not self.tool_call_id:
            raise ValueError("Anthropic tool_result requires tool_call_id (tool_use_id).")
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": self.tool_call_id,
                    "content": self.content,
                }
            ],
        }

    def to_google(self) -> Dict[str, Any]:
        name = self.tool_call_id
        if not name:
            raise ValueError("Google functionResponse requires tool_call_id to be set (use tool name).")
        response_obj: Any
        try:
            raw = self.content.strip()
            response_obj = json.loads(raw) if raw else {}
        except Exception:
            response_obj = {"content": self.content}
        return {
            "role": "user",
            "parts": [
                {
                    "functionResponse": {
                        "name": name,
                        "response": response_obj,
                    }
                }
            ],
        }


@dataclass
class Messages:
    items: List[Any]

    def __post_init__(self) -> None:
        role_to_cls: Dict[str, Type[Message]] = {cls.role: cls for cls in Message.__subclasses__()}
        normalized: List[Message] = []
        for item in self.items:
            if isinstance(item, Message):
                normalized.append(item)
                continue
            if isinstance(item, dict):
                role = item.get("role")
                if not role:
                    raise ValueError("Missing 'role' in message data")
                if role not in role_to_cls:
                    raise ValueError(f"Unsupported role: {role}")
                if role == "tool":
                    tool_call_id = item.get("tool_call_id")
                    if not tool_call_id:
                        raise ValueError("Missing 'tool_call_id' for tool message")
                    content = item.get("content")
                    if content is None:
                        raise ValueError("Missing 'content' in message data")
                    normalized.append(
                        ToolMessage(content=str(content), tool_call_id=str(tool_call_id))
                    )
                    continue
                if role == "assistant":
                    tool_calls_raw = item.get("tool_calls")
                    content = item.get("content")
                    if content is None:
                        content = ""
                    tool_calls: Optional[List[ToolCall]] = None
                    if tool_calls_raw is not None:
                        if not isinstance(tool_calls_raw, list):
                            raise ValueError("assistant.tool_calls must be a list")
                        tool_calls = []
                        for tc in tool_calls_raw:
                            if not isinstance(tc, dict):
                                raise ValueError("assistant.tool_calls items must be dict")
                            if "name" in tc:
                                name = tc.get("name")
                                if not isinstance(name, str) or not name:
                                    raise ValueError("assistant.tool_calls.name must be non-empty str")
                                args = tc.get("arguments", {})
                                if args is None:
                                    args = {}
                                if not isinstance(args, dict):
                                    raise ValueError("assistant.tool_calls.arguments must be dict")
                                call_id = tc.get("call_id")
                            # legacy OpenAI-shaped format
                            else:
                                fn = tc.get("function") or {}
                                name = fn.get("name")
                                if not isinstance(name, str) or not name:
                                    raise ValueError(
                                        "assistant.tool_calls.function.name must be non-empty str"
                                    )
                                raw_args = fn.get("arguments", "{}")
                                if isinstance(raw_args, str):
                                    try:
                                        args = json.loads(raw_args) if raw_args.strip() else {}
                                    except Exception as e:
                                        raise ValueError(
                                            f"assistant.tool_calls.arguments JSON parse failed: {e}"
                                        )
                                elif isinstance(raw_args, dict):
                                    args = raw_args
                                else:
                                    args = {}
                                call_id = tc.get("id")
                            tool_calls.append(
                                ToolCall(
                                    name=name,
                                    arguments=args,
                                    call_id=call_id,
                                )
                            )
                    normalized.append(AIMessage(content=str(content), tool_calls=tool_calls))
                    continue
                content = item.get("content")
                if content is None or content == "":
                    raise ValueError("Missing 'content' in message data")
                cls = role_to_cls[role]
                normalized.append(cls(content=str(content)))
                continue
            raise TypeError(f"Unsupported message type: {type(item)}")
        self.items = normalized

    def to_openai(self) -> List[Dict[str, Any]]:
        return [m.to_openai() for m in self.items]

    def to_anthropic(self) -> Tuple[str | None, List[Dict[str, Any]]]:
        prompt: Optional[str] = None
        messages: List[Dict[str, Any]] = []
        for m in self.items:
            if isinstance(m, Prompt):
                if prompt is not None:
                    raise ValueError("Multiple system prompts are not allowed for Anthropic.")
                prompt = m.to_anthropic()
                continue
            if isinstance(m, ToolMessage):
                messages.append(m.to_anthropic())
                continue
            if isinstance(m, AIMessage):
                messages.append(m.to_anthropic())
                continue
            messages.append(m.to_anthropic())
        return prompt, messages

    def to_google(self) -> Tuple[str | None, List[Dict[str, Any]]]:
        prompt: Optional[str] = None
        messages: List[Dict[str, Any]] = []
        for m in self.items:
            if isinstance(m, Prompt):
                if prompt is not None:
                    raise ValueError("Multiple system prompts are not allowed for Google.")
                prompt = m.to_google()
                continue
            if isinstance(m, ToolMessage):
                messages.append(m.to_google())
                continue
            if isinstance(m, AIMessage):
                messages.append(m.to_google())
                continue
            if isinstance(m, UserMessage):
                messages.append(m.to_google())
            else:
                messages.append({"role": "user", "parts": [{"text": m.content}]})
        return prompt, messages
