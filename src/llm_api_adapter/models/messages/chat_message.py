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
        return {"role": self.role, "parts": [{"text": self.content}]}


@dataclass
class AIMessage(Message):
    """
    Assistant message.
    For OpenAI tool loop, it may carry tool_calls (assistant turn that requested tools).
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

    def to_google(self) -> Dict[str, Any]:
        # Google expects role "model"
        return {"role": "model", "parts": [{"text": self.content}]}


@dataclass
class ToolMessage(Message):
    """
    Tool result message (OpenAI format).
    Must reference preceding assistant tool call via tool_call_id.
    """
    tool_call_id: str
    role: str = field(default="tool", init=False)

    def to_openai(self) -> Dict[str, Any]:
        return {"role": "tool", "tool_call_id": self.tool_call_id, "content": self.content}


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
                            fn = tc.get("function") or {}
                            name = fn.get("name")
                            if not isinstance(name, str) or not name:
                                raise ValueError("assistant.tool_calls.function.name must be non-empty str")
                            raw_args = fn.get("arguments", "{}")
                            if isinstance(raw_args, str):
                                try:
                                    args = json.loads(raw_args) if raw_args.strip() else {}
                                except Exception as e:
                                    raise ValueError(f"assistant.tool_calls.arguments JSON parse failed: {e}")
                            elif isinstance(raw_args, dict):
                                args = raw_args
                            else:
                                args = {}
                            tool_calls.append(
                                ToolCall(
                                    name=name,
                                    arguments=args,
                                    call_id=tc.get("id"),
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
                raise ValueError("ToolMessage is not supported for Anthropic in this version.")
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
                raise ValueError("ToolMessage is not supported for Google in this version.")
            messages.append(m.to_google())
        return prompt, messages
