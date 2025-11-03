from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Message:
    content: str

    def to_openai(self) -> Dict:
        return {"role": self.role, "content": self.content}
    
    def to_anthropic(self) -> Dict:
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

    def to_google(self) -> str:
        return {"role": self.role, "parts": [{"text": self.content}]}


@dataclass
class AIMessage(Message):
    role: str = field(default="assistant", init=False)

    def to_google(self) -> str:
        return {"role": "model", "parts": [{"text": self.content}]}


@dataclass
class Messages:
    items: List[Any]

    def __post_init__(self):
        allowed_roles = {cls.role for cls in Message.__subclasses__()}
        normalized = []
        for item in self.items:
            if isinstance(item, Message):
                normalized.append(item)
            elif isinstance(item, dict):
                role, content = item.get("role"), item.get("content")
                if not role:
                    raise ValueError("Missing 'role' in message data")
                if not content:
                    raise ValueError("Missing 'content' in message data")
                if role not in allowed_roles:
                    raise ValueError(f"Unsupported role: {role}")
                normalized.append(Message(role=role, content=content))
            else:
                raise TypeError(f"Unsupported message type: {type(item)}")
        self.items = normalized
    
    def to_openai(self) -> list[dict]:
        messages = [message.to_openai() for message in self.items]
        return messages
    
    def to_anthropic(self) -> tuple[str | None, list[dict]]:
        prompt = None
        messages = []
        for message in self.items:
            if isinstance(message, Prompt):
                if prompt is not None:
                    raise ValueError(
                        "Multiple system prompts are not allowed for Anthropic."
                    )
                prompt = message.to_anthropic()
            else:
                messages.append(message.to_anthropic())
        return prompt, messages

    def to_google(self) -> tuple[str | None, list[dict]]:
        prompt = None
        messages = []
        for message in self.items:
            if isinstance(message, Prompt):
                if prompt is not None:
                    raise ValueError(
                        "Multiple system prompts are not allowed for Google."
                    )
                prompt = message.to_google()
            else:
                messages.append(message.to_google())
        return prompt, messages
