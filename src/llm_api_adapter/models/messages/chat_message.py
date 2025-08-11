from dataclasses import dataclass, field


@dataclass
class Message:
    content: str


@dataclass
class Prompt(Message):
    role: str = field(default="system", init=False)


@dataclass
class UserMessage(Message):
    role: str = field(default="user", init=False)


@dataclass
class AIMessage(Message):
    role: str = field(default="assistant", init=False)
