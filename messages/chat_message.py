from pydantic import BaseModel


class Message(BaseModel):
    content: str

    # To allow the use of positional arguments
    def __init__(self, content: str, **kwargs):
        super().__init__(content=content, **kwargs)


class Prompt(Message):
    role: str = "system"


class UserMessage(Message):
    role: str = "user"


class AIMessage(Message):
    role: str = "assistant"
