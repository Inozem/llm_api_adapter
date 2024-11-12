from typing import List, Optional

from pydantic import confloat

from ..adapters.base_adapter import LLMAdapterBase
from ..responses.chat_response import ChatResponse
from ..messages.chat_message import Message, Prompt


class AnthropicAdapter(LLMAdapterBase):
    company: str = "anthropic"
    verified_models: List[str] = [
        "claude-3-5-sonnet-20241022",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    ]
    library_name: str = "anthropic"
    library_install_name: str = "anthropic"

    def generate_chat_answer(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = 256,
        temperature: confloat(ge=0, le=2) = 1.0,
        top_p: confloat(ge=0, le=1) = 1.0
    ) -> ChatResponse:
        try:
            response_data = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }

            # Transform messages
            transformed_messages = []
            for msg in messages:
                if isinstance(msg, Prompt):
                    response_data["system"] = msg.content
                else:
                    transformed_msg = {
                        "role": msg.role,
                        "content": msg.content
                    }
                    transformed_messages.append(transformed_msg)
            response_data["messages"] = transformed_messages

            # Prepare the API request
            client = self._library.Anthropic(api_key=self.api_key)
            response = client.messages.create(**response_data)

            return ChatResponse.from_anthropic_response(response)
        except Exception as e:
            self.handle_error(e, self.company)
