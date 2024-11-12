from typing import List, Optional

from pydantic import confloat

from ..adapters.base_adapter import LLMAdapterBase
from ..responses.chat_response import ChatResponse
from ..messages.chat_message import Message


class OpenAIAdapter(LLMAdapterBase):
    company: str = "openai"
    verified_models: List[str] = [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4-turbo",
        "gpt-4",
        "gpt-4-turbo-preview",
        "gpt-3.5-turbo",
    ]
    library_name: str = "openai"
    library_install_name: str = "openai==0.28"

    def generate_chat_answer(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = 256,
        temperature: confloat(ge=0, le=2) = 1.0,
        top_p: confloat(ge=0, le=1) = 1.0
    ) -> ChatResponse:
        try:
            # Transform messages
            transformed_messages = [
                {
                    "role": msg.role,
                    "content": msg.content
                }
                for msg in messages
            ]
            
            # Prepare the API request
            self._library.api_key = self.api_key
            response = self._library.ChatCompletion.create(
                model=self.model,
                messages=transformed_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            return ChatResponse.from_openai_response(response)
        except Exception as e:
            self.handle_error(e, self.company)
