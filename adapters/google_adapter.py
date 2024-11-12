from typing import List, Optional

from pydantic import confloat

from ..adapters.base_adapter import LLMAdapterBase
from ..responses.chat_response import ChatResponse
from ..messages.chat_message import Message, Prompt


class GoogleAdapter(LLMAdapterBase):
    company: str = "google"
    verified_models: List[str] = [
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
        "gemini-1.5-pro",
    ]
    library_name: str = "google.generativeai"
    library_install_name: str = "google-generativeai"

    def generate_chat_answer(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = 256,
        temperature: confloat(ge=0, le=2) = 1.0,
        top_p: confloat(ge=0, le=1) = 1.0
    ) -> ChatResponse:
        try:
            client_data = {"model_name": self.model}
            # Transform messages
            role_mapping = {
                "user": "user",
                "assistant": "model"
            }
            transformed_messages = []
            for msg in messages:
                if isinstance(msg, Prompt):
                    client_data["system_instruction"] = msg.content
                else:
                    transformed_msg = {
                        "role": role_mapping[msg.role],
                        "parts": msg.content
                    }
                    transformed_messages.append(transformed_msg)

            # Prepare the API request
            self._library.configure(api_key=self.api_key)
            client = self._library.GenerativeModel(**client_data)
            chat = client.start_chat(history=transformed_messages[:-1])
            response = chat.send_message(
                transformed_messages[-1]["parts"],
                generation_config={
                    "max_output_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                },
            )
            return ChatResponse.from_google_response(response)
        except Exception as e:
            self.handle_error(e, self.company)
