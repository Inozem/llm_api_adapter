from dataclasses import dataclass
import logging
from typing import List, Optional

from ..adapters.base_adapter import LLMAdapterBase
from ..errors.llm_api_error import LLMAPIError
from ..llms.openai.sync_client import OpenAISyncClient
from ..models.messages.chat_message import Message
from ..models.responses.chat_response import ChatResponse

logger = logging.getLogger(__name__)


@dataclass
class OpenAIAdapter(LLMAdapterBase):
    company: str = "openai"

    def generate_chat_answer(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 1.0
    ) -> ChatResponse:
        temperature = self._validate_parameter(
            name="temperature", value=temperature, min_value=0, max_value=2
        )
        top_p = self._validate_parameter(
            name="top_p", value=top_p, min_value=0, max_value=1
        )
        try:
            transformed_messages = []
            for msg in messages:
                transformed_messages.append(
                    {"role": msg.role, "content": msg.content}
                )
            client = OpenAISyncClient(api_key=self.api_key)
            response = client.chat_completion(
                model=self.model,
                messages=transformed_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            chat_response = ChatResponse.from_openai_response(response)
            if self.pricing:
                chat_response.apply_pricing(
                    price_input_per_token=self.pricing.in_per_token,
                    price_output_per_token=self.pricing.out_per_token,
                    currency=self.pricing.currency
                )
            return chat_response
        except LLMAPIError as e:
            self.handle_error(e)
        except Exception as e:
            error_message = getattr(e, "text", None) or str(e)
            self.handle_error(error=e, error_message=error_message)