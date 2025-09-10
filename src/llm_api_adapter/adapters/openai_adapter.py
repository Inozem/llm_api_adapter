from dataclasses import dataclass
import logging
from typing import FrozenSet, List, Optional

from ..adapters.base_adapter import LLMAdapterBase
from ..errors.llm_api_error import LLMAPIError
from ..llms.openai.sync_client import OpenAISyncClient
from ..models.messages.chat_message import Message
from ..models.responses.chat_response import ChatResponse

logger = logging.getLogger(__name__)


@dataclass
class OpenAIAdapter(LLMAdapterBase):
    company: str = "openai"
    verified_models: FrozenSet[str] = frozenset([
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "gpt-4.5-preview",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-4",
        "gpt-4-turbo-preview",
        "gpt-3.5-turbo",
    ])

    def chat(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = 256,
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
            return ChatResponse.from_openai_response(response)
        except LLMAPIError as e:
            self.handle_error(e, self.company)
        except Exception as e:
            self.handle_error(e, self.company)
