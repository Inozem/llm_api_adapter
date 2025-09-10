from dataclasses import dataclass
import logging
from typing import FrozenSet, List, Optional

from ..adapters.base_adapter import LLMAdapterBase
from ..errors.llm_api_error import LLMAPIError
from ..llms.anthropic.sync_client import ClaudeSyncClient
from ..models.messages.chat_message import Message, Prompt
from ..models.responses.chat_response import ChatResponse

logger = logging.getLogger(__name__)


@dataclass
class AnthropicAdapter(LLMAdapterBase):
    company: str = "anthropic"
    verified_models: FrozenSet[str] = frozenset([
        "claude-opus-4-20250514",
        "claude-sonnet-4-20250514",
        "claude-3-7-sonnet-latest",
        "claude-3-5-haiku-latest",
        "claude-3-5-sonnet-latest",
        "claude-3-haiku-20240307"
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
            system_prompt = ""
            transformed_messages = []
            for msg in messages:
                if isinstance(msg, Prompt):
                    system_prompt = msg.content
                else:
                    transformed_messages.append({
                        "role": msg.role,
                        "content": msg.content
                    })
            client = ClaudeSyncClient(api_key=self.api_key)
            response = client.chat_completion(
                model=self.model,
                messages=transformed_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                system=system_prompt,
            )
            return ChatResponse.from_anthropic_response(response)
        except LLMAPIError as e:
            self.handle_error(e, self.company)
        except Exception as e:
            self.handle_error(e, self.company)
