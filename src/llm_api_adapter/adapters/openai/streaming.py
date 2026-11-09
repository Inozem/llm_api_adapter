"""OpenAI streaming composition and native usage compatibility hooks."""

from __future__ import annotations

from typing import Any, Optional

from .payloads import _cache_token_counts
from ..openai_compatible.chat_completions.streaming import _ChatCompletionsStreamingMixin
from ..openai_compatible.common.usage import _normalize_openai_stream_usage, _token_count
from ...models.responses.chat_response import Usage
from ..openai_compatible.responses.streaming import (
    _ResponsesStreamState,
    _ResponsesStreamingMixin,
)


class _OpenAIStreamingMixin(_ChatCompletionsStreamingMixin, _ResponsesStreamingMixin):
    """Compose the Chat and Responses stream protocols for native OpenAI."""

    @staticmethod
    def _normalize_stream_usage(
        raw_usage: Any,
        *,
        input_field: str,
        output_field: str,
    ) -> Optional[Usage]:
        return _normalize_openai_stream_usage(
            raw_usage,
            input_field=input_field,
            output_field=output_field,
            token_counter=_OpenAIStreamingMixin._token_count,
        )

    _token_count = staticmethod(_token_count)
