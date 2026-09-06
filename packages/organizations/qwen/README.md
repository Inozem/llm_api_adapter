# llm-api-adapter-qwen

Official Model Studio Frankfurt/Global support for Qwen in
[llm-api-adapter](https://github.com/Inozem/llm_api_adapter/). The package uses
the Anthropic-compatible Messages API directly.

## Installation

Install the independently versioned package alongside Core:

```bash
pip install llm-api-adapter-qwen
```

Async requests need HTTPX:

```bash
pip install "llm-api-adapter-qwen[async]"
```

Synchronous requests use `requests` by default. To opt into HTTPX for sync
`chat()` and `stream_chat()`, install `"llm-api-adapter-qwen[httpx]"` and pass
`transport="httpx"`.

## Quick start

```python
import os

from llm_api_adapter.models.messages.chat_message import UserMessage
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter

adapter = UniversalLLMAPIAdapter(
    organization="qwen",
    model="qwen3.8-flash",
    api_key=os.environ["QWEN_API_KEY"],
)

response = adapter.chat(
    messages=[UserMessage("Explain retrieval-augmented generation.")],
    max_tokens=128,
    workspace_id="frankfurt-workspace",
)
print(response.content)
```

Qwen 0.1.0 supports only Model Studio's Frankfurt Global deployment. Pass the
required `workspace_id` explicitly to every `chat`, `stream_chat`, `achat`,
and `astream_chat` call; it is never read from an environment variable. The
package uses:

```text
https://{workspace_id}.eu-central-1.maas.aliyuncs.com/apps/anthropic/v1/messages
```

## Supported models and capabilities

The package deliberately exposes fixed model IDs, not moving aliases:
`qwen3.8-max`, `qwen3.8-flash`, `qwen3.7-plus`, and `qwen3.7-flash`.

| Capability | Supported models |
| --- | --- |
| Text chat, sync/async streaming, application function tools, JSON Schema/Pydantic output, and image URLs or bytes | All four models |
| `reasoning_level` | Qwen 3.8: categorical effort; Qwen 3.7: numeric thinking budget |

All four models default to hybrid thinking. Set `reasoning_level="none"` to
disable thinking, or `capture_reasoning=True` to receive provider-emitted
reasoning separately from visible text. `max_tokens` must be a positive integer
and limits generated output; it is separate from Qwen 3.7's thinking budget.

## PDF input

Qwen 0.1.0 supports images, but not PDFs. `DocumentPart` URLs and bytes are
rejected before any HTTP request:

```text
Qwen does not support DocumentPart; PDF and OCR are unavailable in Qwen 0.1.0.
```

The package does not make a partial PDF/OCR request or upload document bytes.

See the main [llm-api-adapter README](https://github.com/Inozem/llm_api_adapter/#readme)
for the shared API contract and examples.
