# llm-api-adapter-kimi

Official direct Kimi / Moonshot Chat Completions API support for
[llm-api-adapter](https://github.com/Inozem/llm_api_adapter/). The package uses
only `POST /v1/chat/completions`; it does not install the Kimi SDK or add
provider-specific public APIs.

## Installation

Install through the Core package extra:

```bash
pip install "llm-api-adapter[kimi]"
```

Direct installation is also supported when Core is managed separately:

```bash
pip install llm-api-adapter-kimi
```

Async requests need HTTPX:

```bash
pip install "llm-api-adapter[kimi,async]"
```

Synchronous `requests` remains the default. To opt into HTTPX for sync
`chat()` and `stream_chat()`, install `"llm-api-adapter[kimi,httpx]"` and pass
`transport="httpx"`.

## Quick start

```python
import os

from llm_api_adapter.models.messages.chat_message import UserMessage
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter

adapter = UniversalLLMAPIAdapter(
    organization="kimi",
    model="kimi-k3",
    api_key=os.environ["KIMI_API_KEY"],
)

response = adapter.chat(
    messages=[UserMessage("Explain retrieval-augmented generation.")],
    max_tokens=128,
)
print(response.content)
```

## Supported models and capabilities

The package deliberately exposes fixed model IDs, not moving aliases:
`kimi-k3`, `kimi-k2.7-code`, and `kimi-k2.6`.

| Capability | Supported models |
| --- | --- |
| Text chat; sync/async streaming; application tools; portable JSON Schema/Pydantic output; image bytes and data URIs | All three models |
| Public `ImagePart` URLs and every `DocumentPart` PDF URL or byte | Unsupported; rejected before HTTP |

`reasoning_level` is resolved automatically from registry metadata. K3 and
K2.7 Code cannot disable reasoning, so `reasoning_level="none"` warns. K2.6
maps `"none"` to disabled thinking and every other valid level to enabled
thinking. When omitted, no thinking control is sent and Kimi's native default
is preserved. K2.7 Code is code-oriented, but is not restricted to code-only
prompts. Reasoning is never mixed into visible text; use
`capture_reasoning=True` for opt-in observability.

## History, files, and pricing

Kimi Chat Completions is stateless: `previous_response` is accepted for the
shared API but is not serialized. Send the complete `messages` history on
each turn, including assistant tool calls and `ToolMessage` results.

Image bytes are encoded as data URIs. Public image URLs and every
`DocumentPart` are rejected before transport. Kimi's Files API provides
extracted text rather than a Chat Completions attachment, so the adapter does
not upload, retain, download, extract, or delete caller files.

Cost fields use registered standard USD rates. When Kimi reports
`usage.cached_tokens`, the adapter applies cache-hit and cache-miss input
rates; without that split, `cost_input` and `cost_total` remain unset. This
does not enable Kimi context caching, and the result is not an invoice.

Kimi maps authentication/authorization (401/403), rate-limit (429), timeout
(408/504), documented token/quota, and server failures to the matching public
`LLMAPI*Error`; other client or SSE failures become `LLMAPIClientError`.

See the main [llm-api-adapter README](https://github.com/Inozem/llm_api_adapter/#readme)
for the shared API contract, error mapping, and E2E/release documentation.
