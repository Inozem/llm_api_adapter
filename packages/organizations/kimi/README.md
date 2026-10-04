# llm-api-adapter-kimi

An optional Kimi / Moonshot organization package for
[LLM API Adapter](https://github.com/Inozem/llm_api_adapter/), a Python SDK with
one shared interface for calling LLM APIs.

Calls Kimi / Moonshot through `POST /v1/chat/completions` without a provider SDK.

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

## Supported models

- `kimi-k3`
- `kimi-k2.6`

## Capabilities

| Capability | Supported models |
| --- | --- |
| Text chat; sync/async streaming; application tools; portable JSON Schema/Pydantic output; image bytes and data URIs | Both models |
| Public `ImagePart` URLs and every `DocumentPart` PDF URL or byte | Unsupported; rejected before HTTP |

`reasoning_level` is resolved automatically from registry metadata. K3 cannot
disable reasoning, so `reasoning_level="none"` uses the registered minimum and
warns. K2.6 maps `"none"` to disabled thinking and every other valid level to
enabled thinking. When omitted, no thinking control is sent and Kimi's native
default is preserved.
Reasoning is never mixed into visible text; use
`capture_reasoning=True` for opt-in observability.

Both models reject named tool choice before transport. `kimi-k3` accepts
`tool_choice="any"`; `kimi-k2.6` rejects it and accepts only automatic or
disabled tool choice. Exact supported values, request rules, and exceptions
are recorded in the [Kimi registry](https://github.com/Inozem/llm_api_adapter/blob/main/packages/organizations/kimi/src/llm_api_adapter_kimi/registry/organizations/kimi.json).

## History and files

Kimi Chat Completions is stateless: `previous_response` is accepted for the
shared API but is not serialized. Send the complete `messages` history on
each turn, including assistant tool calls and `ToolMessage` results.

Image bytes are encoded as data URIs. Public image URLs and every
`DocumentPart` are rejected before transport. Kimi's Files API provides
extracted text rather than a Chat Completions attachment, so the adapter does
not upload, retain, download, extract, or delete caller files.

Kimi maps authentication/authorization (401/403), rate-limit (429), timeout
(408/504), documented token/quota, and server failures to the matching public
`LLMAPI*Error`; other client or SSE failures become `LLMAPIClientError`.

## Automatic cache usage and pricing

`usage.prompt_tokens_details.cached_tokens` becomes `Usage.cached_tokens`;
the legacy `usage.cached_tokens` field is also accepted. K3 additionally
reports `prompt_tokens_details.cache_write_tokens` as `Usage.cache_write_tokens`
for its automatic cache-write component. Both are disjoint subsets of total
`Usage.input_tokens`. K2.6 has registered automatic cache-read pricing only.

Input cost uses ordinary, read, and applicable write rates from the USD
registry tier without double counting. If a required read or write count is
omitted, `cost_input` and `cost_total` stay `None`; known output cost can still
be exposed. An explicitly reported zero stays zero. The adapter does not enable
opt-in cache modes or account for selectable TTLs or storage charges. See the
shared [usage and pricing guide](https://github.com/Inozem/llm_api_adapter/#token-usage-and-pricing).

See the main [llm-api-adapter README](https://github.com/Inozem/llm_api_adapter/#readme)
for the shared API contract, error mapping, and E2E/release documentation.
