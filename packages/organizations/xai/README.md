# llm-api-adapter-xai

Official xAI Responses API support for
[llm-api-adapter](https://github.com/Inozem/llm_api_adapter/).

## Installation

Install through the core package extra:

```bash
pip install "llm-api-adapter[xai]"
```

Direct installation is also supported when the core package is managed
separately:

```bash
pip install llm-api-adapter-xai
```

Async methods need HTTPX:

```bash
pip install "llm-api-adapter[xai,async]"
```

Synchronous `requests` remains the default. To opt into the HTTPX synchronous
transport, install `"llm-api-adapter[xai,httpx]"` and pass
`transport="httpx"`.

## Quick start

```python
import os

from llm_api_adapter.models.messages.chat_message import UserMessage
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter

adapter = UniversalLLMAPIAdapter(
    organization="xai",
    model="grok-4.6",
    api_key=os.environ["XAI_API_KEY"],
)

response = adapter.chat(
    messages=[UserMessage("Explain retrieval-augmented generation.")]
)
print(response.content)
```

## Supported models and capabilities

The package deliberately exposes fixed model IDs, not moving aliases:
`grok-4.5` and `grok-4.6`.

| Capability | Supported models |
| --- | --- |
| Text chat, sync/async streaming, application function tools, JSON Schema/Pydantic output, image URLs or bytes, and PDF URLs or bytes | Both models |
| `reasoning_level` | `grok-4.5`: `low`–`high`; `grok-4.6`: `low`–`xhigh` |

For `grok-4.5` and `grok-4.6`, xAI cannot disable reasoning: a requested
`"none"` is mapped to the documented minimum and produces a warning.

## Structured-output portability

This package requires `llm-api-adapter>=0.9.2,<1.0.0` and enforces the same
Core portable JSON Schema profile as OpenAI, Anthropic, Google, and Mistral.
The profile guarantees that every object is strict, every property is required,
optional values are nullable, and only
direct, non-recursive local `#/$defs/...` references are resolved before the
request.

xAI's documented immediate schema failures are an additive local overlay, not
a replacement for the Core boundary. The adapter rejects boolean property
schemas, empty `enum` or `anyOf`, `minContains`/`maxContains`, tuple `items`
arrays, and unsupported regular expressions before the request. See xAI's
[structured-output documentation](https://docs.x.ai/developers/model-capabilities/text/structured-outputs)
for xAI-specific details.

Use `json_schema` for parsed JSON only. Use a Pydantic `response_model` when
the final result must also be locally validated and returned as
`ChatResponse.parsed_model`; each nested Pydantic model must use
`ConfigDict(extra="forbid")`. Refusal and incomplete terminal responses set
`ChatResponse.refusal` or `ChatResponse.incomplete_reason` and leave parsed
fields unset. Invalid completed JSON or failed Pydantic validation raises
`JSONSchemaError`. The complete portable vocabulary and examples are in the
main [Structured Output guide](https://github.com/Inozem/llm_api_adapter/#structured-output).

## Conversations, files, and costs

`previous_response` is accepted for the shared API, but xAI continuation is
intentionally not used: the adapter does not send `previous_response_id`.
Keep and provide the complete `messages` history for each turn, including
assistant tool calls and `ToolMessage` results.

The package does not expose or send xAI's `store` option. xAI documents that
Responses are stored server-side by default, so configure data retention with
xAI when that matters to your application. In particular, the library does not
turn on Zero Data Retention (ZDR); enabling ZDR in the xAI Console blocks new
Files API uploads and `file_id` attachments.

PDF URLs are passed to xAI unchanged. For PDF bytes, the adapter uploads a
provider-owned file with a 24-hour expiry; it never deletes or changes a URL or
file identifier supplied by your application. No OCR or local text extraction
is performed.

Attaching a PDF activates xAI's `attachment_search` tool. That makes the
request agentic and adds tool-invocation charges to normal token charges.
Storage for an uploaded file is also billed by xAI until it expires. Treat
`ChatResponse.cost_total` as the exact request cost only when xAI returns it;
consult xAI billing for storage and any charges not included in that response.

See the official xAI documentation for
[Responses storage](https://docs.x.ai/developers/model-capabilities/text/comparison),
[files and expiry](https://docs.x.ai/developers/files/managing-files), and
[file-search pricing](https://docs.x.ai/developers/pricing).

See the main [llm-api-adapter README](https://github.com/Inozem/llm_api_adapter/#readme)
for the shared API contract and examples.
