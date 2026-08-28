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
`grok-4.3`, `grok-4.5`, `grok-4.6`, and `grok-build-0.1`.

| Capability | Supported models |
| --- | --- |
| Text chat, sync/async streaming, application function tools, JSON Schema/Pydantic output, and image URLs or bytes | All four |
| `reasoning_level` | `grok-4.3`: `none`–`high`; `grok-4.5`: `low`–`high`; `grok-4.6`: `low`–`xhigh` |
| Explicit `reasoning_level` on `grok-build-0.1` | Not verified by the package; it is omitted with a warning |
| PDF URLs or bytes | `grok-4.5` and `grok-4.6` |

For `grok-4.5` and `grok-4.6`, xAI cannot disable reasoning: a requested
`"none"` is mapped to the documented minimum and produces a warning.

The JSON Schema subset is limited by xAI. The adapter rejects unsupported
schemas before sending a request. See xAI's
[structured-output documentation](https://docs.x.ai/developers/model-capabilities/text/structured-outputs)
for the supported schema vocabulary.

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
