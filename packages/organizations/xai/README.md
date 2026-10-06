# llm-api-adapter-xai

An optional xAI organization package for
[LLM API Adapter](https://github.com/Inozem/llm_api_adapter/), a Python SDK with
one shared interface for calling LLM APIs.

Uses xAI's official Responses API.

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
    model="grok-4.7",
    api_key=os.environ["XAI_API_KEY"],
)

response = adapter.chat(
    messages=[UserMessage("Explain retrieval-augmented generation.")]
)
print(response.content)
```

## Supported models

- `grok-4.7`
- `grok-4.6`
- `grok-4.5`

## Capabilities

| Capability | Supported models |
| --- | --- |
| Text chat, sync/async streaming, application function tools, JSON Schema/Pydantic output, image URLs or bytes, and PDF URLs or bytes | All three models |

For `grok-4.5`, `grok-4.6`, and `grok-4.7`, xAI cannot disable reasoning: a requested
`"none"` is mapped to the documented minimum and produces a warning.

`grok-4.5` additionally accepts `xhigh` effort with `high` behavior. Exact
reasoning values, limits, prices, and exceptions are recorded in the
[xAI registry](https://github.com/Inozem/llm_api_adapter/blob/main/packages/organizations/xai/src/llm_api_adapter_xai/registry/organizations/xai.json).

## Structured-output portability

The package uses the Core portable profile for `json_schema` and Pydantic
`response_model`. The main [Structured Output guide](https://github.com/Inozem/llm_api_adapter/#structured-output)
defines schema validation and parsed, refusal, and incomplete results.

xAI's documented immediate schema failures are an additive local overlay, not
a replacement for the Core boundary. The adapter rejects boolean property
schemas, empty `enum` or `anyOf`, `minContains`/`maxContains`, tuple `items`
arrays, and unsupported regular expressions before the request. See xAI's
[structured-output documentation](https://docs.x.ai/developers/model-capabilities/text/structured-outputs)
for xAI-specific details.

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
Storage for an uploaded file is also billed by xAI until it expires. Consult
xAI billing for storage and any charges outside the reported request cost.

See the official xAI documentation for
[Responses storage](https://docs.x.ai/developers/model-capabilities/text/comparison),
[files and expiry](https://docs.x.ai/developers/files/managing-files), and
[file-search pricing](https://docs.x.ai/developers/pricing).

## Automatic cache usage and costs

`usage.input_tokens_details.cached_tokens` becomes `Usage.cached_tokens`,
a subset of total input. Registry tiers use total input, including cache hits,
and price ordinary and cached input separately. There is no separately priced
automatic cache write; `cache_write_tokens` remains `None`. Without the required
cache split, estimated `cost_input` and `cost_total` are unknown while known
output cost can remain available. Opt-in cache controls, TTLs, and storage
pricing are outside this estimate.

When xAI reports a valid `usage.cost_in_usd_ticks`, it takes precedence over
the token estimate: `cost_total` is that value divided by 10,000,000,000 in USD.
`cost_input` and `cost_output` remain `None` because this total does not itemize
token costs. A known provider total can therefore coexist with unknown input
and output costs. See the shared [usage and pricing guide](https://github.com/Inozem/llm_api_adapter/#token-usage-and-pricing).

See the main [llm-api-adapter README](https://github.com/Inozem/llm_api_adapter/#readme)
for the shared API contract and examples.
