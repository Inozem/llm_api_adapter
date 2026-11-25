# llm-api-adapter-kimi

An optional Kimi / Moonshot organization package for
[LLM API Adapter](https://github.com/Inozem/llm_api_adapter/), a Python SDK with
one shared interface for calling LLM APIs. Select Kimi natively with
`organization="kimi"` to use its official Chat Completions API at
`POST https://api.moonshot.ai/v1/chat/completions`.
The Kimi adapter and sync/async clients inherit Core's OpenAI Compatible
execution while retaining native selection and Chat Completions.

## Installation

The package requires Core `llm-api-adapter>=0.10.0,<1.0.0` and adds no provider
SDK. Install the package directly or through Core's Kimi extra.

```bash
pip install "llm-api-adapter-kimi>=0.1.0,<0.2.0"
# Or: pip install "llm-api-adapter[kimi]"
```

Requests is the default synchronous transport. Install optional extras for
async support or sync HTTPX. Install both extras together as shown here.

```bash
pip install "llm-api-adapter-kimi[async]"
pip install "llm-api-adapter-kimi[httpx]"
pip install "llm-api-adapter-kimi[async,httpx]"
```

Use `transport="httpx"` to select HTTPX for synchronous calls. `achat` and
`astream_chat` require HTTPX installed with the `[async]` extra.

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

Running this example sends a real request to Kimi.

## Models and capabilities

| Model | Tool choice |
| --- | --- |
| `kimi-k3` | `auto`, `none`, `any` |
| `kimi-k3.1-pro` | `auto`, `none`, `any` |
| `kimi-k3.1-pro-highspeed` | `auto`, `none`, `any` |
| `kimi-k2.6` | `auto`, `none` |

All four models support text chat, application tool calls, portable JSON Schema
and Pydantic structured output, and inline images from bytes or data URIs.
The SDK does not execute application tools. Named tool choice is rejected for
all models. Explicit
`parallel_tool_calls=False` and `True` are both unsupported and rejected before
transport. Omit the option. Public `ImagePart` URLs and PDF `DocumentPart`
bytes or URLs are unsupported and rejected before HTTP. The adapter does not
download, upload, extract, or OCR files.

The [Kimi model registry](https://github.com/Inozem/llm_api_adapter/blob/main/packages/organizations/kimi/src/llm_api_adapter_kimi/registry/organizations/kimi.json)
is authoritative for exact-model context and output limits, capabilities,
request rules, and pricing.

## Reasoning and conversation history

Reasoning settings are resolved automatically for the selected model. Use
`capture_reasoning=True` for separate reasoning observations and `on_reasoning`
for stream callbacks.

Kimi Chat Completions is stateless. Send the complete `messages` history on
each turn, including assistant tool calls and `ToolMessage` results.
`previous_response` is accepted for shared API compatibility and ignored.
Existing history normalization is unchanged. Reasoning content from an
assistant history dictionary is not automatically replayed or preserved by
this package.

## Streaming and validation

`stream_chat` and `astream_chat` use Core's lazy streaming. Validation and
request opening happen during iteration, so catch errors around the `for` or
`async for` loop. See the main [streaming guide](https://github.com/Inozem/llm_api_adapter/#streaming)
for general examples and [Handling Errors](https://github.com/Inozem/llm_api_adapter/#handling-errors)
for shared public error types.

Kimi rejects explicit `parallel_tool_calls=False` and `True` on an otherwise
valid request with `NotImplementedError`. When `temperature=-1` and
`parallel_tool_calls=False`, the shared sampling `ValueError` is raised first,
before provider I/O.

## Usage and pricing

Usage and pricing follow Kimi's registered model data. Responses may expose
`Usage.cached_tokens`; K3 and K3.1 Pro also expose automatic cache writes as
`Usage.cache_write_tokens`. Cache reads and applicable writes are disjoint
subsets of input tokens. If a required count is missing, `cost_input` and
`cost_total` remain `None`, while known output cost can still be exposed. A
reported zero remains zero. K2.6 has registered automatic cache-read pricing
only. See the shared
[usage and pricing guide](https://github.com/Inozem/llm_api_adapter/#token-usage-and-pricing).

## Validation status

Local mocked checks covered Core `0.10.0` with and without HTTPX. Live provider
validation remains in the existing Kimi release lane. See the main
[README](https://github.com/Inozem/llm_api_adapter/#readme) for the shared API
contract and release process.
