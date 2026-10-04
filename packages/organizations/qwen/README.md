# llm-api-adapter-qwen

An optional Qwen organization package for
[LLM API Adapter](https://github.com/Inozem/llm_api_adapter/), a Python SDK with
one shared interface for calling LLM APIs.

Uses Model Studio's Frankfurt Global deployment through its
Anthropic-compatible Messages API.

## Installation

Install through the Core package extra:

```bash
pip install "llm-api-adapter[qwen]"
```

Direct installation is also supported when Core is managed separately:

```bash
pip install llm-api-adapter-qwen
```

Async requests need HTTPX:

```bash
pip install "llm-api-adapter[qwen,async]"
```

Synchronous requests use `requests` by default. To opt into HTTPX for sync
`chat()` and `stream_chat()`, install `"llm-api-adapter[qwen,httpx]"` and pass
`transport="httpx"`.

## Quick start

```python
import os

from llm_api_adapter.models.messages.chat_message import UserMessage
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter

adapter = UniversalLLMAPIAdapter(
    organization="qwen",
    model="qwen3.8-max",
    api_key=os.environ["QWEN_API_KEY"],
)

response = adapter.chat(
    messages=[UserMessage("Explain retrieval-augmented generation.")],
    max_tokens=128,
    workspace_id=os.environ["QWEN_WORKSPACE_ID"],
)
print(response.content)
```

The package supports only Model Studio's Frankfurt Global deployment. Pass the
required `workspace_id` explicitly to every `chat`, `stream_chat`, `achat`,
and `astream_chat` call; it is never read from an environment variable. The
package uses:

```text
https://{workspace_id}.eu-central-1.maas.aliyuncs.com/apps/anthropic/v1/messages
```

## Supported models

- `qwen3.8-max`
- `qwen3.8-flash`
- `qwen3.7-plus`
- `qwen3.7-flash`

## Capabilities

| Capability | Supported models |
| --- | --- |
| Text chat, sync/async streaming, application function tools, JSON Schema/Pydantic output, and image URLs or bytes | All four models |
| `reasoning_level` | Qwen 3.8: categorical effort; Qwen 3.7: numeric thinking budget |

All four models default to hybrid thinking. Set `reasoning_level="none"` to
disable thinking, or `capture_reasoning=True` to receive provider-emitted
reasoning separately from visible text. `max_tokens` must be a positive integer
and limits generated output; it is separate from Qwen 3.7's thinking budget.
With thinking enabled, Model Studio's reported `usage.output_tokens` can also
include thinking tokens, so it can exceed `max_tokens` even when the visible
answer respects that output limit.

Qwen permits `tool_choice="auto"` and `"none"` in thinking mode, but not a
forced `"any"` or named tool. For a forced tool call, the adapter automatically
disables thinking and issues a `UserWarning`; pass `reasoning_level="none"` to
make that choice explicit without a warning.

Both forced tool-choice modes are declared `pass` exceptions: the adapter
handles the provider restriction while preserving the tool-call contract.
`previous_response` is accepted but ignored; supply complete `messages`
history, including assistant tool calls and tool results, on each turn.
Exact rules and exceptions are recorded in the [Qwen registry](https://github.com/Inozem/llm_api_adapter/blob/main/packages/organizations/qwen/src/llm_api_adapter_qwen/registry/organizations/qwen.json).

## PDF input

The package rejects `DocumentPart` URLs and bytes before any HTTP request.
It does not upload documents or run OCR.

## Automatic cache usage and pricing

Model Studio's Messages usage reports ordinary `input_tokens` separately from
`cache_read_input_tokens`. When both counts are present, the adapter adds them
to normalize total `Usage.input_tokens` and exposes cache reads as
`Usage.cached_tokens`. The registry tier is selected from this total, then
ordinary and cached input use their own Frankfurt Global CNY rates.

No automatic cache-write component is priced, and `cache_write_tokens` remains
`None`. An omitted cache-read split leaves `cost_input` and `cost_total` unknown;
known output cost can remain available. Opt-in caching, TTLs, storage, batch,
promotional, and negotiated rates are excluded. See the shared
[usage and pricing guide](https://github.com/Inozem/llm_api_adapter/#token-usage-and-pricing).

See the main [llm-api-adapter README](https://github.com/Inozem/llm_api_adapter/#readme)
for the shared API contract and examples.
