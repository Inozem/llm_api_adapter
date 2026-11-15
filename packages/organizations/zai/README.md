# llm-api-adapter-zai

An optional Z.ai / GLM organization package for
[LLM API Adapter](https://github.com/Inozem/llm_api_adapter/), a Python SDK with
one shared interface for calling LLM APIs.

Calls the official Z.ai API with `organization="zai"`.
Requires Python **3.10+** and
`llm-api-adapter>=0.10.0,<1.0.0`.

## Installation

Install through the Core package extra:

```bash
pip install "llm-api-adapter[zai]"
```

Or install the organization package directly:

```bash
pip install llm-api-adapter-zai
```

Async requests need the optional async dependencies:

```bash
pip install "llm-api-adapter-zai[async]"
```

Synchronous requests use `requests` by default. To opt into HTTPX for sync
`chat()` and `stream_chat()`, install `"llm-api-adapter-zai[httpx]"`
and pass `transport="httpx"`.

The low-level `ZaiSyncClient(api_key, transport="requests", *, sync_transport=None)`
and `ZaiAsyncClient(api_key)` bind credentials when constructed. Call
`complete(model, timeout=None, **payload)` or
`chat_completion(model, timeout=None, **payload)` for a raw JSON object (`dict`);
call `stream(model, timeout=None, **payload)` or
`stream_chat_completion(model, timeout=None, **payload)` for raw SSE events.
For example, use `client.complete("glm-5.3-flash", messages=[{"role": "user", "content": "Hello"}])`.

## Quick start

```python
import os

from llm_api_adapter.models.messages.chat_message import UserMessage
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter

adapter = UniversalLLMAPIAdapter(
    organization="zai",
    model="glm-5.3-flash",
    api_key=os.environ["ZAI_API_KEY"],
)

response = adapter.chat(
    messages=[UserMessage("Explain retrieval-augmented generation.")],
    max_tokens=128,
)
print(response.content)
```

Use `chat`, `achat`, `stream_chat` or `astream_chat`. Explicit
`service_provider="zai"` is also supported. Supply your Z.ai credential through
`api_key`; `base_url` and `api_variant` overrides are unsupported.

`max_tokens` must be a positive integer within the model's output limit. Omit it
to use the provider default. Use `timeout_s` to set the request timeout.

## Streaming

Use the shared [streaming API](https://github.com/Inozem/llm_api_adapter/#streaming).
Handle errors around iterator consumption (`for`, `next` or `list`): request
preparation and validation happen during iteration, with local errors raised
before provider I/O. Keep mutable request inputs stable until consumption.

## Supported models

- `glm-5.3-flash`
- `glm-5.3-flashx`
- `glm-5.3-flashx-legal`

See the [Z.ai registry](https://github.com/Inozem/llm_api_adapter/blob/main/packages/organizations/zai/src/llm_api_adapter_zai/registry/organizations/zai.json)
for model limits, supported reasoning levels and standard prices.

## Capabilities

| Capability | All supported models |
| --- | --- |
| Sync/async chat and streaming | Supported |
| Application function tools | `tool_choice="auto"` only; at most 128 declarations |
| Reasoning controls | `reasoning_level`: provider, canonical or numeric value |
| Reasoning capture | Opt-in `capture_reasoning=True`; separate response events and stream callbacks |
| Image input | User-message URL, bytes, and data-URL forms |
| Usage | Valid reported counts, including zero; unknown components stay unavailable |
| Portable JSON Schema / Pydantic output | Unsupported; rejected before HTTP |
| PDF, other documents and generic files | Unsupported; rejected before HTTP |

Unsupported tool choices and explicit `parallel_tool_calls` are rejected before
HTTP. Function calls are returned to your application; the SDK does not execute
tools. Requests are not retried automatically.

`reasoning_level="none"` uses the minimum registered effort and emits a
warning because these models cannot disable thinking. Capture is off by default.
When enabled, reported reasoning is available through `response.reasoning_events`
and `on_reasoning` for streams, separate from visible text.

`previous_response` is accepted but ignored: send complete `messages`
history, including assistant tool calls and tool results, on each turn.

## Automatic cache usage and pricing

`response.usage` contains valid provider-reported counts. Missing or invalid
counts stay unavailable; explicit zero remains zero. `usage.cached_tokens` is
a confirmed subset of input tokens, charged once at the model's cache-read rate.
Costs use registered USD prices or your caller-local pricing override.

An omitted cache-read split leaves `cost_input` and `cost_total` unknown;
known output cost can remain available. Cache-write counts stay `None`;
explicit cache controls, TTLs and storage pricing are unsupported. See the shared
[usage and pricing guide](https://github.com/Inozem/llm_api_adapter/#token-usage-and-pricing).

See the main [llm-api-adapter README](https://github.com/Inozem/llm_api_adapter/#readme)
for the shared API contract and transport behavior.
