# Anthropic-compatible Messages mode

[Back to the main README](README.md)

Use `organization="anthropic_compatible"` to connect the common SDK interface
to a server that accepts Anthropic Messages requests. This is a generic
connection, not a verified provider or model profile. Use a package version
that includes the feature, or the [current source checkout](CONTRIBUTING.md#local-setup).

## Contents

- [Quick start](#quick-start)
- [Connection selection](#connection-selection)
- [Operations, tools and files](#operations-tools-and-files)
- [History and reasoning](#history-and-reasoning)
- [Streaming and completion](#streaming-and-completion)
- [Usage and caller tariffs](#usage-and-caller-tariffs)

## Quick start

Requests is the default synchronous transport. Install `[httpx]` for optional
synchronous HTTPX or `[async]` for asynchronous calls (see [installation](README.md#installation)).
No provider SDK or optional organization package is required. Set
`MESSAGES_BASE_URL` to the API root before `/messages`, `MESSAGES_MODEL` to
the exact model ID, and `MESSAGES_API_KEY` to a nonempty key.

```python
import os

from llm_api_adapter.models.messages.chat_message import UserMessage
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter

connection = {
    "organization": "anthropic_compatible",
    "model": os.environ["MESSAGES_MODEL"],
    "api_key": os.environ["MESSAGES_API_KEY"],
    "base_url": os.environ["MESSAGES_BASE_URL"],
}
adapter = UniversalLLMAPIAdapter(**connection)
messages = [UserMessage("Explain SSE briefly.")]

response = adapter.chat(messages=messages, max_tokens=256)
print(response.content)

for text in adapter.stream_chat(messages=messages, max_tokens=256):
    print(text, end="", flush=True)
```

Every transport sends `x-api-key` and `anthropic-version: 2023-06-01`; keep
real credentials in environment variables. The SDK does not load these
variables itself.

## Connection selection

Messages is fixed for this mode: omit `api_variant` or pass `None`, and omit
`service_provider` or set it to `"anthropic_compatible"`. Other values fail
configuration validation. The model identifier is sent unchanged. Choosing a
registered-looking name does not inherit native profiles, limits, reasoning
defaults or prices. Native calls retain their usual organization choices.

`base_url` must be an HTTP(S) root with a host. Include any version or proxy
prefix; the SDK appends `/messages` once. For example,
`https://example.invalid/gateway/v1/` becomes
`https://example.invalid/gateway/v1/messages`. Roots containing URL user
information, a query, fragment, or the full `/messages` resource are invalid.
The SDK does not probe support or fall back to another API, model or service.

## Operations, tools and files

The four methods are `chat()` and `stream_chat()` (sync), plus `achat()` and
`astream_chat()` (async). Async uses optional HTTPX even when sync uses
Requests; select synchronous HTTPX with `transport="httpx"`. See the
[async endpoint example](ASYNC_API.md#compatible-endpoints). All requests
need a positive integer `max_tokens`. Default `temperature=1.0` and
`top_p=1.0` are omitted; valid nondefault values are forwarded.

Use typed messages, supported OpenAI-style dictionaries, or a mixed list.
Common `tool_choice` values are `auto`, `none`, `any` or a supplied tool name;
`parallel_tool_calls` is also supported. Use portable `json_schema` or
`response_model` for structured output. See the README's [tool calling](README.md#tool--function-calling)
and [structured output](README.md#structured-output) examples. Images support
URLs and bytes ([vision input](README.md#vision-input)); PDFs support URLs
and bytes ([document input](README.md#document-input)). Tool and schema
validation occurs locally; the SDK returns tool calls but never executes
application functions. Server support for these capabilities depends on that
server.

## History and reasoning

Send the full conversation history on each request, including assistant tool
calls and matching tool results. `previous_response` is accepted but ignored.
Valid `reasoning_level` controls produce a warning and are omitted; invalid
values still fail common validation. `capture_reasoning=True` exposes only
reasoning actually returned by the server through `reasoning_events` and the
streaming `on_reasoning` callback, separately from visible text.

## Streaming and completion

Streams yield visible text only. `buffer_chars`, when set, must be a positive
integer;
`on_chunk` receives `StreamChunk` metadata and `on_delta` receives text.
Callbacks may be synchronous or awaitable in async streaming. See the shared
[callback and lifecycle rules](ASYNC_API.md#async-streaming).

Success requires valid Messages terminal evidence, including a stop reason
and closed content blocks with complete tool arguments. EOF or `[DONE]` alone
does not establish completion. On success, pending text is flushed before
completed tool callbacks and `on_done`. Failure, cancellation or early close
closes resources without a successful final flush or `on_done`. Close a stream
stopped early with `contextlib.closing` (sync) or `contextlib.aclosing` (async).

## Usage and caller tariffs

Usage reflects valid nonnegative integer counters reported by the server.
Missing values remain unknown and explicit zero remains zero. Wire
`input_tokens` is ordinary input; wire `cache_read_input_tokens` maps to
`Usage.cached_tokens`, and `cache_creation_input_tokens` maps to
`Usage.cache_write_tokens`. `Usage.input_tokens` is inclusive input: it is
formed only when all three wire counters are valid, and equals their sum.
The alternate `prompt_tokens_details.cached_tokens` read alias is counted
once; conflicting aliases leave reads unknown. Output is independently
reported, and total is available only when inclusive input and output are
both known. Without usable reported usage, `response.usage` is `None`.
Streaming usage follows the same rules when merging updates.

Without a caller tariff, calculated costs and currency are unavailable.
Assign caller rates through the existing `Pricing` interface on the underlying
adapter; dictionary rates are per million tokens, while `PricingTier` numeric
fields are per token. JSON loading is application-owned using `currency` and
`pricing_tiers` keys, and `Pricing` also provides numeric construction and
per-million setters. See [usage and pricing concepts](README.md#token-usage-and-pricing).

```python
from llm_api_adapter.llm_registry.llm_registry import Pricing

# Illustrative caller rates per million tokens, not provider prices.
adapter.adapter.pricing = Pricing.from_dict(
    [{
        "up_to_prompt_tokens": None,
        "input_per_1m": 1.5,
        "output_per_1m": 3.0,
        "cache_read_input_per_1m": 0.2,
        "cache_write_input_per_1m": 2.0,
    }],
    currency="USD",
)
priced = adapter.chat(messages=messages, max_tokens=256)
print(priced.cost_total, priced.currency)
```

Complete total cost requires complete reported usage, rates for each reported
charged component and a tier covering inclusive input. Missing or conflicting
usage leaves affected costs unavailable; independently priceable components
may still have costs. Rates are never guessed or extrapolated. These are
caller estimates, not verified provider prices.
