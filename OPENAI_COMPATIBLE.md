# OpenAI-compatible mode

[← Back to the main README](README.md)

Install `llm-api-adapter` for synchronous requests, `[httpx]` for explicit
sync HTTPX or `[async]` for asynchronous calls. This guide covers the
caller-selected connection and pricing contract; native organization setup
remains in the [main README](README.md#installation).

## Contents

- [Quick start](#quick-start)
- [Connection and protocol selection](#connection-and-protocol-selection)
- [Common operations and wire forms](#common-operations-and-wire-forms)
- [Stateless history and server-dependent support](#stateless-history-and-server-dependent-support)
- [Compatible usage and caller tariffs](#compatible-usage-and-caller-tariffs)
- [Async examples](ASYNC_API.md#openai-compatible-endpoints)

## Quick start

Use `organization="openai_compatible"` with an already available server's
API root, exact model ID and explicit `api_key`. Omitted or `None`
`api_variant` selects `"chat_completions"`; pass `"responses"` to select
Responses. Explicit `"chat_completions"` is equivalent to the default.

```python
import os

from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter

connection = dict(
    organization="openai_compatible",
    model=os.environ["COMPATIBLE_MODEL"],
    api_key=os.environ["COMPATIBLE_API_KEY"],
    base_url=os.environ["COMPATIBLE_BASE_URL"],
)
adapter = UniversalLLMAPIAdapter(**connection)  # Chat Completions
response = adapter.chat(
    messages=[{"role": "user", "content": "Explain SSE briefly."}],
    max_tokens=256,
)
print(response.content)

responses = UniversalLLMAPIAdapter(**connection, api_variant="responses")
for text in responses.stream_chat(
    messages=[{"role": "user", "content": "Explain SSE briefly."}],
    max_tokens=256,
    on_done=lambda result: print(result.usage),
):
    print(text, end="", flush=True)
```

These calls contact the configured server. Set the environment variables for
your deployment; the SDK sends the supplied key as a Bearer credential. A
local server without authentication can accept a non-secret placeholder, but
`api_key` must still be a nonempty string. Keep real keys out of URLs and code.

## Connection and protocol selection

`base_url` must be an absolute HTTP(S) root with a host and valid optional
port. Include any required proxy prefix and `/v1` yourself; the SDK preserves
the prefix and removes trailing slashes. It does not add a version segment.

| `base_url` | Effective API | Request destination |
| --- | --- | --- |
| `http://localhost:8000/v1/` | Default / `chat_completions` | `http://localhost:8000/v1/chat/completions` |
| `http://localhost:8000/v1/` | `responses` | `http://localhost:8000/v1/responses` |
| `https://example.invalid/gateway/api` | `responses` | `https://example.invalid/gateway/api/responses` |

User information (`user:password@host`), query strings, fragments, and full
resource URLs ending in `/chat/completions` or `/responses` are rejected.
Unknown, blank, case-varied or non-string selectors are invalid. These
configuration errors raise `LLMConfigError` before discovery or I/O. Omit
`service_provider` or set it to `"openai_compatible"`; another provider
conflicts with this mode. For every other organization, including native
`"openai"`, non-`None` `base_url` or `api_variant` raises `LLMConfigError`;
omitting both retains its existing routing.

Any nonempty model ID is sent unchanged. A name matching a registered OpenAI
model still receives no native profile, alias resolution, capability flags,
limits, request rules or tariff. The model name never selects an API. There
is no discovery request, API probe, automatic retry on another dialect, or
fallback to another endpoint, model or official API after a failure.

## Common operations and wire forms

Both APIs support `chat()`, `achat()`, `stream_chat()` and `astream_chat()`
with the existing typed/dictionary messages, tool definitions, portable
schemas, responses and error hierarchy. Sync calls default to `requests`;
`transport="httpx"` requires `[httpx]`. Async calls always use HTTPX and
require `[async]`; see the [compatible async example](ASYNC_API.md#openai-compatible-endpoints).

| Public concept | Chat Completions | Responses |
| --- | --- | --- |
| History | `messages` | `input` items and system `instructions` |
| `max_tokens` | `max_completion_tokens` only | `max_output_tokens` only |
| `ToolSpec` | Nested `tools[].function` | Function fields directly on `tools[]` |
| Named `tool_choice` / `"any"` | Nested function choice / `"required"` | Flat function choice / `"required"` |
| Tool history | Assistant `tool_calls` and tool messages | `function_call` / `function_call_output` with matching `call_id` |
| Portable structured output | `response_format` | `text.format` |
| Streaming | Chat chunks with `stream_options={"include_usage": true}` | Typed Responses events; no Chat `stream_options` |

The output limit includes visible and reasoning tokens. Numeric sampling
defaults `temperature=1.0` and `top_p=1.0` are omitted; explicit nondefaults
are sent after common validation. Servers requiring the deprecated Chat
`max_tokens` wire field are outside this dialect. A server may reject a
supported common option; its error does not trigger a different request form.

Use the same [tools and tool choices](README.md#tool--function-calling) and
[portable schemas or response models](README.md#structured-output) as in native mode.
The SDK validates common inputs, parses completed tool arguments as objects,
and validates response-model results against the original model. It does not
execute application functions. Refusal and incomplete results retain their
terminal fields and leave parsed fields unavailable. Malformed completed
structured output raises `JSONSchemaError`; malformed returned tool arguments
raise `InvalidToolArgumentsError`.

## Stateless history and server-dependent support

Compatible Responses sends `store=false` and the complete caller-supplied
history. Append assistant calls and matching `ToolMessage` results yourself.
`previous_response` is accepted but does not send `previous_response_id`,
`conversation` or opaque reasoning replay, and does not filter history as
native OpenAI continuation does. It provides no server-side history feature.

An explicit `reasoning_level` is omitted with the existing unverified-model
warning. `capture_reasoning=True` exposes only supported reasoning actually
returned by the server, separately from visible text; it does not request a
native summary or inject an effort default. No reasoning or usage is inferred.

Common image/PDF serializers and local rejection rules still apply, while
acceptance depends on the server and selected API. Hosted tools, upload/file
services, background responses, WebSockets and vendor-specific replay or
reasoning dialects are outside this mode. Successful connectivity does not
establish verified model capabilities or prices, or certify another server.

Streams yield visible text only. A valid terminal outcome is required before
successful final flush and `on_done`; EOF or `[DONE]` alone is insufficient.
Protocol failure, cancellation and early close preserve the common cleanup
contract and do not report successful completion. Usage remains reported-data
only, and costs default to unavailable; see [compatible usage and caller tariffs](#compatible-usage-and-caller-tariffs).

## Compatible usage and caller tariffs

Compatible mode has no registered model limits or standard prices. Without
an explicit caller tariff, `cost_input`, `cost_output`, `cost_total`,
`currency` and `cost_breakdown` remain `None`, even if usage is reported.
Both APIs keep only independently reported valid counters: Chat reads
`prompt_tokens` / `completion_tokens`, and Responses reads `input_tokens` /
`output_tokens`. Missing, null or malformed usage is unavailable; a present
empty object yields a `Usage` whose counts are `None`. Invalid counters stay
`None`, reported zeros stay `0`, and no missing count or total is derived.

Initialize the existing pricing object before using its setters. This example
assigns illustrative local rates, in currency units per million tokens:

```python
from llm_api_adapter.llm_registry.llm_registry import Pricing

rates = {
    "currency": "USD",
    "pricing_tiers": [
        {
            "up_to_prompt_tokens": 1000,
            "input_per_1m": 1.5,
            "output_per_1m": 3.0,
            "cache_read_input_per_1m": 0.2,
            "cache_write_input_per_1m": 2.0,
        },
        {
            "up_to_prompt_tokens": None,
            "input_per_1m": 2.0,
            "output_per_1m": 4.0,
            "cache_read_input_per_1m": 0.3,
            "cache_write_input_per_1m": 2.5,
        },
    ],
}
adapter.adapter.pricing = Pricing.from_dict(
    rates["pricing_tiers"], currency=rates["currency"],
)
adapter.pricing.set_out_per_1m(3.5)  # Changes this rate in every tier.
adapter.pricing.set_currency("EUR")  # Changes the label, not the amounts.
```

The same dictionary can be stored as JSON using the existing registry-style
`currency` and `pricing_tiers` fields. Loading it is application code, with
no model registration or automatic price discovery:

```python
import json
from pathlib import Path

rates = json.loads(Path("caller-pricing.json").read_text(encoding="utf-8"))
adapter.adapter.pricing = Pricing.from_dict(
    rates["pricing_tiers"], currency=rates["currency"],
)
```

For parameter-based construction, `PricingTier` fields are **per token**,
unlike the per-million dictionary fields and `.set_*_per_1m` setters:

```python
from llm_api_adapter.llm_registry.llm_registry import Pricing, PricingTier

tariff = Pricing(
    tiers=(PricingTier(
        up_to_prompt_tokens=None,
        in_per_token=0.0,
        out_per_token=3.0 / 1_000_000,
    ),),
    currency="USD",
)
adapter.adapter.pricing = tariff
```

Finite zero rates are valid through direct tiers or the existing rate
setters. `Pricing.from_dict` retains its existing positive ordinary-rate
validation; its cache rates may be zero. Supply finite, nonnegative rates
and a nonempty currency. Tier boundaries are increasing positive integers;
only the last boundary may be `None`. A finite final boundary is allowed,
and usage beyond it leaves costs and currency unavailable.

Assignment to `adapter.adapter.pricing` retains the supplied object by
reference; sharing it across adapters shares subsequent setter changes.
It adds no assignment-time validation hook, so construct a valid tariff first.
Reading `adapter.pricing` delegates to that object, while assigning the whole
object to the facade's own `adapter.pricing` attribute is not forwarded.
Direct `OpenAICompatibleAdapter(..., pricing=tariff)` construction validates
and defensively copies the tariff before I/O; later mutations of the original
do not affect that instance. The facade adds no pricing constructor argument
or setter. Existing native pricing and provider-specific billing rules remain.

Both APIs calculate costs automatically for `chat()`, `achat()`,
`stream_chat()` and `astream_chat()` using the same calculator and supplied
currency. Reported total input, including cache subsets, selects one inclusive
tier for the whole request. Missing input prevents tier selection. Missing
output can leave input cost known but prevents a complete total; a missing
`total_tokens` alone does not prevent pricing known components.

Compatible cache counts come only from the selected API's reported details
(`prompt_tokens_details` or `input_tokens_details`). Cache reads and writes
are disjoint subsets of full input. A supplied cache rate needs a known valid
count; a positive reported cache component needs its own rate. Missing or
contradictory required splits leave input/total cost unknown, even if invalid
counts were discarded during normalization. A confirmed zero requires no
charge, and output cost can remain available independently. These are local
caller estimates, never verified provider prices or invoices. Non-token
charges are outside this compatible addition.

