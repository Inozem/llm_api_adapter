# OpenAI-compatible mode

[Back to the main README](README.md)

Install `llm-api-adapter` for synchronous Requests calls. Add `[httpx]` for
sync HTTPX transport or `[async]` for asynchronous calls. This guide covers
the connection-specific choices; see the [main README](README.md#installation)
for native organization setup.

## Contents

- [Quick start](#quick-start)
- [Connection and protocol selection](#connection-and-protocol-selection)
- [Common operations and wire forms](#common-operations-and-wire-forms)
- [Stateless history and server-dependent support](#stateless-history-and-server-dependent-support)
- [Compatible usage and caller tariffs](#compatible-usage-and-caller-tariffs)
- [Async examples](ASYNC_API.md#compatible-endpoints)

## Quick start

Provide the server's API root, exact model ID and nonempty API key. Omitted
or `None` `api_variant` selects Chat Completions; explicit
`"chat_completions"` is equivalent. Use `"responses"` to select Responses.

```python
import os

from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter

connection = dict(
    organization="openai_compatible",
    model=os.environ["COMPATIBLE_MODEL"],
    api_key=os.environ["COMPATIBLE_API_KEY"],
    base_url=os.environ["COMPATIBLE_BASE_URL"],
)
adapter = UniversalLLMAPIAdapter(**connection)  # Chat Completions by default
response = adapter.chat(
    messages=[{"role": "user", "content": "Hello"}], max_tokens=256,
)
print(response.content)

responses = UniversalLLMAPIAdapter(**connection, api_variant="responses")
```

The key is sent as a Bearer credential. A local unauthenticated server can
use a non-secret placeholder, but the key must still be a nonempty string.
Keep real keys out of URLs and source code.

## Connection and protocol selection

`base_url` must be an absolute HTTP(S) root with a host and valid optional
port. Include any proxy prefix and `/v1` required by the server; trailing
slashes are removed and no version segment is added. For example,
`http://localhost:8000/v1/` targets `/v1/chat/completions` by default and
`/v1/responses` with `api_variant="responses"`.

Roots containing user information, a query or fragment, and full resource
URLs ending in `/chat/completions` or `/responses` are rejected. The model ID
is sent unchanged and need not be registered. Omit `service_provider` or set
it to `"openai_compatible"`; another value conflicts with this mode. Invalid
configuration raises `LLMConfigError` before discovery or network I/O.

Native organizations reject non-`None` `base_url` and `api_variant`; omitting
them preserves native routing. Any model name, including one matching a
registered OpenAI model, gets no native aliases, limits, request rules,
reasoning defaults or tariff here. The model never selects the API. The SDK
does not probe, retry another dialect, or fall back to another endpoint,
model or official service.

## Common operations and wire forms

Both APIs support `chat()`, `achat()`, `stream_chat()` and `astream_chat()`.
They accept the shared [message, tool and structured-output forms](README.md#tool--function-calling)
and response concepts. Sync calls use Requests by default; pass
`transport="httpx"` to use the optional `[httpx]` dependency. Async calls
always use HTTPX and require `[async]`; see the [compatible async examples](ASYNC_API.md#compatible-endpoints).

| Public concept | Chat Completions | Responses |
| --- | --- | --- |
| History | `messages` | `input` items and system `instructions` |
| `max_tokens` | `max_completion_tokens` | `max_output_tokens` |
| Tools and choice | Nested function fields; `"any"` maps to `"required"` | Flat function fields; `"any"` maps to `"required"` |
| Tool history | Assistant `tool_calls` and tool messages | `function_call` / `function_call_output` with matching `call_id` |
| Structured output | `response_format` | `text.format` |
| Stream usage | `stream_options={"include_usage": true}` | Responses events; no Chat `stream_options` |

Sampling defaults to `temperature=1.0` and `top_p=1.0`; defaults are omitted.
Explicit non-defaults are validated and forwarded. Both output-limit fields
include visible and reasoning tokens. A server may not support every common
form; its error does not trigger a different request. Chat servers that
require the legacy wire field `max_tokens` are outside this dialect.

The SDK validates common inputs and returned structured output, parses tool
arguments, and does not execute application tools. See the shared
[structured-output guide](README.md#structured-output) for the common API.

## Stateless history and server-dependent support

Responses requests set `store=false` and send the complete caller-supplied
history. Append assistant calls and matching tool results yourself.
`previous_response` is accepted but ignored: no `previous_response_id`,
conversation state or opaque replay is sent, and history is not filtered.

An explicit `reasoning_level` is omitted with the unverified-model warning.
`capture_reasoning=True` exposes only reasoning returned by the server; it
does not request a summary or add an effort default. Common image/PDF forms
and local rejection rules apply, but server support depends on the selected
API. Hosted tools, uploads, background responses, WebSockets and vendor
extensions are server-dependent and outside this connection contract.

Streams yield visible text only. Successful completion requires a valid
terminal outcome; EOF or `[DONE]` alone is insufficient. Failure, cancellation
or early close does not report successful completion. Follow the shared
[stream lifecycle and early-close guidance](README.md#streaming) when consuming
streams. Usage and cost remain based on reported data; see [caller tariffs](#compatible-usage-and-caller-tariffs).

## Compatible usage and caller tariffs

No caller tariff means calculated costs and currency are unavailable, even
when usage is reported. Chat reads `prompt_tokens` / `completion_tokens`;
Responses reads `input_tokens` / `output_tokens`. Each valid reported counter
is kept independently: missing counts remain `None`, reported zero stays `0`,
and no missing counter or total is fabricated.

Assign an existing `Pricing` object to `adapter.adapter.pricing`. This compact
example uses illustrative local rates, not provider prices; dictionary rates
are per million tokens, including cache rates:

```python
from llm_api_adapter.llm_registry.llm_registry import Pricing

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
response = adapter.chat(messages=[{"role": "user", "content": "Hello"}])
print(response.cost_total)
```

The same `currency` and `pricing_tiers` fields can be stored as JSON and
loaded by application code. `PricingTier` constructor rates are per token;
dictionary fields and `.set_*_per_1m` setters use per-million rates. See
[token usage and pricing](README.md#token-usage-and-pricing)
for shared pricing behavior and types.

Cache reads and writes (`cached_tokens` and `cache_write_tokens`) are subsets
of inclusive total input, not added to it. They come only from
`prompt_tokens_details` in Chat usage or `input_tokens_details` in Responses
usage. The inclusive input count selects one tier for the whole request. If a
finite final tier does not cover it, costs and currency stay unavailable;
output cost also requires a covered input tier. A missing or contradictory
required usage detail or rate leaves the affected cost unknown. A total
requires all incurred token costs to be known. These are caller estimates,
not provider invoices.
