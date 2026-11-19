# Anthropic-compatible Messages mode

[Back to the main README](README.md)

Use `organization="anthropic_compatible"` to call a server through the
Messages protocol with an explicit API root, exact model ID and key. This is
a connection mode in Core. It gives the server the common SDK request and
response interface without registering its models or verifying their
capabilities and prices. This guide describes the current source tree; see
[implementation and release status](#implementation-and-release-status).

## Contents

- [Quick start](#quick-start)
- [Connection selection](#connection-selection)
- [Operations, tools and files](#operations-tools-and-files)
- [History and reasoning](#history-and-reasoning)
- [Streaming and completion](#streaming-and-completion)
- [Usage and caller tariffs](#usage-and-caller-tariffs)
- [Implementation and release status](#implementation-and-release-status)

## Quick start

Install the base package for synchronous Requests calls, `[httpx]` for
optional synchronous HTTPX, or `[async]` for asynchronous calls:

```bash
pip install llm-api-adapter
# Optional alternatives:
pip install "llm-api-adapter[httpx]"
pip install "llm-api-adapter[async]"
```

Use a version containing this connection mode, or the current
[source checkout](CONTRIBUTING.md#local-setup). No provider SDK or optional
organization package is required.
Set `MESSAGES_BASE_URL` to your API root before `/messages`,
`MESSAGES_MODEL` to the server's exact model identifier, and
`MESSAGES_API_KEY` to its credential.

```python
import os

from llm_api_adapter.models.messages.chat_message import UserMessage
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter

connection = dict(
    organization="anthropic_compatible",
    model=os.environ["MESSAGES_MODEL"],
    api_key=os.environ["MESSAGES_API_KEY"],
    base_url=os.environ["MESSAGES_BASE_URL"],
)
adapter = UniversalLLMAPIAdapter(**connection)
messages = [UserMessage("Explain SSE briefly.")]

response = adapter.chat(messages=messages, max_tokens=256)
print(response.content)

for text in adapter.stream_chat(
    messages=messages,
    max_tokens=256,
    on_done=lambda result: print(result.usage),
):
    print(text, end="", flush=True)
```

These calls contact the configured server. The key is sent in `x-api-key`;
keep real credentials out of source and URLs. A local server that does not
authenticate may accept a non-secret placeholder, but the key must still be
a nonempty string. The SDK does not read these environment variables itself;
the example passes their values explicitly.

## Connection selection

Messages is fixed for this mode. Omit `api_variant` or pass `None`; any
other value is a configuration error. Omit `service_provider` or set it to
`"anthropic_compatible"`; a different provider conflicts with this connection.

`base_url` is an absolute HTTP(S) root with a host and optional valid port.
Include any version or proxy prefix yourself. The SDK preserves that prefix,
removes trailing slashes and appends `/messages` exactly once:

| API root | Request destination |
| --- | --- |
| `http://localhost:8000/v1/` | `http://localhost:8000/v1/messages` |
| `https://example.invalid/gateway/anthropic/v1` | `https://example.invalid/gateway/anthropic/v1/messages` |

Blank/malformed roots, non-HTTP(S) schemes, URL user information, query
strings, fragments and full resource URLs ending in `/messages` are
rejected before discovery or I/O. The dialect uses JSON, `x-api-key` and
`anthropic-version: 2023-06-01`; it does not select another API version or
endpoint from the model name. There is no support probe or automatic
retry/fallback to another API, model or official service.

Every nonempty model identifier is sent unchanged. Even a registered Claude
or Qwen name acquires no native profile, aliases, limits, request rules,
reasoning defaults or tariff. Native `organization="anthropic"` and
`organization="qwen"` retain their own setup. Native facade calls reject
non-`None` `base_url` or `api_variant`; OpenAI-compatible mode retains
[its Chat Completions/Responses selection](OPENAI_COMPATIBLE.md).

## Operations, tools and files

All four methods use the common messages, tools, response types and errors:

| Method | Result | Transport |
| --- | --- | --- |
| `chat()` | `ChatResponse` | Requests by default, optional sync HTTPX |
| `stream_chat()` | Iterator of visible `str` chunks | Requests by default, optional sync HTTPX |
| `achat()` | Awaitable returning `ChatResponse` | Optional async HTTPX |
| `astream_chat()` | Async iterator of visible `str` chunks | Optional async HTTPX |

Each request requires a positive integer `max_tokens`; the SDK has no
verified model limit from which to invent a default. Common sampling
validation applies. The numeric defaults `temperature=1.0` and `top_p=1.0`
are omitted; explicitly supplied nondefault values are sent after validation.
Whether the server accepts those controls is its own contract.

To select synchronous HTTPX, reuse the quick-start connection after
installing `[httpx]`:

```python
httpx_adapter = UniversalLLMAPIAdapter(**connection, transport="httpx")
```

`transport` selects the synchronous path. Async calls always use HTTPX,
including when the facade has its default `transport="requests"`; see the
[Messages async example](ASYNC_API.md#compatible-endpoints).
Missing extras produce actionable install guidance when that operation is
selected. Default Requests chat/stream needs neither HTTPX nor a provider SDK.

Use typed `Prompt`, `UserMessage`, `AIMessage` and `ToolMessage`, supported
OpenAI-style dictionaries, or a mixed list. System content is separated into
Messages `system`; supported turn/content order and tool call/result IDs
are retained.

| Common input | Messages form |
| --- | --- |
| `ToolSpec` | `tools[].name`, `description`, `input_schema` |
| `tool_choice="auto"`, `"none"`, `"any"` | Matching `tool_choice.type` |
| Named `tool_choice` | `{"type": "tool", "name": ...}` |
| `parallel_tool_calls` | Inverse `disable_parallel_tool_use` |
| Portable `json_schema` or `response_model` | `output_config.format.type="json_schema"` |
| Image URL, bytes or data URI | Existing Messages image source serialization |
| PDF URL or bytes | Existing Messages document source serialization |

Tool/schema combinations, named choices and portable schemas are validated
locally before transport. Completed calls expose parsed object arguments in
`ChatResponse.tool_calls`; the SDK never executes application functions.
Append assistant calls and matching tool results yourself. See the shared
[tool round-trip](README.md#tool--function-calling) and
[portable structured-output profile](README.md#structured-output).

```python
from llm_api_adapter.models.tools import ToolSpec

lookup = ToolSpec(
    name="lookup",
    description="Look up an item in the application.",
    json_schema={
        "type": "object",
        "properties": {"item": {"type": "string"}},
        "required": ["item"],
        "additionalProperties": False,
    },
)
tool_response = adapter.chat(
    messages=[UserMessage("Look up the delivery policy.")],
    max_tokens=256,
    tools=[lookup],
    tool_choice="any",
)
for call in tool_response.tool_calls:
    print(call.name, call.arguments)
```

Valid completed structured output populates `parsed_json` and, for a
response model, `parsed_model`. Refusal or incomplete outcomes leave parsed
fields unavailable. Invalid completed output raises `JSONSchemaError`;
malformed completed tool arguments raise `InvalidToolArgumentsError`.

```python
schema = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
    "additionalProperties": False,
}
structured = adapter.chat(
    messages=[UserMessage("Summarize SSE in one sentence.")],
    max_tokens=256,
    json_schema=schema,
)
print(structured.parsed_json)
```

Use the existing `ImagePart` and `DocumentPart` types for files, for example:

```python
from pathlib import Path

from llm_api_adapter.models.messages.file_parts import DocumentPart

document = UserMessage(
    "Summarize this PDF.",
    files=[DocumentPart(
        data=Path("report.pdf").read_bytes(),
        media_type="application/pdf",
    )],
)
pdf_response = adapter.chat(messages=[document], max_tokens=256)
print(pdf_response.content)
```

Bytes are encoded locally; URLs are passed through without downloading.
Unsupported local file forms are rejected. There is no upload, OCR or format
conversion service. Serialization support does not establish that a server
accepts tools, schemas, files, reasoning or parallel-tool controls. Server
rejection does not trigger a different request form or certify a capability.

## History and reasoning

Supply full caller history on every request, including assistant tool calls
and matching tool results. `previous_response` is accepted but ignored;
no continuation ID, implicit storage request or opaque replay is added.

A valid `reasoning_level` produces the existing unverified-model warning and
is omitted. Invalid common values still fail validation. Generic mode does
not enable native thinking, choose an effort/budget or request a reasoning
summary. `capture_reasoning=True` exposes supported reasoning actually
returned by the server through `reasoning_events` and streaming
`on_reasoning`, separately from visible text. No returned reasoning means
no synthesized reasoning. Native Claude/Qwen policies remain separate.

## Streaming and completion

Streams yield visible text only. Optional `buffer_chars` is `None` or a
positive integer; buffered chunks never exceed it. `on_chunk` receives
`StreamChunk` metadata, `on_delta` receives text, and streaming async callbacks
may be synchronous or awaitable. See the shared
[callback and lifecycle rules](ASYNC_API.md#async-streaming).

Successful generic completion requires a valid Messages start/terminal
sequence, a stop reason and closed content blocks with complete tool
arguments. EOF or a transport `[DONE]` alone is insufficient. On normal
completion, pending text is flushed, completed tools are delivered through
`on_tool_call`, then `on_done` receives the final normalized response with
available usage/cost and parsed fields. A valid refusal or incomplete
generation remains an outcome with unavailable parsed fields.

On failure, cancellation or early closure, owned resources close without a
successful final flush or `on_done`. Callback failures remain caller failures,
and cleanup failures do not replace the original failure. Close an iterator
when stopping early, using `contextlib.closing` for sync or
`contextlib.aclosing` for async. Generic strict completion does not change
native parser tolerance or raw native client event APIs.

## Usage and caller tariffs

Usage contains only valid reported nonnegative integer counters; booleans,
invalid values and missing components remain unknown. Explicit zero stays
zero. Without reported usable usage, `response.usage` is `None`.

| Wire component | Common result |
| --- | --- |
| `input_tokens` | Ordinary input component, excluding cache reads/writes |
| `cache_read_input_tokens` | `Usage.cached_tokens` |
| `cache_creation_input_tokens` | `Usage.cache_write_tokens` |
| All three valid input components | Their sum is `Usage.input_tokens` |
| `output_tokens` | Independently reported `Usage.output_tokens` |
| Known inclusive input and output | Their sum is `Usage.total_tokens` |

The alternate `prompt_tokens_details.cached_tokens` mirrors cache reads; it
is counted once. Conflicting aliases leave that component unknown. Missing
ordinary input, reads or writes prevents a complete inclusive input, even if
another service would default an omitted field to zero. No native accounting
default or reported total fills missing components. For example, ordinary
input 100, reads 20, writes 10 and output 5 normalize to input 130 and total
135. The streams merge reported start/final usage under the same rules.
Later alias observations replace the earlier read component; a reported
null/malformed update clears stale usage. Wholly absent usage stays absent.

There is no default tariff, even for a registered-looking model name.
Without an explicit caller tariff, calculated `cost_input`, `cost_output`,
`cost_total`, `currency` and `cost_breakdown` remain unavailable. Supply local
rates through the existing pricing types on the underlying adapter:

```python
from llm_api_adapter.llm_registry.llm_registry import Pricing

# Illustrative local rates per million tokens, not provider prices.
rates = {
    "currency": "USD",
    "pricing_tiers": [{
        "up_to_prompt_tokens": None,
        "input_per_1m": 1.5,
        "output_per_1m": 3.0,
        "cache_read_input_per_1m": 0.2,
        "cache_write_input_per_1m": 2.0,
    }],
}
adapter.adapter.pricing = Pricing.from_dict(
    rates["pricing_tiers"], currency=rates["currency"],
)
priced = adapter.chat(messages=messages, max_tokens=256)
print(priced.usage, priced.cost_total, priced.currency)
```

Explicit JSON loading belongs to application code and uses the same schema:

```python
import json
from pathlib import Path

rates = json.loads(Path("caller-pricing.json").read_text(encoding="utf-8"))
adapter.adapter.pricing = Pricing.from_dict(
    rates["pricing_tiers"], currency=rates["currency"],
)
```

For numeric construction, `PricingTier` fields are per token; dictionary
fields and the existing `set_*_per_1m` methods use per-million units:

```python
from llm_api_adapter.llm_registry.llm_registry import Pricing, PricingTier

tariff = Pricing(
    tiers=(PricingTier(
        up_to_prompt_tokens=None,
        in_per_token=0.0,
        out_per_token=3.0 / 1_000_000,
        cache_read_in_per_token=0.0,
        cache_write_in_per_token=0.0,
    ),),
    currency="USD",
)
adapter.adapter.pricing = tariff
adapter.pricing.set_out_per_1m(3.5)
adapter.pricing.set_currency("EUR")  # Relabels the currency; no conversion.
```

Direct numeric tiers and setters accept finite zero rates.
`Pricing.from_dict` retains positive ordinary-rate validation; its cache rates
may be zero. Rates must be finite and nonnegative, currency nonempty, and
boundaries increasing positive integers with `None` only at the end. The
current tariff, including an assigned or subsequently mutated object, is
validated before each generic request is prepared for I/O.

Ordinary assignment to `adapter.adapter.pricing` keeps the object reference;
sharing it shares later setter changes. Reading `adapter.pricing` delegates
to that same object, but whole-object assignment to the facade's own
`adapter.pricing` attribute is not forwarded. Direct
`AnthropicCompatibleAdapter(..., pricing=tariff)` construction defensively
copies a validated tariff. The facade adds no pricing constructor argument.
Native pricing and caller-local native overrides retain their own behavior.

All four methods use the shared calculator. Inclusive input selects one tier
for the whole request, with inclusive upper boundaries. Missing input
prevents tier selection; a finite schedule that does not cover the input
leaves calculated costs and currency unavailable. No rate is extrapolated.
Missing output prevents a complete total. A positive cache component without
its own rate, or incomplete/contradictory required usage, prevents complete
input/total cost; independently priceable output can remain available in a
covered tier. Known zero incurs no cache charge. These are caller estimates,
not verified provider prices or invoices. Explicit cache controls, TTLs,
cache resources and non-token charges are outside this connection addition.

## Implementation and release status

The generic Messages implementation has deterministic source-tree evidence
for requests, transports, conversation forms, lifecycle, accounting and
optional dependency boundaries. The complete generic source-readiness review
is accepted. Installed-candidate acceptance remains pending; the source tree
and its current version do not establish a published feature-bearing artifact.

Native Claude's adapter and both clients use the shared Messages foundation;
its final migration review is still pending. The four existing native Qwen
profiles retain their package/workspace interface while their remaining
shared-foundation migration is pending. Keep using
`organization="qwen"` with its optional package for verified Qwen policy;
see the [Qwen guide](packages/organizations/qwen/README.md).
`qwen4-max` is a planned later addition, not a model registered or certified
by this guide.

The existing native provider lanes cover only the shared paths they reach.
They do not certify an arbitrary compatible server. Final provider acceptance
still requires the existing post-staging-merge exact-TestPyPI candidate
[release flow](CONTRIBUTING.md#release-flow); no separate compatible paid
lane is introduced.
