# llm-api-adapter-mistral

An optional Mistral organization package for
[LLM API Adapter](https://github.com/Inozem/llm_api_adapter/), a Python SDK with
one shared interface for calling LLM APIs.

Uses the official Mistral API directly.

## Installation

Install through the core package extra (recommended):

```bash
pip install "llm-api-adapter[mistral]"
```

Direct installation remains supported when the core package is already managed
separately:

```bash
pip install llm-api-adapter-mistral
```

## Quick start

```python
import os

from llm_api_adapter.models.messages.chat_message import UserMessage
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter

adapter = UniversalLLMAPIAdapter(
    organization="mistral",
    model="mistral-large-4",
    api_key=os.environ["MISTRAL_API_KEY"],
)

response = adapter.chat(messages=[UserMessage("Explain retrieval-augmented generation.")])
print(response.content)
```

## Capabilities

- Chat, tool calling, structured JSON output, and image input.
- Synchronous and asynchronous chat and streaming.
- PDF URLs and bytes through Mistral OCR.
- Tool loops use explicit message history: append the assistant tool calls and
  `ToolMessage` results before the next request. The `previous_response` parameter
  is accepted for the shared API contract but is not sent to Mistral, whose Chat
  Completions API is stateless.

See the main [llm-api-adapter README](https://github.com/Inozem/llm_api_adapter/#readme)
for the shared API contract and examples.

## Structured-output portability

The package uses the Core portable profile for `json_schema` and Pydantic
`response_model`. See the main [Structured Output guide](https://github.com/Inozem/llm_api_adapter/#structured-output)
for schema validation, parsed results, refusal, and incomplete responses.

## Supported models

- `mistral-large-4` (Public Preview)
- `mistral-small-2603`
- `mistral-medium-3-5`
- `mistral-large-2512`
- `glm-5.3-flash` through Mistral (`zai-glm-5-3-flash` on the wire)

For Mistral's native models listed above, the output limit is an upper bound
under the shared context budget; Mistral publishes no independent generation
maximum.

`mistral-large-2512` has no reasoning control: a supplied `reasoning_level`
emits a warning and is not sent to Mistral.

The [canonical Mistral registry](https://github.com/Inozem/llm_api_adapter/blob/main/packages/organizations/mistral/src/llm_api_adapter_mistral/registry/organizations/mistral.json)
records capabilities, request rules, limits, and exceptions.

## GLM 5.3 Flash through Mistral

Install `llm-api-adapter[mistral]`; this route needs only `MISTRAL_API_KEY`:

```python
import os
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter
adapter = UniversalLLMAPIAdapter(
    organization="zai",
    service_provider="mistral",
    model="glm-5.3-flash",
    api_key=os.environ["MISTRAL_API_KEY"],
)
```

Canonical ID `glm-5.3-flash` maps to Mistral wire ID `zai-glm-5-3-flash`; no
fallback occurs. Without the provider override, `organization="zai"` selects
the independent direct Z.ai route, which still requires the Z.ai package and
key.
Supports `chat()`, `stream_chat()`, `achat()`, and `astream_chat()`. Sync uses
`requests` by default; `[httpx]` with `transport="httpx"` selects the other
sync transport. Install `[async]` for async; combine extras with
`pip install "llm-api-adapter[mistral,async,httpx]"`.

See [Capabilities](#capabilities), [Structured-output portability](#structured-output-portability),
and [PDF input](#pdf-input) for tool calling, images, schemas, and OCR.

`max_output_tokens=128000` conservatively interprets Mistral's published “128k”; it
is not a measured server maximum. Reasoning accepts `low`, `high`, or `max`
(default `low`); `reasoning_level="none"` warns and resolves to `low`.
`capture_reasoning=True` opts in to capture, separate from visible text; missing
provider reasoning is not synthesized.

Hosted Flash uses the selected Mistral profile's standard-rate estimates, not
direct Z.ai rates; see [Automatic cache usage and pricing](#automatic-cache-usage-and-pricing)
and [PDF input](#pdf-input) for cost details and boundaries.

## PDF input

`DocumentPart` supports PDF URLs and bytes. The selected Mistral chat model
does not receive PDF `DocumentPart` directly; the package OCRs each PDF first
with Mistral OCR 4.1 (`mistral-ocr-4-1`) and supplies the resulting Markdown to
the selected chat model. This creates a separate OCR API request. The adapter
reads `usage_info.pages_processed` from each OCR response and records an `ocr`
page line in `ChatResponse.cost_breakdown` at the registered standard rate.

For every currently registered Mistral model, PDF URLs and bytes are declared
`pass` exceptions: the OCR adaptation preserves the public PDF contract.

```python
from llm_api_adapter.models.messages.chat_message import UserMessage
from llm_api_adapter.models.messages.file_parts import DocumentPart

# `adapter` is configured as shown in Quick start.
response = adapter.chat(
    messages=[
        UserMessage(
            "Compare these reports.",
            files=[
                DocumentPart(url="https://example.com/report-a.pdf"),
                DocumentPart(url="https://example.com/report-b.pdf"),
            ],
        )
    ]
)

print(response.content)
ocr_line_items = [
    item
    for item in response.cost_breakdown or ()
    if item.operation == "ocr"
]
if ocr_line_items:
    for document_number, item in enumerate(ocr_line_items, start=1):
        print(f"PDF {document_number}: {item.cost} {item.currency}")
    ocr_cost = sum(item.cost for item in ocr_line_items)
    print(f"All PDFs: {ocr_cost} {ocr_line_items[0].currency}")
```

Each PDF produces an `ocr` line item. `ocr_cost` is the OCR-only estimate for
all PDFs; `response.cost_total`, when available, includes both the selected
chat model's token cost and OCR cost.

`ChatResponse.usage`, `cost_input`, and `cost_output` remain the selected chat
model's token values. `cost_total` combines those token costs and OCR lines
only when token pricing is complete, every OCR response has a valid page count,
and the OCR meter uses the same currency. Otherwise known OCR lines remain
visible but `cost_total` is `None`; page counts are never inferred from Markdown
or document bytes. These are standard-rate estimates, not an invoice, and apply
consistently to sync, async, and streaming calls.

## Automatic cache usage and pricing

`usage.prompt_tokens_details.cached_tokens` becomes `Usage.cached_tokens`,
a subset of total `Usage.input_tokens`. Every currently registered Mistral
model uses ordinary and cache-read rates from its registry tiers. There is no
separately priced automatic cache write, so `cache_write_tokens` remains
`None`.

An omitted cache-read count leaves `cost_input` and `cost_total` unknown;
known output and OCR costs can still be exposed. The package accounts for
provider-managed automatic reads without enabling opt-in cache modes, TTLs,
or storage pricing. Cache hits are provider-reported and not guaranteed. See
the shared [usage and pricing guide](https://github.com/Inozem/llm_api_adapter/#token-usage-and-pricing).

## Multi-turn requests

Mistral's `previous_response` parameter is accepted for Core API compatibility
but ignored. The package sends no provider continuation ID, so callers supply the
message history they want the next request to use. Captured native thinking is
not automatically replayed. Leaving out prior messages or thinking may reduce
quality on later turns.
