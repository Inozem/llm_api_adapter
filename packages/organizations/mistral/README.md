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
    model="mistral-large-2512",
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
  `ToolMessage` results before the next request. `previous_response` is
  accepted for the shared API contract but is not sent to Mistral, whose Chat
  Completions API is stateless.

See the main [llm-api-adapter README](https://github.com/Inozem/llm_api_adapter/#readme)
for the shared API contract and examples.

## Structured-output portability

The package uses the Core portable profile for `json_schema` and Pydantic
`response_model`. See the main [Structured Output guide](https://github.com/Inozem/llm_api_adapter/#structured-output)
for schema validation, parsed results, refusal, and incomplete responses.

## Supported models

- `mistral-small-2603`
- `mistral-medium-3-5`
- `mistral-large-2512`

`mistral-large-2512` has no reasoning control: a supplied `reasoning_level`
emits a warning and is not sent to Mistral. Exact capabilities, request rules,
and exceptions are recorded in the [Mistral registry](https://github.com/Inozem/llm_api_adapter/blob/main/packages/organizations/mistral/src/llm_api_adapter_mistral/registry/organizations/mistral.json).

## PDF input

`DocumentPart` supports PDF URLs and bytes. Before the chat request, the
selected Mistral chat model cannot consume `DocumentPart` directly. The package
therefore sends each PDF to Mistral OCR 4.1 (`mistral-ocr-4-1`) and supplies the
resulting Markdown to the selected chat model. This creates a separate OCR API
request. The adapter reads `usage_info.pages_processed` from each OCR response
and records an `ocr` page line in `ChatResponse.cost_breakdown` at the
registered standard rate.

For all three models, PDF URLs and bytes are declared `pass` exceptions:
the OCR adaptation preserves the public PDF contract.

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
a subset of total `Usage.input_tokens`. All three models use ordinary and
cache-read rates from their registry tiers. There is no separately priced
automatic cache write, so `cache_write_tokens` remains `None`.

An omitted cache-read count leaves `cost_input` and `cost_total` unknown;
known output and OCR costs can still be exposed. The package accounts for
provider-managed automatic reads without enabling opt-in cache modes, TTLs,
or storage pricing. See the shared [usage and pricing guide](https://github.com/Inozem/llm_api_adapter/#token-usage-and-pricing).
