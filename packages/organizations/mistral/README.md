# llm-api-adapter-mistral

An optional Mistral organization package for
[LLM API Adapter](https://github.com/Inozem/llm_api_adapter/), a Python SDK with
one shared interface for calling LLM APIs.

## Installation

Install the package through the Core extra:

```bash
pip install "llm-api-adapter[mistral]"
```

Direct installation is also available with `pip install llm-api-adapter-mistral`.

Synchronous calls use `requests` by default. Install the `[httpx]` extra and set
`transport="httpx"` to use HTTPX for synchronous calls. Install the `[async]`
extra for asynchronous methods. The package manifest lists Core constraints and
extras in [`pyproject.toml`](pyproject.toml).

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

Generation now defaults to `https://api.mistral.ai/v1/responses`, replacing the
previous `https://api.mistral.ai/v1/chat/completions` default. To customize the
native `MistralAdapter`, pass the complete Responses resource URL as `endpoint`.
The adapter uses that URL as supplied.

## Supported models and routing

| Model | `organization` | `service_provider` |
| --- | --- | --- |
| `mistral-large-4` (Public Preview) | `mistral` | default |
| `mistral-small-2603` | `mistral` | default |
| `mistral-medium-3-5` | `mistral` | default |
| `mistral-large-2512` | `mistral` | default |
| `glm-5.3-flash` | `zai` | `mistral` |
| `deepseek-flash` (Public Preview) | `deepseek` | `mistral` |

For hosted GLM and DeepSeek, use the developer's `organization` with
`service_provider="mistral"` and your Mistral API key. Omitting
`service_provider` selects the developer's direct integration. Native Mistral
profiles, reasoning behavior, limits, and tariffs are listed in the
[Mistral registry](src/llm_api_adapter_mistral/registry/organizations/mistral.json).

## Supported behavior

The package supports `chat`, `achat`, `stream_chat`, and `astream_chat`.
Features include portable JSON Schema and Pydantic `response_model` output,
application tool calls, reasoning controls, opt-in reasoning capture, images,
and PDFs. The SDK returns tool calls for the application to handle. For a tool
loop, include the assistant tool calls and matching `ToolMessage` results in
message history. Set `capture_reasoning=True` to collect reasoning returned by
the provider. Captured reasoning remains separate from visible text.

Pass the full message history on every request. Mistral ignores
`previous_response`. Reasoning behavior and supported values vary by model and
are recorded in the registry.

See the shared [API guide](https://github.com/Inozem/llm_api_adapter/#readme),
[structured output guide](https://github.com/Inozem/llm_api_adapter/#structured-output),
and [usage and pricing guide](https://github.com/Inozem/llm_api_adapter/#token-usage-and-pricing).

## PDF input

`DocumentPart` accepts PDF URLs, bytes, and data URLs. The package uses the
separate Mistral OCR operation with model `mistral-ocr-4-1`. Reported page
charges appear in `response.cost_breakdown` at the registry rate.

When available, `cost_total` includes both the selected model's token costs and
the separate OCR costs. Incomplete pricing, missing or invalid page counts, and
currency mismatches leave `cost_total` at `None`. Known OCR lines remain visible.
Page counts are not inferred from document content. These standard-rate
estimates apply to synchronous, asynchronous, and streaming calls.
