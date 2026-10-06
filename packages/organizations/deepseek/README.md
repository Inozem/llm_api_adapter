# llm-api-adapter-deepseek

An optional DeepSeek organization package for
[LLM API Adapter](https://github.com/Inozem/llm_api_adapter/), a Python SDK with
one shared interface for calling LLM APIs.

Uses the official DeepSeek Responses API directly.

The package adds no DeepSeek SDK dependency. Select it through the existing
`UniversalLLMAPIAdapter` facade with organization `deepseek` and the canonical
model `deepseek-flash`.

## Installation

```bash
pip install "llm-api-adapter[deepseek]"
```

Direct installation is also supported when Core is managed separately:

```bash
pip install llm-api-adapter-deepseek
```

The Core extra and the direct package install provide the same organization
plugin. The extra is the recommended route when installing Core and its
optional organization support together; direct installation is useful when
Core is already installed or managed by a separate dependency set.

## Core discovery behavior

Core recognizes `deepseek` as an optional organization without importing this
package in its base installation. Selecting `deepseek` before installing the
package raises an actionable error:

```text
Organization 'deepseek' is not installed. Install it with: pip install llm-api-adapter-deepseek
```

After either installation route, the entry point is loaded lazily when the
facade selects `organization="deepseek"`. An unknown name such as
`organization="deepseek-like"` is not treated as an uninstalled DeepSeek
package and instead raises the distinct unsupported-organization error.

## Quick start

```python
import os

from llm_api_adapter.models.messages.chat_message import UserMessage
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter

adapter = UniversalLLMAPIAdapter(
    organization="deepseek",
    model="deepseek-flash",
    api_key=os.environ["DEEPSEEK_API_KEY"],
)

response = adapter.chat(
    messages=[UserMessage("Explain retrieval-augmented generation.")],
    max_tokens=256,
)
print(response.content)
```

## Supported models

- `deepseek-flash`

## Capabilities

The package uses the official Responses API with the shared Core request shape.

| Capability | `deepseek-flash` |
| --- | --- |
| Text `chat()` / `achat()` | Supported |
| Sync and async streaming | Supported |
| Application function tools | Supported; named tool selection and tool-result continuation force `reasoning_level="none"` and emit a warning because DeepSeek thinking rejects that function-tool combination. Provider-built-in tools are rejected. |
| Portable JSON Schema and Pydantic output | Supported |
| Reasoning | Supported levels: `none`, `low`, `high`, `max`; visible reasoning is opt-in |
| Images | User-message URL, bytes, and base64 data URI; JPEG, PNG, GIF, and WebP only |
| Documents and generic files | Unsupported; rejected before transport |
| OCR, Files API upload, local conversion, and document fallback | Not implemented |

The adapter rejects unsupported capabilities before either the synchronous or
asynchronous client is invoked. It does not infer support from model prefixes,
retired aliases, or an endpoint name.

Named tool selection is a `pass` exception: disabling reasoning adapts the
provider restriction while preserving the tool-call contract. Exact request
behavior, limits, and exceptions are recorded in the
[DeepSeek registry](https://github.com/Inozem/llm_api_adapter/blob/main/packages/organizations/deepseek/src/llm_api_adapter_deepseek/registry/organizations/deepseek.json).

## Images and the file boundary

`ImagePart` may be supplied only on a `UserMessage`. DeepSeek accepts an
HTTP(S) image URL, image bytes with an explicit supported MIME type, or a
base64 data URI. The adapter validates the documented boundary locally:

- supported media types are `image/jpeg`, `image/png`, `image/gif`, and
  `image/webp`;
- external image URLs are limited to 8,192 characters;
- inline image data is limited to 32 MiB per image; and
- a request may contain at most 600 images.

Every `DocumentPart`, generic `FilePart`, non-user image, unsupported image
form, OCR/upload/conversion route, and other non-image file input raises a
compatibility error before an outbound request. The package never fetches
document URLs, uploads files, runs OCR, converts documents, or silently falls
back to another endpoint or model.

## History and continuation privacy

DeepSeek Responses requests are stateless. Keep the complete conversation in
the normal `messages` list, including assistant tool calls and `ToolMessage`
results. `previous_response` does not send a provider
`previous_response_id`; it is used only to carry the matching opaque reasoning
replay material required by a subsequent DeepSeek request.

That replay material is kept in `ChatResponse.provider_data`, never merged into
visible assistant text or `reasoning_events`, never included in `repr` or
public serialization, and never logged. Set `capture_reasoning=True` only when
the caller explicitly wants provider reasoning summaries exposed through the
normal reasoning callbacks.

## Usage and standard-rate estimates

DeepSeek reports normalized `input_tokens`, `output_tokens`, and
`total_tokens`, plus optional `usage.input_tokens_details.cached_tokens` and
`usage.output_tokens_details.reasoning_tokens`. Confirmed cache reads become
`Usage.cached_tokens`, a subset of total input. There is no separately priced
automatic write component; `cache_write_tokens` remains `None`.

The USD estimate uses `pricing_tiers` and the UTC `pricing_schedule` in the
registry. The request-dispatch window selects the multiplier for ordinary
input, cached input, and output rates; ordinary input excludes the reported
cache-read subset. No rates or time windows are inferred from model names.

An omitted cache split leaves `cost_input` and `cost_total` unknown while known
output cost can remain available. Missing or invalid pricing context prevents
the estimate; no locally guessed counts or rates are substituted. Caching is
provider-managed, and opt-in controls, selectable TTLs, and storage pricing
are excluded. These are standard-rate estimates, not an invoice. See the
shared [usage and pricing guide](https://github.com/Inozem/llm_api_adapter/#token-usage-and-pricing),
[DeepSeek pricing](https://api-docs.deepseek.com/quick_start/pricing/)
and [Responses usage](https://api-docs.deepseek.com/guides/responses_api/)
documentation for provider semantics.

## Official DeepSeek references

- [Models and pricing](https://api-docs.deepseek.com/quick_start/pricing/)
- [Responses API](https://api-docs.deepseek.com/guides/responses_api/)
- [Vision input](https://api-docs.deepseek.com/guides/vision/)
- [Context caching](https://api-docs.deepseek.com/guides/kv_cache/)
- [Error codes](https://api-docs.deepseek.com/quick_start/error_codes/)
