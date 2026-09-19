# llm-api-adapter-zai

Official direct Z.ai / GLM Chat Completions API support for
[llm-api-adapter](https://github.com/Inozem/llm_api_adapter/). The package
targets Core `>=0.9.7,<1.0.0` and does not install a provider SDK.

## Installation

Install through the Core package extra:

```bash
pip install "llm-api-adapter[zai]"
```

Direct installation is also supported when Core is managed separately:

```bash
pip install llm-api-adapter-zai
```

Async requests need HTTPX:

```bash
pip install "llm-api-adapter[zai,async]"
```

Use `"llm-api-adapter[zai,httpx]"` and `transport="httpx"` to opt into the
HTTPX synchronous transport.

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

## Supported model and capabilities

The 0.1.0 release exposes the fixed model ID `glm-5.3-flash`, selected for the
broadest verified compatibility with the current Core baseline.

| Capability | `glm-5.3-flash` |
| --- | --- |
| Text chat; sync/async streaming | Supported |
| Application function tools | `tool_choice="auto"` only; at most 128 declarations |
| Reasoning | `low`, `high`, `max`; kept separate from visible text |
| Image input | URL and data-URL/bytes forms within the validation boundary |
| Portable JSON Schema / response model | Unsupported; rejected before HTTP |
| Documents and generic files | Unsupported until both direct forms pass the live gate |

The adapter uses only `POST https://api.z.ai/api/paas/v4/chat/completions` with
Bearer authentication from `ZAI_API_KEY`. Unsupported models and capabilities
are rejected before transport when locally decidable; arbitrary endpoints,
deployments, uploads, OCR, retries, and continuation are not supported.

The model has a 1,000,000-token context window and 131,072-token output limit.
Official standard rates are USD per million tokens: input `$0.15`, cached input
`$0.03`, and output `$0.50`. Missing or malformed usage is not estimated.

See the main [llm-api-adapter README](https://github.com/Inozem/llm_api_adapter/#readme)
for the shared API contract and transport behavior.
