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

Synchronous requests use `requests` by default. To opt into HTTPX for sync
`chat()` and `stream_chat()`, install `"llm-api-adapter[zai,httpx]"` and pass
`transport="httpx"`.

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

The package deliberately exposes one exact, verified model ID:
`glm-5.3-flash`.

| Capability | `glm-5.3-flash` |
| --- | --- |
| Text chat; sync/async streaming | Supported through the official Chat Completions endpoint |
| Application function tools | `tool_choice="auto"` only; at most 128 declarations |
| Reasoning | Core `reasoning_level` is mapped to Z.ai effort; kept separate from visible text |
| Image input | User-message URL, bytes, and data-URL forms |
| Usage | Valid provider usage only; missing or malformed usage is unavailable |
| Portable JSON Schema / Pydantic output | Unsupported; rejected before HTTP |
| Documents and generic files | Unsupported; rejected before HTTP |
| Deployments, uploads, OCR, retries, continuation, and video | Unsupported |

Unknown, retired, aliased, or unverified model IDs are not inferred. Unsupported
tool choices, parallel-tool control, provider-built-in tools, and every
`DocumentPart` URL or byte form fail locally before the transport is called.
The adapter accepts only the official endpoint and the `ZAI_API_KEY` bearer
credential.

See the main [llm-api-adapter README](https://github.com/Inozem/llm_api_adapter/#readme)
for the shared API contract and transport behavior.
