# llm-api-adapter-xai

Official xAI support for [llm-api-adapter](https://github.com/Inozem/llm_api_adapter/).

## Installation

```bash
pip install llm-api-adapter-xai
```

For asynchronous requests and streaming, install the core async extra too:

```bash
pip install "llm-api-adapter[async]" llm-api-adapter-xai
```

## Quick start

```python
import os

from llm_api_adapter.models.messages.chat_message import UserMessage
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter

adapter = UniversalLLMAPIAdapter(
    organization="xai",
    model="grok-4.6",
    api_key=os.environ["XAI_API_KEY"],
)

response = adapter.chat(messages=[UserMessage("Explain retrieval-augmented generation.")])
print(response.content)
```

## Capabilities

- Synchronous and asynchronous text chat.
- Synchronous and asynchronous streaming.
- Normalized usage, cost, errors, and streaming callbacks.

Tools, response continuation, structured output, reasoning controls, and file
input are not available yet.

See the main [llm-api-adapter README](https://github.com/Inozem/llm_api_adapter/#readme)
for the shared API contract and examples.

## Supported models

- `grok-4.3`
- `grok-4.5`
- `grok-4.6`
- `grok-build-0.1`
