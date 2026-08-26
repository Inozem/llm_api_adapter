# llm-api-adapter-mistral

Official direct-API support for Mistral in
[llm-api-adapter](https://github.com/Inozem/llm_api_adapter/).

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

## Supported models

- `mistral-small-2603`
- `mistral-medium-3-5`
- `mistral-large-2512`

## PDF input

`DocumentPart` supports PDF URLs and bytes. Before the chat request, the
adapter sends each PDF to Mistral OCR 4.1 (`mistral-ocr-4-1`) and supplies the resulting Markdown to
the selected chat model. This creates a separate OCR API request, subject to
Mistral's OCR limits and pricing; `ChatResponse` usage and cost cover only the
chat completion.
