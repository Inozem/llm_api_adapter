# llm-api-adapter-qwen

An optional Qwen organization package for
[LLM API Adapter](https://github.com/Inozem/llm_api_adapter/), a Python SDK with
one shared interface for calling LLM APIs.

Uses Model Studio's Frankfurt Global deployment through its
Anthropic-compatible Messages API.

## Installation

```bash
pip install "llm-api-adapter[qwen]"
```

Install `llm-api-adapter-qwen` directly if you manage Core separately.
Use `llm-api-adapter[qwen,async]` for async calls. For sync HTTPX, use
`llm-api-adapter[qwen,httpx]` and pass `transport="httpx"`. Sync defaults to Requests.

## Quick start

This example uses a package build with Qwen4-Max support.

```python
import os

from llm_api_adapter.models.messages.chat_message import UserMessage
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter

adapter = UniversalLLMAPIAdapter(
    organization="qwen",
    model="qwen4-max",
    api_key=os.environ["QWEN_API_KEY"],
)

response = adapter.chat(
    messages=[UserMessage("Explain retrieval-augmented generation.")],
    max_tokens=1024,
    workspace_id=os.environ["QWEN_WORKSPACE_ID"],
)
print(response.content)
```

Pass your Frankfurt `workspace_id` explicitly to each chat or streaming call.

## Supported models

- `qwen4-max`
- `qwen3.8-max`
- `qwen3.8-flash`
- `qwen3.7-plus`
- `qwen3.7-flash`

## Capabilities

All five models support text, tools, JSON Schema or Pydantic output, and images
by URL or bytes in sync and async calls.

The SDK maps the shared `reasoning_level` to each model's native controls.
[Reasoning support](https://github.com/Inozem/llm_api_adapter/#reasoning-support)

Pass a positive integer `max_tokens`. For Qwen4-Max, the output limit depends
on the thinking mode.
[Qwen4-Max limits and defaults](https://help.aliyun.com/en/model-studio/qwen4-max)

Qwen4-Max keeps the requested or default thinking mode with forced tool choices.
Qwen 3.8 and 3.7 disable thinking for forced `"any"` or named tools, warning when
the adapter changes the request. The adapter ignores `previous_response`.
Pass full message history on each turn, including tool calls and results.
See the exact rules in the [Qwen registry](https://github.com/Inozem/llm_api_adapter/blob/main/packages/organizations/qwen/src/llm_api_adapter_qwen/registry/organizations/qwen.json).

## PDF input

Qwen4-Max and both Qwen 3.8 models accept PDFs. Add `DocumentPart(url=...)` or
`DocumentPart(data=pdf_bytes, media_type="application/pdf")` to
`UserMessage.files`. Qwen 3.7 and unregistered models reject PDFs locally.

Model Studio bills PDF parsing per page. The SDK adds the charge to
`cost_breakdown` when the provider reports a page count. Without that count,
`cost_total` is unknown. [Model Studio PDF guide](https://help.aliyun.com/en/model-studio/pdf-understanding)

## Automatic cache usage and pricing

The SDK uses Frankfurt Global rates from the Qwen catalogue. Qwen4-Max
includes reported cache writes in input usage. Missing required
counters or unpriced charges leave the full cost unknown.
[Shared usage and pricing guide](https://github.com/Inozem/llm_api_adapter/#token-usage-and-pricing)
