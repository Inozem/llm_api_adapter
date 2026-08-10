# Asynchronous API

[← Back to the main README](README.md)

`llm-api-adapter` provides an optional asynchronous API for OpenAI, Anthropic,
and Google. It uses `httpx` and keeps the same provider-neutral messages,
tools, structured output, file inputs, pricing, usage, and reasoning options
as the synchronous API.

## Contents

- [Installation and compatibility](#installation-and-compatibility)
- [Async request](#async-request)
- [Messages and providers](#messages-and-providers)
- [Async streaming](#async-streaming)
- [Tool calling](#tool-calling)
- [Structured output](#structured-output)
- [Vision input](#vision-input)
- [Document input](#document-input)
- [Usage and pricing](#usage-and-pricing)
- [Lifecycle and errors](#lifecycle-and-errors)
- [Timeouts](#timeouts)
- [Reasoning level](#reasoning-level)
- [Reasoning observability](#reasoning-observability)
- [Logging](#logging)

## Installation and compatibility

Install the optional dependency before using asynchronous methods:

```bash
pip install "llm-api-adapter[async]"
```

The `[async]` extra adds `httpx`; it is imported lazily when an async request
is made. The synchronous API continues to use `requests` and does not require
`httpx`.

`achat()` and `astream_chat()` are available through the same
`UniversalLLMAPIAdapter` facade for OpenAI, Anthropic, and Google. They accept
the same provider-neutral messages, tools, structured-output, file, pricing,
usage, and reasoning options as their synchronous counterparts.

Both async methods accept the common request parameters `messages`,
`max_tokens`, `temperature`, `top_p`, `reasoning_level`, `timeout_s`, `tools`,
`tool_choice`, `parallel_tool_calls`, `previous_response`, `json_schema`,
`response_model`, and `capture_reasoning`. `astream_chat()` additionally
accepts `on_delta`, `on_tool_call`, `on_done`, `buffer_chars`, `on_chunk`, and
`on_reasoning`, and returns an async iterator of visible text strings.

`ImagePart` and `DocumentPart` are supported by both async methods. File bytes
are encoded locally and sent in the async HTTPX request; file URLs are passed
to the provider and are not downloaded by the adapter. As with the synchronous
API, `DocumentPart` URLs require OpenAI Responses API models; OpenAI Chat
Completions supports PDF bytes but not PDF URLs.

## Async request

```python
import asyncio
import os

from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter


async def main():
    adapter = UniversalLLMAPIAdapter(
        organization="openai",
        model="gpt-5",
        api_key=os.environ["OPENAI_API_KEY"],
    )
    response = await adapter.achat(
        messages=[{"role": "user", "content": "Explain SSE in one sentence."}],
        max_tokens=80,
    )
    print(response.content)


asyncio.run(main())
```

## Messages and providers

Async requests use the same typed messages as the synchronous API: `Prompt`
sets context, `UserMessage` adds user input, and `AIMessage` represents an
earlier assistant response. The adapter also accepts OpenAI-style dictionaries,
including a mix of dictionaries and typed messages in one list.

```python
import asyncio
import os

from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter


async def main():
    adapter = UniversalLLMAPIAdapter(
        organization="openai",
        model="gpt-5",
        api_key=os.environ["OPENAI_API_KEY"],
    )
    messages = [
        {
            "role": "system",
            "content": "You are a friendly assistant who answers only yes or no.",
        },
        {"role": "user", "content": "Do you know how AI learns?"},
        {"role": "assistant", "content": "Yes."},
        {"role": "user", "content": "Can you explain it in one sentence?"},
    ]
    response = await adapter.achat(messages=messages, max_tokens=50)
    print(response.content)


asyncio.run(main())
```

Each `UniversalLLMAPIAdapter` instance is tied to one provider and model. To
switch providers, construct another instance and await the same request shape:

```python
import asyncio
import os

from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter


async def main():
    messages = [{"role": "user", "content": "Explain SSE in one sentence."}]

    gpt = UniversalLLMAPIAdapter(
        organization="openai",
        model="gpt-5",
        api_key=os.environ["OPENAI_API_KEY"],
    )
    print((await gpt.achat(messages=messages)).content)

    claude = UniversalLLMAPIAdapter(
        organization="anthropic",
        model="claude-sonnet-4-5",
        api_key=os.environ["ANTHROPIC_API_KEY"],
    )
    print((await claude.achat(messages=messages)).content)

    google = UniversalLLMAPIAdapter(
        organization="google",
        model="gemini-2.5-flash",
        api_key=os.environ["GOOGLE_API_KEY"],
    )
    print((await google.achat(messages=messages)).content)


asyncio.run(main())
```

### Multi-turn conversation

Use typed messages to preserve application-side conversation context, then
await the same request shape:

```python
import asyncio
import os

from llm_api_adapter.models.messages.chat_message import AIMessage, Prompt, UserMessage
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter


async def main():
    messages = [
        Prompt("You are a friendly assistant who explains complex concepts in simple terms."),
        UserMessage("Hi! Can you explain how artificial intelligence works?"),
        AIMessage(
            "Sure! Artificial intelligence (AI) is a system that can perform "
            "tasks requiring human-like intelligence, such as recognizing images "
            "or understanding language."
        ),
        UserMessage("How does AI learn?"),
    ]
    adapter = UniversalLLMAPIAdapter(
        organization="openai",
        model="gpt-5",
        api_key=os.environ["OPENAI_API_KEY"],
    )
    response = await adapter.achat(
        messages=messages,
        max_tokens=256,
        temperature=1.0,
        top_p=1.0,
    )
    print(response.content)


asyncio.run(main())
```

## Async streaming

```python
import asyncio
import os

from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter


async def main():
    adapter = UniversalLLMAPIAdapter(
        organization="anthropic",
        model="claude-sonnet-4-5",
        api_key=os.environ["ANTHROPIC_API_KEY"],
    )

    async def on_chunk(chunk):
        print(f"chunk {chunk.index}: {chunk.text!r}")

    async def on_done(response):
        print(f"\nusage: {response.usage}")

    async for delta in adapter.astream_chat(
        messages=[{"role": "user", "content": "Explain SSE in one sentence."}],
        max_tokens=80,
        buffer_chars=80,
        on_chunk=on_chunk,
        on_done=on_done,
    ):
        print(delta, end="", flush=True)


asyncio.run(main())
```

`astream_chat()` always yields normalized visible text as `str`. It uses a
separate `httpx.AsyncClient` transport, while preserving the same
provider-neutral response and callback contract as `stream_chat()`.

- `buffer_chars` accepts `None` (the default) or a positive integer. Buffered
  chunks never exceed the configured size; remaining text is emitted during
  normal completion.
- `on_chunk(chunk)` receives a `StreamChunk` with `text`, monotonic `index`,
  local `elapsed_s` / `delta_s`, and optional `usage` /
  `output_tokens_delta` fields.
- `on_delta(text)` is called for every yielded visible text chunk. The order is
  always `on_chunk` → `on_delta` → `yield`.
- `on_tool_call(tool_call)` receives completed, normalized `ToolCall` objects
  after the provider stream finishes.
- `on_done(response)` receives the finalized `ChatResponse`, including usage,
  pricing, parsed structured output, and tool calls, after the final buffer
  flush.
- Token metadata is optional and comes only from provider usage payloads. When
  a provider reports cumulative output usage, `output_tokens_delta` is the
  local increment; no token estimation is performed.

Async callbacks may be regular functions or `async def` functions. For every
visible chunk, callbacks are processed serially in this order:
`on_chunk` → `on_delta` → `yield`. Reasoning callbacks follow the same awaited
ordering and reasoning text is never yielded as visible output.

Buffering is pull-based and has no background worker or time-based flush. If a
stream fails or a caller closes it early, pending text is not emitted as a
successful final chunk and `on_done` is not called.

Provider event references: [OpenAI Responses streaming](https://platform.openai.com/docs/api-reference/responses-streaming), [Anthropic streaming](https://platform.claude.com/docs/en/build-with-claude/streaming), and [Google `streamGenerateContent`](https://ai.google.dev/api/generate-content).

## Tool calling

`achat()` and `astream_chat()` accept the same `tools`, `tool_choice`, and
`parallel_tool_calls` arguments as the synchronous API. `achat()` exposes
normalized calls in `ChatResponse.tool_calls`. During async streaming,
completed calls are delivered through `on_tool_call`, which may be a regular
or async function.

The adapter does not execute tools. Your application executes each call,
appends the resulting `ToolMessage`, and makes the follow-up request. This
complete async example handles every tool call in a response:

```python
import asyncio
import json
import os
from typing import Any, Dict

from llm_api_adapter.models.messages.chat_message import (
    AIMessage,
    Prompt,
    ToolMessage,
    UserMessage,
)
from llm_api_adapter.models.tools import ToolSpec
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter


tools = [
    ToolSpec(
        name="get_weather",
        description="Get current weather for a city",
        json_schema={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    )
]


def run_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    if name == "get_weather":
        return {"city": args["city"], "temperature": 22, "unit": "C"}
    raise ValueError(f"Unknown tool: {name}")


async def main():
    adapter = UniversalLLMAPIAdapter(
        organization="openai",
        model="gpt-5.2",
        api_key=os.environ["OPENAI_API_KEY"],
    )
    messages = [
        Prompt("If the user asks about weather, call get_weather."),
        UserMessage("What's the weather in Tel Aviv today?"),
    ]

    first = await adapter.achat(
        messages=messages,
        tools=tools,
        tool_choice="auto",
        max_tokens=1000,
    )

    if not first.tool_calls:
        print(first.content)
        return

    messages.append(AIMessage(content="", tool_calls=first.tool_calls))
    for tool_call in first.tool_calls:
        result = run_tool(tool_call.name, tool_call.arguments)
        messages.append(
            ToolMessage(
                tool_call_id=tool_call.call_id,
                content=json.dumps(result),
            )
        )

    final = await adapter.achat(
        messages=messages,
        previous_response=first,
        max_tokens=1000,
    )
    print(final.content)


asyncio.run(main())
```

For OpenAI Responses API models, `previous_response=first` preserves the
provider-side conversation state. Anthropic and Google accept the parameter
but carry context entirely through `messages`.

## Structured output

`achat()` accepts both `response_model` and `json_schema`. Its response has the
same `parsed_model` and `parsed_json` fields as the synchronous API. With
`astream_chat()`, inspect those fields on the final `ChatResponse` passed to
`on_done`.

### Pydantic model

```python
import asyncio
import os

from pydantic import BaseModel

from llm_api_adapter.models.messages.chat_message import Prompt, UserMessage
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter


class Person(BaseModel):
    name: str
    age: int


async def main():
    adapter = UniversalLLMAPIAdapter(
        organization="openai",
        model="gpt-5",
        api_key=os.environ["OPENAI_API_KEY"],
    )
    response = await adapter.achat(
        messages=[
            Prompt("Extract structured data from the user's message."),
            UserMessage("My name is Alice and I'm 30 years old."),
        ],
        response_model=Person,
        max_tokens=200,
    )
    print(response.parsed_model)  # Person(name='Alice', age=30)
    print(response.parsed_json)   # {"name": "Alice", "age": 30}


asyncio.run(main())
```

Pydantic is optional: install it separately with `pip install pydantic`.

### Raw JSON Schema

```python
import asyncio
import os

from llm_api_adapter.models.messages.chat_message import Prompt, UserMessage
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter


schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
    },
    "required": ["name", "age"],
}


async def main():
    adapter = UniversalLLMAPIAdapter(
        organization="openai",
        model="gpt-5",
        api_key=os.environ["OPENAI_API_KEY"],
    )
    response = await adapter.achat(
        messages=[
            Prompt("Extract structured data from the user's message."),
            UserMessage("My name is Alice and I'm 30 years old."),
        ],
        json_schema=schema,
        max_tokens=200,
    )
    print(response.content)      # '{"name": "Alice", "age": 30}'
    print(response.parsed_json)  # {"name": "Alice", "age": 30}


asyncio.run(main())
```

`response_model`, `json_schema`, and `tools` cannot be combined in one
request. Invalid schemas or non-JSON structured responses raise
`JSONSchemaError`.

## Vision input

Use `ImagePart` with `UserMessage.files`, then await `achat()` exactly as for a
text-only request. URL images and raw bytes work with all supported providers.

```python
import asyncio
import os

from llm_api_adapter.models.messages.chat_message import UserMessage
from llm_api_adapter.models.messages.file_parts import ImagePart
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter


async def main():
    adapter = UniversalLLMAPIAdapter(
        organization="openai",
        model="gpt-5",
        api_key=os.environ["OPENAI_API_KEY"],
    )
    message = UserMessage(
        "What is in this image?",
        files=[ImagePart(url="https://example.com/photo.jpg")],
    )
    response = await adapter.achat(messages=[message], max_tokens=200)
    print(response.content)


asyncio.run(main())
```

```python
import asyncio
import os

from llm_api_adapter.models.messages.chat_message import UserMessage
from llm_api_adapter.models.messages.file_parts import ImagePart
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter


async def main():
    with open("photo.png", "rb") as image_file:
        image_bytes = image_file.read()

    adapter = UniversalLLMAPIAdapter(
        organization="openai",
        model="gpt-5",
        api_key=os.environ["OPENAI_API_KEY"],
    )
    message = UserMessage(
        "Describe this image.",
        files=[ImagePart(data=image_bytes, media_type="image/png")],
    )
    response = await adapter.achat(messages=[message], max_tokens=200)
    print(response.content)


asyncio.run(main())
```

Multiple images use the same `files` list:

```python
message = UserMessage(
    "Compare these two images.",
    files=[
        ImagePart(url="https://example.com/before.jpg"),
        ImagePart(url="https://example.com/after.jpg"),
    ],
)
```

`ImagePart` accepts MIME types that start with `image/`; for URLs without a
file extension, pass `media_type` explicitly. Existing OpenAI-style image
dictionaries are also accepted:

```python
import asyncio
import os

from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter


async def main():
    adapter = UniversalLLMAPIAdapter(
        organization="openai",
        model="gpt-5",
        api_key=os.environ["OPENAI_API_KEY"],
    )
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "What is this?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
        ],
    }]
    response = await adapter.achat(messages=messages, max_tokens=200)
    print(response.content)


asyncio.run(main())
```

## Document input

Use `DocumentPart` with `UserMessage.files` for PDF URLs or bytes:

```python
import asyncio
import os

from llm_api_adapter.models.messages.chat_message import UserMessage
from llm_api_adapter.models.messages.file_parts import DocumentPart
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter


async def main():
    adapter = UniversalLLMAPIAdapter(
        organization="openai",
        model="gpt-5",
        api_key=os.environ["OPENAI_API_KEY"],
    )
    message = UserMessage(
        "Summarize this document in one sentence.",
        files=[DocumentPart(url="https://example.com/report.pdf")],
    )
    response = await adapter.achat(messages=[message], max_tokens=200)
    print(response.content)


asyncio.run(main())
```

```python
import asyncio
import os

from llm_api_adapter.models.messages.chat_message import UserMessage
from llm_api_adapter.models.messages.file_parts import DocumentPart
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter


async def main():
    with open("report.pdf", "rb") as document_file:
        pdf_bytes = document_file.read()

    adapter = UniversalLLMAPIAdapter(
        organization="openai",
        model="gpt-5",
        api_key=os.environ["OPENAI_API_KEY"],
    )
    message = UserMessage(
        "Summarize this document in one sentence.",
        files=[DocumentPart(data=pdf_bytes, media_type="application/pdf")],
    )
    response = await adapter.achat(messages=[message], max_tokens=200)
    print(response.content)


asyncio.run(main())
```

The adapter encodes bytes locally and does not download file URLs. Anthropic,
Google, and OpenAI Responses API models accept PDF URLs; OpenAI Chat
Completions models below `gpt-5` require PDF bytes instead.

## Usage and pricing

`achat()` returns the same `usage`, `currency`, and cost fields as `chat()`.
For `astream_chat()`, the finalized response passed to `on_done` has those
fields, while `StreamChunk.usage` is available when a provider reports usage
during streaming.

### Tiered standard-rate estimates

Async pricing has the same semantics as `chat()`: when the provider reports
`usage.input_tokens`, one tier is selected using its inclusive
`up_to_prompt_tokens` boundary, and that tier's input and output rates price
the entire request. The final tier is unbounded.

`context_window_tokens` is the combined input/output capacity,
`max_output_tokens` is the generated-output capacity, and a tier boundary is
only a price-selection threshold. They are not interchangeable. If the
provider omits usage, the adapter does not estimate it locally and leaves
`usage`, `currency`, and `cost_*` as `None`.

These values are standard text-rate estimates, not provider invoices. They
exclude cached input, cache write/storage, batch, flex, priority,
modality-specific, provider-hosted tool, and negotiated-volume charges.

```python
import asyncio
import os

from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter


async def main():
    google = UniversalLLMAPIAdapter(
        organization="google",
        model="gemini-2.5-flash",
        api_key=os.environ["GOOGLE_API_KEY"],
    )
    response = await google.achat(
        messages=[{"role": "user", "content": "Explain token usage briefly."}],
    )

    if response.usage is None:
        print("Provider did not report usage; cost is unavailable.")
    else:
        print(response.usage.input_tokens, "tokens", f"({response.cost_input} {response.currency})")
        print(response.usage.output_tokens, "tokens", f"({response.cost_output} {response.currency})")
        print(response.usage.total_tokens, "tokens", f"({response.cost_total} {response.currency})")


asyncio.run(main())
```

Override registry pricing or currency on an adapter instance before awaiting
the request. A rate override replaces that rate in every pricing tier for the
selected model:

```python
import asyncio
import os

from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter


async def main():
    google = UniversalLLMAPIAdapter(
        organization="google",
        model="gemini-2.5-flash",
        api_key=os.environ["GOOGLE_API_KEY"],
    )
    google.pricing.set_in_per_1m(1.5)
    google.pricing.set_out_per_1m(3)
    google.pricing.set_currency("EUR")

    response = await google.achat(
        messages=[{"role": "user", "content": "Explain token usage briefly."}],
    )
    print(response.content)
    if response.usage is not None:
        print(response.usage.total_tokens, "tokens", f"({response.cost_total} {response.currency})")


asyncio.run(main())
```

## Lifecycle and errors

Async requests do not add automatic retries, provider fallback, or idempotency
behavior. If a task is cancelled, the HTTP response and client are closed;
pending text is not flushed and `on_done` is not called. The same cleanup
applies when an async stream is closed before normal completion.

Async HTTPX requests use the same `LLMAPIError` hierarchy as synchronous
requests. Authentication, rate-limit, client, server, and timeout failures are
normalized to the same provider-neutral exceptions. See [error handling in the
main README](README.md#handling-errors) for the exception reference.

## Timeouts

`timeout_s` applies to `achat()` and `astream_chat()` as well as the
synchronous methods. It limits the full request lifecycle; if it expires, the
adapter raises `LLMAPITimeoutError`.

```python
import asyncio
import os

from llm_api_adapter.errors import LLMAPITimeoutError
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter


async def main():
    adapter = UniversalLLMAPIAdapter(
        organization="openai",
        model="gpt-5.2",
        api_key=os.environ["OPENAI_API_KEY"],
    )
    try:
        response = await adapter.achat(
            messages=[{"role": "user", "content": "Explain timeouts briefly."}],
            timeout_s=2.5,
        )
        print(response.content)
    except LLMAPITimeoutError:
        # Retry, fall back, or abort according to your application policy.
        print("LLM request timed out")


asyncio.run(main())
```

## Reasoning level

`reasoning_level` is provider-neutral and applies to `achat()` and
`astream_chat()`. It accepts an explicit integer budget or one of the canonical
levels: `"none"`, `"minimal"`, `"low"`, `"medium"`, `"high"`, and
`"very_high"`. Verified model capabilities resolve the value to the native
provider setting.

For categorical models, a native value listed for that model is preserved;
other canonical strings are projected upward through the ordered native values.
An integer is normalized as a 0–100% fraction of the model context window and
rounded upward to a native value. For numeric-budget models, canonical values
from `"minimal"` through `"very_high"` are interpolated across the documented
budget range. Integer budgets below the minimum fall back with a warning;
budgets above the documented maximum are forwarded to the provider unchanged.
`"none"` disables reasoning only where the model supports zero; otherwise it
uses the minimum with a warning. Omitting `reasoning_level` preserves the
provider's existing default request behavior.

```python
import asyncio
import os

from llm_api_adapter.models.messages.chat_message import UserMessage
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter


async def main():
    adapter = UniversalLLMAPIAdapter(
        organization="openai",
        model="gpt-5",
        api_key=os.environ["OPENAI_API_KEY"],
    )

    named_level = await adapter.achat(
        messages=[UserMessage("Explain this")],
        reasoning_level="medium",
    )
    print(named_level.content)

    explicit_level = await adapter.achat(
        messages=[UserMessage("Solve this step-by-step")],
        reasoning_level=2048,
    )
    print(explicit_level.content)

    disabled_reasoning = await adapter.achat(
        messages=[UserMessage("Simple answer, no reasoning")],
        reasoning_level="none",
    )
    print(disabled_reasoning.content)


asyncio.run(main())
```

## Reasoning observability

Reasoning observability is opt-in and additive. Set `capture_reasoning=True` to
retain provider-emitted reasoning summaries or readable thinking content in
`ChatResponse.reasoning_events`.

With `achat()`, the events are available on the returned `ChatResponse`. With
`astream_chat()`, they are delivered through `on_reasoning` and included in the
final response passed to `on_done`.

The stream still yields only visible text. Reasoning events are never sent to
`on_delta` and never appear in the iterator. When supplied,
`on_reasoning(event)` is called as each event is normalized;
`on_done(response)` receives the complete `response.reasoning_events` list.

```python
import asyncio
import os

from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter


async def main():
    adapter = UniversalLLMAPIAdapter(
        organization="openai",
        model="gpt-5",
        api_key=os.environ["OPENAI_API_KEY"],
    )

    async def observe_reasoning(event):
        # Application code decides what to log, redact, retain, or display.
        print(f"reasoning[{event.index}]: {event.text}")

    async def on_done(response):
        print(f"captured {len(response.reasoning_events)} reasoning events")

    async for text in adapter.astream_chat(
        messages=[{"role": "user", "content": "Explain SSE briefly."}],
        capture_reasoning=True,
        on_reasoning=observe_reasoning,
        on_done=on_done,
    ):
        print(text, end="", flush=True)


asyncio.run(main())
```

`ReasoningEvent` contains `text`, `kind`, `index`, `elapsed_s`, and `delta_s`.
Availability and content depend on the provider and model; an empty list is a
valid result. The library does not automatically log, display, redact, or send
reasoning to telemetry, and it does not promise access to a model's private
chain of thought. Applications should apply their own redaction and retention
policy.

## Logging

Async requests use the same standard-library logging setup as synchronous
requests. The library configures no handlers; its loggers are under
`llm_api_adapter.*`. API keys and request bodies are excluded from library logs.

```python
import logging

logging.basicConfig(level=logging.INFO)  # or DEBUG
logging.getLogger("llm_api_adapter").setLevel(logging.DEBUG)
```

```python
import logging

handler = logging.FileHandler("llm_api_adapter.log")
handler.setFormatter(logging.Formatter(
    "%(asctime)s %(levelname)s %(name)s %(message)s"
))
root = logging.getLogger()
root.setLevel(logging.INFO)
root.addHandler(handler)
```

Use `LoggerAdapter` for an application-level request ID:

```python
import logging

logger = logging.LoggerAdapter(
    logging.getLogger("llm_api_adapter"),
    {"request_id": "req-123"},
)
logger.info("starting async request")
```

Add `%(request_id)s` to the formatter to display that value. To reduce noise,
adjust the library and HTTPX logger levels:

```python
import logging

logging.getLogger("llm_api_adapter").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
```
