# LLM API Adapter SDK for Python
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/llm-api-adapter?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/llm-api-adapter)
[![Tests](https://github.com/Inozem/llm_api_adapter/actions/workflows/ci-main.yml/badge.svg)](https://github.com/Inozem/llm_api_adapter/actions/workflows/ci-main.yml)
[![Python versions](https://img.shields.io/pypi/pyversions/llm-api-adapter.svg)](https://pypi.org/project/llm-api-adapter/)
[![codecov](https://codecov.io/github/Inozem/llm_api_adapter/branch/dev/graph/badge.svg?token=T83ZPH1F7Z)](https://codecov.io/github/Inozem/llm_api_adapter)

## Overview

**llm-api-adapter** is a minimal, typed adapter for five organizations: OpenAI, Anthropic, Google, Mistral, and xAI. It provides one provider-neutral contract for messages, tools, structured output, multimodal input, errors, usage, cost, and streaming — with one runtime dependency and no provider SDKs or orchestration framework. Switching organizations means changing two arguments.

**Note:** Mistral and xAI are installed separately with their optional extras.

Supports Python 3.10–3.14.

## Contents

- [Overview](#overview)
- [Why this library?](#why-this-library)
- [Features](#features)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Model-specific request compatibility](#model-specific-request-compatibility)
- [Streaming](#streaming)
- [Async API](#async-api)
- [Optional HTTPX Sync Transport](#optional-httpx-sync-transport)
- [Handling Errors](#handling-errors)
- [Configuration and Management](#configuration-and-management)
- [Example Use Case](#example-use-case)
- [Timeout Support](#timeout-support)
- [Reasoning Support](#reasoning-support)
- [Tool / Function Calling](#tool--function-calling)
- [Structured Output](#structured-output)
- [Vision Input](#vision-input)
- [Document Input](#document-input)
- [Token Usage and Pricing](#token-usage-and-pricing)
- [Logging](#logging)
- [Adoption](#adoption)
- [Related project](#related-project)
- [Development & Testing](#development--testing)
- [License](#license)

## Why this library?

|                           | llm-api-adapter | LiteLLM | LangChain | Provider SDK |
|---------------------------|-----------------|---------|-----------|--------------|
| Runtime dependencies      | 1 (`requests`) | broader gateway/SDK | broader framework | provider-specific |
| Provider-neutral messages and response | ✓ | ✓ | ✓ | ✗ |
| Unified tool calls        | ✓ | ✓ | ✓ | provider-specific |
| Unified structured output | ✓ | ✓ | ✓ | provider-specific |
| Unified image/PDF input   | ✓ | partial | ✓ | provider-specific |
| Unified reasoning control | ✓* | ✓ | partial | ✗ |
| Built-in per-response cost| ✓ | ✓ | via callbacks | ✗ |
| Unified error hierarchy   | ✓ | OpenAI-compatible | framework-specific | ✗ |
| Sync streaming            | ✓ text-first | ✓ | ✓ | ✓ |
| Async API                 | ✓ optional | ✓ | ✓ | ✓ |
| Number of organizations   | 5 | 100+ | 50+ | 1 |

* `reasoning_level` is one application-level parameter, but the available levels, native mapping, and emitted reasoning content remain model/provider-dependent.

This table is a positioning snapshot. Provider capabilities change independently, so the adapter promises a stable application-level contract rather than identical provider internals.

**Small footprint.** The core has one runtime dependency: `requests`. It does not require OpenAI, Anthropic, or Google SDKs, an agent framework, a gateway, or a proxy. Pydantic is optional and is only needed for `response_model` validation.

**One provider-neutral contract.** The same typed messages, `ToolSpec`, `ChatResponse`, `ImagePart`, and `DocumentPart` are converted to and from each provider's native wire format. Application code does not need provider-specific message, tool, or response parsing.

**Model-aware reasoning control.** One `reasoning_level` parameter is accepted across supported providers without provider-specific kwargs. Verified model capabilities resolve the canonical level to a native effort or budget setting; exact availability and token semantics remain model-dependent.

**Cost accounting, built-in.** Responses carry normalized usage and cost fields in the configured currency. When a provider reports an exact total that cannot be split safely, `cost_total` is set while `cost_input` and `cost_output` remain unset.

**Predictable errors.** One explicit exception hierarchy and provider-error mapping across supported organizations. `LLMAPIRateLimitError` means the same application-level condition whether you called OpenAI, Anthropic, Google, Mistral, or xAI.

### Use this when

- You call LLMs directly — no chains, no agents, no orchestration
- You want a provider-neutral contract without adopting a framework or gateway
- You need the same messages, tools, structured output, multimodal input, errors, and response fields across providers
- You need unified reasoning control or per-request cost tracking with minimal dependencies

### Use something else when

- **LiteLLM** — you need providers beyond OpenAI, Anthropic/Claude, and Google, or a gateway/router layer with retries, fallbacks, load balancing, and observability
- **AISuite** — you prefer an OpenAI-compatible client across more providers, or built-in agents, MCP, and automatic tool-execution abstractions
- **LangChain** — you need chains, memory, RAG, agents, or a larger orchestration framework
- **Provider SDK directly** — you use one provider and need an API or feature outside this SDK's provider-neutral contract

## Features

- **Synchronous Streaming**: Iterate normalized text with `stream_chat()` across supported organizations. Optionally coalesce it into bounded chunks and observe per-chunk metadata without changing the yielded `str` contract.
- **Optional HTTPX Sync Transport**: Keep `requests` as the default synchronous transport or opt into HTTPX with `transport="httpx"`. See the [HTTPX sync pilot guide](HTTPX_SYNC_PILOT.md).
- **Asynchronous API**: Use `achat()` and `astream_chat()` with the optional `[async]` installation for non-blocking HTTPX requests and awaitable callbacks. See the [Async API guide](ASYNC_API.md).
- **Reasoning Observability**: Opt in to provider-emitted reasoning summaries through `capture_reasoning=True`, `ReasoningEvent`, and the `on_reasoning` callback without mixing reasoning into visible text.
- **Provider-Neutral Messages and Responses**: Use the same typed messages, `ChatResponse`, usage, pricing, parsed output, and tool-call fields regardless of the provider.
- **Vision Input**: Send images alongside text via `ImagePart` — URL or raw bytes, all providers handled automatically.
- **PDF Documents**: Send PDF URLs or bytes via `DocumentPart`; provider-specific file/document payloads are generated automatically.
- **Tool / Function Calling**: Provider-agnostic tool definitions and normalized tool calls in `ChatResponse.tool_calls`.
- **Portable Structured Output**: Pass a documented portable JSON Schema to `chat()` and get a parsed object in `ChatResponse.parsed_json` across all supported organizations.
- **Pydantic Integration**: Pass a portable Pydantic model as `response_model` and get a typed instance back in `ChatResponse.parsed_model` — no manual schema writing required.
- **Request Timeouts**: Per-request timeout control via `timeout_s`; raises `LLMAPITimeoutError` on expiry.
- **Flexible Configuration**: `temperature`, `max_tokens`, `top_p`, and other parameters are translated to a verified provider payload; known unsupported fields are omitted predictably.
- **Pricing Registry**: Model prices stored in a bundled JSON registry with per-model input/output rates; overridable per instance.

## Installation

To install the SDK, you can use pip:

```bash
pip install llm-api-adapter
```

To use Mistral, install its optional organization package:

```bash
pip install "llm-api-adapter[mistral]"
```

To use xAI, install its optional organization package:

```bash
pip install "llm-api-adapter[xai]"
```

The [Mistral package README](packages/organizations/mistral/README.md) and
[xAI package README](packages/organizations/xai/README.md) list their
supported models and organization-specific behaviour. Direct installation of
`llm-api-adapter-mistral` or `llm-api-adapter-xai` remains supported.

For non-blocking requests, install the optional `[async]` extra and follow the
[Async API guide](ASYNC_API.md).

To opt into HTTPX for synchronous `chat()` and `stream_chat()` calls, install
the optional `[httpx]` extra and pass `transport="httpx"`. The default remains
`requests`; see the [HTTPX sync pilot guide](HTTPX_SYNC_PILOT.md).

**Note:** You need an API key from each LLM provider you use, including
Mistral or xAI when their optional packages are installed. Refer to the
provider's documentation for API-key instructions.


## Getting Started

### Importing and Setting Up the Adapter

To start using the adapter, you need to import the necessary components:

```python
from llm_api_adapter.models.messages.chat_message import (
    AIMessage, Prompt, UserMessage
)
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter
```

### Sending a Simple Request

The SDK supports three types of messages for interacting with the LLM:

- **Prompt**: Use `Prompt` to set the context or initial prompt for the model.
- **UserMessage**: Use `UserMessage` to send messages from the user during a conversation.
- **AIMessage**: Use `AIMessage` to simulate responses from the assistant during a conversation.

Here is an example of how to send a simple request to the adapter:

```python
messages = [
    UserMessage("Hi! Can you explain how artificial intelligence works?")
]

adapter = UniversalLLMAPIAdapter(
    organization="openai",
    model="gpt-5",
    api_key=openai_api_key
)

response = adapter.chat(
    messages=messages,
    max_tokens=max_tokens,
    temperature=temperature,
    top_p=top_p
)
print(response.content)
```

### Parameters

- **max\_tokens**: The maximum number of tokens to generate in the response. This limits the length of the output.

- **temperature**: Controls the randomness of the response. Higher values (e.g., 0.8) make the output more random, while lower values (e.g., 0.2) make it more focused and deterministic. Default value: `1.0` (range: 0 to 2).

- **top\_p**: Limits the response to a certain cumulative probability. This is used to create more focused and coherent responses by considering only the highest probability options. Default value: `1.0` (range: 0 to 1).

### Model-specific request compatibility

The public `chat()`, `achat()`, `stream_chat()`, and `astream_chat()` methods
keep the same arguments across providers. For a verified model, the adapter
applies its registered payload compatibility rules before making the request;
sync and async calls use the same rules.

If a rule omits a parameter, its documented default is removed silently. A
different supplied value is also removed, but emits one `UserWarning` and one
`WARNING` log record per ignored parameter. Treat that warning as a prompt to
remove the setting or select a model that supports it.

Rules match the **exact** model names in the bundled registry, with two narrow
snapshot exceptions for direct provider APIs: Anthropic
`claude-...-YYYYMMDD` and OpenAI `gpt-...-YYYY-MM-DD` IDs inherit a registered
unsuffixed base model's metadata when the date is valid. The requested snapshot
ID is still sent to the provider. The adapter does not infer behavior from a
name prefix; Google aliases and preview IDs, fine-tuned IDs, and any snapshot
whose base model is unregistered receive no compatibility transformation. An
unknown OpenAI model uses Chat Completions rather than the Responses API. Use a
listed model or request a verified registry addition when a provider adds a new
alias or model.

For `gemini-3.7-flash`, `gemini-3.6-flash`, and
`gemini-3.5-flash-lite`, Google does not support sampling controls. The adapter
omits `temperature` and `top_p` from these model requests; non-default values
produce the compatibility warning described above.

### Alternative Message Format

In addition to the built-in message classes, the SDK also supports the standard OpenAI-style message format for quick adoption and compatibility:

```python
messages = [
    {"role": "system", "content": "You are a friendly assistant who answers only yes or no."},
    {"role": "user", "content": "Do you know how AI learns?"},
    {"role": "assistant", "content": "Yes."},
    {"role": "user", "content": "Can you explain it in one sentence?"}
]

response = adapter.chat(messages=messages, max_tokens=50)
print(response.content)
```
> **Note**  
> The adapter automatically normalizes message input — you can mix custom message classes and OpenAI-style dicts in one list.

## Streaming

`stream_chat()` is the synchronous streaming counterpart to `chat()`. It always yields normalized visible text as `str`, independently of the provider's native streaming event format.

By default, `buffer_chars=None` preserves provider text-delta behavior. Set a positive `buffer_chars` value when a consumer prefers bounded, coalesced text; `on_chunk` then receives a `StreamChunk` with the same text and observability metadata.

```python
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter

adapter = UniversalLLMAPIAdapter(
    organization="openai",
    model="gpt-5.6-sol",
    api_key="...",
)

chunk_metadata = []
for text in adapter.stream_chat(
    messages=[{"role": "user", "content": "Explain SSE in one sentence."}],
    buffer_chars=80,
    on_chunk=chunk_metadata.append,
    on_tool_call=lambda call: print(call.name, call.arguments),
    on_done=lambda response: print(response.usage),
):
    print(text, end="", flush=True)  # `text` is still a str.

for chunk in chunk_metadata:
    print(chunk.index, chunk.elapsed_s, chunk.usage, chunk.output_tokens_delta)
```

- `buffer_chars` accepts `None` (the default) or a positive integer. Buffered chunks never exceed the configured size; any remaining text is emitted during normal completion.
- `on_chunk(chunk)` receives a `StreamChunk` with `text`, monotonic `index`, local `elapsed_s` / `delta_s`, and optional `usage` / `output_tokens_delta` fields.
- `on_delta(text)` is called for every yielded visible text chunk. The order is always `on_chunk` → `on_delta` → `yield`.
- `on_tool_call(tool_call)` receives only completed, normalized `ToolCall` objects after the provider stream finishes.
- `on_done(response)` receives the finalized `ChatResponse`, including usage, pricing, parsed structured output, and any tool calls, after the final buffer flush.
- Token metadata is optional and comes only from provider usage payloads. When a provider reports cumulative output usage, `output_tokens_delta` is the local increment; no token estimation is performed.

Buffering is pull-based and has no background worker or time-based flush. If the stream fails or a caller closes the iterator early, pending text is not emitted as a successful final chunk and `on_done` is not called.

`stream_chat()` uses the synchronous `requests` transport by default. To opt
into HTTPX for both `chat()` and `stream_chat()`, see the
[HTTPX sync pilot guide](HTTPX_SYNC_PILOT.md). For the matching
`astream_chat()` contract and its `httpx.AsyncClient` transport, see the
[Async API guide](ASYNC_API.md#async-streaming).

Provider event references: [OpenAI Responses streaming](https://platform.openai.com/docs/api-reference/responses-streaming), [Anthropic streaming](https://platform.claude.com/docs/en/build-with-claude/streaming), and [Google `streamGenerateContent`](https://ai.google.dev/api/generate-content).

## Async API

The optional asynchronous client, installation command, `achat()` and
`astream_chat()` examples, callback ordering, cancellation behavior, and
reasoning observability are documented in the [Async API guide](ASYNC_API.md).

## Optional HTTPX Sync Transport

The default synchronous transport is `requests` and requires no additional
dependency. HTTPX is available as an opt-in pilot for synchronous `chat()` and
`stream_chat()` calls:

```bash
pip install "llm-api-adapter[httpx]"
```

```python
adapter = UniversalLLMAPIAdapter(
    organization="openai",
    model="gpt-5",
    api_key="...",
    transport="httpx",
)
response = adapter.chat([UserMessage("Say hello")])
```

See the [HTTPX sync pilot guide](HTTPX_SYNC_PILOT.md) for compatibility,
limitations, and E2E verification details.

## Handling Errors

### Common Errors

The SDK provides a set of standardized errors for easier debugging and integration:

### API Errors

- **LLMAPIError**: Base class for all API-related errors. This error is also used for any unexpected LLM API errors.

- **LLMAPIAuthorizationError**: Raised when authentication or authorization fails.

- **LLMAPIRateLimitError**: Raised when rate limits are exceeded.

- **LLMAPITokenLimitError**: Raised when token limits are exceeded.

- **LLMAPIClientError**: Raised when the client makes an invalid request.

- **LLMAPIServerError**: Raised when the server encounters an error.

- **LLMAPITimeoutError**: Raised when a request times out.

- **LLMAPIUsageLimitError**: Raised when usage limits are exceeded.

- **InvalidToolSchemaError**: Raised when a provided tool schema is invalid.

- **InvalidToolArgumentsError**: Raised when tool arguments cannot be parsed or validated.

- **ToolChoiceError**: Raised when `tool_choice` is invalid or references an unknown tool.

- **JSONSchemaError**: Raised for incompatible structured-output arguments, a non-portable or non-recursive schema, an invalid Pydantic response model, invalid JSON, or failed Pydantic response validation.

### Config Errors

- **LLMConfigError**: Raised when the request configuration is invalid or incompatible.

- **LLMReasoningLevelError**: Raised only for Anthropic models when max_tokens is less than reasoning_level.

### Provider Error Mapping

| Exception | OpenAI | Anthropic | Google |
|---|---|---|---|
| `LLMAPIAuthorizationError` | HTTP 401; `InvalidAuthenticationError`, `AuthenticationError` | HTTP 401; `AuthenticationError`, `PermissionError` | HTTP 401/403; `PERMISSION_DENIED` |
| `LLMAPIRateLimitError` | HTTP 429; `RateLimitError` | HTTP 429; `RateLimitError` | HTTP 429; `RESOURCE_EXHAUSTED` |
| `LLMAPITokenLimitError` | `MaxTokensExceededError`, `TokenLimitError` | — | — |
| `LLMAPIClientError` | HTTP 4xx; `InvalidRequestError`, `BadRequestError` | HTTP 4xx; `InvalidRequestError`, `RequestTooLargeError`, `NotFoundError` | HTTP 4xx; `INVALID_ARGUMENT`, `FAILED_PRECONDITION`, `NOT_FOUND` |
| `LLMAPIServerError` | HTTP 5xx; `InternalServerError`, `ServiceUnavailableError` | HTTP 5xx; `APIError`, `OverloadedError` | HTTP 5xx; `INTERNAL`, `UNAVAILABLE` |
| `LLMAPITimeoutError` | `requests.Timeout`, `httpx.TimeoutException`; `TimeoutError` | `requests.Timeout`, `httpx.TimeoutException` | `requests.Timeout`, `httpx.TimeoutException`; `DEADLINE_EXCEEDED` |
| `LLMAPIUsageLimitError` | `UsageLimitError`, `QuotaExceededError` | — | — |

> - `LLMAPITokenLimitError` and `LLMAPIUsageLimitError` are OpenAI-only; equivalent cases for Anthropic and Google fall into `LLMAPIClientError` (HTTP 4xx).
> - `LLMAPIClientError` is the default fallback for all unhandled HTTP 4xx responses.
> - `LLMAPIServerError` is the default fallback for all HTTP 5xx responses.
> - `InvalidToolSchemaError`, `InvalidToolArgumentsError`, `ToolChoiceError`, `JSONSchemaError` — client-side errors (validated before the request is sent); inherit from `LLMAPIClientError`.
> - `LLMConfigError`, `LLMReasoningLevelError` — configuration errors (parameter validation before the request); `LLMReasoningLevelError` is only raised for Anthropic models with `budget_tokens`.

## Configuration and Management

### Using Different Providers and Models

The SDK allows you to easily switch between LLM providers and specify the model you want to use. Currently supported providers are OpenAI, Anthropic, Google, Mistral, and xAI. Mistral and xAI require their corresponding optional extras.

- **OpenAI**: You can use models like `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-5.6-luna`, `gpt-5.5`, `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.4-nano`, `gpt-5.2`, `gpt-5.1`, `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`, `gpt-4o`, `gpt-4o-mini`.
- **Anthropic**: Available models include `claude-fable-5`, `claude-sonnet-5`, `claude-opus-4-8`, `claude-opus-4-7`, `claude-opus-4-6`, `claude-sonnet-4-6`, `claude-opus-4-5`, `claude-sonnet-4-5`, `claude-haiku-4-5`.
- **Google**: Models such as `gemini-3.7-flash`, `gemini-3.6-flash`, `gemini-3.5-flash`, `gemini-3.5-flash-lite`, `gemini-3.1-pro-preview`, `gemini-3.1-flash-lite`, `gemini-3-flash-preview`, `gemini-2.5-pro`, `gemini-2.5-flash`, and `gemini-2.5-flash-lite` can be used.
- **Mistral**: Install with `pip install "llm-api-adapter[mistral]"`. Available models are `mistral-small-2603`, `mistral-medium-3-5`, and `mistral-large-2512`; see the [Mistral package README](packages/organizations/mistral/README.md) for Mistral-specific behaviour.
- **xAI**: Install with `pip install "llm-api-adapter[xai]"`. Fixed model IDs are `grok-4.5` and `grok-4.6`; see the [xAI package README](packages/organizations/xai/README.md) for its capability matrix and data-handling notes.

Example:

```python
adapter = UniversalLLMAPIAdapter(
    organization="openai",
    model="gpt-5",
    api_key=openai_api_key
)
```

To switch to another provider, simply change the `organization` and `model` parameters.

### Switching Providers

Here is an example of how to switch between different LLM providers using the SDK:

**Note**: Each instance of `UniversalLLMAPIAdapter` is tied to a specific provider and model. You cannot change the `organization` parameter for an existing adapter object. To use a different provider, you must create a new instance.

```python
gpt = UniversalLLMAPIAdapter(
    organization="openai",
    model="gpt-5",
    api_key=openai_api_key
)
gpt_response = gpt.chat(messages=messages)
print(gpt_response.content)

claude = UniversalLLMAPIAdapter(
    organization="anthropic",
    model="claude-sonnet-4-5",
    api_key=anthropic_api_key
)
claude_response = claude.chat(messages=messages)
print(claude_response.content)

google = UniversalLLMAPIAdapter(
    organization="google",
    model="gemini-2.5-flash",
    api_key=google_api_key
)
google_response = google.chat(messages=messages)
print(google_response.content)
```

## Example Use Case

Here is a comprehensive example that showcases all possible message types and interactions:

```python
from llm_api_adapter.models.messages.chat_message import (
    AIMessage, Prompt, UserMessage
)                                               
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter

messages = [
    Prompt(
        "You are a friendly assistant who explains complex concepts "
        "in simple terms."
    ),
    UserMessage("Hi! Can you explain how artificial intelligence works?"),
    AIMessage(
        "Sure! Artificial intelligence (AI) is a system that can perform "
        "tasks requiring human-like intelligence, such as recognizing images "
        "or understanding language. It learns by analyzing large amounts of "
        "data, finding patterns, and making predictions."
    ),
    UserMessage("How does AI learn?"),
]

adapter = UniversalLLMAPIAdapter(
    organization="openai",
    model="gpt-5",
    api_key=openai_api_key
)

response = adapter.chat(
    messages=messages,
    max_tokens=256,
    temperature=1.0,
    top_p=1.0
)
print(response.content)
```

The `ChatResponse` object returned by `chat` includes:

1. **model**: The model that generated the response.
2. **response\_id**: Unique identifier for the response.
3. **timestamp**: Response generation time.
4. **usage**: Object containing `input_tokens`, `output_tokens`, and `total_tokens`.
5. **currency**: The currency used for cost calculation.
6. **cost\_input**: Cost of input tokens.
7. **cost\_output**: Cost of output tokens.
8. **cost\_total**: Total combined cost.
9. **content**: The generated text response.
10. **finish\_reason**: Provider-native reason why generation stopped (for example, `"stop"` or `"length"`).
11. **refusal**: Provider-normalized refusal detail, or `None` when the response was not a refusal.
12. **incomplete\_reason**: Provider-normalized incomplete-generation reason, or `None` when generation completed.
13. **parsed\_json**: Parsed JSON object for a valid completed structured result, otherwise `None`.
14. **parsed\_model**: Typed Pydantic instance for a valid completed `response_model` result, otherwise `None`.

## Timeout Support

The SDK supports per-request timeouts for all providers.

### timeout\_s parameter

`timeout_s` defines the maximum time (in seconds) the SDK will wait for an LLM response. It applies to `chat()`, `achat()`, `stream_chat()`, and `astream_chat()`.

If the timeout is exceeded, the request is aborted and `LLMAPITimeoutError` is raised.

### Example

```python
chat_params = {
    "messages": messages,
    "timeout_s": 2.5
}
gpt = UniversalLLMAPIAdapter(
    organization="openai",
    model="gpt-5.2",
    api_key=openai_api_key
)
response = gpt.chat(**chat_params)
```

### Notes

- Timeout is applied uniformly across all providers.
- The parameter is optional; if omitted, the provider default is used.
- Timeout affects the full request lifecycle (network + model execution).

### Handling timeout errors

Timeouts raise a dedicated exception that can be handled explicitly:

```python
from llm_api_adapter.errors import LLMAPITimeoutError

try:
    response = gpt.chat(**chat_params)
except LLMAPITimeoutError:
    # retry, fallback, or abort
    print("LLM request timed out")
```

## Reasoning Support

This section describes the provider-neutral `reasoning_level` parameter for `chat()`, `achat()`, `stream_chat()`, and `astream_chat()`. It gives application code one input surface; it does not claim that every model exposes the same levels or consumes the same number of reasoning tokens.

```python
response = adapter.chat(
    messages=[UserMessage("Solve this step-by-step")],
    reasoning_level=2048,
)
```

### Default behavior

When `reasoning_level` is omitted, the adapter bypasses model-aware resolution
and preserves the provider request behavior that existed before it. Pass a
level, including `"none"`, when the request must express reasoning intent.

### `reasoning_level` parameter

`reasoning_level` is optional and provider-agnostic. The adapters translate it to the provider's native effort or thinking-budget format.

Supported forms:

- **int** — explicit reasoning-token budget
- **str** — one of the canonical levels: `"none"`, `"minimal"`, `"low"`,
  `"medium"`, `"high"`, or `"very_high"`

For a verified categorical model, an exact provider value listed for that
model is preserved. This permits native values such as `"xhigh"` or `"max"`
where the model supports them, but those values are not portable. Other
canonical strings are projected upward through the model's ordered native
values; `"none"` becomes the model minimum with a warning if it cannot disable
reasoning. An integer is treated as a fraction of the model context window,
clamped to 0–100%, then rounded upward to an available categorical value.
When the native list has no `"none"`, its first value is the first positive
step: `"minimal"` and a numeric percentage at or below that step resolve to it
rather than skipping it.

For a verified numeric-budget model, canonical values from `"minimal"` through
`"very_high"` are evenly interpolated from that model's documented minimum to
maximum budget (with upward rounding). `"none"` resolves to zero when zero is
supported, otherwise to the minimum with a warning. An integer is a literal
budget: a value below the documented minimum falls back to that minimum with a
warning, while a value above the documented maximum is forwarded so the
provider can return its native validation error.

Named levels express relative intent. Do not treat `"medium"` or a numeric
value as an identical token budget across providers or models.

### Usage examples

```python
# Named level
response = adapter.chat(
    messages=[UserMessage("Explain this")],
    reasoning_level="medium",
)

# Explicit numeric level
response = adapter.chat(
    messages=[UserMessage("Solve this step-by-step")],
    reasoning_level=2048,
)

# Explicitly disable reasoning where the model supports it
response = adapter.chat(
    messages=[UserMessage("Simple answer, no reasoning")],
    reasoning_level="none",
)
```

### Provider independence

`reasoning_level` provides the same application-level entry point for all supported providers:

- same parameter name
- no provider-specific reasoning kwargs in application code
- provider-specific translation is handled inside the adapter

The available levels, native mapping, token budget, and whether reasoning can be disabled remain model/provider-dependent. This allows switching between OpenAI, Anthropic, and Google without rewriting the request shape, while keeping provider limitations explicit.

Provider references: [OpenAI reasoning](https://developers.openai.com/api/docs/guides/reasoning), [Anthropic extended thinking](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking), and [Google thinking](https://ai.google.dev/api/generate-content).

## Tool / Function Calling

The SDK provides a unified provider‑agnostic tool calling interface.

Tools are defined using `ToolSpec`, and tool calls are returned in normalized form through `ChatResponse.tool_calls`.

The adapter **does not execute tools**. Tool execution must be implemented by the caller.

### ToolSpec

```python
from llm_api_adapter.models.tools import ToolSpec

tool = ToolSpec(
    name="get_weather",
    description="Get current weather for a city",
    json_schema={
        "type": "object",
        "properties": {
            "city": {"type": "string"}
        },
        "required": ["city"],
        "additionalProperties": False
    }
)
```

### Tool parameters

All request methods support the same tool parameters:

- `tools`
- `tool_choice`
- `parallel_tool_calls`

Tool calls are returned as normalized `ToolCall` objects. The adapter does not
execute them, so the caller appends each resulting `ToolMessage` and makes the
follow-up request itself.

`parallel_tool_calls` is provider-dependent: Anthropic can enable or
disable parallel tool use, OpenAI supports it for Chat Completions, and Google
ignores it because Gemini has no equivalent option.

### Tool round‑trip example

```python
import json
from typing import Any, Dict

from llm_api_adapter.models.messages.chat_message import (
    Prompt,
    UserMessage,
    AIMessage,
    ToolMessage,
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


adapter = UniversalLLMAPIAdapter(
    organization="openai",
    model="gpt-5.2",
    api_key=openai_api_key,
)

messages = [
    Prompt("If the user asks about weather, call get_weather."),
    UserMessage("What's the weather in Tel Aviv today?")
]

first = adapter.chat(
    messages=messages,
    tools=tools,
    tool_choice="auto",
    max_tokens=1000,
)

if first.tool_calls:
    messages.append(AIMessage(content="", tool_calls=first.tool_calls))

    for tc in first.tool_calls:
        result = run_tool(tc.name, tc.arguments)
        messages.append(
            ToolMessage(
                tool_call_id=tc.call_id,
                content=json.dumps(result)
            )
        )

    final = adapter.chat(
        messages=messages,
        previous_response=first,
        max_tokens=1000
    )
    print(final.content)
```

For a complete async tool round-trip, see the [Async API guide](ASYNC_API.md#tool-calling).

### `previous_response` parameter

`previous_response` accepts the `ChatResponse` returned by an earlier `chat()`
or `achat()` call, and can also be passed to the streaming methods.

For OpenAI models that use the Responses API (o-series and newer GPT models), the adapter extracts `response_id` from the previous response and passes it to the API as `previous_response_id`. This enables stateful multi-turn conversations where the model retains context server-side, which reduces the tokens you need to send in subsequent turns.

For Anthropic, Google, and xAI, the parameter is accepted but ignored — context is carried entirely through the `messages` list regardless. The xAI package also does not expose its provider-specific `store` option; see its README for data-retention details.

If you omit `previous_response`, the call works normally; you just won't get the stateful-session benefit on OpenAI Responses API models.

## Structured Output

The SDK supports structured output from `chat()`, `achat()`, `stream_chat()`,
and `astream_chat()` through a raw `json_schema` dict or a Pydantic model passed
as `response_model`. The documented Core portable profile below is guaranteed
across OpenAI, Anthropic, Google, Mistral, and xAI. Its explicit boundaries
define the schema vocabulary that remains portable between organizations.

For streaming methods, parsed fields and terminal structured-output state are
available on the finalized `ChatResponse` passed to `on_done`.

### Core portable JSON Schema profile

The profile is the provider-neutral intersection enforced before an HTTP
request. A portable schema has:

- an object as its root;
- only `string`, `number`, `integer`, `boolean`, `object`, and `array` types;
  nullable fields are expressed only as `[<one type>, "null"]`;
- `additionalProperties: false` on every object, and a `required` list that
  contains every object property exactly once; use a nullable property instead
  of an optional one;
- a single object schema in `items` for every array;
- a non-empty `enum` when `enum` is used; and
- only the profile's structural and metadata keywords: `type`, `properties`,
  `required`, `additionalProperties`, `items`, `enum`, `title`, `description`,
  `$id`, and `$schema`.

Direct local references of the form `#/$defs/<name>` (including
Pydantic-generated references) are inlined locally before this profile is
checked. External, unresolved, recursive, or
semantically combined `$ref` values are rejected with `JSONSchemaError` before
a provider request. Other JSON Schema composition, validation, and conditional
keywords are intentionally outside the portable profile.

OpenAI, Anthropic, and Google enforce this Core profile. The Mistral and xAI
organization packages require core `>=0.9.2,<1.0.0` and enforce the same
boundary; xAI additionally rejects the documented xAI-only invalid constructs
before sending its request.

### Pydantic Integration (`response_model`)

Pass a Pydantic `BaseModel` subclass as `response_model` — the adapter extracts
the JSON Schema automatically and returns a typed instance in
`ChatResponse.parsed_model`.

```python
from pydantic import BaseModel, ConfigDict
from llm_api_adapter.models.messages.chat_message import Prompt, UserMessage
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter

class Person(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    age: int

adapter = UniversalLLMAPIAdapter(
    organization="openai",
    model="gpt-5",
    api_key=openai_api_key,
)

response = adapter.chat(
    messages=[
        Prompt("Extract structured data from the user's message."),
        UserMessage("My name is Alice and I'm 30 years old."),
    ],
    response_model=Person,
    max_tokens=200,
)

print(response.parsed_model)  # Person(name='Alice', age=30)
print(response.parsed_json)   # {"name": "Alice", "age": 30}
```

> **Note:** Pydantic is not a required dependency of this package. Install it separately: `pip install pydantic`. Every nested Pydantic model must also use `ConfigDict(extra="forbid")` so its generated object schema satisfies the portable profile.

`response_model` cannot be combined with `json_schema` or `tools` — passing both raises `JSONSchemaError`.

### Raw JSON Schema (`json_schema`)

The SDK also supports portable structured JSON output via a `json_schema`
parameter in `chat()`, `achat()`, `stream_chat()`, and `astream_chat()`. The
adapter preserves the schema's semantics: it validates the profile rather than
silently adding required fields or changing `additionalProperties`.

```python
from llm_api_adapter.models.messages.chat_message import Prompt, UserMessage
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
    },
    "required": ["name", "age"],
    "additionalProperties": False,
}

adapter = UniversalLLMAPIAdapter(
    organization="openai",
    model="gpt-5",
    api_key=openai_api_key,
)

response = adapter.chat(
    messages=[
        Prompt("Extract structured data from the user's message."),
        UserMessage("My name is Alice and I'm 30 years old."),
    ],
    json_schema=schema,
    max_tokens=200,
)

print(response.content)      # '{"name": "Alice", "age": 30}'
print(response.parsed_json)  # {"name": "Alice", "age": 30}
```

### `json_schema` parameter

- Accepts a `dict` containing a JSON Schema object.
- Cannot be combined with `tools` or `response_model` — passing both raises `JSONSchemaError`.
- Must satisfy the Core portable profile; unsupported, external, or recursive references raise `JSONSchemaError` before the request.
- Raw schemas are parsed as JSON only. The SDK does not run a local JSON Schema validator against the response.
- If a completed, non-refusal response is not valid JSON, `JSONSchemaError` is raised.

### `parsed_json` field

`ChatResponse.parsed_json` contains the parsed `dict` for a valid completed
structured result. It remains `None` for a refusal or incomplete generation,
and `parsed_model` remains `None` in those cases as well.

### Provider behavior

| Provider | Implementation |
|----------|----------------|
| **OpenAI** (standard) | Native `response_format.type=json_schema` with `strict=true` |
| **OpenAI** (Responses API / o-series) | Native `text.format.type=json_schema` with `strict=true` |
| **Anthropic** | Native `output_config.format.type=json_schema` |
| **Google** | `generationConfig.responseMimeType="application/json"` + `responseSchema` |
| **Mistral** | Native `response_format.type=json_schema` with `strict=true` (optional organization package) |
| **xAI** | Responses `text.format.type=json_schema` with `strict=true` plus xAI's additive local overlay (optional organization package) |

The same **portable** schema can be reused across these organizations without
provider-specific request code. Google drops only `$id` and `$schema`, which
are non-semantic metadata that its endpoint does not accept; no transformer
silently drops a semantic constraint.

### Refusal, incomplete, and invalid results

A provider refusal is exposed as `response.refusal`; an incomplete generation
is exposed as `response.incomplete_reason`. Neither is parsed or Pydantic
validated, so both parsed fields remain `None`. A completed non-refusal result
that is invalid JSON, or fails `response_model.model_validate()`, raises
`JSONSchemaError`. Check the terminal fields before using structured output:

```python
response = adapter.chat(
    messages=[UserMessage("Give me the data.")],
    json_schema=schema,
    max_tokens=200,
)
if response.refusal is not None:
    handle_refusal(response.refusal)
elif response.incomplete_reason is not None:
    handle_incomplete(response.incomplete_reason)
else:
    data = response.parsed_json
```

### Error handling

```python
from llm_api_adapter.errors import JSONSchemaError

try:
    response = adapter.chat(
        messages=[UserMessage("Give me the data.")],
        json_schema=schema,
        max_tokens=200,
    )
except JSONSchemaError as e:
    print(f"JSON schema error: {e}")
```

## Vision Input

The SDK supports sending images alongside text using `ImagePart` and the `files` parameter on `UserMessage`. The input contract is shared across OpenAI, Anthropic, and Google; wire-format differences and provider-specific limitations are handled by the adapter.

### Import

```python
from llm_api_adapter.models.messages.chat_message import UserMessage
from llm_api_adapter.models.messages.file_parts import ImagePart
```

### Image from URL

```python
msg = UserMessage(
    "What is in this image?",
    files=[ImagePart(url="https://example.com/photo.jpg")]
)
response = adapter.chat(messages=[msg], max_tokens=200)
print(response.content)
```

### Image from bytes

```python
with open("photo.png", "rb") as f:
    image_bytes = f.read()

msg = UserMessage(
    "Describe this image.",
    files=[ImagePart(data=image_bytes, media_type="image/png")]
)
response = adapter.chat(messages=[msg], max_tokens=200)
print(response.content)
```

### Multiple images

```python
msg = UserMessage(
    "Compare these two images.",
    files=[
        ImagePart(url="https://example.com/before.jpg"),
        ImagePart(url="https://example.com/after.jpg"),
    ]
)
```

### Supported formats

`ImagePart` accepts any image MIME type starting with `image/` — `image/jpeg`, `image/png`, `image/gif`, `image/webp`, `image/heic`, `image/heif`.

When passing a URL, `media_type` is auto-detected from the file extension. For URLs without an extension, pass `media_type` explicitly:

```python
ImagePart(url="https://api.example.com/image?id=42", media_type="image/jpeg")
```

### OpenAI-style dict compatibility

The adapter also normalizes OpenAI-style content lists, so existing code that uses dicts works without changes:

```python
messages = [{
    "role": "user",
    "content": [
        {"type": "text", "text": "What is this?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
    ]
}]
response = adapter.chat(messages=messages, max_tokens=200)
```

> **Note:** `ImagePart` is supported in v0.5.0; `DocumentPart` is introduced in v0.5.1. Google already supports audio input, but `AudioPart` is postponed because Anthropic does not support audio and OpenAI uses a separate audio API, so there is no common provider-neutral contract yet.

## Document Input

The SDK supports PDF documents alongside text using `DocumentPart` and the `files` parameter on `UserMessage`. Provider-specific wire formats are handled automatically.

### Import

```python
from llm_api_adapter.models.messages.chat_message import UserMessage
from llm_api_adapter.models.messages.file_parts import DocumentPart
```

### PDF from URL

```python
msg = UserMessage(
    "Summarize this document in one sentence.",
    files=[DocumentPart(url="https://example.com/report.pdf")]
)
response = adapter.chat(messages=[msg], max_tokens=200)
print(response.content)
```

The PDF MIME type is detected from the `.pdf` extension. Anthropic, Google, and the OpenAI Responses API accept a document URL. OpenAI Chat Completions models below `gpt-5` do not accept a document URL; use bytes instead.

### PDF from bytes

```python
with open("report.pdf", "rb") as f:
    pdf_bytes = f.read()

msg = UserMessage(
    "Summarize this document in one sentence.",
    files=[DocumentPart(data=pdf_bytes, media_type="application/pdf")]
)
response = adapter.chat(messages=[msg], max_tokens=200)
print(response.content)
```

For bytes, the adapter sends the PDF as base64 data in the provider-specific request format. The same `DocumentPart` works with Anthropic, Google, OpenAI Chat Completions, and the OpenAI Responses API.

### File type support

| File type | Anthropic | OpenAI (< gpt-5) | OpenAI (gpt-5+) | Google |
|-----------|-----------|------------------|-----------------|--------|
| ImagePart (URL) | ✅ | ✅ | ✅ | ✅ |
| ImagePart (bytes) | ✅ | ✅ | ✅ | ✅ |
| DocumentPart (URL) | ✅ | ❌ | ✅ | ✅ |
| DocumentPart (bytes) | ✅ | ✅ | ✅ | ✅ |

## Token Usage and Pricing

`achat()` returns the same `usage`, `currency`, and cost fields as `chat()`.
For `astream_chat()`, the finalized response passed to `on_done` contains the
same fields, while `StreamChunk.usage` is populated when the provider reports
usage during streaming.

### Tiered standard-rate estimates

Costs are a standard-rate estimate, not a provider invoice. When a provider
reports usage, the adapter selects the first pricing tier whose
`up_to_prompt_tokens` is greater than or equal to `usage.input_tokens`. The
boundary is inclusive; the final tier has no boundary. Its input and output
rates are both applied to the entire request.

- `context_window_tokens` is the combined input and generated-output capacity.
- `max_output_tokens` is the generated-output capacity.
- A pricing-tier boundary only selects a price; it is not a request limit.
- If the provider does not report `usage`, the adapter does not estimate tokens
  locally. `usage`, `currency`, and `cost_*` remain `None`.

The estimate covers bundled standard text input/output rates only. It excludes
cached input, cache write/storage, batch, flex, priority, modality-specific,
provider-hosted tool, and negotiated-volume charges. Do not use it to reconcile
a provider invoice.

### Token Usage and Pricing Example

```python
google = UniversalLLMAPIAdapter(
    organization="google",
    model="gemini-2.5-flash",
    api_key=google_api_key
)

response = google.chat(**chat_params)

if response.usage is None:
    print("Provider did not report usage; cost is unavailable.")
else:
    print(response.usage.input_tokens, "tokens", f"({response.cost_input} {response.currency})")
    print(response.usage.output_tokens, "tokens", f"({response.cost_output} {response.currency})")
    print(response.usage.total_tokens, "tokens", f"({response.cost_total} {response.currency})")
```

Prices are updated with each release to reflect provider changes. Bundled prices reflect each provider's standard API rates — batch pricing, cached-token discounts, and volume agreements are not accounted for. Use `set_in_per_1m` / `set_out_per_1m` to apply your actual rates if needed:

### Overriding Pricing or Currency

An override replaces that rate in every pricing tier for the selected model.

```python
google = UniversalLLMAPIAdapter(
    organization="google",
    model="gemini-2.5-flash",
    api_key=google_api_key
)

google.pricing.set_in_per_1m(1.5)
google.pricing.set_out_per_1m(3)
google.pricing.set_currency("EUR")

response = google.chat(**chat_params)
print(response.content)
if response.usage is not None:
    print(response.usage.input_tokens, "tokens", f"({response.cost_input} {response.currency})")
    print(response.usage.output_tokens, "tokens", f"({response.cost_output} {response.currency})")
    print(response.usage.total_tokens, "tokens", f"({response.cost_total} {response.currency})")
```

## Logging

The library uses Python's standard `logging` module and does not configure handlers.
Loggers are module-based under `llm_api_adapter.*` (e.g., `llm_api_adapter.universal_adapter`).

* **Default behavior:** No handlers installed, effective level = `WARNING`.
* **No secrets are logged** — API keys and request bodies are excluded. Only event metadata and errors are logged. Adapter and client objects mask the key in `__repr__` (e.g. `api_key='sk-12345...cdef'`), so they are safe to log or print.

### Enable logs (console)

```python
import logging

logging.basicConfig(level=logging.INFO)  # or DEBUG
# Optionally limit logging to this library
logging.getLogger("llm_api_adapter").setLevel(logging.DEBUG)
```

### Write logs to a file

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

### Per-request correlation (optional)

```python
import logging
logger = logging.getLogger("llm_api_adapter")

req_id = "req-123"
logger = logging.LoggerAdapter(logger, {"request_id": req_id})
logger.info("starting call")
```

To include `request_id` in log output, add `%(request_id)s` to your log formatter.

### Reduce noise / silence logs

```python
import logging
logging.getLogger("llm_api_adapter").setLevel(logging.WARNING)   # silence info logs
logging.getLogger("urllib3").setLevel(logging.WARNING)           # if using requests
```

### Env-based log level toggle

```python
# app.py
import logging, os
level = os.getenv("LLM_ADAPTER_LOGLEVEL", "WARNING").upper()
logging.getLogger("llm_api_adapter").setLevel(level)
```

**Tip:** For HTTP-level debugging with `requests`, also set:

```python
import http.client as http_client, logging
http_client.HTTPConnection.debuglevel = 1
logging.getLogger("urllib3").setLevel(logging.DEBUG)
```

Use this only in development.

## Adoption

[Vertec](https://www.vertec.com/en-ch/kb/llm-client/) bundles llm-api-adapter with Vertec Cloud Suite and On-Premises, allowing customers to call OpenAI, Anthropic, and Google models from custom Python extensions through a unified interface.

## Related project

For production applications that need retries, multi-provider failover,
circuit breakers, and tool-session recovery, see
[`llm-api-resilience`](https://github.com/Inozem/llm_api_resilience).

Install it with:

```bash
python -m pip install llm-api-resilience
```

`llm-api-resilience` is built on top of this adapter and keeps its
provider-neutral interface unchanged.

## Development & Testing

Developer setup, deterministic unit and mocked-integration commands, paid E2E requirements, provider-key safety, documentation rules, and the release flow are documented in [CONTRIBUTING.md](CONTRIBUTING.md).

## License

This project is licensed under the terms of the MIT License.  
See the [LICENSE](LICENSE) file for details.
