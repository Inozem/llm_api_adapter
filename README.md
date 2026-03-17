# LLM API Adapter SDK for Python
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/llm-api-adapter?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/llm-api-adapter)
[![Tests](https://github.com/Inozem/llm_api_adapter/actions/workflows/ci-main.yml/badge.svg)](https://github.com/Inozem/llm_api_adapter/actions/workflows/ci-main.yml)

## Overview

This lightweight SDK for Python allows you to use LLM APIs from multiple providers through a unified interface. It is designed to be minimal, dependency‑free, and easy to integrate into any Python project.

Currently, the project supports OpenAI, Anthropic, and Google with a consistent chat API, unified error handling, token/cost accounting, reasoning control, timeout support, and unified tool/function calling. Switching between providers or models requires no code changes beyond replacing the organization and model values when creating the adapter.

### Version

Current version: 0.3.1


## Features

- **Unified Interface**: Work seamlessly with different LLM providers using a single, consistent API.
- **Multiple Provider Support**: Currently supports OpenAI, Anthropic, and Google APIs, allowing easy switching between them.
- **Chat Functionality**: Provides an easy way to interact with chat-based LLMs.
- **Tool / Function Calling**: Provider-agnostic tool definitions and normalized tool calls.
- **Extensible Design**: Built to easily extend support for additional providers and new functionalities in the future.
- **Error Handling**: Standardized error messages across all supported LLMs, simplifying integration and debugging.
- **Flexible Configuration**: Manage request parameters like temperature, max tokens, and other settings for fine-tuned control.
- **Token and Cost Accounting**: Automatic calculation of token usage and cost per request.
- **Pricing Registry**: Model prices are stored in a unified JSON registry with per-model input/output pricing and currency support.
- **Unified Reasoning Support**: A single `reasoning_level` parameter that works identically across all providers.
- **Request Timeouts:** Per-request timeout control with a unified `timeout_s` parameter.

## Installation

To install the SDK, you can use pip:

```bash
pip install llm-api-adapter
```

**Note:** You will need to obtain API keys from each LLM provider you wish to use (OpenAI, Anthropic, Google). Refer to their respective documentation for instructions on obtaining API keys.


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

### Config Errors

- **LLMConfigError**: Raised when the request configuration is invalid or incompatible.

- **LLMReasoningLevelError**: Raised only for Anthropic models when max_tokens is less than reasoning_level.

## Configuration and Management

### Using Different Providers and Models

The SDK allows you to easily switch between LLM providers and specify the model you want to use. Currently supported providers are OpenAI, Anthropic, and Google.

- **OpenAI**: You can use models like `gpt-5.4`, `gpt-5.2`, `gpt-5.1`, `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`, `gpt-4o`, `gpt-4o-mini`.
- **Anthropic**: Available models include `claude-opus-4-6`, `claude-sonnet-4-6`, `claude-opus-4-5`, `claude-sonnet-4-5`, `claude-haiku-4-5`, `claude-opus-4-1`, `claude-opus-4-0`, `claude-sonnet-4-0`, `claude-3-haiku-20240307`.
- **Google**: Models such as `gemini-3.1-pro-preview`, `gemini-3.1-flash-lite-preview`, `gemini-3-flash-preview`, `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.5-flash-lite` can be used.

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
10. **finish\_reason**: Reason why generation stopped (e.g., `"stop"`, `"length"`).

## Timeout Support

The SDK supports per-request timeouts for all providers.

### timeout\_s parameter

`timeout_s` defines the maximum time (in seconds) the SDK will wait for an LLM response.

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
from llm_api_adapter.errors.llm_api_error import LLMAPITimeoutError

try:
    response = gpt.chat(**chat_params)
except LLMAPITimeoutError:
    # retry, fallback, or abort
    print("LLM request timed out")
```

## Reasoning Support

This section describes the unified `reasoning_level` parameter that works the same way for all supported providers and their models.

```python
response = adapter.chat(
    messages=[UserMessage("Solve this step-by-step")],
    reasoning_level=2048,
)
```

### Default behavior

If `reasoning_level` is not passed, reasoning is:

- fully disabled where the provider allows it, or
- reduced to the minimal supported level if it cannot be turned off.

This keeps behavior consistent when switching providers or models.

### `reasoning_level` parameter

`reasoning_level` is optional and provider‑agnostic.

Supported forms:

- **int** — explicit numeric level
- **str** — one of: `"none"`, `"low"`, `"medium"`, `"high"`

Internal mapping:

```json
{
  "none": 0,
  "low": 100,
  "medium": 1000,
  "high": 10000
}
```

String values are automatically converted to numbers, and numeric values can be normalized back to named levels.

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

# Reasoning disabled (default)
response = adapter.chat(
    messages=[UserMessage("Simple answer, no reasoning")],
)
```

### Provider independence

`reasoning_level` has the same semantics for all providers:

- same parameter name
- same string levels
- same numeric mapping

This allows switching between OpenAI, Anthropic, and Google without changing reasoning configuration in your code.

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

`chat()` supports:

- `tools`
- `tool_choice`

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

## Token Usage and Pricing

### Token Usage and Pricing Example

```python
google = UniversalLLMAPIAdapter(
    organization="google",
    model="gemini-2.5-flash",
    api_key=google_api_key
)

response = google.chat(**chat_params)

print(response.usage.input_tokens, "tokens", f"({response.cost_input} {response.currency})")
print(response.usage.output_tokens, "tokens", f"({response.cost_output} {response.currency})")
print(response.usage.total_tokens, "tokens", f"({response.cost_total} {response.currency})")
```

### Overriding Pricing or Currency

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
print(response.usage.input_tokens, "tokens", f"({response.cost_input} {response.currency})")
print(response.usage.output_tokens, "tokens", f"({response.cost_output} {response.currency})")
print(response.usage.total_tokens, "tokens", f"({response.cost_total} {response.currency})")
```

## Logging

The library uses Python’s standard `logging` module and does not configure handlers.
Loggers are module-based under `llm_api_adapter.*` (e.g., `llm_api_adapter.universal_adapter`).

* **Default behavior:** No handlers installed, effective level = `WARNING`.
* **No secrets are logged** — API keys and request bodies are excluded. Only event metadata and errors are logged.

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


## Development & Testing

> **Note**  
> This section is intended for developers working with the source code from GitHub.  
> It is **not** relevant for users installing the package from PyPI.

This project uses `pytest` for testing. Tests are located in the `tests/` directory.

### Test suites

- **unit**: fast, offline, no real provider calls
- **integration**: adapter-level integration tests (may use mocked/provider-shaped responses)
- **e2e**: real API calls against providers (requires API keys)

### Running tests

Run everything:

```bash
pytest
```

Run by marker:

```bash
pytest -m unit
pytest -m integration
pytest -m e2e
```

### E2E requirements

E2E tests require provider API keys to be present in environment variables:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`

## License

This project is licensed under the terms of the MIT License.  
See the [LICENSE](LICENSE) file for details.
