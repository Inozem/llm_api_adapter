# LLM API Adapter SDK for Python
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/llm-api-adapter?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/llm-api-adapter)

[![PyPI Downloads](https://static.pepy.tech/personalized-badge/llm-api-adapter?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/llm-api-adapter)

## Overview

This lightweight SDK for Python allows you to use LLM APIs from various providers and models through a unified interface. It is designed to be minimal, dependency-free, and easy to integrate into any Python project. Currently, the project supports API integration for OpenAI, Anthropic, and Google, focusing on chat functionality with consistent cost tracking and unified error handling.

### Version

Current version: 0.2.1


## Features

- **Unified Interface**: Work seamlessly with different LLM providers using a single, consistent API.
- **Multiple Provider Support**: Currently supports OpenAI, Anthropic, and Google APIs, allowing easy switching between them.
- **Chat Functionality**: Provides an easy way to interact with chat-based LLMs.
- **Extensible Design**: Built to easily extend support for additional providers and new functionalities in the future.
- **Error Handling**: Standardized error messages across all supported LLMs, simplifying integration and debugging.
- **Flexible Configuration**: Manage request parameters like temperature, max tokens, and other settings for fine-tuned control.
- **Token and Cost Accounting**: Automatic calculation of token usage and cost per request.
- **Pricing Registry**: Model prices are stored in a unified JSON registry with per-model input/output pricing and currency support.

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

response = adapter.generate_chat_answer(
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

## Handling Errors

### Common Errors

The SDK provides a set of standardized errors for easier debugging and integration:

- **LLMAPIError**: Base class for all API-related errors. This error is also used for any unexpected LLM API errors.

- **LLMAPIAuthorizationError**: Raised when authentication or authorization fails.

- **LLMAPIRateLimitError**: Raised when rate limits are exceeded.

- **LLMAPITokenLimitError**: Raised when token limits are exceeded.

- **LLMAPIClientError**: Raised when the client makes an invalid request.

- **LLMAPIServerError**: Raised when the server encounters an error.

- **LLMAPITimeoutError**: Raised when a request times out.

- **LLMAPIUsageLimitError**: Raised when usage limits are exceeded.

## Configuration and Management

### Using Different Providers and Models

The SDK allows you to easily switch between LLM providers and specify the model you want to use. Currently supported providers are OpenAI, Anthropic, and Google.

- **OpenAI**: You can use models like `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`, `gpt-4o`, `gpt-4o-mini`.
- **Anthropic**: Available models include `claude-sonnet-4-5`, `claude-opus-4-1`, `claude-opus-4-0`, `claude-sonnet-4-0`, `claude-3-7-sonnet-latest`, `claude-3-5-haiku-latest`, `claude-3-haiku-20240307`.
- **Google**: Models such as `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.5-flash-lite`, `gemini-2.0-flash`, `gemini-2.0-flash-lite` can be used.

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
gpt_response = gpt.generate_chat_answer(messages=messages)
print(gpt_response.content)

claude = UniversalLLMAPIAdapter(
    organization="anthropic",
    model="claude-sonnet-4-5",
    api_key=anthropic_api_key
)
claude_response = claude.generate_chat_answer(messages=messages)
print(claude_response.content)

google = UniversalLLMAPIAdapter(
    organization="google",
    model="gemini-2.5-flash",
    api_key=google_api_key
)
google_response = google.generate_chat_answer(messages=messages)
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

response = adapter.generate_chat_answer(
    messages=messages,
    max_tokens=256,
    temperature=1.0,
    top_p=1.0
)
print(response.content)
```

The `ChatResponse` object returned by `generate_chat_answer` includes:

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

## Token Usage and Pricing

### Token Usage and Pricing Example

```python
google = UniversalLLMAPIAdapter(
    organization="google",
    model="gemini-2.5-flash",
    api_key=google_api_key
)

response = google.generate_chat_answer(**chat_params)

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

response = google.generate_chat_answer(**chat_params)
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

To run all tests, use the following command:

```bash
pytest
```

Alternatively, you can run the tests using the `tests_runner.py` script:

```bash
python tests/tests_runner.py
```

### Dependencies

Ensure you have the required dependencies installed. You can install them using:

```bash
pip install -r requirements-test.txt
```

### Test Structure

*   `unit/`: Contains unit tests for individual components.
*   `integration/`: Contains integration tests to verify the interaction between different parts of the system.

## License

This project is licensed under the terms of the MIT License.  
See the [LICENSE](LICENSE) file for details.
