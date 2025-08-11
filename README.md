# LLM API Adapter SDK for Python

## Overview

This SDK for Python allows you to use LLM APIs from various companies and models through a unified interface. Currently, the project supports API integration for the following companies: OpenAI, Anthropic, and Google. At this stage, only the chat functionality is implemented.

### Version

Current version: 0.2.0


## Features

- **Unified Interface**: Work seamlessly with different LLM providers using a single, consistent API.
- **Multiple Provider Support**: Currently supports OpenAI, Anthropic, and Google APIs, allowing easy switching between them.
- **Chat Functionality**: Provides an easy way to interact with chat-based LLMs.
- **Extensible Design**: Built to easily extend support for additional providers and new functionalities in the future.
- **Error Handling**: Standardized error messages across all supported LLMs, simplifying integration and debugging.
- **Flexible Configuration**: Manage request parameters like temperature, max tokens, and other settings for fine-tuned control.

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
    model="gpt-3.5-turbo",
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

- **max\_tokens**: The maximum number of tokens to generate in the response. This limits the length of the output. Default value: `256`.

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

- **OpenAI**: You can use models like `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`, `gpt-4.5-preview`, `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `gpt-4`, `gpt-4-turbo-preview`, `gpt-3.5-turbo`. Set the `organization` parameter to `openai` and specify the `model` name.

- **Anthropic**: Available models include `claude-opus-4-20250514`, `claude-sonnet-4-20250514`, `claude-3-sonnet-20240229`, `claude-3-haiku-20240307`, `claude-3-5-sonnet-latest`, `claude-3-haiku-20240307`. Set the `organization` parameter to `anthropic` and specify the desired `model`.

- **Google**: Models such as `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.5-flash-lite`, `gemini-2.0-flash`, `gemini-2.0-flash-lite` can be used. Set the `organization` parameter to `google` and specify the `model`.

Example:

```python
adapter = UniversalLLMAPIAdapter(
    organization="openai",
    model="gpt-3.5-turbo",
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
    model="gpt-3.5-turbo",
    api_key=openai_api_key
)
gpt_response = gpt.generate_chat_answer(messages=messages)
print(gpt_response.content)

claude = UniversalLLMAPIAdapter(
    organization="anthropic",
    model="claude-3-haiku-20240307",
    api_key=anthropic_api_key
)
claude_response = claude.generate_chat_answer(messages=messages)
print(claude_response.content)

google = UniversalLLMAPIAdapter(
    organization="google",
    model="gemini-1.5-flash",
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
    model="gpt-3.5-turbo",
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

The `ChatResponse` object returned by `generate_chat_answer` includes several attributes that provide additional details about the response. These attributes will contain data only if they are included in the response from the LLM:

- **model**: The model that generated the response.
- **response_id**: A unique identifier for the response.
- **timestamp**: The time at which the response was generated.
- **tokens_used**: The number of tokens used for the response.
- **content**: The actual text content generated by the model.
- **finish_reason**: The reason why the generation was finished (e.g., "stop" or "length").

## Testing

This project uses `pytest` for testing. Tests are located in the `tests/` directory.

### Running Tests

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
