# Optional HTTPX Sync Transport (Pilot)

[← Back to the main README](README.md)

`llm-api-adapter` uses `requests` for synchronous `chat()` and `stream_chat()`
calls by default. The optional HTTPX sync transport is an opt-in pilot for
applications that need HTTPX on their synchronous code path without changing
the provider-neutral adapter API.

## Installation

The base package needs only `requests`:

```bash
pip install llm-api-adapter
```

Install the optional HTTPX extra before selecting the pilot transport:

```bash
pip install "llm-api-adapter[httpx]"
```

If an application uses both asynchronous methods and the sync HTTPX pilot,
install both extras:

```bash
pip install "llm-api-adapter[async,httpx]"
```

`httpx` is imported only when `transport="httpx"` is selected. Existing
synchronous code that uses the default transport does not require it.

## Usage

Pass `transport="httpx"` when constructing `UniversalLLMAPIAdapter`:

```python
from llm_api_adapter.models.messages.chat_message import UserMessage
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter


adapter = UniversalLLMAPIAdapter(
    organization="openai",
    model="gpt-5",
    api_key="...",
    transport="httpx",
)

response = adapter.chat([UserMessage("Explain HTTPX in one sentence.")])
print(response.content)

for text in adapter.stream_chat([UserMessage("Reply with exactly: OK")]):
    print(text, end="", flush=True)
```

The only supported synchronous selections are:

```python
transport="requests"  # default
transport="httpx"     # opt-in pilot
```

Any other value raises `ValueError`. Selecting `"httpx"` without its optional
dependency raises an actionable `ImportError` that names the installation
command.

## Compatibility and scope

The pilot keeps the public synchronous facade unchanged for OpenAI, Anthropic,
and Google:

- `chat()` and `stream_chat()` keep their parameters, normalized response
  objects, provider-specific request rules, and public error hierarchy.
- `stream_chat()` keeps its callback order: `on_chunk` → `on_delta` → yielded
  text; `on_done` runs only after normal completion.
- HTTP response and streaming resources are closed on success, failure, and
  early iterator close.
- The pilot does not add retries, fallbacks, or change the default transport.

`achat()` and `astream_chat()` are unchanged. They continue to use their own
optional HTTPX async implementation; see the [Async API guide](ASYNC_API.md).

## E2E verification profile

The bounded E2E profile is deliberately separate from deterministic tests and
makes one paid `chat()` request through `transport="httpx"` for each provider
with a configured API key. It chooses the latest model registered for OpenAI,
Anthropic, and Google, and is intended for the post-publish release-candidate
job described in [CONTRIBUTING.md](CONTRIBUTING.md#e2e-tests-and-provider-keys).

Override a model only when needed:

```bash
SYNC_HTTPX_E2E_OPENAI_MODEL=...
SYNC_HTTPX_E2E_ANTHROPIC_MODEL=...
SYNC_HTTPX_E2E_GOOGLE_MODEL=...
```

The profile is located at `tests/e2e/test_sync_httpx.py`. It is not part of
the deterministic pull-request matrix and requires the provider API-key
environment variables documented in the contributing guide.
