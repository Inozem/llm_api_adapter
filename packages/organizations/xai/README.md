# llm-api-adapter-xai

The independently released organization package for xAI's official Responses
API in [llm-api-adapter](https://github.com/Inozem/llm_api_adapter/).

This workspace commit establishes packaging only. It deliberately contains no
xAI SDK dependency, direct HTTP client, transport implementation, model
metadata, or plugin entry point. Those arrive with the verified adapter
implementation.

The package requires a compatible core release:

```text
llm-api-adapter >=0.9.0,<1.0.0
```

Extras are forwarded to the core and remain independently composable:

- `async` enables the core async transport support.
- `httpx` enables the core HTTPX sync-transport pilot.
- A future service-provider extra remains independent, so callers can compose
  it with a transport extra.
