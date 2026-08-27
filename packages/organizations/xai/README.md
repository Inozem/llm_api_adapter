# llm-api-adapter-xai

The independently released organization package for xAI's official Responses
API in [llm-api-adapter](https://github.com/Inozem/llm_api_adapter/).

The package currently provides synchronous text chat through `POST
/v1/responses`. It uses the core transport contract rather than the xAI SDK or
a package-local HTTP implementation.

Streaming, asynchronous calls, application tools, response continuation,
structured output, reasoning controls, and file inputs arrive in subsequent
implementation commits. Unsupported options fail explicitly rather than being
silently ignored.

The package requires a compatible core release:

```text
llm-api-adapter >=0.9.0,<1.0.0
```

Extras are forwarded to the core and remain independently composable:

- `async` enables the core async transport support.
- `httpx` enables the core HTTPX sync-transport pilot.
- A future service-provider extra remains independent, so callers can compose
  it with a transport extra.
