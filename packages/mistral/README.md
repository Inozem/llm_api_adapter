# llm-api-adapter-mistral

The independently released provider package for Mistral's official direct API.

This workspace commit establishes packaging only. It deliberately contains no
Mistral SDK dependency, direct HTTP client, transport implementation, or plugin
entry point. Those arrive with the verified adapter implementation.

The package requires a compatible core release:

```text
llm-api-adapter >=0.9.0,<1.0.0
```

Extras are forwarded to the core and remain independently composable:

- `async` enables the core async transport support.
- `httpx` enables the core HTTPX sync-transport pilot.
- A future deployment extra will remain independent, so callers can compose it
  with a transport extra, for example `[infomaniak,async]`.
