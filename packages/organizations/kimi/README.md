# llm-api-adapter-kimi

Kimi / Moonshot support for
[llm-api-adapter](https://github.com/Inozem/llm_api_adapter/).

## Development status

This is an unpublished package scaffold. The direct Kimi Chat Completions API
implementation, supported-model registry, and public usage documentation are
introduced in later commits of the Kimi 0.1.0 plan.

## Intended installation

Install the external organization package directly:

```bash
pip install llm-api-adapter-kimi
```

The Core extra is also available as a convenience:

```bash
pip install "llm-api-adapter[kimi]"
```

Async and opt-in synchronous HTTPX transports will forward to the matching
Core extras.
