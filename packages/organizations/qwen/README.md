# llm-api-adapter-qwen

Qwen Model Studio support for
[llm-api-adapter](https://github.com/Inozem/llm_api_adapter/).

## Development status

This is an unpublished package scaffold. The direct Model Studio
Anthropic-compatible Messages API implementation, supported-model registry, and
public usage documentation are added in later commits of the Qwen 0.1.0 plan.

## Intended installation

```bash
pip install "llm-api-adapter[qwen]"
```

Direct installation will also be supported:

```bash
pip install llm-api-adapter-qwen
```

Async and opt-in synchronous HTTPX transports will forward to the matching
Core extras.
