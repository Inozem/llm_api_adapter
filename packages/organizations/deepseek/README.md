# llm-api-adapter-deepseek

Official direct DeepSeek API support for
[llm-api-adapter](https://github.com/Inozem/llm_api_adapter/).

This independently versioned package targets Core `>=0.9.6,<1.0.0` and adds no
DeepSeek SDK dependency. Select it through the existing
`UniversalLLMAPIAdapter` facade with organization `deepseek` and the canonical
model `deepseek-flash`.

## Installation

```bash
pip install "llm-api-adapter[deepseek]"
```

Direct installation is also supported when Core is managed separately:

```bash
pip install llm-api-adapter-deepseek
```

The Core extra and the direct package install provide the same organization
plugin. The extra is the recommended route when installing Core and its
optional organization support together; direct installation is useful when
Core is already installed or managed by a separate dependency set.

## Core discovery behavior

Core recognizes `deepseek` as an optional organization without importing this
package in its base installation. Selecting `deepseek` before installing the
package raises an actionable error:

```text
Organization 'deepseek' is not installed. Install it with: pip install llm-api-adapter-deepseek
```

After either installation route, the entry point is loaded lazily when the
facade selects `organization="deepseek"`. An unknown name such as
`organization="deepseek-like"` is not treated as an uninstalled DeepSeek
package and instead raises the distinct unsupported-organization error.
