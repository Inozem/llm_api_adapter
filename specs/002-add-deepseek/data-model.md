# Data Model: DeepSeek Provider Release

This feature stores no application data. The following in-memory and package-metadata entities define the externally observable integration boundary.

## Core entities

### Known organization package

| Field | Rules |
| --- | --- |
| `organization` | Exact lowercase `deepseek`. |
| `distribution` | Exact `llm-api-adapter-deepseek`. |
| installation command | Derived by Core as `pip install llm-api-adapter-deepseek`. |

**Lifecycle**: Static Core metadata in `0.9.6`; consulted only after lazy plugin discovery does not register `deepseek`. It distinguishes a missing optional package from an unknown organization.

### Chat response provider data

| Field | Rules |
| --- | --- |
| `ChatResponse.provider_data` | Optional opaque mapping; absent for existing providers unless they need transport-only continuation material. It is declared with `repr=False` and is never public serialized, rendered, or logged. |
| `deepseek.reasoning_replay` | Package-owned opaque data reconstructed from a matching `deepseek-flash` response; never text yielded by a stream, never part of `reasoning_events`, debug representation, public serialization, or logs. |

**Lifecycle**: The DeepSeek adapter records it when the provider supplies reasoning replay content. A subsequent DeepSeek request may consume it only from the caller-provided matching `previous_response`, alongside normal caller-supplied history. The package never persists it or serializes a server-side continuation ID. Existing `ChatResponse` users remain valid when this mapping is `None`.

## Package entities

### DeepSeek plugin descriptor

| Field | Rules |
| --- | --- |
| entry-point group | `llm_api_adapter.organizations` |
| entry-point name | `deepseek` |
| entry-point target | `llm_api_adapter_deepseek.plugin:PLUGIN` |
| API version | Must equal Core's organization-plugin API version. |
| registration | Registers only the first-party `deepseek` service provider and package-owned model metadata. |

### DeepSeek model specification

| Field | Rules |
| --- | --- |
| canonical ID | `deepseek-flash` only. |
| limits | Verified positive context and maximum-output limits. Input plus output must fit the context limit. |
| pricing | Verified published peak/off-peak USD input, cached-input, and output rates; no static Core rate is silently substituted. |
| reasoning capability | Exact documented thinking/non-thinking mapping in registry-backed metadata. |
| request rules | Only closed, validated request restrictions: no inferred alias/prefix behavior. |
| capability record | Maps each public mode/capability to supported or explicitly rejected evidence. |

**Lifecycle**: Loaded lazily with the plugin. An unregistered model remains usable only under the existing Core unknown-model behavior; it receives no DeepSeek-specific capability, pricing, reasoning, or request-rule inference.

### DeepSeek request context

| Field | Rules |
| --- | --- |
| normalized messages | Core `Messages` after common validation. System instructions, user text/images, assistant history, and tool outputs are converted to the official Responses input format. |
| effective schema / response model | Core portable schema plus original response model, if supplied. The package sends only the Responses JSON Schema shape and performs Core final validation. |
| tool selection | Core-normalized application function tools and tool choice. Explicit `parallel_tool_calls` control is rejected because DeepSeek does not provide a verified equivalent. |
| reasoning request | Registry-resolved native thinking configuration; replay material comes only from `previous_response.provider_data`. |
| dispatch timestamp | UTC instant captured immediately before the request and used solely to select published time-of-use standard rates. |

**Validation**: Reject `DocumentPart`, non-image file forms, unsupported image media/type/size or position, unsupported built-in tools, unverified model capabilities, malformed schema, invalid tool calls, and unsupported explicit parallel control before outbound HTTP.

### DeepSeek usage and cost estimate

| Field | Rules |
| --- | --- |
| normalized `Usage` | Input, output, and total tokens come only from provider-reported usage. |
| package usage details | Retain reported cached-input and reasoning-token counts when valid. |
| `cost_input` / `cost_output` | Calculated only from valid provider usage and the dispatch-time published rate schedule. |
| `cost_total` | Present only when every included token component can be priced in USD. |

**Validation**: Missing, negative, non-integer, or internally inconsistent usage never produces local estimates. The value is a standard-rate estimate, not an invoice.

### Responses stream state

| Field | Rules |
| --- | --- |
| response metadata | Response ID, model, terminal state, usage, and opaque replay data. |
| visible text | Collected separately from reasoning and emitted only through the Core chunk lifecycle. |
| reasoning | Collected only for `capture_reasoning=True`; never emitted as visible text. |
| function calls | Indexed fragments with name, call ID, and argument fragments; finalized only after valid reconstruction. |
| terminal status | Completed, incomplete, failed, cancelled, or caller-closed. |

**Lifecycle**: Created per stream; terminal completion flushes visible text then provides the normalized result. Failed, cancelled, or closed streams do not flush pending text or invoke final-completion callbacks.

## Relationships

```text
Core optional extra + Known organization package
        │
        └── DeepSeek plugin ── registers ──> DeepSeek adapter + model specification
                                                     │
Caller messages + tools + schema + previous response ─┤
                                                     ├──> request context ──> Responses request/stream
                                                     │                              │
                                                     └──> normalized response <── usage, output, replay data
```
