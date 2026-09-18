# DeepSeek Provider Contract

## Release artifacts and installation

| Artifact | Contract |
| --- | --- |
| Core | `llm-api-adapter==0.9.6` advertises optional `deepseek` support without importing the package. |
| Organization package | `llm-api-adapter-deepseek==0.1.0` is independently installable and depends on `llm-api-adapter>=0.9.6,<1.0.0`. |
| Recommended install | `pip install "llm-api-adapter[deepseek]"` |
| Direct install | `pip install llm-api-adapter-deepseek` |
| Optional transports | `[async]` and `[httpx]` forward only to the matching Core extras. |

The existing facade remains the only public entry point:

```python
UniversalLLMAPIAdapter(
    organization="deepseek",
    model="deepseek-flash",
    api_key=...
)
```

If the integration is known to Core but not installed, construction raises the existing actionable not-installed error for `llm-api-adapter-deepseek`. An unknown organization remains a distinct unsupported-organization error.

## Supported boundary

| Capability | `deepseek-flash` contract |
| --- | --- |
| Text chat | Supported through `chat()` and `achat()`. |
| Streaming | Supported through `stream_chat()` and `astream_chat()` with normal Core callback, buffering, close, cancellation, and finalization semantics. |
| Conversation history | Caller retains full normal message history; provider server-side continuation IDs are never sent. |
| Reasoning | Supported through registry-resolved thinking settings. Required opaque replay material is carried only by a matching `previous_response`; visible reasoning remains opt-in. |
| Application tools | Function tools, normal tool results, and auto/none/required/named selection are supported. Named selection and tool-result continuation force `reasoning_level="none"` with a warning because DeepSeek thinking rejects that function-tool combination. Explicit parallel-call control is rejected. Provider built-in tools are unsupported. |
| Structured output | Core portable raw JSON Schema and Pydantic response models use the official Responses JSON Schema feature and Core final validation. |
| Images | Verified user-message image URL/data input forms are supported within official media and size limits. |
| Documents/files | Every `DocumentPart`, non-image file, OCR/upload path, and automatic conversion is rejected before transport. |
| Usage and cost | Valid provider usage yields a documented USD standard-rate estimate using published UTC time-of-use rates; missing or invalid usage leaves relevant values unavailable. |
| Errors | Documented provider failures map to existing normalized public errors; no package retry policy is added. |

## Continuation metadata contract

`previous_response` remains optional and does not become a provider ID. For a reasoned DeepSeek continuation, it supplies opaque provider data from the immediately preceding matching DeepSeek `ChatResponse`; the caller also supplies complete normal messages. The package consumes this data solely to build the next official DeepSeek request. It must not render, include it in `repr`, log, serialize as visible content, or persist the opaque material.

If a caller requests a continuation whose required DeepSeek reasoning replay is unavailable, the adapter must fail with a clear local compatibility/configuration error when it can identify the condition; it must never invent reasoning or silently switch to a different model or mode.

## Non-goals

- No DeepSeek SDK, proxy/compatible endpoint, deployment provider, or self-hosted runtime.
- No PDF/document handling, OCR, local extraction, file retention, or fallback.
- No automatic retry, background operation, server-side state, or model alias inference.
- No provider-built-in tool, retrieval/storage, conversation, computer, code, or MCP feature.
- No change to the facade constructor or existing organization behavior.
