# Phase 0 Research: DeepSeek Provider Release

## Scope and source rule

This research selects a direct official API implementation for Core `0.9.6` and `llm-api-adapter-deepseek` `0.1.0`. It relies on official DeepSeek documentation and the repository's existing plugin architecture; no provider SDK, compatible endpoint, deployment service, or client-side document path is considered.

## Decision 1: expose only `deepseek-flash`

**Decision**: Register `deepseek-flash` as the sole canonical model for `0.1.0`.

**Rationale**: The current [model list](https://api-docs.deepseek.com/api/list-models/) and [pricing table](https://api-docs.deepseek.com/quick_start/pricing/) identify DeepSeek Flash (V4.1-Flash) as the supported model with the broadest compatible coverage: thinking/non-thinking modes, function tools, JSON output, Responses API, streaming, and vision. Its documented 1M context window and 384K maximum output make it the strongest Core-aligned candidate. `deepseek-v4-pro` lacks vision and is being retired; legacy Flash aliases are accepted by the provider but are not stable canonical package API.

**Alternatives considered**:

- Publish the legacy aliases: rejected because the package must expose verified, stable model IDs rather than provider routing aliases.
- Include `deepseek-v4-pro`: rejected because it provides less Core coverage and is retired/routed to Flash.
- Publish a broad, moving model list: rejected because the registry contract is exact-model and evidence-based.

## Decision 2: use Responses API, not Chat Completions

**Decision**: Implement the official Responses API as the sole wire API.

**Rationale**: [Responses API documentation](https://api-docs.deepseek.com/guides/responses_api/) confirms support for Responses SSE, function output items, images, JSON object, and JSON Schema output. The [JSON mode guide](https://api-docs.deepseek.com/guides/json_mode/) limits Chat Completions to JSON object mode; it is therefore insufficient for the Core portable JSON Schema/Pydantic contract. The package will follow xAI's Responses adapter shape but implement DeepSeek's own event, error, and usage protocol.

**Alternatives considered**:

- Direct Chat Completions only: rejected because it loses portable JSON Schema capability.
- Choose API per request: rejected because a single verified Responses surface is smaller, more testable, and avoids capability drift.

## Decision 3: preserve reasoning continuity with opaque metadata

**Decision**: Add backward-compatible `provider_data: dict | None` to `ChatResponse`. The DeepSeek package stores only the opaque replay material needed for a reasoned continuation. A caller passes the previous response via the already public `previous_response` parameter; the adapter uses the metadata with caller-supplied full message history and does not serialize a server-side continuation identifier.

**Rationale**: In [thinking mode](https://api-docs.deepseek.com/guides/thinking_mode/), DeepSeek requires replaying prior `reasoning_content` for continued tool or multi-turn interactions. The Core contract intentionally keeps `previous_response` as an optional provider optimization and has no server-side requirement. Opaque response metadata supports DeepSeek's documented protocol without altering the facade, exposing it as visible output, or persisting it. The existing `ToolCall.provider_data` demonstrates the same transport-only metadata pattern.

**Alternatives considered**:

- Drop reasoning entirely: rejected because Flash's reasoning support is part of its maximum Core-compatible capability coverage.
- Send `previous_response_id`: rejected because DeepSeek Responses is stateless and the package must retain caller-supplied full history.
- Put raw reasoning into normal assistant text or logs: rejected because it breaks visible-text and sensitive-data boundaries.

## Decision 4: capabilities and file boundary

**Decision**: Declare text chat, sync/async streaming, application function tools, portable structured output, model-aware reasoning, image URL/data input, usage, and standard-rate cost estimates for `deepseek-flash`; reject `DocumentPart`, non-image Files API inputs, provider-built-in tools, and unverified model/capability combinations before HTTP.

**Rationale**: The [tool-calling guide](https://api-docs.deepseek.com/guides/tool_calls/), [vision guide](https://api-docs.deepseek.com/guides/vision/), and [Files API guide](https://api-docs.deepseek.com/guides/files_api/) support this boundary. Files are images only and the Responses API does not accept file inputs, so accepting PDF/documents would make the Core interface dishonest. DeepSeek permits parallel calls but offers no verified per-request parallel-call control; explicit control is rejected rather than silently ignored.

**Alternatives considered**:

- OCR, file upload, or local conversion: rejected by feature scope and the former DeepSeek plan's explicit no-client-document-strategy boundary.
- Fall back to a Chat Completions-compatible endpoint: rejected because it weakens structured output and complicates the supported surface.
- Pass provider built-in tools through: rejected because they are not portable application tools and have no matching Core contract.

## Decision 5: package-local time-of-use pricing

**Decision**: Model the published Flash peak/off-peak USD rates in a package-local extension, patterned after Kimi's cache-pricing extension. Compute the estimate from the UTC request-dispatch time and provider-reported token usage; expose cached input and reasoning tokens as provider-specific usage details where reported.

**Rationale**: The official [pricing page](https://api-docs.deepseek.com/quick_start/pricing/) publishes distinct peak and off-peak rates. Core's `PricingTier` selects by prompt-token count, not day/time, so one static rate would misrepresent part of the schedule. A package-local policy retains Core's lightweight registry while returning a documented standard estimate. The rate is never represented as an invoice, and unavailable/malformed usage produces no fabricated cost.

**Alternatives considered**:

- Store one peak or off-peak rate in Core pricing: rejected because it is predictably wrong during the other published period.
- Add a general Core time-pricing framework: rejected as a broader architectural change unsupported by this release's scope.
- Omit all cost information: rejected because the selected model reports usage and has published standard rates that can be estimated honestly.

## Decision 6: errors, retries, and release evidence

**Decision**: Map documented 400/401/402/422/429/500/503 responses to the existing normalized client, authentication, usage-limit, rate-limit, and server error families. Do not add adapter retries. Test transient retry only through the existing maintainer-controlled E2E harness.

**Rationale**: DeepSeek's [error-code guide](https://api-docs.deepseek.com/quick_start/error_codes/) and [rate-limit guide](https://api-docs.deepseek.com/quick_start/rate_limit/) distinguish client errors from retryable operational failures. The constitution prohibits unapproved automatic retry behavior. Deterministic fixtures will cover all mappings and SSE failures; release candidate E2E is bounded to one dedicated lane with the DeepSeek secret.

**Alternatives considered**:

- Retry inside the provider package: rejected because it changes request cost, timing, and caller semantics.
- Treat every provider failure as a generic error: rejected because it loses the normalized public error contract.

## Repository integration findings

- Core already lazily discovers `llm_api_adapter.organizations` entry points in `src/llm_api_adapter/organization_registry.py` and `src/llm_api_adapter/universal_adapter.py`; only the known package record and optional extra are needed for missing-package guidance.
- Kimi is the closest package/release scaffold. xAI is the closest Responses API and Responses SSE structural reference. Neither provider's wire payloads or event parser may be copied unchanged.
- `tests/e2e/conftest.py` contains profile-level feature gates. Generic tools, reasoning, structured-output, stream, and async scenarios are not all gated; `deepseek-flash` must support the profile's selected common capabilities or those scenarios must gain capability gates before the live lane is enabled.
- `.github/scripts/select_e2e_lanes.py` and the release workflows explicitly enumerate external packages. DeepSeek requires complete candidate, deterministic, release, tag, and E2E wiring rather than only a package folder.
