# Contract: Core / Plugin Refactor 0.9.8

This contract records the observable guarantees of the 0.9.8 refactor. The project constitution remains the sole source of the full provider-neutral API and provider admission baseline.

## Caller-facing compatibility

- `UniversalLLMAPIAdapter` construction, organization selection, typed request/response concepts, synchronous and asynchronous chat/streaming, transport selection, and normalized error classes retain their existing calling patterns.
- OpenAI, Anthropic, and Google remain built in. Mistral, xAI, Qwen, Kimi, DeepSeek, and Z.ai remain independently installable organization packages with their existing Core extras. A known but uninstalled package continues to produce actionable installation guidance.
- `requests` remains the synchronous default; HTTPX remains an explicit synchronous option and the async transport implementation. No provider contract or deployment backend is added.
- Existing positional `Usage(input_tokens, output_tokens, total_tokens)` construction and zero defaults remain valid. The common response may additionally expose optional `usage.cached_tokens` when the provider confirms it; absent or incomplete usage is not converted into a cache hit.
- Provider-parsed partial usage exposes `None` for each omitted input or output count, including when an existing `ChatResponse.from_*` factory is called directly. An explicitly reported zero stays `0`; wholly absent usage stays absent. An omitted total stays `None` except where an existing documented parser computes the exact sum of both confirmed components.
- **Migration for 0.9.8:** Code that performs arithmetic on token counts from partial provider responses must check for `None` first. This intentional public correction prevents missing usage from appearing as measured zero; directly constructed legacy `Usage` instances retain their existing behavior.
- `ChatResponse.cost_input` remains the combined input-token cost, including a verified cached portion where applicable. `cost_output` remains the output-token cost. `cost_total` is set only for complete priceable accounting, including any separately metered operations already supported. Cache input is not a non-token `cost_breakdown` item.
- Existing direct `ChatResponse.apply_pricing(input_rate, output_rate, currency)` calls remain valid. Shared pricing and cost-breakdown methods also respect verified cached rates and unknown provider components, including when invoked outside an adapter.
- A model with a distinct cached-input rate and no valid provider-reported cache split may now return `cost_input=None` and `cost_total=None` where package code previously assumed all input was uncached. This is an intentional correctness change; confirmed output cost may remain available. No cache-control request option is introduced.

## Model exceptions against the baseline

- The canonical baseline defines required behavior and mandatory Core invariants. Each first-party model's registry entry contains an explicit `capability_exceptions` list; the list may be empty.
- An applicable capability absent from a model's exception list keeps its baseline-positive check. A declared exception names the capability, a stable value-independent `behavior_id`, and its verified provider limitation and adapter behavior. Shared outcomes use `pass`, `ignored`, `rejected_before_transport`, or `none_falls_back_to_minimum`; provider-specific IDs are reserved for behavior that those outcomes plus structured metadata cannot express. Exact values and limits come from their dedicated registry fields. A package workaround that fulfills the public contract remains an exception with `behavior_id: "pass"`.
- A missing profile, unknown/duplicate exception, always-on exception, missing/malformed `behavior_id`, or malformed behavior is a profile-validation error for conformance and E2E selection. The selector never interprets an unlisted capability as a reason to skip.
- Older third-party plugins that implement the current entry-point API may still register and serve requests without new metadata. A missing profile prevents profile-based certification, not runtime plugin loading.

## Conformance and E2E selection

- For an exact model, each applicable capability without a declared exception selects its baseline-positive scenario. A `pass` exception keeps that scenario without an additional model-specific E2E route. A non-`pass` exception selects a common or package-local check through its `(capability_id, behavior_id)` pair, replacing only the positive scenario for that capability. A request that exercises several marked capabilities runs only when none of them redirects to a different non-`pass` scenario. Test node IDs stay in the test catalogue; the selector does not parse `behavior` prose.
- Mistral PDF via OCR remains a declared `pass` exception: the baseline-positive PDF check runs, while package-local tests verify OCR routing and costs.
- Shared facade, normalization, error, transport, pricing, and lifecycle invariants run regardless of discretionary model capabilities.
- Missing baseline-positive evidence or missing/duplicate evidence for a non-`pass` behavior pair is a deterministic validation failure. Expected refusals/incomplete results are asserted by their explicit exception scenarios; unexpected ones fail positive scenarios.
- Model decisions choose scenarios **inside** a provider lane. Changed paths select the lane; provider-specific CI jobs alone control exact candidate installation, credentials, and paid E2E execution. Pull requests remain credential-free.

## Organization metadata consistency

- Every known external organization must have one matching Core identity/distribution mapping, optional Core extra, actual package manifest and entry point, E2E profile, and CI lane/job.
- A missing package, name mismatch, or absent profile/lane fails deterministic repository validation with the affected organization and source named.
- Runtime Core does not import an uninstalled provider package to validate this repository structure. Organization distributions retain independent versions, tags, and publication jobs.

## Exclusions

The 0.9.8 contract does not move xAI or another provider into Core, alter provider wire protocols or plugin registration signatures, introduce user-directed cache controls, switch the default synchronous transport, or begin service-provider/deployment profiles.
