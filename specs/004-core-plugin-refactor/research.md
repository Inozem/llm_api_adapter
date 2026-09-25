# Research: Core / Plugin Architecture Refactor (0.9.8)

**Date**: 2026-09-25  
**Inputs**: [feature specification](./spec.md), [canonical baseline](../001-baseline-contract/spec.md), project constitution, current source and tests, and the [implementation roadmap](https://app.notion.com/p/34f33dd99fc8812ea5f2eae262910ab3).

## 1. Model capability profiles

**Decision**: Add explicit exact-model capability decisions to model-registry metadata. Maintain a versioned catalogue of model-dependent capabilities derived from the existing baseline, including chat and streaming modes, application tools and tool-choice variants, structured output, image and document forms, reasoning controls/events, continuation, usage, refusal, and incomplete outcomes. Each decision is either confirmed `supported` or a documented `exception` with its expected rejection or special behavior. Do not infer support from organization, package placement, model-name prefix, or another model's profile.

**Rationale**: `ModelSpec` currently stores limits, pricing, reasoning, and request rules, but no baseline profile (`src/llm_api_adapter/llm_registry/llm_registry.py`). The current E2E feature set is organization-wide (`tests/e2e/conftest.py`) and therefore cannot represent exact model differences. Repository inspection found 57 registered model entries in nine organization catalogues (43 built-in, 14 external). The baseline remains `specs/001-baseline-contract/spec.md`; its admission rules are not rewritten here.

**Compatibility boundary**: First-party bundled and organization-package catalogues must have complete validated profiles for the 0.9.8 release. An older third-party plugin may continue to register and serve requests without a new plugin API requirement. Its missing capability profile is an error when profile-based conformance/E2E selection or release certification is requested; it never means “unsupported” or “skip.” Shared transport, facade, message/error normalization, pricing, and cleanup invariants remain mandatory checks and cannot be disabled by a model capability decision.

**Alternatives considered**: Copying organization-wide feature flags to each model would preserve false assumptions. Rejecting every legacy third-party plugin at registration would change the external plugin contract. Treating a missing flag as false would hide missing tests. All three were rejected.

## 2. Shared scenario selection and evidence

**Decision**: Define one deterministic mapping from canonical capability identifiers to positive shared scenarios and documented exception scenarios. Select scenarios for each exact registered model from its profile. Fail validation for an unknown status, unknown capability ID, missing scenario mapping, or missing exception evidence. Keep package-local tests for provider-specific behavior, while retaining every applicable shared scenario. Make common cross-cutting scenarios unconditional for the selected lane.

**Rationale**: `tests/e2e/conftest.py` currently deselects `@e2e_feature` tests by organization-level set inclusion. Some common tests iterate models without such markers; `tests/unit/conformance/test_portable_profile_matrix.py` has a separate hand-maintained terminal-outcome map. These sources can drift. `tests/e2e/test_async.py` also has outcome-driven skips for malformed structured results; a model confirmed to support that capability must instead fail the relevant positive check.

**Lane boundary**: `.github/scripts/select_e2e_lanes.py` continues to choose the affected provider lane by changed paths. `.github/workflows/ci-dev-release.yml` continues to own separate jobs, exact candidate installation, and one-provider credential access. Capability data selects scenarios *inside* a lane and never grants access to a provider secret or changes which paid lane runs.

**Alternatives considered**: Growing the organization-level `supported_features` sets or placing test node IDs in provider packages would duplicate model facts and couple registry releases to test names. Both were rejected.

## 3. Cached input and pricing

**Decision**: Add an optional verified cached-input rate to each applicable pricing tier, and add an optional provider-confirmed `cached_tokens` field to the common `Usage` representation. Preserve existing `ChatResponse.cost_input` as the combined ordinary-plus-cached input charge, `cost_output` as output charge, and `cost_total` as a complete total only when every incurred component is known and priceable. Do not add public cache-control operations or require callers to change requests. Use the full reported input-token count to select a tier before splitting it into cached and ordinary portions. A model with a distinct cached rate and missing, malformed, or inconsistent cache split has unknown input and total cost; independently confirmed output cost may remain available.

**Rationale**: Core `PricingTier` has only ordinary input/output rates, and Core `Usage` has no cached field. Z.ai, Kimi, and DeepSeek already report `usage.cached_tokens` through package-specific subclasses and calculate cache costs separately. Kimi's registry has a separate `cache_pricing` block; Z.ai uses `registry/cache_pricing.py`. Generalizing verified static rates removes this duplication while preserving the public aggregate cost fields. Tests in Z.ai and Kimi currently expect a full-rate total without a cache split; that expectation conflicts with the 0.9.8 requirement not to assume a cache miss and must be updated explicitly.

**Non-static rates**: DeepSeek's peak/off-peak rate selection depends on request-dispatch time (`packages/organizations/deepseek/src/llm_api_adapter_deepseek/registry/cache_pricing.py`). Keep that selection package-owned and pass the selected verified rates through shared accounting where possible; do not encode a time-dependent rate as one static registry value. Existing tier boundaries and currency behavior remain authoritative.

**Alternatives considered**: Treating missing cached usage as zero would fabricate a cache miss. Applying the cached rate to all input would double-discount. Flattening DeepSeek to a static rate would misprice time-dependent requests. These were rejected. A new public per-component cost field is deferred because 0.9.8 can report cached tokens and correct aggregate costs with the existing response fields.

## 4. External organization metadata consistency

**Decision**: Keep `KNOWN_ORGANIZATION_PACKAGES` as the Core source for known external organization and distribution names. Derive safe test-profile metadata from it where practical, and add a deterministic repository check that compares that mapping with Core extras, each package's manifest and entry point, E2E profile names/distributions, and the CI lane selector. Fail with the organization and mismatching source. Leave workflow jobs, version extraction, publication boundaries, and secrets explicit per distribution.

**Rationale**: The six external organizations (`mistral`, `xai`, `qwen`, `kimi`, `deepseek`, `zai`) currently appear separately in `organization_registry.py`, `pyproject.toml`, `tests/e2e/conftest.py`, `.github/scripts/select_e2e_lanes.py`, package manifests, and `.github/workflows/ci-dev-release.yml`. The current values align, but no single deterministic check detects a new missing package or changed name. Static package metadata cannot safely be generated from runtime Core imports without changing build behavior.

**Alternatives considered**: Generating CI jobs or package metadata from a runtime registry would entangle build, security, and release policy. Accepting duplicate strings without a drift check would leave the reported failure mode intact. Both were rejected.

## 5. Verification and release scope

**Decision**: Run focused registry, response, accounting, conformance, metadata, and lane-selection tests first, then the credential-free unit and mocked-integration suite across Python 3.10–3.14. Check E2E collection without making provider calls. Paid live E2E remains in the established maintainer-controlled post-publish lanes against exact TestPyPI candidates. Release Core 0.9.8 only after backward compatibility and the common conformance gate are satisfied; keep organization package versions and publication jobs independent.

**Rationale**: The constitution requires deterministic PR checks and segregated paid E2E. `.github/workflows/ci-dev.yml` already runs the Python matrix and a canonical 3.10 coverage floor of 90%. The existing release workflow already owns provider-specific secrets and candidate installation.

**Remaining implementation work, not unresolved design**: Verify the exact capability decisions and any new cached-input rates against each organization's authoritative documentation before editing registry data. Map every shared scenario to its baseline capability or cross-cutting invariant, and update affected package expectations. These are explicit tasks for `$speckit-tasks`; no design decision remains unresolved.
