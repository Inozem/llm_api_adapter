# Implementation Plan: DeepSeek Provider Release

**Branch**: `feat/deepseek` | **Date**: 2026-09-16 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/002-add-deepseek/spec.md`

## Summary

Ship a coordinated release train comprising Core `llm-api-adapter` `0.9.6` and the independently versioned `llm-api-adapter-deepseek` `0.1.0`. Core advertises and diagnoses the optional integration but does not import it. The package owns the direct official DeepSeek Responses API mapping and exposes only canonical `deepseek-flash`, the currently verified model with the broadest compatible Core capability coverage.

The package will reuse the Core facade, plugin registry, transports, message and response normalizers, stream lifecycle, and portable structured-output profile. It will add no provider SDK or base-install dependency. A small additive Core response metadata field preserves opaque DeepSeek reasoning-continuation data through the existing `previous_response` argument; it must never be logged or mixed into visible content. Direct PDF/documents, non-image files, deployment backends, provider-built-in tools, automatic retries, and server-side continuation IDs remain out of scope.

## Technical Context

**Language/Version**: Python `>=3.10` through 3.14

**Primary Dependencies**: Core `requests>=2.32`; optional `httpx>=0.28` for async and opt-in synchronous transport. The DeepSeek package adds no provider SDK or direct transport dependency.

**Storage**: N/A. The adapter holds request-local stream and continuation state only; it does not persist conversations, files, API responses, or reasoning.

**Testing**: `pytest` deterministic unit/mocked-integration suites; Core 3.10 coverage remains at least 90%. One bounded, maintainer-authorized post-publish DeepSeek E2E lane uses only `DEEPSEEK_API_KEY`.

**Target Platform**: OS-independent Python library distributed through PyPI.

**Project Type**: Monorepo workspace: Core package at repository root plus one independently versioned organization package under `packages/organizations/`.

**Performance Goals**: Preserve existing synchronous/asynchronous streaming lifecycle and bounded buffering behavior. No new latency or throughput target is introduced; the release must not add a local document-processing, retry, or background-work path.

**Constraints**:

- Direct official API only; use the Responses API because it supports the portable JSON Schema profile and images for the selected model.
- Canonical public model ID is `deepseek-flash`; retired aliases and the retiring `deepseek-v4-pro` are not public registry entries.
- Direct PDF/document and non-image Files API inputs are rejected before either sync or async transport is called.
- Model-specific limits, pricing policy, reasoning mapping, and request restrictions are registry-backed or package-local data, never model-name conditionals in adapters.
- Fixtures, diagnostics, and documentation contain neither API keys nor raw reasoning/tool payloads. Reasoning remains opt-in for visible observability.
- Pull requests are deterministic and credential-free; paid E2E runs only in the post-publish release-candidate workflow.

**Scale/Scope**: One new known optional organization, one provider package, one canonical model, four public request modes, and one dedicated CI/E2E lane.

## Constitution Check

### Pre-design gate

| Principle | Plan evidence | Status |
| --- | --- | --- |
| Stable provider-neutral public contract | `UniversalLLMAPIAdapter` constructor and request signatures remain unchanged. The additive `ChatResponse.provider_data` is used only by `previous_response` continuation and receives full compatibility coverage. | Pass |
| Shared contract, isolated organization behavior | DeepSeek endpoint, headers, wire payloads, SSE, error parsing, files, pricing schedule, and reasoning encoding live exclusively in the external package. | Pass |
| Registry and abstraction first | Core adds only the known-package record and optional extra; the package contributes lazy metadata, exact model facts, and closed request rules. No model-prefix inference or Core DeepSeek branch is introduced. | Pass |
| Deterministic contract evidence | Package-local facade tests, Core discovery tests, transport parity, compatibility matrix, and bounded E2E are planned before release. | Pass |
| Lightweight, safe extensibility | No new runtime dependency, SDK, deployment backend, retry loop, persisted data, or credential-bearing test data is introduced. | Pass |

### Post-design gate

The Phase 0 and Phase 1 artifacts retain all five gates. The only shared-model change is additive and is justified by an official protocol requirement for reasoning continuity; it is opaque, opt-in for display, and tested across the existing response/message contract. No exception or complexity waiver is needed.

## Design Decisions

1. **Use `deepseek-flash` and the official Responses API only.** It is the verified model/API pair with the largest compatible capability set: text, async, Responses SSE, application function tools, JSON Schema structured output, reasoning, and image input. Chat Completions is not used as a fallback because its JSON mode cannot meet the portable structured-output contract.
2. **Keep the Core facade and plugin boundary unchanged.** Core `0.9.6` advertises the extra and turns `organization="deepseek"` into the existing actionable not-installed error until plugin discovery succeeds. The plugin provides the adapter and its model registry lazily.
3. **Represent reasoned continuation as opaque response metadata.** The package stores DeepSeek-required reasoning replay material in `ChatResponse.provider_data`; the field is an optional opaque mapping declared with `repr=False`. A following call receives it only through the existing `previous_response` object while the caller retains normal message history. It never sends a provider `previous_response_id`, displays this material as visible text or reasoning, includes it in a debug representation or public serialization, or logs it. This is additive and keeps provider protocol data separate from the portable message body.
4. **Reject documents and unsupported capabilities locally.** No OCR, file upload, file conversion, endpoint fallback, provider-built-in tool, or automatic retry is introduced. Unsupported model/capability combinations fail before outbound HTTP whenever the adapter can identify them.
5. **Use a package-local time-of-use pricing extension.** DeepSeek Flash has published peak/off-peak rates that cannot be represented by Core's static token tier alone. The package records both verified schedules and calculates a standard estimate using the UTC request-dispatch time. It exposes the rate basis in documentation; it never calls the result an invoice. Missing usage, an unavailable schedule, or malformed usage leaves cost fields unset.

## Project Structure

### Documentation (this feature)

```text
specs/002-add-deepseek/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── deepseek-provider.md
└── checklists/
    └── requirements.md
```

### Source Code (repository root)

```text
pyproject.toml
README.md
CONTRIBUTING.md
pytest.ini

src/llm_api_adapter/
├── organization_registry.py                  # known optional DeepSeek package
└── models/responses/chat_response.py         # additive opaque provider_data field

packages/organizations/deepseek/
├── pyproject.toml
├── MANIFEST.in
├── LICENSE
├── README.md
├── src/llm_api_adapter_deepseek/
│   ├── __init__.py
│   ├── py.typed
│   ├── plugin.py
│   ├── adapter.py
│   ├── streaming.py
│   ├── clients/{__init__.py,sync_client.py,async_client.py}
│   └── registry/{__init__.py,organizations/deepseek.json,cache_pricing.py}
└── tests/
    ├── e2e/{conftest.py,test_live_contract.py,test_file_contract.py}
    └── {test_package_scaffold.py,test_capability_discovery.py,test_deepseek_adapter.py}

tests/
├── unit/{test_organization_plugins.py,test_ci_e2e_lane_selection.py}
└── e2e/conftest.py

.github/
├── scripts/select_e2e_lanes.py
└── workflows/{ci-deepseek-dev.yml,ci-deepseek-main.yml,ci-dev.yml,ci-main.yml,ci-dev-release.yml}

docs/{architecture,organization_packages,registry,adapters,messages,errors,ci_cd}.json
```

**Structure Decision**: Mirror the independent Kimi package's workspace, packaging, discovery, test, and release shapes. Reuse the xAI Responses API adapter/stream parser as the structural reference, but implement DeepSeek's wire protocol, SSE events, usage, limits, pricing schedule, and error mapping in package-local code. The `docs/*.json` files are generated architecture artifacts; refresh and review them with `graphify update .` only after the source/documentation changes are complete.
