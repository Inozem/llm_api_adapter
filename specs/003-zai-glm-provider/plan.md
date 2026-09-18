# Implementation Plan: Z.ai / GLM Provider Release

**Branch**: `feat/zai-glm-provider` | **Date**: 2026-09-18 | **Spec**: [spec.md](./spec.md)

## Summary

Ship Core `llm-api-adapter` `0.9.7` with independently versioned
`llm-api-adapter-zai` `0.1.0`. Core knows the optional `zai` organization and provides an
actionable installation remedy, but never imports the provider package. The package implements
the official Z.ai Chat Completions API using the existing transport contracts and initially
registers only `glm-5.3-flash`, the documented GLM model with the broadest verified baseline
coverage.

It supports text chat, sync/async operation, SSE streaming, application function tools, reasoning,
and image input. Portable JSON Schema/response-model output is explicitly unsupported because the
authoritative vision API schema has no `response_format`. Document input remains unsupported until
both supported direct forms pass the focused live gate. No provider SDK, deployment backend,
arbitrary endpoint, upload, OCR, retry loop, or continuation protocol is introduced.

## Technical Context

**Language/Version**: Python `>=3.10` through 3.14

**Primary Dependencies**: Core `requests>=2.32`; optional `httpx>=0.28` for async and opt-in
synchronous transport. The provider package adds no SDK or direct transport dependency.

**Storage**: N/A. Request, stream, usage, and pricing state is request-local; no provider data is
persisted.

**Testing**: `pytest` deterministic unit and mocked-integration suites; shared Core E2E plus
package-local Z.ai E2E. Pull requests are credential-free; paid E2E receives only `ZAI_API_KEY`.

**Target Platform**: OS-independent Python library distributed through PyPI.

**Project Type**: Core at repository root plus an independently versioned organization package
under `packages/organizations/`.

**Performance Goals**: Preserve existing transport and bounded streaming lifecycle. No local
upload, fetch, conversion, OCR, retry, background work, or new latency target is added.

**Constraints**:

- Use only `POST https://api.z.ai/api/paas/v4/chat/completions` with bearer API-key auth.
- Register exact model ID `glm-5.3-flash`; exclude FlashX, text-only GLM 5.3/5.2, and older VLMs.
- Store official USD rates per million tokens: input `$0.15`, cached input `$0.03`, output `$0.50`.
  Missing or invalid usage leaves the affected estimate unavailable.
- Support `tool_choice="auto"` only and at most 128 functions; other tool choices fail before HTTP.
- Accept only `low`, `high`, and `max` reasoning levels. Reasoning is separate from visible text;
  provider continuation is not claimed.
- Reject portable structured-output requests before HTTP. Reject document input before HTTP unless
  the focused gate proves both Z.ai direct URL and byte forms.
- Keep generic registry schema unchanged; capability and cache-pricing extensions are package-local.

**Scale/Scope**: One Core optional-extra/discovery change, one provider package, one registered
model, four public request modes, and one provider CI/E2E lane.

## Constitution Check

### Pre-design gate

| Principle | Plan evidence | Status |
| --- | --- | --- |
| Stable provider-neutral public contract | No facade, constructor, or normalized-model breaking change. | Pass |
| Shared contract and isolated behavior | Endpoint, auth, payloads, SSE, error mapping, capabilities, and pricing live in the external package. | Pass |
| Registry and abstraction first | Core adds only known-package and extra records; the plugin lazily provides standard metadata and factory registration. | Pass |
| Deterministic evidence and baseline profiles | A named profile runs every applicable shared scenario; explicit unsupported capabilities are the only exclusions. | Pass |
| Lightweight, safe extensibility | No provider SDK, Core dependency, deployment, persisted data, secret fixture, upload, or retry is added. | Pass |

### Post-design gate

All gates remain satisfied. Shared E2E tests whose capability is currently implicit must be marked
with their real `e2e_feature` requirement before the Z.ai profile excludes Flash's unsupported
structured-output path. This corrects the generic test harness; it is not a provider bypass.

## Design Decisions

1. **Use only Flash and the direct official API.** It has the highest confirmed baseline coverage:
   1M context, 128K output, multimodal input, tools, reasoning, streaming, and published pricing.
2. **Keep the plugin boundary unchanged.** Core `0.9.7` advertises `zai`; entry-point discovery
   supplies the adapter and metadata after installation.
3. **Use existing transports and package-local OpenAI-compatible serialization.** No OpenAI or Z.ai
   SDK is introduced.
4. **Declare a conservative capability profile.** Tools are auto-only; structured output and
   continuation are rejected. Documents become supported only after both direct forms pass live E2E.
5. **Price only verified usage.** Standard rates are registry data; package-local cache pricing is
   used only if valid cache-token usage is reported.
6. **Repair shared E2E capability marking.** JSON-schema tests are currently unmarked and cannot
   be excluded by an organization profile; annotate them before the profile declares no structured
   output.

## Project Structure

### Documentation (this feature)

```text
specs/003-zai-glm-provider/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
└── contracts/zai-provider.md
```

### Source Code (repository root)

```text
pyproject.toml
README.md
CONTRIBUTING.md
pytest.ini
src/llm_api_adapter/organization_registry.py

packages/organizations/zai/
├── {pyproject.toml,MANIFEST.in,LICENSE,README.md}
├── src/llm_api_adapter_zai/
│   ├── {__init__.py,py.typed,plugin.py,adapter.py,streaming.py}
│   ├── clients/{__init__.py,sync_client.py,async_client.py}
│   └── registry/{__init__.py,cache_pricing.py,organizations/zai.json}
└── tests/{test_package_scaffold.py,test_capability_discovery.py,test_zai_adapter.py,
           fixtures/zai_capability_discovery.py,e2e/...}

tests/{unit/test_organization_plugins.py,unit/test_ci_e2e_lane_selection.py,e2e/...}
.github/{scripts/select_e2e_lanes.py,workflows/ci-zai-dev.yml,workflows/ci-zai-main.yml,
         workflows/ci-dev.yml,workflows/ci-main.yml,workflows/ci-dev-release.yml}
```

**Structure Decision**: Mirror the implemented DeepSeek package for packaging, discovery,
metadata, tests, CI, and release. Reuse generic Core transports and normalizers, but keep Z.ai
protocol details in `packages/organizations/zai`.

## Complexity Tracking

No constitution violations or complexity waivers are required.
