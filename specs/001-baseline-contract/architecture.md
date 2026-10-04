# llm-api-adapter Living Architecture

**Scope**: Stable architecture and dependency boundaries for the current public SDK contract.

**Normative baseline**: The canonical provider-neutral baseline is maintained in `.specify/memory/constitution.md`. This living architecture records implementation boundaries only.

**Source precedence**: The constitution defines the normative provider-neutral baseline. Registry
data owns verified exact-model facts, while code and tests implement and evidence the contract.
This artifact records durable design boundaries only.

## Architectural Purpose

The project provides one application-facing LLM contract while keeping organization-specific
protocols, model behavior, and optional dependencies outside that shared boundary. Its design
allows an application to select an organization and model without taking a direct dependency on
that organization's SDK or wire format.

## Component Boundaries

```text
Application
    |
UniversalLLMAPIAdapter
    |
Service-provider registry and optional organization-plugin discovery
    |
Organization adapter (common contract validation and normalization)
    |
Organization client / transport (HTTP and streaming protocol)
    |
Organization API
```

### Public facade

`UniversalLLMAPIAdapter` is the single routing facade. It validates caller selection, defaults a
first-party service provider to the organization name, resolves a registered adapter, and delegates
the public request methods. It does not own organization payload rules or protocol behavior.

### Shared adapter contract

`LLMAdapterBase` owns cross-organization validation and lifecycle behavior: message
normalization, tool validation and selection normalization, structured-output preparation,
reasoning resolution, stream buffering and completion, response finalization, and standard
pricing finalization. Shared behavior belongs here only when it is observable as a common
contract for the supported organizations.

### Organization adapters and clients

Organization adapters transform the shared request and response models to and from an
organization's supported behavior. Organization clients and transports own endpoint interaction,
HTTP/SSE resource handling, raw event consumption, and translation of raw failures to public error
categories. They must not leak raw provider events or payload objects through the common API.

### Shared models

Messages, file parts, tool definitions and calls, response/usage/cost values, stream chunks, and
reasoning events are the bidirectional normalization boundary. Application code uses these models;
organization code performs the serialization and parsing around them.

## Organization and Service-Provider Ownership

An **organization** owns its model identity, verified limits, pricing, reasoning capability, and
organization-specific compatibility metadata. A **service provider** owns connection selection,
authentication, endpoint and protocol handling, raw error handling, and usage extraction. A direct
first-party API normally uses the same identifier for both; the distinction remains required for
other service providers and runtimes.

Built-in adapters cover OpenAI, Anthropic, and Google. Mistral, xAI, Qwen, Kimi, DeepSeek, and
Z.ai are separate organization distributions. An installed optional package registers its model
metadata and first-party service-provider adapter through the established organization-plugin
entry point. The facade does not change when a package is added.

## Registry Boundary

The model registry is the authoritative project record for verified model limits, standard token
rates, reasoning capability, exact-model request rules, and capability exceptions. Structured
fields own exact values and allowed sets. Capability exceptions describe only semantic deviations
from the constitutional baseline and must not duplicate those structured values. Registry-owned
exceptions are data, not adapter-local model-name logic. The closed request-rule mechanism may
select an API variant, restrict normalized tool choice, rename a supported request field, or drop
a documented unsupported field. It may not execute arbitrary callbacks or infer behavior from a
model prefix.

Every first-party exact-model profile supplies `capability_exceptions`, including `[]` when no
deviation applies. Exception IDs describe semantic, value-independent behavior, not model values
or pytest IDs. An absent exception or `pass` retains positive baseline evidence. Legacy third-party
registrations without a profile remain usable but uncertified; profile requirements apply at
certification, not ordinary plugin discovery.

Unknown models remain selectable through the chosen adapter but receive no inferred pricing,
reasoning capability, or request transformation. Only documented direct organization snapshot
forms inherit registered base metadata.

## Request and Response Flow

1. The facade selects an adapter by organization and service provider.
2. The adapter normalizes messages and validates common request options.
3. The adapter resolves verified model metadata and applies only its declared compatibility rules.
4. The organization adapter serializes a permitted common request to its native request.
5. The client or transport exchanges the native request and maps raw failures to public errors.
6. The adapter normalizes the response, usage, tool calls, optional reasoning, structured result,
   and applicable cost data into the shared response model.

For streaming, raw organization events are consumed within the organization boundary. The shared
stream lifecycle emits normalized visible text, collects optional reasoning separately, finalizes
the response, delivers completed tool calls, and then invokes completion. No background producer,
partial tool-call API, or universal raw-event API is part of the common contract.

## Structured Output, Tools, and Files

The Core portable JSON Schema profile is a shared compatibility boundary. The common layer
validates the profile before the request; organization layers may perform only documented
non-semantic transformations required by their wire format. A Pydantic response model is a
convenience layer over the same portable structured-output boundary.

Tool definitions and completed calls are normalized by the shared model. The SDK carries tool
requests and results but never executes application functions. Organization-specific tool wire
formats and unsupported normalized choices remain outside core behavior.

Image and PDF inputs use shared file-part types, while their supported forms are an organization
capability boundary. A package may provide explicit preprocessing that is necessary to honor the
public contract, such as Mistral's PDF OCR path. A package must reject an unsupported file form
before dispatching an invalid native request.

## Usage and Cost Boundary

Usage is provider-reported; the common layer does not estimate it. `cached_tokens` means confirmed
automatic cache-read/cache-hit input; separately reported automatic cache writes use
`cache_write_tokens`. Disjoint components normalize to inclusive input only when provider semantics
establish exact counts, without double counting; reads and separately counted writes are disjoint
subsets of inclusive input and their sum cannot exceed it. Organization-owned normalization supplies common
`Usage`/`ChatResponse`; `ChatResponse` owns arithmetic, and `LLMAdapterBase` selects the verified
pricing tier by full input and forwards its rates.
Registry cache-read and cache-write rates are independent and present only for components that can
occur automatically during ordinary adapter requests and whose quantities are provider-reported;
rates are not inferred for opt-in controls, TTLs, resources, or storage. Missing required split
data, an unverified rate for a positive component, or contradictory usage leaves calculated input
and total unavailable; output can remain independently known when its usage and rate are known.
Parsed omitted input/output counts remain `None`, explicit zero remains zero, and wholly absent
usage remains absent, while direct `Usage` construction keeps its first
three positional zero defaults. Existing direct `apply_pricing(input_rate, output_rate, currency)`
and `apply_cost_breakdown(...)` calls remain compatible. Cache is token input, while other metered
operations are cost line items. An SDK-calculated total requires every incurred component and a
compatible currency to be known. An authoritative complete provider total, such as xAI's reported
`cost_in_usd_ticks`, may be present with component costs `None`; absent component costs are not
fabricated or treated as zero. DeepSeek's conditional rate schedule remains registry-owned, with
package-owned dispatch selecting the applicable rate rather than flattening the schedule.

## Dependency and Extension Boundary

The core package has a minimal default dependency boundary. Optional HTTPX support enables the
documented alternate synchronous transport and asynchronous requests. Optional organization
packages own their extra dependencies, registries, clients, and special preprocessing. A provider
SDK, gateway, agent framework, or optional organization dependency must not become a mandatory
core dependency without an explicit approved contract change.

## Quality and Release Boundaries

Unit and mocked integration tests are the deterministic contract evidence. They do not require
provider credentials or network access. `tests/capability_scenarios.py` maps the subset of
capabilities covered by shared E2E scenarios to positive, exception, and always-on evidence; it is
not a second product contract or an inventory of every test. Exact-model registry exceptions select
the applicable shared or package-local route. Provider E2E validation is deliberately bounded,
organization-specific, and performed only through explicit manual verification or designated
post-publish CI lanes. Core and organization packages are independently versioned and released.

That test-only map assigns `BASELINE`, `EXCEPTION`, and `ALWAYS_ON` roles to existing shared
scenarios through `BASELINE_SCENARIOS`, `EXCEPTION_SCENARIOS`, and `ALWAYS_ON_SCENARIOS`. It is
scoped to current shared request forms and one common application-tool loop per model, not every
capability/test. A non-pass exception redirects only that capability; missing or duplicate evidence in
the scoped shared E2E selection fails. Selection uses no runtime pytest IDs or new requests/tool
variants, stays inside the provider lane, and does not select CI jobs or grant secrets.
`tests/external_organization_metadata.py` compares
`KNOWN_ORGANIZATION_PACKAGES`, Core extras, package manifests and entry points, E2E profiles, and
the CI selector by reading Python AST/TOML without executing or importing optional providers. E2E
profiles derive distribution names from `KNOWN_ORGANIZATION_PACKAGES`; the validator detects drift.
Runtime does not read repository TOML or install absent plugins. Publication jobs and credential
boundaries remain explicit.

The executable source of truth for workflow triggers and release mechanics remains the repository
workflow configuration. This artifact preserves the stable boundary, not a copied workflow index.

## Deliberate Non-Goals

This architecture does not promise provider feature parity beyond the documented common contract.
It does not provide automatic retries, fallback routing, idempotency, background stream workers,
application tool execution, a universal raw stream-event type, or support for undocumented native
provider features.
