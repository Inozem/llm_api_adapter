# llm-api-adapter Living Architecture

**Scope**: Stable architecture and dependency boundaries for the current public SDK contract.

**Companion specification**: [Existing SDK Baseline Contract](spec.md)

**Source precedence**: Current code and tests define behavior. This artifact records durable
design boundaries; it does not replace source, test cases, package metadata, or live provider
documentation.

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

Built-in adapters cover OpenAI, Anthropic, and Google. Mistral, xAI, Qwen, and Kimi are separate
organization distributions. An installed optional package registers its model metadata and
first-party service-provider adapter through the established organization-plugin entry point.
The facade does not change when a package is added.

## Registry Boundary

The model registry is the authoritative project record for verified model limits, standard token
rates, reasoning capability, and exact-model request rules. Registry-owned exceptions are data,
not adapter-local model-name logic. The closed request-rule mechanism may select an API variant,
restrict normalized tool choice, rename a supported request field, or drop a documented unsupported
field. It may not execute arbitrary callbacks or infer behavior from a model prefix.

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

Usage is provider-reported; the common layer does not estimate it. Standard text token cost is
calculated from a selected verified tier when usable provider input usage exists. Separately
metered provider operations are represented as independent cost line items. A total is available
only when accounting information is complete and compatible; absent data stays absent.

## Dependency and Extension Boundary

The core package has a minimal default dependency boundary. Optional HTTPX support enables the
documented alternate synchronous transport and asynchronous requests. Optional organization
packages own their extra dependencies, registries, clients, and special preprocessing. A provider
SDK, gateway, agent framework, or optional organization dependency must not become a mandatory
core dependency without an explicit approved contract change.

## Quality and Release Boundaries

Unit and mocked integration tests are the deterministic contract evidence. They do not require
provider credentials or network access. Provider E2E validation is deliberately bounded,
organization-specific, and performed only through explicit manual verification or designated
post-publish CI lanes. Core and organization packages are independently versioned and released.

The executable source of truth for workflow triggers and release mechanics remains the repository
workflow configuration. This artifact preserves the stable boundary, not a copied workflow index.

## Deliberate Non-Goals

This architecture does not promise provider feature parity beyond the documented common contract.
It does not provide automatic retries, fallback routing, idempotency, background stream workers,
application tool execution, a universal raw stream-event type, or support for undocumented native
provider features.
