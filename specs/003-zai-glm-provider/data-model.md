# Data Model: Z.ai / GLM Provider

## Known package (Core)

| Field | Rule |
| --- | --- |
| Organization key | Exact string `zai` |
| Distribution | `llm-api-adapter-zai` |
| Core optional extra | `zai` |
| Absent-package outcome | Existing actionable organization-not-installed error |

Core does not import Z.ai code or carry Z.ai model metadata.

## Plugin metadata (provider package)

| Field | Rule |
| --- | --- |
| Entry-point name and adapter key | `zai` |
| Entry-point object | Versioned `OrganizationPlugin` |
| Metadata owner | Provider package registry |
| Currency | USD |

Plugin discovery registers valid model metadata before its factory and fails atomically on invalid
metadata, matching the existing plugin contract.

## Model profile

| Field | `glm-5.3-flash` value |
| --- | --- |
| Exact identifier | `glm-5.3-flash` |
| Context / output | 1,000,000 / 131,072 tokens |
| Input / cached / output | USD 0.15 / 0.03 / 0.50 per 1M tokens |
| Reasoning | `low`, `high`, `max` |
| Tools | Auto choice, up to 128 declarations |
| Structured output | Unsupported |
| Documents | Unsupported until both direct forms pass live E2E |

The generic registry contains limits, standard tiers, and reasoning. The package README,
credential-free capability fixture, adapter preflight, and E2E profile own the complete matrix.

## Request and result boundaries

| Entity | Required behavior |
| --- | --- |
| Adapter request | Send only declared supported Core inputs to the official Z.ai wire format |
| Tool call | Normalize provider id, name, and JSON arguments into the existing completion contract |
| Reasoning event | Keep provider reasoning separate from visible text |
| Usage / cost | Price only valid reported input, output, and cached-token values |
| Capability error | Reject unsupported model, schema, tool choice, reasoning level, or file form before transport when locally decidable |
