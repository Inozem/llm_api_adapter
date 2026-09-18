# Phase 0 Research: Z.ai / GLM Provider

## Official API and transport

**Decision**: Call Z.ai's public Chat Completions endpoint directly through the repository's
existing sync and async transports.

- Endpoint: `POST https://api.z.ai/api/paas/v4/chat/completions`
- Authentication: `Authorization: Bearer <ZAI_API_KEY>`
- Non-streaming: JSON response; streaming: SSE ending in `data: [DONE]`.

**Rationale**: Z.ai documents OpenAI SDK compatibility, but the project already owns transport
contracts. Direct package-local serialization avoids a provider SDK or a base-install dependency.

**Alternatives rejected**: Z.ai SDK, OpenAI SDK, and arbitrary OpenAI-compatible endpoints.

Sources: [HTTP introduction](https://docs.z.ai/guides/develop/http/introduction),
[OpenAI Python compatibility](https://docs.z.ai/guides/develop/openai/python),
[OpenAPI schema](https://docs.z.ai/openapi.json), and
[streaming](https://docs.z.ai/guides/capabilities/streaming).

## Initial model matrix

**Decision**: Register only exact ID `glm-5.3-flash` in `0.1.0`.

**Rationale**: Flash has the broadest confirmed baseline coverage: 1M context, 128K output,
text/image/file input, tools, reasoning, sync/async, SSE streaming, and published USD pricing.
FlashX is faster but costlier and unverified; text-only GLM 5.3/5.2 lack image/file capability;
older VLMs have smaller limits and are unverified. The release selects highest verified coverage,
not a broad catalogue.

Sources: [Flash](https://docs.z.ai/guides/vlm/glm-5.3-flash),
[GLM 5.3](https://docs.z.ai/guides/llm/glm-5.3),
[GLM 5.2](https://docs.z.ai/guides/llm/glm-5.2), and
[OpenAPI schema](https://docs.z.ai/openapi.json).

## Capability boundary

| Capability | `glm-5.3-flash` decision |
| --- | --- |
| Text, sync/async chat, SSE streaming | Supported |
| Application tools | Supported; `tool_choice="auto"` only; maximum 128 tools |
| Reasoning | Supported; `low`, `high`, `max`; never visible text |
| Image input | URL and data-URL forms, subject to deterministic and live validation |
| Document input | Withheld until direct URL and bytes forms both pass focused live E2E |
| Portable JSON Schema / response model | Unsupported; reject before HTTP |
| Provider continuation, deployments, video | Out of scope |

**Decision**: Do not treat a generic JSON-format claim as portable structured output. The
authoritative vision request schema has no `response_format`; sending a text-model JSON-mode
request to Flash would violate the contract. Mark shared JSON-schema tests with
`e2e_feature("structured_output")` before the Z.ai profile excludes them.

Sources: [OpenAPI schema](https://docs.z.ai/openapi.json),
[function calling](https://docs.z.ai/guides/capabilities/function-calling),
[thinking](https://docs.z.ai/guides/capabilities/thinking), and
[structured output](https://docs.z.ai/guides/capabilities/struct-output).

## Limits and pricing

**Decision**: Use official Flash rates in USD per million tokens: input `$0.15`, cached input
`$0.03`, and output `$0.50`. A package-local cache-pricing helper prices cached tokens only when
valid provider usage is present; invalid or incomplete accounting remains unavailable.

**Rationale**: The generic registry retains its standard schema; this follows the existing Kimi
package pattern. No universal rate limit is recorded because limits are account-specific.

Sources: [pricing](https://docs.z.ai/guides/overview/pricing) and
[rate limits](https://docs.z.ai/api-reference/rate-limit).

## Repository and release decisions

**Decision**: Follow the implemented DeepSeek release shape: entry-point plugin, root extra,
known-package record, package-local metadata/tests, named Core E2E profile, per-provider CI,
TestPyPI candidate install, and post-publish E2E receiving only `ZAI_API_KEY`.

**Repository evidence**: `specs/002-add-deepseek/plan.md`,
`packages/organizations/deepseek/`, `tests/e2e/conftest.py`, and
`.github/workflows/ci-dev-release.yml`.
