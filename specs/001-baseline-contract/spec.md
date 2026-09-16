# Feature Specification: Existing SDK Baseline Contract

**Feature Branch**: `001-baseline-contract`

**Created**: 2026-09-16

**Status**: Draft

**Input**: User description: "Create a baseline specification for the current externally
observable behavior and stable contracts of the existing llm-api-adapter project."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Use One Contract Across Organizations (Priority: P1)

An application developer selects an organization, model, API key, and optional service provider,
then uses one documented interface to submit a conversation and receive a normalized result.

**Why this priority**: The provider-neutral contract is the SDK's primary externally observable
value.

**Independent Test**: Construct adapters for each installed supported organization with a
provider-shaped mocked response; verify that the same application-facing request shape produces a
normalized response or a documented normalized error.

**Acceptance Scenarios**:

1. **Given** a supported built-in organization and valid adapter settings, **When** a developer
   submits typed messages or supported OpenAI-style message dictionaries, **Then** the SDK accepts
   them through the common chat interface and returns a normalized response.
2. **Given** a supported optional organization package is installed, **When** it is selected by
   organization, **Then** the SDK discovers its adapter without changing the facade API.
3. **Given** an unsupported organization, service provider, or an optional organization whose
   package is absent, **When** it is selected, **Then** construction fails with a clear error that
   distinguishes the unsupported selection from a missing optional package.

---

### User Story 2 - Make Synchronous, Asynchronous, and Streaming Requests (Priority: P1)

An application developer can choose a single response, an asynchronous response, or visible-text
streaming while retaining the same message, tool, structured-output, usage, and final-response
semantics where those capabilities apply.

**Why this priority**: Sync, async, and streaming are documented public entry points, not
provider-specific implementation options.

**Independent Test**: Use mocked organization responses and event streams to verify chat,
asynchronous chat, synchronous streaming, and asynchronous streaming results and callback order.

**Acceptance Scenarios**:

1. **Given** a valid common request, **When** it is made synchronously or asynchronously,
   **Then** the completed result exposes the same normalized response concepts.
2. **Given** a streaming request, **When** visible text arrives, **Then** the iterator yields only
   normalized visible text and never exposes provider event objects or reasoning as visible text.
3. **Given** streaming callbacks, **When** visible text is emitted, **Then** chunk observation,
   text-delta observation, and yielding occur in that order; completed tool calls are delivered
   before final completion.

---

### User Story 3 - Use Portable Messages, Tools, and Structured Results (Priority: P1)

An application developer sends multi-turn messages with optional images or documents, supplies
tool definitions, and requests portable structured output without manually interpreting each
organization's native response format.

**Why this priority**: These types are the stable boundary that enables applications to switch
organizations.

**Independent Test**: Submit representative message, tool, and structured-output requests through
mocked organizations and verify validation, normalized tool calls, parsed output, and documented
unsupported-input errors.

**Acceptance Scenarios**:

1. **Given** valid typed or dictionary messages, **When** they include system, user, assistant,
   and tool-result turns, **Then** the SDK preserves their roles and normalized content.
2. **Given** valid tool definitions and a supported tool-choice mode, **When** an organization
   returns a tool call, **Then** the completed response or stream callback exposes a normalized
   tool name, arguments, and call identifier where available.
3. **Given** a portable JSON Schema or compatible response model, **When** a completed response is
   valid, **Then** the SDK exposes parsed JSON and, for a response model, the validated typed
   result; invalid or incompatible output raises the documented client-side schema error.

---

### User Story 4 - Select Verified Models and Understand Limits (Priority: P2)

An application developer can select a registered model and rely on documented limits, reasoning
capabilities, request compatibility behavior, usage, and standard token-cost information.

**Why this priority**: Model metadata lets callers receive predictable behavior without
provider-specific application logic.

**Independent Test**: Exercise registered, valid direct snapshot, and unverified model selections
with mocked requests and verify registry-derived behavior and warnings.

**Acceptance Scenarios**:

1. **Given** a registered model, **When** the SDK handles a request, **Then** registered limits,
   reasoning capability, and request compatibility rules apply.
2. **Given** an unverified model identifier, **When** the SDK is constructed, **Then** it remains
   usable with a warning and without inferred pricing, reasoning, or model-specific compatibility
   behavior.
3. **Given** an organization omits usage, **When** a response completes, **Then** usage and cost
   values remain unavailable rather than being locally estimated.

### Edge Cases

- Invalid adapter configuration, messages, tool definitions, tool selection, sampling values,
  output schema, or reasoning level is rejected before an organization request where the common
  contract can validate it.
- A requested capability that an organization package explicitly does not support, including
  particular file forms, fails before sending an unsupported request.
- A caller closes or cancels a stream early, or the stream fails, so pending text is not presented
  as a completed response and final-completion callbacks do not run.
- A provider reports an authorization, rate-limit, token-limit, client, server, timeout, or usage
  limit failure, which is exposed through the corresponding normalized error category when known.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The SDK MUST provide one facade that selects an adapter using organization, model,
  API key, an optional service provider, and a documented synchronous transport choice.
- **FR-002**: The SDK MUST support OpenAI, Anthropic, and Google as built-in organizations and
  MUST support Mistral, xAI, Qwen, Kimi, and DeepSeek through separately installable organization
  packages.
- **FR-003**: The SDK MUST accept typed messages and supported OpenAI-style message dictionaries,
  including mixed input, and normalize system, user, assistant, and tool-result turns.
- **FR-004**: The SDK MUST expose synchronous and asynchronous completed-chat operations and
  synchronous and asynchronous visible-text streaming operations with common request concepts.
- **FR-005**: The SDK MUST return a normalized completed response that can include content, tool
  calls, provider response identity, refusal or incomplete state, usage, cost information,
  optional parsed structured output, and opt-in reasoning events.
- **FR-006**: Streaming operations MUST yield normalized visible text only. Optional buffering
  MUST not yield chunks longer than the configured positive limit; reasoning events MUST remain
  separate from visible text.
- **FR-007**: The SDK MUST validate tool definitions and normalize supported tool-choice inputs.
  Completed tool calls MUST expose parsed arguments and be available in completed responses or
  stream completion callbacks.
- **FR-008**: The SDK MUST accept a documented portable JSON Schema or compatible response model
  for structured output, reject incompatible combinations or schemas before request submission,
  and validate completed structured results.
- **FR-009**: The SDK MUST accept supported image and PDF message parts through the common message
  model. Organization-specific support boundaries MUST be explicit and MUST reject unsupported
  forms before request submission.
- **FR-010**: The SDK MUST maintain a verified organization-scoped model registry for model
  limits, standard token pricing, reasoning capability, and exact-model request compatibility.
- **FR-011**: The SDK MUST use exact registered model behavior, with only documented direct-model
  snapshot inheritance. It MUST not infer special behavior for unrelated aliases or unverified
  model identifiers.
- **FR-012**: The SDK MUST normalize known organization failures into its public error hierarchy
  and preserve client-side validation errors as public client or configuration errors.
- **FR-013**: The base installation MUST retain its documented minimal dependency boundary. Async
  and HTTPX transport capabilities, and optional organization support, MUST be available through
  their documented optional installations.
- **FR-014**: The SDK MUST accept a prior normalized response as a continuation optimization where
  supported and otherwise retain compatibility by accepting it without serializing an unsupported
  organization continuation request.
- **FR-015**: The SDK MUST not estimate missing provider token usage or costs. Separately metered
  provider operations, when reported and supported, MUST remain distinguishable from token cost.
- **FR-016**: The SDK MUST expose public error categories for authorization, rate limit, token
  limit, invalid client request, server, timeout, usage limit, invalid tool input, invalid tool
  arguments, invalid tool choice, incompatible structured output, and invalid configuration.
- **FR-017**: A successful stream MUST flush any pending visible text before producing the final
  response. A failed, cancelled, or caller-closed stream MUST not present pending text as a
  successful final chunk or invoke the final-completion callback.
- **FR-018**: Reasoning capture MUST be opt-in. When enabled and organization data is available,
  reasoning events MUST be available separately from visible text; when it is unavailable, the
  completed response MUST expose an empty reasoning-event collection rather than synthesized
  reasoning.
- **FR-019**: Registered request compatibility restrictions MUST either reject an unsupported
  request before it is sent or omit a parameter only under the documented warning behavior. A
  default value omitted by a registered rule MUST not produce a warning.
- **FR-020**: Optional organization packages MUST document any public capability difference that
  changes the common contract, including required operation settings and unsupported message-file,
  tool, reasoning, continuation, or structured-output forms.

### Contract Boundaries

- A standard completed chat response and an asynchronously completed chat response have the same
  normalized result concepts. The asynchronous operations require the documented optional
  installation; selecting an unavailable optional transport reports an actionable configuration
  error.
- Streaming callbacks may be synchronous or awaitable for asynchronous streaming. Callback
  failures are caller failures and are not reclassified as organization failures.
- `tool_choice` accepts the documented automatic, disabled, any-tool, and named-tool forms.
  Named tools must be declared in the supplied tool list, and model-specific availability is
  enforced before request submission.
- A raw JSON Schema result is parsed only after a completed valid JSON result. A response-model
  result also requires validation against the requested model. Refusal or incomplete terminal
  outcomes leave parsed fields unavailable.
- Standard token pricing uses organization-reported input usage and the matching registered tier
  for the whole request. Provider-reported separately billed operations remain separate cost-line
  items; incomplete accounting leaves total cost unavailable.
- Direct registered Anthropic and OpenAI dated snapshots may inherit the matching registered base
  model's metadata while retaining the requested wire model identifier. Other aliases and
  fine-tuned identifiers do not inherit behavior.

### Organization Package Contract

The common contract applies to every installed organization within its documented capability
boundary. The following current differences are externally observable and must remain explicit.

| Organization | Supported contract boundary |
| --- | --- |
| OpenAI | Uses the documented common contract and selects its supported request variant from exact registered model metadata. Provider-side continuation is used only where that variant supports it. |
| Anthropic | Uses the common contract; its documented output-limit and reasoning constraints are enforced before submission. |
| Google | Uses the common contract; function-call identifiers use the function name because the organization does not expose a separate call identifier. |
| Mistral package | Provides the common chat contract. PDF input is converted through its documented OCR capability before chat, and reported OCR page charges remain separate from token cost. Provider-side continuation is not used. |
| xAI package | Provides the common contract through its supported response capability. Public PDF URLs and PDF bytes have distinct documented attachment handling. Provider-side continuation is accepted but not sent. |
| Qwen package | Requires an explicit workspace setting for every operation and supports only its documented endpoint region. It rejects PDF input before request submission. |
| Kimi package | Supports image bytes and data URIs within its documented boundary, but rejects public image URLs and all PDF document forms before request submission. Provider-side continuation is accepted but not sent. |
| DeepSeek package | Supports only its documented canonical model and capability matrix. It rejects direct PDF/document and non-image file input before request submission. A reasoned continuation may use opaque response metadata from a matching prior DeepSeek response; that metadata is never rendered, included in debug output, or logged. |

Model-specific tool-choice and reasoning restrictions remain registry-derived contract behavior.
They must reject or warn according to the applicable common request rule rather than silently
substitute a different caller intent.

### Key Entities

- **Adapter selection**: The caller's organization, service-provider, model, API-key, and
  transport configuration that determines the organization connection while retaining the common
  facade.
- **Message conversation**: A normalized ordered sequence of system, user, assistant, and tool
  turns, with optional supported file parts.
- **Tool definition and tool call**: An application-provided callable description and a
  provider-returned normalized request to invoke it; the SDK does not execute application tools.
- **Completed response**: The normalized outcome of a chat or completed stream, including visible
  content and optional metadata, parsed output, usage, cost, tools, reasoning, or terminal state.
  A documented provider may retain opaque transport-only metadata for continuation, but it never
  becomes visible content, reasoning, debug output, or logged data.
- **Model specification**: Verified organization-owned metadata describing limits, standard rates,
  reasoning, and exact compatibility restrictions for a model.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A caller can use the same documented chat request shape with each installed
  supported organization and receive either a normalized completed response or a normalized
  documented error category.
- **SC-002**: For each public streaming mode, 100% of visible iterator values are text and all
  completed tool calls are delivered before the final completion callback in contract tests.
- **SC-003**: Contract tests cover all four public request modes—synchronous, asynchronous,
  synchronous streaming, and asynchronous streaming—for every core organization and each
  installed organization package's supported conformance scope.
- **SC-004**: For a response without provider usage, 100% of contract-test results leave usage and
  cost fields unavailable rather than producing an estimated value.
- **SC-005**: A caller attempting each documented invalid common input receives a public validation
  error before an outbound organization request in contract tests.

## Assumptions

- This specification records current stable externally observable behavior; it does not authorize
  a new feature, migration, refactor, dependency change, or documentation cleanup.
- Actual code and tests take precedence over prose documentation when they disagree.
- Provider-specific capability differences remain in scope only where they alter the documented
  shared contract or an explicit package boundary; native payload shapes and endpoints are out of
  scope.
- Exact supported model inventory and provider pricing are registry data that may change through
  a separately verified update; this specification defines their behavior, not a frozen list.
- No performance, availability, retention, or roadmap target is asserted because the current
  public contract and tests do not establish one.
