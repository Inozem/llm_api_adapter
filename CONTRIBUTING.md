# Contributing

This guide covers local development, deterministic tests, paid provider checks, documentation maintenance, and releases for `llm-api-adapter`.

## Local setup

The project supports Python 3.10 and newer. It uses a `src/` layout and keeps provider SDKs out of the runtime dependencies.

Create an isolated environment and install the test and editable package dependencies:

```bash
python -m venv .venv
```

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r tests/requirements-test.txt
python -m pip install -e .
```

POSIX shells:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r tests/requirements-test.txt
python -m pip install -e .
```

## Deterministic test suite

Unit tests are offline. Integration tests use mocked HTTP and provider-shaped responses. Neither suite requires provider credentials or makes paid API calls.

Run both deterministic suites:

```bash
python -m pytest -q -m "unit or integration"
```

Run one suite while iterating:

```bash
python -m pytest -v -m unit
python -m pytest -v -m integration
```

`tests/tests_runner.py` is the same unit-then-integration sequence used by the main CI workflow:

```bash
python tests/tests_runner.py
```

The `dev` and `main` CI workflows run this deterministic coverage across Python 3.10, 3.11, 3.12, 3.13, and 3.14. Provider keys and live E2E calls are excluded from every matrix job.

Keep deterministic tests free of network access, provider credentials, and model-specific prompt tuning. New provider behavior should use sanitized fixtures or mocked transports.

## Coverage baseline

The recorded deterministic baseline (2026-08-03) is **93.29% line coverage** (`2,946 / 3,158` lines), measured on Python 3.10 across the unit and mocked-integration suites. The measured scope is `src/llm_api_adapter`; E2E tests, provider calls, and credentials are excluded. Python 3.10 is the canonical coverage lane, while the other matrix versions verify compatibility. CI now enforces a **90% non-regression threshold** on this canonical report. Do not update the baseline for ordinary releases; refresh it only when intentionally approving a new baseline and threshold.

To reproduce the combined report locally:

```bash
python -m pytest -q -m unit \
  --cov=src/llm_api_adapter \
  --cov-report=
python -m pytest -q -m integration \
  --cov=src/llm_api_adapter \
  --cov-append \
  --cov-report=term-missing \
  --cov-report=xml:coverage.xml
python -m coverage report --show-missing --fail-under=90
```

## E2E tests and provider keys

E2E tests make real provider requests and may incur charges. Run them only deliberately, outside pull-request and Python-version matrix jobs.

The full E2E marker runs the scenarios under `tests/e2e/`. The synchronous feature and structured-output scenarios traverse the models registered for each organization profile, so this command can make more than one request per provider:

```bash
python -m pytest -v -m e2e
```

The required environment variables are:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`
- `MISTRAL_API_KEY` (with the independently installed Mistral package)
- `XAI_API_KEY` (with the independently installed xAI package)

Run one built-in organization independently with its dedicated marker:

```bash
python -m pytest -v -m e2e_openai
python -m pytest -v -m e2e_anthropic
python -m pytest -v -m e2e_google
```

`e2e_builtin` remains the aggregate marker for all three built-in organization
profiles.

`test_json_schema.py` makes one portable structured-output request for every configured registered model. It must return the exact expected JSON without a refusal or incomplete state; advertised structured-output support is not skipped after the request.

The heavyweight async suite uses one latest registered model for each provider
by default; override it only when needed with
`ASYNC_E2E_<ORGANIZATION>_MODEL`. The synchronous HTTPX suite uses the same
bounded selection through `SYNC_HTTPX_E2E_<ORGANIZATION>_MODEL`, and makes one
paid `chat()` request plus one paid `achat()` request for that selected model
per configured provider. The HTTPX requests reserve 512 generated tokens so
models that think by default still have room for visible text.

For a release candidate, open a pull request to `main` first. After review and deterministic CI pass, the maintainer promotes that exact candidate commit through a staging pull request to `dev`. The `dev` branch is protected by an active repository ruleset: direct updates are restricted, pull requests are required, and only repository administrators are on the bypass list. The [dev workflow](.github/workflows/ci-dev.yml) runs deterministic core tests with coverage. The Mistral and xAI package workflows run their respective unit and mocked-integration suites on Python 3.10–3.14 when that package or code it uses changes.

Only after the staging pull request is merged does the [dev release workflow](.github/workflows/ci-dev-release.yml) publish changed distributions to TestPyPI and run paid E2E tests. A core change selects the affected independent OpenAI, Anthropic, and Google E2E lanes. A shared Core dependency additionally runs the Mistral and xAI lanes; a provider-specific built-in adapter, client, or registry change runs only that Core organization lane. Shared E2E infrastructure runs every applicable lane. Each core lane receives only its own API key; `e2e_builtin` is not used in CI. A Mistral or xAI package change publishes only that organization package and runs its corresponding lane. Each lane installs the exact TestPyPI versions through the matching optional extra, verifies plugin discovery when needed, then makes provider calls. Every changed distribution needs a new version because TestPyPI artifacts are immutable; do not raise the version of an unchanged package. The installer retries twice with two-minute waits for TestPyPI propagation and never falls back to an older candidate. After the workflow passes, the maintainer manually installs the TestPyPI packages and verifies the changed behavior and critical flows before merging the pull request to `main`. Do not push directly to `dev`, and do not run these paid provider calls as part of a deterministic PR matrix or multiply them across Python versions.

## Provider-key safety

- Keep keys in local environment variables, an ignored local `.env`, or GitHub Actions Secrets.
- Never commit keys or place them in fixtures, examples, issue reports, logs, CI artifacts, or documentation.
- Review `git diff` before sharing changes or artifacts.
- Treat reasoning output, tool arguments, and `--dump-raw` SSE payloads as potentially sensitive. Redact them before sharing.
- Do not add provider credentials to a test just to make it pass locally.

## Registry verification

The bundled registry contains only published standard text input/output rates;
it is not an invoice calculator. For every registry-data update, verify the
model identifier, context window, maximum output, tier boundaries, reasoning
capabilities, request rules, aliases, and deprecation status against the
provider's official documentation. Do not infer a value when the provider does
not publish it.

Update the root manifest's `effective_date` whenever the built-in core registry
data changes, and keep the source list below current for both core and plugin
organizations.

| Provider | Registry source | Official sources |
| --- | --- | --- |
| OpenAI | `src/llm_api_adapter/llm_registry/organizations/openai.json` | [Model catalog](https://developers.openai.com/api/docs/models) and [API pricing](https://developers.openai.com/api/docs/pricing) |
| Anthropic | `src/llm_api_adapter/llm_registry/organizations/anthropic.json` | [Models overview](https://platform.claude.com/docs/en/about-claude/models/overview), [Claude API pricing](https://platform.claude.com/docs/en/about-claude/pricing), [effort](https://platform.claude.com/docs/en/build-with-claude/effort), and [model deprecations](https://platform.claude.com/docs/en/about-claude/model-deprecations) |
| Google | `src/llm_api_adapter/llm_registry/organizations/google.json` | [Gemini models](https://ai.google.dev/gemini-api/docs/models), [Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing), [Gemini thinking](https://ai.google.dev/gemini-api/docs/generate-content/thinking), and [deprecations](https://ai.google.dev/gemini-api/docs/deprecations) |
| Mistral | `packages/organizations/mistral/src/llm_api_adapter_mistral/registry/organizations/mistral.json` | [Model cards](https://docs.mistral.ai/models/), [pricing](https://docs.mistral.ai/inference/pricing), [reasoning](https://docs.mistral.ai/studio/conversations/reasoning), and [model lifecycle](https://docs.mistral.ai/inference/model-lifecycle) |
| xAI | `packages/organizations/xai/src/llm_api_adapter_xai/registry/organizations/xai.json` | [Models](https://docs.x.ai/developers/models), [pricing](https://docs.x.ai/developers/pricing), and [reasoning](https://docs.x.ai/developers/model-capabilities/text/reasoning) |

This verification excludes cached input, cache write/storage, batch, flex,
priority, modality-specific, provider-hosted tool, and negotiated-volume
charges. Record any unsupported modality-specific rate as an explicit registry
limitation rather than silently applying a text rate.

## Reasoning smoke script

`scripts/reasoning_smoke.py` is a manual live check, not part of the pytest E2E suite or CI. It uses the default dog-and-potato prompt unless another prompt is supplied:

```powershell
python scripts/reasoning_smoke.py `
  --provider openai `
  --model gpt-5.6-sol `
  --require-reasoning `
  --dump-raw
```

Use `--prompt` to test another task. The script prints reasoning summaries and visible answers as they arrive and can dump raw provider events for diagnostics. Raw output may contain model-generated reasoning, tool arguments, or other sensitive response data; keep it local and redact it before sharing.

## Documentation maintenance

- Keep README focused on the public package contract, installation, and user-facing examples.
- Keep contributor setup, test commands, provider-key rules, CI details, and release procedures in this guide.
- Update the relevant documentation when public behavior, provider mappings, or test workflows change. Keep `docs/architecture.json` minimal and place topical details in its linked detail files.
- When structured-output behavior changes, update the README's portable-profile contract, the organization-package READMEs, and deterministic conformance tests together. Do not claim arbitrary JSON Schema compatibility.
- Examples must not require credentials merely to import. Live calls should be explicit and documented.
- When code or project artifacts change, run `graphify update .` and review the resulting diff.

## Release flow

1. Confirm the version and release notes for every changed distribution describe the user-visible changes.
2. Run the same deterministic unit-then-integration sequence used by the main CI workflow, without provider keys:

   ```bash
   python tests/tests_runner.py
   python -m pytest -v -m unit packages/organizations/mistral/tests
   python -m pytest -v -m integration packages/organizations/mistral/tests
   python -m pytest -v -m unit packages/organizations/xai/tests
   ```

   The dev workflow collects coverage separately while running its unit and integration jobs.

3. Build and inspect the core distribution and each changed organization package:

   ```bash
   python -m pip install build
   python -m build
   # Run for each changed organization package:
   python -m build packages/organizations/mistral
   python -m build packages/organizations/xai
   ```

4. Open or update the pull request to `main`. Wait for review and the deterministic main CI to pass.
5. The maintainer opens and merges a staging pull request containing that exact candidate commit into protected `dev`. The dev release workflow publishes only changed distributions to TestPyPI, then runs the E2E lanes affected by those changes. Core organization lanes run independently; provide the matching key for each through CI Secrets only. The current synchronous scenarios may exercise every registered model of that lane.
6. After the E2E jobs pass, the maintainer manually installs the changed package set from TestPyPI. For Mistral, verify the public installation path:

   ```bash
   pip install --index-url https://test.pypi.org/simple/ \\
     --extra-index-url https://pypi.org/simple \\
     "llm-api-adapter[mistral]==<core-version>" \\
     "llm-api-adapter-mistral==<mistral-version>"
   ```

   For xAI, include the transports used by the release candidate:

   ```bash
   pip install --index-url https://test.pypi.org/simple/ \\
     --extra-index-url https://pypi.org/simple \\
     "llm-api-adapter[async,httpx]==<core-version>" \\
     "llm-api-adapter-xai==<xai-version>"
   ```

   Then verify the changed behavior, critical flows, and absence of regressions.
7. Merge the already verified pull request into `main`.
8. After the pull request is merged, create one final tag for each changed distribution: `v<core-version>` for the core package, `mistral-v<mistral-version>` for Mistral, and `xai-v<xai-version>` for xAI. The tags may point to the same commit. The main workflow publishes only the distribution selected by its tag.

The post-publish E2E job is a release-candidate gate, not a general development check. Keep it out of pull-request jobs and Python-version matrices so paid provider calls remain bounded.

To keep this gate maintainer-controlled, protect `dev` with an active repository ruleset that targets only `dev`, requires a pull request before merging, restricts updates, blocks force pushes, and grants bypass only to the approved repository administrators. The `Require a pull request before merging` rule is essential: `Restrict updates` alone still permits an administrator to push directly to `dev`.
