# Contributing

This guide covers local development, deterministic tests, paid provider checks, documentation maintenance, and releases for `llm-api-adapter`.

## Local setup

The project supports Python 3.9 and newer. It uses a `src/` layout and keeps provider SDKs out of the runtime dependencies.

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

Keep deterministic tests free of network access, provider credentials, and model-specific prompt tuning. New provider behavior should use sanitized fixtures or mocked transports.

## E2E tests and provider keys

E2E tests make real provider requests and may incur charges. Run them only deliberately, outside pull-request and Python-version matrix jobs.

The full E2E marker runs the scenarios under `tests/e2e/`; many scenarios iterate over the models registered for each provider, so this command can make more than one request per provider:

```bash
python -m pytest -v -m e2e
```

The required environment variables are:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`

Async E2E checks use the latest registered model for each provider by default; providers without configured API keys are skipped. Override a provider's model only when needed with `ASYNC_E2E_OPENAI_MODEL`, `ASYNC_E2E_ANTHROPIC_MODEL`, or `ASYNC_E2E_GOOGLE_MODEL`.

For a release candidate, open a pull request to `main` first. After review and deterministic CI pass, the maintainer promotes that exact candidate commit to `dev`. The current [dev workflow](.github/workflows/ci-dev.yml) runs deterministic unit and integration tests with coverage, publishes the package to TestPyPI, and then runs the paid E2E marker. That marker currently exercises all registered models in the synchronous scenarios, while async scenarios select one latest registered model per provider. After the workflow passes, the maintainer manually installs the TestPyPI package and verifies the changed behavior and critical flows before merging the pull request. Do not ask contributors to push to `dev`, open pull requests to `dev`, run the paid job as part of a deterministic PR matrix, or multiply it across Python versions.

## Provider-key safety

- Keep keys in local environment variables, an ignored local `.env`, or GitHub Actions Secrets.
- Never commit keys or place them in fixtures, examples, issue reports, logs, CI artifacts, or documentation.
- Review `git diff` before sharing changes or artifacts.
- Treat reasoning output, tool arguments, and `--dump-raw` SSE payloads as potentially sensitive. Redact them before sharing.
- Do not add provider credentials to a test just to make it pass locally.

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
- Examples must not require credentials merely to import. Live calls should be explicit and documented.
- When code or project artifacts change, run `graphify update .` and review the resulting diff.

## Release flow

1. Confirm the package version and release notes describe the user-visible changes.
2. Run the same deterministic unit-then-integration sequence used by the main CI workflow, without provider keys:

   ```bash
   python tests/tests_runner.py
   ```

   The dev workflow collects coverage separately while running its unit and integration jobs.

3. Build and inspect the distribution:

   ```bash
   python -m pip install build
   python -m build
   ```

4. Open or update the pull request to `main`. Wait for review and the deterministic main CI to pass.
5. The maintainer pushes that exact candidate commit to `dev`. The dev workflow publishes it to TestPyPI and runs the current post-publish E2E job. Provide keys through CI Secrets only, and remember that the current synchronous E2E scenarios may exercise every registered model. Do not use a pull request to `dev` for this promotion.
6. After the E2E job passes, the maintainer manually installs the package from TestPyPI and verifies the changed behavior, critical flows, and absence of regressions.
7. Merge the already verified pull request into `main`.
8. After the pull request is merged, create the release tag so the main workflow publishes the final package.

The post-publish E2E job is a release-candidate gate, not a general development check. Keep it out of pull-request jobs and Python-version matrices so paid calls remain bounded.
