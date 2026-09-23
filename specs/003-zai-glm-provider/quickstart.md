# Z.ai Provider Validation Guide

## Prerequisites

- Python 3.10–3.14
- Core candidate `llm-api-adapter` `0.9.7`
- Z.ai candidate `llm-api-adapter-zai` `0.1.0`
- `ZAI_API_KEY` only for authorized live verification

## Deterministic validation

```powershell
python -m pytest -v --ignore=packages/organizations/zai/tests/e2e -m unit packages/organizations/zai/tests
python -m pytest -v --ignore=packages/organizations/zai/tests/e2e -m integration packages/organizations/zai/tests
python -m pytest -v -m unit tests/unit/test_organization_plugins.py tests/unit/test_ci_e2e_lane_selection.py
python -m build
python -m build packages/organizations/zai
```

Expected results: the missing-package message names `llm-api-adapter-zai`; installation triggers
plugin discovery; Flash metadata validates; both transports normalize supported behavior; and
unsupported schemas, tool choices, and file forms fail before a client call.

## Authorized live validation

With `ZAI_API_KEY` supplied only by a maintainer environment or release secret, install exact
candidate artifacts and run:

```powershell
python -m pytest -v --import-mode=importlib -m e2e_zai --rootdir=. tests/e2e packages/organizations/zai/tests/e2e
```

The lane covers sync, async, streaming, tools, reasoning, image URL/data-URL forms, valid usage
and pricing, and normalized errors. It proves structured-output rejection. Document forms become
public only if both URL and byte checks pass; otherwise the profile stays document-free.

## Release evidence

Build both distributions, install the TestPyPI candidates in a clean environment, confirm the
`zai` extra resolves, and rerun the full live command. Pull-request matrices stay network- and
credential-free; the release lane receives only `ZAI_API_KEY`.
