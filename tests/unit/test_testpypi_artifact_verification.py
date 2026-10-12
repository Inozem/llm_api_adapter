import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from urllib.error import HTTPError

import pytest


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = (
    _REPOSITORY_ROOT / ".github" / "scripts" / "verify_testpypi_artifact.py"
)
_MODULE_SPEC = importlib.util.spec_from_file_location(
    "testpypi_artifact_verification", _SCRIPT_PATH
)
assert _MODULE_SPEC is not None
assert _MODULE_SPEC.loader is not None
_VERIFIER = importlib.util.module_from_spec(_MODULE_SPEC)
sys.modules[_MODULE_SPEC.name] = _VERIFIER
_MODULE_SPEC.loader.exec_module(_VERIFIER)


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def _remote_metadata(artifacts: dict[str, bytes]):
    return {
        "urls": [
            {
                "filename": filename,
                "digests": {"sha256": hashlib.sha256(contents).hexdigest()},
            }
            for filename, contents in artifacts.items()
        ]
    }


@pytest.mark.unit
def test_accepts_exact_distribution_files_from_testpypi(tmp_path: Path):
    artifacts = {
        "llm_api_adapter-0.9.8-py3-none-any.whl": b"wheel",
        "llm_api_adapter-0.9.8.tar.gz": b"sdist",
    }
    for filename, contents in artifacts.items():
        (tmp_path / filename).write_bytes(contents)

    _VERIFIER.verify_testpypi_artifact(
        "llm-api-adapter",
        "0.9.8",
        tmp_path,
        open_url=lambda request, timeout: _Response(_remote_metadata(artifacts)),
    )


@pytest.mark.unit
def test_rejects_existing_version_built_from_different_sources(tmp_path: Path):
    (tmp_path / "llm_api_adapter-0.9.8-py3-none-any.whl").write_bytes(b"new wheel")

    with pytest.raises(
        _VERIFIER.ArtifactVerificationError,
        match="different build",
    ):
        _VERIFIER.verify_testpypi_artifact(
            "llm-api-adapter",
            "0.9.8",
            tmp_path,
            open_url=lambda request, timeout: _Response(
                _remote_metadata(
                    {"llm_api_adapter-0.9.8-py3-none-any.whl": b"old wheel"}
                )
            ),
        )


@pytest.mark.unit
def test_rejects_missing_or_extra_testpypi_distribution_files(tmp_path: Path):
    (tmp_path / "llm_api_adapter-0.9.8-py3-none-any.whl").write_bytes(b"wheel")
    remote = _remote_metadata(
        {
            "llm_api_adapter-0.9.8-py3-none-any.whl": b"wheel",
            "llm_api_adapter-0.9.8.tar.gz": b"old source archive",
        }
    )

    with pytest.raises(
        _VERIFIER.ArtifactVerificationError,
        match="distribution files do not match",
    ):
        _VERIFIER.verify_testpypi_artifact(
            "llm-api-adapter",
            "0.9.8",
            tmp_path,
            open_url=lambda request, timeout: _Response(remote),
        )


@pytest.mark.unit
def test_rejects_candidate_missing_from_testpypi(tmp_path: Path):
    (tmp_path / "llm_api_adapter-0.9.8-py3-none-any.whl").write_bytes(b"wheel")

    def not_found(request, timeout):
        raise HTTPError(request.full_url, 404, "Not Found", None, None)

    with pytest.raises(
        _VERIFIER.ArtifactVerificationError,
        match="does not contain",
    ):
        _VERIFIER.verify_testpypi_artifact(
            "llm-api-adapter",
            "0.9.8",
            tmp_path,
            open_url=not_found,
        )
