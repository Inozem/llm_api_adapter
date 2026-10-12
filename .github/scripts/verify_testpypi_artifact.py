"""Verify that TestPyPI serves the exact distributions built by this run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


class ArtifactVerificationError(RuntimeError):
    """Raised when TestPyPI does not contain the candidate built by this run."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_testpypi_artifact(
    package: str,
    version: str,
    dist_dir: Path,
    *,
    open_url: Callable = urlopen,
) -> None:
    """Require TestPyPI's files to match every local distribution byte for byte."""
    local_files = {
        path.name: _sha256(path)
        for path in dist_dir.iterdir()
        if path.is_file()
    }
    if not local_files:
        raise ArtifactVerificationError(
            f"No built distributions found in {dist_dir}."
        )

    project = quote(package, safe="")
    candidate = quote(version, safe="")
    url = f"https://test.pypi.org/pypi/{project}/{candidate}/json"
    request = Request(url, headers={"User-Agent": "llm-api-adapter-ci/1.0"})
    try:
        with open_url(request, timeout=15) as response:
            remote = json.load(response)
    except HTTPError as error:
        if error.code == 404:
            raise ArtifactVerificationError(
                f"TestPyPI does not contain {package}=={version}."
            ) from error
        raise ArtifactVerificationError(
            f"TestPyPI returned HTTP {error.code} while verifying "
            f"{package}=={version}."
        ) from error
    except URLError as error:
        raise ArtifactVerificationError(
            f"Could not verify {package}=={version} on TestPyPI: {error.reason}"
        ) from error

    remote_files = {
        item["filename"]: item["digests"]["sha256"]
        for item in remote.get("urls", [])
    }
    if set(remote_files) != set(local_files):
        missing = sorted(set(local_files) - set(remote_files))
        unexpected = sorted(set(remote_files) - set(local_files))
        raise ArtifactVerificationError(
            "TestPyPI distribution files do not match this run's build "
            f"(missing: {missing}; unexpected: {unexpected})."
        )

    mismatched = sorted(
        filename
        for filename, digest in local_files.items()
        if remote_files[filename] != digest
    )
    if mismatched:
        raise ArtifactVerificationError(
            "TestPyPI contains a different build for "
            f"{package}=={version}: {mismatched}."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--dist-dir", type=Path, required=True)
    args = parser.parse_args()

    verify_testpypi_artifact(args.package, args.version, args.dist_dir)
    print(
        f"TestPyPI {args.package}=={args.version} matches this run's build; "
        "Core E2E may continue."
    )


if __name__ == "__main__":
    main()
