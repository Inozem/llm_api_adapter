"""Install candidate core and Mistral wheels in a clean offline environment."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import venv
from pathlib import Path


def _venv_python(environment: Path) -> Path:
    """Return the interpreter created by :mod:`venv` on the active platform."""
    directory = "Scripts" if sys.platform == "win32" else "bin"
    executable = "python.exe" if sys.platform == "win32" else "python"
    return environment / directory / executable


def _wheel_path(value: str) -> Path:
    """Validate one explicit candidate wheel path."""
    path = Path(value).resolve()
    if not path.is_file() or path.suffix != ".whl":
        raise argparse.ArgumentTypeError(
            f"Expected an existing .whl file, got: {path}",
        )
    return path


def check_installation(core_wheel: Path, mistral_wheel: Path) -> None:
    """Install exactly the two candidate wheels and verify their distributions."""
    with tempfile.TemporaryDirectory(prefix="llm-api-adapter-wheel-check-") as raw:
        environment = Path(raw) / "venv"
        venv.EnvBuilder(with_pip=True).create(environment)
        python = _venv_python(environment)

        subprocess.run(
            [
                python,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-index",
                str(core_wheel),
                str(mistral_wheel),
            ],
            check=True,
        )
        subprocess.run(
            [
                python,
                "-c",
                (
                    "import importlib.metadata as metadata; "
                    "import llm_api_adapter; "
                    "import llm_api_adapter_mistral; "
                    "assert metadata.version('llm-api-adapter'); "
                    "assert metadata.version('llm-api-adapter-mistral')"
                ),
            ],
            check=True,
        )


def main() -> None:
    """Parse candidate wheels and run the isolated install check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-wheel", required=True, type=_wheel_path)
    parser.add_argument("--mistral-wheel", required=True, type=_wheel_path)
    arguments = parser.parse_args()
    check_installation(arguments.core_wheel, arguments.mistral_wheel)


if __name__ == "__main__":
    main()
