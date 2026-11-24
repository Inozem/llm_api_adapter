"""Exercise the native default and optional dependency boundaries in isolation."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[2]


@pytest.mark.unit
def test_default_sync_and_optional_paths_with_httpx_unavailable() -> None:
    script = r'''from __future__ import annotations
import asyncio
import importlib.abc
import importlib.util
import ipaddress
import os
from pathlib import Path
import socket
import sys
from typing import Any, Iterator, Optional

package_source, core_source, repository_root, *secret_paths = sys.argv[1:]
sys.path[:0] = [package_source, core_source, repository_root]
secret_files = {Path(item).resolve() for item in secret_paths}

class BlockHTTPX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: object = None, target: object = None) -> None:
        if fullname == "httpx" or fullname.startswith("httpx."):
            raise ModuleNotFoundError("No module named 'httpx'", name="httpx")
        return None

sys.meta_path.insert(0, BlockHTTPX())

def is_loopback(host: Any) -> bool:
    if host in {"localhost", b"localhost"}:
        return True
    if isinstance(host, bytes):
        host = host.decode("ascii", "ignore")
    try:
        return ipaddress.ip_address(host).is_loopback
    except (ValueError, TypeError):
        return False

def audit(event: str, args: tuple[Any, ...]) -> None:
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        if Path(os.fsdecode(args[0])).resolve() in secret_files:
            raise PermissionError("isolated optional-path test forbids opening .env")
    if event == "socket.getaddrinfo" and args and not is_loopback(args[0]):
        raise PermissionError("isolated optional-path test forbids external DNS")
    if event == "socket.connect" and len(args) > 1:
        address = args[1]
        if isinstance(address, tuple) and address and not is_loopback(address[0]):
            raise PermissionError("isolated optional-path test forbids external sockets")

sys.addaudithook(audit)
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
try:
    importlib.util.find_spec("httpx")
except ModuleNotFoundError as error:
    assert error.name == "httpx"
else:
    raise AssertionError("HTTPX must be unavailable in this child process")
assert "httpx" not in sys.modules

from llm_api_adapter.models.messages.chat_message import UserMessage
import llm_api_adapter.universal_adapter as universal_module
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter
import llm_api_adapter.llms.requests_transport as requests_transport
from llm_api_adapter.service_provider_registry import ServiceProviderRegistry
from llm_api_adapter_mistral.adapter import MistralAdapter

class OfflineResponse:
    payload: dict[str, Any]
    lines: tuple[str, ...]
    closed: bool
    streaming: bool

    def __init__(self, streaming: bool) -> None:
        self.payload = {
            "object": "response", "id": "resp_offline", "status": "completed",
            "output": [{"type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": "offline"}]}],
        }
        response = {"response": self.payload}
        self.lines = (
            "event: response.output_text.delta", 'data: {"delta":"offline"}', "",
            "event: response.completed", f"data: {__import__('json').dumps(response)}", "",
            "data: [DONE]", "",
        )
        self.closed = False
        self.streaming = streaming

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload

    def iter_lines(self, decode_unicode: bool = True) -> Iterator[str]:
        yield from self.lines

    def close(self) -> None:
        self.closed = True

request_calls: list[tuple[str, dict[str, str], dict[str, Any], Optional[float], bool]] = []

def fake_post(
    url: str,
    *,
    headers: dict[str, str],
    json: dict[str, Any],
    timeout: Optional[float],
    stream: bool = False,
) -> OfflineResponse:
    request_calls.append((url, headers, json, timeout, stream))
    return OfflineResponse(streaming=stream)

requests_transport.requests.post = fake_post
adapter = MistralAdapter(model="mistral-large-4", api_key="synthetic-key")
assert adapter.chat([UserMessage("offline")]).content == "offline"
assert "".join(adapter.stream_chat([UserMessage("offline")])) == "offline"
assert len(request_calls) == 2
assert all(call[0] == "https://api.mistral.ai/v1/responses" for call in request_calls)
assert all(call[1]["Authorization"] == "Bearer synthetic-key" for call in request_calls)

service_registry = ServiceProviderRegistry({"mistral": MistralAdapter})
universal_module.SERVICE_PROVIDER_REGISTRY = service_registry
try:
    UniversalLLMAPIAdapter(
        organization="mistral", model="mistral-large-4", api_key="synthetic-key",
        transport="httpx",
    )
except ImportError as error:
    assert "optional 'httpx' dependency" in str(error)
else:
    raise AssertionError("selected HTTPX transport should fail before I/O")
assert len(request_calls) == 2

async def check_async_optional_paths() -> None:
    native = MistralAdapter(model="mistral-large-4", api_key="synthetic-key")
    try:
        await native.achat([UserMessage("offline")])
    except ImportError as error:
        assert "optional 'httpx' dependency" in str(error)
    else:
        raise AssertionError("achat should fail before I/O when HTTPX is absent")
    try:
        async for _text in native.astream_chat([UserMessage("offline")]):
            raise AssertionError("unavailable async transport yielded output")
    except ImportError as error:
        assert "optional 'httpx' dependency" in str(error)
    else:
        raise AssertionError("astream_chat should fail before I/O when HTTPX is absent")

asyncio.run(check_async_optional_paths())
assert len(request_calls) == 2
'''
    env = os.environ.copy()
    for key in tuple(env):
        normalized = key.upper()
        if (
            "API_KEY" in normalized
            or normalized.endswith("_TOKEN")
            or normalized
            in {
                "GOOGLE_APPLICATION_CREDENTIALS",
                "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY",
                "AWS_SESSION_TOKEN",
                "QWEN_WORKSPACE_ID",
            }
        ):
            env.pop(key, None)
    env["PYTHON_DOTENV_DISABLED"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            script,
            str(PACKAGE_ROOT / "src"),
            str(REPOSITORY_ROOT / "src"),
            str(REPOSITORY_ROOT),
            str(REPOSITORY_ROOT / ".env"),
            str(REPOSITORY_ROOT.parent / ".env"),
        ],
        cwd=REPOSITORY_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
