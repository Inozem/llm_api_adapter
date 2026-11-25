"""Fresh-process checks for Kimi's optional HTTPX boundary."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[2]


_OPTIONAL_DEPENDENCY_SCRIPT = r'''from __future__ import annotations
import asyncio
import ipaddress
import importlib.abc
import importlib.util
from importlib.machinery import ModuleSpec
import socket
import sys
from collections.abc import Iterator
from typing import Any

package_source, core_source, repository_root = sys.argv[1:]
sys.path[:0] = [package_source, core_source, repository_root]

httpx_import_attempts: list[str] = []
class BlockHTTPX(importlib.abc.MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: object = None,
        target: object = None,
    ) -> ModuleSpec | None:
        if fullname == "httpx" or fullname.startswith("httpx."):
            httpx_import_attempts.append(fullname)
            raise ModuleNotFoundError("HTTPX deliberately blocked", name="httpx")
        return None

sys.meta_path.insert(0, BlockHTTPX())

def is_loopback(host: object) -> bool:
    if host in {"localhost", b"localhost"}:
        return True
    if isinstance(host, bytes):
        host = host.decode("ascii", "ignore")
    try:
        return ipaddress.ip_address(host).is_loopback
    except (ValueError, TypeError):
        return False

external_network_attempts: list[str] = []
def audit(event: str, args: tuple[Any, ...]) -> None:
    if event == "socket.getaddrinfo" and args and not is_loopback(args[0]):
        external_network_attempts.append("socket.getaddrinfo")
        raise PermissionError("isolated Kimi test forbids external DNS")
    if event == "socket.connect" and len(args) > 1:
        address = args[1]
        if isinstance(address, tuple) and address and not is_loopback(address[0]):
            external_network_attempts.append("socket.connect")
            raise PermissionError("isolated Kimi test forbids external sockets")

sys.addaudithook(audit)
try:
    importlib.util.find_spec("httpx")
except ModuleNotFoundError as error:
    assert error.name == "httpx"
else:
    raise AssertionError("HTTPX must be unavailable in this child process")
assert "httpx" not in sys.modules
assert httpx_import_attempts == ["httpx"]
httpx_import_attempts.clear()

# Import the real optional plugin only after the blocker is active.
from llm_api_adapter_kimi.plugin import PLUGIN
from llm_api_adapter.organization_registry import (
    ORGANIZATION_PLUGIN_ENTRY_POINT_GROUP,
    OrganizationPlugin,
)
from llm_api_adapter.service_provider_registry import ServiceProviderRegistry
from llm_api_adapter.llm_registry.llm_registry import RegistrySpec
from llm_api_adapter.llms.openai_compatible.sync_client import OpenAICompatibleSyncClient
import llm_api_adapter.adapters.base_adapter as base_adapter_module
import llm_api_adapter.organization_registry as organization_registry_module
import llm_api_adapter.universal_adapter as universal_module
from llm_api_adapter_kimi.clients.async_client import KimiAsyncClient
from unittest.mock import patch

class KimiEntryPoint:
    name = "kimi"
    value = "llm_api_adapter_kimi.plugin:PLUGIN"
    def load(self) -> OrganizationPlugin:
        return PLUGIN

organization_registry_module.entry_points = lambda *, group: (
    [KimiEntryPoint()]
    if group == ORGANIZATION_PLUGIN_ENTRY_POINT_GROUP
    else []
)
universal_module.SERVICE_PROVIDER_REGISTRY = ServiceProviderRegistry()
universal_module.ORGANIZATION_PLUGIN_DISCOVERY = (
    organization_registry_module.OrganizationPluginDiscovery()
)
model_registry = RegistrySpec()
assert PLUGIN.model_metadata is not None
model_registry.register_organization_metadata(PLUGIN.model_metadata)
universal_module.LLM_REGISTRY = model_registry
base_adapter_module.LLM_REGISTRY = model_registry

# Observe the actual native sync-transport factory while constructing the
# no-argument async client. It must not be touched by that constructor.
with patch.object(
    OpenAICompatibleSyncClient,
    "_create_sync_transport",
    autospec=True,
) as sync_transport_factory:
    async_client = KimiAsyncClient()
    sync_transport_factory.assert_not_called()
assert not hasattr(async_client, "_sync_transport")

from llm_api_adapter.llms import requests_transport

MODEL = "kimi-k3"
COMPLETED = {
    "id": "chatcmpl-optional-test",
    "model": MODEL,
    "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": "offline reply"},
        "finish_reason": "stop",
    }],
}
STREAM_LINES = (
    'data: {"id":"chatcmpl-stream","model":"kimi-k3","choices":[{"index":0,"delta":{"content":"offline "}}]}',
    "",
    'data: {"id":"chatcmpl-stream","model":"kimi-k3","choices":[{"index":0,"delta":{"content":"stream"},"finish_reason":"stop"}]}',
    "",
    "data: [DONE]",
    "",
)
calls: list[tuple[str, dict[str, Any]]] = []
class FakeResponse:
    def raise_for_status(self) -> None:
        return None
    def json(self) -> dict[str, Any]:
        return COMPLETED
    def iter_lines(self, decode_unicode: bool = True) -> Iterator[str]:
        _ = decode_unicode
        yield from STREAM_LINES
    def close(self) -> None:
        return None
def fake_post(url: str, **kwargs: Any) -> FakeResponse:
    calls.append((url, kwargs))
    return FakeResponse()
requests_transport.requests.post = fake_post

from llm_api_adapter.models.responses.chat_response import ChatResponse
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter

def native_facade(**kwargs: Any) -> UniversalLLMAPIAdapter:
    return UniversalLLMAPIAdapter(
        organization="kimi", model=MODEL, api_key="synthetic-key", **kwargs
    )

# Default facade construction discovers the real plugin and uses Requests.
facade = native_facade()
assert facade.transport == "requests"
completed = facade.chat(messages=[{"role": "user", "content": "hello"}])
assert completed.content == "offline reply"
done: list[ChatResponse] = []
visible = list(facade.stream_chat(
    messages=[{"role": "user", "content": "hello"}], on_done=done.append
))
assert visible == ["offline ", "stream"]
assert len(done) == 1 and done[0].content == "offline stream"
assert len(calls) == 2
assert all(call[0] == "https://api.moonshot.ai/v1/chat/completions" for call in calls)
assert all(call[1]["headers"]["Authorization"] == "Bearer synthetic-key" for call in calls)
assert calls[1][1]["stream"] is True
assert httpx_import_attempts == [], "default Requests operations attempted HTTPX"

# A selected sync HTTPX transport fails during facade/client construction.
try:
    native_facade(transport="httpx")
except ImportError as error:
    assert "optional 'httpx' dependency" in str(error)
    assert "llm-api-adapter[httpx]" in str(error)
else:
    raise AssertionError("sync HTTPX construction unexpectedly succeeded")
assert httpx_import_attempts, "sync HTTPX construction did not reach the import blocker"
httpx_import_attempts.clear()
assert len(calls) == 2

async def check_default_async_paths() -> None:
    default_facade = native_facade()
    pending = default_facade.achat(
        messages=[{"role": "user", "content": "hello"}]
    )
    assert httpx_import_attempts == [], "achat attempted HTTPX before being awaited"
    try:
        await pending
    except ImportError as error:
        assert "optional 'httpx' dependency" in str(error)
        assert "llm-api-adapter[async]" in str(error)
    else:
        raise AssertionError("default achat unexpectedly succeeded without HTTPX")
    assert httpx_import_attempts, "awaited achat did not reach the import blocker"
    httpx_import_attempts.clear()
    assert len(calls) == 2

    stream = default_facade.astream_chat(
        messages=[{"role": "user", "content": "hello"}]
    )
    assert httpx_import_attempts == [], "astream_chat attempted HTTPX before first consumption"
    assert len(calls) == 2, "astream_chat must remain lazy until first consumption"
    try:
        await anext(stream)
    except ImportError as error:
        assert "optional 'httpx' dependency" in str(error)
        assert "llm-api-adapter[async]" in str(error)
    else:
        raise AssertionError("default astream_chat unexpectedly succeeded without HTTPX")
    assert httpx_import_attempts, (
        "first astream_chat consumption did not reach the import blocker"
    )
    httpx_import_attempts.clear()
    assert len(calls) == 2

asyncio.run(check_default_async_paths())
assert not any(name == "httpx" or name.startswith("httpx.") for name in sys.modules)
assert external_network_attempts == []
print("Kimi optional dependency boundary passed")
'''


@pytest.mark.unit
def test_kimi_optional_httpx_boundary_in_fresh_process() -> None:
    env = os.environ.copy()
    for key in tuple(env):
        normalized = key.upper()
        if "API_KEY" in normalized or normalized.endswith("_TOKEN"):
            env.pop(key, None)
    env["PYTHON_DOTENV_DISABLED"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            _OPTIONAL_DEPENDENCY_SCRIPT,
            str(PACKAGE_ROOT / "src"),
            str(REPOSITORY_ROOT / "src"),
            str(REPOSITORY_ROOT),
        ],
        cwd=REPOSITORY_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.splitlines() == [
        "Kimi optional dependency boundary passed"
    ]
