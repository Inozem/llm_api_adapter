"""Fresh-interpreter operation checks for optional Messages dependencies."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from tests.fixtures.anthropic_compatible import (
    events_to_sse_lines,
    messages_event_trace,
    messages_response,
)


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"

_OPTIONAL_IMPORT_BLOCKER = r'''import importlib.abc, json, socket, sys
from pathlib import Path
source = Path(sys.argv[1]).resolve()
assert source.is_dir()
sys.path.insert(0, str(source))
blocked = (
    "httpx", "llm_api_adapter_mistral", "llm_api_adapter_xai",
    "llm_api_adapter_qwen", "llm_api_adapter_kimi",
    "llm_api_adapter_deepseek", "llm_api_adapter_zai",
    "openai", "anthropic", "google", "mistralai", "xai", "qwen",
    "kimi", "deepseek", "zai",
)
def under_blocked_root(name):
    return any(name == root or name.startswith(root + ".") for root in blocked)
assert not any(under_blocked_root(name) for name in sys.modules)
attempts = []
class BlockOptionalModules(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if under_blocked_root(fullname):
            attempts.append(fullname)
            raise ModuleNotFoundError("optional dependency blocked: " + fullname)
sys.meta_path.insert(0, BlockOptionalModules())
network_attempts = []
def forbid_network(*args, **kwargs):
    network_attempts.append("network")
    raise AssertionError("real network access attempted")
socket.getaddrinfo = forbid_network
socket.create_connection = forbid_network
import requests
from unittest.mock import patch
input_data = json.load(sys.stdin)
calls = []
class FakeResponse:
    status_code = 200
    headers = {"Content-Type": "application/json"}
    def __init__(self, content, *, streaming=False):
        self.content = content.encode()
        self.text = content
        self.streaming = streaming
    def raise_for_status(self):
        return None
    def json(self):
        return json.loads(self.text)
    def iter_lines(self, **kwargs):
        return iter(self.content.splitlines())
    def close(self):
        return None
def fake_post(url, **kwargs):
    calls.append((url, kwargs))
    if kwargs.get("stream"):
        return FakeResponse(input_data["sse"], streaming=True)
    return FakeResponse(json.dumps(input_data["response"]))
requests.post = fake_post
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter
import llm_api_adapter.universal_adapter as universal_module
sdk_path = Path(universal_module.__file__).resolve()
assert sdk_path == source / "llm_api_adapter" / "universal_adapter.py"
print("sdk_path=" + str(sdk_path))
'''

_REQUESTS_OPERATIONS = _OPTIONAL_IMPORT_BLOCKER + r'''
operation = input_data["operation"]
organization = input_data["organization"]
model = input_data["model"]
root = input_data["root"]
kwargs = dict(organization=organization, model=model, api_key="synthetic-key")
if organization == "anthropic_compatible":
    kwargs["base_url"] = root
facade = UniversalLLMAPIAdapter(**kwargs)
request_messages = [{"role": "user", "content": "Say hello."}]
if operation == "chat":
    result = facade.chat(messages=request_messages, max_tokens=32)
    visible = result.content
    done = []
else:
    done = []
    visible = list(facade.stream_chat(
        messages=request_messages, max_tokens=32, on_done=done.append
    ))
    result = done[0]
expected_url = root.rstrip("/") + "/messages"
assert len(calls) == 1
url, request = calls[0]
assert url == expected_url
assert request["headers"] == {
    "x-api-key": "synthetic-key",
    "Content-Type": "application/json",
    "anthropic-version": "2023-06-01",
}
body = request["json"]
expected_body = {
    "model": model,
    "max_tokens": 32,
    "messages": [{"role": "user", "content": "Say hello."}],
}
if organization == "anthropic":
    expected_body["temperature"] = 1.0
if operation == "chat":
    assert request.get("stream") is None
    assert visible == "Fresh-process reply"
else:
    assert request["stream"] is True
    expected_body["stream"] = True
    assert visible == ["Fresh-process reply"]
    assert len(done) == 1 and done[0] is result
assert body == expected_body, body
assert result.content == "Fresh-process reply"
assert not any(under_blocked_root(name) for name in sys.modules)
assert attempts == []
assert network_attempts == []
print("isolated requests operation passed")
'''

_MISSING_OPTIONAL_OPERATION = _OPTIONAL_IMPORT_BLOCKER + r'''
operation = input_data["operation"]
organization = input_data["organization"]
model = input_data["model"]
root = input_data["root"]
transport = "httpx" if operation in ("httpx_chat", "httpx_stream") else "requests"
kwargs = dict(
    organization=organization, model=model, api_key="synthetic-key", transport=transport
)
if organization == "anthropic_compatible":
    kwargs["base_url"] = root
facade = UniversalLLMAPIAdapter(**kwargs)
try:
    if operation == "httpx_chat":
        facade.chat(messages=[{"role": "user", "content": "Say hello."}], max_tokens=32)
    elif operation == "httpx_stream":
        list(facade.stream_chat(
            messages=[{"role": "user", "content": "Say hello."}], max_tokens=32
        ))
    elif operation == "async_chat":
        import asyncio
        async def chat():
            await facade.achat(
                messages=[{"role": "user", "content": "Say hello."}], max_tokens=32
            )
        asyncio.run(chat())
    else:
        import asyncio
        async def consume():
            async for _text in facade.astream_chat(
                messages=[{"role": "user", "content": "Say hello."}], max_tokens=32
            ):
                pass
        asyncio.run(consume())
    raise AssertionError("selected missing optional transport unexpectedly succeeded")
except ImportError as error:
    expected_extra = "async" if operation.startswith("async_") else "httpx"
    assert "llm-api-adapter[" + expected_extra + "]" in str(error), str(error)
else:
    raise AssertionError("selected missing optional transport did not report ImportError")
assert len(calls) == 0
httpx_attempts = [name for name in attempts if name == "httpx" or name.startswith("httpx.")]
assert httpx_attempts, attempts
assert not any(under_blocked_root(name) for name in sys.modules)
assert network_attempts == []
print("isolated missing optional operation passed")
'''


def _run_isolated(source: str, payload: dict) -> subprocess.CompletedProcess[str]:
    assert _SOURCE_ROOT.is_dir()
    return subprocess.run(
        [sys.executable, "-I", "-c", source, str(_SOURCE_ROOT)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        timeout=20,
        cwd=_REPOSITORY_ROOT,
        check=False,
    )


@pytest.mark.unit
@pytest.mark.parametrize("organization", ("anthropic_compatible", "anthropic"))
@pytest.mark.parametrize("operation", ("chat", "stream_chat"))
def test_requests_default_operations_work_without_optional_imports(organization, operation):
    """Default requests operations import and execute with optional roots blocked."""
    root = (
        "https://messages.example.test/v1"
        if organization == "anthropic_compatible"
        else "https://api.anthropic.com/v1"
    )
    sse = "\n".join(events_to_sse_lines(messages_event_trace(
        blocks=[{"type": "text", "text": "Fresh-process reply"}],
        stop_reason="end_turn",
    ))) + "\n"
    completed = _run_isolated(
        _REQUESTS_OPERATIONS,
        {
            "organization": organization,
            "operation": operation,
            "model": "claude-sonnet-4-5" if organization == "anthropic" else "unregistered-model",
            "root": root,
            "response": messages_response(text="Fresh-process reply"),
            "sse": sse,
        },
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == [
        f"sdk_path={_SOURCE_ROOT / 'llm_api_adapter' / 'universal_adapter.py'}",
        "isolated requests operation passed",
    ]


@pytest.mark.unit
@pytest.mark.parametrize("organization", ("anthropic_compatible", "anthropic"))
@pytest.mark.parametrize(
    "operation", ("httpx_chat", "httpx_stream", "async_chat", "async_stream")
)
def test_missing_optional_httpx_errors_only_after_selected_operation(organization, operation):
    """Selected HTTPX-backed operations fail with their extra guidance and no I/O."""
    root = (
        "https://messages.example.test/v1"
        if organization == "anthropic_compatible"
        else "https://api.anthropic.com/v1"
    )
    completed = _run_isolated(
        _MISSING_OPTIONAL_OPERATION,
        {
            "organization": organization,
            "operation": operation,
            "model": "claude-sonnet-4-5" if organization == "anthropic" else "unregistered-model",
            "root": root,
            "response": messages_response(),
            "sse": "",
        },
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == [
        f"sdk_path={_SOURCE_ROOT / 'llm_api_adapter' / 'universal_adapter.py'}",
        "isolated missing optional operation passed",
    ]
