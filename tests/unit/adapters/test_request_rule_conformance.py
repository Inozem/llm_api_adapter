"""Conformance checks for registry-backed payload rules at the adapter boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping
from unittest.mock import AsyncMock, Mock, patch
import warnings

import pytest

from src.llm_api_adapter.adapters.anthropic.adapter import AnthropicAdapter
from src.llm_api_adapter.adapters.google.adapter import GoogleAdapter
from src.llm_api_adapter.adapters.openai.adapter import OpenAIAdapter
from src.llm_api_adapter.errors.llm_api_error import ToolChoiceError
from src.llm_api_adapter.llm_registry.llm_registry import (
    DEFAULT_REGISTRY_PATH,
    RegistrySpec,
)
from src.llm_api_adapter.llm_registry.request_rules import (
    RequestRule,
    RequestRuleRegistry,
)
from src.llm_api_adapter.llms.anthropic.async_client import ClaudeAsyncClient
from src.llm_api_adapter.llms.anthropic.sync_client import ClaudeSyncClient
from src.llm_api_adapter.llms.google.async_client import GeminiAsyncClient
from src.llm_api_adapter.llms.google.sync_client import GeminiSyncClient
from src.llm_api_adapter.llms.openai.async_client import OpenAIAsyncClient
from src.llm_api_adapter.llms.openai.sync_client import OpenAISyncClient
from src.llm_api_adapter.models.messages.chat_message import UserMessage
from src.llm_api_adapter.models.tools.tool_spec import ToolSpec


@dataclass(frozen=True)
class _Response:
    data: dict[str, Any]

    def json(self) -> dict[str, Any]:
        return self.data


@dataclass(frozen=True)
class _AdapterProfile:
    adapter_class: type[Any]
    sync_client_class: type[Any]
    async_client_class: type[Any]
    message_key: str
    required_kwargs: Mapping[str, Any]
    public_kwargs_by_payload_path: Mapping[str, str]
    response_factory: Callable[[str, str | None], dict[str, Any]]


@dataclass(frozen=True)
class _PayloadRuleCase:
    organization: str
    model: str
    api_variant: str | None
    rules: tuple[RequestRule, ...]
    profile: _AdapterProfile


@dataclass(frozen=True)
class _ToolChoiceRestrictionProfile:
    adapter_class: type[Any]
    sync_client_class: type[Any]
    required_kwargs: Mapping[str, Any]


@dataclass(frozen=True)
class _ToolChoiceRestrictionCase:
    organization: str
    model: str
    allowed_modes: frozenset[str]
    profile: _ToolChoiceRestrictionProfile


def _openai_response(model: str, api_variant: str | None) -> dict[str, Any]:
    if api_variant == "responses":
        return {
            "id": "resp_123",
            "model": model,
            "created_at": 123,
            "status": "completed",
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            "output": [{
                "type": "message",
                "content": [{"type": "output_text", "text": "ok"}],
            }],
        }
    return {
        "id": "chatcmpl_123",
        "model": model,
        "created": 123,
        "choices": [{
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _anthropic_response(model: str, _: str | None) -> dict[str, Any]:
    return {
        "id": "msg_123",
        "model": model,
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": "ok"}],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def _google_response(model: str, _: str | None) -> dict[str, Any]:
    return {
        "modelVersion": model,
        "candidates": [{
            "content": {"parts": [{"text": "ok"}]},
            "finishReason": "STOP",
        }],
        "usageMetadata": {
            "promptTokenCount": 1,
            "candidatesTokenCount": 1,
            "totalTokenCount": 2,
        },
    }


_ADAPTER_PROFILES = {
    "openai": _AdapterProfile(
        adapter_class=OpenAIAdapter,
        sync_client_class=OpenAISyncClient,
        async_client_class=OpenAIAsyncClient,
        message_key="messages",
        required_kwargs={},
        public_kwargs_by_payload_path={
            "max_tokens": "max_tokens",
            "temperature": "temperature",
            "top_p": "top_p",
        },
        response_factory=_openai_response,
    ),
    "anthropic": _AdapterProfile(
        adapter_class=AnthropicAdapter,
        sync_client_class=ClaudeSyncClient,
        async_client_class=ClaudeAsyncClient,
        message_key="messages",
        required_kwargs={"max_tokens": 64},
        public_kwargs_by_payload_path={"top_p": "top_p"},
        response_factory=_anthropic_response,
    ),
    "google": _AdapterProfile(
        adapter_class=GoogleAdapter,
        sync_client_class=GeminiSyncClient,
        async_client_class=GeminiAsyncClient,
        message_key="contents",
        required_kwargs={},
        public_kwargs_by_payload_path={
            "generationConfig.maxOutputTokens": "max_tokens",
            "generationConfig.temperature": "temperature",
            "generationConfig.topP": "top_p",
        },
        response_factory=_google_response,
    ),
}

_API_VARIANT_CLIENTS = {
    "openai": (OpenAISyncClient, OpenAIAsyncClient),
}

_TOOL_CHOICE_RESTRICTION_PROFILES = {
    "anthropic": _ToolChoiceRestrictionProfile(
        adapter_class=AnthropicAdapter,
        sync_client_class=ClaudeSyncClient,
        required_kwargs={"max_tokens": 64},
    ),
}


def _registered_payload_rule_cases() -> tuple[pytest.ParameterSet, ...]:
    registry = RegistrySpec(path=str(DEFAULT_REGISTRY_PATH))
    transformation_handlers = {
        RequestRuleRegistry.DROP_PARAMETER,
        RequestRuleRegistry.RENAME_PARAMETER,
    }
    cases = []

    for organization, profile in _ADAPTER_PROFILES.items():
        for model_name, model_spec in registry.organizations[organization].models.items():
            rules = tuple(
                rule
                for rule in model_spec.request_rules.rules
                if rule.handler in transformation_handlers
            )
            if not rules:
                continue
            cases.append(
                pytest.param(
                    _PayloadRuleCase(
                        organization=organization,
                        model=model_name,
                        api_variant=model_spec.request_rules.api_variant,
                        rules=rules,
                        profile=profile,
                    ),
                    id=f"{organization}-{model_name}",
                )
            )

    return tuple(cases)


CASES = _registered_payload_rule_cases()


def _registered_api_variant_cases() -> tuple[pytest.ParameterSet, ...]:
    registry = RegistrySpec(path=str(DEFAULT_REGISTRY_PATH))
    cases = []
    for organization, client_classes in _API_VARIANT_CLIENTS.items():
        for model_name, model_spec in registry.organizations[organization].models.items():
            api_variant = model_spec.request_rules.api_variant
            if api_variant is None:
                continue
            for client_class in client_classes:
                cases.append(
                    pytest.param(
                        client_class,
                        model_name,
                        api_variant,
                        id=(
                            f"{organization}-{model_name}-"
                            f"{client_class.__name__.removesuffix('Client').lower()}"
                        ),
                    )
                )
    return tuple(cases)


API_VARIANT_CASES = _registered_api_variant_cases()


def _registered_tool_choice_restriction_cases() -> tuple[pytest.ParameterSet, ...]:
    registry = RegistrySpec(path=str(DEFAULT_REGISTRY_PATH))
    cases = []
    for organization, profile in _TOOL_CHOICE_RESTRICTION_PROFILES.items():
        for model_name, model_spec in registry.organizations[organization].models.items():
            allowed_modes = model_spec.request_rules.allowed_tool_choice_modes
            if allowed_modes is None:
                continue
            cases.append(
                pytest.param(
                    _ToolChoiceRestrictionCase(
                        organization=organization,
                        model=model_name,
                        allowed_modes=allowed_modes,
                        profile=profile,
                    ),
                    id=f"{organization}-{model_name}",
                )
            )
    return tuple(cases)


TOOL_CHOICE_RESTRICTION_CASES = _registered_tool_choice_restriction_cases()


def _make_adapter(case: _PayloadRuleCase):
    return case.profile.adapter_class(api_key="test_api_key", model=case.model)


def _sync_chat(case: _PayloadRuleCase, kwargs: dict[str, Any]):
    transport = Mock(
        return_value=_Response(
            case.profile.response_factory(case.model, case.api_variant)
        )
    )
    with patch.object(case.profile.sync_client_class, "_send_request", new=transport):
        response = _make_adapter(case).chat([UserMessage("hi")], **kwargs)
    return response, transport.call_args.args[1]


async def _async_chat(case: _PayloadRuleCase, kwargs: dict[str, Any]):
    transport = AsyncMock(
        return_value=case.profile.response_factory(case.model, case.api_variant)
    )
    with patch.object(case.profile.async_client_class, "_send_request", new=transport):
        response = await _make_adapter(case).achat([UserMessage("hi")], **kwargs)
    return response, transport.await_args.args[1]


def _payload_path(case: _PayloadRuleCase, path: str) -> str:
    try:
        return case.profile.public_kwargs_by_payload_path[path]
    except KeyError as error:
        raise AssertionError(
            f"No public adapter parameter is defined for registry path {path!r} "
            f"in organization {case.organization!r}."
        ) from error


def _non_default_value(default: Any) -> Any:
    if default is None:
        return 64
    if isinstance(default, bool):
        return not default
    if isinstance(default, (int, float)):
        return 0.2 if default != 0.2 else 0.3
    if isinstance(default, str):
        return f"{default}-unsupported"
    raise AssertionError(f"No non-default test value is defined for {default!r}.")


def _request_kwargs(case: _PayloadRuleCase, *, non_default: bool) -> dict[str, Any]:
    kwargs = dict(case.profile.required_kwargs)
    for rule in case.rules:
        if rule.handler == RequestRuleRegistry.RENAME_PARAMETER:
            kwargs[_payload_path(case, rule.arguments["from"])] = 64
            continue

        path = rule.arguments["path"]
        default = rule.arguments.get("default", 64)
        kwargs[_payload_path(case, path)] = (
            _non_default_value(default) if non_default else default
        )
    return kwargs


def _value_at_path(payload: dict[str, Any], path: str) -> Any:
    target: Any = payload
    for segment in path.split("."):
        assert isinstance(target, dict), f"Expected mapping at {segment!r} in {path!r}."
        target = target[segment]
    return target


def _assert_path_is_absent(payload: dict[str, Any], path: str) -> None:
    target: Any = payload
    *parents, leaf = path.split(".")
    for parent in parents:
        if not isinstance(target, dict) or parent not in target:
            return
        target = target[parent]
    assert not isinstance(target, dict) or leaf not in target


def _assert_payload_rules_applied(
    payload: dict[str, Any],
    case: _PayloadRuleCase,
    kwargs: dict[str, Any],
) -> None:
    for rule in case.rules:
        if rule.handler == RequestRuleRegistry.DROP_PARAMETER:
            _assert_path_is_absent(payload, rule.arguments["path"])
            continue

        source = rule.arguments["from"]
        target = rule.arguments["to"]
        _assert_path_is_absent(payload, source)
        assert _value_at_path(payload, target) == kwargs[_payload_path(case, source)]


def _expected_warning_messages(case: _PayloadRuleCase) -> list[str]:
    return [
        f"Parameter {rule.arguments['path']!r} is not supported for model "
        f"{case.model!r} and will be ignored."
        for rule in case.rules
        if rule.handler == RequestRuleRegistry.DROP_PARAMETER
        and "default" in rule.arguments
    ]


def _message_key(case: _PayloadRuleCase) -> str:
    if case.organization == "openai" and case.api_variant == "responses":
        return "input"
    return case.profile.message_key


@pytest.mark.unit
@pytest.mark.parametrize(
    ("client_class", "model_name", "api_variant"),
    API_VARIANT_CASES,
)
def test_client_selects_each_registered_api_variant(
    client_class,
    model_name,
    api_variant,
):
    client = client_class(api_key="test_api_key")

    assert client._should_use_responses_api(model_name) is (api_variant == "responses")


def _tool_choice_arguments(mode: str) -> tuple[list[ToolSpec] | None, str]:
    tool = ToolSpec(name="conformance_tool", json_schema={"type": "object"})
    if mode == "tool":
        return [tool], tool.name
    if mode == "any":
        return [tool], mode
    return None, mode


def _make_tool_choice_adapter(case: _ToolChoiceRestrictionCase):
    return case.profile.adapter_class(api_key="test_api_key", model=case.model)


@pytest.mark.unit
@pytest.mark.parametrize("case", TOOL_CHOICE_RESTRICTION_CASES)
def test_adapter_accepts_each_registered_tool_choice_mode(case):
    adapter = _make_tool_choice_adapter(case)
    with patch.object(
        case.profile.sync_client_class,
        "chat_completion",
        return_value=_anthropic_response(case.model, None),
    ) as chat_completion:
        for mode in case.allowed_modes:
            tools, tool_choice = _tool_choice_arguments(mode)
            response = adapter.chat(
                [UserMessage("hi")],
                tools=tools,
                tool_choice=tool_choice,
                **case.profile.required_kwargs,
            )

    assert response.content == "ok"
    assert chat_completion.call_count == len(case.allowed_modes)


@pytest.mark.unit
@pytest.mark.parametrize("case", TOOL_CHOICE_RESTRICTION_CASES)
def test_adapter_rejects_all_unregistered_tool_choice_modes(case):
    adapter = _make_tool_choice_adapter(case)
    with patch.object(
        case.profile.sync_client_class,
        "chat_completion",
    ) as chat_completion:
        for unsupported_mode in ("auto", "none", "any", "tool"):
            if unsupported_mode in case.allowed_modes:
                continue
            tools, tool_choice = _tool_choice_arguments(unsupported_mode)
            with pytest.raises(
                ToolChoiceError,
                match="does not support forced tool use",
            ):
                adapter.chat(
                    [UserMessage("hi")],
                    tools=tools,
                    tool_choice=tool_choice,
                    **case.profile.required_kwargs,
                )

    chat_completion.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("case", CASES)
async def test_adapter_chat_silently_applies_registered_payload_defaults(case):
    kwargs = _request_kwargs(case, non_default=False)
    with warnings.catch_warnings(record=True) as sync_warnings:
        warnings.simplefilter("always")
        sync_response, sync_payload = _sync_chat(case, dict(kwargs))
    with warnings.catch_warnings(record=True) as async_warnings:
        warnings.simplefilter("always")
        async_response, async_payload = await _async_chat(case, dict(kwargs))

    assert sync_response.content == async_response.content == "ok"
    assert sync_payload == async_payload
    assert sync_payload["model"] == case.model
    assert sync_payload[_message_key(case)]
    _assert_payload_rules_applied(sync_payload, case, kwargs)
    assert sync_warnings == []
    assert async_warnings == []


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("case", CASES)
async def test_adapter_chat_warns_for_registered_non_default_payload_values(case):
    kwargs = _request_kwargs(case, non_default=True)
    with warnings.catch_warnings(record=True) as sync_warnings:
        warnings.simplefilter("always")
        sync_response, sync_payload = _sync_chat(case, dict(kwargs))
    with warnings.catch_warnings(record=True) as async_warnings:
        warnings.simplefilter("always")
        async_response, async_payload = await _async_chat(case, dict(kwargs))

    assert sync_response.content == async_response.content == "ok"
    assert sync_payload == async_payload
    _assert_payload_rules_applied(sync_payload, case, kwargs)
    expected_messages = _expected_warning_messages(case)
    assert [str(warning.message) for warning in sync_warnings] == expected_messages
    assert [str(warning.message) for warning in async_warnings] == expected_messages
