"""Migrate legacy deterministic provider fixtures at native I/O boundaries."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Any, Optional

import pytest

from llm_api_adapter.llms.transports import SSEEvent
from llm_api_adapter_mistral.adapter import MistralAdapter


def _responses_usage(value: object) -> object:
    if not isinstance(value, Mapping):
        return value
    result: dict[str, Any] = {}
    for source, target in (
        ("prompt_tokens", "input_tokens"),
        ("completion_tokens", "output_tokens"),
        ("total_tokens", "total_tokens"),
    ):
        if source in value:
            result[target] = value[source]
    details = value.get("prompt_tokens_details")
    if "prompt_tokens_details" in value:
        result["input_tokens_details"] = (
            dict(details) if isinstance(details, Mapping) else details
        )
    if "input_tokens" in value or "output_tokens" in value:
        result.update(value)
    return result


def _responses_content(value: object) -> list[dict[str, Any]]:
    if isinstance(value, str):
        return [{"type": "output_text", "text": value}] if value else []
    if not isinstance(value, list):
        return []
    output: list[dict[str, Any]] = []
    for part in value:
        if not isinstance(part, Mapping):
            continue
        if part.get("type") == "text" and isinstance(part.get("text"), str):
            output.append({"type": "output_text", "text": part["text"]})
        elif part.get("type") == "thinking":
            thinking = part.get("thinking")
            snippets = [
                {"type": "reasoning_text", "text": item["text"]}
                for item in thinking or ()
                if isinstance(item, Mapping) and isinstance(item.get("text"), str)
            ]
            if snippets:
                output.append({"type": "reasoning", "content": snippets})
    return output


def _responses_response(value: object) -> object:
    if not isinstance(value, dict) or "choices" not in value:
        return value
    choices = value.get("choices")
    if not isinstance(choices, list) or (choices and not isinstance(choices[0], Mapping)):
        return value
    result: dict[str, Any] = {
        "object": "response",
        "id": value.get("id", "fixture-response"),
        "model": value.get("model"),
        "created_at": value.get("created", 1),
        "status": "completed",
        "output": [],
    }
    choice = (value.get("choices") or [{}])[0]
    if isinstance(choice, Mapping):
        message = choice.get("message")
        if isinstance(message, Mapping):
            content = _responses_content(message.get("content"))
            refusal = message.get("refusal")
            if isinstance(refusal, str) and refusal:
                content.append({"type": "refusal", "refusal": refusal})
            reasoning = [part for part in content if part.get("type") == "reasoning"]
            content = [part for part in content if part.get("type") != "reasoning"]
            if content:
                result["output"].append({
                    "type": "message", "role": "assistant", "status": "completed",
                    "content": content,
                })
            result["output"].extend(reasoning)
            raw_calls = message.get("tool_calls")
            if isinstance(raw_calls, list):
                for call in raw_calls:
                    if not isinstance(call, Mapping):
                        continue
                    function = call.get("function")
                    if isinstance(function, Mapping):
                        result["output"].append({
                            "type": "function_call", "call_id": call.get("id"),
                            "name": function.get("name"),
                            "arguments": function.get("arguments", "{}"),
                        })
        finish = choice.get("finish_reason")
        if finish == "length":
            result["status"] = "incomplete"
            result["incomplete_details"] = {"reason": "max_output_tokens"}
        elif finish == "content_filter":
            refusal_content = [{"type": "refusal", "refusal": "Request refused."}]
            message_output = next((
                item for item in result["output"]
                if item.get("type") == "message"
            ), None)
            if isinstance(message_output, dict):
                message_output.setdefault("content", []).extend(refusal_content)
            else:
                result["output"].append({
                    "type": "message", "role": "assistant",
                    "content": refusal_content,
                })
    usage = _responses_usage(value.get("usage"))
    if isinstance(usage, Mapping):
        result["usage"] = dict(usage)
    return result


class _ResponsesStreamFixtureConverter:
    """Convert each synthetic Chat chunk while retaining request-local state."""

    response_id: str

    def __init__(self) -> None:
        self.text_parts: list[str] = []
        self.reasoning_parts: list[str] = []
        self.tool_calls: dict[int, dict[str, Any]] = {}
        self.model: Optional[str] = None
        self.response_id = "fixture-stream-response"
        self.usage: object = None
        self.terminal: Optional[str] = None

    def convert(self, event: SSEEvent) -> list[SSEEvent]:
        output: list[SSEEvent] = []
        payload = event.data
        if (
            not isinstance(payload, Mapping)
            or "choices" not in payload
            or payload.get("object") == "response"
            or (
                isinstance(payload.get("type"), str)
                and payload["type"].startswith("response.")
            )
        ):
            return [event]
        choices = payload.get("choices")
        if not isinstance(choices, list):
            return [event]
        if any(
            not isinstance(choice, Mapping)
            or (
                choice.get("delta") is not None
                and not isinstance(choice.get("delta"), Mapping)
            )
            for choice in choices
        ):
            return [event]
        model_value = payload.get("model")
        if isinstance(model_value, str):
            self.model = model_value
        id_value = payload.get("id")
        if isinstance(id_value, str):
            self.response_id = id_value
        if "usage" in payload:
            self.usage = _responses_usage(payload.get("usage"))
        event_usage = self.usage if "usage" in payload else None
        choices = payload.get("choices")
        for choice in choices if isinstance(choices, list) else ():
            if not isinstance(choice, Mapping):
                continue
            delta = choice.get("delta")
            if isinstance(delta, Mapping):
                content_parts = _responses_content(delta.get("content"))
                for part in content_parts:
                    if part.get("type") == "output_text":
                        text = part.get("text")
                        if isinstance(text, str):
                            self.text_parts.append(text)
                            data = {"delta": text}
                            if isinstance(event_usage, Mapping):
                                data["usage"] = dict(event_usage)
                            output.append(SSEEvent(event="response.output_text.delta", data=data))
                    elif part.get("type") == "reasoning":
                        for reasoning in part.get("content", []):
                            text = reasoning.get("text")
                            if isinstance(text, str):
                                self.reasoning_parts.append(text)
                                output.append(SSEEvent(event="response.reasoning_text.delta", data={"delta": text}))
                raw_calls = delta.get("tool_calls")
                if isinstance(raw_calls, list):
                    for raw_call in raw_calls:
                        if not isinstance(raw_call, Mapping):
                            continue
                        index = raw_call.get("index", len(self.tool_calls))
                        if not isinstance(index, int):
                            index = len(self.tool_calls)
                        call = self.tool_calls.setdefault(index, {"arguments": ""})
                        if isinstance(raw_call.get("id"), str):
                            call["call_id"] = raw_call["id"]
                        function = raw_call.get("function")
                        if isinstance(function, Mapping):
                            if isinstance(function.get("name"), str):
                                call["name"] = function["name"]
                            arguments = function.get("arguments")
                            if isinstance(arguments, str):
                                call["arguments"] += arguments
            finish = choice.get("finish_reason")
            if isinstance(finish, str):
                self.terminal = finish
        return output

    def finish(self) -> list[SSEEvent]:
        """Emit completion only after a normal source EOF and explicit finish."""
        if self.terminal is None:
            return []
        output_items: list[dict[str, Any]] = []
        if self.text_parts:
            output_items.append({
                "type": "message", "role": "assistant",
                "content": [{"type": "output_text", "text": "".join(self.text_parts)}],
            })
        if self.reasoning_parts:
            output_items.append({
                "type": "reasoning", "content":[
                    {"type": "reasoning_text", "text": text}
                    for text in self.reasoning_parts
                ],
            })
        output_items.extend({"type": "function_call", **call} for _, call in sorted(self.tool_calls.items()))
        if self.terminal == "content_filter" and not any(
            part.get("type") == "refusal"
            for item in output_items
            if isinstance(item, Mapping)
            for part in item.get("content", [])
            if isinstance(part, Mapping)
        ):
            output_items.append({
                "type": "message", "role": "assistant",
                "content": [{"type": "refusal", "refusal": "Request refused."}],
            })
        response: dict[str, Any] = {
            "object": "response", "id": self.response_id, "model": self.model,
            "status": "incomplete" if self.terminal == "length" else "completed",
            "output": output_items,
        }
        if self.terminal == "length":
            response["incomplete_details"] = {"reason": "max_output_tokens"}
        if isinstance(self.usage, Mapping):
            response["usage"] = dict(self.usage)
        return [SSEEvent(
            event="response.incomplete" if self.terminal == "length" else "response.completed",
            data={"response": response},
        )]


def _responses_stream(events: Iterator[SSEEvent]) -> Iterator[SSEEvent]:
    converter = _ResponsesStreamFixtureConverter()
    try:
        for event in events:
            yield from converter.convert(event)
        yield from converter.finish()
    finally:
        close = getattr(events, "close", None)
        if close is not None:
            close()


@pytest.fixture(autouse=True)
def convert_mistral_legacy_wire_fixtures(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Emit controlled Responses fixtures at the adapter's native submission seam."""
    if any(marker.name.startswith("e2e") for marker in request.node.iter_markers()):
        return
    if not any(request.node.get_closest_marker(marker) for marker in ("unit", "integration")):
        return
    original_post = MistralAdapter._post_payload
    original_apost = MistralAdapter._apost_payload
    original_stream = MistralAdapter._stream_payload
    original_astream = MistralAdapter._astream_payload

    def post(self: MistralAdapter, payload: dict[str, Any], timeout_s: Optional[float]) -> dict[str, Any]:
        response = _responses_response(original_post(self, payload, timeout_s))
        if not isinstance(response, dict):
            raise AssertionError("Native fixture response conversion returned a non-object")
        return response

    async def apost(self: MistralAdapter, payload: dict[str, Any], timeout_s: Optional[float]) -> dict[str, Any]:
        response = _responses_response(await original_apost(self, payload, timeout_s))
        if not isinstance(response, dict):
            raise AssertionError("Native fixture response conversion returned a non-object")
        return response

    def stream(self: MistralAdapter, payload: dict[str, Any], timeout_s: Optional[float]) -> Iterator[SSEEvent]:
        return _responses_stream(original_stream(self, payload, timeout_s))

    def astream(self: MistralAdapter, payload: dict[str, Any], timeout_s: Optional[float]) -> AsyncIterator[SSEEvent]:
        async def convert() -> AsyncIterator[SSEEvent]:
            converter = _ResponsesStreamFixtureConverter()
            source = original_astream(self, payload, timeout_s)
            try:
                async for event in source:
                    for converted in converter.convert(event):
                        yield converted
                for converted in converter.finish():
                    yield converted
            finally:
                close = getattr(source, "aclose", None)
                if close is not None:
                    await close()
        return convert()

    monkeypatch.setattr(MistralAdapter, "_post_payload", post)
    monkeypatch.setattr(MistralAdapter, "_apost_payload", apost)
    monkeypatch.setattr(MistralAdapter, "_stream_payload", stream)
    monkeypatch.setattr(MistralAdapter, "_astream_payload", astream)
