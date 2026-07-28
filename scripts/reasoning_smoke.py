"""Run a stand-alone live reasoning-observability smoke test.

This script is intentionally separate from pytest and CI. It prints reasoning
and visible text as they arrive, and captures every decoded provider SSE event
so a missing normalized reasoning event can be diagnosed against the provider
payload.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional convenience for this script
    load_dotenv = None

from llm_api_adapter.errors import LLMAPIRateLimitError, LLMAPIServerError
from llm_api_adapter.llm_registry.llm_registry import LLM_REGISTRY
from llm_api_adapter.llms.anthropic.sync_client import ClaudeSyncClient
from llm_api_adapter.llms.google.sync_client import GeminiSyncClient
from llm_api_adapter.llms.openai.sync_client import OpenAISyncClient
from llm_api_adapter.models.messages.chat_message import UserMessage
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter


DEFAULT_PROMPT = (
    "Compare a dog and a potato. Identify three non-obvious properties they "
    "share, state a comparison criterion, and briefly justify each conclusion. "
    "Return exactly three numbered points."
)

API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
}
CLIENT_CLASSES = {
    "openai": OpenAISyncClient,
    "anthropic": ClaudeSyncClient,
    "google": GeminiSyncClient,
}
TRANSIENT_ERRORS = (LLMAPIRateLimitError, LLMAPIServerError)


class _LiveStreamPrinter:
    """Print normalized reasoning and visible text as soon as they arrive."""

    def __init__(self) -> None:
        self._channel: str | None = None

    def _select_channel(self, channel: str) -> None:
        if self._channel != channel:
            if self._channel is not None:
                print()
            if channel == "final answer":
                print("-------------")
            print(f"[{channel}] ", end="", flush=True)
            self._channel = channel

    def on_reasoning(self, event: Any) -> None:
        if not event.text:
            return
        self._select_channel("summary")
        print(event.text, end="", flush=True)

    def on_delta(self, text: str) -> None:
        if not text:
            return
        self._select_channel("final answer")
        print(text, end="", flush=True)

    def finish(self) -> None:
        if self._channel is not None:
            print()
            self._channel = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a live reasoning-observability smoke test for one model."
    )
    parser.add_argument("--provider", required=True, choices=sorted(API_KEY_ENV))
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument(
        "--require-reasoning",
        action="store_true",
        help="Exit with code 2 if no normalized reasoning event is returned.",
    )
    parser.add_argument(
        "--dump-raw",
        action="store_true",
        help="Print complete decoded provider events when no reasoning is captured.",
    )
    return parser.parse_args()


def _event_metadata(event: Any) -> dict[str, Any]:
    return {
        "kind": event.kind,
        "index": event.index,
        "text_chars": len(event.text),
        "elapsed_s": event.elapsed_s,
        "delta_s": event.delta_s,
    }


def _raw_event_metadata(event: Any) -> dict[str, Any]:
    event_name = event.get("event")
    done = event.get("done", False)
    data = event.get("data")
    return {
        "event": event_name,
        "done": done,
        "data_type": type(data).__name__,
        "data_keys": (
            sorted(str(key) for key in data)
            if isinstance(data, Mapping)
            else None
        ),
    }


@contextmanager
def _capture_provider_events(
    provider: str,
    raw_events: list[dict[str, Any]],
) -> Iterator[None]:
    client_class = CLIENT_CLASSES[provider]
    original_stream = client_class.stream

    def stream_with_capture(self, *args, **kwargs):
        events = original_stream(self, *args, **kwargs)
        for event in events:
            raw_events.append(
                {
                    "event": event.event,
                    "done": event.done,
                    "data": event.data,
                }
            )
            yield event

    client_class.stream = stream_with_capture
    try:
        yield
    finally:
        client_class.stream = original_stream


def _stream_with_retry(
    adapter: UniversalLLMAPIAdapter,
    request_kwargs: dict[str, Any],
    completed_responses: list[Any],
    observed_deltas: list[str],
    observed_reasoning: list[Any],
    raw_events: list[dict[str, Any]],
    live_output: _LiveStreamPrinter,
) -> list[str]:
    delays = (2, 4, 8)
    for attempt in range(len(delays) + 1):
        try:
            return list(adapter.stream_chat(**request_kwargs))
        except TRANSIENT_ERRORS:
            if attempt == len(delays):
                raise
            completed_responses.clear()
            observed_deltas.clear()
            observed_reasoning.clear()
            raw_events.clear()
            live_output.finish()
            delay = delays[attempt]
            print(f"Transient provider error; retrying in {delay}s...", file=sys.stderr)
            time.sleep(delay)
    return []


def main() -> int:
    args = _parse_args()
    if load_dotenv is not None:
        load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv(API_KEY_ENV[args.provider])
    if not api_key:
        print(
            f"Missing {API_KEY_ENV[args.provider]} for provider {args.provider}.",
            file=sys.stderr,
        )
        return 2

    provider_spec = LLM_REGISTRY.providers.get(args.provider)
    model_spec = provider_spec.models.get(args.model) if provider_spec else None
    expected_reasoning = bool(model_spec and model_spec.is_reasoning)

    adapter = UniversalLLMAPIAdapter(
        organization=args.provider,
        model=args.model,
        api_key=api_key,
    )
    completed_responses: list[Any] = []
    observed_deltas: list[str] = []
    observed_reasoning: list[Any] = []
    raw_events: list[dict[str, Any]] = []
    live_output = _LiveStreamPrinter()

    def on_delta(text: str) -> None:
        observed_deltas.append(text)
        live_output.on_delta(text)

    def on_reasoning(event: Any) -> None:
        observed_reasoning.append(event)
        live_output.on_reasoning(event)

    def on_done(response: Any) -> None:
        completed_responses.append(response)
        live_output.finish()

    request_kwargs: dict[str, Any] = {
        "messages": [UserMessage(args.prompt)],
        "max_tokens": args.max_tokens,
        "timeout_s": args.timeout,
        "capture_reasoning": True,
        "on_delta": on_delta,
        "on_reasoning": on_reasoning,
        "on_done": on_done,
    }
    if expected_reasoning:
        request_kwargs["reasoning_level"] = "high"

    with _capture_provider_events(args.provider, raw_events):
        text_chunks = _stream_with_retry(
            adapter,
            request_kwargs,
            completed_responses,
            observed_deltas,
            observed_reasoning,
            raw_events,
            live_output,
        )

    live_output.finish()

    streamed_text = "".join(text_chunks)
    if not streamed_text.strip():
        print("The provider returned no visible text.", file=sys.stderr)
        return 1
    if len(completed_responses) != 1:
        print("The adapter did not finalize exactly one response.", file=sys.stderr)
        return 1

    response = completed_responses[0]
    report = {
        "provider": args.provider,
        "model": args.model,
        "registry_is_reasoning": expected_reasoning,
        "visible_text_chars": len(streamed_text),
        "reasoning_event_count": len(observed_reasoning),
        "reasoning_events": [_event_metadata(event) for event in observed_reasoning],
        "raw_event_count": len(raw_events),
        "raw_events": [_raw_event_metadata(event) for event in raw_events],
    }

    response_reasoning_mismatch = response.reasoning_events != observed_reasoning
    reasoning_expected = expected_reasoning or args.require_reasoning
    diagnostic_needed = response_reasoning_mismatch or (
        reasoning_expected and not observed_reasoning
    )

    if diagnostic_needed:
        print("REASONING_SMOKE " + json.dumps(report, ensure_ascii=False, sort_keys=True))

    if reasoning_expected and not observed_reasoning:
        if args.dump_raw:
            raw_report = {
                "provider": args.provider,
                "model": args.model,
                "prompt": args.prompt,
                "events": raw_events,
            }
            print(
                "REASONING_SMOKE_RAW "
                + json.dumps(raw_report, default=str, ensure_ascii=False, sort_keys=True)
            )
        else:
            print(
                "No normalized reasoning events captured. Re-run with --dump-raw "
                "to print every decoded provider event."
            )

    if response_reasoning_mismatch:
        print("Final response and callback reasoning events differ.", file=sys.stderr)
        return 1
    if args.require_reasoning and not observed_reasoning:
        print("Reasoning was required but no reasoning event was returned.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
