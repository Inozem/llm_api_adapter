"""Protocol-neutral validation and diagnostics for compatible API roots."""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import quote, urlsplit

from ..errors.config_errors import LLMConfigError
from ..llm_registry.llm_registry import (
    Pricing,
    PricingTier,
    _finite_non_negative_rate,
    _positive_int,
)


_URI_RE = re.compile(r"(?i)\bhttps?://[^\s<>]+")


def _validate_compatible_pricing(pricing: object) -> Pricing:
    """Validate a caller tariff without exposing its content in diagnostics."""
    error = LLMConfigError(message="pricing must be a valid Pricing object.")
    if not isinstance(pricing, Pricing):
        raise error
    try:
        if not isinstance(pricing.tiers, tuple) or not pricing.tiers:
            raise ValueError
        if not isinstance(pricing.currency, str) or not pricing.currency:
            raise ValueError
        for tier in pricing.tiers:
            if not isinstance(tier, PricingTier):
                raise ValueError
            if tier.up_to_prompt_tokens is not None:
                _positive_int(tier.up_to_prompt_tokens, "up_to_prompt_tokens")
            _finite_non_negative_rate(tier.in_per_token, "in_per_token")
            _finite_non_negative_rate(tier.out_per_token, "out_per_token")
            if tier.cache_read_in_per_token is not None:
                _finite_non_negative_rate(
                    tier.cache_read_in_per_token, "cache_read_in_per_token"
                )
            if tier.cache_write_in_per_token is not None:
                _finite_non_negative_rate(
                    tier.cache_write_in_per_token, "cache_write_in_per_token"
                )
        Pricing._validate_tiers(pricing.tiers)
    except (TypeError, ValueError, OverflowError):
        raise error from None
    return pricing


def _redact_compatible_diagnostic(text: object, api_key: str) -> str:
    """Redact one compatible instance's key and credential-bearing URLs."""
    def redact_url(match: re.Match[str]) -> str:
        original = match.group(0)
        token = original.rstrip(".,;:!?)]")
        suffix = original[len(token):]
        url = token
        try:
            parsed = urlsplit(url)
            has_userinfo = parsed.username is not None or parsed.password is not None
            if has_userinfo or "?" in url or "#" in url:
                return "[redacted URL]" + suffix
        except ValueError:
            return "[redacted URL]" + suffix
        return original

    result = _URI_RE.sub(redact_url, str(text))
    encoded_key = quote(api_key, safe="")
    if "%" in encoded_key:
        pieces = re.split(r"(%[0-9A-Fa-f]{2})", encoded_key)
        pattern = "".join(
            "%"
            + "".join(
                f"[{char.lower()}{char.upper()}]"
                if char.lower() in "abcdef"
                else char
                for char in piece[1:]
            )
            if piece.startswith("%")
            else re.escape(piece)
            for piece in pieces
        )
        result = re.sub(pattern, "[redacted]", result)
    if api_key:
        result = result.replace(api_key, "[redacted]")

    return result


def _validate_ip_literal(netloc: str, hostname: str) -> None:
    """Validate IPv6 and numeric IPv4 literals without resolving hostnames."""
    if netloc.startswith("["):
        literal = netloc[1:netloc.find("]")]
        ipaddress.IPv6Address(literal)
    elif hostname.count(".") == 3 and all(
        part.isdecimal() for part in hostname.split(".")
    ):
        ipaddress.IPv4Address(hostname)


def _validate_api_root(
    base_url: object,
    *,
    resource_paths: tuple[str, ...] = (),
) -> str:
    """Validate and normalize an absolute HTTP(S) API root without I/O.

    Resource suffixes are supplied by the protocol-specific caller; generic
    URL validation does not assume any API's resource paths.
    """
    error = LLMConfigError(
        message="base_url must be an absolute HTTP(S) API root."
    )
    if not isinstance(base_url, str) or not base_url:
        raise error
    if any(
        character.isspace() or ord(character) < 0x20 or ord(character) == 0x7F
        for character in base_url
    ):
        raise error
    if "\\" in base_url or "?" in base_url or "#" in base_url:
        raise error

    try:
        parsed = urlsplit(base_url)
        scheme = parsed.scheme.lower()
        hostname = parsed.hostname
        port = parsed.port
    except (TypeError, ValueError):
        raise error from None

    if (
        scheme not in {"http", "https"}
        or not parsed.netloc
        or not hostname
        or parsed.netloc.endswith(":")
    ):
        raise error
    if "@" in parsed.netloc or parsed.username is not None or parsed.password is not None:
        raise LLMConfigError(message="base_url must not contain user information.")
    if port is not None and not 0 <= port <= 65535:
        raise error

    try:
        _validate_ip_literal(parsed.netloc, hostname)
    except ValueError:
        raise error from None

    if resource_paths and parsed.path.rstrip("/").endswith(resource_paths):
        raise LLMConfigError(
            message="base_url must be an API root, not a resource URL."
        )

    return base_url.rstrip("/")
