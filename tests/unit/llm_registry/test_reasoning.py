from dataclasses import replace

import pytest

from src.llm_api_adapter.errors.config_errors import LLMReasoningLevelError
from src.llm_api_adapter.llm_registry.llm_registry import (
    CategoricalReasoningCapability,
    DEFAULT_REGISTRY_PATH,
    RegistrySpec,
)
from src.llm_api_adapter.llm_registry.reasoning import (
    CANONICAL_REASONING_LEVELS,
    ReasoningResolution,
    resolve_reasoning_level,
)


@pytest.fixture(scope="module")
def registry():
    return RegistrySpec(path=str(DEFAULT_REGISTRY_PATH))


def _model(registry, organization, model_name):
    return registry.organizations[organization].models[model_name]


@pytest.mark.unit
def test_canonical_reasoning_scale_is_public_and_ordered():
    assert CANONICAL_REASONING_LEVELS == (
        "none",
        "minimal",
        "low",
        "medium",
        "high",
        "very_high",
    )


@pytest.mark.unit
def test_categorical_string_uses_direct_match_before_projection(registry):
    resolution = resolve_reasoning_level(
        _model(registry, "openai", "gpt-5.5"),
        "high",
    )

    assert resolution == ReasoningResolution(
        provider_value="high",
        reason="categorical_direct_match",
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("model_name", "reasoning_level"),
    [
        ("gpt-5.5", "xhigh"),
        ("gpt-5.6-sol", "max"),
    ],
)
def test_categorical_string_accepts_provider_native_levels_outside_canonical_scale(
    registry,
    model_name,
    reasoning_level,
):
    resolution = resolve_reasoning_level(
        _model(registry, "openai", model_name),
        reasoning_level,
    )

    assert resolution == ReasoningResolution(
        provider_value=reasoning_level,
        reason="categorical_direct_match",
    )


@pytest.mark.unit
def test_categorical_string_projects_canonical_level_with_upward_rounding(registry):
    resolution = resolve_reasoning_level(
        _model(registry, "openai", "gpt-5.5"),
        "minimal",
    )

    assert resolution == ReasoningResolution(
        provider_value="low",
        reason="categorical_canonical_projection",
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("reasoning_level", "expected_value"),
    [
        ("minimal", "low"),
        ("very_high", "high"),
    ],
)
def test_categorical_projection_without_none_uses_the_working_scale(
    registry,
    reasoning_level,
    expected_value,
):
    resolution = resolve_reasoning_level(
        _model(registry, "google", "gemini-3.1-pro-preview"),
        reasoning_level,
    )

    assert resolution == ReasoningResolution(
        provider_value=expected_value,
        reason="categorical_canonical_projection",
    )


@pytest.mark.unit
def test_categorical_none_falls_back_to_minimum_when_disable_is_unavailable(registry):
    resolution = resolve_reasoning_level(
        _model(registry, "google", "gemini-3.1-pro-preview"),
        "none",
    )

    assert resolution.provider_value == "low"
    assert resolution.reason == "none_to_minimum"
    assert "cannot disable reasoning" in resolution.warning


@pytest.mark.unit
@pytest.mark.parametrize(
    ("reasoning_level", "expected_value"),
    [
        (0, "none"),
        (1, "low"),
        (262_500, "low"),
        (262_501, "medium"),
        (525_000, "medium"),
        (525_001, "high"),
        (787_500, "high"),
        (787_501, "xhigh"),
        (1_050_000, "xhigh"),
        (2_000_000, "xhigh"),
    ],
)
def test_categorical_integer_uses_virtual_context_window_percentage(
    registry,
    reasoning_level,
    expected_value,
):
    resolution = resolve_reasoning_level(
        _model(registry, "openai", "gpt-5.5"),
        reasoning_level,
    )

    assert resolution.provider_value == expected_value
    assert resolution.reason == "categorical_context_window_percentage"
    assert isinstance(resolution.provider_value, str)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("reasoning_level", "expected_value"),
    [
        (0, "low"),
        (1, "low"),
        (349_525, "low"),
        (349_526, "medium"),
        (699_050, "medium"),
        (699_051, "high"),
    ],
)
def test_categorical_integer_without_none_uses_the_working_scale(
    registry,
    reasoning_level,
    expected_value,
):
    resolution = resolve_reasoning_level(
        _model(registry, "google", "gemini-3.1-pro-preview"),
        reasoning_level,
    )

    assert resolution.provider_value == expected_value
    assert resolution.reason == "categorical_context_window_percentage"


@pytest.mark.unit
def test_categorical_projection_needs_no_provider_specific_mapping(registry):
    model = replace(
        _model(registry, "openai", "gpt-5.5"),
        reasoning_capability=CategoricalReasoningCapability(
            allowed_values=("conserve", "focus"),
        ),
    )

    resolution = resolve_reasoning_level(model, "high")

    assert resolution.provider_value == "focus"
    assert resolution.reason == "categorical_canonical_projection"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("reasoning_level", "expected_budget"),
    [
        ("minimal", 128),
        ("low", 8_288),
        ("medium", 16_448),
        ("high", 24_608),
        ("very_high", 32_768),
    ],
)
def test_numeric_string_interpolates_actual_budget(
    registry,
    reasoning_level,
    expected_budget,
):
    resolution = resolve_reasoning_level(
        _model(registry, "google", "gemini-2.5-pro"),
        reasoning_level,
    )

    assert resolution == ReasoningResolution(
        provider_value=expected_budget,
        reason="numeric_canonical_interpolation",
    )


@pytest.mark.unit
def test_numeric_none_disables_only_when_zero_is_supported(registry):
    disabled = resolve_reasoning_level(
        _model(registry, "google", "gemini-2.5-flash"),
        "none",
    )
    fallback = resolve_reasoning_level(
        _model(registry, "google", "gemini-2.5-pro"),
        "none",
    )

    assert disabled == ReasoningResolution(
        provider_value=0,
        reason="numeric_disable",
    )
    assert fallback.provider_value == 128
    assert fallback.reason == "none_to_minimum"
    assert "cannot disable reasoning" in fallback.warning


@pytest.mark.unit
def test_numeric_thinking_capabilities_preserve_disable_values(registry):
    disabled = resolve_reasoning_level(
        _model(registry, "google", "gemini-2.5-flash-lite"),
        "none",
    )
    explicit_disabled = resolve_reasoning_level(
        _model(registry, "google", "gemini-2.5-flash-lite"),
        0,
    )

    assert disabled == ReasoningResolution(
        provider_value=0,
        reason="numeric_disable",
    )
    assert explicit_disabled == ReasoningResolution(
        provider_value=0,
        reason="numeric_disable",
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("provider", "model_name"),
    [
        ("google", "gemini-2.5-pro"),
        ("openai", "gpt-5.6-sol"),
    ],
)
def test_negative_reasoning_budget_is_rejected(registry, provider, model_name):
    with pytest.raises(LLMReasoningLevelError, match="non-negative"):
        resolve_reasoning_level(_model(registry, provider, model_name), -1)


@pytest.mark.unit
def test_numeric_integer_preserves_budget_and_clamps_only_below_minimum(registry):
    minimum_fallback = resolve_reasoning_level(
        _model(registry, "google", "gemini-2.5-pro"),
        127,
    )
    provider_native_maximum_error = resolve_reasoning_level(
        _model(registry, "google", "gemini-2.5-pro"),
        32_769,
    )

    assert minimum_fallback.provider_value == 128
    assert minimum_fallback.reason == "numeric_minimum_fallback"
    assert "below the model minimum" in minimum_fallback.warning
    assert provider_native_maximum_error == ReasoningResolution(
        provider_value=32_769,
        reason="numeric_budget",
    )


@pytest.mark.unit
def test_numeric_integer_preserves_declared_budget_boundaries(registry):
    minimum = resolve_reasoning_level(
        _model(registry, "google", "gemini-2.5-pro"),
        128,
    )
    maximum = resolve_reasoning_level(
        _model(registry, "google", "gemini-2.5-pro"),
        32_768,
    )
    zero = resolve_reasoning_level(
        _model(registry, "google", "gemini-2.5-flash"),
        0,
    )

    assert minimum == ReasoningResolution(
        provider_value=128,
        reason="numeric_budget",
    )
    assert maximum == ReasoningResolution(
        provider_value=32_768,
        reason="numeric_budget",
    )
    assert zero == ReasoningResolution(
        provider_value=0,
        reason="numeric_budget",
    )


@pytest.mark.unit
def test_non_reasoning_model_returns_disabled_resolution_with_warning(registry):
    resolution = resolve_reasoning_level(
        _model(registry, "openai", "gpt-4.1"),
        "high",
    )

    assert resolution.provider_value is None
    assert resolution.reason == "unsupported_model"
    assert "does not support reasoning" in resolution.warning


@pytest.mark.unit
@pytest.mark.parametrize("reasoning_level", [True, 1.5, "unknown"])
def test_reasoning_resolver_rejects_invalid_public_levels(registry, reasoning_level):
    with pytest.raises(LLMReasoningLevelError, match="Invalid reasoning level"):
        resolve_reasoning_level(
            _model(registry, "openai", "gpt-5.5"),
            reasoning_level,
        )
