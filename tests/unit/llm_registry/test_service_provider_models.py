import pytest

from src.llm_api_adapter.llm_registry.llm_registry import (
    ModelRouteSpec,
    OrganizationModelMetadata,
    RegistrySpec,
    ServiceProviderModelMetadata,
    resolve_model_spec,
    resolve_service_provider_model,
)

_UNSET = object()


def _model_data(*, context_window_tokens=1_048_576, input_per_1m=0.15):
    return {
        "limits": {
            "context_window_tokens": context_window_tokens,
            "max_output_tokens": 128_000,
        },
        "pricing_tiers": [
            {
                "up_to_prompt_tokens": None,
                "input_per_1m": input_per_1m,
                "cache_read_input_per_1m": 0.03,
                "output_per_1m": 0.50,
            }
        ],
        "reasoning_capability": {
            "allowed_values": ["low", "high", "max"],
        },
        "capability_exceptions": [],
    }


def _route_metadata(
    *,
    organization="zai",
    service_provider="mistral",
    model="glm-5.3-flash",
    provider_model="zai-glm-5-3-flash",
    model_data=_UNSET,
):
    return ServiceProviderModelMetadata(
        organization=organization,
        service_provider=service_provider,
        model=model,
        provider_model=provider_model,
        model_data=_model_data() if model_data is _UNSET else model_data,
        currency="USD",
    )


def _owner_metadata(*routes):
    return OrganizationModelMetadata(
        organization="mistral",
        organization_data={"models": {"mistral-small-2603": _model_data()}},
        model_routes=tuple(routes),
    )


@pytest.mark.unit
def test_service_provider_route_parses_a_structured_model_profile():
    registry = RegistrySpec()

    assert registry.register_organization_metadata(_owner_metadata(_route_metadata()))

    route = resolve_service_provider_model(
        registry,
        "zai",
        "mistral",
        "glm-5.3-flash",
    )
    assert isinstance(route, ModelRouteSpec)
    assert route.organization == "zai"
    assert route.service_provider == "mistral"
    assert route.model == "glm-5.3-flash"
    assert route.provider_model == "zai-glm-5-3-flash"
    assert route.model_spec.name == "glm-5.3-flash"
    assert route.model_spec.limits.context_window_tokens == 1_048_576
    assert route.model_spec.limits.max_output_tokens == 128_000
    assert route.model_spec.reasoning_capability.allowed_values == (
        "low",
        "high",
        "max",
    )
    assert route.model_spec.pricing_tiers.tiers[0].in_per_token == pytest.approx(
        0.15 / 1_000_000
    )
    assert route.model_spec.pricing_tiers.tiers[0].cache_read_in_per_token == pytest.approx(
        0.03 / 1_000_000
    )
    assert route.model_spec.pricing_tiers.tiers[0].out_per_token == pytest.approx(
        0.50 / 1_000_000
    )
    assert route.model_spec.capability_exceptions == ()


@pytest.mark.unit
@pytest.mark.parametrize(
    "field_name",
    ("organization", "service_provider", "model", "provider_model"),
)
def test_route_rejects_empty_identifiers(field_name):
    route_kwargs = {
        "organization": "zai",
        "service_provider": "mistral",
        "model": "glm-5.3-flash",
        "provider_model": "zai-glm-5-3-flash",
        "model_data": _model_data(),
    }
    route_kwargs[field_name] = ""
    registry = RegistrySpec()

    with pytest.raises(ValueError):
        registry.register_organization_metadata(
            _owner_metadata(ServiceProviderModelMetadata(**route_kwargs, currency="USD"))
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "invalid_model_data",
    (
        None,
        {"limits": {"context_window_tokens": 0, "max_output_tokens": 10}},
        {
            "limits": {"context_window_tokens": 100, "max_output_tokens": 10},
            "pricing_tiers": [],
        },
    ),
)
def test_route_rejects_invalid_model_data(invalid_model_data):
    registry = RegistrySpec()

    with pytest.raises(ValueError):
        registry.register_organization_metadata(
            _owner_metadata(_route_metadata(model_data=invalid_model_data))
        )


@pytest.mark.unit
def test_route_provider_must_match_the_owning_catalogue():
    registry = RegistrySpec()

    with pytest.raises(ValueError):
        registry.register_organization_metadata(
            _owner_metadata(_route_metadata(service_provider="zai"))
        )

    assert "mistral" not in registry.organizations
    assert not registry.service_provider_models


@pytest.mark.unit
def test_service_provider_resolver_uses_exact_route_identity_without_fallback():
    registry = RegistrySpec()
    direct_data = {
        "models": {"glm-5.3-flash": _model_data(input_per_1m=9)}
    }
    assert registry.register_organization_metadata(
        OrganizationModelMetadata(organization="zai", organization_data=direct_data)
    )
    assert registry.register_organization_metadata(
        _owner_metadata(_route_metadata())
    )

    route = resolve_service_provider_model(
        registry, "zai", "mistral", "glm-5.3-flash"
    )
    assert route is not None
    assert route.model_spec.pricing_tiers.tiers[0].in_per_token == pytest.approx(
        0.15 / 1_000_000
    )
    assert resolve_service_provider_model(
        registry, "missing-organization", "mistral", "glm-5.3-flash"
    ) is None
    assert resolve_service_provider_model(
        registry, "zai", "missing-provider", "glm-5.3-flash"
    ) is None
    assert resolve_service_provider_model(
        registry, "zai", "mistral", "missing-model"
    ) is None
    # The direct Z.ai catalogue does not become a Mistral route by name alone.
    assert resolve_service_provider_model(
        registry, "zai", "zai", "glm-5.3-flash"
    ) is None


@pytest.mark.unit
def test_registering_identical_route_metadata_is_idempotent():
    registry = RegistrySpec()
    metadata = _owner_metadata(_route_metadata())

    assert registry.register_organization_metadata(metadata) is True
    assert registry.register_organization_metadata(metadata) is False
    assert len(registry.service_provider_models) == 1


@pytest.mark.unit
def test_route_conflict_does_not_partially_publish_a_batch():
    registry = RegistrySpec()
    original = _route_metadata()
    assert registry.register_organization_metadata(_owner_metadata(original)) is True
    original_route = resolve_service_provider_model(
        registry, "zai", "mistral", "glm-5.3-flash"
    )
    routes_before = dict(registry.service_provider_models)

    conflicting_route = _route_metadata(provider_model="zai-glm-5-3-flash-v2")
    additional_route = _route_metadata(
        model="another-flash-model",
        provider_model="another-flash-model",
    )
    with pytest.raises(ValueError):
        registry.register_organization_metadata(
            _owner_metadata(conflicting_route, additional_route)
        )

    assert registry.service_provider_models == routes_before
    assert resolve_service_provider_model(
        registry, "zai", "mistral", "glm-5.3-flash"
    ) == original_route
    assert resolve_service_provider_model(
        registry, "zai", "mistral", "another-flash-model"
    ) is None


@pytest.mark.unit
def test_direct_and_routed_models_with_the_same_name_remain_independent():
    registry = RegistrySpec()
    direct_metadata = OrganizationModelMetadata(
        organization="zai",
        organization_data={
            "models": {"glm-5.3-flash": _model_data(input_per_1m=9)}
        },
    )
    assert registry.register_organization_metadata(direct_metadata) is True
    assert registry.register_organization_metadata(
        _owner_metadata(_route_metadata())
    ) is True

    direct_model = resolve_model_spec(registry, "zai", "glm-5.3-flash")
    routed_model = resolve_service_provider_model(
        registry, "zai", "mistral", "glm-5.3-flash"
    )
    assert direct_model is not None
    assert routed_model is not None
    assert direct_model.name == routed_model.model
    assert direct_model.pricing_tiers.tiers[0].in_per_token == pytest.approx(
        9 / 1_000_000
    )
    assert routed_model.model_spec.pricing_tiers.tiers[0].in_per_token == pytest.approx(
        0.15 / 1_000_000
    )
