"""Deterministic routing checks for the shared live E2E profiles."""

import pytest

from tests.e2e import conftest as e2e_conftest


@pytest.mark.unit
def test_xai_profile_is_package_scoped_and_uses_grok_4_6_for_bounded_transports(
    monkeypatch,
):
    profile = e2e_conftest._XAI_E2E_PROFILE
    models = ["grok-4.3", "grok-4.5", "grok-4.6", "grok-build-0.1"]
    organization = e2e_conftest.E2EOrganization(
        {
            "name": "xai",
            "models": models,
            "latest_model": e2e_conftest._profile_default_model(
                profile,
                "xai",
                models,
            ),
        }
    )

    assert profile.organization_names == ("xai",)
    assert list(e2e_conftest.iter_organization_models.__wrapped__([organization])()) == [
        (organization, model) for model in models
    ]
    assert e2e_conftest._select_latest_e2e_models(
        [organization],
        "SYNC_HTTPX_E2E",
    ) == [(organization, "grok-4.6")]

    monkeypatch.setenv("SYNC_HTTPX_E2E_XAI_MODEL", "grok-4.5")
    assert e2e_conftest._select_latest_e2e_models(
        [organization],
        "SYNC_HTTPX_E2E",
    ) == [(organization, "grok-4.5")]


@pytest.mark.unit
def test_xai_profile_rejects_a_bounded_transport_override_for_an_unknown_model(
    monkeypatch,
):
    organization = e2e_conftest.E2EOrganization(
        {
            "name": "xai",
            "models": ["grok-4.3", "grok-4.6"],
            "latest_model": "grok-4.6",
        }
    )
    monkeypatch.setenv("ASYNC_E2E_XAI_MODEL", "not-a-grok")

    with pytest.raises(pytest.UsageError, match="not registered for xai"):
        e2e_conftest._select_latest_e2e_models([organization], "ASYNC_E2E")


@pytest.mark.unit
def test_xai_profile_requires_its_api_key_when_the_package_is_installed(
    monkeypatch,
):
    monkeypatch.setattr(e2e_conftest, "version", lambda distribution: "0.1.0")
    monkeypatch.setitem(e2e_conftest.API_KEY_ENV, "xai", None)

    with pytest.raises(pytest.UsageError, match="XAI_API_KEY is not configured"):
        e2e_conftest.organizations.__wrapped__(e2e_conftest._XAI_E2E_PROFILE)
