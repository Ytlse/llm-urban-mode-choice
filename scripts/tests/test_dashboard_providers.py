"""Tests of the dashboard Providers pane: filtering out providers with RPD = 0."""

import os
from unittest.mock import MagicMock, patch

import pandas as pd

# Importing `app` runs the dashboard. Under `DASHBOARD_EAGER`, an import outside the Streamlit
# runtime draws the forms on the main generator and pollutes every following
# AppTest. This import only tests three functions: it must therefore stay lazy.
with patch.dict(os.environ, {"DASHBOARD_EAGER": ""}):
    from scripts.dashboard.app import (
        _is_rpd_zero,
        render_overview,
        render_providers_tab,
    )
from scripts.dashboard.live import ApiHealth, ProviderLive
from scripts.dashboard.metrics import DockerStatus, ProvidersStatic


def test_l_import_du_tableau_de_bord_ne_laisse_pas_le_generateur_dans_un_formulaire():
    from streamlit.delta_generator_singletons import get_dg_singleton_instance

    principal = get_dg_singleton_instance().main_dg
    assert principal._form_data is None, (
        "importing `scripts.dashboard.app` marked the main generator as being inside "
        f"a `st.form` ({principal._form_data}): every AppTest of the suite will die of it"
    )


def test_is_rpd_zero():
    """_is_rpd_zero detects RPD limits that are explicitly zero."""
    assert _is_rpd_zero(0) is True
    assert _is_rpd_zero("0") is True
    assert _is_rpd_zero(0.0) is True

    # None = no daily quota (e.g. Mistral, LM Studio local models)
    assert _is_rpd_zero(None) is False
    # Positive quotas
    assert _is_rpd_zero(20) is False
    assert _is_rpd_zero(500) is False
    assert _is_rpd_zero("1000") is False
    # Invalid / non-numeric strings
    assert _is_rpd_zero("illimité") is False
    assert _is_rpd_zero("") is False


def test_render_providers_tab_filtre_rpd_zero_live():
    """When the API is reachable, providers with RPD == 0 are excluded from the table and the metrics."""
    p_actif = ProviderLive(
        name="google_gemini31_key1",
        current_rpm=0,
        rpm_limit=15,
        active_tasks=0,
        usage_pct=0.0,
        cooldown=False,
        daily_requests=10,
        rpd_limit=500,
        daily_tokens=100,
        tpd_limit=1000000,
        quota_exhausted=False,
        available=True,
    )
    p_inutile = ProviderLive(
        name="deprecated_model_key1",
        current_rpm=0,
        rpm_limit=15,
        active_tasks=0,
        usage_pct=0.0,
        cooldown=False,
        daily_requests=0,
        rpd_limit=0,
        daily_tokens=0,
        tpd_limit=0,
        quota_exhausted=False,
        available=False,
    )
    p_sans_limite = ProviderLive(
        name="mistral_key1",
        current_rpm=0,
        rpm_limit=60,
        active_tasks=0,
        usage_pct=0.0,
        cooldown=False,
        daily_requests=25,
        rpd_limit=None,
        daily_tokens=5000,
        tpd_limit=100000000,
        quota_exhausted=False,
        available=True,
    )

    health = ApiHealth(available=True, providers=[p_actif, p_inutile, p_sans_limite])
    static = ProvidersStatic(available=True, providers=[])

    captured_dfs = []

    def mock_dataframe(data, **kwargs):
        if isinstance(data, pd.DataFrame):
            captured_dfs.append(data)

    with patch("scripts.dashboard.app.cached_health", return_value=health), \
         patch("scripts.dashboard.app.cached_providers_static", return_value=static), \
         patch("streamlit.columns", side_effect=lambda n: [MagicMock() for _ in range(n)]), \
         patch("streamlit.dataframe", side_effect=mock_dataframe), \
         patch("streamlit.caption"), \
         patch("streamlit.divider"), \
         patch("streamlit.markdown"), \
         patch("streamlit.checkbox", return_value=False), \
         patch("streamlit.button", return_value=False), \
         patch("scripts.dashboard.app.current_run", return_value=None):
        render_providers_tab()

    assert len(captured_dfs) == 1, "A providers table must be displayed"
    df = captured_dfs[0]
    # The useless provider (RPD = 0) must be excluded
    providers_affiches = list(df["Provider"])
    assert "google_gemini31_key1" in providers_affiches
    assert "mistral_key1" in providers_affiches
    assert "deprecated_model_key1" not in providers_affiches
    assert 0 not in list(df["RPD"])


def test_render_providers_tab_filtre_rpd_zero_statique():
    """When the API is down, providers with rpd_limit == 0 are excluded from the static table."""
    health = ApiHealth(available=False, error="API injoignable")
    static = ProvidersStatic(
        available=True,
        providers=[
            {"name": "google_gemini31_key1", "adapter": "google", "model": "gemini-3.1", "rpm_limit": 15, "tpm_limit": None, "rpd_limit": 500, "tpd_limit": None, "weight": 1.0},
            {"name": "zero_key1", "adapter": "zero", "model": "zero-model", "rpm_limit": 15, "tpm_limit": None, "rpd_limit": 0, "tpd_limit": None, "weight": 0.0},
            {"name": "mistral_key1", "adapter": "mistral", "model": "mistral-small", "rpm_limit": 60, "tpm_limit": 500000, "rpd_limit": None, "tpd_limit": 100000000, "weight": 4.0},
        ],
    )

    captured_dfs = []

    def mock_dataframe(data, **kwargs):
        if isinstance(data, pd.DataFrame):
            captured_dfs.append(data)

    with patch("scripts.dashboard.app.cached_health", return_value=health), \
         patch("scripts.dashboard.app.cached_providers_static", return_value=static), \
         patch("streamlit.warning"), \
         patch("streamlit.dataframe", side_effect=mock_dataframe), \
         patch("streamlit.caption"), \
         patch("streamlit.divider"), \
         patch("streamlit.markdown"), \
         patch("streamlit.columns", side_effect=lambda n: [MagicMock() for _ in range(n)]), \
         patch("streamlit.checkbox", return_value=False), \
         patch("streamlit.button", return_value=False), \
         patch("scripts.dashboard.app.current_run", return_value=None):
        render_providers_tab()

    assert len(captured_dfs) == 1, "The static table must be displayed"
    df = captured_dfs[0]
    adapters_affiches = list(df["Provider"])
    assert "google" in adapters_affiches
    assert "mistral" in adapters_affiches
    assert "zero" not in adapters_affiches
    assert 0 not in list(df["RPD"])


def test_render_overview_filtre_rpd_zero():
    """In the overview, the available-providers counter also filters out RPD == 0."""
    fn = getattr(render_overview, "__wrapped__", render_overview)

    p_actif = ProviderLive(
        name="google_gemini31_key1",
        current_rpm=0,
        rpm_limit=15,
        active_tasks=0,
        usage_pct=0.0,
        cooldown=False,
        daily_requests=10,
        rpd_limit=500,
        daily_tokens=100,
        tpd_limit=1000000,
        quota_exhausted=False,
        available=True,
    )
    p_inutile = ProviderLive(
        name="deprecated_model_key1",
        current_rpm=0,
        rpm_limit=15,
        active_tasks=0,
        usage_pct=0.0,
        cooldown=False,
        daily_requests=0,
        rpd_limit=0,
        daily_tokens=0,
        tpd_limit=0,
        quota_exhausted=False,
        available=False,
    )

    health = ApiHealth(available=True, providers=[p_actif, p_inutile])

    markdown_calls = []

    with patch("scripts.dashboard.app.cached_health", return_value=health), \
         patch("scripts.dashboard.app.cached_run_process", return_value=MagicMock(active=False)), \
         patch("scripts.dashboard.app.cached_docker", return_value=DockerStatus(available=False)), \
         patch("scripts.dashboard.app.current_run", return_value=None), \
         patch("scripts.dashboard.app.cached_git", return_value=MagicMock(available=False)), \
         patch("scripts.dashboard.app.onglets_prives", None), \
         patch("streamlit.columns", side_effect=lambda n: [MagicMock() for _ in range(n)]), \
         patch("streamlit.container", return_value=MagicMock()), \
         patch("streamlit.markdown", side_effect=lambda text, *args, **kwargs: markdown_calls.append(str(text))), \
         patch("streamlit.caption"):
        fn()

    # The counter must read 1/1 available (not 1/2)
    assert any("**1/1** disponibles" in m for m in markdown_calls), markdown_calls
