"""The open tab lives in the URL: a refresh comes back to where we were.

An F5 opens a fresh Streamlit session — all client state is lost, and the page
fell back to `Vue d'ensemble` whatever tab was being viewed. The only marker that
survives a refresh is the address bar: that is what these tests check, in
both directions (the URL opens the right tab, a tampered slug does not break the page).
"""

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

RACINE = Path(__file__).resolve().parents[2]
APP = RACINE / "scripts" / "dashboard" / "app.py"


def slug_dans_l_url(at: AppTest) -> str | None:
    """The slug as the address bar would carry it.

    The harness returns raw query params — a list of values per key, where
    `st.query_params` on the application side returns the last value. We normalise here.
    """
    valeur = at.query_params.get("onglet")
    return valeur[-1] if isinstance(valeur, list) else valeur


@pytest.fixture
def lancer(tmp_path_factory):
    """Returns an `AppTest` run with the wanted query string, without touching the real disk.

    Same precaution as `test_dashboard_app.py`: the experiment form and the registry
    filters are remembered in files, which the harness must neither read nor overwrite.
    """
    sys.path.insert(0, str(RACINE))
    from scripts.dashboard import experiences

    origine, origine_vue = experiences.ETAT_FORMULAIRE, experiences.ETAT_VUE_REGISTRE
    experiences.ETAT_FORMULAIRE = tmp_path_factory.mktemp("brouillon") / "formulaire.yaml"
    experiences.ETAT_VUE_REGISTRE = tmp_path_factory.mktemp("vue") / "tableau.yaml"

    def _lancer(onglet: str | None, ouvert: str | None = None) -> AppTest:
        at = AppTest.from_file(str(APP), default_timeout=240)
        if onglet is not None:
            at.query_params["onglet"] = onglet
        if ouvert is not None:  # what the browser sends back when the user has clicked
            at.session_state["onglet_actif"] = ouvert
        at.run()
        assert not at.exception, "\n".join(str(e.value) for e in at.exception)
        return at

    yield _lancer
    experiences.ETAT_FORMULAIRE = origine
    experiences.ETAT_VUE_REGISTRE = origine_vue


@pytest.mark.parametrize(
    "slug, libelle",
    [("providers", "🤖 Providers"), ("experiences", "🧪 Expériences"), ("vue", "🏠 Vue d'ensemble")],
)
def test_l_url_ouvre_l_onglet_qu_elle_nomme(lancer, slug, libelle):
    assert lancer(slug).session_state["onglet_actif"] == libelle


@pytest.mark.parametrize("onglet", [None, "nimportequoi", ""])
def test_sans_slug_utilisable_la_page_revient_a_l_accueil(lancer, onglet):
    """A missing, empty or tampered URL is not an error: it is the home page."""
    assert lancer(onglet).session_state["onglet_actif"] == "🏠 Vue d'ensemble"


def table_onglets() -> list[tuple[str, str]]:
    """Reads the `ONGLETS` table in the source without running the Streamlit script.

    The public tabs are a literal of `app.py`; the private ones, when their module is present,
    come from `onglets_prives.ONGLETS`.
    """
    arbre = ast.parse(APP.read_text(encoding="utf-8"))
    for noeud in arbre.body:
        cibles = noeud.targets if isinstance(noeud, ast.Assign) else []
        if any(isinstance(c, ast.Name) and c.id == "ONGLETS" for c in cibles):
            publics = [tuple(paire) for paire in ast.literal_eval(noeud.value)]
            break
    else:
        raise AssertionError("`ONGLETS` not found in app.py: the URL no longer drives anything")
    if importlib.util.find_spec("scripts.dashboard.onglets_prives") is None:
        return publics
    from scripts.dashboard import onglets_prives

    return onglets_prives.inserer(publics)


def test_les_slugs_couvrent_tous_les_onglets_dessines(lancer):
    """No tab may remain without a slug: it would be impossible to return to it via the URL."""
    at = lancer(None)
    libelles_dessines = [t.label for t in at.tabs]
    declares = table_onglets()
    assert set(libelles_dessines) == {libelle for _, libelle in declares}, (
        "a drawn tab is missing from the ONGLETS table of app.py, or the reverse "
        f"(drawn: {libelles_dessines} — declared: {[lib for _, lib in declares]})"
    )
    slugs = [slug for slug, _ in declares]
    assert all(slugs) and len(set(slugs)) == len(slugs), \
        f"two tabs can neither share a slug nor go without one: {slugs}"


def test_l_onglet_ouvert_se_recopie_dans_l_url(lancer):
    """Clicking a tab updates the URL: without that, there would be nothing to reload."""
    at = lancer(None, ouvert="📊 Métriques")
    assert slug_dans_l_url(at) == "metriques"


def test_l_url_perimee_est_corrigee_par_l_onglet_reellement_ouvert(lancer):
    """The URL follows the tab, never the reverse: the page state is authoritative."""
    at = lancer("vue", ouvert="🤖 Providers")
    assert slug_dans_l_url(at) == "providers"


def test_sans_ses_onglets_prives_la_page_se_sert_de_ses_onglets_publics(lancer, monkeypatch):
    """The public copy does not ship `onglets_prives.py`: the page must then draw its public
    tabs, with no error and no empty tab."""
    monkeypatch.setattr(importlib.util, "find_spec", _sans_module_prive(importlib.util.find_spec))
    at = lancer(None)
    libelles = [t.label for t in at.tabs]
    assert "🧪 Expériences" in libelles and "📊 Métriques" in libelles, libelles
    assert not {"🧬 Calibration", "🎫 Tickets", "🗂️ Mes travaux"} & set(libelles), libelles
    assert not any("Calibration" in str(m.value) for m in at.markdown), "calibration light left on"


def _sans_module_prive(origine):
    """`find_spec` as the public copy answers it: the private dashboard modules do not exist."""
    def find_spec(nom, *args, **kwargs):
        if nom in ("scripts.dashboard.onglets_prives", "scripts.dashboard.tickets_par_experience"):
            return None
        return origine(nom, *args, **kwargs)

    return find_spec
