"""The local/remote scope, seen from the dashboard (D1–D7).

The form has always asked for the side — two "language model" entries, two model
selectors — but the answer died in `type_plateforme`, which squashed both into
`passerelle`. The file only kept the model name, and the experiment could switch from one
side to the other along the way.

The thread of these tests: what is DISPLAYED of a run must come from the run, not from
today's providers.yaml. `exp_qwen38-27b_minper_jtir_t0_nosim`, served 100 % by Groq on
2026-09-09, was displayed `groq · local` because an LM Studio instance carrying the same
model identifier was declared the next day.
"""

import json
import sys
from pathlib import Path

import pytest
import yaml

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE))

from scripts.dashboard import experiences  # noqa: E402

PROVIDERS = {
    "providers": {
        "groq_qwen_key1": {
            "adapter": "groq",
            "base_url": "https://api.groq.com/openai/v1",
            "default_model": "qwen/qwen3.8-27b",
        },
        "lmstudio_qwen_key1": {
            "adapter": "openai_compatible",
            "base_url": "http://host.docker.internal:1234/v1",
            "default_model": "qwen/qwen3.8-27b",
        },
        "lmstudio_muse_key1": {
            "adapter": "openai_compatible",
            "base_url": "http://host.docker.internal:1234/v1",
            "default_model": "meta/muse-glimmer",
        },
        "google_g35_key1": {
            "adapter": "google",
            "base_url": "https://generativelanguage.googleapis.com",
            "default_model": "gemini-3.5-flash-lite",
        },
    }
}


@pytest.fixture
def providers(tmp_path, monkeypatch):
    chemin = tmp_path / "providers.yaml"
    chemin.write_text(yaml.safe_dump(PROVIDERS), encoding="utf-8")
    monkeypatch.setattr(experiences, "PROVIDERS_YAML", chemin)
    for cache in (
        experiences._FAMILLES_CACHE,
        experiences._PORTEE_CACHE,
        experiences._FOURNISSEUR_EXEC_CACHE,
    ):
        cache.clear()
    yield chemin
    for cache in (
        experiences._FAMILLES_CACHE,
        experiences._PORTEE_CACHE,
        experiences._FOURNISSEUR_EXEC_CACHE,
    ):
        cache.clear()


# ── D1–D2: the form choice survives the write ────────────────────────────────


def test_D1_le_bord_choisi_est_ecrit_dans_le_fichier():
    """Without this field, `instances_pour_modele` then widens again to both sides."""
    assert experiences.portee_plateforme("passerelle_local") == "local"
    assert experiences.portee_plateforme("passerelle_distant") == "distant"
    # Outside the gateway there is no side to name.
    for choix in ("aleatoire", "duree_minimale", "rejeu", "antigravity", "modele"):
        assert experiences.portee_plateforme(choix) is None, choix


def test_D2_la_portee_ecrite_fait_foi_a_la_relecture(providers):
    """The definition states its side; it is not guessed again from the day's providers.yaml."""
    locaux = experiences.modeles_par_portee()[0]
    c = experiences.choix_decideur
    assert c("passerelle", "qwen/qwen3.8-27b", locaux, "local") == "passerelle_local"
    assert c("passerelle", "qwen/qwen3.8-27b", locaux, "distant") == "passerelle_distant"
    # A decision-maker without an LLM keeps its type, scope or not.
    assert c("aleatoire", None, locaux, None) == "aleatoire"


def test_D3_sans_portee_un_modele_des_deux_bords_se_range_du_cote_distant(providers):
    """The old heuristic tested `local` first: a Groq archive reopened as "local"."""
    locaux = experiences.modeles_par_portee()[0]
    c = experiences.choix_decideur
    assert c("passerelle", "qwen/qwen3.8-27b", locaux, None) == "passerelle_distant"
    # A model served ONLY locally stays local: the heuristic is not broken.
    assert c("passerelle", "meta/muse-glimmer", locaux, None) == "passerelle_local"
    assert c("passerelle", "gemini-3.5-flash-lite", locaux, None) == "passerelle_distant"


# ── D4–D5: what is displayed of a definition ─────────────────────────────────


def test_D4_le_fournisseur_affiche_suit_la_portee(providers):
    f = experiences.fournisseur_de
    m = "qwen/qwen3.8-27b"
    assert f({"type": "passerelle", "modele": m, "portee": "local"}) == "local"
    assert f({"type": "passerelle", "modele": m, "portee": "distant"}) == "groq"


def test_D5_sans_portee_l_ambiguite_est_dite_et_non_tranchee(providers):
    """`groq · local` read like a composite provider — which does not exist."""
    rendu = experiences.fournisseur_de({"type": "passerelle", "modele": "qwen/qwen3.8-27b"})
    assert rendu.startswith(experiences.MARQUE_AMBIGU)
    assert "ou" in rendu and "groq" in rendu and "local" in rendu
    # Two REMOTE families stay a list: interchangeable quota buckets, not
    # two quantizations — there is nothing to decide.
    assert experiences.fournisseur_de(
        {"type": "passerelle", "modele": "gemini-3.5-flash-lite"}
    ) == "google"


# ── D6–D7: what is displayed of a RUN ────────────────────────────────────────


def _execution(dossier: Path, instances: list[str]) -> Path:
    dossier.mkdir(parents=True, exist_ok=True)
    lignes = [
        json.dumps({"fournisseur": i, "methode": "llm"}, ensure_ascii=False)
        for i in instances
    ]
    (dossier / "decisions.jsonl").write_text("\n".join(lignes) + "\n", encoding="utf-8")
    return dossier


def test_D6_le_fournisseur_d_une_execution_est_lu_dans_ses_decisions(
    tmp_path, providers
):
    """The real case: an archive served by Groq alone, a providers.yaml that has become ambiguous since."""
    d = _execution(tmp_path / "exec", ["groq_qwen_key1"] * 3)
    assert experiences.fournisseur_execute(d) == "groq"
    # …whereas the definition alone can only note the ambiguity.
    assert experiences.fournisseur_de(
        {"type": "passerelle", "modele": "qwen/qwen3.8-27b"}
    ).startswith(experiences.MARQUE_AMBIGU)


def test_D7_une_execution_qui_a_change_de_bord_le_montre(tmp_path, providers):
    """A mix is not hidden: it is exactly what we want to see."""
    d = _execution(tmp_path / "exec", ["groq_qwen_key1", "lmstudio_qwen_key1"])
    assert experiences.fournisseur_execute(d) == "groq · local"
    # Decisions without a provider (single choice, no solution): we don't know, we say so.
    vide = tmp_path / "vide"
    vide.mkdir()
    (vide / "decisions.jsonl").write_text(
        json.dumps({"methode": "choix_unique"}) + "\n", encoding="utf-8"
    )
    assert experiences.fournisseur_execute(vide) is None
    assert experiences.fournisseur_execute(tmp_path / "inexistant") is None


def test_D7b_une_instance_disparue_reste_lisible(tmp_path, providers):
    """An archive must be readable even when the instance that served it is no longer declared."""
    d = _execution(tmp_path / "exec", ["groq_modele_retire_key1", "lmstudio_vieux_key1"])
    assert experiences.fournisseur_execute(d) == "groq · local"


def test_D8_modele_local_de_respecte_portee_distante(providers):
    """A model served on both sides must not be seen as local if the scope is remote."""
    exp_distante = {
        "decideur": {"type": "passerelle", "modele": "qwen/qwen3.8-27b", "portee": "distant"}
    }
    assert experiences.modele_local_de(exp_distante) is None

    exp_locale = {
        "decideur": {"type": "passerelle", "modele": "qwen/qwen3.8-27b", "portee": "local"}
    }
    assert experiences.modele_local_de(exp_locale) == "qwen/qwen3.8-27b"


def test_D9_parallelisme_conseille_respecte_portee_distante(providers):
    """A model served on both sides must not receive the local constraints if the scope is remote."""
    conseil_distant = experiences.parallelisme_conseille("qwen/qwen3.8-27b", None, portee="distant")
    assert conseil_distant is None or not conseil_distant.get("local")

    conseil_local = experiences.parallelisme_conseille("qwen/qwen3.8-27b", None, portee="local")
    assert conseil_local is not None and conseil_local.get("local") is True

