"""The repository configuration does not redefine the memory rules.

The specification says nothing changes in the memory rules. On 18 September,
`config.yaml` nevertheless lowered the shock threshold from 0.70 to 0.50 — with no effect on
the targeted shock, which already reached 0.70, its only measurable effect: a wider pool C.
"""

import sys
from pathlib import Path

import pytest
import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.chocs import charger
from llm.gravite import gravite_deterministe
from settings import settings

RACINE = Path(__file__).resolve().parents[1]
CONFIG = RACINE / "config" / "config.yaml"
CONFIG_CHOCS = RACINE / "config" / "chocs"


@pytest.fixture
def journal():
    lignes: list[tuple[str, str]] = []
    sink = logger.add(lambda m: lignes.append((m.record["level"].name, m.record["message"])), level="INFO")
    yield lignes
    logger.remove(sink)


# ══════════════════════ J1 to J4 — the memory rules ════════════════════════════


def test_J1_config_yaml_ne_redefinit_aucune_regle_de_memoire():
    """J1 — a memory rule is varied through the environment, not through the repository default.

    Through the environment, it belongs to the run and shows up in its identity. In
    `config.yaml`, it becomes the repository norm without anyone having decided it — and it
    turns red the tests that state the rule, which makes them unreadable.
    """
    agent = (yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}).get("agent") or {}
    fautives = sorted(k for k in agent if k.startswith("memoire__"))
    assert not fautives, (
        "config.yaml redéfinit des règles de mémoire : "
        + ", ".join(fautives)
        + " — à passer par l'environnement (AGENT__MEMOIRE__…), où le run les enregistre."
    )


def test_J2_le_seuil_de_choc_est_celui_du_071():
    """J2 — 0.70, and Θ equals it. Covered by the memory constants; here it is stated once more."""
    assert settings.agent.memoire__importance_choc == pytest.approx(0.70)
    assert settings.agent.memoire__theta_gravite_cumulee == pytest.approx(0.70)


def test_J3_le_premier_choc_de_c6_franchit_le_seuil_sans_qu_on_l_abaisse():
    """J3 — checked on the COMPUTED value, not on an intention written in a comment."""
    choc = charger(CONFIG_CHOCS / "c6_voiture_suspecte.yaml")
    premier = choc.jours[choc.premier_jour]
    gravite, _ = gravite_deterministe(
        premier.retard_s,
        incident_reseau=premier.incident_reseau,
        correspondance_ratee=premier.correspondance_ratee,
    )
    assert gravite >= settings.agent.memoire__importance_choc, (
        f"severity {gravite:.3f} below the threshold {settings.agent.memoire__importance_choc} — "
        f"the shock would not enter the core memory"
    )


def test_J4_le_second_choc_de_c6_reste_sous_le_seuil_et_c_est_voulu():
    """J4 — original documented behaviour: recalled by the PER-OBJECT pool, not out of context.

    Written so that no one "fixes" it again by lowering the threshold.
    """
    choc = charger(CONFIG_CHOCS / "c6_voiture_suspecte.yaml")
    second = choc.jours[choc.dernier_jour]
    gravite, _ = gravite_deterministe(
        second.retard_s,
        incident_reseau=second.incident_reseau,
        correspondance_ratee=second.correspondance_ratee,
    )
    assert gravite < settings.agent.memoire__importance_choc


# ══════════════════════ J5 — the survey milestones ═════════════════════════════


def _module(monkeypatch, **env):
    from urban_mobility_agents import enquetes

    for cle in ("EXPERIMENT_SURVEY_DAYS",):
        monkeypatch.delenv(cle, raising=False)
    for cle, val in env.items():
        monkeypatch.setenv(cle, val)
    enquetes.reinitialiser()
    return enquetes


def test_J5_1_le_defaut_suit_le_protocole_et_se_journalise(monkeypatch, journal):
    """J5.1 — D12 end of baseline · D17 day after the second shock · D29 window exit · D40 late."""
    enquetes = _module(monkeypatch)
    assert enquetes.jours_jalons() == (12, 17, 29, 40)
    assert any("milestone" in m.lower() for _, m in journal)


def test_J5_2_les_jalons_se_declarent(monkeypatch):
    """J5.2 — a changing protocol must no longer require modifying code."""
    enquetes = _module(monkeypatch, EXPERIMENT_SURVEY_DAYS="3,11")
    assert enquetes.jours_jalons() == (3, 11)


@pytest.mark.parametrize("brut", ["douze", "0", "-4", "3,douze", ",,"])
def test_J5_3_une_declaration_illisible_replie_et_alarme(monkeypatch, journal, brut):
    """J5.3 — a survey silently switched off cannot be told from a survey without results."""
    enquetes = _module(monkeypatch, EXPERIMENT_SURVEY_DAYS=brut)
    assert enquetes.jours_jalons() == (12, 17, 29, 40)
    assert [m for n, m in journal if n == "ERROR" and "[ALARME]" in m]


def test_J5_4_un_jalon_le_week_end_est_signale(monkeypatch, journal):
    """J5.4 — the simulation skips weekends: this milestone would never trigger."""
    enquetes = _module(monkeypatch, EXPERIMENT_SURVEY_DAYS="42")
    enquetes.verifier_jalons_atteignables(jour_courant=1, date_du_jour_courant="2026-03-16")
    signales = [m for _, m in journal if "weekend" in m or "never reachable" in m]
    assert signales, "a milestone falling on a Sunday was reported nowhere"
    assert "42" in signales[0]


def test_J5_5_un_jalon_en_jour_ouvre_ne_signale_rien(monkeypatch, journal):
    """J5.5 — the report only concerns the unreachable."""
    enquetes = _module(monkeypatch, EXPERIMENT_SURVEY_DAYS="12,17,29,40")
    enquetes.verifier_jalons_atteignables(jour_courant=1, date_du_jour_courant="2026-03-16")
    assert not [m for _, m in journal if "weekend" in m or "never reachable" in m]
