"""An experiment arm runs with its settings, or it does not run.

Written after the first campaign attempt, which revealed three defects in forty-two
seconds: `make run` does not block, the settings did not reach the container, and nothing
said so. The arm ran at the repository's default values.
"""

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RACINE = Path(__file__).resolve().parents[1]
DEPOT = RACINE.parents[1]
COMPOSE = DEPOT / "infra" / "docker-compose.yml"
COHORTE = DEPOT / "scripts" / "experiment" / "run_sequential_cohort.py"

# Setting → path in `settings`, so that the compose default cannot drift.
REGLAGES = {
    "VEHICLE_CHAIN_ENABLED": "vehicle_chain_enabled",
    "VEHICLE_RETURN_HOME_LOCK": "vehicle_return_home_lock",
    "MODE_CHOICE_TRUNCATION_THRESHOLD": "mode_choice_truncation_threshold",
    "STM_REFLECTION_MIN_ENTRIES": "stm_reflection_min_entries",
    "WEATHER_PER_AGENT_DATES": "weather_per_agent_dates",
    "MEMOIRE__IMPORTANCE_CHOC": "memoire__importance_choc",
    "MEMOIRE__FENETRE_CHANGEMENTS_JOURS": "memoire__fenetre_changements_jours",
    "MEMOIRE__CHANGEMENTS_MAX": "memoire__changements_max",
    # The duration of a shock memory derived from its severity.
    "MEMOIRE__MODE_FENETRE_CHANGEMENTS": "memoire__mode_fenetre_changements",
    "MEMOIRE__SEUIL_SERVICE_CHANGEMENT": "memoire__seuil_service_changement",
    "MEMOIRE__PLANCHER_CHANGEMENT_JOURS": "memoire__plancher_changement_jours",
    "MEMOIRE__PLAFOND_CHANGEMENT_JOURS": "memoire__plafond_changement_jours",
    # The delay severity model.
    "MEMOIRE__RETARD_SATURATION": "memoire__retard_saturation",
    "MEMOIRE__RETARD_GRAVITE_MAX": "memoire__retard_gravite_max",
    "MEMOIRE__RETARD_REF_S": "memoire__retard_ref_s",
    # The sharing within the household, and the guard of decision D7.
    # Added on 2026-09-22: the end-to-end run of the `lu` channel needed to turn the
    # sharing on, and the switch did not reach the container.
    "MEMOIRE__PARTAGE_FOYER_ENABLED": "memoire__partage_foyer_enabled",
    "MEMOIRE__PARTAGE_FOYER_OBSERVATIONS_MIN": "memoire__partage_foyer_observations_min",
    "MEMOIRE__PARTAGE_FOYER_MAX_BLOC": "memoire__partage_foyer_max_bloc",
    # Analysis of 2026-09-25 — the account is given in the evening, once.
    "MEMOIRE__RECIT_SOIR_HEURE": "memoire__recit_soir_heure",
    "MEMOIRE__RECIT_SOIR_ALERTE_PAR_MEMBRE": "memoire__recit_soir_alerte_par_membre",
    "MEMOIRE__ECART_JUGEMENT_ALARME": "memoire__ecart_jugement_alarme",
}


def _defauts_du_compose() -> dict[str, str]:
    texte = COMPOSE.read_text(encoding="utf-8")
    trouves = {}
    for nom in REGLAGES:
        m = re.search(rf"^\s*{nom}:\s*\$\{{{nom}:-(.*?)\}}\s*$", texte, re.M)
        if m:
            trouves[nom] = m.group(1)
    return trouves


# ══════════════════════ L1 — les réglages traversent ════════════════════════════


def test_L1_1_le_compose_declare_chaque_reglage_en_passe_plat():
    """L1.1 — compose only passes to the container what it declares."""
    declares = _defauts_du_compose()
    manquants = sorted(set(REGLAGES) - set(declares))
    assert not manquants, (
        "réglages non déclarés dans infra/docker-compose.yml : "
        + ", ".join(manquants)
        + " — posés sur l'hôte, ils n'atteindraient rien."
    )


@pytest.mark.parametrize("nom,champ", sorted(REGLAGES.items()))
def test_L1_2_le_defaut_du_compose_est_celui_de_settings(nom, champ):
    """L1.2 — deux valeurs par défaut écrites à deux endroits divergent."""
    from settings import settings

    brut = _defauts_du_compose()[nom]
    attendu = getattr(settings.agent, champ)
    if isinstance(attendu, bool):
        obtenu = brut.strip().lower() == "true"
    elif isinstance(attendu, int):
        obtenu = int(brut)
    elif isinstance(attendu, str):
        obtenu = brut.strip()
    else:
        obtenu = float(brut)
    assert obtenu == attendu, (
        f"{nom} vaut {brut!r} dans le compose et {attendu!r} dans settings.py"
    )


def test_L1_3_aucun_prefixe_agent_nulle_part():
    """L1.3 — `AGENT__…` is read by nobody; this script set it for weeks with no effect."""
    for fichier in (COMPOSE, COHORTE):
        if not fichier.is_file():
            continue
        fautifs = re.findall(r"\bAGENT__[A-Z_]+", fichier.read_text(encoding="utf-8"))
        vrais = [f for f in fautifs if "n'est lu par personne" not in f]
        # Mentions in comments are tolerated: they explain the trap.
        lignes = [
            l for l in fichier.read_text(encoding="utf-8").splitlines()
            if "AGENT__" in l and not l.strip().startswith("#")
        ]
        assert not lignes, f"{fichier.name} pose encore un AGENT__ actif :\n" + "\n".join(lignes)


def test_L1_4_le_yaml_prime_sur_l_environnement(monkeypatch):
    """L1.4 — the reason why these keys are removed from config.yaml."""
    import yaml

    agent = (yaml.safe_load((RACINE / "config" / "config.yaml").read_text(encoding="utf-8")) or {}).get("agent") or {}
    poses = {c for c in REGLAGES.values() if c in agent}
    assert not poses, (
        "config.yaml pose encore " + ", ".join(sorted(poses)) + " : aucune variable "
        "d'environnement ne pourra les changer."
    )


# ══════════════════════ L2 and L3 — waiting and checking ══════════════════════


def _cohorte():
    if not COHORTE.is_file():
        pytest.skip("run_sequential_cohort.py absent")
    sys.path.insert(0, str(COHORTE.parent))
    import run_sequential_cohort as rc

    return rc


def test_L2_1_l_orchestrateur_attend_la_disparition_du_lanceur():
    """L2.1 — `make run` returns in 20 s; the launcher lives as long as the run."""
    rc = _cohorte()
    assert hasattr(rc, "attendre_fin_du_run")
    assert "launch_headless.py" in COHORTE.read_text(encoding="utf-8")


def test_L2_4_un_lanceur_deja_en_cours_fait_refuser_le_bras(monkeypatch):
    """L2.4 — otherwise the arm would watch someone else's run, `make run` returning 0."""
    rc = _cohorte()
    monkeypatch.setattr(rc, "lanceur_en_cours", lambda: True)
    code = rc.executer_run_persona("899549", "treated", Path("/tmp/x"), is_resume=False)
    assert code == 3


def test_L3_2_un_reglage_non_conforme_est_nomme(tmp_path):
    """L3.2 — three lost hours are detected in twenty seconds."""
    rc = _cohorte()
    attendus = rc.reglages_attendus()
    # The faulty identity is DEDUCED from the expected values: hard-coded, it would become
    # compliant again the day the campaign changes a setting — and the test would pass for nothing.
    fautive = {
        champ: (not v) if isinstance(v, bool) else (v + 1 if isinstance(v, int) else v + 0.5)
        for champ, v in attendus.items()
    }
    (tmp_path / "identite_run.json").write_text(json.dumps(fautive), encoding="utf-8")
    ecarts = rc.ecarts_de_reglages(tmp_path, attendus)
    assert len(ecarts) == len(attendus), f"{len(ecarts)} écarts pour {len(attendus)} réglages"
    assert any("chaine_vehicules" in e for e in ecarts)
    assert any("seuil_troncature" in e for e in ecarts)


def test_L3_3_des_reglages_conformes_ne_produisent_aucun_ecart(tmp_path):
    """L3.3 — and the agreement is logged: a mute check cannot be told from a missing check."""
    rc = _cohorte()
    attendus = rc.reglages_attendus()
    (tmp_path / "identite_run.json").write_text(json.dumps(attendus), encoding="utf-8")
    assert rc.ecarts_de_reglages(tmp_path, attendus) == []


def test_L3_4_une_identite_absente_est_un_echec_nomme(tmp_path):
    """L3.4 — without an identity, nothing can be asserted about the arm."""
    rc = _cohorte()
    ecarts = rc.ecarts_de_reglages(tmp_path, rc.reglages_attendus())
    assert ecarts and "illisible" in ecarts[0]


def test_L3_5_le_bras_d_ablation_attend_sa_propre_valeur():
    """L3.5 — the expected value follows the variable set, it is not hard-coded."""
    rc = _cohorte()
    attendus = rc.reglages_attendus({"MEMOIRE__FENETRE_CHANGEMENTS_JOURS": "7"})
    assert attendus["fenetre_changements_jours"] == 7
    assert "fenetre_changements_jours" not in rc.reglages_attendus()


# ══════════════════════ L4 — l'identité lue est la bonne ═══════════════════════


def _poser_identite(archive: Path, mtime: float) -> None:
    import os

    archive.mkdir(parents=True, exist_ok=True)
    (archive / "identite_run.json").write_text("{}", encoding="utf-8")
    os.utime(archive / "identite_run.json", (mtime, mtime))


def test_L4_1_une_identite_anterieure_est_ignoree(tmp_path, monkeypatch):
    """L4.1 — celle d'un contrôleur lancé avant `make run` annonce « choc : aucun »."""
    import time as _t

    rc = _cohorte()
    archive = tmp_path / "2026-09-19_16_57"
    _poser_identite(archive, mtime=1000.0)
    monkeypatch.setattr(rc, "archive_du_run", lambda: archive)
    assert rc.attendre_identite(depuis=2000.0, timeout_s=1) is None


def test_L4_2_une_identite_posterieure_est_acceptee(tmp_path, monkeypatch):
    """L4.2 — that of the controller that actually serves the run."""
    rc = _cohorte()
    archive = tmp_path / "2026-09-19_16_58"
    _poser_identite(archive, mtime=3000.0)
    monkeypatch.setattr(rc, "archive_du_run", lambda: archive)
    assert rc.attendre_identite(depuis=2000.0, timeout_s=5) == archive


def test_L4_2b_sans_reference_toute_identite_convient(tmp_path, monkeypatch):
    """L4.2 (continued) — `depuis=None`: we cannot date, we do not refuse."""
    rc = _cohorte()
    archive = tmp_path / "2026-09-19_16_59"
    _poser_identite(archive, mtime=1000.0)
    monkeypatch.setattr(rc, "archive_du_run", lambda: archive)
    assert rc.attendre_identite(depuis=None, timeout_s=5) == archive


def test_L4_4_un_refus_d_ecriture_se_journalise(tmp_path):
    """L4.4 — mute, this refusal cannot be told from a successful write."""
    from loguru import logger

    from urban_mobility_agents.utils import identite_run as I

    lignes: list[str] = []
    sink = logger.add(lambda m: lignes.append(m.record["message"]), level="INFO")
    try:
        assert I.ecrire(tmp_path, {"a": 1}) is True
        assert I.ecrire(tmp_path, {"a": 2}) is False
    finally:
        logger.remove(sink)
    assert any("already set" in m for m in lignes)
