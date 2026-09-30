"""Ticket 118 — the night chain withstands a 4-hour outage without losing anything (2026-09-29).

The outage targeted by the author: « no progress for 4 hours ». The provider no longer
answers, each resume falls back into the same outage on the same simulated day, then it comes back. The chain
must wait (1 h between two attempts), display its pause, and resume the experiment when the
provider comes back, without giving up on the way.

The bench plays the REAL chain (`enchainer_experiences_memoire.sh`) against a fake orchestrator,
in compressed time: ATTENTE_S = 2 s stands for one hour. The fake orchestrator returns 7
(arm suspended, reason `enquete_incomplete`, same simulated day) as long as the outage lasts, then 0.
Duration: about ten seconds per case.

What the bench does not play: GAMA and the controller. That is the object of `make banc-reprise`.

2026-09-29, evening — the chain only retries stops of nature `passager` (overload). The
v6 control of a13 had stopped three times at the same instant on a 409 `rejeu_obligatoire_absent`,
filed as `decision_absente` and retried hour after hour. The cases below replay the REAL
markers of that night: a defect is never retried, an old marker without a nature is
judged by its reason, and when in doubt (no marker, unqualified cause) the chain stops.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CHAINE = REPO_ROOT / "scripts/experiment/enchainer_experiences_memoire.sh"
EXP = "exp_banc_chaine_118"
HEURE_S = 2  # une heure de la vraie chaîne, compressée

FAUX_ORCHESTRATEUR = """
import json, sys, time
from pathlib import Path
racine = Path(sys.argv[0]).resolve().parent
etat = json.loads((racine / "panne.json").read_text())
appels = racine / "appels.log"
appels.write_text(appels.read_text() + f"{time.time()} {' '.join(sys.argv[1:])}\\n" if appels.exists() else f"{time.time()} {' '.join(sys.argv[1:])}\\n")
if time.time() < etat["fin"]:
    if etat["marqueur"] is not None:
        marqueur = racine / "experiments/current/en_attente_quota.json"
        marqueur.write_text(json.dumps(etat["marqueur"]))
    sys.exit(7)
sys.exit(0)
"""

# The marker the controller now writes on a survey question left unanswered
# because the provider is saturated.
SURCHARGE_ENQUETE = {"motif": "enquete_incomplete", "nature": "passager",
                     "cause": "surcharge_fournisseur", "jour_simule": 2, "resume_at": None}


def _orchestrateur_vivant() -> bool:
    motif = "orchestrateur_memoire.py|run_sequential_cohort.py|experiences (lancer|campagne)"
    return subprocess.run(["pgrep", "-f", motif], capture_output=True, check=False).returncode == 0


@pytest.fixture
def racine(tmp_path: Path) -> Path:
    if _orchestrateur_vivant():
        pytest.skip("une campagne tourne : la chaîne refuserait de démarrer (règle d'une seule à la fois)")
    exp = tmp_path / "data/experiences/evenements_non_tabules/banc" / EXP
    exp.mkdir(parents=True)
    (exp / "experience_memoire.yaml").write_text(f"nom: {EXP}\n", encoding="utf-8")
    (exp / "etat.json").write_text('{"etat": "definie"}', encoding="utf-8")
    (tmp_path / "experiments/current").mkdir(parents=True)
    (tmp_path / "faux_orchestrateur.py").write_text(FAUX_ORCHESTRATEUR, encoding="utf-8")
    return tmp_path


def _jouer(
    racine: Path, panne_heures: float, essais_max: int = 6, marqueur: dict | None = SURCHARGE_ENQUETE
) -> dict:
    """Plays the chain; records the pause files seen while it runs."""
    (racine / "panne.json").write_text(
        json.dumps({"fin": time.time() + panne_heures * HEURE_S, "marqueur": marqueur}),
        encoding="utf-8",
    )
    env = dict(
        os.environ,
        RACINE=str(racine),
        PYTHON=sys.executable,
        ORCHESTRATEUR=str(racine / "faux_orchestrateur.py"),
        ATTENTE_S=str(HEURE_S),
        ESSAIS_MAX=str(essais_max),
        ORCHESTRATEUR_ARGS="--branche treated",
    )
    proc = subprocess.Popen(
        ["bash", str(CHAINE), EXP], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    pauses_vues: dict[str, dict] = {}
    while proc.poll() is None:
        for f in (racine / "experiments").glob("enchainement_nuit_*.pause.json"):
            try:
                contenu = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue  # en cours d'écriture
            pauses_vues[f"{contenu['essai']}"] = contenu
        time.sleep(0.1)
    sortie = proc.stdout.read().decode() if proc.stdout else ""
    journal = next((racine / "experiments").glob("enchainement_nuit_*.log")).read_text("utf-8")
    appels = (racine / "appels.log").read_text("utf-8").splitlines()
    return {
        "code": proc.returncode,
        "journal": journal,
        "sortie": sortie,
        "appels": appels,
        "pauses_vues": pauses_vues,
        "pause_restante": list((racine / "experiments").glob("enchainement_nuit_*.pause.json")),
    }


def test_une_panne_de_4_heures_est_tenue_et_lexperience_reprend(racine):
    r = _jouer(racine, panne_heures=4)
    assert r["code"] == 0, r["journal"]
    assert f"✅ {EXP} TERMINÉE" in r["journal"]
    assert "sans jour simulé gagné" not in r["journal"], "la chaîne a abandonné pendant la panne"
    # One attempt at t0, then one per « hour »: 4 or 5 failures depending on the relaunch time.
    echecs = r["journal"].count("⏸ PAUSE")
    assert 3 <= echecs <= 5, r["journal"]
    assert len(r["appels"]) == echecs + 1
    assert all("--branche treated" in a for a in r["appels"]), "ORCHESTRATEUR_ARGS transmis"
    # The pause was seen, with its cause and its day, and it disappeared at the resume.
    assert r["pauses_vues"], "aucun fichier de pause pendant l'attente : l'interface n'aurait rien affiché"
    une = next(iter(r["pauses_vues"].values()))
    assert (une["experience"], une["motif"], une["jour_simule"]) == (EXP, "enquete_incomplete", "2")
    assert r["pause_restante"] == []


def test_au_dela_du_plafond_la_chaine_abandonne_et_le_dit(racine):
    """Counter-check: ESSAIS_MAX bounds the wait. 6 attempts × 1 h cover a little more than 5 h."""
    r = _jouer(racine, panne_heures=100, essais_max=3)
    assert r["code"] == 0  # the chain finishes its queue; it is the experiment that is suspended
    assert "[ALARME]" in r["journal"] and "3 suspensions de suite sans jour simulé gagné" in r["journal"]
    assert f"suspendue(s), à relancer : 1 {EXP}" in r["journal"]
    assert r["pause_restante"] == []


def test_le_plafond_par_defaut_couvre_4_heures():
    """At real scale: 1 h of waiting, 6 attempts without progress — a 4 h outage uses up 5 of them at worst."""
    script = CHAINE.read_text("utf-8")
    assert "ATTENTE_S=${ATTENTE_S:-3600}" in script and "ESSAIS_MAX=${ESSAIS_MAX:-6}" in script
    attente_h, essais_max, panne_h = 1, 6, 4
    echecs_au_pire = panne_h // attente_h + 1  # t0, 1 h, 2 h, 3 h, 4 h
    assert echecs_au_pire < essais_max


# ── What is not retried (2026-09-29) ─────────────────────────────────────────────────

# The two real markers of the night of 29/09 (archive 2026-09-29_17_48, then 11_48), as is.
DECISION_ABSENTE_V6 = {"motif": "decision_absente", "resume_at": None, "replis_consecutifs": 1,
                       "person_id": "1127260", "timestamp": 1774510516, "jour_simule": 11}
CONSOLIDATION_V6 = {"motif": "consolidation_memoire", "resume_at": None, "replis_consecutifs": 0,
                    "person_id": "1320713", "timestamp": 1775574900, "jour_simule": 23,
                    "detail": {"categorie": "stm_reflection",
                               "erreur": "ConsolidationMemoryUnavailable: STM sans résultat pour 1320713"}}
# The same 409, as the fixed controller now writes it.
PREFIXE_CASSE = {"motif": "prefixe_commun", "nature": "defaut", "cause": "rejeu_obligatoire_absent",
                 "person_id": "1127260", "jour_simule": 11, "resume_at": None}


@pytest.mark.parametrize("marqueur", [
    pytest.param(PREFIXE_CASSE, id="409-rejeu-obligatoire-nouveau-marqueur"),
    pytest.param(DECISION_ABSENTE_V6, id="409-de-la-nuit-marqueur-ancien"),
    pytest.param(CONSOLIDATION_V6, id="consolidation-ancienne-sans-cause"),
    pytest.param({"motif": "decision_absente", "nature": "defaut", "cause": None,
                  "jour_simule": 4}, id="decision-absente-non-qualifiee"),
    pytest.param(None, id="aucun-marqueur"),
])
def test_un_defaut_n_est_jamais_retente(racine, marqueur):
    """A single run, no pause, an alarm that says so, and the experiment filed apart.

    Before the fix, each of these cases started again every hour up to ESSAIS_MAX: six
    hours for nothing on the v6 control, which fell back on the same 409 at the same instant.
    """
    r = _jouer(racine, panne_heures=100, marqueur=marqueur)
    assert r["code"] == 0
    assert len(r["appels"]) == 1, r["journal"]
    assert r["pauses_vues"] == {} and r["pause_restante"] == [], "aucun bandeau de pause"
    assert "⏸ PAUSE" not in r["journal"]
    assert "[ALARME] ⛔" in r["journal"] and "DÉFAUT" in r["journal"]
    assert f"arrêtée(s) sur un DÉFAUT, à corriger puis relancer : 1 {EXP}" in r["journal"]
    assert "suspendue(s), à relancer : 0" in r["journal"]


def test_un_marqueur_ancien_de_surcharge_se_retente(racine):
    """Without a `nature` field, the reason `surcharge_fournisseur` states its cause by itself: it is retried."""
    ancien = {"motif": "surcharge_fournisseur", "jour_simule": 5, "resume_at": None}
    r = _jouer(racine, panne_heures=1, marqueur=ancien)
    assert f"✅ {EXP} TERMINÉE" in r["journal"]
    assert len(r["appels"]) >= 2 and "⏸ PAUSE" in r["journal"]
    assert "nature=passager" in r["journal"]


def test_un_quota_du_jour_passe_a_l_experience_suivante_sans_attendre(racine):
    quota = {"motif": "decision_absente", "nature": "quota", "cause": "quota_journalier",
             "jour_simule": 7, "resume_at": "2026-09-30T09:00:00+02:00"}
    r = _jouer(racine, panne_heures=100, marqueur=quota)
    assert len(r["appels"]) == 1
    assert "quota du jour épuisé" in r["journal"]
    assert f"suspendue(s), à relancer : 1 {EXP}" in r["journal"]
    assert r["pauses_vues"] == {}
