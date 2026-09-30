#!/usr/bin/env python3
"""Functional bench of ticket 118 — two days, an outage, a pause, a resume.

    make banc-reprise                 # ≈ 30 min, a single arm, 6 personas, 2 simulated days

What the a13 v5 campaign (2 × 25 days) discovered in the morning, checked in half an hour and
BEFORE relaunching it:

  C1  the simulated outage cuts the survey of milestone D2: the milestone is not written, the run stops;
  C2  the night chain pauses, tells the interface, and relaunches by itself;
  C3  at the resume, the household state is reread from the resume point (O1);
  C4  during the frozen-memory replay, no resume point is rewritten (O1);
  C5  milestone D2 is completed after the outage: no answer lost, none duplicated (O2);
  C6  the reader speaks to their household in English (O7);
  C7  the evening account is never truncated;
  C8  the arm goes to the end.

The outage is the one the author designated: « no progress for 4 hours », in compressed
time — 1 h of waiting of the chain is worth ATTENTE_S = 60 s here, the outage lasts 4 × 60 s.
No question is asked of the model during the outage (`EXPERIMENT_SURVEY_PANNE`). Withstanding
a 4 h outage over several attempts is checked separately, in seconds, by
`scripts/tests/test_chaine_memoire_panne_fournisseur.py`: a GAMA resume replays the run from its start
(≈ 4.5 min per simulated day), so that here the outage is over before the second attempt.

The bench plays the REAL chain (`enchainer_experiences_memoire.sh`), the real orchestrator, GAMA
in headless mode. It calls the model (≈ 200 calls, gemini-3.1-flash-lite). It refuses to
start if a campaign is running (only one at a time).

Outputs: docs/traces/<AAAA-MM-JJ_HH_MM>_banc_reprise_118/ (outside git) — BILAN.md, the journal of
the chain, and the bench experiment itself, removed from data/ so that no night
resumes it. Exit code 0 if all checks pass, 1 otherwise.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

RACINE = Path(__file__).resolve().parents[2]
NOM = "exp_banc_reprise_118"
DOSSIER_EXP = RACINE / "data/experiences/evenements_non_tabules/banc" / NOM
CHAINE = RACINE / "scripts/experiment/enchainer_experiences_memoire.sh"
EXPERIMENTS = RACINE / "experiments"

ATTENTE_S = int(os.getenv("BANC_ATTENTE_S", "60"))  # « une heure » de la chaîne
PANNE_S = 4 * ATTENTE_S                              # « pas d'avancement pendant 4 heures »
JOUR_PANNE = 2
HORIZON = 2

CONFIG = {
    "nom": NOM,
    "canal": "lu",
    "evenement": "a13_punaises_metro__banc_reprise",
    # The models of the a13 v5 campaign: the bench tests the same routing as the measurement.
    "modeles": {
        "itinary_multi_agent": "gemini-3.1-flash-lite",
        "evenement_jugement": "gemini-3.1-flash-lite",
        "stm_reflection": "gemini-3.5-flash-lite",
        "ltm_self_reflection": "gemini-3.1-flash-lite",
        "enquete_affinite": "gemini-3.1-flash-lite",
        "evenement_relais": "gemini-3.1-flash-lite",
    },
    "temperature_decision": 0.0,
    "variante_prompt": "prompt_expert_05",
    "population": "population_6_foyers_a13",
    "horizon_jours": HORIZON,
    "arret_sur_extinction": False,
    "jours_apres_extinction": 5,
    "graine_calendrier": 42,
    "graine_tirage": 42,
    "memoire_importance_choc": 0.7,
    "stm_reflection_min_entries": 5,
    "contrefactuel_ab": True,
    "partage_foyer": True,
    "rejeu_ab": True,
    "adaptateur": "google",
    "prefixe_commun": True,
    "enquetes_quotidiennes": True,
}

MOTS_FRANCAIS = {"le", "la", "les", "des", "est", "une", "que", "pour", "pas", "dans", "sur",
                 "avec", "tu", "vous", "nous", "il", "elle", "j'ai", "c'est", "et", "du", "au"}


def log(msg: str) -> None:
    print(f"{datetime.now().astimezone():%H:%M:%S} [banc-118] {msg}", flush=True)


def orchestrateur_vivant() -> bool:
    motif = "orchestrateur_memoire.py|run_sequential_cohort.py|experiences (lancer|campagne)"
    return subprocess.run(["pgrep", "-f", motif], capture_output=True, check=False).returncode == 0


def empreintes_points(run: Path) -> dict[str, tuple[str, float]]:
    """Fingerprint and date of each resume point of the run."""
    sortie = {}
    for jour in sorted((run / "checkpoints_memoire").glob("jour_*")):
        h = hashlib.sha256()
        derniere = 0.0
        for f in sorted(p for p in jour.rglob("*") if p.is_file()):
            h.update(f.relative_to(jour).as_posix().encode())
            h.update(f.read_bytes())
            derniere = max(derniere, f.stat().st_mtime)
        sortie[jour.name] = (h.hexdigest()[:16], derniere)
    return sortie


def heure_de(ligne: str) -> float | None:
    m = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", ligne)
    # Local wall-clock time of the controller, compared with the host's file dates: same time zone.
    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp() if m else None  # noqa: DTZ007


def part_francaise(texte: str) -> float:
    mots = re.findall(r"[a-zà-ÿ']+", texte.lower())
    return sum(m in MOTS_FRANCAIS for m in mots) / max(1, len(mots))


def preparer() -> None:
    if DOSSIER_EXP.exists():
        reste = DOSSIER_EXP.with_name(f"{NOM}.reste_{datetime.now().astimezone():%Y%m%d_%H%M%S}")
        log(f"un banc précédent traîne dans data/ — déplacé vers {reste.name}, puis retiré")
        DOSSIER_EXP.rename(reste)
        shutil.rmtree(reste)
    DOSSIER_EXP.mkdir(parents=True)
    import yaml

    (DOSSIER_EXP / "experience_memoire.yaml").write_text(
        "# BANC du ticket 118 — engendré par scripts/experiment/banc_reprise_chaine.py, jamais une mesure.\n"
        + yaml.safe_dump(CONFIG, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    (DOSSIER_EXP / "etat.json").write_text(
        json.dumps({"etat": "definie", "cree_le": datetime.now().astimezone().astimezone().isoformat()}),
        encoding="utf-8",
    )


def jouer_la_chaine() -> dict:
    """Launches the night chain on the bench; watches the pause while it happens."""
    avant = set(EXPERIMENTS.glob("enchainement_nuit_*.log"))
    env = dict(
        os.environ,
        ATTENTE_S=str(ATTENTE_S),
        ESSAIS_MAX="4",
        ORCHESTRATEUR_ARGS="--branche treated",
        EXPERIMENT_SURVEY_PANNE=f"{JOUR_PANNE}:{PANNE_S}",
        EXPERIMENT_SURVEY_RETRIES_S="5,5",
    )
    env.pop("MAKEFLAGS", None)
    log(f"chaîne lancée — panne des enquêtes au jalon J{JOUR_PANNE} pendant {PANNE_S} s, "
        f"pause de la chaîne {ATTENTE_S} s")
    proc = subprocess.Popen(["bash", str(CHAINE), NOM], cwd=RACINE, env=env)
    obs: dict = {"pauses": [], "points_pendant_pause": None, "run": None}
    while proc.poll() is None:
        for f in EXPERIMENTS.glob("enchainement_nuit_*.pause.json"):
            try:
                pause = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if pause not in obs["pauses"]:
                obs["pauses"].append(pause)
                run = (EXPERIMENTS / "current").resolve()
                obs["run"] = run
                obs["points_pendant_pause"] = empreintes_points(run)
                log(f"⏸ pause vue : {pause}")
        time.sleep(2)
    obs["code"] = proc.returncode
    nouveaux = sorted(set(EXPERIMENTS.glob("enchainement_nuit_*.log")) - avant)
    obs["journal_chaine"] = nouveaux[-1] if nouveaux else None
    obs["run"] = obs["run"] or (EXPERIMENTS / "current").resolve()
    obs["pause_restante"] = list(EXPERIMENTS.glob("enchainement_nuit_*.pause.json"))
    return obs


def controler(obs: dict) -> list[tuple[str, bool, str]]:
    run: Path = obs["run"]
    # The controller sets its journal aside at each restart (`app.log.<horodatage>.bak`):
    # the attempts read in order, the last one in `app.log`.
    journaux = [*sorted(run.glob("app.log.*.bak")), run / "app.log"]
    app = [
        ligne
        for f in journaux if f.is_file()
        for ligne in f.read_text(encoding="utf-8", errors="replace").splitlines()
    ]
    chaine = obs["journal_chaine"].read_text(encoding="utf-8") if obs["journal_chaine"] else ""
    res: list[tuple[str, bool, str]] = []

    def trouve(motif: str, depuis: int = 0) -> list[int]:
        return [i for i, l in enumerate(app) if i >= depuis and motif in l]

    # C1 — the outage cuts milestone D2, which is not written, and the run stops.
    panne = trouve("[banc] PANNE SIMULÉE")
    incomplet = trouve(f"jalon J{JOUR_PANNE} INCOMPLET")
    hib = trouve(f"Survey of milestone J{JOUR_PANNE}")
    res.append(("C1 panne → jalon non écrit → arrêt ordonné", bool(panne and incomplet and hib),
                f"panne {len(panne)}, jalon incomplet {len(incomplet)}, hibernation {len(hib)}"))

    # C2 — the chain pauses, says so, and relaunches.
    motifs = {p.get("motif") for p in obs["pauses"]}
    ok = bool(obs["pauses"]) and motifs == {"enquete_incomplete"} and not obs["pause_restante"] \
        and chaine.count("▶ ") >= 2
    res.append(("C2 pause affichée puis relance automatique", ok,
                (f"{len(obs['pauses'])} pause(s) vue(s), motifs {sorted(map(str, motifs))}, "
                f"{chaine.count('▶ ')} lancement(s), fichier de pause restant : {len(obs['pause_restante'])}")))

    # C3 — l'état du foyer relu à la reprise, non vide (deux soirs ont été servis).
    restaure = trouve("[reprise] état restauré depuis")
    relu = [app[i] for i in trouve("[foyer] état relu du point de reprise", restaure[0] if restaure else 0)]
    m = re.search(r"(\d+) receveur\(s\) déjà servi", relu[0]) if relu else None
    sans_foyer = trouve("ne porte pas l'état du foyer")
    ok = bool(restaure and relu and m and int(m.group(1)) > 0 and not sans_foyer)
    res.append(("C3 état du foyer relu à la reprise (O1)", ok,
                (relu[0].split(" - ", 1)[-1] if relu else "aucune relecture journalisée")
                + (f" ; {len(sans_foyer)} point(s) sans état du foyer" if sans_foyer else "")))

    # C4 — aucun point réécrit pendant le gel.
    degel = trouve("[reprise] DÉGEL au")
    non_ecrit = trouve("NON écrit : rejeu à mémoire gelée")
    t_rest = heure_de(app[restaure[-1]]) if restaure else None
    t_degel = heure_de(app[degel[-1]]) if degel else None
    apres = empreintes_points(run)
    pendant_pause = obs["points_pendant_pause"] or {}
    reecrits = [j for j, (_h, t) in apres.items() if t_rest and t_degel and t_rest <= t <= t_degel]
    changes = [j for j, (h, _t) in pendant_pause.items() if apres.get(j, (h,))[0] != h]
    ok = bool(degel and non_ecrit and not reecrits and not changes)
    res.append(("C4 aucun point de reprise réécrit pendant le gel (O1)", ok,
                (f"dégel {len(degel)}, « NON écrit » {len(non_ecrit)}, points datés du gel {reecrits}, "
                f"points modifiés depuis la pause {changes}")))

    # C5 — milestone D2 completed, nothing lost, nothing duplicated.
    complet = [app[i] for i in trouve(f"jalon J{JOUR_PANNE} COMPLET")]
    lignes = list(csv.DictReader((run / "affinites_declarees.csv").open(encoding="utf-8"))) \
        if (run / "affinites_declarees.csv").is_file() else []
    par_jour = Counter(l["jour_simule"] for l in lignes)
    cles = Counter((l["jour_simule"], l["persona_id"], l["mode"], l["critere"]) for l in lignes)
    doublons = sum(1 for n in cles.values() if n > 1)
    personas = {l["persona_id"] for l in lignes}
    attendu = len(personas) * 5 * 6
    en_attente = sorted(p.name for p in run.glob("enquete_en_attente_J*.json"))
    ok = bool(complet and "reposée" in complet[0]) and set(par_jour.values()) == {attendu} \
        and len(par_jour) == HORIZON and not doublons and not en_attente
    res.append(("C5 jalon J2 complété après la panne, aucune réponse perdue (O2)", ok,
                (f"lignes par jour {dict(par_jour)} (attendu {attendu}), doublons {doublons}, "
                f"en attente {en_attente}, « {complet[0].split('] ', 1)[-1] if complet else 'jamais complet'} »")))

    # C6 — the relay in English.
    messages = []
    if (run / "relais_foyer.jsonl").is_file():
        for l in (run / "relais_foyer.jsonl").read_text(encoding="utf-8").splitlines():
            if l.strip():
                messages += [m["texte"] for m in json.loads(l).get("messages", []) if m.get("parle")]
    francais = [t for t in messages if part_francaise(t) > 0.15]
    res.append(("C6 relais écrit en anglais (O7)", bool(messages) and not francais,
                (f"{len(messages)} message(s), {len(francais)} en français"
                + (f" : « {francais[0][:80]}… »" if francais else ""))))

    # C7 — jamais de troncature.
    bilans = [app[i] for i in trouve("[foyer] bilan du run")]
    tronque = [l for l in bilans if "rien n'est tronqué" not in l]
    ok = bool(bilans) and not tronque and not trouve("anormalement long")
    res.append(("C7 récit du soir jamais tronqué", ok,
                (f"{len(bilans)} bilan(s) du foyer, {len(tronque)} à l'ancien format ; "
                f"{len(trouve('anormalement long'))} alarme(s) de récit long")))

    # C8 — the arm goes to the end.
    res.append(("C8 le bras va au bout", obs["code"] == 0 and "TERMINÉE" in chaine,
                f"code de la chaîne {obs['code']}"))
    return res


def rendre(obs: dict, res: list, debut: float) -> Path:
    trace = RACINE / "docs/traces" / f"{datetime.now().astimezone():%Y-%m-%d_%H_%M}_banc_reprise_118"
    trace.mkdir(parents=True, exist_ok=True)
    duree = (time.time() - debut) / 60
    lignes = [
        "# Banc du ticket 118 — une panne, une pause, une reprise",
        "",
        (
            f"**{datetime.now().astimezone():%Y-%m-%d %H:%M}**, {duree:.0f} min, run `{obs['run'].name}`, "
            f"panne des enquêtes {PANNE_S} s au jalon J{JOUR_PANNE}, pause de la chaîne {ATTENTE_S} s."
        ),
        "",
        "| Contrôle | Verdict | Relevé |",
        "|---|---|---|",
        *[f"| {n} | {'✅' if ok else '❌'} | {d} |" for n, ok, d in res],
        "",
    ]
    (trace / "BILAN.md").write_text("\n".join(lignes), encoding="utf-8")
    if obs["journal_chaine"]:
        shutil.copy2(obs["journal_chaine"], trace / "chaine.log")
    return trace


def main() -> int:
    if orchestrateur_vivant():
        log("[ALARME] une campagne tourne — banc refusé (une seule à la fois).")
        return 2
    debut = time.time()
    preparer()
    obs: dict = {"run": (EXPERIMENTS / "current").resolve(), "journal_chaine": None}
    res: list[tuple[str, bool, str]] = []
    try:
        obs = jouer_la_chaine()
        res = controler(obs)
    except Exception as err:  # noqa: BLE001 — a bench that crashes is a failed check, not a silence
        res.append(("C0 le banc s'exécute", False, repr(err)))
    finally:
        # Never resumed by a night: the state says « arrêtée », then the bench leaves data/.
        etat = DOSSIER_EXP / "etat.json"
        if etat.is_file():
            d = json.loads(etat.read_text(encoding="utf-8"))
            d.update(etat="arretee", motif="banc du ticket 118 — jamais une mesure")
            etat.write_text(json.dumps(d, indent=2), encoding="utf-8")
    trace = rendre(obs, res, debut)
    shutil.move(str(DOSSIER_EXP), str(trace / NOM))
    for nom, ok, detail in res:
        (log if ok else lambda m: log(f"[ALARME] {m}"))(f"{'✅' if ok else '❌'} {nom} — {detail}")
    echecs = [n for n, ok, _ in res if not ok]
    log(f"{len(res) - len(echecs)}/{len(res)} contrôle(s) passé(s) en {(time.time() - debut) / 60:.0f} min — "
        f"bilan : {trace / 'BILAN.md'}")
    return 1 if echecs else 0


if __name__ == "__main__":
    sys.exit(main())
