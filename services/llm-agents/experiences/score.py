"""Composite scoring of a run — composite score, per-stratum detail, formula replay.

Spec: `specs/scoring_composite_experiences.md`.

The composite score is **not** reimplemented (R1): it comes from the ``Scorer`` of
`scripts/synthesis/frames.py`, which applies the scoring formula
(`scripts/synthesis/formule_score/metrics.py`, repatriated from the calibration engine on
2026-09-29). This module is an **adapter**:
it reads a run's `moves.csv`, feeds it to the same pipeline as the historical
synthesis page, and persists a `scores.json` stamped with a scoring formula (R5).

Replaying a formula is **exact and offline** (R6, R10): the composite score is
recomposed by ``weighted_composite`` from the raw per-dimension scores already
stored, without rereading `moves.csv` or calling any model.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import yaml
from experiences import formule as F
from experiences.chemins import racine_depot
from loguru import logger

# ── Access to the synthesis pipeline and the scoring formula ─────────────────
# `scripts.synthesis.frames` lives at the repo root. We add it to the path once;
# the formula (`scripts/synthesis/formule_score`, repatriated from the calibration engine
# on 2026-09-29) is then imported by `sources.import_formule_score`, as in
# `scripts/synthesis/build.py`.
# Never `parents[2]`: it is `<repo>/services` on the host since ticket 039 and `/`
# in the container. `racine_depot()` anchors on `scripts/synthesis`, correct on both sides.
_RACINE_MODULE = racine_depot()
if str(_RACINE_MODULE) not in sys.path:
    sys.path.insert(0, str(_RACINE_MODULE))

from scripts.synthesis import frames, sources

# Path resolution anchor: the synthesis pipeline's, correct on BOTH sides
# (repo root on the host, `/app` in the container where `llm-agents` is mounted). Derived
# from `__file__`, it was `/` in the container: the survey reference data could not be
# found and any scoring launched in the container — hence at the close of a run —
# failed with FileNotFoundError.
REPO_ROOT = sources.REPO_ROOT

# Rows without a modal decision, excluded from scoring — same list as
# `scripts/synthesis/sources.yaml` (common_set.exclude_selection_methods).
# "LLM Error (Default index)" is a fallback, not a decision (ticket 008).
EXCLURE_METHODES = [
    "Pas de déplacement (même localisation)",
    "Pas de solution de déplacement",
    "LLM Error (Default index)",
]

# Label written to `moves.csv` when only one option existed (`runner.py`, method
# `choix_unique`). It is NOT in `EXCLURE_METHODES`: these rows stay in the
# composite score, which describes the day as it happened. They serve to produce the
# SECOND reading, that of what the decision-maker actually decided (ticket 047).
METHODE_CHOIX_UNIQUE_MOVES = "Un seul itinéraire disponible"

# Beyond this gap between the two readings of the composite score, the published figure
# depends largely on rows nobody decided: we say so in ERROR rather than letting
# someone cite a ranking that the second reading reverses. Threshold set on the measurement of
# 2026-09-12 (16 runs of 11/09, same set and population): chain off, the gap
# caps at 0.32 EMD point; chain on, it ranges from 2.62 to 12.22 and changes the ranking
# of the arms. A threshold of 1.0 separates the two regimes exactly.
ECART_LECTURES_ALARME = 1.0

# ── Ticket 081 — a truncated log never produces a score ──────────────────────
# `moves.csv` is the SUBSTRATE of the composite score; `couverture.decides` counts what the run
# actually decided. The two sat side by side in the same `scores.json` without ever being
# compared. The run of 2026-09-12 was published at 5.35 on 274 rows when its archive
# held 3,299: the figure was cited in three documents before being recognised as wrong (it is
# 6.17 on the rebuilt log).
#
# Threshold set on the sweep of the repo's 38 runs (2026-09-15): a healthy run
# has 3,161 rows for 3,151 to 3,155 decided, i.e. a deficit ALWAYS NEGATIVE, from −0.2 to
# −0.3 % — the log also carries the `sans_solution` rows, which `decides` excludes.
# The faulty run is at +91.3 %. A floor at 2 % leaves a margin of 63 rows out of
# 3,154 and separates the two regimes by a factor of 45: it cannot fire wrongly on
# the known history, and it cannot miss a truncation.
TOLERANCE_JOURNAL = 0.02

# Runs already reported in this process — the alarm fires on the RISING EDGE. Without this,
# `score --toutes` would replay the same ERROR line on every pass and `make error` would drown the
# signal in its own repetition. The name leaves the set as soon as the check passes again, so
# that a log regenerated then truncated again raises the alarm again.
_JOURNAUX_SIGNALES: set[str] = set()


class JournalIncomplet(ValueError):
    """The moves log does not cover the archived decisions: no score.

    Distinct from the R21 refusal ("run not finished") because it calls for a different
    remedy: R21 waits for the end of the run, this one asks to regenerate the log
    (`python -m experiences journal <execution> --regenerer`) and invalidates the `scores.json`
    that would have been written on the truncated log.
    """


# Canonical path of the reference data in the repo (EF-77: read, never copied).
CEREMA_DEPOT = REPO_ROOT / "scripts" / "data" / "population" / "cerema_values.yaml"

F_SYNTHESE = "synthese.json"
F_MOVES = "moves.csv"
F_SCORES = "scores.json"

# Metric name → readable key in scores.json.
_CLE_METRIQUE = {"emd_jsd": "emd_jsd", "l1_composite": "l1"}


def _lire_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _sha256_fichier(p: Path) -> str | None:
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()
    except OSError:
        return None


def est_terminee(synthese: dict) -> bool:
    """R21 — only a run closed as "terminee" is scorable."""
    return (synthese.get("etat") or {}).get("etat") == "terminee"


def volet_pour(synthese: dict) -> str:
    """R15 — `modele` decision-maker → volet 3; any other decision-maker → volet 1."""
    decideur = (synthese.get("empreintes") or {}).get("decideur") or {}
    return "3" if decideur.get("type") == "modele" else "1"


def _resoudre_cerema(synthese: dict) -> Path:
    """Path of the reference data: the cited source if present, else the repo canon."""
    source = ((synthese.get("referentiel") or {}).get("source") or "").strip()
    if source:
        # The source is written container-side ("app/…"): we map it back to the repo.
        rel = source.removeprefix("app/")
        cand = REPO_ROOT / rel
        if cand.exists():
            return cand
    return CEREMA_DEPOT


def scorer_pour(formule: F.Formule) -> tuple[Any | None, str | None]:
    """Builds a ``Scorer`` with the formula's weights. R18: without the engine,
    returns ``(None, message)`` — the caller then writes no scores.json."""
    formule_score, erreur = sources.import_formule_score()
    if formule_score is None:
        return None, erreur
    scorer = frames.Scorer(
        formule_score, formule.poids_scorer(), "emd_jsd", "l1_composite"
    )
    return scorer, None


_MOTIF_COUPE = {
    "aucune": "toutes les décisions du journal, tentative la plus récente",
    "horizon": "premier jour simulé (horizon déclaré au-delà d'un jour), tentative la plus récente",
    "repetitions": "premier jour simulé (couples répétés dans le journal), tentative la plus récente",
    "forcee": "premier jour simulé (coupe imposée par l'appelant), tentative la plus récente",
}


def _libelle_perimetre(stats: dict) -> str:
    """R5 — the scope actually scored, stated plainly next to the count it carries.

    A count placed next to a composite score it does not cover is the error ticket 047
    closed; an announced scope that is no longer the one applied would be its relapse.
    """
    motif = str(stats.get("coupe") or "aucune")
    base = _MOTIF_COUPE.get(motif, _MOTIF_COUPE["aucune"])
    repetes = int(stats.get("couples_repetes") or 0)
    doublons = int(stats.get("exclues_doublon") or 0)
    detail = f" ; {repetes} couple(s) répété(s)" if repetes else ""
    detail += f" ; {doublons} doublon(s) écarté(s)" if doublons else ""
    return f"lecture du score : {base}{detail}"


def _horizon_jours(dossier: Path) -> int | None:
    """Number of days the run declared to cover — `None` if the archive does not say.

    Used by the scope cut criterion (ticket 057, R2): beyond one day, the log
    carries several days and the score keeps only one. Read from `execution.yaml`, which
    freezes the experiment definition at launch — not from `data/experiences/<nom>/`, which
    may have been redefined since.
    """
    try:
        config = yaml.safe_load(
            (dossier / "execution.yaml").read_text(encoding="utf-8")
        )
    except (OSError, yaml.YAMLError):
        return None
    valeur = ((config or {}).get("experience") or {}).get("horizon_jours")
    try:
        return int(valeur)
    except (TypeError, ValueError):
        return None


def _couverture_de(synthese: dict) -> dict:
    """The `couverture` block of a synthesis, wherever it was written."""
    return (
        (synthese.get("parts_modales") or {}).get("couverture")
        or synthese.get("couverture")
        or {}
    )


def mesurer_perimetre(lignes_journal: int, synthese: dict) -> dict:
    """Raw finding, no judgement: does the log cover the archived decisions?

    Kept apart from the rule so that the sweep (`journal --verifier`) can observe without
    raising and without alarming. `complet` is `None` when `couverture.decides` is missing or zero:
    there is then nothing to compare, and saying "complete" would mistake emptiness for a
    check — in this repo, that is the most persistent error pattern.
    """
    decides = (_couverture_de(synthese) or {}).get("decides")
    decides = int(decides) if isinstance(decides, (int, float)) else None
    ecart = None if not decides else decides - lignes_journal
    relatif = None if not decides else ecart / decides
    return {
        "lignes_journal": int(lignes_journal),
        "decides": decides,
        "ecart": ecart,
        "ecart_relatif": relatif,
        "tolerance": TOLERANCE_JOURNAL,
        "complet": None if relatif is None else relatif <= TOLERANCE_JOURNAL,
    }


def verifier_perimetre(dossier: Path, stats: dict, synthese: dict) -> dict:
    """R24 — refuses to score a log that does not cover the archived decisions.

    The count compared is `stats["total"]`, the RAW number of rows in the file, before the
    cut to the first simulated day and before `latest_attempts`: this is indeed the question "was
    the log written in full?", not "how many rows enter the composite score?".

    Raises `JournalIncomplet` below the threshold, after a rising-edge `[ALARME]`.
    """
    constat = mesurer_perimetre(int(stats.get("total") or 0), synthese)
    cle = str(Path(dossier).resolve())
    if constat["complet"] is not False:
        _JOURNAUX_SIGNALES.discard(cle)
        return constat
    if cle not in _JOURNAUX_SIGNALES:
        _JOURNAUX_SIGNALES.add(cle)
        causes = [
            str(i.get("cause"))
            for i in (synthese.get("interruptions") or [])
            if i.get("cause")
        ]
        logger.error(
            f"[ALARME] Journal des mouvements incomplet — "
            f"{synthese.get('experience')}/{synthese.get('execution') or Path(dossier).name} : "
            f"{constat['lignes_journal']} ligne(s) dans moves.csv pour "
            f"{constat['decides']} décision(s) archivées, soit "
            f"{100 * constat['ecart_relatif']:.1f} % de déficit (tolérance "
            f"{100 * TOLERANCE_JOURNAL:.0f} %). Cause probable : "
            + (
                f"interruption(s) {', '.join(causes)} — les décisions resservies à la "
                f"reprise n'écrivaient pas de ligne de journal."
                if causes
                else "aucune interruption consignée ; le journal a été tronqué autrement."
            )
            + " AUCUN score n'est écrit : un composite calculé sur ce journal porterait sur "
            "une fraction du travail sans le dire. Régénérer le journal avec "
            f"`python -m experiences journal {Path(dossier).name} --regenerer`, puis rescorer."
        )
    raise JournalIncomplet(
        f"incomplete log, not scorable (R24): {Path(dossier).name} — "
        f"{constat['lignes_journal']} row(s) for {constat['decides']} archived decisions"
    )


def _detail_par_dimension(frame_attendu: list[dict], cerema: dict) -> dict:
    """Per-stratum detail for the 7 dimensions (R5, R16).

    The `scored:False` dimensions (place of residence, dwelling type) are
    included for display — they carry no weight in the composite score.
    """
    detail: dict[str, dict] = {}
    for dim in frames.DIMENSIONS:
        strates = frames.dimension_detail(frame_attendu, cerema, dim)
        detail[dim["key"]] = {
            "label": dim["label"],
            "scored": dim["scored"],
            "strates": strates,
        }
    return detail


def lire_perimetre(
    dossier: str | Path, exclusions: list[str]
) -> tuple[list[dict], dict]:
    """The scoring scope of a run — **the only place** that decides it (ticket 057).

    Two readings come out of here, the main one and the one excluding single itineraries; tests and
    any analysis that wants to redo the computation go through this function rather than
    re-specifying the cut. It is that duplication that had let the scorer's scope
    diverge from the one the checks believed they reproduced.

    The cut to the first simulated day applies ONLY if there is something to cut (R2/R3):
    declared horizon beyond one day, or repeated pairs in the log. Applied
    systematically, it removed 866 decisions out of 3,299 from runs without a simulator
    for zero duplicates — including 797 morning departures, which
    `jeu.py:deplacements_attendus()` dates to the next day because the origin "home"
    activity spans midnight.
    """
    dossier = Path(dossier)
    return frames.read_moves(
        dossier / F_MOVES,
        exclusions,
        first_day_only="auto",
        horizon_jours=_horizon_jours(dossier),
    )


def calculer(
    dossier: str | Path, formule: F.Formule, scorer: Any | None = None
) -> dict:
    """Scores a finished run under ``formule`` and returns the scores.json content.

    Raises ``ValueError`` if the run is not finished (R21) or if the loss
    engine is unavailable (R18) — in that case the caller must write nothing.
    """
    dossier = Path(dossier)
    synthese = _lire_json(dossier / F_SYNTHESE)
    if not est_terminee(synthese):
        raise ValueError(f"run not finished, not scorable (R21): {dossier.name}")

    if scorer is None:
        scorer, erreur = scorer_pour(formule)
        if scorer is None:
            raise ValueError(f"moteur de loss indisponible (R18) : {erreur}")

    moves = dossier / F_MOVES
    cerema_path = _resoudre_cerema(synthese)
    cerema = frames.load_cerema(cerema_path)
    rows, stats = lire_perimetre(dossier, EXCLURE_METHODES)
    # R24 (ticket 081) — BEFORE any computation: a log that does not cover the archived
    # decisions produces no score. Placed here, the only place that the close,
    # `score --execution` and the global rescoring all go through.
    perimetre = verifier_perimetre(dossier, stats, synthese)

    variants = frames.simulation_frames(rows)
    attendu = variants["attendu"]

    # Raw scores per dimension, for BOTH metrics: they carry each
    # s[dim], which makes formula replay exact and free (R6).
    bruts = scorer.score(attendu, cerema) if attendu else {}
    emd = bruts.get("emd_jsd", {})
    l1 = bruts.get("l1_composite", {})

    # R8 — unmeasured dimensions (fallback to max loss in the engine, never 0):
    # we cite them explicitly next to the composite score.
    non_mesurees: list[str] = []
    if attendu:
        _, mesure = scorer.primary.compute_detailed(
            scorer._pd.DataFrame(attendu), cerema
        )
        non_mesurees = list(mesure.undefined)

    tire = variants.get("tiré") or variants.get("tire")
    composite_tire = None
    if tire:
        bruts_tire = scorer.score(tire, cerema)
        composite_tire = (bruts_tire.get("emd_jsd") or {}).get("composite")

    # ── Second reading: what the decision-maker ACTUALLY decided (ticket 047) ──────
    # Systematic, never on request: the measurement of 2026-09-12 shows that removing the
    # single-itinerary choices moves the composite by −3.75 to +12.22 EMD points depending on
    # the arm, and that it changes the ranking (`lgbm` 4.50 → 10.46 falls behind `klr`). A
    # composite published without its second reading does not say if it rates a
    # decision-maker or a supply.
    rows_hors, _ = lire_perimetre(
        dossier, EXCLURE_METHODES + [METHODE_CHOIX_UNIQUE_MOVES]
    )
    attendu_hors = frames.simulation_frames(rows_hors)["attendu"]
    bruts_hors = scorer.score(attendu_hors, cerema) if attendu_hors else {}
    emd_hors = bruts_hors.get("emd_jsd", {})
    l1_hors = bruts_hors.get("l1_composite", {})

    # The count is over the scope OF THE SCORE (first simulated day, most recent
    # attempt), which is not that of `synthese.json` (all archived decisions):
    # measured on the 16 runs of 11/09, the two differ by 14 to 19 rows on
    # each. Both are therefore published, each naming its scope — a count placed
    # next to a composite score it does not cover is precisely the error this ticket closes.
    n_scorees, n_hors = len(rows), len(rows_hors)
    n_forces = n_scorees - n_hors
    forces_execution = synthese.get("choix_forces") or {}
    choix_forces = {
        "n": n_forces,
        "part": (n_forces / n_scorees) if n_scorees else None,
        "n_scorees": n_scorees,
        "n_hors_choix_unique": n_hors,
        "perimetre": _libelle_perimetre(stats),
        "n_execution": forces_execution.get("n"),
        "part_execution": forces_execution.get("part"),
        "perimetre_execution": "toutes les décisions archivées (synthese.json)",
        "lecture": (
            "décisions à itinéraire unique : une seule option existait, personne n'a choisi. "
            "Elles RESTENT dans le composite principal — la journée a eu lieu — et sont "
            "retirées de la seconde lecture. Leur nombre dépend du bras."
        ),
    }

    couverture = _couverture_de(synthese)

    contenu = {
        "execution": synthese.get("execution") or dossier.name,
        "experience": synthese.get("experience"),
        "volet": volet_pour(synthese),
        "genere_le": frames_now(),
        "formule": {"nom": formule.nom, "sha256": formule.sha256},
        "moves_sha256": _sha256_fichier(moves),
        "referentiel": {
            "source": (synthese.get("referentiel") or {}).get("source"),
            "sha256": _sha256_fichier(cerema_path),
        },
        "couverture": couverture,
        # R24 (ticket 081) — the check that AUTHORISED this score, published with it. A score
        # without this field was computed before the rule: `scores_perimes` declares it stale
        # to force a full computation, once.
        "perimetre_verifie": perimetre,
        "composite": {
            "emd_jsd": emd.get("composite"),
            "l1": l1.get("composite"),
            "emd_jsd_tire": composite_tire,
            # Second reading (ticket 047). `None` and never 0.0 when it has no
            # rows: with no decision left, there is nothing to rate — and in this repo
            # a 0.0 is the PERFECT score, so emptiness would pass itself off as excellence.
            "emd_jsd_hors_choix_unique": emd_hors.get("composite"),
            "l1_hors_choix_unique": l1_hors.get("composite"),
        },
        "scores_bruts": {"emd_jsd": emd, "l1": l1},
        # The raw scores of the second reading too: without them, `rejouer()` could only
        # recompose one of the two composite scores, and a replayed formula would leave the
        # second frozen under the OLD formula — two figures side by side that no longer
        # compare, with nothing saying so.
        "scores_bruts_hors_choix_unique": {"emd_jsd": emd_hors, "l1": l1_hors},
        "choix_forces": choix_forces,
        "dimensions_non_mesurees": non_mesurees,
        "global": frames.global_view(attendu, cerema) if attendu else {},
        "detail": _detail_par_dimension(attendu, cerema) if attendu else {},
        "lecture": dict(stats),
    }
    _journaliser_lectures(contenu)
    return contenu


def _journaliser_lectures(contenu: dict) -> None:
    """Reports success, not only failure — and alerts when the composite rests on forced choices.

    A silent scoring cannot tell "the second reading was computed" from
    "it no longer is". So we always log both composite scores and the count;
    the ERROR fires only when the threshold is crossed, once per scoring.
    """
    comp = contenu.get("composite") or {}
    forces = contenu.get("choix_forces") or {}
    principal, seconde = comp.get("emd_jsd"), comp.get("emd_jsd_hors_choix_unique")
    part = forces.get("part")
    logger.info(
        f"Score {contenu.get('execution')} — composite EMD {_texte(principal)} "
        f"(all decisions, n={forces.get('n_scorees')}) · "
        f"{_texte(seconde)} (excluding single itinerary, n={forces.get('n_hors_choix_unique')}) "
        f"· forced choices {forces.get('n')}"
        + (f" ({100 * part:.1f} %)" if part is not None else "")
    )
    if principal is None or seconde is None:
        return
    ecart = seconde - principal
    if abs(ecart) >= ECART_LECTURES_ALARME:
        logger.error(
            f"[ALARME] Composite dépendant des choix forcés : {contenu.get('execution')} "
            f"({contenu.get('experience')}) passe de {principal:.2f} à {seconde:.2f} "
            f"({ecart:+.2f} points EMD) quand on retire les {forces.get('n')} décisions à "
            f"itinéraire unique, soit {100 * (part or 0):.1f} % du périmètre scoré. "
            f"Le classement des bras n'est pas invariant à ce retrait : citer ce composite "
            f"sans sa seconde lecture attribuerait au décideur ce que l'offre lui imposait."
        )


def _texte(valeur: float | None) -> str:
    """'—' rather than '0.00': in this repo, 0 is the perfect score, not the absence."""
    return "—" if valeur is None else f"{valeur:.2f}"


def _metrics() -> Any:
    """The ``formule_score.metrics`` module (the scoring formula).

    The offline replay builds no ``Scorer``, hence never calls ``scorer_pour``: it imports
    the formula here. Without it, `score --toutes` and the "Recalculer toutes les
    expériences" button stopped at the first already-scored run. No network call (R10).
    """
    formule_score, erreur = sources.import_formule_score()
    if formule_score is None:
        raise ValueError(erreur)
    return formule_score.metrics


def rejouer(scores: dict, formule: F.Formule) -> dict:
    """Recomposes the composite score under a NEW formula, without rereading moves.csv (R6, R10).

    Exact because the composite score is linear and each s[dim] is stored in
    ``scores_bruts``: ``weighted_composite`` redoes the weighted sum. No network
    call, no model.

    Raises ``ValueError`` if the loss engine is missing (R18): the caller writes nothing.
    """
    m = _metrics()

    bruts = scores.get("scores_bruts") or {}
    emd = bruts.get("emd_jsd") or {}
    l1 = bruts.get("l1") or {}
    # Second reading replayed by the SAME path (ticket 047): the two composite scores of a
    # scores.json always carry the same formula. Leaving it behind would produce two
    # figures side by side computed under two weightings, whose gap would no longer measure
    # the forced choices but the change of formula.
    bruts_hors = scores.get("scores_bruts_hors_choix_unique") or {}
    emd_hors = bruts_hors.get("emd_jsd") or {}
    l1_hors = bruts_hors.get("l1") or {}
    poids = formule.poids_scorer()
    nouveau = dict(scores)
    nouveau["formule"] = {"nom": formule.nom, "sha256": formule.sha256}
    nouveau["composite"] = dict(scores.get("composite") or {})
    nouveau["composite"]["emd_jsd"] = m.weighted_composite(emd, poids) if emd else None
    nouveau["composite"]["l1"] = m.weighted_composite(l1, poids) if l1 else None
    nouveau["composite"]["emd_jsd_hors_choix_unique"] = (
        m.weighted_composite(emd_hors, poids) if emd_hors else None
    )
    nouveau["composite"]["l1_hors_choix_unique"] = (
        m.weighted_composite(l1_hors, poids) if l1_hors else None
    )
    nouveau["genere_le"] = frames_now()
    nouveau["rejoue"] = True
    _journaliser_lectures(nouveau)
    return nouveau


def est_reference(scores: dict, registre: F.RegistreFormules) -> bool:
    """R7 — derived on read: is the stored SHA that of the current reference?"""
    return (scores.get("formule") or {}).get("sha256") == registre.reference.sha256


def ecrire(dossier: str | Path, contenu: dict) -> Path:
    """Persists scores.json (deterministic: sorted keys, R10/R5)."""
    p = Path(dossier) / F_SCORES
    p.write_text(
        json.dumps(contenu, ensure_ascii=False, indent=1, sort_keys=True),
        encoding="utf-8",
    )
    return p


def _couronnes_fantomes(scores: dict) -> bool:
    """Ticket 082 — does this `scores.json` carry an untranslated residence ring?

    Exact signature of the fault: the "outside reference data" row of the dimension
    `lieu_residence` carries a key like `1st_ring`, which the old `normalize_place`
    built from an English label and which the reference does not break down. The three
    rings outside Toulouse are then missing from the strata.

    This is not a criterion of appearance — "fewer than four strata" would also occur
    on a legitimately small run. It is the trace of the faulty key itself.
    """
    strates = ((scores.get("detail") or {}).get("lieu_residence") or {}).get("strates")
    for strate in strates or []:
        if strate.get("cat") != frames.OFF_REFERENCE_ROW:
            continue
        if set(strate.get("categories") or {}) & frames.PLACE_CLES_FANTOMES:
            return True
    return False


def scores_perimes(dossier: str | Path) -> bool:
    """R17 — a scores.json computed on a moves.csv that has changed since is stale.

    Ticket 047 — so is one that does NOT carry the second reading. The offline replay
    recomposes the composite scores from the stored raw scores: a file older than the
    ticket has no `scores_bruts_hors_choix_unique`, so a replay would leave it at "not
    measured" forever, and the whole history would stay silent on the quantity this
    ticket makes mandatory. Declaring it stale forces a full computation — once.

    Ticket 082 — so, finally, is one whose residence rings were not translated.
    Same reasoning, same remedy: the replay recomputes ONLY the composite score, never the
    per-stratum detail. Without this criterion, `--toutes` would rewrite the scored v6 runs with
    their pages missing three rings out of four, and the fix would never
    reach a single published file.

    Ticket 081 — so is one that does not carry `perimetre_verifie`, i.e. any
    score written before rule R24 existed. The offline replay never rereads
    `moves.csv`: without this criterion, a score computed on a truncated log would be replayed
    indefinitely under the new formulas, keeping its wrong composite score and never
    triggering the check. The full computation it forces is what puts the
    entire history in front of the safeguard, once.

    Ticket 057 — so, finally, is one whose `lecture` does not say which cut was applied,
    hence any score written when the cut to the first simulated day was systematic. The
    scope itself has changed: these scores carry 2,347 decisions where the log
    counts 3,154, and no formula replay would correct them — it recomposes the composite scores
    from raw scores computed on the old scope. Same remedy as the previous ones:
    a full computation, once.
    """
    dossier = Path(dossier)
    scores = _lire_json(dossier / F_SCORES)
    if not scores:
        return True
    if not (scores.get("scores_bruts_hors_choix_unique") or {}).get("emd_jsd"):
        return True
    if _couronnes_fantomes(scores):
        return True
    if not scores.get("perimetre_verifie"):
        return True
    if not (scores.get("lecture") or {}).get("coupe"):
        return True
    return scores.get("moves_sha256") != _sha256_fichier(dossier / F_MOVES)


def invalider(dossier: str | Path, motif: str) -> Path | None:
    """Withdraws from circulation a `scores.json` that the rule now refuses (R24).

    Renamed, not deleted: the file stays auditable — it is the one that produced the published
    figure, and erasing it would make the correction impossible to trace. The HTML page, on the
    other hand, is deleted: it has no other role than being read, and a rendering that outlives its
    score reads as a valid score.
    """
    dossier = Path(dossier)
    source = dossier / F_SCORES
    if not source.is_file():
        return None
    cible = dossier / "scores.invalide.json"
    contenu = _lire_json(source)
    contenu["invalide"] = {"motif": motif, "le": frames_now()}
    cible.write_text(
        json.dumps(contenu, ensure_ascii=False, indent=1, sort_keys=True),
        encoding="utf-8",
    )
    source.unlink()
    (dossier / "synthese_scores.html").unlink(missing_ok=True)
    logger.error(
        f"[ALARME] Score invalidated — {dossier.name}: {motif}. Published composite "
        f"{(contenu.get('composite') or {}).get('emd_jsd')} discarded; the old file is "
        f"kept as {cible.name} for audit, the synthesis page is deleted."
    )
    return cible


def score_execution(
    dossier: str | Path,
    formule: F.Formule | None = None,
    registre: F.RegistreFormules | None = None,
    scorer: Any | None = None,
) -> Path | None:
    """Scores a finished run and writes scores.json. Returns the path, or None
    if the run is not scorable (R21) or the engine is missing (R18)."""
    dossier = Path(dossier)
    registre = registre or F.charger()
    formule = formule or registre.reference
    try:
        contenu = calculer(dossier, formule, scorer=scorer)
    except JournalIncomplet as exc:
        # R24 — refusing is not enough: a `scores.json` computed BEFORE the rule is still
        # on disk, readable by the page and the dashboard as if it were valid. That is
        # exactly what happened on 2026-09-12. We withdraw it from circulation.
        invalider(dossier, str(exc))
        return None
    except ValueError as exc:
        logger.info(f"[score] {dossier.name} not scored: {exc}")
        return None
    chemin = ecrire(dossier, contenu)
    logger.info(
        f"[score] {contenu['experience']}/{contenu['execution']} scored under "
        f"formula {formule.nom} ({formule.sha256[:12]}): "
        f"composite emd_jsd={contenu['composite']['emd_jsd']}"
    )
    return chemin


def scorer_a_la_cloture(dossier: str | Path) -> Path | None:
    """Scores a run that has just been closed, and writes its page (R21, R23).

    Called by the runner right after the synthesis. **Fail-open by construction**:
    it never raises. Scoring is a rendering — offline, ~0.2 s for 2,700
    decisions — and its failure must not turn a successful run into a failure.
    When it fails, the run stays `terminee` without `scores.json`, the table
    shows "—", and the "Recalculer toutes les expériences" button remains the
    fallback.

    Does nothing if the run is not `terminee` (R21): a stopped or paused
    run is not scorable, even if largely filled.
    """
    dossier = Path(dossier)
    synthese = _lire_json(dossier / F_SYNTHESE)
    if not est_terminee(synthese):
        logger.info(
            f"[score] {dossier.name} not scored at close: run not finished "
            f"(state {(synthese.get('etat') or {}).get('etat')!r})"
        )
        return None
    debut = time.monotonic()
    logger.info(f"[score] Closing scoring of {dossier.name} — start")
    try:
        registre = F.charger()
        chemin = score_execution(dossier, registre.reference, registre)
        if chemin is None:
            logger.error(
                f"[ALARME] Closing scoring without result — {dossier.parent.parent.name}/"
                f"{dossier.name}: run not scorable or loss engine missing; "
                f"the run stays finished, without a score"
            )
            return None
        from experiences import rendu_scores

        rendu_scores.ecrire(dossier, registre)
        composite = (_lire_json(chemin).get("composite") or {}).get("emd_jsd")
        logger.info(
            f"[score] Closing scoring of {dossier.name} finished in "
            f"{time.monotonic() - debut:.1f} s — composite emd_jsd={composite}"
        )
        return chemin
    except Exception as exc:  # noqa: BLE001 — fail-open : l'exécution reste terminée
        logger.error(
            f"[ALARME] Closing scoring failed — {dossier.parent.parent.name}/"
            f"{dossier.name} after {time.monotonic() - debut:.1f} s: "
            f"{type(exc).__name__}: {exc}; the run stays finished, without a score "
            f"(rerun 'Recalculer toutes les expériences')"
        )
        return None


def _executions(racine: Path):
    """Iterates the run directories under data/experiences/*/executions/*."""
    for exp in sorted(racine.iterdir()) if racine.is_dir() else []:
        execs = exp / "executions"
        if not execs.is_dir():
            continue
        for d in sorted(execs.iterdir()):
            if d.is_dir():
                yield d


def rescorer_tout(
    formule: F.Formule | None = None,
    registre: F.RegistreFormules | None = None,
    racine: Path | None = None,
) -> dict:
    """Rescores all finished runs under ``formule`` (R10, R22).

    Offline: if a scores.json exists and the moves.csv has not changed, we
    **replay** from the raw scores (R6) — no ``Scorer`` rebuilt, no
    file reread. Otherwise we compute in full. No network call in either case.
    """
    registre = registre or F.charger()
    formule = formule or registre.reference
    racine = racine or (REPO_ROOT / "data" / "experiences")
    scorer = None  # built lazily, only if a full computation is required
    bilan = {"rejouees": 0, "calculees": 0, "ignorees": 0}
    logger.info(
        f"[score] Global rescoring under formula {formule.nom} "
        f"({formule.sha256[:12]}) — start"
    )
    for dossier in _executions(racine):
        synthese = _lire_json(dossier / F_SYNTHESE)
        if not est_terminee(synthese):
            bilan["ignorees"] += 1
            continue
        existant = _lire_json(dossier / F_SCORES)
        if existant and not scores_perimes(dossier):
            try:
                contenu = rejouer(existant, formule)
            except ValueError as exc:  # engine missing (R18): nothing written, we stop
                logger.error(
                    f"[ALARME] Replay impossible — loss engine missing: {exc}"
                )
                break
            ecrire(dossier, contenu)
            bilan["rejouees"] += 1
            continue
        if scorer is None:
            scorer, erreur = scorer_pour(formule)
            if scorer is None:
                logger.error(
                    f"[ALARME] Rescoring impossible — loss engine missing: {erreur}"
                )
                break
        if score_execution(dossier, formule, registre, scorer=scorer):
            bilan["calculees"] += 1
        else:
            bilan["ignorees"] += 1
    logger.info(
        f"[score] Global rescoring finished: {bilan['rejouees']} replayed, "
        f"{bilan['calculees']} computed, {bilan['ignorees']} skipped"
    )
    return bilan


def frames_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


__all__ = [
    "TOLERANCE_JOURNAL",
    "JournalIncomplet",
    "calculer",
    "ecrire",
    "est_reference",
    "est_terminee",
    "invalider",
    "mesurer_perimetre",
    "rejouer",
    "rescorer_tout",
    "score_execution",
    "scorer_a_la_cloture",
    "scorer_pour",
    "scores_perimes",
    "verifier_perimetre",
    "volet_pour",
]
