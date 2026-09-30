"""The move log of a run — writing one row, and regeneration (ticket 081).

`moves.csv` is the SCORE SUBSTRATE: `score.calculer` reads nothing else. Yet it was only
written on the path of a **fresh** decision. A cold resume re-served the archived decisions
without writing a row, and the log stayed as the interrupted process had left it.
Run `2026-09-12_11_24_28` was thus scored 5.35 on 274 rows while its archive
held 3,299; rescored on the reconstructed log, it is worth 6.17.
The figure had already been cited in three documents.

This module fixes the failure at its cause, not its symptom. It holds **the only** function
that writes a log row, `ecrire_ligne`, and both paths call it: the runner when the
decision has just been made, the regeneration when it is read back from the archive.
Two writes that diverge are exactly the class of bug this ticket closes — making them
share the code is the only way they do not diverge again.

A row is reconstructed **entirely** from the archived trace and the sealed set:
the trace holds the chosen option, the presented ones, the distribution, the sources, the
discarded ones, the chain constraint and the anticipation; the set holds the complete
`TravelPlan`s, hence the distances. Nothing is borrowed from a sibling run, nothing is
approximated.

What regeneration CANNOT restore, and does not pretend to restore:
the archive does not timestamp decisions one by one (`archive.ajouter_decision` writes the
trace as is), so the "Heure de calcul" column of a regenerated log is uniform.
With no duplicate in the rebuilt file, `frames.latest_attempts` does not suffer, but the
`reprise` flag of the score reading disappears. The interruptions in `synthese.json` still
carry it, and they are the source of truth on this point.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import yaml
from experiences import decision as D
from experiences.archive import (
    METHODE_INEXPLOITABLE,
    METHODE_NON_COUVERT,
    Execution,
)
from loguru import logger
from models import Person

F_MOVES = "moves.csv"
SUFFIXE_REGENERATION = ".regen"

# Archived methods that NEVER produced a log row, live or at regeneration: the set
# does not cover the trip, or the engines proposed no itinerary. They are archived
# (they explain a hole in the coverage) but there is neither a chosen mode nor a
# distance to write. Counting them as skipped, rather than hiding them, is what makes
# it possible to check afterwards that the log is complete.
METHODES_SANS_LIGNE = (METHODE_NON_COUVERT, METHODE_INEXPLOITABLE)


# ── The decision-maker label, as written in "Méthode de sélection" ───────────────────────


@dataclass(frozen=True)
class EtiquetteDecideur:
    """Enough to label a row without building the real decision-maker.

    `_methode_moves` only looks at the decision-maker's name and its `sans_quota`. Rebuilding
    a `DecideurPasserelle` to regenerate a log would require an `LlmAgent`, hence a network
    configuration, just to write a label: regeneration would become unusable offline,
    precisely when it is needed. This label is read from `execution.yaml`, which freezes
    the decision-maker ACTUALLY used (R6).
    """

    nom: str
    sans_quota: bool

    @classmethod
    def depuis_execution(cls, config: dict) -> EtiquetteDecideur:
        regime = config.get("regime_applique") or {}
        experience = config.get("experience") or {}
        type_decideur = (experience.get("decideur") or {}).get("type")
        # Only the gateway consumes a quota (`decideurs.DecideurPasserelle`); this single
        # value decides between the label "LLM" and "Décideur <nom>".
        # Checked on the archived logs: `passerelle` → "LLM", `antigravity` →
        # "Décideur antigravity:gemini-3.8-flash", `modele` → "Décideur modele:rf@…".
        return cls(
            nom=str(regime.get("decideur") or type_decideur or "?"),
            sans_quota=type_decideur != "passerelle",
        )


def methode_moves(methode: str, decideur) -> str:
    """The "Méthode de sélection" label of a log row.

    Takes the METHOD (a trace string), not a `Decision` object: this is what lets the
    runner and the regeneration call the same function. `decideur` is the real
    decision-maker or an `EtiquetteDecideur` — only `nom` and `sans_quota` are read.
    """
    if methode == D.METHODE_CHOIX_UNIQUE:
        return "Un seul itinéraire disponible"
    if methode == D.METHODE_SANS_SOLUTION:
        return "Pas de solution de déplacement"
    if methode == D.METHODE_REPLI_UNIFORME:
        return "LLM Error (Default index) — repli uniforme"
    if methode == D.METHODE_MODELE_NON_IMPUTABLE:
        # Model non-decision (out of domain): no chosen mode, excluded from the score.
        return "Modèle : hors domaine (non imputable)"
    return (
        "LLM"
        if not getattr(decideur, "sans_quota", True)
        else f"Décideur {getattr(decideur, 'nom', '?')}"
    )


# ── The single write of a row ────────────────────────────────────────────────────────────


def _par_code(props: Sequence[D.Proposition], code: str | None) -> D.Proposition | None:
    return next((p for p in props if p.code == code), None) if code else None


async def ecrire_ligne(
    moves,
    *,
    personne: Person,
    trace: dict,
    props: Sequence[D.Proposition],
    decideur,
) -> None:
    """Writes ONE row of `moves.csv` from a decision trace and the set's proposals.

    Single point of passage (ticket 081): the runner calls it on the trace it has just
    built, the regeneration on the one it reads back from the archive. Both produce the
    same row, because it is the same code.

    `props` is the RAW list from the set, not the presented ones: the log reports the offer
    as it existed ("Modes proposés", "Plus rapide", "Options présentées"), not what
    survived the vehicle-chain filtering and the candidate cap.
    """
    retenue = _par_code(props, (trace.get("retenue") or {}).get("code"))
    plan = retenue.plan if retenue is not None else None
    plus_rapide = min(
        (p.plan for p in props),
        key=lambda pl: pl.duration or float("inf"),
        default=None,
    )
    await moves.ecrire(
        person=personne,
        plan=plan,
        purpose=trace.get("purpose"),
        selection_method=methode_moves(str(trace.get("methode") or ""), decideur),
        provider_model=str(trace.get("fournisseur") or ""),
        faster_itinerary=plus_rapide,
        reasoning=str(trace.get("raison") or ""),
        chain_constraint=trace.get("contrainte_chaine", ""),
        anticipation=trace.get("anticipation", "") or "",
        move_id=f"{trace.get('person_id')}:{trace.get('activity_id')}",
        simulated_time=trace.get("timestamp"),
        start_time=plan.start_time if plan is not None else None,
        available_options=[p.plan for p in props],
        activity_id=trace.get("activity_id"),
        # `{}` and `None` are treated identically by `_mode_probability_cells` (empty
        # cells): a decision without a distribution does not fabricate a 0, which would mean
        # "the model explicitly discarded this mode".
        mode_probabilities=trace.get("distribution") or None,
        sources=",".join(
            f"{k}:{v}"
            for k, v in sorted(
                Counter(
                    str(v).split(":")[0] for v in (trace.get("sources") or {}).values()
                ).items()
            )
        ),
        ecartees=D.resumer_ecartees(trace.get("ecartees") or []),
        lot=str(trace.get("identifiant_lot") or ""),
    )


# ── Full regeneration ────────────────────────────────────────────────────────────────────


class JournalIrregenerable(ValueError):
    """The regeneration cannot be faithful: it is not attempted."""


def _verifier_empreintes(config: dict, jeu, info_population) -> None:
    """Refuses to regenerate from a cohort or a set other than those of the run.

    Reconstructing a log from a neighbouring population would produce a file that is
    plausible, scorable, and wrong — exactly the silent corruption this ticket
    closes. The check covers the fingerprints frozen in `execution.yaml` (E19).
    """
    empreintes = config.get("empreintes") or {}
    attendu_jeu = (empreintes.get("jeu") or {}).get("sha256")
    if attendu_jeu and jeu.empreinte and attendu_jeu != jeu.empreinte:
        raise JournalIrregenerable(
            f"set differs from the run's: the archive cites "
            f"{attendu_jeu[:12]}…, the set provided is {jeu.empreinte[:12]}… "
            f"({jeu.nom}). Regenerating the log from another set would produce "
            f"distances and options that were never presented."
        )
    attendu_pop = (empreintes.get("population") or {}).get("fichier_sha256")
    reel_pop = getattr(info_population, "fichier_sha256", None)
    if attendu_pop and reel_pop and attendu_pop != reel_pop:
        raise JournalIrregenerable(
            f"population different from that of the execution: the archive cites "
            f"{attendu_pop[:12]}…, the cohort supplied is {reel_pop[:12]}… "
            f"({getattr(info_population, 'nom', '?')}). The persona columns "
            f"would describe other people than those who decided."
        )


@contextmanager
def _reglages_figes(config: dict):
    """Applies, for one regeneration, the settings that `moves.csv` copies from the process.

    `MoveLogger.log_move` does not receive everything as arguments: "Mémoire à long terme" and
    "Température" are read from `settings`, which `cli.cmd_lancer` had set from the
    experiment definition. Regenerating without them produces a log that describes the
    regeneration process instead of the run — measured on a `memoire: false` run, the
    column switched to `True` on all 3,161 rows.

    The values come from the `execution.yaml` snapshot, which freezes the regime ACTUALLY
    applied (R6), and are restored as they were afterwards: called from the runner, where
    they are already correct, the regeneration changes nothing.
    """
    from settings import settings

    exp = config.get("experience") or {}
    decideur = exp.get("decideur") or {}
    avant_memoire = settings.agent.long_term_memory_enabled
    avant_params = dict(settings.agent.llm_params)
    try:
        if "memoire" in exp:
            settings.agent.long_term_memory_enabled = bool(exp.get("memoire"))
        if decideur.get("type") in ("passerelle", "antigravity"):
            settings.agent.llm_params = {
                **settings.agent.llm_params,
                **(decideur.get("parametres") or {}),
            }
        yield
    finally:
        settings.agent.long_term_memory_enabled = avant_memoire
        settings.agent.llm_params = avant_params


def _reference_existante(chemin: Path) -> str | None:
    """The "Référence" value carried by the current log, if it carries one.

    This column names the process that produced the decisions. A regeneration produces the
    FILE, not the decisions: writing today's date there would make the log claim it comes
    from a run that never decided anything. So the original value is carried over.
    """
    if not Path(chemin).is_file():
        return None
    import csv

    with Path(chemin).open(encoding="utf-8") as fh:
        for ligne in csv.DictReader(fh):
            valeur = (ligne.get("Référence") or "").strip()
            return valeur or None
    return None


def _reecrire_reference(chemin: Path, reference: str) -> None:
    import csv

    with Path(chemin).open(encoding="utf-8") as fh:
        lecteur = csv.DictReader(fh)
        colonnes = list(lecteur.fieldnames or [])
        lignes = list(lecteur)
    if "Référence" not in colonnes:
        return
    for ligne in lignes:
        ligne["Référence"] = reference
    with Path(chemin).open("w", encoding="utf-8", newline="") as fh:
        redacteur = csv.DictWriter(fh, fieldnames=colonnes)
        redacteur.writeheader()
        redacteur.writerows(lignes)


async def regenerer(
    execution: Execution,
    jeu,
    personnes: Sequence[Person],
    decideur,
    *,
    info_population=None,
) -> dict:
    """Fully rebuilds `moves.csv` from `decisions.jsonl` and the sealed set.

    Writes to a neighbouring file then replaces atomically: the existing log is
    never destroyed by a regeneration that fails.

    Returns the counters — a program that stays silent when all goes well cannot tell
    "it worked" from "it no longer runs".
    """
    if info_population is not None:
        _verifier_empreintes(execution.config, jeu, info_population)

    debut = time.monotonic()
    cible = execution.dossier / F_MOVES
    provisoire = execution.dossier / (F_MOVES + SUFFIXE_REGENERATION)
    provisoire.unlink(missing_ok=True)
    par_personne = {p.person_id: p for p in personnes}
    avant = _compter_lignes(cible)
    reference = _reference_existante(cible)
    logger.info(
        f"[journal] Regeneration of {execution.nom} — start: "
        f"{len(execution.decisions)} archived decisions, {avant} row(s) in the current log"
    )

    from experiences.archive import JournalMoves

    moves = JournalMoves(provisoire)
    compteurs = Counter()
    absents: list[str] = []
    with _reglages_figes(execution.config):
        for trace in execution.decisions:
            methode = str(trace.get("methode") or "")
            if methode in METHODES_SANS_LIGNE:
                compteurs[f"sautees_{methode}"] += 1
                continue
            person_id = str(trace.get("person_id"))
            activity_id = str(trace.get("activity_id"))
            personne = par_personne.get(person_id)
            if personne is None:
                compteurs["persona_absente"] += 1
                absents.append(f"{person_id}:{activity_id}")
                continue
            props = jeu.propositions(person_id, activity_id)
            if not props:
                # The set no longer holds this trip although a decision settled it: the
                # log would be truncated without saying so. We refuse to write it silently.
                compteurs["sans_proposition"] += 1
                absents.append(f"{person_id}:{activity_id}")
                continue
            await ecrire_ligne(
                moves, personne=personne, trace=trace, props=props, decideur=decideur
            )
            compteurs["reconstruites"] += 1

    if absents:
        logger.error(
            f"[ALARME] Regenerated log incomplete — {execution.nom}: "
            f"{len(absents)} archived decision(s) with no persona or proposal in the "
            f"(population, set) pair provided; the rebuilt log will lack them. "
            f"First cases: {', '.join(absents[:10])}"
            f"{'…' if len(absents) > 10 else ''}"
        )

    if not provisoire.exists():
        # No row written: never replace an existing log with nothing.
        raise JournalIrregenerable(
            f"no row rebuilt for {execution.nom} — existing log kept "
            f"({avant} row(s)). Archived decisions: {len(execution.decisions)}."
        )
    if reference:
        _reecrire_reference(provisoire, reference)
    os.replace(provisoire, cible)
    apres = _compter_lignes(cible)
    duree = time.monotonic() - debut
    logger.info(
        f"[journal] Regeneration of {execution.nom} finished in {duree:.1f} s — "
        f"{avant} → {apres} row(s); rebuilt {compteurs['reconstruites']}, "
        f"not covered {compteurs[f'sautees_{METHODE_NON_COUVERT}']}, "
        f"unusable {compteurs[f'sautees_{METHODE_INEXPLOITABLE}']}, "
        f"without proposal {compteurs['sans_proposition']}, "
        f"persona missing {compteurs['persona_absente']}"
    )
    return {
        "execution": execution.nom,
        "lignes_avant": avant,
        "lignes_apres": apres,
        "decisions_archivees": len(execution.decisions),
        "duree_s": round(duree, 3),
        **{k: int(v) for k, v in compteurs.items()},
    }


def _compter_lignes(chemin: Path) -> int:
    """Data rows of a CSV (header excluded); 0 if the file does not exist."""
    if not Path(chemin).is_file():
        return 0
    import csv

    with Path(chemin).open(encoding="utf-8") as fh:
        return sum(1 for _ in csv.DictReader(fh))


# ── Verification (sweep) ─────────────────────────────────────────────────────────────────


def _lire_json(p: Path) -> dict:
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def verifier(dossier: str | Path) -> dict:
    """Observation, with no write decision: how many rows, how many decisions, the gap.

    Does not raise and writes nothing: this is the sweep. The RULE itself lives in
    `score.verifier_perimetre`, the only place that can refuse a score.
    """
    from experiences import score as S

    dossier = Path(dossier)
    synthese = _lire_json(dossier / "synthese.json")
    lignes = _compter_lignes(dossier / F_MOVES)
    constat = S.mesurer_perimetre(lignes, synthese)
    constat["execution"] = synthese.get("execution") or dossier.name
    constat["experience"] = synthese.get("experience")
    constat["etat"] = (synthese.get("etat") or {}).get("etat")
    constat["dossier"] = str(dossier)
    constat["interruptions"] = [
        str(i.get("cause")) for i in (synthese.get("interruptions") or [])
    ]
    return constat


def executions(racine: str | Path):
    """Iterates over the run folders under `<racine>/*/executions/*`."""
    racine = Path(racine)
    for exp in sorted(racine.iterdir()) if racine.is_dir() else []:
        execs = exp / "executions"
        if not execs.is_dir():
            continue
        for d in sorted(execs.iterdir()):
            if d.is_dir():
                yield d


# ── Locating the set and the population of an archived run ───────────────────────────────


def resoudre_sources(
    execution: Execution,
    *,
    jeu: str | Path | None = None,
    population: str | Path | None = None,
    motif_archive: str | None = None,
):
    """(Set, persons, InfoPopulation) of a run, to regenerate it.

    The paths frozen in `execution.yaml` are CONTAINER paths (`/data/eqasim-output/…`)
    and do not exist on the host: `--jeu` and `--population` replace them. The fingerprints
    are checked afterwards by `regenerer`, so a wrong path is refused, not silently used.
    """
    from experiences.jeu import Jeu
    from experiences.population import charger_population, info_population

    config = execution.config
    exp = config.get("experience") or {}

    chemin_jeu = Path(jeu) if jeu else None
    if chemin_jeu is None:
        declare = (exp.get("jeu") or {}).get("dossier")
        nom = (exp.get("jeu") or {}).get("nom")
        for candidat in (declare, f"data/jeux/{nom}" if nom else None):
            if candidat and Path(candidat).is_dir():
                chemin_jeu = Path(candidat)
                break
    if chemin_jeu is None:
        raise JournalIrregenerable(
            f"set not found for {execution.nom}: the archive cites "
            f"{(exp.get('jeu') or {}).get('nom')!r}, whose folder cannot be located "
            f"on this host. Specify it with `--jeu <dossier>`."
        )

    chemin_pop = Path(population) if population else None
    if chemin_pop is None:
        declare = (exp.get("population") or {}).get("chemin")
        if declare and Path(declare).exists():
            chemin_pop = Path(declare)
    if chemin_pop is None:
        raise JournalIrregenerable(
            f"population not found for {execution.nom}: the archive cites "
            f"{(exp.get('population') or {}).get('chemin')!r} (container path). "
            f"Specify it with `--population <dossier>`."
        )

    objet_jeu = Jeu.charger(chemin_jeu, archive_confirmee=motif_archive)
    info = info_population(chemin_pop, archivee_confirmee=motif_archive)
    personnes, _ = charger_population(chemin_pop, archivee_confirmee=motif_archive)
    return objet_jeu, personnes, info


def ouvrir(dossier: str | Path, *, motif_archive: str | None = None) -> Execution:
    """Opens a run, including under cold archive if a reason is given."""
    from experiences import froid

    froid.verifier(
        dossier,
        motif_archive,
        quoi="an archived execution",
        comment_lever='pass `--motif-archive "<reason>"`',
    )
    return Execution.ouvrir(dossier)


def config_de(dossier: str | Path) -> dict:
    """Raw `execution.yaml`, without going through `Execution.ouvrir` (no decision read)."""
    p = Path(dossier) / "execution.yaml"
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}


__all__ = [
    "METHODES_SANS_LIGNE",
    "EtiquetteDecideur",
    "JournalIrregenerable",
    "config_de",
    "ecrire_ligne",
    "executions",
    "methode_moves",
    "ouvrir",
    "regenerer",
    "resoudre_sources",
    "verifier",
]
