"""Archive of a run (ticket 035, specs 05 and 06).

    data/experiences/<exp>/executions/<horodatage>/
        execution.yaml    frozen configuration, fingerprints, regime, randomness sources, interruptions, closure
        etat.json         current state and its reason (definie · en_cours · en_pause · epuisee · arretee · terminee)
        decisions.jsonl   one complete decision trace per line — atomic append (Q9)
        erreurs.jsonl     one FAILED attempt per line (R3): nothing is skipped without leaving a trace
        compteurs.json    final counters and coverage (E14)
        moves.csv         same columns as the simulation → `make report`, synthesis (S7)
        llm_exchanges.jsonl  gateway exchanges (tokens), same format as the simulation

A decision is archived entirely or not at all; on loading, a truncated last line
is discarded with a WARNING that names it (Q9). A closed archive carries the fingerprints of its
files: a modified byte is detected (E19).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import yaml
from experiences.decision import valider_trace
from experiences.population import sha256_fichier
from loguru import logger

VERSION_EXECUTION = "execution1"
ETAT_DEFINIE = "definie"
ETAT_EN_COURS = "en_cours"
ETAT_EN_PAUSE = "en_pause"
ETAT_EPUISEE = "epuisee"
ETAT_ARRETEE = "arretee"
ETAT_TERMINEE = "terminee"
ETAT_EN_ATTENTE_QUOTA = "en_attente_quota"  # in-process: sleeps until the quota window (R4, --attendre-fenetre)
ETAT_EN_ATTENTE_AGENT = (
    "en_attente_agent"  # in-process: sleeps waiting for an Antigravity sub-agent
)
ETAT_INTERROMPUE = (
    "interrompue"  # process died without closure, reconciled by the scheduler (R7)
)
ETATS = (
    ETAT_DEFINIE,
    ETAT_EN_COURS,
    ETAT_EN_PAUSE,
    ETAT_EPUISEE,
    ETAT_ARRETEE,
    ETAT_TERMINEE,
    ETAT_EN_ATTENTE_QUOTA,
    ETAT_EN_ATTENTE_AGENT,
    ETAT_INTERROMPUE,
)
ETATS_FINAUX = (
    ETAT_EPUISEE,
    ETAT_ARRETEE,
    ETAT_TERMINEE,
    ETAT_INTERROMPUE,
)  # en_attente_quota and en_attente_agent are NOT final: the run resumes on its own

F_EXECUTION = "execution.yaml"
F_ETAT = "etat.json"
F_DECISIONS = "decisions.jsonl"
F_ERREURS = "erreurs.jsonl"  # one failed attempt per line (R3) — successes go to decisions.jsonl
F_COMPTEURS = "compteurs.json"
F_MOVES = "moves.csv"
F_ECHANGES = "llm_exchanges.jsonl"
F_SYNTHESE = "synthese.json"
F_SYNTHESE_HTML = "synthese.html"
F_SCORES = (
    "scores.json"  # composite score + per-stratum detail (spec scoring_composite_experiences)
)
F_SCORES_HTML = "synthese_scores.html"  # score synthesis page of a run

METHODE_NON_COUVERT = (
    "non_couvert"  # trace of a trip absent from the set: not a decision (S2)
)
METHODE_INEXPLOITABLE = "inexploitable"  # covered by the set but NO proposal from the engines: excluded from expected trips (decision 3)

CLES_EXECUTION_OBLIGATOIRES = (
    "version",
    "experience",
    "empreintes",
    "regime_demande",
    "regime_applique",
    "sources_alea",
    "interruptions",
    "cree_le",
)


class ArchiveInvalide(ValueError):
    """Incomplete, altered or unreadable archive."""


def _iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ecrire_atomique(chemin: Path, texte: str) -> None:
    tmp = chemin.with_name(chemin.name + ".tmp")
    tmp.write_text(texte, encoding="utf-8")
    os.replace(tmp, chemin)


def lire_decisions(chemin: Path, *, reparer: bool = True) -> tuple[list[dict], int]:
    """Valid traces + valid bytes. Truncated last line: discarded, named, truncated if `reparer`."""
    traces: list[dict] = []
    if not chemin.exists():
        return traces, 0
    brut = chemin.read_bytes()
    octets = 0
    lignes = brut.split(b"\n")
    for numero, ligne in enumerate(lignes, start=1):
        if not ligne.strip():
            octets += len(ligne) + 1
            continue
        try:
            trace = json.loads(ligne.decode("utf-8"))
            manquants = valider_trace(trace)
            if manquants:
                raise ArchiveInvalide(
                    f"{chemin.name} line {numero}: missing trace fields {manquants}"
                )
        except (ValueError, UnicodeDecodeError) as e:
            derniere = octets + len(ligne) >= len(brut.rstrip(b"\n"))
            if derniere and not isinstance(e, ArchiveInvalide):
                logger.warning(
                    f"[archive] {chemin.name} : dernière entrée (ligne {numero}) tronquée — écartée, elle sera redemandée"
                )
                if reparer:
                    with open(chemin, "r+b") as f:
                        f.truncate(octets)
                return traces, octets
            raise ArchiveInvalide(f"{chemin.name} line {numero}: {e}") from e
        traces.append(trace)
        octets += len(ligne) + 1
    return traces, min(octets, len(brut))


class JournalMoves:
    """The run's `moves.csv`, with the simulation's columns (S7, `make report`)."""

    def __init__(self, chemin: Path):
        from urban_mobility_agents.utils.move_logger import MoveLogger

        class _Logger(MoveLogger):
            def _resolve_path(self_inner) -> Path:
                return chemin

        self._logger = _Logger()
        self.chemin = chemin

    async def ecrire(self, **kw) -> None:
        await self._logger.log_move(**kw)


class Execution:
    """The directory of a run, open for reading/writing."""

    def __init__(
        self,
        dossier: Path,
        config: dict,
        decisions: list[dict] | None = None,
        compteurs: dict | None = None,
    ):
        self.dossier = Path(dossier)
        self.config = config
        self._index: dict[tuple[str, str], dict] = {}
        for t in decisions or []:
            self._index[(str(t.get("person_id")), str(t.get("activity_id")))] = t
        self.compteurs: dict = dict(compteurs or {})
        self._fh = None
        self._moves: JournalMoves | None = None

    # ── creation / opening ──
    @classmethod
    def creer(
        cls,
        dossier_experience: Path,
        experience: dict,
        empreintes: dict,
        regime_demande: dict,
        sources_alea: dict,
        reglages_herites: dict | None = None,
        horodatage: str | None = None,
    ) -> Execution:
        base = horodatage or datetime.now(timezone.utc).strftime("%Y-%m-%d_%H_%M_%S")
        nom, i = base, 1
        while (Path(dossier_experience) / "executions" / nom).exists():
            i += 1
            nom = f"{base}_{i}"  # never overwrite: a relaunch is a NEW run (E18)
        dossier = Path(dossier_experience) / "executions" / nom
        dossier.mkdir(parents=True, exist_ok=False)
        config = {
            "version": VERSION_EXECUTION,
            "cree_le": _iso(),
            "experience": experience,
            "empreintes": empreintes,
            "regime_demande": regime_demande,
            "regime_applique": {},
            "reglages_herites": reglages_herites or {},
            "sources_alea": sources_alea,
            "interruptions": [],
            "cloture": None,
        }
        ex = cls(dossier, config)
        ex._sauver_config()
        ex.changer_etat(ETAT_DEFINIE)
        return ex

    @classmethod
    def ouvrir(cls, dossier: str | Path) -> Execution:
        dossier = Path(dossier)
        chemin = dossier / F_EXECUTION
        if not chemin.is_file():
            raise ArchiveInvalide(f"no {F_EXECUTION} in {dossier}")
        try:
            config = yaml.safe_load(chemin.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as e:
            raise ArchiveInvalide(f"{F_EXECUTION} unreadable: {e}") from e
        manquantes = [c for c in CLES_EXECUTION_OBLIGATOIRES if c not in config]
        if manquantes:
            raise ArchiveInvalide(f"{F_EXECUTION}: missing keys {manquantes}")
        decisions, _ = lire_decisions(
            dossier / F_DECISIONS, reparer=not config.get("cloture")
        )
        compteurs = {}
        if (dossier / F_COMPTEURS).is_file():
            compteurs = json.loads((dossier / F_COMPTEURS).read_text(encoding="utf-8"))
        return cls(dossier, config, decisions, compteurs)

    # ── configuration ──
    def _sauver_config(self) -> None:
        _ecrire_atomique(
            self.dossier / F_EXECUTION,
            yaml.safe_dump(self.config, allow_unicode=True, sort_keys=False),
        )

    def noter_reglages_herites(self, **champs) -> None:
        """Records the settings IMPOSED on the run, at creation as well as on resume.

        Ticket 085 (B4): a resume reopens an existing run, and its trace would be wrong if
        the routing restriction had changed since launch. An archived measurement must say
        under which restriction it was taken, not under which it started.
        """
        herites = dict(self.config.get("reglages_herites") or {})
        herites.update(champs)
        self.config["reglages_herites"] = herites
        self._sauver_config()

    def mettre_a_jour_regime(self, **champs) -> None:
        self.config.setdefault("regime_applique", {}).update(champs)
        self._sauver_config()

    @property
    def nom(self) -> str:
        return self.dossier.name

    @property
    def cloturee(self) -> bool:
        return bool(self.config.get("cloture"))

    # ── state ──
    def etat(self) -> dict:
        p = self.dossier / F_ETAT
        if not p.is_file():
            return {"etat": ETAT_DEFINIE}
        return json.loads(p.read_text(encoding="utf-8"))

    def changer_etat(
        self,
        etat: str,
        raison: str | None = None,
        reprise_possible_a: str | None = None,
    ) -> None:
        if etat not in ETATS:
            raise ValueError(f"unknown state: {etat!r}")
        contenu = {
            "etat": etat,
            "raison": raison,
            "reprise_possible_a": reprise_possible_a,
            "maj": _iso(),
            "decisions_archivees": len(self._index),
        }
        _ecrire_atomique(
            self.dossier / F_ETAT, json.dumps(contenu, ensure_ascii=False, indent=1)
        )
        niveau = logger.error if etat == ETAT_EPUISEE else logger.info
        niveau(
            f"[execution] {self.nom} → {etat}"
            + (f" — {raison}" if raison else "")
            + (
                f" — reprise possible à {reprise_possible_a}"
                if reprise_possible_a
                else ""
            )
        )

    def ajouter_interruption(self, cause: str, **infos) -> None:
        self.config.setdefault("interruptions", []).append(
            {
                "instant": _iso(),
                "cause": cause,
                "decisions_archivees": len(self._index),
                **infos,
            }
        )
        self._sauver_config()

    # ── decisions ──
    def decision(self, person_id: str, activity_id: str) -> dict | None:
        return self._index.get((str(person_id), str(activity_id)))

    @property
    def decisions(self) -> list[dict]:
        return list(self._index.values())

    def ajouter_decision(self, trace: dict) -> None:
        """Appends a complete line + fsync: entirely or not at all (Q9)."""
        if self.cloturee:
            raise ArchiveInvalide("closed archive: immutable (E19)")
        manquants = valider_trace(trace)
        if manquants:
            raise ArchiveInvalide(f"incomplete trace: {manquants}")
        cle = (str(trace["person_id"]), str(trace["activity_id"]))
        if cle in self._index:
            return
        if self._fh is None:
            self._fh = open(self.dossier / F_DECISIONS, "ab")  # noqa: SIM115 — long-lived handle: append at each decision, closed by fermer()
        self._fh.write(
            (json.dumps(trace, ensure_ascii=False, default=str) + "\n").encode("utf-8")
        )
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._index[cle] = trace

    def fermer(self) -> None:
        if self._fh is not None and not self._fh.closed:
            self._fh.close()
        self._fh = None

    def ajouter_erreur(self, erreur: dict) -> None:
        """Records a FAILED attempt in `erreurs.jsonl` (R3) — pure, tolerant append.

        Unlike `ajouter_decision`, no decision is archived here: this log
        makes visible the attempts that did not succeed (timestamp, person, activity,
        attempt, type, message, provider, wait). It never enters the decision
        index and never blocks the run: its failure is swallowed (WARNING), an untraced
        error is better than a lost trip.
        """
        try:
            ligne = (json.dumps(erreur, ensure_ascii=False, default=str) + "\n").encode(
                "utf-8"
            )
            with open(self.dossier / F_ERREURS, "ab") as f:
                f.write(ligne)
                f.flush()
                os.fsync(f.fileno())
        except OSError as e:
            logger.warning(f"[archive] {F_ERREURS} not written for {self.nom}: {e}")

    # ── counters / moves ──
    def ecrire_compteurs(self, compteurs: dict) -> None:
        self.compteurs = dict(compteurs)
        _ecrire_atomique(
            self.dossier / F_COMPTEURS,
            json.dumps(self.compteurs, ensure_ascii=False, indent=1, default=str),
        )

    def journal_moves(self) -> JournalMoves:
        if self._moves is None:
            self._moves = JournalMoves(self.dossier / F_MOVES)
        return self._moves

    # ── closure (E19) ──
    def cloturer(
        self,
        etat_final: str,
        raison: str | None = None,
        reprise_possible_a: str | None = None,
    ) -> None:
        if etat_final not in ETATS_FINAUX:
            raise ValueError(f"closure impossible in state {etat_final!r}")
        self.fermer()
        empreintes = {}
        for f in (F_DECISIONS, F_ERREURS, F_MOVES, F_COMPTEURS, F_ECHANGES):
            p = self.dossier / f
            if p.is_file():
                empreintes[f] = sha256_fichier(p)
        self.config["cloture"] = {
            "le": _iso(),
            "etat": etat_final,
            "sha256": empreintes,
        }
        self._sauver_config()
        self.changer_etat(etat_final, raison, reprise_possible_a)


def valider_archive(dossier: str | Path) -> list[str]:
    """Problems of a reread archive (E10, E19) — empty if it is valid."""
    dossier = Path(dossier)
    problemes: list[str] = []
    try:
        ex = Execution.ouvrir(dossier)
    except ArchiveInvalide as e:
        return [str(e)]
    for cle in ("population", "jeu", "gabarit", "decideur", "depot"):
        if cle not in (ex.config.get("empreintes") or {}):
            problemes.append(f"empreinte manquante : {cle}")
    if not (dossier / F_ETAT).is_file():
        problemes.append(f"{F_ETAT} absent")
    if not (dossier / F_DECISIONS).is_file():
        problemes.append(f"{F_DECISIONS} absent")
    if not ex.compteurs:
        problemes.append(f"{F_COMPTEURS} absent ou vide")
    elif "couverture" not in ex.compteurs:
        problemes.append("compteurs sans couverture")
    for t in ex.decisions:
        if t.get("methode") in ("decideur", "repli_uniforme") and not t.get("presente"):
            problemes.append(
                f"décision {t.get('person_id')}/{t.get('activity_id')} sans le texte présenté (E11)"
            )
            break
    cloture = ex.config.get("cloture") or {}
    for f, sha in (cloture.get("sha256") or {}).items():
        p = dossier / f
        if not p.is_file():
            problemes.append(f"{f} manquant alors que la clôture l'annonce")
        elif sha256_fichier(p) != sha:
            problemes.append(f"{f} altéré depuis la clôture (E19)")
    return problemes


__all__ = [
    "ETATS",
    "ETATS_FINAUX",
    "ETAT_ARRETEE",
    "ETAT_DEFINIE",
    "ETAT_EN_ATTENTE_AGENT",
    "ETAT_EN_ATTENTE_QUOTA",
    "ETAT_EN_COURS",
    "ETAT_EN_PAUSE",
    "ETAT_EPUISEE",
    "ETAT_INTERROMPUE",
    "ETAT_TERMINEE",
    "F_COMPTEURS",
    "F_DECISIONS",
    "F_ECHANGES",
    "F_ERREURS",
    "F_ETAT",
    "F_EXECUTION",
    "F_MOVES",
    "F_SYNTHESE",
    "F_SYNTHESE_HTML",
    "METHODE_INEXPLOITABLE",
    "METHODE_NON_COUVERT",
    "VERSION_EXECUTION",
    "ArchiveInvalide",
    "Execution",
    "JournalMoves",
    "lire_decisions",
    "valider_archive",
]
