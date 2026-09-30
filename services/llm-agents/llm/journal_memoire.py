"""Memory journal — one Markdown file per agent, written continuously (ticket 075).

WHY
---
The repository already knows how to log memory WRITES (`agent_memory_events.jsonl`). It
can say neither what **triggered** a consolidation, nor what a concept operation
changed, nor what the memory **looked like afterwards**. On a one-day run, this does not
matter: memory has no time to move. Over sixty days, it is the whole point of
the observation, and reconstructing it afterwards from a JSONL is an archaeologist's job.

WHAT THIS FILE GUARANTEES
-------------------------
1. **Nothing when switched off.** `agent.journal_memoire_enabled` is false by default: a run with
   a thousand agents pays neither a byte nor a disk call.
2. **Never an exception towards the caller.** A journal that brings down a sixty-day
   simulation would be worse than no journal at all. Any write error is
   logged as WARNING and swallowed.
3. **One agent, one file.** No mixing between two memories is possible.
4. **Freezing.** During the replay of a hot resume, the journal is FROZEN: days already
   written are not rewritten.

WHAT IT IS NOT
--------------
It is not a source of measurement. The figures of a paper come from the memory
metadata and from `moves.csv`, not from a text formatted for reading.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from llm.memory import MemoryEntry
from loguru import logger
from settings import _run_artifacts_disabled, settings
from urban_mobility_agents.utils.reprise import gel_actif

# Length of a content shown in a table. Beyond it, the table becomes unreadable and the
# file impossible to browse — yet it is written TO BE browsed.
_LARGEUR_CONTENU = 110


def _tronquer(texte: Any, largeur: int = _LARGEUR_CONTENU) -> str:
    """Content reduced to a table cell: one line, a bounded length."""
    plat = " ".join(str(texte or "").split())
    plat = plat.replace("|", "\\|")
    return plat if len(plat) <= largeur else plat[: largeur - 1] + "…"


def _jour_de(quand: datetime) -> str:
    return quand.strftime("%Y-%m-%d")


def _heure_de(quand: datetime) -> str:
    return quand.strftime("%H:%M")


class JournalMemoire:
    """Writes `<workdir>/memoires/<person_id>.md`, one file per agent."""

    _instance: JournalMemoire | None = None

    def __init__(self, repertoire: Path):
        self.repertoire = Path(repertoire)
        # ⚠ The directory is created at the FIRST WRITE, never at construction. Creating it
        # here made a run directory at every import — a host-side test discovery
        # dropped one per minute into `services/llm-agents/experiments/archive/`
        # (seen on 2026-09-14, during the pilot run of ticket 075).
        self._entetes: set[str] = set()
        self._jour_courant: dict[str, str] = {}
        self._gele = False

    # ── Life cycle ───────────────────────────────────────────────────────────────────────

    @classmethod
    def get(cls) -> JournalMemoire | None:
        """The instance, or `None` if the journal is off. No side effect in that case."""
        if not getattr(settings.agent, "journal_memoire_enabled", False):
            return None
        if cls._instance is None and _run_artifacts_disabled():
            # The journal is a RUN ARTIFACT: under test, it does not open by itself. Without
            # this guard, a test suite started while a run was going dropped an
            # experiment directory per run (seen on 2026-09-14). A test that wants
            # a journal injects it itself into `_instance`, on a directory of its own.
            return None
        if cls._instance is None:
            cls._instance = cls(Path(settings.agent.journal_memoire_dir))
            logger.info(
                f"[journal-mémoire] active — one file per agent in {cls._instance.repertoire}"
            )
        return cls._instance

    @classmethod
    def reinitialiser(cls) -> None:
        """Forgets the instance. Reserved for tests."""
        cls._instance = None

    def geler(self, motif: str) -> None:
        """Suspends all writing — replay of a hot resume."""
        if not self._gele:
            logger.info(f"[journal-mémoire] FROZEN: {motif}")
        self._gele = True

    def degeler(self, motif: str) -> None:
        if self._gele:
            logger.info(f"[journal-mémoire] unfrozen: {motif}")
        self._gele = False

    @property
    def gele(self) -> bool:
        return self._gele

    # ── Writing ──────────────────────────────────────────────────────────────────────────

    def _fichier(self, person_id: str) -> Path:
        return self.repertoire / f"{person_id}.md"

    def _ecrire(self, person_id: str, texte: str) -> None:
        """Appends to the agent's file. A disk error NEVER propagates to the caller."""
        if self._gele or gel_actif():
            return
        try:
            self.repertoire.mkdir(parents=True, exist_ok=True)
            with self._fichier(person_id).open("a", encoding="utf-8") as f:
                f.write(texte)
        except OSError as err:
            logger.warning(
                f"[journal-mémoire] write impossible for {person_id} ({err}) — "
                f"the event is lost, the simulation goes on."
            )

    def ouvrir_agent(self, person_id: str, identite: dict | None = None) -> None:
        """File header, written only once."""
        if person_id in self._entetes:
            return
        self._entetes.add(person_id)
        if self._fichier(person_id).exists():
            # Hot resume: the restored file continues, it does not start over.
            return
        traits = identite or {}
        lignes = [
            f"# Mémoire de {traits.get('name') or person_id} — agent {person_id}",
            "",
            "> Journal écrit en continu (ticket 075) : une entrée à chaque événement qui",
            "> **modifie** la mémoire, avec ce qui l'a déclenché. Les heures sont en temps",
            "> SIMULÉ. Ce fichier se lit ; il ne se mesure pas.",
            "",
        ]
        if traits:
            lignes += [
                (
                    f"**Profil.** {traits.get('age', '?')} ans · {traits.get('main_occupation', '?')} · "
                    f"{traits.get('residence_commune', '?')} ({traits.get('residence_zone', '?')}) · "
                    f"voiture : {traits.get('car_availability', '?')} · "
                    f"vélo : {traits.get('personal_bike', '?')} · "
                    f"abonnement TC : {'oui' if traits.get('has_pt_subscription') else 'non'}"
                ),
                "",
            ]
        self._ecrire(person_id, "\n".join(lignes))

    def _jour(self, person_id: str, quand: datetime) -> None:
        """Opens a day section when the simulated date MOVES FORWARD.

        ⚠ Never backwards, and this is not a presentation detail. A write line
        carries the timestamp of the EVENT — often the day before, since a 22:00 consolidation
        writes memories dated from the morning —, while the consolidation section carries
        the current time. Sectioning on every date change therefore made the header
        oscillate: 76 sections for 32 simulated days, measured on 2026-09-15 on the run of
        ticket 075. A memory dated earlier stays in the open section; its own time is
        written on its line anyway.
        """
        jour = _jour_de(quand)
        connu = self._jour_courant.get(person_id)
        if connu is not None and jour <= connu:
            return
        self._jour_courant[person_id] = jour
        self._ecrire(person_id, f"\n## {quand.strftime('%A %d %B %Y')}\n\n")

    # ── Events ───────────────────────────────────────────────────────────────────────────

    def ecriture(self, person_id: str, quand: datetime, entree: MemoryEntry) -> None:
        """An entry is written to long-term memory. Compact line: there are thousands."""
        if self._gele:
            return
        self._jour(person_id, quand)
        self._ecrire(
            person_id,
            f"- `{_heure_de(quand)}` **écriture** ({entree.memory_type}) — "
            f"{_tronquer(entree.content)} "
            f"<sub>gravité {float(entree.importance or 0.0):.2f} · "
            f"force {float(entree.force or 0.0):.1f} j</sub>\n",
        )

    def rappel(
        self, person_id: str, quand: datetime, servis: Iterable[MemoryEntry]
    ) -> None:
        """Memories were served to the model: their lifetime and counter move."""
        if self._gele:
            return
        servis = list(servis)
        if not servis:
            return
        self._jour(person_id, quand)
        detail = " · ".join(
            f"{_tronquer(e.content, 46)} (force {float(e.force or 0.0):.1f} j, "
            f"rappels {int(e.rappels or 0)})"
            for e in servis[:4]
        )
        reste = f" · +{len(servis) - 4} autre(s)" if len(servis) > 4 else ""
        self._ecrire(
            person_id,
            f"- `{_heure_de(quand)}` **rappel** — {len(servis)} souvenir(s) servi(s) : "
            f"{detail}{reste}\n",
        )

    def purge(
        self, person_id: str, quand: datetime, supprimes: Iterable[MemoryEntry]
    ) -> None:
        """Episodic entries fell below the weight threshold."""
        if self._gele:
            return
        supprimes = list(supprimes)
        if not supprimes:
            return
        self._jour(person_id, quand)
        lignes = [
            f"- `{_heure_de(quand)}` **purge** — {len(supprimes)} entrée(s) oubliée(s) :"
        ]
        for e in supprimes:
            age = (quand - e.horodatage_de_reference).total_seconds() / 86400.0
            lignes.append(
                f"    - {_tronquer(e.content, 90)} "
                f"<sub>{age:.1f} j depuis le dernier rappel, force {float(e.force or 0.0):.1f} j</sub>"
            )
        self._ecrire(person_id, "\n".join(lignes) + "\n")

    def consolidation_debut(
        self,
        person_id: str,
        quand: datetime,
        motif: str,
        declencheur: str,
        entrees_consommees: Iterable[Any],
    ) -> None:
        """Opens the section of a consolidation, with what triggered it."""
        if self._gele:
            return
        self._jour(person_id, quand)
        entrees = list(entrees_consommees)
        lignes = [
            "",
            f"### `{_heure_de(quand)}` CONSOLIDATION — déclencheur : **{motif}**",
            "",
            f"**Ce qui l'a déclenchée.** {declencheur}",
            "",
            f"**Entrées de mémoire courte consommées ({len(entrees)}).**",
            "",
        ]
        for e in entrees:
            horodatage = getattr(e, "timestamp", None)
            heure = (
                _heure_de(horodatage) if isinstance(horodatage, datetime) else "  ?  "
            )
            lignes.append(
                f"- `{heure}` {_tronquer(getattr(e, 'content', e), 100)} "
                f"<sub>gravité {float(getattr(e, 'importance', 0.0) or 0.0):.2f}</sub>"
            )
        self._ecrire(person_id, "\n".join(lignes) + "\n")

    def reflexion(self, person_id: str, texte: str) -> None:
        if self._gele or not texte:
            return
        self._ecrire(person_id, f"\n**Réflexion écrite.** {_tronquer(texte, 600)}\n")

    def operation_concept(
        self,
        person_id: str,
        operation: str,
        *,
        avant: str = "",
        apres: str = "",
        observations: str = "",
        contre_exemples: str = "",
        confiance: str = "",
        note: str = "",
    ) -> None:
        """One of the four operations of lot 3 of ticket 071, with its before/after."""
        if self._gele:
            return
        champs = [
            f"opération **{operation}**",
            f"avant : {avant}" if avant else "",
            f"après : {apres}" if apres else "",
            f"observations {observations}" if observations else "",
            f"contre-exemples {contre_exemples}" if contre_exemples else "",
            f"confiance {confiance}" if confiance else "",
            note,
        ]
        self._ecrire(person_id, "- " + " · ".join(c for c in champs if c) + "\n")

    def consolidation_fin(
        self, person_id: str, quand: datetime, entrees: Iterable[MemoryEntry]
    ) -> None:
        """Closes the section with the COMPLETE state of the agent's memory."""
        if self._gele:
            return
        entrees = list(entrees)
        lignes = [
            "",
            f"**État de la mémoire après consolidation ({len(entrees)} entrées).**",
            "",
            "| type | contenu | gravité | force (j) | rappels | dernier rappel | confiance | axes | statut |",
            "|---|---|---:|---:|---:|---|---:|---|---|",
        ]
        for e in entrees:
            axes = " / ".join(
                str(a)
                for a in (
                    e.axe_objet,
                    e.axe_lieu,
                    e.axe_creneau,
                    e.axe_motif,
                    e.axe_meteo,
                )
                if a
            )
            if e.est_episodique:
                statut = "épisodique"
                confiance = "—"
            elif e.est_depasse:
                statut = f"**dépassé** (écarté le {str(e.depasse_le)[:16]})"
                confiance = f"{e.confiance:.2f}"
            elif not e.est_servi:
                statut = f"hors service (écarté le {str(e.depasse_le)[:16]})"
                confiance = f"{e.confiance:.2f}"
            else:
                statut = "servi"
                confiance = f"{e.confiance:.2f}"
            lignes.append(
                f"| {e.memory_type} | {_tronquer(e.content)} | "
                f"{float(e.importance or 0.0):.2f} | {float(e.force or 0.0):.1f} | "
                f"{int(e.rappels or 0)} | "
                f"{e.dernier_rappel.strftime('%d/%m %H:%M') if e.dernier_rappel else '—'} | "
                f"{confiance} | {axes or '—'} | {statut} |"
            )
        self._ecrire(person_id, "\n".join(lignes) + "\n")


def journal() -> JournalMemoire | None:
    """Call shortcut: `journal() and journal().ecriture(...)`."""
    return JournalMemoire.get()
