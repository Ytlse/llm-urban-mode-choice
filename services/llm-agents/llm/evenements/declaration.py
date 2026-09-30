"""What an event declares, and what is refused to it — ticket 100, lot 1.

ONE CHANNEL, TWO INJECTION POINTS
---------------------------------
An event is **a text** placed in the memory of designated agents on designated days,
with **at most one measured fact**. The rest of the machinery is that of memory, unchanged:
severity, force, service duration, consolidation, belief, contradiction.

What the channel says is where the text comes from — `vecu` (the agent suffered it) or `lu` (it
read it). What the moment says is when it enters — `arrivee` (after the decision, the agent chose
while seeing the nominal offer then takes the hit) or `reveil` (before the first decision, it
knows before choosing). The two pairs that matter are `vecu`/`arrivee`, the shock of ticket 079,
and `lu`/`reveil`, the article of ticket 059.

WHAT LOT 1 DELIVERS, AND WHAT IT REFUSES BY NAMING IT
-----------------------------------------------------
Lot 1 is a MIGRATION: it delivers no new function. It moves ticket 079 into
this package, without changing a single one of its behaviours, and proves it by an exact replay.
Everything that belongs to the following lots — `lu` channel, `reveil` point, `foyers` rule,
quoted text, drawn window, judgement at injection — is therefore **refused**, with a message naming
the lot where it arrives. An accepted, inert field is worse than a refused one: nothing reports it.

THE NON-NEGOTIABLE RULE, TAKEN OVER FROM 079
--------------------------------------------
The **injected** delay and the **measured** delay are never confused. Two distinct fields,
wherever they go. Without this separation, no trace would any longer allow saying what the
simulation produced and what it was made to say.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from llm.evenements.calendrier import jour_tire
from llm.evenements.exposition import REGLES_EXPOSITION, REGLES_LIVREES
from llm.evenements.gardes import verifier_empreinte, verifier_texte

# Where the text comes from.
CANAUX: tuple[str, ...] = ("vecu", "lu")
CANAUX_LIVRES: tuple[str, ...] = ("vecu", "lu")

# When it enters memory, relative to the agent's decision.
MOMENTS: tuple[str, ...] = ("arrivee", "reveil")
MOMENTS_LIVRES: tuple[str, ...] = ("arrivee", "reveil")

# Does the agent judge what it has just experienced or read? `aucun` is the DECLARED ABLATION of
# ticket 100 (Q4): it measures what judgement adds, and the golden test of the migration runs
# under it. `a_l_injection` arrives in lot 3.
JUGEMENTS: tuple[str, ...] = ("aucun", "a_l_injection")
JUGEMENTS_LIVRES: tuple[str, ...] = ("aucun", "a_l_injection")

# Application cadence, per agent and per day. `trajet` = every eligible arrival suffers
# the event (a traffic jam lasts all day); `jour` = only the FIRST eligible arrival of
# the day suffers it (a repaired breakdown does not happen again identically three hours
# later). The 19 September run applied the same thirty-minute breakdown four times in
# the same day, where the protocol counted one: the real intensity was four times
# the announced intensity, and nothing said so.
CADENCES: tuple[str, ...] = ("trajet", "jour")
CADENCE_PAR_DEFAUT = "trajet"

# Ticket 111 — how the reader passes it on to its household. One mode delivered: a message per
# member, written by the reader in its own words, and for a minor the parents' decision.
RELAIS_MODES: tuple[str, ...] = ("par_destinataire",)


class RefusDEvenement(ValueError):
    """Invalid declaration. Refusing at load time beats a ghost event during the run."""


@dataclass(frozen=True)
class EffetPhysique:
    """What the world inflicts, that day, on the exposed agent.

    `None` in place of this object means that the world does not move — this is the case of a
    read article. It is NOT the same thing as a zero effect, which says we measured and found
    zero. The distinction carries the whole difference between the two regimes, and it is lost if
    one stores it in an integer.
    """

    retard_min: int = 0
    incident_reseau: bool = True
    correspondance_ratee: bool = False

    @property
    def retard_s(self) -> int:
        return int(self.retard_min) * 60


@dataclass(frozen=True)
class JourDEvenement:
    """What the event lays down on a given day.

    The profile is declared DAY BY DAY and not by a duration plus an intensity: a traffic jam
    is not the same on the first and the third day, and the agent must be able to say so in its
    own words. It is also what allows a decline — 60, 40, 25 minutes — that nothing else could
    produce.
    """

    jour: int  # rank in the run, 1 = first simulated day
    texte: str
    effet: EffetPhysique | None = None

    # ── 079 compatibility, for one version ───────────────────────────────────────────────
    # These four properties exist so that `llm/chocs.py` stays a THIN adapter and
    # the 36 tests of ticket 079 run without their expected values moving by one character. They
    # go with `chocs.py`, in lot 6.
    @property
    def vecu(self) -> str:
        return self.texte

    @property
    def retard_min(self) -> int:
        return self.effet.retard_min if self.effet else 0

    @property
    def retard_s(self) -> int:
        return self.effet.retard_s if self.effet else 0

    @property
    def incident_reseau(self) -> bool:
        return bool(self.effet.incident_reseau) if self.effet else False

    @property
    def correspondance_ratee(self) -> bool:
        return bool(self.effet.correspondance_ratee) if self.effet else False


@dataclass(frozen=True)
class TexteCite:
    """A text we QUOTE, with what it takes to prove we did not rewrite it.

    This is the form of the `lu` channel: a press article, the same for all readers and all
    days. It contrasts with the text declared day by day, which we write ourselves and
    which has no source to betray.

    `mention` is carried INSIDE the entry, visible to the model — "Translated from
    French". It is not a repository comment: it is information the agent has, and the
    setup has no reason to hide it from it (059, lot 1).
    """

    fichier: str
    sha256: str
    contenu: str
    mention: str = ""

    @property
    def servi(self) -> str:
        """The text as it enters memory, mention included."""
        if not self.mention:
            return self.contenu
        return f"({self.mention})\n{self.contenu}"


@dataclass(frozen=True)
class Calendrier:
    """When the event reaches a given target.

    Two forms. `jours`: the same days for everyone, it is the shock of 079. `fenetre`
    plus `graine`: a day DRAWN per target, within the declared window — two households do not read
    on the same morning, and a calendar effect can no longer be confused with that of
    the article.
    """

    jours: tuple[int, ...] = ()
    fenetre: tuple[int, int] | None = None
    graine: int = 59

    def jour_de(self, evenement_id: str, cible: str) -> int:
        """The day of THIS target. Without a window, the first declared day."""
        if self.fenetre is not None:
            return jour_tire(self.graine, evenement_id, str(cible), self.fenetre)
        return min(self.jours)

    @property
    def premier_jour_possible(self) -> int:
        return self.fenetre[0] if self.fenetre is not None else min(self.jours)

    @property
    def dernier_jour_possible(self) -> int:
        return self.fenetre[1] if self.fenetre is not None else max(self.jours)


@dataclass(frozen=True)
class Exposition:
    """Who is affected.

    `mode` — any agent whose arriving trip was made in one of the declared modes.
    `tirage` — a share of the agents, drawn DETERMINISTICALLY and stable from one run to the next.
    `agents` — named identifiers, for a reproducible individual incident.
    `foyers` — one or several members per designated household (lot 2). `lecteurs`, if
    declared, names who reads in each household instead of the draw (2026-09-25).
    """

    regle: str
    modes: frozenset[str] = frozenset()
    part: float = 1.0
    graine: int = 79
    agents: frozenset[str] = frozenset()
    foyers: frozenset[str] = frozenset()
    lecteurs_par_foyer: int = 1
    lecteurs: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Service:
    """How long the article remains GUARANTEED in the prompt — ticket 111.

    Counted in TRAVEL days: the reading day is day 1, and weekends do not
    count when no departure takes place at weekends. During these days, the line is
    served whatever the judged severity; afterwards, memory decides alone — that is what we
    measure.
    """

    jours_de_deplacement: int


@dataclass(frozen=True)
class Relais:
    """What the reader tells its household — ticket 111."""

    mode: str
    # Some protocols measure the effect of information actually spread through the whole
    # household. In that case the model keeps its words, but silence is not a modality of the
    # treatment: each recipient must receive a non-empty message.
    parole_obligatoire: bool = False


@dataclass(frozen=True)
class Evenement:
    evenement_id: str
    libelle: str
    source: str | None
    canal: str
    moment: str
    jugement: str
    exposition: Exposition
    jours: dict[int, JourDEvenement]
    # Form B (`lu` channel): a single, quoted text, and a calendar drawn per target. `jours`
    # is then empty, and `calendrier` says when each one receives it.
    texte_cite: TexteCite | None = None
    calendrier: Calendrier | None = None
    cadence: str = CADENCE_PAR_DEFAUT
    empreinte: str = ""
    # Format of the file this declaration comes from: "100", or "079" for a shock
    # declaration read through the compatibility path. Logged, never interpreted: it serves to
    # know, when re-reading a run, in which form the protocol had been written.
    format_source: str = "100"
    # Ticket 111. `None` = key absent: behaviour from before the ticket, logged at
    # load time (a silent file must not change behaviour silently).
    service: Service | None = None
    relais: Relais | None = None

    @property
    def premier_jour(self) -> int:
        if self.calendrier is not None:
            return self.calendrier.premier_jour_possible
        return min(self.jours)

    @property
    def dernier_jour(self) -> int:
        if self.calendrier is not None:
            return self.calendrier.dernier_jour_possible
        return max(self.jours)

    @property
    def texte_unique(self) -> bool:
        """A single text for everyone and every day (`lu` channel), or one text per day?"""
        return self.texte_cite is not None

    @property
    def choc_id(self) -> str:
        """079 compatibility — goes in lot 6 with `llm/chocs.py`."""
        return self.evenement_id


@dataclass(frozen=True)
class EvenementApplique:
    """What an agent actually suffers or reads, at a given injection."""

    evenement_id: str
    canal: str
    moment: str
    jour_run: int
    jour_relatif: int
    retard_injecte_s: int
    texte: str
    incident_reseau: bool
    correspondance_ratee: bool
    raison: str

    @property
    def choc_id(self) -> str:
        """079 compatibility — goes in lot 6."""
        return self.evenement_id

    @property
    def vecu(self) -> str:
        """079 compatibility — goes in lot 6."""
        return self.texte


def _modes_admis() -> frozenset[str]:
    """The canonical modes, DERIVED from the repository hierarchy and never copied here.

    A list written in this module would diverge from the hierarchy the day it moves, without
    anything saying so — this is the lesson of lot A of ticket 077, where two mode vocabularies
    coexisted without anyone knowing.
    """
    from llm.axes import mode_canonique

    candidats = (
        "walking", "cycling", "car", "public_transport", "train", "motorbike",
    )
    return frozenset(m for m in candidats if mode_canonique(m) == m)


def _lire_exposition(brut: dict, evenement_id: str) -> Exposition:
    regle = str(brut.get("regle") or "").strip()
    if regle not in REGLES_EXPOSITION:
        raise RefusDEvenement(
            f"event « {evenement_id} »: unknown exposure rule « {regle} » — "
            f"expected one of {REGLES_EXPOSITION}"
        )
    if regle not in REGLES_LIVREES:
        raise RefusDEvenement(
            f"event « {evenement_id} »: the exposure rule « {regle} » is declared "
            f"but not delivered yet — it comes with batch 2 of ticket 100, with "
            f"`Person.household_id`. Use `agents` in the meantime, or wait for the batch"
        )

    modes = frozenset(str(m).strip() for m in (brut.get("modes") or []) if str(m).strip())
    admis = _modes_admis()
    inconnus = modes - admis
    if inconnus:
        raise RefusDEvenement(
            f"event « {evenement_id} »: mode(s) {sorted(inconnus)} outside the repository's "
            f"hierarchy — expected among {sorted(admis)}"
        )
    if regle == "mode" and not modes:
        raise RefusDEvenement(
            f"event « {evenement_id} »: `mode` rule without any declared mode"
        )

    part = float(brut.get("part", 1.0))
    if regle == "tirage" and not (0.0 < part <= 1.0):
        raise RefusDEvenement(
            f"event « {evenement_id} »: `part` = {part} outside ]0, 1] — a draw that "
            f"reaches nobody or more than everybody makes no sense"
        )

    agents = frozenset(str(a).strip() for a in (brut.get("agents") or []) if str(a).strip())
    if regle == "agents" and not agents:
        raise RefusDEvenement(
            f"event « {evenement_id} »: `agents` rule without any identifier"
        )

    foyers = frozenset(str(f).strip() for f in (brut.get("foyers") or []) if str(f).strip())

    # DESIGNATED readers (2026-09-25): when the article targets a mode, the reader must be
    # a user of it, and a draw among the adults may land on the one who never takes it.
    lecteurs = frozenset(str(a).strip() for a in (brut.get("lecteurs") or []) if str(a).strip())
    lecteurs_par_foyer = int(brut.get("lecteurs_par_foyer", 1))
    if lecteurs and regle != "foyers":
        raise RefusDEvenement(
            f"event « {evenement_id} »: `lecteurs` declared with the `{regle}` rule — it "
            f"only designates someone under the `foyers` rule, and would have no effect"
        )
    if lecteurs and lecteurs_par_foyer != 1:
        raise RefusDEvenement(
            f"event « {evenement_id} »: `lecteurs` and `lecteurs_par_foyer` = "
            f"{lecteurs_par_foyer} both say how many read — designating the readers is enough"
        )

    return Exposition(
        regle=regle,
        modes=modes,
        part=part,
        graine=int(brut.get("graine", 79)),
        agents=agents,
        foyers=foyers,
        lecteurs_par_foyer=lecteurs_par_foyer,
        lecteurs=lecteurs,
    )


def _lire_cadence(brut: Any, evenement_id: str) -> str:
    cadence_brute = str(brut or "").strip().lower()
    if cadence_brute and cadence_brute not in CADENCES:
        raise RefusDEvenement(
            f"event « {evenement_id} »: unknown `cadence` « {cadence_brute} » — "
            f"expected {' or '.join(CADENCES)}. A misspelt cadence would silently change "
            f"the intensity of the event."
        )
    cadence = cadence_brute or CADENCE_PAR_DEFAUT
    logger.info(
        f"[evenements] « {evenement_id} »: cadence « {cadence} »"
        + ("" if cadence_brute else f" (not declared, default value {CADENCE_PAR_DEFAUT})")
    )
    return cadence


def _lire_jours(brut: list, evenement_id: str, canal: str) -> dict[int, JourDEvenement]:
    if not brut:
        raise RefusDEvenement(
            f"event « {evenement_id} »: no day declared in `jours`"
        )
    jours: dict[int, JourDEvenement] = {}
    for entree in brut:
        jour = int(entree.get("jour", 0))
        if jour < 1:
            raise RefusDEvenement(
                f"event « {evenement_id} »: `jour` = {jour} — the first day of the run "
                f"is number 1"
            )
        if jour in jours:
            raise RefusDEvenement(
                f"event « {evenement_id} »: two entries for day {jour} — a "
                f"day carries one profile and only one"
            )
        # `texte` is the name of ticket 100, `vecu` that of 079. Both are read; a file
        # carrying both is refused, because we would not know which one served.
        if "texte" in entree and "vecu" in entree:
            raise RefusDEvenement(
                f"event « {evenement_id} », day {jour}: `texte` AND `vecu` declared — "
                f"keep `texte` (ticket 100) or `vecu` (ticket 079), not both"
            )
        texte = str(entree.get("texte", entree.get("vecu", "")) or "").strip()
        verifier_texte(texte, evenement_id, jour, RefusDEvenement)

        retard_min = int(entree.get("retard_min", 0))
        if retard_min < 0:
            raise RefusDEvenement(
                f"event « {evenement_id} », day {jour}: `retard_min` = {retard_min} — a "
                f"negative delay would be an advance, which the severity cannot represent"
            )
        # A `canal: lu` touches neither OTP, nor GTFS, nor OSMnx, and inflicts no delay: the
        # world does not change because one read the paper. The physical effect is therefore
        # ABSENT, and not zero — the distinction is that of § 9 of the ticket.
        effet = None
        if canal == "vecu":
            effet = EffetPhysique(
                retard_min=retard_min,
                incident_reseau=bool(entree.get("incident_reseau", True)),
                correspondance_ratee=bool(entree.get("correspondance_ratee", False)),
            )
        elif retard_min or entree.get("incident_reseau") or entree.get("correspondance_ratee"):
            raise RefusDEvenement(
                f"event « {evenement_id} », day {jour}: physical effect declared on a "
                f"`canal: lu`. Reading the newspaper delays nobody — the world does not change "
                f"before the decision (§ 9 of ticket 100)"
            )
        jours[jour] = JourDEvenement(jour=jour, texte=texte, effet=effet)
    return jours


def _lire_texte_cite(brut: dict, evenement_id: str) -> TexteCite:
    """The quoted text, read ONCE at load time and checked twice."""
    chemin = str(brut.get("fichier") or "").strip()
    if not chemin:
        raise RefusDEvenement(
            f"event « {evenement_id} »: `texte.fichier` missing. A `lu` channel cites a "
            f"text of the repository, it does not carry it in its declaration"
        )
    contenu = verifier_empreinte(
        chemin, str(brut.get("sha256") or ""), evenement_id, RefusDEvenement
    )
    # Declared exemptions, on the model of the forbidden words of the paraphrase (059, lot 1):
    # a marker is exempted only if it APPEARS in the text, and never without a written reason.
    # Without the first guard, absent markers would be exempted as a precaution and the list
    # would lose its meaning without anyone seeing it.
    exemptes: dict[str, str] = {}
    for entree in brut.get("marqueurs_exemptes") or []:
        marqueur = str(entree.get("marqueur") or "").strip().lower()
        motif = str(entree.get("motif") or "").strip()
        if not marqueur:
            raise RefusDEvenement(
                f"event « {evenement_id} »: an exemption without a marker"
            )
        if not motif:
            raise RefusDEvenement(
                f"event « {evenement_id} »: the marker « {marqueur} » is exempted without "
                f"a reason. An exemption that does not say why cannot be reviewed"
            )
        if marqueur not in contenu.lower():
            raise RefusDEvenement(
                f"event « {evenement_id} »: the marker « {marqueur} » is exempted but "
                f"DOES NOT APPEAR in the cited text. One does not exempt what is not there — "
                f"a list of preventive exemptions would lose its meaning without anything "
                f"signalling it"
            )
        exemptes[marqueur] = motif

    # The content guards of 079 apply ALSO to the quoted text. An article telling the
    # reader which mode to take would fabricate the result we claim to measure, exactly
    # like an experience that concludes in place of the agent. A single list for both channels.
    verifier_texte(
        contenu, evenement_id, 0, RefusDEvenement,
        exemptes=exemptes,
        attendre_premiere_personne=False,
    )
    return TexteCite(
        fichier=chemin,
        sha256=str(brut.get("sha256") or "").strip().lower(),
        contenu=contenu.strip(),
        mention=str(brut.get("mention") or "").strip(),
    )


def _lire_calendrier(brut: dict, evenement_id: str) -> Calendrier:
    jours = tuple(int(j) for j in (brut.get("jours") or []))
    fenetre_brute = brut.get("fenetre_jours")
    if jours and fenetre_brute:
        raise RefusDEvenement(
            f"event « {evenement_id} »: `calendrier.jours` AND `calendrier.fenetre_jours` "
            f"declared — a fixed day and a drawn day are two protocols, not two ways "
            f"of writing the same one"
        )
    if fenetre_brute:
        if len(fenetre_brute) != 2:
            raise RefusDEvenement(
                f"event « {evenement_id} »: `fenetre_jours` expects two bounds, "
                f"« {fenetre_brute} » carries {len(fenetre_brute)}"
            )
        debut, fin = int(fenetre_brute[0]), int(fenetre_brute[1])
        if debut < 1:
            raise RefusDEvenement(
                f"event « {evenement_id} »: `fenetre_jours` starts at day {debut} — the "
                f"first day of the run is number 1"
            )
        if fin < debut:
            raise RefusDEvenement(
                f"event « {evenement_id} »: `fenetre_jours` = [{debut}, {fin}] is empty. "
                f"An empty window draws no day, and the event would happen for "
                f"nobody without any symptom appearing"
            )
        return Calendrier(fenetre=(debut, fin), graine=int(brut.get("graine", 59)))
    if not jours:
        raise RefusDEvenement(
            f"event « {evenement_id} »: `calendrier` without `jours` or `fenetre_jours`"
        )
    if any(j < 1 for j in jours):
        raise RefusDEvenement(
            f"event « {evenement_id} »: `calendrier.jours` carries a day < 1 — the first "
            f"day of the run is number 1"
        )
    return Calendrier(jours=jours, graine=int(brut.get("graine", 59)))


def _lire_service(brut: Any, evenement_id: str) -> Service:
    if not isinstance(brut, dict):
        raise RefusDEvenement(
            f"event « {evenement_id} »: `service` expects a block carrying "
            f"`jours_de_deplacement`, not « {brut} »"
        )
    valeur = brut.get("jours_de_deplacement")
    # `bool` is an `int` in Python: `True` would pass for one day. Refused like the rest.
    if isinstance(valeur, bool) or not isinstance(valeur, int) or valeur < 1:
        raise RefusDEvenement(
            f"event « {evenement_id} »: `service.jours_de_deplacement` = {valeur!r} — "
            f"expected an integer ≥ 1. Zero days of service would amount to guaranteeing nothing "
            f"while appearing to"
        )
    return Service(jours_de_deplacement=valeur)


def _lire_relais(brut: Any, evenement_id: str) -> Relais:
    mode = str((brut or {}).get("mode") or "").strip() if isinstance(brut, dict) else ""
    if mode not in RELAIS_MODES:
        raise RefusDEvenement(
            f"event « {evenement_id} »: unknown `relais.mode` « {mode} » — expected "
            f"{' or '.join(RELAIS_MODES)}"
        )
    parole = brut.get("parole_obligatoire", False)
    if not isinstance(parole, bool):
        raise RefusDEvenement(
            f"event « {evenement_id} »: `relais.parole_obligatoire` = {parole!r} — "
            f"expected a boolean"
        )
    return Relais(mode=mode, parole_obligatoire=parole)


def _refuser_les_champs_des_lots_suivants(data: dict, evenement_id: str) -> None:
    """A field of lot 2 or lot 3 in a repository at lot 1: refused, and the lot is named.

    Accepting it while ignoring it would be the worst of the three options: the declaration
    would seem to say something the code does not do, and a run would start on a protocol that
    is not the one that was written.
    """
    if "effet_physique" in data:
        raise RefusDEvenement(
            f"event « {evenement_id} »: a top-level `effet_physique` is not read — the effect is "
            f"declared day by day, under `jours`, because it changes from one day to the next"
        )
    if "gravite" in data:
        raise RefusDEvenement(
            f"event « {evenement_id} »: a `gravite` field is set by hand. The severity "
            f"is COMPUTED — from what the simulation measured, and from what the agent judges of it. "
            f"Setting it here would restore the parameter that decision D4 removes, and would amount "
            f"to measuring one's own instruction"
        )


def _depuis_le_format_079(data: dict, chemin: Path) -> dict:
    """A shock declaration of ticket 079, re-read as an event.

    The path exists for a precise reason: the golden test of lot 1 must be able to replay THE
    ORIGINAL FILE, not its translation. A migration verified only on rewritten files
    verifies only the rewriting.
    """
    logger.warning(
        f"[evenements] {chemin.name} is in the ticket 079 format (key `choc:`). It is read and "
        f"converted — channel `vecu`, moment `arrivee`, judgement `aucun`. This "
        f"compatibility path goes with `llm/chocs.py`, in lot 6 of ticket 100: migrate the file "
        f"to `config/evenements/` when you touch it."
    )
    converti = dict(data)
    converti["evenement"] = data.get("choc")
    converti.pop("choc", None)
    converti.setdefault("canal", "vecu")
    converti.setdefault("moment", "arrivee")
    converti.setdefault("jugement", "aucun")
    return converti


def charger(chemin: str | Path) -> Evenement:
    """Reads and CHECKS an event declaration. Raises `RefusDEvenement` at the slightest doubt.

    Accepts both formats: `evenement:` (ticket 100) and `choc:` (ticket 079, converted on
    the fly with a warning naming the file).
    """
    p = Path(chemin)
    if not p.is_file():
        raise RefusDEvenement(f"event file not found: {p}")
    brut = p.read_bytes()
    empreinte = hashlib.sha256(brut).hexdigest()
    data: dict[str, Any] = yaml.safe_load(brut.decode("utf-8")) or {}

    format_source = "100"
    if "choc" in data and "evenement" not in data:
        data = _depuis_le_format_079(data, p)
        format_source = "079"

    evenement_id = str(data.get("evenement") or "").strip()
    if not evenement_id:
        raise RefusDEvenement(
            f"{p}: field `evenement` (identifier) missing or empty — a ticket 079 "
            f"declaration carries `choc:` instead"
        )

    _refuser_les_champs_des_lots_suivants(data, evenement_id)

    canal = str(data.get("canal") or "vecu").strip().lower()
    if canal not in CANAUX:
        raise RefusDEvenement(
            f"event « {evenement_id} »: unknown `canal` « {canal} » — expected "
            f"{' or '.join(CANAUX)}"
        )
    moment = str(data.get("moment") or "arrivee").strip().lower()
    if moment not in MOMENTS:
        raise RefusDEvenement(
            f"event « {evenement_id} »: unknown `moment` « {moment} » — expected "
            f"{' or '.join(MOMENTS)}"
        )
    jugement = str(data.get("jugement") or "aucun").strip().lower()
    if jugement not in JUGEMENTS:
        raise RefusDEvenement(
            f"event « {evenement_id} »: unknown `jugement` « {jugement} » — expected "
            f"{' or '.join(JUGEMENTS)}"
        )
    exposition = _lire_exposition(data.get("exposition") or {}, evenement_id)
    cadence = _lire_cadence(data.get("cadence"), evenement_id)

    # ── The two forms of text, and what separates them ───────────────────────────────────
    texte_cite = None
    calendrier = None
    jours: dict[int, JourDEvenement] = {}
    if "texte" in data:
        texte_cite = _lire_texte_cite(data.get("texte") or {}, evenement_id)
        if data.get("jours"):
            raise RefusDEvenement(
                f"event « {evenement_id} »: `texte` (a cited text) AND `jours` (one text "
                f"per day) declared. An article is the same every morning; a lived experience changes "
                f"from one day to the next. Choose the form of the regime you play"
            )
        calendrier = _lire_calendrier(data.get("calendrier") or {}, evenement_id)
    else:
        if canal == "lu":
            raise RefusDEvenement(
                f"event « {evenement_id} »: a `canal: lu` cites a text — declare a "
                f"`texte` block with its file and its fingerprint. An article written by us "
                f"in the declaration would not be local press, it would be us"
            )
        jours = _lire_jours(data.get("jours") or [], evenement_id, canal)
        calendrier = Calendrier(jours=tuple(sorted(jours)))

    # ── Cross refusals, and their reasons ───────────────────────────────────────────────
    if moment == "reveil" and any(j.effet is not None for j in jours.values()):
        raise RefusDEvenement(
            f"event « {evenement_id} »: physical effect declared on an injection at "
            f"WAKE-UP. The world does not change before the decision: what the agent knows when "
            f"choosing inflicts no delay on it, otherwise the day of the event "
            f"would measure both an anticipation and a constraint (§ 9 of ticket 100)"
        )
    if canal == "lu" and moment != "reveil":
        raise RefusDEvenement(
            f"event « {evenement_id} »: a `canal: lu` set at moment « {moment} ». An "
            f"article is read BEFORE deciding — that is what separates it from the shock, and the whole "
            f"contrast that chapter 7 measures"
        )
    if exposition.regle == "foyers" and not exposition.foyers:
        raise RefusDEvenement(
            f"event « {evenement_id} »: `foyers` rule without any household identifier"
        )
    if exposition.regle == "foyers" and moment != "reveil":
        raise RefusDEvenement(
            f"event « {evenement_id} »: the `foyers` rule draws readers, and a "
            f"reader reads at wake-up. On an arrival, use `agents` or `mode`"
        )

    # ── Ticket 111: guaranteed service and household relay ──────────────────────────────
    service = None
    relais = None
    if "service" in data:
        if moment != "reveil":
            raise RefusDEvenement(
                f"event « {evenement_id} »: `service` on an event of moment "
                f"« {moment} ». The guaranteed service bears on what is known BEFORE deciding; "
                f"a shock applies after the decision, and that is intended"
            )
        service = _lire_service(data.get("service"), evenement_id)
    if "relais" in data:
        if canal != "lu":
            raise RefusDEvenement(
                f"event « {evenement_id} »: `relais` on a `canal: {canal}`. Only a "
                f"READ article is told to the household the same morning"
            )
        if exposition.regle != "foyers":
            raise RefusDEvenement(
                f"event « {evenement_id} »: `relais` without the `foyers` exposure "
                f"rule. The relay goes from the reader to the other members of THEIR household: without "
                f"a drawn household, it has no recipient"
            )
        if service is None:
            raise RefusDEvenement(
                f"event « {evenement_id} »: `relais` without `service`. The message received is "
                f"served during the recipient's days of service; without their number, it "
                f"would have no guaranteed presence, and nothing would say so"
            )
        relais = _lire_relais(data.get("relais"), evenement_id)
    if moment == "reveil":
        absentes = [c for c, v in (("service", service), ("relais", relais)) if v is None]
        if absentes:
            logger.info(
                f"[evenements] « {evenement_id} » : clé(s) {absentes} absente(s) — "
                f"comportement d'avant le ticket 111 : "
                + ("aucune présence garantie au prompt, la gravité et le rappel décident seuls"
                   if service is None else f"service {service.jours_de_deplacement} j")
                + ("" if relais is not None else ", aucun relais au foyer")
                + "."
            )
        else:
            logger.info(
                f"[evenements] « {evenement_id} »: served {service.jours_de_deplacement} travel "
                f"day(s) to the reader and to the informed, relay « {relais.mode} »"
            )

    # ── Judgement: accepted at reading, refused when arming a run (lot 3) ───────────────
    if canal == "lu" and jugement == "aucun":
        # In figures, not in arguments: an article inflicts no delay, its deterministic
        # severity is 0.00, and `force_initiale(0.00)` is 2.8 days. The entry leaves the
        # « Ce qui a changé récemment » block two days later, and its silence would pass for
        # an absence of effect — whereas it only measures the lifetime it was given.
        logger.error(
            f"[ALARME] [evenements] « {evenement_id} » : canal `lu` SANS jugement. L'article "
            f"entrera en mémoire avec une gravité de 0,00, donc une durée de vie de 2,8 jours "
            f"— il aura disparu du prompt le surlendemain, et aucune mesure longitudinale "
            f"n'est possible. Cette combinaison sert à exercer l'injection, pas à jouer une "
            f"campagne. Pour une campagne : `jugement: a_l_injection` (lot 3)."
        )

    return Evenement(
        evenement_id=evenement_id,
        libelle=str(data.get("libelle") or evenement_id),
        source=data.get("source"),
        canal=canal,
        moment=moment,
        jugement=jugement,
        exposition=exposition,
        jours=jours,
        texte_cite=texte_cite,
        calendrier=calendrier,
        cadence=cadence,
        empreinte=empreinte,
        format_source=format_source,
        service=service,
        relais=relais,
    )
