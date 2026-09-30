"""Core memory — ticket 071, lot 4.

The ten raw memories served to the model are replaced by a **structured permanent block**,
completed by two or three episodic entries recalled for the current decision. This is the
*working context* of MemGPT (Packer et al., 2023, § 2.1): a block always present in the
main context, the rest being paged on demand.

**All three blocks are COMPUTED, none is written by the model.** The specification only
computed the habits and entrusted knowledge to the model; since lot 3, a
concept carries its observation counter, its confidence and its service state, so that the
block can be computed exactly. The safeguard argument holds for all three: a text rewritten
periodically by a model drifts and invents, a computed block remains verifiable against its
source. Lot 4 therefore does not touch the self-reflection schema, and costs no inference.

**The block carries no metadata about itself**: neither update date nor number of days
of experience. This information changes no decision, costs tokens, and breaks the fiction
the templates maintain — a person does not think "my habit summary is three
days old". The useful age is already carried, statement by statement, by the observation counters.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta

from loguru import logger
from llm.gravite import force_initiale
from settings import settings

# One occurrence is not a habit. Below the threshold, the line is not written: saying "bike,
# 1 time out of 1" would give a random event the authority of a routine.
OCCURRENCES_MIN_HABITUDE = 3
# Beyond that, the block costs tokens without teaching the model anything.
HABITUDES_MAX = 4
CONNAISSANCES_MAX = 6

# The window of the "what changed recently" block and its line cap are SETTINGS
# (`memoire__fenetre_changements_jours`, `memoire__changements_max`) and not constants:
# the window decides how long a shock weighs on decisions, and it was
# hard-coded. Cf. ticket 077, lot I.

# Window of SET-ASIDE BELIEFS, unchanged since lot 4 of 071 and deliberately
# left as a constant: the ablation arm only varies the window of shock memories.
FENETRE_CROYANCES_ECARTEES_JOURS = 14

# Shock memories whose exit from the window has already been announced: (agent, memory instant).
# Rising edge — without it, the line would repeat at every decision and drown what it says.
_SORTIES_ANNONCEES: set[tuple[str, str]] = set()
_FENETRE_NEGATIVE_DITE = False

# ── The two service modes of the changes block (ticket 095, lot A) ──────────────
MODE_DERIVEE = "derivee"
MODE_FIXE = "fixe"
_MODES = (MODE_DERIVEE, MODE_FIXE)

# Out-of-domain settings already reported. An alarm that repeats at every decision stops being
# one; a silent alarm lets a fallback pass for an accepted setting.
_REGLAGES_FAUTIFS_DITS: set[str] = set()


def reinitialiser() -> None:
    """Forgets what has already been announced. Reserved for tests."""
    global _FENETRE_NEGATIVE_DITE
    _SORTIES_ANNONCEES.clear()
    _REGLAGES_FAUTIFS_DITS.clear()
    _FENETRE_NEGATIVE_DITE = False


def _alarmer_une_fois(cle: str, message: str) -> None:
    """Reports an out-of-domain setting ONCE, in ERROR: it is a drifting measurement."""
    if cle in _REGLAGES_FAUTIFS_DITS:
        return
    _REGLAGES_FAUTIFS_DITS.add(cle)
    logger.error(message)


def _fenetre_jours() -> int:
    """The window in force, read at EACH call.

    Frozen at import, an environment override — hence the ablation arm — would be
    useless.
    """
    global _FENETRE_NEGATIVE_DITE
    brut = int(settings.agent.memoire__fenetre_changements_jours)
    if brut < 0:
        if not _FENETRE_NEGATIVE_DITE:
            _FENETRE_NEGATIVE_DITE = True
            logger.warning(
                f"[noyau] fenêtre « ce qui a changé récemment » négative ({brut} j) — ramenée "
                f"à 0, donc aucun souvenir de choc dans le bloc. Si c'est l'ablation qui est "
                f"voulue, déclarer 0 ; une valeur négative ne doit pas se lire comme un réglage "
                f"accepté."
            )
        return 0
    return brut

def _mode_fenetre() -> str:
    """`derivee` or `fixe`, read at EACH call. An unknown mode is not interpreted."""
    brut = str(getattr(settings.agent, "memoire__mode_fenetre_changements", MODE_DERIVEE)).strip()
    if brut in _MODES:
        return brut
    _alarmer_une_fois(
        f"mode:{brut}",
        f"[ALARME] [noyau] mode de fenêtre « ce qui a changé récemment » inconnu ({brut!r}) — "
        f"repli sur {MODE_DERIVEE!r}. Modes admis : {list(_MODES)}. Un bras lancé sous ce "
        f"réglage ne mesure PAS ce qu'il déclare mesurer.",
    )
    return MODE_DERIVEE


def _seuil_service() -> float:
    """The weight threshold below which a memory leaves the block. Open domain ]0, 1[."""
    defaut = 0.35
    brut = float(getattr(settings.agent, "memoire__seuil_service_changement", defaut))
    if 0.0 < brut < 1.0:
        return brut
    _alarmer_une_fois(
        f"seuil:{brut}",
        f"[ALARME] [noyau] seuil de service du bloc de changements hors de ]0, 1[ ({brut}) — "
        f"repli sur {defaut}. À 1 la durée serait nulle (une ablation que personne n'a "
        f"déclarée), à 0 elle serait infinie.",
    )
    return defaut


def _bornes_duree() -> tuple[float, float]:
    """Floor and ceiling of the served duration, in days, put back in order if needed."""
    plancher = float(getattr(settings.agent, "memoire__plancher_changement_jours", 2.0))
    plafond = float(getattr(settings.agent, "memoire__plafond_changement_jours", 30.0))
    if plancher > plafond:
        _alarmer_une_fois(
            f"bornes:{plancher}:{plafond}",
            f"[ALARME] [noyau] plancher de durée ({plancher} j) supérieur au plafond "
            f"({plafond} j) — les deux bornes sont échangées. Sans cet échange, aucun souvenir "
            f"de choc ne serait jamais servi.",
        )
        plancher, plafond = plafond, plancher
    return max(0.0, plancher), max(0.0, plafond)


@dataclass(frozen=True)
class DureeService:
    """What is needed to justify an exit date, and not merely announce it."""

    jours: float        # the SERVED duration, bounds applied
    brute: float        # `force × ln(1/seuil)`, before bounds
    force: float        # the forgetting time constant of this memory, in days
    gravite: float
    borne: str          # "plancher", "plafond", or "" when no bound bit


def duree_service_jours(entree) -> DureeService:
    """How many days this shock memory stays served in "what changed recently".

    `poids(t) = exp(-t / force)` decreases; the memory is served as long as this weight exceeds
    the threshold, i.e. `t ≤ force × ln(1 / seuil)`. The shape comes from MemoryBank (Zhong et al.,
    2024) — it is the same as that of recall, and that is the point: the duration of an effect
    stops being an integer set beside the forgetting model and becomes a consequence of it.

    The `force` is the one CARRIED by the entry, which may have grown at recall. When missing —
    entry written before lot 1 of ticket 071 —, it is recomputed from severity rather than
    treated as zero: a zero force would give a zero duration, that is an ablation.
    """
    gravite = float(getattr(entree, "importance", 0.0) or 0.0)
    force = getattr(entree, "force", None)
    force = float(force) if force else force_initiale(gravite)
    brute = force * math.log(1.0 / _seuil_service())
    plancher, plafond = _bornes_duree()
    borne = ""
    jours = brute
    if jours > plafond:
        jours, borne = plafond, "plafond"
    elif jours < plancher:
        jours, borne = plancher, "plancher"
    return DureeService(
        jours=jours, brute=brute, force=force, gravite=gravite, borne=borne
    )


# The block is rendered in the prompt language: ENGLISH since 2026-09-25. Its titles had
# stayed in French in an otherwise all-English prompt. The prompt readers
# (`scripts/analysis/mesures/souvenir.py`, `scripts/analysis/tableau_quatre_voies.py`) accept
# both languages: archives from before that date carry the French titles.
TITRE_HABITUDES = "My habits"
TITRE_CONNAISSANCES = "What I know"
TITRE_CHANGEMENTS = "What changed recently"
TITRES_FRANCAIS = {
    TITRE_HABITUDES: "Mes habitudes",
    TITRE_CONNAISSANCES: "Ce que je sais",
    TITRE_CHANGEMENTS: "Ce qui a changé récemment",
}

_LIBELLE_MODE = {
    "walking": "on foot",
    "cycling": "by bike",
    "car": "by car",
    "public_transport": "by public transport",
    "train": "by train",
    "motorbike": "by motorbike",
}
_LIBELLE_CRENEAU = {
    "matin": "in the morning",
    "midi": "at midday",
    "soir": "in the evening",
    "nuit": "at night",
}


# ── Trip journal ─────────────────────────────────────────────────────────────────


def cle_journal(motif: str | None, creneau: str | None) -> str:
    """Journal key: the purpose-time slot pair, JSON-serialisable."""
    return f"{motif or '?'}|{creneau or '?'}"


def noter_trajet(
    journal: dict,
    motif: str | None,
    creneau: str | None,
    mode: str | None,
    retard_s: float = 0.0,
) -> dict:
    """Records a trip in an agent's journal.

    ⚠ This journal did not exist. `MoveLogger` writes to `moves.csv` and nothing else, and
    `PersonState` keeps no history: re-reading a CSV at every decision is ruled out, it is
    the critical path. It is therefore kept in memory and **persisted with the agent's
    metadata**, which reuses the deferred write already in place. Without persistence, a resumed
    run would restart without habits and the block would lie by omission on the first day.
    """
    if not mode:
        # A trip whose mode was not resolved does not count — but it does not make the
        # denominator drift either: it is not recorded at all.
        return journal
    clef = cle_journal(motif, creneau)
    entree = journal.setdefault(clef, {"modes": {}, "retards": 0, "total": 0})
    entree["modes"][mode] = entree["modes"].get(mode, 0) + 1
    entree["total"] += 1
    if retard_s >= 600:  # ten minutes
        entree["retards"] += 1
    return journal


def bloc_habitudes(journal: dict) -> list[str]:
    """"My habits", computed from the trip journal. Never written by the model."""
    lignes = []
    for clef, entree in sorted(
        journal.items(), key=lambda kv: kv[1].get("total", 0), reverse=True
    ):
        total = int(entree.get("total", 0))
        if total < OCCURRENCES_MIN_HABITUDE:
            continue
        modes = entree.get("modes") or {}
        if not modes:
            continue
        mode, n = max(modes.items(), key=lambda kv: kv[1])
        motif, creneau = clef.split("|", 1)
        libelle = (
            f"{motif} {_LIBELLE_CRENEAU.get(creneau, creneau)}: "
            f"{_LIBELLE_MODE.get(mode, mode)}, {n} times out of {total}"
        )
        retards = int(entree.get("retards", 0))
        if retards:
            libelle += f". {retards} delay(s) of more than 10 min"
        lignes.append(libelle)
        if len(lignes) >= HABITUDES_MAX:
            break
    return lignes


# ── What I know ──────────────────────────────────────────────────────────────────


def _enonce(entree) -> str:
    """Readable text of a concept, whatever the format it was written in."""
    contenu = entree.content or ""
    if contenu.startswith("["):
        try:
            cinq = json.loads(contenu)
            return str(cinq[0]) if cinq else contenu
        except (ValueError, IndexError):
            return contenu
    return contenu


def bloc_connaissances(entrees) -> list[str]:
    """"What I know", computed from the CONCEPTS and their counters.

    A concept taken out of service is not in it: we do not serve the model what the agent no
    longer believes. The most confident first.
    """
    concepts = [
        e for e in entrees
        if not e.est_episodique and e.est_servi and not e.depasse_le
    ]
    concepts.sort(
        key=lambda e: (e.confiance, e.observations, e.horodatage_de_reference),
        reverse=True,
    )
    return [
        f"{_enonce(e)}  ({e.observations} obs.)"
        for e in concepts[:CONNAISSANCES_MAX]
    ]


# ── What changed recently ────────────────────────────────────────────────────────


def _annoncer_sortie(
    person_id: str | None,
    dans: list,
    hors: list[tuple],
    mode: str,
    fenetre: int,
) -> None:
    """The day a shock memory leaves the block, and that day only.

    This is the event that, on the 2026-09-19 run, coincides to the prompt with the return of
    the car. It was invisible until now: one had to re-read the prompt text to
    reconstruct it, and the run report attributed it to another cause.

    ⚠ The line now carries **what produced the duration** — severity, force, computed duration,
    and the name of the bound when a bound bit — and no longer the date alone. A served duration
    without its cause cannot be checked afterwards: this is exactly what made indistinguishable,
    for two days, the decay of the memory and the window cut-off.
    """
    if not person_id or not hors:
        return
    for entree, duree in sorted(hors, key=lambda kv: kv[0].timestamp, reverse=True):
        cle = (str(person_id), entree.timestamp.isoformat())
        if cle in _SORTIES_ANNONCEES:
            continue
        _SORTIES_ANNONCEES.add(cle)
        if mode == MODE_FIXE:
            cause = f"fenêtre fixe de {fenetre} j"
        elif duree is None:
            cause = "durée indéterminée"
        else:
            cause = (
                f"durée {duree.jours:.2f} j dérivée d'une gravité de {duree.gravite:.2f} "
                f"(force {duree.force:.2f} j, durée calculée {duree.brute:.2f} j)"
            )
            if duree.borne:
                cause += f", ramenée par le {duree.borne}"
        reste = (
            "" if dans else " — plus aucun souvenir de choc ne pèse sur ses décisions"
        )
        logger.info(
            f"[noyau] {person_id} : le souvenir de choc du "
            f"{entree.timestamp:%Y-%m-%d} est sorti du bloc « ce qui a changé récemment » "
            f"({cause}){reste}."
        )


def bloc_changements(
    entrees,
    maintenant: datetime | None,
    person_id: str | None = None,
    lignes: tuple[str, ...] | list[str] = (),
) -> list[str]:
    """"What changed recently": the shocks, and the set-aside beliefs.

    Setting a concept aside is the observable the hysteresis experiment looks for.
    This is where it becomes readable in the prompt itself, and no longer only in the
    output statistics.

    ⚠ TWO windows, on purpose. The one for shock memories depends on the declared MODE — in
    `derivee`, it is computed memory by memory from severity; in `fixe`, it equals
    `memoire__fenetre_changements_jours` for all. The one for set-aside beliefs remains the
    historical constant: moving it at the same time would confound two changes in a
    single measurement.

    ⚠ The age of a memory is counted from `timestamp`, the instant of the EVENT, and not from
    `dernier_rappel`. This block announces the age of a CHANGE, not that of its last
    reading: a shock re-read yesterday is not yesterday's shock. Reinforcement at recall keeps
    playing, but through the `force`, hence on the duration — not by making the event younger.

    `person_id` only serves the log; it is optional so that pure calls stay pure.

    `lignes` (ticket 111) — what is GUARANTEED to the prompt that day: the article read, or what a
    household member said about it. They go FIRST, whatever their judged severity, and are
    never evicted by `memoire__changements_max`. A long-term memory entry whose
    content is identical to a line is not served a second time. Without `lignes`, the block
    is byte-for-byte identical to the one before the ticket.
    """
    garanties = [str(ligne) for ligne in (lignes or ()) if ligne]
    if maintenant is None:
        return garanties
    mode = _mode_fenetre()
    fenetre = _fenetre_jours()
    depuis_chocs = maintenant - timedelta(days=fenetre)
    depuis_croyances = maintenant - timedelta(days=FENETRE_CROYANCES_ECARTEES_JOURS)
    seuil_choc = float(settings.agent.memoire__importance_choc)
    lignes: list[tuple[datetime, str]] = []
    chocs_dans: list = []
    chocs_hors: list[tuple] = []

    for e in entrees:
        if garanties and e.content in garanties:
            continue
        if e.est_episodique:
            if float(e.importance or 0.0) >= seuil_choc:
                duree = None
                if mode == MODE_FIXE:
                    # Window at 0: declared ABLATION, no shock memory gets through — not even
                    # the one of the very instant, which `>= maintenant` would let in.
                    servi = fenetre > 0 and e.timestamp >= depuis_chocs
                else:
                    duree = duree_service_jours(e)
                    age_jours = (maintenant - e.timestamp).total_seconds() / 86400.0
                    servi = age_jours <= duree.jours
                if servi:
                    chocs_dans.append(e)
                    lignes.append((e.timestamp, _enonce(e)))
                else:
                    chocs_hors.append((e, duree))
            continue
        if e.depasse_le:
            try:
                quand = datetime.fromisoformat(e.depasse_le)
            except (TypeError, ValueError):
                continue
            if quand >= depuis_croyances:
                lignes.append(
                    (quand, f"I no longer believe that: {_enonce(e)}")
                )

    _annoncer_sortie(person_id, chocs_dans, chocs_hors, mode, fenetre)
    lignes.sort(key=lambda kv: kv[0], reverse=True)
    ordinaires = [texte for _, texte in lignes[: int(settings.agent.memoire__changements_max)]]
    if not garanties:
        return ordinaires
    return garanties + [t for t in ordinaires if t not in garanties]


# ── The complete block ───────────────────────────────────────────────────────────


def memoire_noyau(
    journal: dict,
    entrees,
    maintenant: datetime | None,
    person_id: str | None = None,
    lignes: tuple[str, ...] | list[str] = (),
) -> list[str]:
    """The permanent block, in lines ready for the template.

    An empty block is ABSENT and not present with a title: a title without content tells the model
    that there should be something there, and invites it to fill it.
    """
    sections = (
        (TITRE_HABITUDES, bloc_habitudes(journal or {})),
        (TITRE_CONNAISSANCES, bloc_connaissances(entrees or [])),
        (
            TITRE_CHANGEMENTS,
            bloc_changements(entrees or [], maintenant, person_id, lignes),
        ),
    )
    sortie: list[str] = []
    for titre, lignes in sections:
        if not lignes:
            continue
        sortie.append(titre)
        sortie.extend(f"- {ligne}" for ligne in lignes)
    return sortie
