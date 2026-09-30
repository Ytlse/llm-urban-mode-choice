"""A single event channel, for the experienced and the read — ticket 100.

WHAT THIS PACKAGE REPLACES, AND WHY
-----------------------------------
A shock suffered on a trip and an article read in the morning came in through two distinct
code paths: `llm/chocs.py`, delivered in ticket 079, and `llm/informations.py`, planned in lot 4
of ticket 059 as **its copy** — same content guards, same exposure, same counters, same
run day, same trace, same alarm. Yet everything downstream of them was already shared: severity,
force, service duration, consolidation, belief, contradiction, household.

This package lays down the single channel, with two injection points:

- `arrivee` — after the decision, with a measured fact. The agent chose while seeing the nominal
  offer, then takes the hit. It is the shock of ticket 079.
- `reveil` — before the first decision, with nothing measured. The agent knows before choosing.
  It is the article of ticket 059. Delivered in lot 2.

WHAT THIS PACKAGE DOES NOT DO, AND THAT IS THE POINT
----------------------------------------------------
It cuts no line, degrades no frequency, touches neither OTP, nor GTFS, nor OSMnx. The
world stays nominal; what changes is what the agent knows of it and what it keeps. An
event that also degraded the offer would inextricably mix adaptation to the constraint with
the inertia of memory, and neither would be measurable.

THE NON-NEGOTIABLE RULE
-----------------------
The **injected** delay and the **measured** delay are never confused. Two variables, two
columns, two fields in the trace. Without this separation, no re-reading could any longer
tell what the simulation produced from what it was made to say.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from llm.evenements.declaration import (  # noqa: F401 — façade du paquet
    CADENCES,
    CADENCE_PAR_DEFAUT,
    CANAUX,
    JUGEMENTS,
    JUGEMENTS_LIVRES,
    MOMENTS,
    Calendrier,
    TexteCite,
    Evenement,
    EvenementApplique,
    EffetPhysique,
    Exposition,
    JourDEvenement,
    RefusDEvenement,
    charger,
)
from llm.evenements.exposition import REGLES_EXPOSITION  # noqa: F401
from llm.evenements.injection import (  # noqa: F401
    PREFIXE_FOYER,
    a_l_arrivee,
    au_reveil,
    entree_de_lecture,
    joindre,
    ligne_de_foyer,
    ligne_de_lecture,
)
from llm.evenements.jugement import (  # noqa: F401
    Jugement,
    JugementRefuse,
    juger,
)
from llm.evenements.registre import CompteursJournee, RegistreEvenements  # noqa: F401
from llm.evenements.relais import Message, RelaisFoyer, RelaisRefuse  # noqa: F401
from settings import settings

# ── Process registry ────────────────────────────────────────────────────────────────────────
_registre: RegistreEvenements | None = None
_initialise = False


def registre() -> RegistreEvenements | None:
    return _registre


def incident_reseau_a_une_source() -> bool:
    """True as soon as a loaded event carries the `incident_reseau` component.

    This is what `llm/gravite.py` checks to stop declaring it inactive. Without an event,
    it stays declared inactive: a component without a source must keep saying so, otherwise
    zero would be confused with "perfect trip".
    """
    if _registre is None:
        return False
    return any(j.incident_reseau for j in _registre.evenement.jours.values())


def cache_coupe(timestamp: int) -> bool:
    """Must the decision cache be bypassed at this instant? (ticket 100, Q5)

    False when no event is declared — a nominal run keeps its whole cache.
    """
    if _registre is None:
        return False
    try:
        return _registre.cache_coupe(int(timestamp))
    except Exception:  # noqa: BLE001 — a guard never brings down a decision
        return False


def noter_decision(timestamp: int, depuis_cache: bool) -> None:
    """Counts a decision, and says whether the cache served it. Without an event, does nothing."""
    if _registre is None:
        return
    _registre.noter_decision(int(timestamp), depuis_cache)


async def lignes_du_jour(person_id: str, timestamp: int, *, compter: bool = True) -> list[str]:
    """The lines guaranteed to this agent's prompt for a trip at `timestamp` — ticket 111.

    Empty list without an event, outside service days, or for an unexposed agent. NEVER
    raises towards the caller: a line that cannot be computed must not cost the
    decision — but it does not vanish silently either, an [ALARME] says so.
    """
    if _registre is None:
        return []
    try:
        return await _registre.lignes_du_jour(str(person_id), int(timestamp), compter=compter)
    except Exception as err:  # noqa: BLE001
        logger.error(
            f"[ALARME] [evenements] service lines not computed for {person_id} at "
            f"{timestamp} ({type(err).__name__}: {err}) — the decision goes out WITHOUT the line "
            f"guaranteed by ticket 111."
        )
        return []


def noter_instant(timestamp: int) -> None:
    """The simulated instant of the current /sync. Without an event, does nothing."""
    if _registre is not None:
        _registre.noter_instant(int(timestamp))


def noter_rendu(person_id: str, timestamp: int, lignes, historique) -> None:
    """Checks that a decision does carry its service lines. Without an event, does nothing."""
    if _registre is not None and lignes:
        _registre.noter_rendu(str(person_id), int(timestamp), lignes, historique)


def noter_contournement_cache() -> None:
    if _registre is not None:
        _registre.noter_contournement_cache()


def noter_reflexion_presse(
    person_id: str,
    timestamp: int,
    lignes: list[str],
    *,
    etape: str,
    prise_en_compte: bool | None = None,
    reflection: str = "",
) -> None:
    """Traces the article shown to a reflection and the model's structured read receipt."""
    if _registre is not None and lignes:
        _registre.tracer_reflexion_presse(
            str(person_id),
            int(timestamp),
            lignes,
            etape=etape,
            prise_en_compte=prise_en_compte,
            reflection=reflection,
        )


def _declaration_demandee() -> str | None:
    """The file to load, `None` if none.

    Two configuration keys during one version: `evenements` (ticket 100) and `chocs`
    (ticket 079). The second is read with a warning, never silently — a campaign
    started under the old key must keep running, but whoever re-reads the log must
    know which one served.
    """
    bloc_100 = getattr(settings, "evenements", None)
    if bloc_100 is not None and getattr(bloc_100, "enabled", False):
        chemin = getattr(bloc_100, "fichier", None)
        if chemin:
            return str(chemin)
        logger.error(
            "[ALARME] [evenements] `evenements.enabled` is true but `evenements.fichier` is "
            "empty: NO event will be played, and the run will behave like a nominal run. "
            "Declare the file, or turn the flag off."
        )
        return None

    bloc_079 = getattr(settings, "chocs", None)
    if bloc_079 is not None and getattr(bloc_079, "enabled", False):
        chemin = getattr(bloc_079, "fichier", None)
        if chemin:
            logger.warning(
                "[evenements] declaration read under the `chocs:` key of ticket 079. It is still "
                "served; the `evenements:` key replaces it (ticket 100, lot 6)."
            )
            return str(chemin)
    return None


def initialiser(workdir: Path | None = None) -> RegistreEvenements | None:
    """Loads the declared event, if there is one. Without a file, NOTHING changes.

    The refusal is BLUNT: an invalid declaration stops the loading instead of letting a
    sixty-day run go on that will do nothing and of which nobody will know why.
    """
    global _registre, _initialise
    if _initialise:
        return _registre
    _initialise = True
    chemin = _declaration_demandee()
    if not chemin:
        logger.info(
            "[evenements] no event declared — the run behaves as without this mechanism."
        )
        return None
    evenement = charger(chemin)
    # ⚠ Reading and ARMING are two things. `charger()` checks a declaration —
    # fingerprints, exposure, calendar — and must be able to do so for a file describing
    # a protocol not yet playable; otherwise we could not even test that the five
    # articles of the corpus match their manifest. Arming a RUN is something else:
    # there, what is not delivered must stop the startup, not fail halfway.
    if evenement.canal == "lu" and evenement.jugement == "aucun":
        # Refusal at ARMING, not at reading. An article without judgement enters at severity
        # 0.00 and lives 2.8 days: it will have left the prompt two days later, and its
        # silence would pass for an absence of effect. The declaration stays readable — this
        # is what allows checking a corpus's fingerprints — but it does not arm a run.
        raise RefusDEvenement(
            f"event « {evenement.evenement_id} »: a `canal: lu` without judgement cannot "
            f"arm a run. The article would enter memory at severity 0.00, hence for "
            f"2.8 days, and no longitudinal measurement would be possible. Declare "
            f"`jugement: a_l_injection`."
        )
    journal = None
    if workdir is not None:
        journal = Path(workdir) / "evenements.jsonl"
        try:
            Path(workdir).mkdir(parents=True, exist_ok=True)
            declaration = Path(workdir) / "evenement.yaml"
            declaration.write_bytes(Path(chemin).read_bytes())
            # 079 compatibility: the extractors and the reports already written look for
            # `choc.yaml` and `chocs.jsonl` in the run directory. Two links, not two
            # copies — and they go in lot 6 with the rest of the aliases.
            for alias, cible in (("choc.yaml", declaration), ("chocs.jsonl", journal)):
                lien = Path(workdir) / alias
                if not lien.exists():
                    try:
                        lien.symlink_to(cible.name)
                    except OSError:
                        # File systems without symbolic links: a copy is better than
                        # nothing for the declaration; the journal would be recopied at every
                        # line — it stays under its new name, and the extractor will find it.
                        if alias.endswith(".yaml"):
                            lien.write_bytes(cible.read_bytes())
        except Exception as err:  # noqa: BLE001
            logger.warning(f"[evenements] declaration not archived in the run ({err})")
    _registre = RegistreEvenements(evenement, journal=journal)
    jours = ", ".join(
        f"j{j.jour}:+{j.retard_min}min"
        for j in sorted(evenement.jours.values(), key=lambda x: x.jour)
    )
    logger.info(
        f"[evenements] « {evenement.libelle} » ({evenement.evenement_id}) chargé — canal "
        f"{evenement.canal}, moment {evenement.moment}, jugement {evenement.jugement}, "
        f"exposition {evenement.exposition.regle}"
        + (f" {sorted(evenement.exposition.modes)}" if evenement.exposition.modes else "")
        + f", jours {evenement.premier_jour}→{evenement.dernier_jour} [{jours}], "
        f"empreinte {evenement.empreinte[:12]}, format {evenement.format_source}, "
        f"source : {evenement.source or 'non déclarée'}"
    )
    if getattr(settings.cache, "enabled", False):
        # Author's decision of 2026-09-22: the cache is cut ON THE DAY of the event, and
        # on that day only. Automatic, hence without depending on a `CACHE=0` one would remember
        # to set — the 19 September run showed what a guard costs that relies on the
        # operator's memory.
        jours = ", ".join(f"j{j}" for j in sorted(evenement.jours))
        logger.info(
            f"[evenements] cache de décisions ACTIF, coupé automatiquement les jours "
            f"d'événement ({jours}). Le reste du run le garde."
        )
        logger.warning(
            "[evenements] ⚠ the cut applies to the DAY of the event, not to the window "
            "AFTER it, which is the one measured. The exact cache key carries neither the date "
            "nor the memory: a decision taken before the event remains servable after it. What "
            "prevents it is the semantic branch (0.95 memory similarity), and "
            "nobody has measured it on this window. The daily counters "
            "« décision(s) servie(s) DEPUIS LE CACHE » say, day by day, how many "
            "decisions the caveat applies to. `make run CACHE=0` lifts it entirely."
        )
    return _registre


def reinitialiser() -> None:
    """Forgets the event. Reserved for tests and for the end of a run."""
    global _registre, _initialise
    if _registre is not None:
        _registre.journaliser_compteurs()
    _registre = None
    _initialise = False
