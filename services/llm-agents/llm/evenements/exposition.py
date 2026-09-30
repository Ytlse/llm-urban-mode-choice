"""Who is affected — ticket 100, lot 1.

Three rules delivered, taken over as they are from ticket 079. The fourth, `foyers`, arrives in
lot 2: it needs `Person.household_id`, which lot 1 carries up to the runtime.

An unknown rule is **refused at load time** rather than ignored, otherwise a typo would produce
an event that affects nobody and an entire run without the slightest symptom.

The draw is DETERMINISTIC and independent of the order in which observations arrive. A draw that
depended on the order would turn two replays of the same scenario into two different experiments.
"""

from __future__ import annotations

import hashlib

# `foyers` is declared here so that the lot 1 refusal can name the rule and say where it
# arrives, instead of treating it as a typo.
REGLES_EXPOSITION: tuple[str, ...] = ("mode", "tirage", "agents", "foyers")
REGLES_LIVREES: tuple[str, ...] = ("mode", "tirage", "agents", "foyers")


def tirage_stable(graine: int, evenement_id: str, cible: str) -> float:
    """A reproducible real in [0, 1[, a function of the declaration and the target alone.

    Same formula as in 079 — eight hexadecimal digits of the SHA-256, divided by 0xFFFFFFFF.
    It does not change: changing it would redistribute the exposed agents of every campaign
    already played, and their internal witnesses with them.
    """
    empreinte = f"{graine}:{evenement_id}:{cible}"
    return int(hashlib.sha256(empreinte.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF


def expose(exposition, evenement_id: str, person_id: str, mode: str | None) -> tuple[bool, str]:
    """Is this agent exposed to this arrival, and for what reason?

    Always returns a REASON, including when the answer is no: the no is the internal witness
    of the run, and a witness one cannot name cannot be analysed.
    """
    if exposition.regle == "mode":
        if mode and mode in exposition.modes:
            return True, f"mode:{mode}"
        return False, f"mode:{mode or 'inconnu'}"

    if exposition.regle == "agents":
        if str(person_id) not in exposition.agents:
            return False, "non_designe"
        # `modes` was PARSED, validated, then ignored by this branch: an accepted field without
        # effect is worse than a refused one, because nothing reports it. When declared, it
        # restricts the designated agent to trips made in these modes. The case that requires
        # it: a CAR incident placed on a multimodal agent, who otherwise read « the engine made
        # a grinding noise » on the way back from a bus trip.
        if exposition.modes and (not mode or mode not in exposition.modes):
            return False, f"designe:mode:{mode or 'inconnu'}"
        return True, "designe"

    if exposition.regle == "foyers":
        # The `foyers` rule is not resolved on a single agent: it draws readers per household,
        # which requires the whole population. The registry did it once and for all
        # (`lecteurs()`), and what arrives here is the verdict already taken.
        raise RuntimeError(
            "the `foyers` rule is computed by `lecteurs(population)`, never agent by agent"
        )

    # tirage
    if tirage_stable(exposition.graine, evenement_id, str(person_id)) < exposition.part:
        return True, f"tirage:{exposition.part:.2f}"
    return False, "tirage:epargne"


# A press article is read by an ADULT of the household (decision of 2026-09-24): a nine-year-old
# child drawn as reader in a family of four would put the transmission measurement on an agent
# who does not decide the household's trips.
AGE_ADULTE = 18


def age_de(personne) -> int | None:
    """The age declared by the population (`identity.traits_json.age`), or None if missing."""
    traits = getattr(getattr(personne, "identity", None), "traits_json", None) or {}
    try:
        return int(traits.get("age"))
    except (TypeError, ValueError):
        return None


def lecteurs(exposition, evenement_id: str, population) -> dict[str, tuple[str, str]]:
    """Who reads, in which household, and for what reason — `foyers` rule (ticket 100, lot 2).

    Returns `{person_id: (household_id, reason)}`.

    ONE READER PER HOUSEHOLD by default (059 Q2), and this is what gives stage 3 its object: the
    unexposed co-resident is the INTERNAL WITNESS of the setup. If everyone read, there would be
    nobody left in whom to observe what is transmitted.

    The draw is deterministic and runs over the NUMERIC order of identifiers, like
    `extraire_sous_population.py`: an insertion order would turn two loads of the same
    cohort into two different experiments.

    DESIGNATED READERS (2026-09-25). If `exposition.lecteurs` is declared, it replaces the draw
    in ALL households: an article about the metro read by the only adult who never takes it
    measures nothing. A designated minor, immobile or absent reader is not replaced by a drawn
    one: the household stays without a reader, and the alarm says not to count it as exposed.
    """
    from loguru import logger

    designes = {str(d) for d in (getattr(exposition, "lecteurs", None) or ())}
    retenus: dict[str, tuple[str, str]] = {}
    par_foyer: dict[str, list] = {}
    for personne in population:
        foyer = getattr(personne, "household_id", None)
        if foyer and str(foyer) in exposition.foyers:
            par_foyer.setdefault(str(foyer), []).append(personne)

    for foyer in sorted(exposition.foyers):
        membres = par_foyer.get(foyer) or []
        mobiles = [p for p in membres if not getattr(p, "immobile", False)]
        if not mobiles:
            logger.error(
                f"[ALARME] [evenements] « {evenement_id} »: household {foyer} is declared "
                f"exposed but has NO mobile member in the loaded population "
                f"({len(membres)} member(s) present). Nobody will read there: do not count this "
                f"household as exposed in the analysis."
            )
            continue
        # Only adults read. A MISSING age does not exclude: we do not know, and dropping it
        # would silently empty the populations that do not carry it; it is reported instead.
        mineurs = [p for p in mobiles if (a := age_de(p)) is not None and a < AGE_ADULTE]
        sans_age = [str(p.person_id) for p in mobiles if age_de(p) is None]
        adultes = [p for p in mobiles if p not in mineurs]
        if sans_age:
            logger.warning(
                f"[evenements] « {evenement_id} »: household {foyer} — unknown age for "
                f"{sans_age}, admitted to the reader draw for lack of a way to exclude them."
            )
        if not adultes:
            logger.error(
                f"[ALARME] [evenements] « {evenement_id} » : le foyer {foyer} est déclaré "
                f"exposé mais ne compte AUCUN adulte mobile (âges "
                f"{[age_de(p) for p in mobiles]}). Personne n'y lira : ne pas compter ce foyer "
                f"comme exposé dans l'analyse."
            )
            continue
        if designes:
            _designes_du_foyer(designes, foyer, membres, adultes, evenement_id, retenus, logger)
            continue
        ordonnes = sorted(adultes, key=lambda p: (len(str(p.person_id)), str(p.person_id)))
        combien = max(1, min(int(exposition.lecteurs_par_foyer), len(ordonnes)))
        classes = sorted(
            ordonnes,
            key=lambda p: tirage_stable(
                exposition.graine, evenement_id, f"{foyer}:{p.person_id}"
            ),
        )
        for personne in classes[:combien]:
            retenus[str(personne.person_id)] = (foyer, f"foyer:{foyer}")
        ecartes = [str(p.person_id) for p in classes[combien:]]
        enfants = [str(p.person_id) for p in mineurs]
        logger.info(
            f"[evenements] « {evenement_id} » : foyer {foyer} — lecteur(s) "
            f"{[str(p.person_id) for p in classes[:combien]]}, co-résident(s) témoin(s) "
            f"{ecartes or 'aucun'}, mineur(s) hors tirage {enfants or 'aucun'}"
        )
    if designes:
        dans_les_foyers = {
            str(p.person_id) for membres in par_foyer.values() for p in membres
        }
        hors = sorted(designes - dans_les_foyers)
        if hors:
            logger.error(
                f"[ALARME] [evenements] « {evenement_id} » : lecteur(s) désigné(s) {hors} "
                f"absent(s) des foyers exposés {sorted(exposition.foyers)} de la population "
                f"chargée. Ils ne liront pas : vérifier la déclaration contre le MANIFEST."
            )
    if not retenus:
        logger.error(
            f"[ALARME] [evenements] « {evenement_id} »: `foyers` rule on "
            f"{sorted(exposition.foyers)} and NO reader retained. The event will happen "
            f"for nobody, and the entire run will unfold without the slightest symptom. "
            f"Check that the loaded population does carry `household.id`."
        )
    return retenus


def _designes_du_foyer(designes, foyer, membres, adultes, evenement_id, retenus, logger) -> None:
    """Retains the designated readers of this household, if they are mobile adults."""
    nommes = [p for p in membres if str(p.person_id) in designes]
    if not nommes:
        logger.error(
            f"[ALARME] [evenements] « {evenement_id} » : lecteurs désignés {sorted(designes)}, "
            f"AUCUN dans le foyer {foyer} (membres {[str(p.person_id) for p in membres]}). "
            f"Personne n'y lira, aucun tirage ne le remplace : ne pas compter ce foyer comme "
            f"exposé dans l'analyse."
        )
        return
    admis = [p for p in nommes if p in adultes]
    for p in nommes:
        if p not in admis:
            motif = "immobile" if getattr(p, "immobile", False) else f"mineur ({age_de(p)} ans)"
            logger.error(
                f"[ALARME] [evenements] « {evenement_id} » : lecteur désigné {p.person_id} du "
                f"foyer {foyer} écarté — {motif}. Aucun tirage ne le remplace."
            )
    for p in admis:
        retenus[str(p.person_id)] = (foyer, f"foyer:{foyer}")
    if admis:
        temoins = [str(p.person_id) for p in membres if p not in admis]
        logger.info(
            f"[evenements] « {evenement_id} » : foyer {foyer} — lecteur(s) DÉSIGNÉ(S) "
            f"{[str(p.person_id) for p in admis]}, co-résident(s) témoin(s) {temoins or 'aucun'}"
        )
