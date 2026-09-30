"""A single way of deciding (ticket 035, spec 02).

The simulator-free mode and the GAMA simulation call **these** functions, and nothing else, to
go from the raw proposals of a trip to the archived decision:

    eligibilite()  →  plafonner()  →  ordre_presentation()  →  decider()  →  avancer_chaine()

No rule of the vehicle chain is written here: everything is delegated to
`urban_mobility_agents.vehicle_chain` (D3). This module adds what the controller did not
produce: the **reason** for each exclusion, the deterministic presentation **order**, the complete
decision **trace** (D6) and the `Decideur` contract (D1). It reads neither the clock nor the
process time zone (D11): every time comes from the provided context.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Protocol

from models import Location, Person, TravelPlan
from settings import settings
from urban_mobility_agents.candidats import _select_candidates, _selection_group
from urban_mobility_agents.vehicle_chain import (
    _VEHICLE_MODES,
    MOTIF_PAS_DE_CONDUCTEUR,
    MOTIF_PLAFOND,
    MOTIF_RETOUR_FORCE,
    RETURN_LOCK_MIN_DISTANCE_KM,
    _can_drive,
    _is_car_passenger,
    _orphaned_vehicles,
    _owns_vehicle,
    _park_vehicles,
    _road_distance_km,
    _vehicle_mode,
    _vehicle_unavailable_reason,
    _vehicles_parked_at,
)

# ── Sources of a proposal (spec 04, G6) ──────────────────────────────────────
SOURCE_ENREGISTREE = "enregistree"  # served from the recorded set
SOURCE_EN_VOL = "en_vol"  # computed by the engines during the run (no set)
SOURCE_HORS_JEU = "hors_jeu"  # trip that the set does not cover (question 9)
SOURCE_LOCALE = "locale"  # produced without an engine (synthetic school bus)
PREFIXE_RECALCUL_OFFRE = "recalculee:offre"
PREFIXE_RECALCUL_HORAIRE = "recalculee:horaire"
PREFIXE_RECALCUL_OFFRE_JOUR = (
    "recalculee:offre_jour"  # simulated day other than the set's (decision 18)
)

# ── Decision methods (trace vocabulary) ──────────────────────────────────────
METHODE_SANS_SOLUTION = (
    "sans_solution"  # no practicable proposal, decision-maker not called
)
METHODE_CHOIX_UNIQUE = "choix_unique"  # a single proposal, decision-maker not called
METHODE_DECIDEUR = "decideur"  # usable response from the decision-maker
METHODE_REPLI_UNIFORME = (
    "repli_uniforme"  # unusable response → uniform draw (D10)
)
METHODE_ERREUR = "erreur"  # the decision-maker did not respond: NOT a decision
# R13: the model CANNOT decide (out of domain) — TERMINAL non-decision, counted,
# neither retried (unlike ERREUR) nor fabricated (unlike the uniform fallback).
METHODE_MODELE_NON_IMPUTABLE = "modele_non_imputable"

# Chain constraint as logged in moves.csv (vehicle-chain.md) —
# a single value per decision, priority passager > retour_force > sortie_bloquee.
CONTRAINTE_AUCUNE = ""
CONTRAINTE_RETOUR_FORCE = "retour_force"
CONTRAINTE_SORTIE_BLOQUEE = "sortie_bloquee"
CONTRAINTE_PASSAGER = "passager"


@dataclass
class Proposition:
    """One way of making the trip, and where it comes from."""

    plan: TravelPlan
    source: str = SOURCE_EN_VOL

    @property
    def code(self) -> str:
        try:
            return self.plan.get_code() or self.plan.id
        except TypeError:
            # Leg without `transit_route` (test or fallback plan): the identifier is authoritative.
            return self.plan.id

    @property
    def mode_vehicule(self) -> str:
        return _vehicle_mode(self.plan)

    @property
    def mode(self) -> str:
        return self.plan.mode_label() or "unknown"


@dataclass(frozen=True)
class Ecart:
    """An excluded proposal, and why (D2, D6)."""

    code: str
    mode: str
    motif: str


@dataclass
class ResultatFiltre:
    """Output of `eligibilite`: what remains, what left, and what the controller counts."""

    eligibles: list[Proposition]
    ecartees: list[Ecart]
    # Events of the `agent_vehicle_chain_total{mode,event}` metric — the controller
    # increments them, the runner counts them: the semantics remain those of vehicle-chain.md.
    evenements: list[tuple[str, str]]
    contrainte: str = CONTRAINTE_AUCUNE


def modes_vehicules_eligibles(
    person: Person, from_location: Location | None
) -> dict[str, bool]:
    """Exit lock per vehicle mode — what the controller passes to OTP (`include_*`)."""
    return {
        mode: _vehicle_unavailable_reason(person, mode, from_location) is None
        for mode in _VEHICLE_MODES
    }


def eligibilite(
    person: Person,
    from_location: Location | None,
    propositions: Sequence[Proposition],
    purpose: str | None,
    destination: Location | None,
) -> ResultatFiltre:
    """D2 — excludes what the person cannot use, in the documented order.

    1. exit lock, per vehicle mode: ownership, driver/passenger, position;
    2. return lock: return home of more than `RETURN_LOCK_MIN_DISTANCE_KM` with a
       vehicle parked at the origin ⇒ only the proposals of that mode remain.
    Walking and public transport are never excluded here (D2, last sentence).
    """
    traits = person.identity.traits_json
    evenements: list[tuple[str, str]] = []
    contrainte = CONTRAINTE_AUCUNE

    motifs = {
        mode: _vehicle_unavailable_reason(person, mode, from_location)
        for mode in _VEHICLE_MODES
    }
    for mode, motif in motifs.items():
        # Exit events: counted per mode OWNED but excluded, as the controller
        # did before routing (they do not depend on the proposals present).
        if motif is None or not _owns_vehicle(traits, mode):
            continue
        if motif == MOTIF_PAS_DE_CONDUCTEUR:
            evenements.append(("car", "no_driver"))
        else:
            evenements.append((mode, "unavailable"))
            contrainte = CONTRAINTE_SORTIE_BLOQUEE

    eligibles: list[Proposition] = []
    ecartees: list[Ecart] = []
    for prop in propositions:
        vm = prop.mode_vehicule
        motif = motifs.get(vm)
        if motif is not None:
            ecartees.append(Ecart(prop.code, prop.mode, motif))
        else:
            eligibles.append(prop)

    if (
        settings.agent.vehicle_chain_enabled
        and settings.agent.vehicle_return_home_lock
        and eligibles
        and (purpose or "").lower() == "home"
    ):
        a_ramener = _vehicles_parked_at(person, from_location)
        od_km = _road_distance_km(from_location, destination)
        if a_ramener and od_km is not None and od_km < RETURN_LOCK_MIN_DISTANCE_KM:
            evenements.extend((mode, "short_return") for mode in sorted(a_ramener))
            a_ramener = set()
        if a_ramener:
            gardees = [p for p in eligibles if p.mode_vehicule in a_ramener]
            evenements.extend(
                (mode, "forced_return" if gardees else "return_failed")
                for mode in sorted(a_ramener)
            )
            if gardees:
                ecartees.extend(
                    Ecart(p.code, p.mode, MOTIF_RETOUR_FORCE)
                    for p in eligibles
                    if p.mode_vehicule not in a_ramener
                )
                eligibles = gardees
                contrainte = CONTRAINTE_RETOUR_FORCE

    return ResultatFiltre(
        eligibles=eligibles,
        ecartees=ecartees,
        evenements=evenements,
        contrainte=contrainte,
    )


def plafonner(
    eligibles: Sequence[Proposition], max_n: int
) -> tuple[list[Proposition], list[Ecart]]:
    """Option cap (`_select_candidates`), with the excluded ones traced `plafond`.

    The input order is first made canonical (duration, code) so that a duration tie is
    broken the same way in both modes (RG-5) — `_select_candidates` is stable but otherwise
    depends on the order returned by the engines.
    """
    canon = sorted(
        eligibles,
        key=lambda p: (
            (p.plan.duration if p.plan.duration is not None else float("inf")),
            p.code,
        ),
    )
    plans = [p.plan for p in canon]
    gardes = _select_candidates(plans, max_n)
    garde_ids = {id(pl) for pl in gardes}
    retenues = [p for p in canon if id(p.plan) in garde_ids]
    ecartees = [
        Ecart(p.code, p.mode, MOTIF_PLAFOND)
        for p in canon
        if id(p.plan) not in garde_ids
    ]
    return retenues, ecartees


def graine_ordre(graine: int, person_id: str, activity_id: str | None) -> int:
    raw = f"{graine}|{person_id}|{activity_id or ''}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def ordre_presentation(
    propositions: Sequence, graine: int, person_id: str, activity_id: str | None
) -> list:
    """D7 — deterministic presentation order: same seed, same trip ⇒ same order.

    The input is first sorted by code so that the result does not depend on the received order;
    accepts `Proposition`s as well as bare `TravelPlan`s (the controller passes its plans).
    """

    def _code(x) -> str:
        plan = x.plan if isinstance(x, Proposition) else x
        return plan.get_code() or plan.id or ""

    items = sorted(propositions, key=_code)
    random.Random(graine_ordre(graine, person_id, activity_id)).shuffle(items)
    return items


@dataclass
class ContexteDecision:
    """What the decision-maker is allowed to know — and nothing else (D9, D11)."""

    timestamp: int  # SIMULATED clock of the decision
    activity_id: str
    purpose: str
    departure_time: int
    from_location: Location | None
    destination: Location | None
    anticipation: dict | None = None
    evenement: dict | None = None
    periode_evenement: str | None = None  # avant | pendant | apres (spec 04, G9)
    graine_ordre: int = 42
    graine_tirage: int = 42
    max_candidats: int = 6
    jour_offre: str | None = None


@dataclass
class ReponseDecideur:
    """What a decision-maker returns for a list of PRESENTED proposals (indexed in that order)."""

    index: int | None  # None ⇒ unusable response or error
    fournisseur: str = ""
    distribution: dict = field(default_factory=dict)  # per canonical mode
    poids: list[float] = field(default_factory=list)  # per presented proposal
    reponse_brute: str | None = None
    raison: str = ""
    souvenirs: list[str] = field(default_factory=list)
    presente: dict | None = None  # what was shown (payload / texts), for E11
    repli_uniforme: bool = False  # D10
    erreur: str | None = (
        None  # the decision-maker did NOT respond (network, quota, substitution)
    )
    identifiant_lot: str | None = (
        None  # S5: gateway batch identifier if known
    )
    non_imputable: bool = (
        False  # R13: TERMINAL non-decision (not a fallback, not a retry)
    )
    reprise_a: str | None = (
        None  # daily quota: ISO-UTC reopening instant, announced by the provider
    )
    modele_verifie: bool | None = (
        None  # antigravity P1: False if the model is declared without gateway attestation
    )
    parametres_appliques: dict | None = (
        # Antigravity: what the sub-agent declares it ACTUALLY applied of the
        # `parametres` sent (temperature, top_p, max_tokens). The IPC channel cannot
        # impose a sampling setting on an IDE agent: the only honest thing is
        # therefore to ask it what it applied and to archive that. `None` = the sub-agent
        # declared nothing, which IS NOT "temperature 0" but "we do not know".
        None
    )
    sortie_litterale: str | None = (
        # P8 (hygiene spec §8) — the model output AS EMITTED, never reformatted nor
        # re-serialised. `reponse_brute` showed its limit: on the antigravity channel, its
        # 2,287 responses were bare JSON in two forms depending on the segment, hence a
        # re-serialisation by the intermediate agent — no piece of the archive showed
        # what the model had actually written. None when the channel cannot provide it:
        # a declared gap is better than a copy one would believe literal.
        None
    )
    # Ticket 107 — weather traced decision by decision
    jour_offre: str | None = None
    jour_meteo_tire: str | None = None
    jour_meteo_lu: str | None = None
    source_date_meteo: str | None = None


class Decideur(Protocol):
    """Contract D1: the same interface for the gateway, a heuristic or a replay."""

    nom: str
    sans_quota: bool

    async def choisir(
        self, person: Person, ctx: ContexteDecision, presentees: list[Proposition]
    ) -> ReponseDecideur: ...


@dataclass
class Decision:
    retenue: Proposition | None
    methode: str
    trace: dict
    reponse: ReponseDecideur | None = None
    sollicite: bool = False

    @property
    def est_decision(self) -> bool:
        """An error is not a decision: nothing is archived, it will be requested again."""
        return self.methode != METHODE_ERREUR


def _prop_dict(p: Proposition) -> dict:
    return {
        "code": p.code,
        "mode": p.mode,
        "source": p.source,
        "duree_s": p.plan.duration,
    }


def construire_trace(
    person: Person,
    ctx: ContexteDecision,
    presentees: Sequence[Proposition],
    ecartees: Sequence[Ecart],
    retenue: Proposition | None,
    methode: str,
    reponse: ReponseDecideur | None,
    contrainte: str,
) -> dict:
    """D6 — the five elements (presented, excluded, chosen, distribution, raw response) + sources."""
    t = {
        "person_id": person.person_id,
        "activity_id": ctx.activity_id,
        "purpose": ctx.purpose,
        "timestamp": ctx.timestamp,
        "departure_time": ctx.departure_time,
        "methode": methode,
        "presentees": [_prop_dict(p) for p in presentees],
        "ecartees": [asdict(e) for e in ecartees],
        "retenue": (
            _prop_dict(retenue)
            | {
                "index_presente": next(
                    (i for i, p in enumerate(presentees) if p.code == retenue.code),
                    None,
                )
            }
        )
        if retenue
        else None,
        "distribution": dict(reponse.distribution) if reponse else {},
        "poids_presentes": list(reponse.poids) if reponse else [],
        "reponse_brute": reponse.reponse_brute if reponse else None,
        "sortie_litterale": reponse.sortie_litterale if reponse else None,
        "sources": {p.code: p.source for p in presentees},
        "fournisseur": reponse.fournisseur if reponse else "",
        "raison": reponse.raison if reponse else "",
        "souvenirs": list(reponse.souvenirs) if reponse else [],
        "presente": reponse.presente if reponse else None,
        "identifiant_lot": reponse.identifiant_lot if reponse else None,
        "graine_ordre": ctx.graine_ordre,
        "graine_tirage": ctx.graine_tirage,
        "contrainte_chaine": contrainte,
        "periode_evenement": ctx.periode_evenement,
        "anticipation": (ctx.anticipation or {}).get("trace", "")
        if ctx.anticipation
        else "",
    }
    if reponse and reponse.modele_verifie is not None:
        t["modele_verifie"] = reponse.modele_verifie
    if reponse and reponse.parametres_appliques is not None:
        t["parametres_appliques"] = reponse.parametres_appliques

    # Ticket 107 — three dates (offer, drawn weather, read weather) + source of the date
    if reponse is not None and getattr(reponse, "jour_offre", None) is not None:
        jour_offre = str(reponse.jour_offre)
    elif ctx.jour_offre is not None:
        jour_offre = str(ctx.jour_offre)
    else:
        from sim_clock import wall_clock

        jour_offre = wall_clock(ctx.timestamp).strftime("%Y-%m-%d")

    if reponse is not None and getattr(reponse, "source_date_meteo", None) is not None:
        jour_meteo_tire = reponse.jour_meteo_tire
        jour_meteo_lu = reponse.jour_meteo_lu
        source_date_meteo = reponse.source_date_meteo
    else:
        from urban_mobility_agents.utils.weather_draw import resoudre_meteo_decision

        info_m = resoudre_meteo_decision(
            person_id=str(person.person_id),
            timestamp=int(ctx.timestamp),
            graine=ctx.graine_tirage,
        )
        jour_meteo_tire = info_m["jour_meteo_tire"]
        jour_meteo_lu = info_m["jour_meteo_lu"]
        source_date_meteo = info_m["source_date_meteo"]

    t["jour_offre"] = jour_offre
    t["jour_meteo_tire"] = jour_meteo_tire
    t["jour_meteo_lu"] = jour_meteo_lu
    t["source_date_meteo"] = source_date_meteo

    return t


def resumer_ecartees(ecartees: Sequence) -> str:
    """`vehicule_ailleurs:car;plafond:2` — for the "Écartées (motifs)" column of moves.csv.

    Accepts `Ecart`s or trace dicts. Those excluded by the cap are counted, the
    others named by mode: this is what lets one read back why an option was missing.
    """
    par_motif: dict[str, list[str]] = {}
    for e in ecartees:
        motif = e.motif if isinstance(e, Ecart) else str(e.get("motif"))
        mode = e.mode if isinstance(e, Ecart) else str(e.get("mode"))
        par_motif.setdefault(motif, []).append(mode)
    parts = []
    for motif in sorted(par_motif):
        modes = par_motif[motif]
        parts.append(
            f"{motif}:{len(modes)}"
            if motif == MOTIF_PLAFOND
            else f"{motif}:{','.join(sorted(set(modes)))}"
        )
    return ";".join(parts)


CHAMPS_TRACE_OBLIGATOIRES = (
    "presentees",
    "ecartees",
    "retenue",
    "distribution",
    "reponse_brute",
    "sources",
    "methode",
)


def valider_trace(trace: dict) -> list[str]:
    """D6 fields missing from a reread trace — empty if the archive is valid."""
    return [c for c in CHAMPS_TRACE_OBLIGATOIRES if c not in trace]


async def decider(
    person: Person,
    ctx: ContexteDecision,
    propositions: Sequence[Proposition],
    decideur: Decideur,
) -> Decision:
    """D1/D5/D6/D10 — from the raw list to the traced decision. Does not modify the person (see `avancer_chaine`)."""
    filtre = eligibilite(
        person, ctx.from_location, propositions, ctx.purpose, ctx.destination
    )
    retenues, ecartees_plafond = plafonner(filtre.eligibles, ctx.max_candidats)
    ecartees = list(filtre.ecartees) + ecartees_plafond
    presentees = ordre_presentation(
        retenues, ctx.graine_ordre, person.person_id, ctx.activity_id
    )

    if not presentees:
        trace = construire_trace(
            person,
            ctx,
            [],
            ecartees,
            None,
            METHODE_SANS_SOLUTION,
            None,
            filtre.contrainte,
        )
        return Decision(retenue=None, methode=METHODE_SANS_SOLUTION, trace=trace)

    if len(presentees) == 1:
        seule = presentees[0]
        trace = construire_trace(
            person,
            ctx,
            presentees,
            ecartees,
            seule,
            METHODE_CHOIX_UNIQUE,
            None,
            filtre.contrainte,
        )
        return Decision(retenue=seule, methode=METHODE_CHOIX_UNIQUE, trace=trace)

    reponse = await decideur.choisir(person, ctx, presentees)
    if reponse.non_imputable:
        # R13 — TERMINAL non-decision: archived, counted, excluded from the shares (nothing
        # chosen), never retried (est_decision True) nor fabricated as a uniform fallback.
        trace = construire_trace(
            person,
            ctx,
            presentees,
            ecartees,
            None,
            METHODE_MODELE_NON_IMPUTABLE,
            reponse,
            filtre.contrainte,
        )
        return Decision(
            retenue=None,
            methode=METHODE_MODELE_NON_IMPUTABLE,
            trace=trace,
            reponse=reponse,
            sollicite=True,
        )
    if (
        reponse.erreur is not None
        or reponse.index is None
        or not (0 <= reponse.index < len(presentees))
    ):
        if reponse.erreur is None:
            # Response returned but unusable (out-of-bounds index not realigned): D10 fallback.
            reponse.repli_uniforme = True
            reponse.index = random.Random(
                graine_ordre(ctx.graine_tirage, person.person_id, ctx.activity_id)
            ).randrange(len(presentees))
        else:
            trace = construire_trace(
                person,
                ctx,
                presentees,
                ecartees,
                None,
                METHODE_ERREUR,
                reponse,
                filtre.contrainte,
            )
            trace["erreur"] = reponse.erreur
            return Decision(
                retenue=None,
                methode=METHODE_ERREUR,
                trace=trace,
                reponse=reponse,
                sollicite=True,
            )

    retenue = presentees[reponse.index]
    methode = METHODE_REPLI_UNIFORME if reponse.repli_uniforme else METHODE_DECIDEUR
    contrainte = filtre.contrainte
    if retenue.mode_vehicule == "car" and _is_car_passenger(person):
        contrainte = CONTRAINTE_PASSAGER
    trace = construire_trace(
        person, ctx, presentees, ecartees, retenue, methode, reponse, contrainte
    )
    return Decision(
        retenue=retenue, methode=methode, trace=trace, reponse=reponse, sollicite=True
    )


def avancer_chaine(
    person: Person,
    plan: TravelPlan | None,
    from_location: Location | None,
    destination: Location | None,
    purpose: str | None,
) -> set[str]:
    """D4 — the vehicle used follows the person, the others stay; orphans recovered at home.

    Returns the set of orphan modes observed (empty most of the time). No effect if the chain
    is disabled. The GAMA controller keeps its own metrics around this same call.
    """
    if not settings.agent.vehicle_chain_enabled or plan is None or destination is None:
        return set()
    _park_vehicles(person, plan, from_location, destination)
    if (purpose or "").lower() != "home":
        return set()
    orphelins = _orphaned_vehicles(person)
    if orphelins and settings.agent.vehicle_orphan_reset_at_home:
        for mode in orphelins:
            person.state.planning_vehicle_at.pop(mode, None)
    return orphelins


__all__ = [
    "CONTRAINTE_AUCUNE",
    "CONTRAINTE_PASSAGER",
    "CONTRAINTE_RETOUR_FORCE",
    "CONTRAINTE_SORTIE_BLOQUEE",
    "METHODE_CHOIX_UNIQUE",
    "METHODE_DECIDEUR",
    "METHODE_ERREUR",
    "METHODE_MODELE_NON_IMPUTABLE",
    "METHODE_REPLI_UNIFORME",
    "METHODE_SANS_SOLUTION",
    "PREFIXE_RECALCUL_HORAIRE",
    "PREFIXE_RECALCUL_OFFRE",
    "PREFIXE_RECALCUL_OFFRE_JOUR",
    "SOURCE_ENREGISTREE",
    "SOURCE_EN_VOL",
    "SOURCE_HORS_JEU",
    "SOURCE_LOCALE",
    "ContexteDecision",
    "Decideur",
    "Decision",
    "Ecart",
    "Proposition",
    "ReponseDecideur",
    "ResultatFiltre",
    "_can_drive",
    "_selection_group",
    "avancer_chaine",
    "construire_trace",
    "decider",
    "eligibilite",
    "graine_ordre",
    "modes_vehicules_eligibles",
    "ordre_presentation",
    "plafonner",
    "resumer_ecartees",
    "valider_trace",
]
