import asyncio
import csv
import itertools
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from mobility_core.housing_type import TRAIT_KEY as HOUSING_TRAIT_KEY, key_for
from mobility_core.mode_hierarchy import hierarchy as _mode_hierarchy
from mobility_core.population_reference import COURONNES, OUT_OF_PERIMETER
from mobility_core.residence_zone import TRAIT_KEY as RESIDENCE_TRAIT_KEY
from models import Person, TravelPlan
from settings import settings

# Values accepted in the "Lieu de résidence" column. `hors périmètre` is one of
# them: a known home located outside the survey's 453 municipalities has no target
# per zone, and its mass must be COUNTED rather than diluted into the 3rd ring
# (axis A4 of ticket 020). A value outside this list is reset to empty rather than
# logged: the summary page joins this column on the EMC² labels, an exotic value
# would vanish there without being counted.
RESIDENCE_VALUES = frozenset((*COURONNES, OUT_OF_PERIMETER))

logger = logging.getLogger(__name__)

_MODE_HIERARCHY = _mode_hierarchy()

# Unknown mode sets already reported — the alarm fires on a rising edge.
_UNKNOWN_MODES_SEEN: set[str] = set()

# Mode families, DERIVED from the survey's hierarchy and no longer written here
# (ticket 022). Five literal lists coexisted in the repository and any single one of
# them was enough to make a modal share drift: an incomplete list yields a plausible
# and wrong number (the Téléo counted as walking, the TER counted as walking). The order
# of the tests now follows the "Hiérarchie des modes" appendix of the AUAT/CEREMA report
# (p. 53), checked against the microdata: metro > tram > cable car > bus > train > car >
# powered two-wheeler > bike > walk.
#
# The five names survive because they are cited elsewhere (`llm_agent`, the parity
# test); they are no longer a source, only a view.
_BUS_MODES = frozenset().union(*(_MODE_HIERARCHY.legs_by_family[f]
                                 for f in ("metro", "tram", "cableway", "bus")))
_RAIL_MODES = _MODE_HIERARCHY.legs_by_family["rail"]
_CAR_MODES = _MODE_HIERARCHY.legs_by_family["car"]
_BIKE_MODES = _MODE_HIERARCHY.legs_by_family["bicycle"]
_WALK_MODES = _MODE_HIERARCHY.legs_by_family["foot"]

_PURPOSE_FR = {
    "work": "Travail",
    "education": "Etude",
    "shop": "Achats",
    "shopping": "Achats",
    "escort": "Accompagnement",
    "accompany": "Accompagnement",
}

# Canonical modes (mobility_llm.mode_choice) → column labels, aligned on the
# vocabulary of "Mode de transport Choisi". The labels come from the hierarchy
# (ticket 022): a single table decides them. The ORDER, however, stays that of the
# display and not that of the hierarchy — it fixes the CSV columns, and changing them
# would make archived `moves.csv` files incomparable with new ones. `other` is not a
# survey family: it is the repository's catch-all, so it has no rank.
_COLUMN_ORDER = ("walking", "cycling", "car", "public_transport", "train", "motorbike")
_LABEL_BY_CANONICAL = {_MODE_HIERARCHY.canonical_mode[family]:
                       _MODE_HIERARCHY.journal_label[family]
                       for family in _MODE_HIERARCHY.families}
# A canonical mode missing from the hierarchy raises here, at import: a silent column
# would be better than a wrong label, but a missing column is better than both.
_CANONICAL_FR = {canonical: _LABEL_BY_CANONICAL[canonical] for canonical in _COLUMN_ORDER}
_CANONICAL_FR["other"] = "Autres modes"

# One column per mode: the distribution estimated by the LLM before the draw (sum = 100).
# A mode not offered is 0 (not empty) — this is what distinguishes "was possible but
# judged null" from "decision without distribution" (single choice, LLM error, inherited
# cache), where all these columns are empty.
MODE_PROBABILITY_HEADERS = [f"P({label}) %" for label in _CANONICAL_FR.values()]

# Allowed values of "Contrainte de chaîne" (ticket 008, A4) — one per row:
#   ""              no constraint, the choice set is OTP's;
#   retour_force    return lock applied, options restricted to the parked vehicle's mode;
#   passager        car trip driven by another member of the household;
#   sortie_bloquee  an owned vehicle mode was dropped for lack of a vehicle on site.
# The column **explains**, it does not filter: these rows stay in the scoring, and the
# summary page shows their breakdown next to the selection methods.
CHAIN_CONSTRAINTS = ("", "retour_force", "passager", "sortie_bloquee")

# Allowed values of "Anticipation" (ticket 014) — what the prompt of THIS trip
# contained as anticipation context:
#   ""        decision without a prompt (cache, single option) or anticipation disabled;
#   agenda    full block — rolling agenda + vehicle positions (+ the day's weather);
#   meteo     the day's weather only (agent with no vehicle to chain).
# Essential to segment the before/after A/B: non-motorised agents do not have the block.
ANTICIPATION_VALUES = ("", "agenda", "meteo")

CSV_HEADERS = [
    "Référence",
    "Trajet",
    "ID Trajet",
    "Mode de transport Choisi",
    "Plus rapide",
    "Modes proposés au LLM",
    *MODE_PROBABILITY_HEADERS,
    "Lieu de résidence",
    "Genre",
    "Âge",
    "Occupation principale",
    "Type de logement",
    "Motifs de déplacement",
    "Distance parcourue",
    "Méthode de sélection",
    "Contrainte de chaîne",
    "Anticipation",
    "Fournisseur & Modèle",
    "Température",
    "Mémoire à court terme",
    "Mémoire à long terme",
    "Filtre de perception",
    "Traits de personnalité",
    "Météo Température (°C)",
    "Météo Condition",
    "Météo Précipitations (mm)",
    "Raisonnement",
    "Retard planification (s)",
    "Heure de calcul",
    "Temps simulé",
    "Heure de départ",
    "ID Personne",
    "ID Activité",
    # Ticket 035 (spec 04, G6): where the presented proposals come from — "enregistree:5",
    # "enregistree:3,recalculee:horaire:2", "en_vol:6" (in-flight computation, no set), "hors_jeu:…".
    # Added in LAST position: no consumer reads moves.csv by index.
    "Source des propositions",
    # Proposals dropped before presentation, by reason (spec 02, D2/D6):
    # "vehicule_ailleurs:car;retour_force:foot,bus;plafond:2" — empty if nothing was dropped.
    "Écartées (motifs)",
    # Identifier of the gateway batch that carried the decision (spec 03, S5) — empty otherwise.
    "Identifiant lot",
    # Ticket 077, lot E2 — the CHOSEN index among the presented options, for ANY decision.
    # It only existed in `pipeline_timing.csv`, and only for 66 of the 514 trips of the
    # ticket 075 run: impossible to read there whether the agent changed itinerary at constant purpose.
    "Index retenu",
    # Number of options actually presented. Without it, an index of 0 cannot be told from a
    # constrained choice: the absence of alternatives would produce a perfect "loyalty".
    "Options présentées",
    # Ticket 077, lot E5 — does the decision come from the semantic cache or a direct call.
    # "cache", "direct", or empty when no model was asked.
    "Origine de la décision",
    # Ticket 077, lot E3 — description of the presented options, index by index.
    "Options (descriptif)",
    # Ticket 079, extended in ticket 100 — the event in force, and the day RELATIVE to its
    # first day (−2, −1, 0, +1…). Filled in for ANY decision, including nominal days: the
    # relative day is the x-axis of the drop-off and return curves, and an x-axis that only
    # existed on event days would plot nothing. Empty when no event is declared.
    #
    # ⚠ The first two keep their NAME from ticket 079. Renaming them would break the
    # re-reading of already archived runs — and they carry the published figures of § 7.2.
    # The new vocabulary lives in `evenements.jsonl`; here, we add without substituting.
    "Choc",
    "Jour relatif au choc",
    # Ticket 100, lot 5 — the agent's role in the setup: `expose`, `co_resident`
    # (same roof as an exposed agent, received nothing — it is there that diffusion is read)
    # or `temoin`. Empty when the setup does not yet know who reads: an empty column is not a
    # role, and writing `temoin` would pass ignorance off as a measurement.
    "Rôle",
    "Raison d'exposition",
]


def _residence_zone(traits: dict) -> str:
    """Residence ring — READ from the persona, never recomputed (ticket 021).

    The trait is set at population generation (`scripts/data/population/
    enrich_residence_zone.py`) from the survey's **list-of-municipalities**
    partition, the one against which the modal shares per zone are published.

    ⚠ **This module no longer imports `geo_reference.residence_zone`, and that is
    deliberate.** That function classifies by DISTANCE to the city centre (8 / 20 / 40 km),
    which is not the survey's definition: ticket 020 measured 24.4% of misclassified
    personas and 66 "false Toulouse residents" living in Blagnac or Balma. As long as the
    import existed, a "reasonable" fallback could be restored in one line by
    inadvertence; by removing it, the fallback becomes impossible rather than discouraged.

    Empty when the persona does not carry the trait — population generated before ticket
    021, or a home without coordinates. Empty is therefore not a category, exactly as an
    empty probability cell is not a 0. `hors périmètre` is one, however: the home is known
    and it is outside, it has no EMC² target, and its mass is counted instead of being
    diluted into the 3rd ring.
    """
    value = str(traits.get(RESIDENCE_TRAIT_KEY) or "").strip()
    return value if value in RESIDENCE_VALUES else ""


def _housing_type(traits: dict) -> str:
    """Housing type of the persona, in the categories of the EMC² reference.

    The trait is set at population generation (`scripts/data/population/
    enrich_housing_type.py`): it is **imputed** from the distribution the survey observes
    in the home's fine zone, never drawn here. This module only copies it.

    Empty when the persona carries none — population generated before action A2, or a
    home outside the fine-zone layer, where nothing is guessed. Empty is therefore not a
    category, exactly as an empty probability cell is not a 0. A value outside the
    reference data is reset to empty rather than logged: the summary page joins this
    column on the EMC² labels, an exotic value would vanish there without being
    counted.
    """
    label = str(traits.get(HOUSING_TRAIT_KEY) or "").strip()
    return label if key_for(label) else ""


def _log_unknown_modes(modes: set) -> None:
    """Rising-edge alarm: a mode the hierarchy ignores must not pass silently.

    A mode missing from the hierarchy lands in "Autres modes", which is **excluded** from
    the EMC² scoring (`frames.CHOSEN_MODE_MAP` files it under `autres`). Its mass thus
    vanishes from a modal share without breaking anything: this is the Téléo defect
    (2026-08-26) and the TER one (2026-09-04), twice the same mechanism. A single line per
    unknown mode set, so as not to flood the log of a run of several thousand trips.
    """
    cle = ",".join(sorted(str(m) for m in modes))
    if cle in _UNKNOWN_MODES_SEEN:
        return
    _UNKNOWN_MODES_SEEN.add(cle)
    logger.error(
        "[ALARME] Modes outside the hierarchy in a plan: {%s} → column \"Autres modes\", "
        "hence outside EMC² scoring. Add them to mobility_core/data/mode_hierarchy_emc2.json "
        "(scripts/progedo_logit/export_mode_hierarchy.py, table FAMILLES).", cle)


def _plan_transport_mode(plan: Optional[TravelPlan]) -> str:
    if plan is None:
        return ""
    non_transfer = [leg for leg in (plan.legs or []) if not leg.is_transfer]
    if not non_transfer:
        return "Marche"
    modes = {(leg.mode or "").lower() for leg in non_transfer}
    # A single call, a single order: the survey's (cf. `mobility_core.
    # mode_hierarchy`). "Autres modes" is no longer the output of an exhausted cascade but
    # that of a mode the hierarchy does not know — a case to look at, not to absorb.
    label = _MODE_HIERARCHY.primary_label(modes)
    if label is None:
        _log_unknown_modes(modes)
        return "Autres modes"
    return label


def _available_modes_summary(options: Optional[list]) -> str:
    if not options:
        return ""
    return " | ".join(_plan_transport_mode(opt) for opt in options)


def _contexte_choc(start_time_ms: Optional[int], person_id: str = "") -> tuple:
    """Tickets 079 and 100 — `(event, relative day, role, reason)` for this decision.

    Filled in for ANY decision of a run with an event, including nominal days: the relative
    day is the x-axis of the drop-off and return curves, and an x-axis that only existed on
    event days would plot nothing. Four empty values when no event is declared, which leaves
    nominal runs strictly identical to what they were.

    Never raises: a missing context column is better than a lost decision.
    """
    try:
        from llm import evenements as evenements_module

        registre = evenements_module.registre()
        if registre is None or start_time_ms is None:
            # ⚠ Four values, like every output of this function: a `moves.csv` row that does
            # not have the number of columns of its header shifts everything that follows,
            # and the shift is only seen on re-reading, when it is too late.
            return ("", "", "", "")
        role, raison = registre.role_de(str(person_id))
        return (
            registre.evenement.evenement_id,
            registre.jour_relatif(int(start_time_ms) // 1000, str(person_id)),
            role,
            raison,
        )
    except Exception:  # noqa: BLE001
        return ("", "", "", "")


def _options_descriptif(options: Optional[list]) -> str:
    """Description of the PRESENTED options — ticket 077, lot E3.

    "0:car:487s:6.5km | 1:foot,bus,foot:3480s:7.5km". One field per option, in list order,
    so the index there is the one carried by the "Index retenu" column.

    ⚠ Why this column exists. Until now, the description of the options could only be
    rebuilt by re-reading the prompt TEXT of `llm_exchanges.jsonl`, and matching a trip to
    its prompt block only found 247 of the 430 trips of the ticket 075 run. Without a
    description, one cannot tell whether an agent changed ITINERARY at constant mode — which
    is precisely the question memory raises.

    Duration is in seconds and distance in kilometres: two units already used by the
    neighbouring columns, and no rounding that would hide two close options.
    """
    if not options:
        return ""
    parts = []
    for index, option in enumerate(options):
        try:
            duree = int(max(0, (option.end_time - option.start_time) // 1000))
        except (AttributeError, TypeError):
            duree = ""
        parts.append(
            f"{index}:{option.mode_label() or '?'}:{duree}s:{_plan_distance_km(option)}km"
        )
    return " | ".join(parts)


def _mode_probability_cells(distribution: Optional[dict]) -> list:
    """Spreads the per-mode distribution over one column per mode (`_CANONICAL_FR` order).

    Without a distribution (single choice, LLM error, inherited cache point), all cells
    are empty — to be told apart from a 0, which means "the LLM explicitly ruled out this mode".
    """
    if not distribution:
        return [""] * len(_CANONICAL_FR)
    return [round(distribution.get(mode, 0.0) * 100, 1) for mode in _CANONICAL_FR]


def _plan_distance_km(plan: Optional[TravelPlan]) -> str:
    if plan is None:
        return ""
    if plan.distance is not None:
        return str(round(plan.distance / 1000, 2))
    total = sum(leg.get_distance() for leg in (plan.legs or []))
    return str(round(total / 1000, 2))


class GamaArrivalsLogger:
    _instance: Optional["GamaArrivalsLogger"] = None

    # Ticket 079 — `retard_injecte_s` is in LAST position and distinct from `delay_s`:
    # one is what the simulation measured, the other what a declared shock inflicted.
    # Confusing them would make any re-reading of a shock campaign impossible.
    _HEADERS = ["move_id", "person_id", "arrive_at", "expected_arrive_at", "delay_s", "started_at", "schedule_at", "departure_delay_s", "timed_out", "retard_injecte_s"]

    def __init__(self):
        self._path: Optional[Path] = None
        self._lock = asyncio.Lock()

    @classmethod
    def get_instance(cls) -> "GamaArrivalsLogger":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _ensure_header(self):
        path = Path(settings.app.log_file).parent / "gama_results" / "gama_arrivals.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        needs_header = not path.exists()
        self._path = path
        if needs_header:
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self._HEADERS)

    def _write_arrival(self, move_id: str, person_id: str, arrive_at: int, expected_arrive_at: int,
                       started_at: Optional[int], schedule_at: Optional[int], timed_out: bool,
                       retard_injecte_s: int = 0):
        self._ensure_header()
        delay_s = arrive_at - expected_arrive_at
        departure_delay_s = (started_at - schedule_at) if started_at is not None and schedule_at is not None else None
        with open(self._path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([move_id, person_id, arrive_at, expected_arrive_at, delay_s,
                                    started_at, schedule_at, departure_delay_s, timed_out,
                                    int(retard_injecte_s)])

    async def log_arrival(self, move_id: str, person_id: str, arrive_at: int, expected_arrive_at: int,
                          started_at: Optional[int] = None, schedule_at: Optional[int] = None,
                          timed_out: bool = False, retard_injecte_s: int = 0):
        # Writing offloaded out of the event loop (blocking open/write); the asyncio lock
        # guarantees the row order and that the header is written only once.
        async with self._lock:
            await asyncio.to_thread(self._write_arrival, move_id, person_id, arrive_at,
                                    expected_arrive_at, started_at, schedule_at, timed_out,
                                    retard_injecte_s)


class MoveLogger:
    _instance: Optional["MoveLogger"] = None

    def __init__(self):
        self._path: Optional[Path] = None
        self._lock = asyncio.Lock()
        self._counter = itertools.count(1)

    @classmethod
    def get_instance(cls) -> "MoveLogger":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _resolve_path(self) -> Path:
        return Path(settings.app.log_file).parent / "moves.csv"

    def _ensure_header(self):
        path = self._resolve_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        needs_header = not path.exists()
        self._path = path
        if needs_header:
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(CSV_HEADERS)

    async def log_move(
        self,
        person: Person,
        plan: Optional[TravelPlan],
        purpose: Optional[str],
        selection_method: str,
        provider_model: str,
        faster_itinerary: Optional[TravelPlan],
        reasoning: str,
        chain_constraint: str = "",
        anticipation: str = "",
        weather_temp: Optional[float] = None,
        weather_condition: Optional[str] = None,
        weather_precip_mm: Optional[float] = None,
        late_s: int = 0,
        move_id: str = "",
        simulated_time: Optional[int] = None,
        start_time: Optional[int] = None,
        available_options: Optional[list] = None,
        activity_id: Optional[str] = None,
        mode_probabilities: Optional[dict] = None,
        sources: str = "",
        ecartees: str = "",
        lot: str = "",
        selected_index: Optional[int] = None,
        origine_decision: str = "",
    ):
        async with self._lock:
            computed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            trip_id = next(self._counter)
            traits = person.identity.traits_json
            home = person.identity.home

            gender_raw = traits.get("gender", "")
            gender = "Homme" if gender_raw == "Male" else "Femme" if gender_raw == "Female" else gender_raw

            purpose_fr = _PURPOSE_FR.get((purpose or "").lower(), purpose or "")

            no_move = selection_method == "Pas de déplacement (même localisation)"
            row = [
                settings.workdir.name,
                trip_id,
                move_id,
                "Aucun" if no_move else _plan_transport_mode(plan),
                _plan_transport_mode(faster_itinerary),
                _available_modes_summary(available_options),
                *_mode_probability_cells(mode_probabilities),
                _residence_zone(traits),
                gender,
                traits.get("age", ""),
                traits.get("main_occupation", ""),
                _housing_type(traits),
                purpose_fr,
                _plan_distance_km(plan),
                selection_method,
                chain_constraint if chain_constraint in CHAIN_CONSTRAINTS else "",
                anticipation if anticipation in ANTICIPATION_VALUES else "",
                provider_model,
                settings.agent.llm_params.get("temperature", ""),
                True,
                settings.agent.long_term_memory_enabled,
                settings.agent.long_term_memory_filter_by_datetime,
                "personality" in traits,
                weather_temp if weather_temp is not None else "",
                weather_condition if weather_condition is not None else "",
                weather_precip_mm if weather_precip_mm is not None else "",
                reasoning,
                late_s,
                computed_at,
                simulated_time if simulated_time is not None else "",
                # `tz=timezone.utc` on a GAMA clock timestamp returns the WALL-CLOCK
                # time — that is the definition of `sim_clock.wall_clock` — and does not
                # depend on the process `TZ`. Left as is: the "Heure de départ" column
                # of moves.csv is already the agents' time, and rewriting it to an
                # identical value would break the comparison with archived runs.
                datetime.fromtimestamp(start_time / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S") if start_time is not None else "",
                person.person_id,
                activity_id if activity_id is not None else "",
                sources or "",
                ecartees or "",
                lot or "",
                "" if selected_index is None else int(selected_index),
                len(available_options) if available_options else 0,
                origine_decision or "",
                _options_descriptif(available_options),
                *_contexte_choc(start_time, person.person_id),
            ]

            # Writing offloaded out of the event loop (blocking open/write)
            await asyncio.to_thread(self._write_row, row)

    def _write_row(self, row: list):
        self._ensure_header()
        with open(self._path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
