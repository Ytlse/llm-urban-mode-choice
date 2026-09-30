import asyncio
import hashlib
import json
import os
import re
import time
import traceback
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import demjson3
import numpy as np
from experiences.decision import ordre_presentation
from helper import (
    categorize_date_time_short,
    get_weekday_category,
    humanize_date,
    humanize_date_short,
    humanize_time,
)
from llm.axes import (
    creneau_de,
    meteo_de,
    mode_canonique,
    normaliser_lieu,
    normaliser_motif,
)
from llm.cache import LlmSemanticCache
from llm import evenements as evenements_module
from llm import foyer
from llm.evenements import temoin
from llm.concepts import (
    CONCEPTS_MONTRES_PAR_PANIER,
    CONFIRMER,
    CONTREDIRE,
    CREER,
    PRECISER,
    normaliser_operation,
    panier_de,
)
from llm.gravite import (
    CONTRAINTES_MODE_FORCE,
    force_apres_rappel,
    gravite_concept,
    gravite_deterministe,
    gravite_jugee,
)
from llm.journal_memoire import journal
from llm.longterm import MultiUserLongTermMemory
from llm.memory import MemoryEntry, MemoryType
from llm.noyau import TITRE_CHANGEMENTS, memoire_noyau
from urban_mobility_agents.utils.modeles import origine as origine_modele
from urban_mobility_agents.utils.routage import instances_pour, toutes_les_instances
from llm.reflection_store import ReflectionMemoStore
from llm.shortterm import UserShortTermMemory
from llm.trace_concepts import tracer_operation
from llm_gateway.sdk import LLMGatewayClient
from loguru import logger
from mobility_llm import prompt_manager as _mobility_prompt_manager
from mobility_llm.mode_choice import (
    UniformFallback,
    draw_index,
    mode_distribution,
    normalize_option_probabilities,
)
from models import Person, TravelPlan
from pydantic import BaseModel
from settings import settings
from sim_clock import gama_timestamp, wall_clock
from text_helper import env_ob_to_text
from urban_mobility_agents.agents.prompt_manager import PromptManager
from urban_mobility_agents.agents.prompt_types import PromptName
from urban_mobility_agents.utils import rejeu_decisions
from urban_mobility_agents.utils.ancre_run import jours_ecoules
from urban_mobility_agents.utils.history_log import HistoryStreamLog
from urban_mobility_agents.utils.pipeline_logger import PipelineLogger
from urban_mobility_agents.utils.reprise import gel_actif
from urban_mobility_agents.utils.weather_draw import (
    date_declaree,
    jours_eligibles,
    timestamp_meteo,
)
from urban_mobility_agents.utils.weather_loader import (
    get_weather,
    weather_to_natural_language,
)
from utils import create_background_task
from world.population import PersonScheduler

history_log = HistoryStreamLog.get_instance()


class ConsolidationMemoryUnavailable(RuntimeError):
    """The gateway returned no consolidation: the buffer must stay intact.

    `genre` is the gateway's `error_kind` of the last failed call (`surcharge_fournisseur`,
    `quota_journalier`, …), None when the failure is not the provider's. It decides whether the
    experiment stop is retried (`utils/nature_arret.py`, 2026-09-29).
    """

    def __init__(self, message: str, genre: str | None = None) -> None:
        super().__init__(message)
        self.genre = genre


def log_llm_cache_hit(
    agent_id: str,
    activity_id: str | None,
    sim_ts: float,
    mode: str,
    category: str = "itinary_multi_agent",
) -> None:
    """Traces a decision served by the semantic cache in workdir/llm_cache_hits.jsonl.

    A hit triggers no LLM call (hence no line in llm_exchanges.jsonl): this log makes it
    possible to count the calls saved and to break the saving down by simulation day
    (sim_day). The value in tokens saved is estimated on the analysis side from the average
    per-agent cost of the calls actually made for the same category."""
    try:
        entry = {
            "time": datetime.now(timezone.utc).isoformat(),
            "sim_ts": sim_ts,
            # sim_day in UTC to line up with llm_exchanges.jsonl (cf. logger.log_llm_exchange).
            # `tz=timezone.utc` on a GAMA timestamp ALREADY gives the WALL-CLOCK day — that is
            # the very definition of `sim_clock.wall_clock` — and does not depend on the
            # process `TZ`: left as is on purpose, to stay byte for byte the field that
            # `llm_gateway.telemetry.logger` writes on its side (a separate package, which
            # cannot import `sim_clock`).
            "sim_day": datetime.fromtimestamp(sim_ts, tz=timezone.utc).strftime(
                "%Y-%m-%d"
            )
            if sim_ts
            else None,
            "agent_id": str(agent_id),
            "activity_id": str(activity_id or ""),
            "category": category,
            "mode": mode,
        }
        with open(settings.app.llm_cache_hits_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except OSError as e:
        logger.warning(f"Cannot write the LLM cache hit: {e}")


def _format_distribution(distribution: dict) -> str:
    """"car 60% · public_transport 30% · walking 10%" — modes at 0% left out of the text.

    The complete dictionary (modes at 0% included) remains the source for the metrics:
    this format is only meant for readable traces (short-term memory, logs).
    """
    parts = [
        f"{mode} {pct * 100:.0f}%"
        for mode, pct in sorted(distribution.items(), key=lambda kv: -kv[1])
        if pct > 0
    ]
    return " · ".join(parts) or "aucune"


@lru_cache(maxsize=1)
def _weather_eligible_days() -> tuple[tuple[int, int], ...]:
    """Days of the year from which the per-agent weather draw picks.

    Resolved once: the window does not change during a run, and re-reading it at
    each decision would cost one YAML read per agent and per activity.
    `"enquete"` delegates the bounds to `mobility_core.population_reference`
    rather than copying them — renaming a key of the framing must not break
    this device silently.
    """
    fenetre = settings.agent.weather_window
    jours_semaine = None
    if fenetre == "enquete":
        from mobility_core.population_reference import survey_window, surveyed_weekdays

        debut, fin = survey_window()
        if settings.agent.weather_weekdays_only:
            jours_semaine = tuple(surveyed_weekdays())
    elif fenetre == "annee":
        debut, fin = "2024-01-01", "2024-12-31"
        if settings.agent.weather_weekdays_only:
            jours_semaine = (1, 2, 3, 4, 5)
    else:
        debut, fin = fenetre
        if settings.agent.weather_weekdays_only:
            jours_semaine = (1, 2, 3, 4, 5)

    jours = jours_eligibles(debut, fin, jours_semaine)
    logger.info(
        f"[météo] one date per agent: {len(jours)} eligible day(s) in "
        f"{debut} → {fin}"
        + (f", weekdays {jours_semaine}" if jours_semaine else "")
    )
    return jours


def _traits_de(person) -> dict:
    """The persona's traits for the log header, or empty.

    Defensive ON PURPOSE: the log knows nothing of a person's structure and must NEVER
    bring down a consolidation for a header. A test double without `identity` was enough
    to break ten tests on 2026-09-14 — exactly the kind of accident an observer has no
    right to cause.
    """
    identite = getattr(person, "identity", None)
    traits = getattr(identite, "traits_json", None) if identite is not None else None
    return traits if isinstance(traits, dict) else {}


class Context(BaseModel):
    person: Person
    timestamp: int
    activity_id: str | None = None
    data: dict | None = None


def log_chat(prompt: str, response: str, context: Context) -> str:
    log_dir = settings.agent.chat_log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    type_suffix = (
        f"-{context.data['type']}" if context.data and context.data.get("type") else ""
    )
    # GAMA WALL-CLOCK time (`sim_clock`): the file name must read alongside the
    # prompt it contains, and that prompt carries the same time.
    sim_time = datetime.strftime(wall_clock(context.timestamp), "%d_%H%M")
    file_name = f"{datetime.now().strftime('%Y%m%d%H%M%S')}-{sim_time}-{context.person.person_id}-{context.activity_id}{type_suffix}.txt"

    with open(os.path.join(log_dir, file_name), "a") as f:
        f.write("--------------------\n")
        f.write(f"Prompt: \n{prompt}\n")
        f.write("--------------------\n")
        f.write(f"Response: \n{response}\n\n")
        f.write("--------------------\n")
        f.write(f"Data: \n{context.model_dump_json()}\n")

    return file_name


# Public transport legs, to decide whether the PT pass is relevant to announce on an
# option. `cableway` = Téléo, which is part of the Tisséo network.
#
# ⚠ This list serves the PROMPT, not the score. Three lists of PT modes coexist in the
# repository and must stay consistent: this one, `move_logger._BUS_MODES` (production
# log) and `categorize_mode` (calibration loss). The latter ignored `cableway` until
# 2026-08-26 — the Téléo was counted there as walking; a parity test now locks the last
# two. The loss is the measuring instrument: any change is quantified there before being
# applied (amendment A13 of the protocol).
_PT_LEG_MODES = (
    "bus",
    "metro",
    "métro",
    "tram",
    "cableway",
    "transit",
    "public_transport",
    "rail",
    "train",
)


# Persona traits EXCLUDED from the cache signature. `name` only, and for a verified
# reason: it comes from unseeded Faker at generation, so it differs from one population to
# another without any decision depending on it — including it would invalidate the whole
# cache at each population regeneration. Check made on 2026-08-27: the `name` is identical
# between the source population and the run's (930/930), so it is NOT re-drawn at loading,
# contrary to what the docs claimed.
#
# Everything else goes in, including traits that only serve the narrative: sorting by
# "what reaches the prompt" is exactly the trade-off that produced the defect fixed
# here. Over-invalidating is the safe direction.
_TRAITS_EXCLUDED_FROM_CACHE_KEY = ("name",)


def _traits_signature(traits: dict | None) -> str:
    """Stable signature of the persona's traits, for the LLM cache's `state_hash`.

    Without it, a trait that does not condition the offer — the PT pass — changes the
    prompt without changing the cache key, and decisions already stored are served again
    under the old prompt silently. Measured on 2026-08-27: 352 passes corrected in the
    population of 1,000, none of which would have reached the cached decisions.

    Sorted keys: a Python dict keeps insertion order, and two populations serialised
    differently would give two signatures for the same traits.
    """
    if not traits:
        return ""
    kept = {
        k: v
        for k, v in sorted(traits.items())
        if k not in _TRAITS_EXCLUDED_FROM_CACHE_KEY
    }
    raw = json.dumps(kept, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _pt_subscription_note(mode_label: str, has_pt: bool) -> str:
    """PT pass mention to attach to an option, or an empty string.

    The information lives on the OPTION and no longer on the persona (2026-08-26): it
    cannot be deduced from the option set — a bus option is offered whether or not one
    has a pass — but it only weighs on the decision where public transport is actually
    offered. A persona line made it be read even without any PT option.
    """
    ml = (mode_label or "").lower()
    # School bus (ticket 030): counted as PT for the metrics, but FREE — no pass. The
    # guard comes before the _PT_LEG_MODES test, whose "bus" substring would otherwise
    # catch "school_bus" and attach a wrong mention.
    if "school_bus" in ml:
        return ""
    if not any(k in ml for k in _PT_LEG_MODES):
        return ""
    return (
        " Has a public transport pass." if has_pt else " Has no public transport pass."
    )


def _build_profile_narrative(traits: dict) -> str:
    """Social identity of the persona, in English (ticket 074, B-5).

    Two points of the switch to English are decided here.

    **The occupation comes from `professional_activity`, with `main_occupation` as a
    fallback.** Since the v6 cohort, both are English at the source; the order stays this
    one because EARLIER cohorts carry `main_occupation` in French, and an archived trace
    re-read must keep returning an English narrative.

    **The usual trip purposes are served** (`travel_purposes`). They were produced by the
    generator, recomputed by `align_minor_traits`, documented as "the list seen by the
    LLM" — and read by nobody. A field maintained for nothing is more misleading than a
    missing field: it makes one believe the information is served.

    **`_income_map` is gone.** This table only existed to frenchify a field already in
    English in the population (`Very Low`, `Medium-High`…). Removing it returns the value
    as it is; there is nothing to translate, only a translation to remove.
    """
    name = traits.get("name", "")
    first_name = name.split()[0] if name else ""
    age = traits.get("age", "")
    occupation = traits.get("professional_activity") or traits.get(
        "main_occupation", ""
    )
    household = traits.get("household_size")
    income = traits.get("income") or ""

    extras = []
    if household == 1:
        extras.append("lives alone")
    elif household:
        extras.append(f"household of {household}")
    if income:
        extras.append(f"{income.lower()} income")
    line1 = f"{first_name}, {age}, {occupation}"
    if extras:
        line1 += f" ({', '.join(extras)})"

    # Usual purposes. A person who never travels has none: the sentence is then OMITTED rather
    # than rendered empty — "Usual trip purposes:" with nothing after would read as missing
    # information, when it is information saying "this person does not go out".
    motifs = [
        str(m).strip() for m in (traits.get("travel_purposes") or []) if str(m).strip()
    ]
    if motifs:
        line1 += f". Usual trip purposes: {', '.join(motifs)}"

    # Residence ring. SERVED to the model since ticket 074: it places the person in the urban
    # structure, which no other information in the prompt says — the address is not there,
    # and a trip's distance says nothing about where one lives.
    #
    # That is also what switches it to English AT THE SOURCE: as long as it was only a join
    # key to the survey, its language committed nobody; served to the model, it falls under
    # the same rule as the rest of the prompt.
    #
    # `hors périmètre` is NOT a ring: it is the fifth category, and serving it as is is
    # better than hiding it — the home is known, it is simply outside.
    couronne = str(traits.get("residence_zone") or "").strip()
    if couronne:
        line1 += f". Lives in: {couronne}"

    # The "Mobility:" line disappeared on 2026-08-26. What it carried:
    #
    # * `car_availability` and the driver status — REMOVED. The option set already says
    #   whether the car can be taken (the controller's `_owns_car` / `_can_drive` offer it
    #   or not), and the narrative channel was measured then rejected: +0.12 pt of car
    #   share, at noise level (ticket 018, docs/traces/2026-08-24_car_availability).
    # * the personal bike — REMOVED for the same reason. ⚠ With an accepted loss: an
    #   agent who owns a bike parked elsewhere (vehicle chain) has no bike option, and
    #   the prompt no longer says it has one. Since it cannot use it, the information
    #   carried no decision.
    # * the PT pass — MOVED to the PT option (cf. `pt_subscription_suffix`): it can NOT
    #   be deduced from the option set (a bus option exists, pass or not), so it stays
    #   served — but where it weighs, and only when PT is offered.
    #
    # Only the social identity remains, the only information of the block that the options
    # do not carry. An empty line is not rendered.
    return line1


def _axes_de_la_decision(context: Context, plan, destination, weather) -> dict:
    """Normalised axes of an entry written by an itinerary decision (lot 2).

    This is the only place where the chosen mode, the purpose and the day's weather are
    known together. Normalise them here, once, rather than at each recall.

    `axe_lieu` stays empty for a decision: it carries an itinerary, not a stop. It will be
    filled by concepts, whose spatial scope designates one. Empty is better than the
    destination: two trips to the same workplace by two different modes are not the same
    place memory.
    """
    return {
        "axe_objet": mode_canonique(plan.mode_label() if plan is not None else None),
        "axe_lieu": None,
        "axe_creneau": creneau_de(wall_clock(context.timestamp)),
        "axe_motif": normaliser_motif(destination),
        "axe_meteo": meteo_de(weather),
    }


@dataclass
class ConceptLu:
    """A concept returned by the model, brought to a single form.

    TWO input FORMATS are accepted, and this is not leniency:

    - the OBJECT, since lot 1 of ticket 071, which carries `severity`, `valence`, then `mode`
      (lot 2) and `operation` / `target_id` (lot 3);
    - the ARRAY of five strings, the earlier format, which a RESUMED run re-reads in its own
      reflection cache and in the concepts already written. Without this tolerance, resuming
      a run would lose the accumulated concepts.

    The stored `content` stays the 5-tuple in both cases: severity, valence and axes live on
    the entry's FIELDS, not in its text. No downstream reader changes.
    """

    cinq: list
    niveau: str | None = None
    valence: str = "neutre"
    mode: str | None = None
    operation: str = CREER
    cible: str = ""
    # Ticket 100, lot 4 — the PROVENANCE. `vecu` by default: a concept written before this lot,
    # or by a model that ignores the field, comes from the agent's day — that is what was true
    # before the household existed, and it is the least inventive default.
    origine: str = "vecu"


# Schema vocabulary (English, ticket 074) → `MemoryEntry` vocabulary (French).
_ORIGINES_MODELE: dict[str, str] = {"lived": "vecu", "heard": "entendu"}


def _normaliser_origine(brut) -> str:
    """The provenance returned by the model, brought to the entry's vocabulary.

    An unknown or missing value falls back on `vecu`, and LEAVES A TRACE. It is the least
    inventive fallback — the concept is taken as born from the agent's day, which was true
    before the household existed — but it is not harmless: a hearsay concept taken as lived
    WOULD GO BACK into the household, which decision D2 forbids. A model that would never
    respect this field must therefore show.
    """
    clef = str(brut or "").strip().lower()
    if clef in _ORIGINES_MODELE:
        return _ORIGINES_MODELE[clef]
    if clef in ("vecu", "entendu", "lu"):
        return clef
    if clef:
        logger.warning(
            f"[concepts] UNKNOWN provenance \"{brut}\" — outside "
            f"{sorted(_ORIGINES_MODELE)}; the concept is taken as LIVED, so it may go back "
            f"into the household. Check that the model respects the `source` field."
        )
    return "vecu"


def _normaliser_concept(concept) -> ConceptLu:
    if isinstance(concept, dict):
        cinq = [
            str(concept.get("content", "")),
            str(concept.get("keywords", "")),
            str(concept.get("spatial_scope", "")),
            str(concept.get("temporal_scope", "")),
            str(concept.get("purpose", "")),
        ]
        mode = str(concept.get("mode") or "").strip().lower() or None
        if mode == "any":
            # "any" is an ANSWER, not an absence: the concept speaks of no mode.
            # It is `None` as an axis — and a missing axis matches nothing, which is
            # correct: this concept has no reason to come up on behalf of a mode.
            mode = None
        return ConceptLu(
            cinq=cinq,
            niveau=concept.get("severity"),
            valence=str(concept.get("valence") or "neutre"),
            mode=mode,
            operation=normaliser_operation(concept.get("operation")),
            cible=str(concept.get("target_id") or "").strip(),
            origine=_normaliser_origine(concept.get("source")),
        )
    if isinstance(concept, (list, tuple)):
        cinq = [str(x) for x in list(concept)[:5]]
        cinq += [""] * (5 - len(cinq))
        # Format from before lot 1: neither level, nor mode, nor operation were asked. The
        # concept will fall back on its group's deterministic severity — which is correct — and
        # on `creer`, the least destructive fallback.
        return ConceptLu(cinq=cinq)
    return ConceptLu(cinq=[str(concept), "", "", "", ""])


def _concepts_du_jour(long_term_memory, person_id: str, paniers: set) -> list:
    """Existing concepts of the baskets touched by the day (ticket 071, lot 3).

    ⚠ **The basket cannot be computed afterwards.** It is defined on the (mode, purpose)
    pair of the NEW concept, but the candidates must be shown to the model BEFORE it writes
    its own — and a second call is ruled out by the ticket's cross-cutting constraint: no
    lot costs an extra call.

    Hence the chosen scope: the baskets of the **entries consumed during the day**, that is
    the modes and purposes the agent actually used. It is exactly the right scope — it
    reflects on its day, the concepts it could correct are those that speak of it.

    OUT-OF-SERVICE concepts are not included: there is no reason to offer the model to
    correct what is no longer served to it.
    """
    if long_term_memory is None or not paniers:
        return []
    long_term_memory.ensure_user_initialized(person_id)
    entrees = long_term_memory.user_metadata.get(person_id, {}).get("entries", [])

    par_panier: dict = {}
    for e in entrees:
        if str(e.memory_type) != str(MemoryType.CONCEPT.value) or not e.doc_id:
            continue
        if not e.est_servi or e.depasse_le:
            continue
        clef = panier_de(e.axe_objet, e.axe_motif)
        if clef in paniers:
            par_panier.setdefault(clef, []).append(e)

    montres = []
    for clef, candidats in par_panier.items():
        # The most confident first, the last observation breaking ties: it is what the
        # agent holds as surest, hence what it has the most reasons to correct.
        candidats.sort(
            key=lambda e: (
                e.confiance,
                e.derniere_observation or e.timestamp,
            ),
            reverse=True,
        )
        montres.extend(candidats[:CONCEPTS_MONTRES_PAR_PANIER])
    return montres


def _gravite_de_la_contrainte(context: Context) -> float:
    """Deterministic severity of a decision, from the chain constraint (ticket 071, lot 1).

    It is the fourth component of `I_det`, "constrained mode change". It comes neither
    through the arrival nor through the missed bus but through the DECISION path: it is
    there, and only there, that one knows the agent had to give up a mode — it goes back with
    what it took (`retour_force`), or cannot leave as it would have wanted (`sortie_bloquee`).

    Being a passenger of another household member is excluded: it is an arrangement, not a
    suffered degradation (cf. `CONTRAINTES_MODE_FORCE`).
    """
    contrainte = str((context.data or {}).get("contrainte_chaine") or "")
    valeur, _detail = gravite_deterministe(
        mode_contraint=contrainte in CONTRAINTES_MODE_FORCE
    )
    return valeur


# Version of the output schema of the SHORT reflection. To be incremented as soon as its
# contract changes (fields, or a field's semantics): it goes into the memoisation key and thus
# invalidates the cache accumulated under the old contract, instead of serving it wrongly.
#   1 — ticket 071, lot 1: each concept becomes an object and carries `severity` and `valence`.
#   2 — ticket 071, lot 2: each concept declares its `mode`. Without it, the recall's
#       per-object pool would never find anything: long-term memory only contains
#       reflections and concepts, and none would say which mode it speaks of.
#   3 — ticket 071, lot 3: each concept declares its `operation` and its target. Concepts
#       stop piling up: they are confirmed, refined or contradicted.
#   4 — ticket 100, lot 4: each concept declares its PROVENANCE (`lived` | `heard`). Without
#       it, a concept born from hearsay is indistinguishable from one born from a trip, and
#       decision D2 — a single hop — has no way to apply. Ticket 078 § 4.1 refused this
#       genealogy; it becomes necessary once circulation is bounded to one hop.
#       ⚠ The field is asked in BOTH arms, sharing flag on or off: that is what keeps a
#       single schema version between them, hence comparable. Flag off, no household block
#       enters the call and the answer is `lived` everywhere.
#   5 — the paper served for five days carries a structured acknowledgement. A v4 answer
#       could ignore it while being served again by the cache.
SCHEMA_REFLEXION_VERSION = 5


class LlmAgent:
    DEFAULT_IDENTITY = ""

    def __init__(self):
        self.short_term_memory: dict[str, UserShortTermMemory] = {}
        if settings.agent.long_term_memory_enabled:
            self.long_term_memory = MultiUserLongTermMemory(
                storage_dir=settings.agent.long_term_memory_storage_dir,
                long_term_memory_filter_by_datetime=settings.agent.long_term_memory_filter_by_datetime,
                max_loaded_metadata=settings.agent.long_term_max_loaded_metadata,
            )
        else:
            self.long_term_memory = None
            logger.info("Long-term memory disabled — ChromaDB initialization skipped")

        # LLM client instance (natural singleton for this Agent) — typed SDK
        # (httpx AsyncClient reused across calls, TaskResult results)
        self.llm_client = LLMGatewayClient(
            base_url=os.getenv("LLM_API_URL", "http://localhost:8000"),
            wait_timeout=settings.agent.remote_llm_poll_timeout,
            backpressure_max_inflight=settings.world.worker_concurrency,
            backpressure_release_ratio=settings.agent.remote_llm_backpressure_ratio,
            circuit_failure_threshold=settings.agent.remote_llm_circuit_failure_threshold,
            circuit_probe_interval=settings.agent.remote_llm_circuit_probe_interval,
            # Ticket 092 — set on the CLIENT: it covers any call added later.
            # Ticket 095, lot C — the known calls now set their allowlist PER CATEGORY in
            # their payload, which the SDK respects as is. This one remains the safety net:
            # the UNION of everything allowed, so that a new call is restricted by default
            # rather than free.
            instances_admises=toutes_les_instances(),
            # The run name signs each exchange: the worker's log is shared by all clients,
            # and this field is what makes it possible to read only one's own there.
            origine=Path(settings.workdir).name,
            # Both arms of an A/B share this space: the control receives the treated arm's
            # answers as long as its prompts are the same, word for word (exact-prompt replay).
            espace_rejeu=settings.llm.rejeu_ab or None,
            rejeu_strict_avant_ts=settings.llm.rejeu_strict_avant_ts or None,
        )
        self.prompt_manager = PromptManager(
            os.path.join(os.path.dirname(__file__), "prompts")
        )

        if settings.cache.enabled:
            population_name = f"{settings.data.synthetic_file_prefix}population_{settings.data.population_size}"
            # Cache isolation per system prompt version: if the active prompt
            # (mobility_llm/prompts/prompts.yaml) changes, the checksum changes and the cache
            # starts afresh instead of reusing obsolete decisions.
            prompt_checksum = _mobility_prompt_manager().active_prompt_checksum()
            cache_dir = os.path.join(
                settings.cache.cache_dir, prompt_checksum, population_name
            )
            logger.info(
                f"LLM cache isolé par prompt — checksum={prompt_checksum}, dir={cache_dir}"
            )
            self.llm_cache = LlmSemanticCache(
                cache_dir=cache_dir,
                semantic_threshold=settings.cache.semantic_threshold,
                embed_model_name=settings.cache.embed_model_name,
            )
            # Exact memoisation of reflections (ticket 012) — same directory as the
            # decision cache: the isolation by prompt checksum is inherited.
            self.reflection_memo = (
                ReflectionMemoStore(cache_dir=cache_dir)
                if settings.cache.reflection_memo_enabled
                else None
            )
        else:
            self.llm_cache = None
            self.reflection_memo = None

    def get_short_term_memory(self, user_id: str) -> UserShortTermMemory:
        if user_id not in self.short_term_memory:
            self.short_term_memory[user_id] = UserShortTermMemory(user_id)
        return self.short_term_memory[user_id]

    def _weather_info(self, context: Context) -> dict:
        """Complete weather information for this decision context (ticket 107).

        Returns:
            "source_date_meteo": "tiree" | "declaree" | "horloge",
            "jour_meteo_tire": JourMeteo("MM-JJ") | None,
            "jour_meteo_lu": str (ISO du CSV) | None,
            "weather_timestamp": int,
        """
        from urban_mobility_agents.utils.weather_draw import resoudre_meteo_decision

        return resoudre_meteo_decision(
            person_id=str(context.person.person_id),
            timestamp=int(context.timestamp),
        )

    def _weather_timestamp(self, context: Context) -> int:
        """Timestamp used to read this agent's weather forecast.

        By default, the simulated clock — historical behaviour. When
        `weather_per_agent_dates` is on, only the DATE is replaced by a day of the
        year — **the one the person actually described** when a table of declared
        dates comes with the population (ticket 058), otherwise a day drawn
        deterministically from the agent's identifier: on a single simulated day,
        all agents would otherwise share a single weather, and the weather effect
        would be unmeasurable by construction (ticket 023). The departure time is
        kept, and so is the transport offer: only the weather varies.
        """
        return int(self._weather_info(context)["weather_timestamp"])

    def add_short_term_memory(
        self,
        context: Context,
        msg: str,
        timestamp: int | None = None,
        importance: float = 0.0,
        axes: dict | None = None,
        valence: str = "neutre",
        origine: str | None = None,
    ):
        # Ticket 075 — replay of a hot resume: the agent relives a day it has ALREADY
        # learnt. The choke point is here: without a short-term memory entry, no agent
        # becomes eligible for consolidation, so nothing is rewritten in long-term memory.
        # Decisions and trips, for their part, go on — the simulation does need to get
        # back to its state.
        if gel_actif():
            return
        memory = self.get_short_term_memory(context.person.person_id)
        # The memory's timestamp is GAMA's WALL-CLOCK time. This is not cosmetic:
        # this `datetime` ends up in the PROMPT ("- Time 16 March 2026, 05:12: …" via
        # `humanize_date`, and the day name of reflection memories), and it is the
        # left-hand side of the LTM filters by day and by age. Read in the process
        # time zone, it announced 06:12 for 5:12 wall-clock.
        memory.add_message(
            msg,
            wall_clock(timestamp or context.timestamp),
            activity_id=context.activity_id,
            importance=importance,
            axes=axes,
            valence=valence,
            origine=origine,
        )
        history_log.log_shortterm_memory(
            timestamp=context.timestamp,
            person_id=context.person.person_id,
            activity_id=context.activity_id,
            message=msg,
            data=context.data,
        )

    def note_decision_contrainte(self, context: Context, plan, destination) -> None:
        """Writes to short-term memory a trip whose mode was not CHOSEN.

        Ticket 077, lot C. A single-itinerary trip does not call the model: it therefore
        went through neither of the two places that write a decision entry, and memory
        simply ignored that a trip had taken place. On the thirty-day run of 075,
        **116 trips out of 514** were in this case — all the returns of rural agents, that
        is half of what they lived.

        Two consequences, both measured. The evening reflection received the trip's
        physical observations without the decision that explains them. And the habit log,
        which is fed on arrival by looking up the matching decision, found none.

        **The text says the choice was constrained, and does not claim a model chose.** An
        agent re-reading itself must be able to tell what it decided from what it
        underwent: making it read "chosen by gateway LLM" on a trip without an alternative
        would make it derive a preference from an absence of option — exactly the pattern
        this repository hunts, where the absence of measurement readily yields the perfect
        score.
        """
        if plan is None:
            return
        plan_summary = env_ob_to_text("travel_plan", plan.model_dump())
        stm_msg = (
            f"[ TRAVEL_PLAN ] Plan to head <{destination}>. "
            f"No choice was possible: this was the only available itinerary.\n"
            f"{plan_summary}\n"
            f"Reasoning: the mode was imposed by the absence of any alternative, "
            f"not preferred."
        )
        self.add_short_term_memory(
            context,
            stm_msg,
            timestamp=context.timestamp,
            importance=_gravite_de_la_contrainte(context),
            axes=_axes_de_la_decision(
                context,
                plan,
                destination,
                get_weather(self._weather_timestamp(context)),
            ),
        )

    async def aadd_long_term_memory(self, context: Context, msg: MemoryEntry):
        await self.long_term_memory.aadd_memory(msg)
        history_log.log_longterm_memory(
            timestamp=context.timestamp,
            person_id=context.person.person_id,
            message=msg.content,
            data=context.data,
        )

    def parse_response_json(self, response: str) -> tuple[dict | None, str]:
        try:
            match = re.search(r"\{.*\}", response, re.DOTALL)
            assert match is not None, "No JSON found in response"

            json_str = match.group(0)
        except Exception as e:
            traceback.print_exc()
            print(f"Error parsing response: {e}, response raw: {response}")
            json_str = response.strip()

        try:
            parsed = json.loads(json_str)
            return parsed, ""
        except Exception as e:
            traceback.print_exc()
            print(f"Error parsing response: {e}, response raw: {response}")

        try:
            parsed = demjson3.decode(json_str)
            return parsed, ""
        except demjson3.JSONDecodeError as e:
            traceback.print_exc()
            print(f"Error parsing response: {e}, response raw: {response}")

        return None, response.strip()

    def get_person_identity_description(self, person: Person) -> str:
        return _build_profile_narrative(person.identity.traits_json)

    async def query_past_experiences_for_travel(
        self,
        context: Context,
        options: list[TravelPlan],
        lignes: tuple[str, ...] | list[str] = (),
    ) -> list[str]:
        def get_plan_text(plan: TravelPlan) -> str:
            return env_ob_to_text("travel_plan_query", plan.model_dump())

        index = 1
        travel_options = ""
        for option in options:
            travel_options += f"{index}. \n{get_plan_text(option)}\n"
            index += 1

        text = self.prompt_manager.get_prompt(
            PromptName.QUERY_EXPERIENCES,
            current_time=humanize_date_short(context.timestamp),
            temporal_keyword=categorize_date_time_short(context.timestamp),
            weekday=get_weekday_category(context.timestamp).upper(),
            destination=options[0].purpose,
            travel_options=travel_options,
        )

        # logger.debug(f"Querying experiences with travel plans for user {context.person.person_id}, activity {context.activity_id}, query text: {text}")

        # Ticket 071, lot 2 — the OFFERED modes open the per-object pool, and the current
        # context feeds the two affinities. Without the modes, pool B would not know what to
        # look for; without the context, the affinities would be zero everywhere, which is
        # the score of a total mismatch and not that of an absence of information.
        modes_offerts = {
            m for m in (mode_canonique(o.mode_label()) for o in options) if m
        }
        contexte_axes = {
            "axe_objet": modes_offerts,
            "axe_lieu": None,
            "axe_creneau": creneau_de(wall_clock(context.timestamp)),
            "axe_motif": normaliser_motif(options[0].purpose if options else None),
            "axe_meteo": meteo_de(get_weather(self._weather_timestamp(context))),
        }

        hist = await self.long_term_memory.aquery_user_memories(
            person_id=context.person.person_id,
            query=text,
            top_k=settings.agent.long_term_max_entries_query,
            max_past_days=settings.agent.long_term_max_days_query,
            query_at=context.timestamp,
            modes_offerts=sorted(modes_offerts),
            contexte=contexte_axes,
        )

        # deduplicate entries based on content
        unique_hist = {}
        for entry in hist:
            if entry.content not in unique_hist:
                unique_hist[entry.content] = entry
        hist = sorted(
            list(unique_hist.values()),
            key=lambda x: x.metadata["timestamp"],
            reverse=True,
        )

        # logger.debug(f"Found {len(hist)} relevant experiences for travel plans for user {context.person.person_id}, activity {context.activity_id}")

        # resp = [
        #     # f"[{datetime.strftime(datetime.fromisoformat(entry.metadata['timestamp']), '%A, %H:%M:%S')}] {entry.content}"
        #     # TODO: we asked the LLM to return the time within the day, so only need to append the day of week here
        #     f"[{datetime.strftime(datetime.fromisoformat(entry.metadata['timestamp']), '%A')} at {time_to_bucket_text(datetime.fromisoformat(entry.metadata['timestamp']).timestamp())}] {entry.content}" if entry.content else ""
        #     for entry in hist
        # ]

        resp = []
        ts = []
        for entry in hist:
            if str(entry.metadata["memory_type"]) == str(MemoryType.REFLECTION.value):
                date_str = datetime.strftime(
                    datetime.fromisoformat(entry.metadata["timestamp"]), "%A, %B %d"
                )
                resp.append(f"[{date_str}] {entry.content}")
                ts.append(
                    datetime.fromisoformat(entry.metadata["timestamp"]).timestamp()
                )
            elif str(entry.metadata["memory_type"]) == str(MemoryType.CONCEPT.value):
                concept = json.loads(entry.content)
                resp.append(f"[Concept] {concept[0]}" if concept else "")
                ts.append(
                    datetime.fromisoformat(entry.metadata["timestamp"]).timestamp()
                )
            else:
                logger.debug(
                    f"Unknown memory type for entry: {entry.metadata['memory_type']}"
                )

        # sort the entries by timestamp asc
        ts = np.array(ts)
        sorted_indices = np.argsort(ts)
        resp = [resp[i] for i in sorted_indices]

        # ── Ticket 071, lot 4 — the CORE MEMORY ──────────────────────────────────
        # The ten raw memories give way to a permanent structured block, completed with two
        # or three episodic entries recalled for the current decision. This is MemGPT's
        # *working context*: a block always present, the rest paged on demand.
        #
        # The three blocks are COMPUTED, none is written by the model — a text periodically
        # rewritten by a model drifts and invents, a computed block stays checkable against
        # its source. The lot therefore costs NO extra call.
        try:
            _entrees = self.long_term_memory.user_metadata.get(
                context.person.person_id, {}
            ).get("entries", [])
            _bloc = memoire_noyau(
                self.long_term_memory.journal_trajets(context.person.person_id),
                _entrees,
                wall_clock(context.timestamp),
                context.person.person_id,
                lignes,
            )
        except Exception as err:  # noqa: BLE001
            # A block that fails to build must not lose the decision — but it must not vanish
            # silently either: without it, the agent falls back to the behaviour from before
            # lot 4 without anything saying so.
            logger.warning(
                f"[noyau] mémoire noyau non construite pour {context.person.person_id} "
                f"({err}) — repli sur les souvenirs bruts seuls"
            )
            # Ticket 111 — the GUARANTEED line survives a block that fails to build: that is
            # precisely what it guarantees.
            _bloc = (
                [TITRE_CHANGEMENTS, *(f"- {ligne}" for ligne in lignes)]
                if lignes
                else []
            )

        if not _bloc:
            return resp

        # The episodic ones are the MOST RECENT of the top-K, and there are two or three, not
        # ten: the block now carries what was repetitive in the ten.
        _n = int(settings.agent.memoire__episodiques_avec_noyau)
        return _bloc + resp[-_n:] if _n > 0 else _bloc

    def get_personal_system_prompt(self, person: Person) -> str:
        identity_description = self.get_person_identity_description(person)
        return self.prompt_manager.get_prompt(
            PromptName.PERSONAL_SYSTEM, identity_description=identity_description
        )

    async def build_travel_plan_payload(
        self,
        context: Context,
        options: list[TravelPlan],
        destination: str,
        departure_time: int = 0,
        anticipation: dict | None = None,
        lignes: tuple[str, ...] | list[str] = (),
    ) -> dict[str, Any]:
        agent_id = context.person.person_id
        perception = self.get_person_identity_description(
            context.person
        )  # TODO To be remplace by feeling and perception about transport modes
        current_time = humanize_time(context.timestamp)
        city_context = (
            weather_to_natural_language(get_weather(self._weather_timestamp(context)))
            or "None"
        )

        history = []
        if settings.agent.long_term_memory_enabled:
            _pl = PipelineLogger.get()
            _rec = _pl.get_record(context.person.person_id) if _pl is not None else None
            if _rec is not None:
                _rec.T_ltm_start = time.time()
            history = await self.query_past_experiences_for_travel(
                context, options, lignes
            )
            if _rec is not None:
                _rec.T_ltm_end = time.time()

        # PT pass: carried by the option, not by the persona (cf. `_pt_subscription_note`).
        _has_pt = bool(
            (context.person.identity.traits_json or {}).get(
                "has_pt_subscription", False
            )
        )

        def _describe(opt: TravelPlan) -> str:
            """Text of the option, with the pass mention on its FIRST line.

            The following lines are the itinerary's legs, re-indented as sub-bullets by
            the template: attaching the mention there would make it look like a leg. It
            is therefore attached to the summary sentence.
            """
            text = env_ob_to_text("travel_plan", opt.model_dump())
            note = _pt_subscription_note(opt.mode_label() or "", _has_pt)
            if not note:
                return text
            head, sep, tail = text.partition("\n")
            return f"{head.rstrip()}{note}{sep}{tail}"

        # List comprehension for performance and clarity
        trajectories = [
            {
                "index": i,
                "mode": opt.mode_label() or "unknown",
                "description": _describe(opt),
                # Total trip distance (in metres) — used for the Prometheus metrics
                "total_distance_m": (
                    opt.distance
                    if opt.distance is not None
                    else sum(leg.get_distance() for leg in (opt.legs or []))
                ),
            }
            for i, opt in enumerate(options)
        ]

        dest_zone = (
            options[0].end_location.zone
            if options and options[0].end_location
            else None
        )

        return {
            "category": "itinary_multi_agent",
            # Ticket 095, lot C — the allowlist is set PER CATEGORY, and not once for the
            # whole run. The decision is the paper's measured variable: it does not share
            # its queue with the nightly reflection, which consumes as many tokens.
            "instances_admises": instances_pour("itinary_multi_agent"),
            "agents": [
                {
                    "agent_id": agent_id,
                    # `Contraintes : None` removed on 2026-08-26: a hard-coded literal,
                    # never implemented, measured constant on 2,487 records out of 2,487.
                    "perception": perception,
                    "destination": destination,
                    "destination_zone": dest_zone,
                    "departure_time": humanize_time(departure_time)
                    if departure_time
                    else None,
                    "departure_timestamp": float(departure_time)
                    if departure_time
                    else None,
                    "current_time": current_time,
                    # Weather/traffic specific to the agent (and no longer at request level):
                    # the batch key hashes `parameters`, so by taking the weather out of
                    # parameters, requests with different weathers can merge into the
                    # same LLM call — each persona keeps its own in the prompt
                    # (cf. itinary_multi_agent.md.j2, per-block injection).
                    "context": city_context,
                    # Chain anticipation (ticket 014): weather of the remaining slices
                    # of the day and rolling agenda of the remaining trips — built by
                    # the controller (_build_anticipation), rendered per block in the
                    # template. The vehicles' position is no longer stated (measured
                    # bike bias): the chain rule lives in the system prompt
                    # (expert_m4 variant).
                    "day_outlook": (anticipation or {}).get("outlook"),
                    "agenda": (anticipation or {}).get("agenda") or [],
                    "history": history,
                    "trajectories": trajectories,
                }
            ],
            "parameters": {**settings.agent.llm_params},
        }

    async def evaluate_and_choose_travel_plan(
        self,
        context: Context,
        options: list[TravelPlan],
        destination: str,
        departure_time: int = 0,
        anticipation: dict | None = None,
        *,
        force_provider: str | None = None,
        allowed_providers: set | None = None,
        trace: dict | None = None,
        presentation_figee: bool = False,
        option_order_seed: int | None = None,
    ) -> tuple[int, str, str, dict]:
        """Chooses an itinerary and returns (index, justification, provider, distribution).

        Ticket 035 (spec 02/05) — optional kwargs, with no effect when absent:
        `force_provider` pins a gateway instance; `allowed_providers` refuses any response
        served by an instance outside this set (substitution refused, Q3); `trace` (dict
        provided by the caller) receives what was presented, the raw response, the weights
        and the provider (D6); `presentation_figee` keeps the order received (the caller
        has already ordered it); `option_order_seed` replaces the settings' seed.

        The distribution is the probability distribution per canonical mode used for the
        draw (modes not offered included, at 0) — empty if the decision does not come from
        one (old-format response, inherited cache point, error). It is traced as is in
        `moves.csv`.

        `anticipation` (ticket 014): anticipation context built by the controller —
        injected into the prompt, and its `signature` goes into the decision cache key
        (two different anticipations = two distinct entries).
        """
        assert options, "No travel options provided for planning trip."
        anticipation_key = (anticipation or {}).get("signature", "")
        # Ticket 090 — the decision time, read BEFORE any wait: it is this one, and not the
        # activity instant, that bounds the trace on resume (2026-09-28).
        _decide_a = rejeu_decisions.horloge()

        # ACCIDENT GUARD (ticket 070, work E). The decision cache key is built on the OPTION
        # CODES — routes and stops — hence insensitive to DURATIONS by construction
        # (`llm/cache.py`, `_make_state_hash`). Without this complement, an agent whose trip
        # has just been lengthened by twenty minutes would be served again the decision it
        # had taken without the delay, and no log would report it.
        #
        # This is the FOURTH occurrence of the same trap in the repository — after the terminal
        # time, the persona traits and the anticipation context, one of which cost a manual
        # cache flush. `extra_key` exists precisely for that.
        if departure_time:
            from trip_helper import accidents as _accidents

            _registre = _accidents.registre()
            if _registre is not None:
                _sig = _registre.signature_active(int(departure_time))
                if _sig:
                    anticipation_key = f"{anticipation_key}|accidents:{_sig}"

        # Deterministic order for the cache keys (independent of the shuffle)
        sorted_options = sorted(options, key=lambda p: p.get_code() or "")
        # DETERMINISTIC presentation order (ticket 035, D7): derived from the seed, the
        # agent and the activity — the position bias is still shuffled, but the same trip
        # is presented in the same order in both execution modes.
        if presentation_figee:
            shuffled_options = list(options)
        else:
            shuffled_options = ordre_presentation(
                options,
                settings.agent.option_order_seed
                if option_order_seed is None
                else option_order_seed,
                context.person.person_id,
                context.activity_id,
            )

        activity_purpose = options[0].purpose or ""
        info_meteo = self._weather_info(context)
        if trace is not None:
            trace["jour_meteo_tire"] = info_meteo["jour_meteo_tire"]
            trace["jour_meteo_lu"] = info_meteo["jour_meteo_lu"]
            trace["source_date_meteo"] = info_meteo["source_date_meteo"]
        weather = get_weather(info_meteo["weather_timestamp"])

        # Draw seed: the simulated day is part of it, so the same context replayed the
        # next day draws another mode (including on a cache hit), while a run relaunched
        # identically reproduces exactly the same trips.
        # `tz=timezone.utc` already returns GAMA's WALL-CLOCK DAY (the definition of
        # `sim_clock.wall_clock`) and does not depend on the process `TZ`. Left word for
        # word: this string goes into the mode draw seed, and rewriting it — even to an
        # identical value — would bring nothing but a risk of reshuffling all the draws
        # already measured.
        seed_parts = (
            settings.agent.mode_draw_seed,
            context.person.person_id,
            context.activity_id,
            datetime.fromtimestamp(context.timestamp, tz=timezone.utc).strftime(
                "%Y-%m-%d"
            ),
        )

        # --- Hybrid cache (before the LLM call) ---
        # Without memories, the decision only depends on factual conditions: exact match,
        # and the payload — hence the LTM query — is only built on a miss.
        # With memories, the agent's experience weighs on the decision: the payload must be
        # built first to compare the current LTM with the one that produced the cached decision.
        has_memories = (
            self.long_term_memory is not None
            and self.long_term_memory.has_memories(context.person.person_id)
        )
        # ── Ticket 111 — what is GUARANTEED to the prompt that day, computed BEFORE the cache ──
        # The article read (reader) or what a household member said about it (informed), during
        # its service days. The day is that of the TRIP, not of the computation: the decision of
        # the reading day is computed the day before, before the injection, and that is what
        # deprived it of the article. Empty outside the setup — the rest of the method is then
        # unchanged.
        from llm import evenements as _evenements

        _lignes = await _evenements.lignes_du_jour(
            context.person.person_id, int(departure_time or context.timestamp)
        )
        payload = None
        memory_text = None
        if has_memories:
            payload = await self.build_travel_plan_payload(
                context, shuffled_options, destination, departure_time, anticipation,
                lignes=_lignes,
            )
            # Memory text: serialisation of the history field already computed in the payload
            memory_text = json.dumps(
                payload["agents"][0].get("history", []), ensure_ascii=False
            )

        # ── Ticket 075 — during the REPLAY of a hot resume ───────────────────────────────
        # The similarity branch compares the CURRENT memory with the one that produced the
        # cached decision. But during a replay memory is frozen at the resume point, hence
        # never again identical to what it was at that instant of the original run: the
        # 0.95 threshold is not reached and the decision goes back to the model. Measured on
        # 2026-09-14 on a one-day replay: only 33% served, eleven decisions out of eighteen
        # paid again. On a forty-day replay, that would be more than a day of quota spent
        # relearning what is already known.
        #
        # ⚠ This is NOT a degradation, and it is the opposite of a fallback: the EXACT branch
        # addresses the point written by the original run for this same agent, this same
        # activity, this same slot, this same weather and these same options. What it returns
        # is therefore the decision the original run DID TAKE. A new call, for its part, would
        # return a decision taken on a memory the agent did not yet have at that moment.
        # A miss remains a miss: the model is called normally.
        _memory_text_cache = None if gel_actif() else memory_text

        # Ticket 100 (Q5, settled on 2026-09-22) — the decision cache is BYPASSED on event
        # days, automatically. That day, each decision goes through the model: it is the day
        # the agent decides seeing the nominal offer, and a decision served again would pass
        # off a choice from another day as a choice from this one.
        #
        # ⚠ The cut ONLY applies to that day. The window after it — the one being measured —
        # keeps its cache, and the `evenements` counters say day by day how many decisions
        # were served there from the cache.
        _cache_coupe = _evenements.cache_coupe(context.timestamp)
        if self.llm_cache is not None and _lignes:
            # Ticket 111 — a decision that must carry a line NEVER consults the cache.
            # `cache_coupe` only covers the draw window: a service day located after it
            # could otherwise bring out a decision taken without the article.
            _evenements.noter_decision(context.timestamp, depuis_cache=False)
            _evenements.noter_contournement_cache()

        if self.llm_cache is not None and not _cache_coupe and not _lignes:
            cache_hit = await self.llm_cache.lookup(
                agent_id=context.person.person_id,
                activity_id=context.activity_id,
                timestamp=context.timestamp,
                options=sorted_options,
                memory_text=_memory_text_cache,
                weather=weather,
                activity_purpose=activity_purpose,
                seed_parts=seed_parts,
                extra_key=anticipation_key,
                traits_key=_traits_signature(context.person.identity.traits_json),
            )
            _evenements.noter_decision(context.timestamp, depuis_cache=cache_hit is not None)
            if cache_hit is not None:
                if trace is not None:
                    trace["cache"] = True
                chosen_plan = sorted_options[cache_hit["index"]]
                original_index = options.index(chosen_plan)
                if cache_hit.get("distribution"):
                    reason = (
                        "Mode tiré au sort dans la distribution mise en cache : "
                        + _format_distribution(cache_hit["distribution"])
                    )
                else:
                    reason = "Décision récupérée depuis le cache sémantique LLM."
                plan_summary = env_ob_to_text("travel_plan", chosen_plan.model_dump())
                stm_msg = f"[ TRAVEL_PLAN ] Plan to head <{destination}> served from LLM cache.\n{plan_summary}\nReasoning: {reason}"
                self.add_short_term_memory(
                    context,
                    stm_msg,
                    timestamp=context.timestamp,
                    importance=_gravite_de_la_contrainte(context),
                    axes=_axes_de_la_decision(
                        context, chosen_plan, destination, weather
                    ),
                )
                logger.debug(
                    f"Cache hit for person {context.person.person_id}, activity {context.activity_id}, returning cached plan with reason: {reason}"
                )
                # jsonl writing offloaded out of the event loop (blocking open/write)
                await asyncio.to_thread(
                    log_llm_cache_hit,
                    agent_id=context.person.person_id,
                    activity_id=context.activity_id,
                    sim_ts=float(context.timestamp),
                    mode=cache_hit.get("mode", ""),
                )
                return (
                    original_index,
                    reason,
                    f"cache:{cache_hit.get('mode', '')}",
                    cache_hit.get("distribution") or {},
                )

        # Ticket 090 — during the replay of a hot resume, serve again the decision THIS run
        # has already taken, instead of paying for it again. The `gel_actif()` condition is
        # essential: outside the replay window, the trace would become a permanent cache, and
        # the run would stop being logged over its full scope. A missed key is not a failure —
        # the model is called — but it is counted, and alarmed above a threshold: a replay that
        # does not find its own choices does not rebuild the state believed to be resumed.
        if gel_actif():
            _rejoue = rejeu_decisions.chercher(
                context.person.person_id, context.activity_id, float(context.timestamp)
            )
            if _rejoue is not None:
                _plan = next(
                    (o for o in options if o.get_code() == _rejoue.get("code_plan")), None
                )
                if _plan is not None:
                    # No memory write here: the replay must relearn nothing, and the ticket 075
                    # freeze already forbids it on its side.
                    return (
                        options.index(_plan),
                        _rejoue.get("raison", ""),
                        f"rejeu:{_rejoue.get('fournisseur', '')}",
                        _rejoue.get("distribution") or {},
                    )

        # Cache miss on the "empty memory" branch: the payload is still to be built.
        if payload is None:
            payload = await self.build_travel_plan_payload(
                context, shuffled_options, destination, departure_time, anticipation,
                lignes=_lignes,
            )
        if _lignes:
            _evenements.noter_rendu(
                context.person.person_id,
                int(departure_time or context.timestamp),
                _lignes,
                payload["agents"][0].get("history", []),
            )

        _pl = PipelineLogger.get()
        _rec = _pl.get_record(context.person.person_id) if _pl is not None else None

        try:
            if _rec is not None:
                _rec.T_llm_start = time.time()
            # Ticket 084 — the routing restriction goes WITH the request: the gateway then
            # refuses to serve from another instance, at selection and hence before a call is
            # paid. Ticket 092: it is no longer set HERE but on the client, the single gate
            # through which the STM and LTM reflections ALSO go out — setting them call by
            # call had let memory be written by another model. The `allowed_providers`
            # filter below remains a defence in depth: it notes afterwards what the
            # restriction prevents.

            attente_instance = None
            if force_provider:
                payload["force_provider"] = force_provider
                # The pinned instance carries its wait: a local model only serves one call at
                # a time, the following tasks wait in the queue and the client's default
                # (tuned for a remote provider) would make them expire before their turn.
                _cfg = settings.llm.providers.get(force_provider)
                attente_instance = getattr(_cfg, "wait_timeout", None) if _cfg else None
            llm_result = await self.llm_client.execute(
                payload, wait_timeout=attente_instance
            )
            _t_after_llm = time.time()
            provider_used = llm_result.provider_used or ""
            if trace is not None:
                trace["payload"] = payload
                trace["fournisseur"] = provider_used
                trace["identifiant_lot"] = llm_result.task_id
                trace["souvenirs"] = list(payload["agents"][0].get("history", []) or [])
                trace["presentees_modes"] = [
                    t.get("mode") for t in payload["agents"][0]["trajectories"]
                ]
                trace["reponse_brute"] = (
                    json.dumps(
                        [a.model_dump() for a in llm_result.agents],
                        ensure_ascii=False,
                        default=str,
                    )
                    if llm_result.agents
                    else None
                )
            if (
                allowed_providers is not None
                and llm_result.ok
                and provider_used not in allowed_providers
            ):
                # Silent substitution by the gateway (switch to another model after a parse
                # error): refused, never archived as a decision (Q3).
                if trace is not None:
                    trace["substitution_refusee"] = provider_used
                logger.warning(
                    f"[035] Response served by {provider_used!r}, outside the allowed instances "
                    f"{sorted(allowed_providers)} — substitution refused for {context.person.person_id}"
                )
                return -1, f"substitution_refusee:{provider_used}", provider_used, {}

            if _rec is not None:
                _post_ms = (llm_result.timing.post_ms if llm_result.timing else 0) or 0
                _rec.T_llm_sent = _rec.T_llm_start + _post_ms / 1000
                _rec.T_llm_result = _t_after_llm
                _timing_p5 = llm_result.timing.timing_p5 if llm_result.timing else None
                if _timing_p5 and _pl is not None:
                    _pl.apply_timing_p5(context.person.person_id, _timing_p5)

            if llm_result.ok:
                agent_result = llm_result.agents[0]
                # The LLM scores all options (sum = 100); the actual mode is drawn at
                # random from this distribution. The weights are re-aligned on
                # `sorted_options` (deterministic order by code) so that the draw does
                # not depend on the anti-position-bias shuffle applied to the prompt.
                weights = None
                weights_are_fallback = False
                reasons = None
                if agent_result.probabilities:
                    # Modes as they were sent in the prompt: they make it possible to
                    # realign a response whose indices are out of bounds (the model has
                    # renumbered the options) instead of losing its mass.
                    sent_modes = [
                        t.get("mode") for t in payload["agents"][0]["trajectories"]
                    ]
                    shuffled_weights = normalize_option_probabilities(
                        agent_result.probabilities,
                        len(shuffled_options),
                        modes=sent_modes,
                        context=f"agent={context.person.person_id} activity={context.activity_id}",
                    )
                    # Uniform fallback (unusable LLM vector): the draw remains valid for
                    # THIS trip, but the distribution is not a decision of the model — it
                    # must never reach the persistent cache.
                    weights_are_fallback = isinstance(shuffled_weights, UniformFallback)
                    if trace is not None:
                        trace["poids_presentes"] = [float(w) for w in shuffled_weights]
                        trace["repli_uniforme"] = weights_are_fallback
                    position_in_sorted = {
                        id(opt): i for i, opt in enumerate(sorted_options)
                    }
                    weights = [0.0] * len(sorted_options)
                    for opt, w in zip(shuffled_options, shuffled_weights):
                        weights[position_in_sorted[id(opt)]] += w
                    index = draw_index(
                        weights,
                        *seed_parts,
                        min_prob_threshold=settings.agent.mode_choice_truncation_threshold,
                    )
                    decision_list = sorted_options

                    # Justification PER OPTION (2026-08-26): `normalize_option_probabilities`
                    # only returns the weights, the `reason` of each entry would be lost there.
                    # Located by the index sent (source of truth on the prompt side), then
                    # carried over to `sorted_options` like the weights, to recover the
                    # justification of the option actually drawn.
                    reasons = [None] * len(sorted_options)
                    for entry in agent_result.probabilities:
                        entry_reason = getattr(entry, "reason", None)
                        if not entry_reason:
                            continue
                        try:
                            entry_idx = int(entry.index)
                        except (TypeError, ValueError):
                            continue
                        if 0 <= entry_idx < len(shuffled_options):
                            opt = shuffled_options[entry_idx]
                            reasons[position_in_sorted[id(opt)]] = entry_reason
                else:
                    # Old-format response (one chosen index) — fallback without a draw.
                    index = agent_result.chosen_index
                    decision_list = shuffled_options

                if isinstance(index, int) and 0 <= index < len(decision_list):
                    reason = (
                        (reasons[index] if reasons is not None else None)
                        or agent_result.reason
                        or "Pas de justification fournie."
                    )

                    # Normalisation of the reason (alignment with aplan_trip_old)
                    if "is chosen because it" in reason:
                        reason = f"This plan {reason.split('is chosen because it', 1)[1].strip()}"

                    chosen_plan = decision_list[index]

                    distribution = {}
                    if weights is not None:
                        modes = [opt.mode_label() for opt in sorted_options]
                        distribution = mode_distribution(weights, modes)
                        reason = (
                            f"{reason} [Répartition estimée : "
                            f"{_format_distribution(distribution)} — mode tiré au sort.]"
                        )

                    # Writing the decision into short-term memory to feed the daily reflection
                    plan_summary = env_ob_to_text(
                        "travel_plan", chosen_plan.model_dump()
                    )
                    stm_msg = f"[ TRAVEL_PLAN ] Plan to head <{destination}> chosen by gateway LLM.\n{plan_summary}\nReasoning: {reason}"
                    self.add_short_term_memory(
                        context,
                        stm_msg,
                        timestamp=context.timestamp,
                        importance=_gravite_de_la_contrainte(context),
                        axes=_axes_de_la_decision(
                            context, chosen_plan, destination, weather
                        ),
                    )

                    original_index = options.index(chosen_plan)
                    if _rec is not None:
                        _rec.T_extract_end = time.time()

                    # --- Asynchronous insertion into the cache (fire-and-forget) ---
                    # It is the distribution that is cached, not the decision: on the next
                    # hit, a new draw will take place on these same probabilities.
                    # Never for a uniform fallback: the cache has no degraded mode, a
                    # persisted fallback would serve randomness to the following runs.
                    if self.llm_cache is not None and weights_are_fallback:
                        logger.info(
                            f"[cache] store refused — uniform fallback distribution not persisted | "
                            f"agent={context.person.person_id} activity={context.activity_id}"
                        )
                    elif self.llm_cache is not None and _lignes:
                        # Ticket 111 — a decision taken WITH a service line does not go into
                        # the cache: the cache is shared by the runs of the same population,
                        # and the control arm could be served again a decision taken while
                        # reading the article.
                        logger.info(
                            f"[cache] store refused — decision carrying a service line "
                            f"(ticket 111) | agent={context.person.person_id} "
                            f"activity={context.activity_id}"
                        )
                    elif self.llm_cache is not None:
                        mode = chosen_plan.mode_label()
                        _cache_task = create_background_task(
                            self.llm_cache.store(
                                agent_id=context.person.person_id,
                                activity_id=context.activity_id,
                                timestamp=context.timestamp,
                                options=sorted_options,
                                memory_text=memory_text,
                                chosen_plan_code=chosen_plan.get_code(),
                                mode=mode,
                                weather=weather,
                                probabilities=weights,
                                extra_key=anticipation_key,
                                traits_key=_traits_signature(
                                    context.person.identity.traits_json
                                ),
                            )
                        )
                        _cache_task.add_done_callback(
                            lambda t: (
                                logger.warning(f"Cache store failed: {t.exception()}")
                                if not t.cancelled() and t.exception()
                                else None
                            )
                        )
                        logger.debug(
                            f"Cache store task created for person {context.person.person_id}, activity {context.activity_id}, chosen plan mode: {mode}"
                        )

                    if trace is not None:
                        trace["distribution"] = distribution
                        trace["index_presente"] = shuffled_options.index(chosen_plan)
                        trace["raison"] = reason
                    # Ticket 090 — record the decision so that a hot resume brings it back
                    # without paying for it again. Nothing is traced during the freeze: a
                    # replayed day is not traced twice.
                    if not gel_actif():
                        rejeu_decisions.tracer(
                            context.person.person_id,
                            context.activity_id,
                            float(context.timestamp),
                            code_plan=chosen_plan.get_code(),
                            raison=reason,
                            fournisseur=provider_used,
                            distribution=distribution,
                            decide_a=_decide_a,
                        )
                    # ── Ticket 095, lot E — the decision, with the MODEL that produced it ──
                    # `llm_exchanges.jsonl` already carried the provider; the application
                    # log did not. A decision read in `app.log` did not say which model
                    # had taken it, and two files had to be cross-checked to establish it.
                    # Reconstructing a run afterwards depends on it, and it costs
                    # nothing.
                    logger.info(
                        f"[decision] agent={context.person.person_id} "
                        f"activite={context.activity_id} "
                        f"mode={chosen_plan.mode_label()} "
                        f"raison={reason} "
                        f"origine={'cache' if (trace or {}).get('cache') else 'direct'} "
                        f"modele={origine_modele(provider_used)}"
                    )
                    # Returns the index in the original (unshuffled) list for consistency with the caller
                    return original_index, reason, provider_used, distribution

                if _rec is not None:
                    _rec.T_extract_end = time.time()

            error_msg = llm_result.error or "Format de réponse invalide ou timeout."
            logger.warning(
                f"aplan_trip: gateway returned an invalid result for {context.person.person_id}: {error_msg}"
            )
            if trace is not None:
                trace["erreur"] = error_msg
                # Nature of the failure as the gateway qualified it. The text alone is not
                # enough: "Providers saturés ou indisponibles" describes a queue as well as
                # a quota dead for the day, and the caller must wait in one case, and be
                # patient for a few seconds in the other.
                if llm_result.error_kind:
                    trace["genre_erreur"] = llm_result.error_kind
                if llm_result.resume_at:
                    trace["reprise_a"] = llm_result.resume_at
            return -1, error_msg, provider_used, {}

        except Exception as e:
            logger.exception(f"Error while calling the LLM Gateway API: {e}")
            if trace is not None:
                trace["erreur"] = str(e)
            return -1, str(e), "", {}

    async def trigger_short_term_reflection_for_all_people(
        self,
        timestamp: int,
        people: list[Person],
        motif: str = "",
        declencheur: str = "",
    ):
        """
        Reflect on all short-term memories of all people at the given timestamp.
        This is used to process all short-term memories at once, e.g. at the end of the day.

        `motif` and `declencheur` (ticket 075): what made the agent eligible — `seuil`,
        `plancher journalier` or `rupture` — is computed by the CONTROLLER, the only one to
        know the buffer state at the time of the test. Without these two fields, the memory
        log notes a consolidation without being able to say what caused it, that is, the
        essence of what one comes to read there. Empty, nothing changes: the consolidation
        takes place identically.
        """
        if settings.agent.long_term_memory_enabled is False:
            logger.info("Long-term memory is disabled, skipping reflection.")
            return

        import asyncio

        sem = asyncio.Semaphore(10)

        async def _reflect_one(person):
            async with sem:
                context = Context(
                    person=person,
                    timestamp=timestamp,
                    data={
                        "type": "reflection",
                        "motif": motif,
                        "declencheur": declencheur,
                    },
                )
                await self.reflect_on_short_term_memory(context)

        await asyncio.gather(*[_reflect_one(p) for p in people])

    async def trigger_long_term_reflection_for_all_people(
        self, timestamp: int, from_date: datetime, people: list[Person]
    ):
        if (
            settings.agent.long_term_memory_enabled is False
            or settings.agent.long_term_self_reflect_enabled is False
        ):
            logger.info(
                "Long-term memory is disabled or Self reflection is disable, skipping self reflection."
            )
            return
        if gel_actif():
            # Replay of a resume (ticket 075): this self-reflection has already taken place and
            # its entry is already in memory. Redoing it would write it twice.
            return

        for person in people:
            context = Context(
                person=person, timestamp=timestamp, data={"type": "self_reflection"}
            )
            await self.reflect_on_long_term_memory(context, from_date)

    async def reflect_on_long_term_memory(self, context: Context, from_date: datetime):
        if settings.agent.long_term_memory_enabled is False:
            logger.info("Long-term memory is disabled, skipping reflection.")
            return

        all_entries = self.long_term_memory.get_last_user_memories(
            person_id=context.person.person_id,
            from_date=from_date,
        )
        if not all_entries:
            logger.info(
                f"No long-term memory available for reflection for {context.person.person_id}"
            )
            return

        # `gama_timestamp` and not `.timestamp()`: the memory's `datetime` carries WALL-CLOCK
        # fields, and `.timestamp()` would reread them in the process time zone —
        # `humanize_date` would then translate again, and the two conventions would add up
        # to a one-hour shift in the text read by the model.
        entries_text = "\n".join(
            f"- Time {humanize_date(gama_timestamp(entry.timestamp))}: {entry.content}"
            for entry in all_entries
        )
        identity_description = self.get_person_identity_description(context.person)

        # Exact memoisation (ticket 012) — same principle as the STM reflection.
        memo_key = None
        reflection: str | None = None
        if self.reflection_memo is not None:
            memo_key = ReflectionMemoStore.make_key(
                person_id=context.person.person_id,
                category="ltm_self_reflection",
                identity=identity_description,
                context_text=entries_text,
                departure_timestamp=float(context.timestamp),
                llm_params=settings.agent.llm_params,
            )
            hit = await asyncio.to_thread(
                self.reflection_memo.lookup, memo_key, "ltm_self_reflection"
            )
            if hit is not None:
                reflection = hit["reflection"]
                logger.info(
                    f"[reflection-memo] hit LTM — self-reflection served without an LLM call | "
                    f"person={context.person.person_id} (paid by {hit['provider'] or '?'})"
                )

        if reflection is None:
            payload = {
                "category": "ltm_self_reflection",
                "instances_admises": instances_pour("ltm_self_reflection"),
                "agents": [
                    {
                        "agent_id": context.person.person_id,
                        "perception": identity_description,
                        "context": entries_text,
                        "departure_timestamp": float(context.timestamp),
                    }
                ],
                "parameters": {**settings.agent.llm_params},
            }

            llm_result = await self.llm_client.execute(payload)
            results = llm_result.agents
            if not results:
                logger.error(
                    f"LTM self-reflection gateway returned no result for {context.person.person_id}"
                )
                raise ConsolidationMemoryUnavailable(
                    f"LTM without a result for {context.person.person_id}",
                    genre=getattr(llm_result, "error_kind", None),
                )
            # AgentResponse accepts fields outside the schema (extra=allow) —
            # "reflection" is carried by the ltm_self_reflection category.
            reflection = getattr(results[0], "reflection", "") or ""
            # Ticket 095, lot E — rare (13 per run) and high-stakes: it is the one that rereads
            # the whole long-term memory.
            logger.info(
                f"[auto-reflexion] agent={context.person.person_id} "
                f"modele={origine_modele(llm_result.provider_used)}"
            )
            if self.reflection_memo is not None:
                await asyncio.to_thread(
                    self.reflection_memo.store,
                    memo_key,
                    context.person.person_id,
                    "ltm_self_reflection",
                    reflection,
                    None,
                    llm_result.provider_used or "",
                )

        try:
            entry = MemoryEntry(
                person_id=context.person.person_id,
                content=reflection,
                timestamp=wall_clock(context.timestamp),
                memory_type=MemoryType.REFLECTION,
            )
            # Ticket 075 — the long-term self-reflection is a consolidation of another kind: it
            # consumes no short-term memory entry, it rereads the long-term memory. It has its
            # own section for this very reason, otherwise its trace would be confused with an
            # ordinary write.
            _journal = journal()
            if _journal is not None:
                _journal.ouvrir_agent(
                    context.person.person_id, _traits_de(context.person)
                )
                _journal.consolidation_debut(
                    context.person.person_id,
                    wall_clock(context.timestamp),
                    "auto-réflexion long terme",
                    "relecture périodique de la mémoire longue "
                    f"(tous les {settings.agent.long_term_self_reflect_interval_days} j simulés, "
                    f"fenêtre de {settings.agent.long_term_self_reflect_window_days} j)",
                    [],
                )
                _journal.reflexion(context.person.person_id, reflection)
            await self.aadd_long_term_memory(context, entry)
        except Exception as e:
            logger.error(
                f"Failed to store LTM self-reflection for person {context.person.person_id}, err: {e}"
            )

    def _marquer_concept_modifie(self, person_id: str) -> None:
        """A concept updated IN PLACE must be persisted like a new write.

        The `confirmer`, `preciser` and `contredire` operations add no entry: they modify an
        object already in live memory. Without this marking, the counter would rise in RAM
        and fall back to zero on reload — the consolidation would be perfectly invisible.
        """
        if self.long_term_memory is None:
            return
        self.long_term_memory._dirty.add(person_id)
        self.long_term_memory._schedule_flush()

    _JOURS_SANS_CONTRADICTION_ALARME = 2

    def _compter_operations(self, operations: dict, jour_simule) -> None:
        """Operation counter per cycle, and the alarm for the model that confirms everything.

        A model that never contradicts lets the agent act on beliefs that have stopped being
        true, and the hysteresis the experiment seeks to observe becomes uninterpretable: the
        recovery is won through the concepts' confidence, not through forgetting.
        """
        if not operations:
            return
        if not hasattr(self, "_jours_sans_contradiction"):
            self._jours_sans_contradiction: set = set()
            self._alarme_confirme_tout = False

        logger.info(
            "[concepts] opérations du cycle — "
            + ", ".join(f"{k} {v}" for k, v in sorted(operations.items()))
        )

        if operations.get(CONTREDIRE):
            self._jours_sans_contradiction.clear()
            if self._alarme_confirme_tout:
                self._alarme_confirme_tout = False
                logger.info("[concepts] the model contradicts its beliefs again")
            return

        # Concepts were produced, none contradicts: the day counts.
        if operations.get(CONFIRMER) or operations.get(CREER):
            self._jours_sans_contradiction.add(jour_simule)

        if (
            len(self._jours_sans_contradiction) > self._JOURS_SANS_CONTRADICTION_ALARME
            and not self._alarme_confirme_tout
        ):
            self._alarme_confirme_tout = True
            logger.error(
                f"[ALARME] no concept contradiction over "
                f"{len(self._jours_sans_contradiction)} simulated days, while concepts "
                f"are produced — the model confirms everything. The agents' beliefs are no "
                f"longer revised, and the recovery measured by the hysteresis experiment "
                f"becomes uninterpretable."
            )

    # Observation window of the rate of shown beliefs, in number of reflection calls.
    _FENETRE_CROYANCES = 50
    # Beyond it, the concept correction mechanism is out of service IN FACT: the model
    # can neither confirm nor refine what it is not shown. Measured at 68% on the ticket
    # 075 run, and at 100% for the agent without a car — hence 0 refinements out of 231.
    _SEUIL_CROYANCES_VIDES = 0.5

    def _compter_croyances_montrees(self, person_id: str, montrees: int) -> None:
        """Rate of reflection calls where the agent was shown NO belief.

        Ticket 077, lot E4. This is the denominator missing from the 075 run: without it, the
        redundancy of concepts is blamed on the model, whereas the measurement clears it —
        when `known_beliefs` is not empty, it confirms 74 times out of 76. This counter is
        the GUARD of lot A: if the baskets empty again, for this reason or another, the
        alarm says so before a thirty-day run has to be redone.
        """
        if not hasattr(self, "_croyances_fenetre"):
            self._croyances_fenetre: deque = deque(maxlen=self._FENETRE_CROYANCES)
            self._alarme_croyances_vides = False
        self._croyances_fenetre.append(1 if montrees == 0 else 0)
        if len(self._croyances_fenetre) < self._FENETRE_CROYANCES:
            return
        part_vide = sum(self._croyances_fenetre) / len(self._croyances_fenetre)
        if part_vide >= self._SEUIL_CROYANCES_VIDES and not self._alarme_croyances_vides:
            self._alarme_croyances_vides = True
            logger.error(
                f"[ALARME] no belief shown to the model in {part_vide:.0%} of the "
                f"{len(self._croyances_fenetre)} last reflections (threshold: "
                f"{self._SEUIL_CROYANCES_VIDES:.0%}) — the (mode, purpose) basket names "
                f"no candidate. The model can neither confirm nor refine: all it can "
                f"do is create, and concepts pile up as rewordings. First check "
                f"that the concepts' object axes are not empty."
            )
        elif part_vide < self._SEUIL_CROYANCES_VIDES and self._alarme_croyances_vides:
            self._alarme_croyances_vides = False
            logger.info(
                f"[concepts] beliefs shown to the model again "
                f"({1 - part_vide:.0%} of calls) — alarm re-armed"
            )

    async def reflect_on_short_term_memory(self, context: Context):
        mem = self.get_short_term_memory(context.person.person_id)
        group_messages, all_messages = mem.get_all_message_and_group()

        if not all_messages:
            logger.info("No short-term memory available for reflection.")
            return

        exp = []
        for group in group_messages:
            if group:
                activity = (
                    PersonScheduler(context.person).get_activity(group[0].activity_id)
                    if group[0].activity_id
                    else None
                )
                exp.append(
                    {
                        "purpose": activity.purpose if activity else None,
                        "observations": [msg.content for msg in group],
                    }
                )
        # Ticket 071, lot 3 — the concepts the agent already holds on the modes and purposes
        # of its day are shown in the call that ALREADY takes place. The model names the one
        # it updates, or names none: zero marginal cost, no arbitrary similarity threshold to
        # calibrate, and it has the context a threshold lacks.
        _paniers = {panier_de(m.axe_objet, m.axe_motif) for m in all_messages}
        _connus = _concepts_du_jour(
            self.long_term_memory, context.person.person_id, _paniers
        )
        # Short handles (K1, K2…) rather than document identifiers: the model has less
        # latitude to invent one, and the match is checked on return.
        _par_poignee = {f"K{i + 1}": e for i, e in enumerate(_connus)}
        self._compter_croyances_montrees(context.person.person_id, len(_par_poignee))
        # ── Ticket 100, lot 4 — what the household told this evening ────────────────────
        # The block is an INPUT of the call, just like the day's experiences. Nothing is
        # written in the receiver's memory: the model decides whether it draws a belief from
        # it, and a sentence heard that resonates with nothing disappears with the call.
        # That is the right default — memory does not grow from having listened.
        #
        # Empty when the flag is off, when the agent lives alone, or when nobody has produced
        # anything new since its last consolidation.
        _bloc_foyer = ""
        try:
            _bloc_foyer = foyer.bloc_du_soir(
                self.long_term_memory, context.person, wall_clock(context.timestamp)
            )
        except Exception as err:  # noqa: BLE001 — the household never brings down a reflection
            logger.error(
                f"[ALARME] [foyer] evening block impossible for "
                f"{context.person.person_id} ({err}) — the consolidation continues WITHOUT it, "
                f"hence without what the household had to say this evening."
            )

        _contexte_reflexion = {
            "today": exp,
            "known_beliefs": [
                {
                    "id": poignee,
                    "belief": json.loads(e.content)[0]
                    if e.content.startswith("[")
                    else e.content,
                    "times_observed": e.observations,
                    "times_contradicted": e.contre_exemples,
                }
                for poignee, e in _par_poignee.items()
            ],
        }
        if _bloc_foyer:
            _contexte_reflexion["household"] = _bloc_foyer
        # The press article is a daily service distinct from a shock. During its five travel
        # days, it is presented at EACH reflection, without consulting the memory's severity
        # threshold. The threshold still governs shocks and the usual recalls.
        _lignes_presse = await evenements_module.lignes_du_jour(
            str(context.person.person_id), int(context.timestamp), compter=False
        )
        if _lignes_presse:
            _contexte_reflexion["press_service"] = _lignes_presse
            evenements_module.noter_reflexion_presse(
                str(context.person.person_id),
                int(context.timestamp),
                _lignes_presse,
                etape="presentee",
            )
        experiences_text = json.dumps(
            _contexte_reflexion, indent=2, ensure_ascii=False
        )

        identity_description = self.get_person_identity_description(context.person)
        custom_guidelines = (
            f"\n**IMPORTANT CUSTOM GUIDELINES** {settings.agent.reflection_custom_guidelines}"
            if settings.agent.reflection_custom_guidelines
            else ""
        )

        # Exact memoisation (ticket 012): same agent, same experience, same guidelines
        # ⇒ same introspection. Hit ⇒ LLM call avoided; the effects (STM consumption,
        # LTM writes) remain strictly identical to a real call.
        memo_key = None
        reflection: str | None = None
        concepts: list = []
        if self.reflection_memo is not None:
            memo_key = ReflectionMemoStore.make_key(
                person_id=context.person.person_id,
                category="stm_reflection",
                identity=identity_description,
                context_text=experiences_text,
                guidelines=custom_guidelines,
                departure_timestamp=float(context.timestamp),
                llm_params=settings.agent.llm_params,
                # The output schema changed in lot 1 of ticket 071: each concept carries a
                # severity level. A response memoised under the old schema has none, and
                # serving it would produce zero-severity concepts WITHOUT ERROR — yet zero is
                # the severity of a perfect trip. The version therefore makes the cache MISS
                # rather than serve wrongly. The cache accumulated under the old schema is
                # lost: that is the price of the field, and it is announced.
                schema_version=SCHEMA_REFLEXION_VERSION,
            )
            hit = await asyncio.to_thread(
                self.reflection_memo.lookup, memo_key, "stm_reflection"
            )
            if hit is not None:
                reflection, concepts = hit["reflection"], hit["concepts"]
                logger.info(
                    f"[reflection-memo] hit STM — reflection served without an LLM call | "
                    f"person={context.person.person_id} (paid by {hit['provider'] or '?'})"
                )

        if reflection is None:
            payload = {
                "category": "stm_reflection",
                "instances_admises": instances_pour("stm_reflection"),
                "min_tpm_required": settings.agent.stm_reflection_min_tpm,
                "agents": [
                    {
                        "agent_id": context.person.person_id,
                        "perception": identity_description,
                        "context": experiences_text,
                        "departure_timestamp": float(context.timestamp),
                    }
                ],
                "parameters": {
                    "custom_guidelines": custom_guidelines,
                    **settings.agent.llm_params,
                },
            }

            results = []
            max_tentatives = 3
            for tentative in range(max_tentatives):
                llm_result = await self.llm_client.execute(payload)
                results = llm_result.agents
                if results:
                    break
                if tentative + 1 < max_tentatives:
                    logger.warning(
                        f"STM reflection gateway returned no result for {context.person.person_id} "
                        f"(attempt {tentative + 1}/{max_tentatives}) — retrying in 1s..."
                    )
                    await asyncio.sleep(1.0)

            if not results:
                logger.error(
                    f"STM reflection gateway returned no result for {context.person.person_id} "
                    f"after {max_tentatives} attempts"
                )
                # STM entries are only removed after this block. The exception therefore
                # guarantees that a resume can retry exactly the missing consolidation.
                raise ConsolidationMemoryUnavailable(
                    f"STM without a result for {context.person.person_id}",
                    genre=getattr(llm_result, "error_kind", None),
                )

            agent_result = results[0]
            # AgentResponse accepts fields outside the schema (extra=allow) —
            # "reflection"/"concepts" are carried by the stm_reflection category.
            reflection = (getattr(agent_result, "reflection", "") or "").strip()
            concepts = getattr(agent_result, "concepts", []) or []
            presse_prise_en_compte = bool(
                getattr(agent_result, "press_service_considered", False)
            )
            if _lignes_presse and not presse_prise_en_compte:
                evenements_module.noter_reflexion_presse(
                    str(context.person.person_id),
                    int(context.timestamp),
                    _lignes_presse,
                    etape="invalide",
                    prise_en_compte=False,
                    reflection=reflection,
                )
                raise ConsolidationMemoryUnavailable(
                    f"STM reflection of {context.person.person_id} without explicit "
                    f"consideration of the press service"
                )
            if _lignes_presse:
                evenements_module.noter_reflexion_presse(
                    str(context.person.person_id),
                    int(context.timestamp),
                    _lignes_presse,
                    etape="validee",
                    prise_en_compte=True,
                    reflection=reflection,
                )
            # Ticket 095, lot E — changing the reflections' model changes the CONTENT of
            # memory, hence the decisions. Knowing which one wrote what is not an
            # infrastructure detail.
            logger.info(
                f"[reflexion-stm] agent={context.person.person_id} "
                f"concepts={len(concepts)} "
                f"modele={origine_modele(llm_result.provider_used)}"
            )

            if self.reflection_memo is not None:
                # The store refuses emptiness (D3): a generation failure is not replayed.
                await asyncio.to_thread(
                    self.reflection_memo.store,
                    memo_key,
                    context.person.person_id,
                    "stm_reflection",
                    reflection,
                    concepts,
                    llm_result.provider_used or "",
                )

        self.get_short_term_memory(context.person.person_id).remove_batch(all_messages)
        start_timestamp = all_messages[0].timestamp

        # ── Ticket 075 — opening of the log section ──────────────────────────────────────
        # The REASON is computed by the controller at eligibility time (entry threshold,
        # 10 pm floor, cumulative severity break): it comes down through `context.data`.
        # Without it, the log would say that a consolidation took place without being able to
        # say why — that is, the essence of what one comes to read there.
        _journal = journal()
        if _journal is not None:
            _motif = (context.data or {}).get("motif") or "non précisé"
            _journal.ouvrir_agent(
                context.person.person_id, _traits_de(context.person)
            )
            _journal.consolidation_debut(
                context.person.person_id,
                wall_clock(context.timestamp),
                _motif,
                (context.data or {}).get("declencheur")
                or "motif non transmis par le contrôleur",
                all_messages,
            )
            _journal.reflexion(context.person.person_id, reflection)

        # Ticket 071, lot 1 — severity FLOOR: the highest deterministic severity among the
        # entries this reflection consumes. It is a FACT measured by the simulation, which the
        # model's judgement cannot lower.
        plancher = max((float(m.importance or 0.0) for m in all_messages), default=0.0)

        # Axes of the consumed day (lot 2). They are taken from the MOST SEVERE entry, and
        # not from the last or the most frequent: it is the episode that marked the day that
        # characterises it. At equal severity, the last one wins, because it is the closest
        # to the moment the reflection is written.
        _pivot = max(
            all_messages,
            key=lambda m: (float(m.importance or 0.0), m.timestamp),
            default=None,
        )
        axes_du_groupe = {
            "axe_objet": getattr(_pivot, "axe_objet", None),
            "axe_lieu": getattr(_pivot, "axe_lieu", None),
            "axe_creneau": getattr(_pivot, "axe_creneau", None),
            "axe_motif": getattr(_pivot, "axe_motif", None),
            "axe_meteo": getattr(_pivot, "axe_meteo", None),
        }

        entries = []
        try:
            # The narrative reflection inherits the floor: it tells the day, and a day that
            # contains a shock is not an ordinary day. The schema does not ask for a level for
            # the reflection itself, only for the concepts.
            entries.append(
                MemoryEntry(
                    person_id=context.person.person_id,
                    content=reflection,
                    timestamp=start_timestamp,
                    memory_type=MemoryType.REFLECTION,
                    importance=plancher,
                    **axes_du_groupe,
                )
            )

            # The RANK only serves as a tie-break within the same level: one must therefore
            # know how many concepts share each level, and in which order they arrived. An
            # order is relative to the batch; a value must be comparable from one day and
            # one agent to another.
            normalises = [_normaliser_concept(c) for c in concepts]
            effectif: dict = {}
            for _lu in normalises:
                effectif[_lu.niveau] = effectif.get(_lu.niveau, 0) + 1
            rang_courant: dict = {}
            operations_vues: dict = {}

            for lu in normalises:
                cinq, niveau, valence, mode = lu.cinq, lu.niveau, lu.valence, lu.mode
                origine_concept = lu.origine
                rang_courant[niveau] = rang_courant.get(niveau, 0) + 1
                i_llm = gravite_jugee(
                    niveau, n_niveau=effectif[niveau], rang=rang_courant[niveau]
                )
                importance = gravite_concept(i_llm, plancher)

                # ── Ticket 071, lot 3 — the concept is CORRECTED instead of piling up ────────
                operation = lu.operation
                cible = _par_poignee.get(lu.cible) if lu.cible else None
                if operation != CREER and cible is None:
                    # An unknown target must neither lose the concept nor touch an existing one
                    # at random: we create, and we SAY so. A model that systematically named
                    # phantom targets would otherwise go unnoticed.
                    logger.warning(
                        f"[concepts] opération « {operation} » sur une cible inconnue "
                        f"« {lu.cible} » | person={context.person.person_id} — repli sur "
                        f"« {CREER} », aucun concept existant n'est modifié"
                    )
                    operation = CREER
                operations_vues[operation] = operations_vues.get(operation, 0) + 1

                if operation in (CONFIRMER, PRECISER):
                    # Nothing new is written: the existing concept is updated in place.
                    # Ticket 075 — IN-PLACE update: without the before, no trace would ever
                    # say what this operation changed.
                    _avant = (cible.content, cible.observations, cible.confiance)
                    if operation == CONFIRMER:
                        cible.observations = int(cible.observations or 0) + 1
                    else:
                        # `preciser`: the content is replaced, counters and history
                        # KEPT — it is the same belief, stated more finely.
                        cible.content = json.dumps(cinq, ensure_ascii=False)
                        cible.tags = ",".join(cinq[1:])
                    cible.derniere_observation = start_timestamp
                    cible.force = force_apres_rappel(cible.force)
                    cible.importance = max(float(cible.importance or 0.0), importance)
                    self._marquer_concept_modifie(context.person.person_id)
                    tracer_operation(
                        context.person.person_id,
                        "confirmé" if operation == CONFIRMER else "précisé",
                        start_timestamp,
                        doc_id=str(getattr(cible, "doc_id", "") or ""),
                    )
                    if _journal is not None:
                        _journal.operation_concept(
                            context.person.person_id,
                            "confirmé" if operation == CONFIRMER else "précisé",
                            avant=str(_avant[0]),
                            apres=str(cible.content)
                            if operation == PRECISER
                            else "(inchangé)",
                            observations=f"{_avant[1]} → {cible.observations}",
                            confiance=f"{_avant[2]:.2f} → {cible.confiance:.2f}",
                        )
                    continue

                if operation == CONTREDIRE:
                    _contre_avant = int(cible.contre_exemples or 0)
                    _confiance_avant = cible.confiance
                    cible.contre_exemples = int(cible.contre_exemples or 0) + 1
                    cible.derniere_observation = start_timestamp
                    if not cible.est_servi and not cible.depasse_le:
                        # Marked and DATED, never deleted: this setting aside is the readable
                        # trace of the habit change the experiment is looking for.
                        #
                        # ⚠ The date is set when the concept STOPS BEING SERVED, and not when
                        # it becomes "outdated" in the sense of the three counter-examples.
                        # Defect found while writing the lot 3 tests: a concept leaves the
                        # basket as soon as it is no longer served, hence it is no longer
                        # shown to the model, hence it can no longer be contradicted. With
                        # few observations, the threshold of three contradictions is
                        # REACHABLE ONLY if the removal from service comes after — otherwise
                        # the concept stayed at zero service, never dated, and the observable
                        # the hysteresis experiment seeks was never written.
                        #
                        # The specification in fact ties the dated trace to the end of
                        # service — "below a threshold, the concept stops being served
                        # without being deleted: its dated setting aside is the trace of the
                        # habit change" — and not to the threshold of three.
                        cible.depasse_le = start_timestamp.isoformat()
                        logger.info(
                            f"[concepts] concept SET ASIDE for "
                            f"{context.person.person_id}: {cible.contre_exemples} "
                            f"contradictions against {cible.observations} confirmations, "
                            f"confidence {cible.confiance:.2f} — it stops being served and "
                            f"stays KEPT in memory"
                            + (
                                "; outdated in the sense of the three counter-examples"
                                if cible.est_depasse
                                else ""
                            )
                        )
                    self._marquer_concept_modifie(context.person.person_id)
                    tracer_operation(
                        context.person.person_id,
                        "contredit",
                        start_timestamp,
                        doc_id=str(getattr(cible, "doc_id", "") or ""),
                    )
                    if _journal is not None:
                        _journal.operation_concept(
                            context.person.person_id,
                            "contredit",
                            avant=str(cible.content),
                            contre_exemples=f"{_contre_avant} → {cible.contre_exemples}",
                            confiance=f"{_confiance_avant:.2f} → {cible.confiance:.2f}",
                            note=(
                                f"**mis à l'écart** le {str(cible.depasse_le)[:16]} — il cesse "
                                f"d'être servi, il n'est PAS supprimé"
                                if cible.depasse_le and not cible.est_servi
                                else ""
                            ),
                        )
                    # and the concept that takes over is written below
                if i_llm is not None and importance > i_llm:
                    # The fact has taken over from the judgement: this is exactly what the
                    # safety rule must produce, and it must be countable over a run.
                    logger.info(
                        f"[gravite] judgement RAISED by the measured fact | "
                        f"person={context.person.person_id} niveau={niveau} "
                        f"I_llm={i_llm:.2f} → I={importance:.2f}"
                    )
                entries.append(
                    MemoryEntry(
                        person_id=context.person.person_id,
                        # The content remains the canonical 5-tuple: severity, valence and axes
                        # live on the entry's FIELDS, not in its text. No downstream reader
                        # changes, and a concept written before lot 1 is reread
                        # identically.
                        content=json.dumps(cinq, ensure_ascii=False),
                        timestamp=start_timestamp,
                        memory_type=MemoryType.CONCEPT,
                        tags=",".join(cinq[1:]),
                        importance=importance,
                        valence=valence,
                        # D2 — a single hop. A belief born from what the agent HEARD carries
                        # its provenance and will never go back into the household, even
                        # if later confirmed by a trip (Q2, settled on 2026-09-21).
                        origine=origine_concept,
                        # Axes normalised on write (lot 2). The mode comes from the model, which
                        # knows what its concept is about; the place from its spatial scope; the
                        # purpose from its object. The slot and the weather come from the consumed
                        # day: a concept has no time of its own, it has that of what produced it.
                        axe_objet=mode_canonique(mode),
                        axe_lieu=normaliser_lieu(cinq[2] or None),
                        axe_creneau=axes_du_groupe.get("axe_creneau"),
                        axe_motif=normaliser_motif(cinq[4] or None),
                        axe_meteo=axes_du_groupe.get("axe_meteo"),
                    )
                )
            self._compter_operations(
                operations_vues, wall_clock(context.timestamp).date().toordinal()
            )
        except Exception as e:
            logger.exception(f"Failed to parse STM reflection response: {e}")

        for entry in entries:
            if entry.memory_type == MemoryType.CONCEPT:
                # The trace is written HERE and not in the log: the log is off by default and
                # is read, it is not measured (its own header says so). Making the operations
                # curve depend on its activation would make the measurement dependent on a
                # convenience setting.
                tracer_operation(
                    context.person.person_id,
                    "créé",
                    start_timestamp,
                    doc_id=str(getattr(entry, "doc_id", "") or ""),
                )
            if _journal is not None and entry.memory_type == MemoryType.CONCEPT:
                _journal.operation_concept(
                    context.person.person_id,
                    "créé",
                    apres=str(entry.content),
                    note=f"gravité {float(entry.importance or 0.0):.2f}",
                )
            await self.aadd_long_term_memory(context, entry)

        # ── Ticket 106 — has the injected memory reached the long-term memory? ───────────
        # Here and nowhere else: the entries have just been WRITTEN, and it is their written
        # content that is queried, not an intention. Returns nothing — and costs nothing — on
        # ordinary consolidations, which consume no injected entry.
        #
        # The check ALARMS and does not stop, deliberately: the witness is heuristic where
        # that of ticket 105 was certain. Wiring the stop will come once its false alarm rate
        # has been observed on real runs.
        temoin.controler(
            person_id=str(context.person.person_id),
            sim_ts=int(context.timestamp),
            contenus_courts=[m.content for m in all_messages],
            textes_longs=[e.content for e in entries],
            seuil=settings.agent.temoin_souvenir_mots_min,
        )

        # Ticket 075 — the FULL state of memory after the consolidation, the only moment it is
        # written: it is the point of comparison from one consolidation to the next. It is
        # read from the agent's metadata, never rebuilt from the day's entries.
        if _journal is not None:
            _journal.consolidation_fin(
                context.person.person_id,
                wall_clock(context.timestamp),
                self.long_term_memory.user_metadata.get(
                    context.person.person_id, {}
                ).get("entries", []),
            )
