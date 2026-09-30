"""Severity of a memory — ticket 071, lot 1.

A memory is no longer merely dated, it is **qualified**. Its severity decides its
lifetime and, in lot 2, its weight at recall.

Two sources, in two places, never competing:

- the **raw entries** of short-term memory receive a DETERMINISTIC severity, computed from
  what the simulation measures — it costs nothing and it is objective;
- **concepts** receive a named level JUDGED by the model, requested inside the
  reflection that already takes place, hence with no extra call.

And a safety rule that arbitrates between them: the measured fact always prevails over judgement.

This module is PURE: it reads only the configuration, never the clock, never the disk, never
the network. This is what makes it testable without a simulator or a model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from loguru import logger

from settings import settings

# ── The five levels ──────────────────────────────────────────────────────────────
# Anchored by an OBSERVABLE CONSEQUENCE and not by a felt intensity. This is a
# refinement of Park et al. (2023), who ask for an integer from 1 to 10 anchoring only the
# two extremes. The argument is comparability: one must be able to weigh a shock on day 2
# against an ordinary trip on day 5, hence an ABSOLUTE value and not a rank.
# The textual anchors live in the `stm_reflection` template; here, their values.
NIVEAUX: dict[str, float] = {
    "anodin": 0.10,
    "notable": 0.30,
    "genant": 0.50,
    "grave": 0.75,
    "marquant": 1.00,
}

# The setup addresses the model in ENGLISH (ticket 074): the template therefore asks for the
# levels under their English label. The canonical vocabulary of the specification remains
# French — the language in which the scale was defined and discussed — and these aliases
# map one to the other. Both are accepted as input: a model answering in the
# other language is understood rather than rejected.
ALIAS_NIVEAUX: dict[str, str] = {
    "negligible": "anodin",
    "noticeable": "notable",
    "inconvenient": "genant",
    "serious": "grave",
    "memorable": "marquant",
}

# Textual anchors, as set in the reflection template. Each level
# is anchored by an OBSERVABLE CONSEQUENCE, and not by a felt intensity: this is what
# makes the value comparable from one agent and one day to another, where a rank is not.
ANCRES: dict[str, str] = {
    "anodin": "Everything went as I had planned.",
    "notable": "A noticeable deviation, with no consequence for the rest of my day.",
    "genant": "It cost me time, or forced me to shift a schedule.",
    "grave": "It made me miss something, or put me in difficulty.",
    "marquant": "I will remember this in a month; it changes how I travel.",
}

# The anchors, RE-KEYED in the English vocabulary of the output schemas. They are not
# copied: they are derived from `ANCRES` and `ALIAS_NIVEAUX`, which remain the only source.
#
# ⚠ WHY THIS FUNCTION EXISTS. `evenement_jugement` passed `ANCRES` as is in its
# payload, hence FRENCH keys — the prompt asked "choose among anodin,
# notable, genant, grave, marquant" while its output schema accepted only
# `negligible`…`memorable`. The model received two vocabularies for the same scale and
# had to guess one. Measured on Groq on 2026-09-22, fixed the same day.
def ancres_anglaises() -> dict[str, str]:
    """The five anchors under the label expected by the schemas (`negligible`…`memorable`)."""
    vers_anglais = {fr: en for en, fr in ALIAS_NIVEAUX.items()}
    return {vers_anglais[niveau]: texte for niveau, texte in ANCRES.items()}


# ── Weights of the deterministic severity ────────────────────────────────────────
POIDS_RETARD = 0.50
POIDS_CORRESPONDANCE_RATEE = 0.20
POIDS_INCIDENT_RESEAU = 0.20
POIDS_MODE_CONTRAINT = 0.10

# ⚠ Components WITHOUT A SOURCE today. `experiences/experience.py` declares an `incident`
# event format, but refuses to run it: GAMA does not play them yet. The chain is
# laid end to end — parameter, detail, counter, log — so that there is only one
# source to plug in when the day comes, and not a formula to reopen.
#
# ⚠⚠ A component without an observable contributes ZERO, and zero is exactly the value of a
# perfect trip. Without an explicit declaration, severity would be underestimated without any
# symptom showing. Hence `journal_des_composantes()`, called at startup.
# The other three components are NOT reweighted to compensate: reweighting would change
# the severity scale silently.
COMPOSANTES_INACTIVES: tuple[str, ...] = ("incident_reseau",)

# Values of the "chain constraint" that constitute a CONSTRAINED MODE CHANGE.
# `retour_force`: the agent returns with what it took, its options are restricted to the mode
# of the parked vehicle. `sortie_bloquee`: it cannot leave in the mode it would have chosen.
# `passager` is deliberately EXCLUDED: being the passenger of another household member is an
# arrangement, not a suffered degradation — counting it would degrade the measurement.
CONTRAINTES_MODE_FORCE: tuple[str, ...] = ("retour_force", "sortie_bloquee")


@dataclass(frozen=True)
class DetailGravite:
    """Contribution of each component, so that instrumentation says which one played."""

    retard: float = 0.0
    correspondance_ratee: float = 0.0
    incident_reseau: float = 0.0
    mode_contraint: float = 0.0

    def total(self) -> float:
        return self.retard + self.correspondance_ratee + self.incident_reseau + self.mode_contraint

    def composantes_actives(self) -> tuple[str, ...]:
        """Those that actually contributed on THIS entry."""
        return tuple(
            nom
            for nom in ("retard", "correspondance_ratee", "incident_reseau", "mode_contraint")
            if getattr(self, nom) > 0.0
        )


MODE_RETARD_ASYMPTOTE = "asymptote"
MODE_RETARD_PALIER = "palier"
_MODES_RETARD = (MODE_RETARD_ASYMPTOTE, MODE_RETARD_PALIER)

# Out-of-domain delay settings already reported. An alarm repeated at every memory stops being
# one; a silent alarm lets a fallback pass for an accepted setting.
_RETARD_FAUTIFS_DITS: set[str] = set()


def reinitialiser_alarmes_retard() -> None:
    """Forgets the setting alarms already raised. Reserved for tests."""
    _RETARD_FAUTIFS_DITS.clear()


def _alarme_retard(cle: str, message: str) -> None:
    if cle in _RETARD_FAUTIFS_DITS:
        return
    _RETARD_FAUTIFS_DITS.add(cle)
    logger.error(message)


def part_de_retard(retard_s: float) -> float:
    """The delay component of severity, in [0, max].

    TWO SHAPES, and it is a setting (`memoire__retard_saturation`).

    `palier` — the original shape: `0.50 × min(t / ref, 1)`. Clean, but it **erases everything
    beyond the reference**. Declaring 45, 60 or 90 minutes gave exactly the same
    severity, and a decreasing shock profile staying above 30 minutes existed only in
    the text — it was the first trap of any new shock declaration.

    `asymptote` — DEFAULT since 2026-09-21. **Below the reference, nothing changes**:
    the component remains `0.50 × t / ref`. Above it, the plateau is replaced by a rise that
    decelerates towards `max` without ever reaching it. Two different delays therefore give two
    different severities, whatever their duration.

    This split is not a convenience. A pure exponential going through the anchor point
    would be 1.75 times steeper at the origin: nine minutes of delay would go from 0.15 to 0.22,
    and ALL small incidents would become more severe — an effect nobody asked for,
    and which would make days cross the break threshold that did not cross it. The
    piecewise shape touches only what we want to correct.

    The time constant of the extension is set so that the SLOPE is continuous at the anchor
    point: without this the curve would have a kink at thirty minutes, and one more second
    would be worth a jump in severity.

    ⚠ Removing the plateau without changing the shape does not remove the wall, it MOVES it. An
    unbounded linear component would reach 1.0 from 60 minutes, total severity being bounded at
    1: 60, 90 and 120 minutes would become indistinguishable again, and the other three components
    would stop weighing anything at all. The asymptotic shape is what truly removes
    the limit.

    The anchor point is PRESERVED from one shape to the other: at `memoire__retard_ref_s`, both
    return 0.50. No shock in the catalogue therefore changes classification through the mere
    change of shape — they only separate from one another above the
    reference.
    """
    t = max(0.0, float(retard_s))
    ref = float(settings.agent.memoire__retard_ref_s)
    if ref <= 0:
        # A zero or negative `ref` would make the component undefined. We do not guess: it is
        # zero AND we say so, otherwise delay would silently stop counting.
        _alarme_retard(
            f"ref:{ref}",
            f"[ALARME] memoire__retard_ref_s = {ref} : la composante de retard de la "
            f"gravité est INACTIVE — toute gravité de retard vaudra zéro",
        )
        return 0.0

    mode = str(getattr(settings.agent, "memoire__retard_saturation", MODE_RETARD_ASYMPTOTE)).strip()
    if mode not in _MODES_RETARD:
        _alarme_retard(
            f"mode:{mode}",
            f"[ALARME] memoire__retard_saturation inconnu ({mode!r}) — repli sur "
            f"{MODE_RETARD_ASYMPTOTE!r}. Modes admis : {list(_MODES_RETARD)}.",
        )
        mode = MODE_RETARD_ASYMPTOTE

    if mode == MODE_RETARD_PALIER:
        return POIDS_RETARD * min(t / ref, 1.0)

    maxi = float(getattr(settings.agent, "memoire__retard_gravite_max", 0.70))
    if maxi <= POIDS_RETARD:
        # Without margin above the anchor point, there is nothing to raise: the time
        # constant of the extension would be zero.
        _alarme_retard(
            f"max:{maxi}",
            f"[ALARME] memoire__retard_gravite_max = {maxi} n'est pas strictement supérieur à "
            f"POIDS_RETARD = {POIDS_RETARD} : il ne reste aucune marge au-dessus du point "
            f"d'ancrage. Repli sur le mode {MODE_RETARD_PALIER!r}, qui efface tout au-delà de "
            f"{ref:.0f} s.",
        )
        return POIDS_RETARD * min(t / ref, 1.0)

    # BELOW the reference: strictly unchanged. This is what makes the change
    # safe — no shock declared under 30 minutes moves by a thousandth, and three quarters of the
    # catalogue are in that case.
    if t <= ref:
        return POIDS_RETARD * t / ref

    # ABOVE: the plateau is replaced by a rise that decelerates towards `maxi` without ever
    # reaching it. τ is set so that the SLOPE is continuous at the anchor point — otherwise the
    # curve would have a kink at 30 minutes, and one more second would be worth a jump in severity.
    tau = (maxi - POIDS_RETARD) * ref / POIDS_RETARD
    return maxi - (maxi - POIDS_RETARD) * math.exp(-(t - ref) / tau)


def gravite_deterministe(
    retard_s: float = 0.0,
    correspondance_ratee: bool = False,
    incident_reseau: bool = False,
    mode_contraint: bool = False,
) -> tuple[float, DetailGravite]:
    """Severity of a raw entry, from what the simulation measures. Bounded on [0, 1].

    A contribution specific to this setup: neither Park et al. nor Vu et al. have a simulator that
    measures the delay suffered, and Park et al. (2023, § 4.1) note that "there are many
    possible implementations of an importance score".

    An EARLY arrival is not a bonus: a negative delay is worth zero, never a negative
    severity that would offset a real incident in the same entry.
    """
    part_retard = part_de_retard(retard_s)

    detail = DetailGravite(
        retard=part_retard,
        correspondance_ratee=POIDS_CORRESPONDANCE_RATEE * bool(correspondance_ratee),
        incident_reseau=POIDS_INCIDENT_RESEAU * bool(incident_reseau),
        mode_contraint=POIDS_MODE_CONTRAINT * bool(mode_contraint),
    )
    return borne_0_1(detail.total()), detail


def borne_0_1(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def ecart_de_rang(n_niveau: int, rang: int) -> float:
    """Tie-break within the same level: +1 for the first, -1 for the last, 0 in the middle.

    The rank serves ONLY THAT. Asking the model to rank its concepts among themselves produces a
    reliable order, but relative to the batch: on an entirely ordinary day, the least ordinary
    would get the top mark. The order therefore spreads concepts by only ±0.05.

    ⚠ Fixed on 2026-09-14 (ticket 071, specs/ticket_071/tests_lot1.md § 9): the original
    formula, `2 × (n - rang) / max(n - 1, 1) - 1`, gave **-1** for a concept ALONE in its
    level, which it thus penalised by 0.05 — whereas it is the most common case and there is
    nothing to break when there is only one candidate.
    """
    if n_niveau <= 1:
        return 0.0
    rang = max(1, min(int(rang), int(n_niveau)))
    return 2.0 * (n_niveau - rang) / (n_niveau - 1) - 1.0


def gravite_jugee(niveau: str | None, n_niveau: int = 1, rang: int = 1) -> float | None:
    """Severity of a concept from the named level returned by the model, or `None`.

    `None` when the model answers off the grid or does not answer: the concept then falls back on
    the sole deterministic severity of its group (cf. `gravite_concept`). An off-grid answer must
    not cost the whole reflection, but it must leave a TRACE — this is what
    will make it visible, on a run, that a model does not respect the scale.

    ⚠ Bounded on [0, 1] after tie-break: without this bound, a `marquant` first of three
    was worth 1.05 and its lifetime went from 19.60 to 20.44 days.
    """
    if niveau is None or not str(niveau).strip():
        logger.warning(
            "[gravite] severity level MISSING from the model's answer — the concept falls back "
            "on the deterministic severity of its group"
        )
        return None
    clef = str(niveau).strip().lower()
    clef = ALIAS_NIVEAUX.get(clef, clef)
    if clef not in NIVEAUX:
        logger.warning(
            f"[gravite] UNKNOWN severity level « {niveau} » — outside the five levels "
            f"{sorted(ALIAS_NIVEAUX)}; the concept falls back on the deterministic severity "
            f"of its group"
        )
        return None
    return borne_0_1(NIVEAUX[clef] + 0.05 * ecart_de_rang(n_niveau, rang))


def gravite_concept(i_llm: float | None, i_det_groupe: float = 0.0) -> float:
    """Safety rule, NON-NEGOTIABLE: `I = max(I_llm, max(I_det of the consumed group))`.

    A model that underestimates a forty-five-minute incident cannot downgrade it: the
    measured fact always prevails over judgement.

    ⚠ The protection is ASYMMETRIC, and deliberately so: it only works downwards. A model
    that OVERESTIMATES is bounded by nothing other than the 1.0 ceiling. A symmetric ceiling
    is not specified and is not applied here.
    """
    plancher = borne_0_1(i_det_groupe)
    if i_llm is None:
        return plancher
    return max(borne_0_1(i_llm), plancher)


# ── Lifetime ─────────────────────────────────────────────────────────────────────


def force_initiale(importance: float) -> float:
    """Forgetting time constant, in days, fixed at write time according to severity.

    `force = min(S0 × (1 + k × I), FORCE_MAX)`. The `exp(-Δt / force)` shape and the strength
    that grows at recall come from MemoryBank (Zhong et al., 2024), which takes up Ebbinghaus's
    forgetting curve (1885). The modulation by severity does NOT come from ACT-R, whose base
    activation encodes only recency and frequency: it relies on emotional memory
    (McGaugh, 2004).

    The ceiling applies FROM WRITE TIME and not only at reinforcement: at the published
    sensitivity point (`S0 = 8.3` d, Park et al.), a `marquant` memory would otherwise start at
    58 days.
    """
    s0 = float(settings.agent.long_term_retrieval__force_base_jours)
    k = float(settings.agent.memoire__force_k_importance)
    plafond = float(settings.agent.memoire__force_max_jours)
    return min(s0 * (1.0 + k * borne_0_1(importance)), plafond)


def force_apres_rappel(force: float | None) -> float:
    """`force ← min(force + δ, FORCE_MAX)`. ADDITIVE, and it is a choice.

    A multiplicative factor (× 1.15) saturated at the ceiling in seventeen recalls everything
    recalled often. Each recall adds one day: an ordinary trip reaches the ceiling in
    twenty-eight recalls, a `marquant` memory in eleven. It is also closer to
    ACT-R's base learning, where frequency has diminishing returns.
    """
    delta = float(settings.agent.memoire__force_delta_rappel_jours)
    plafond = float(settings.agent.memoire__force_max_jours)
    depart = float(settings.agent.long_term_retrieval__force_base_jours) if force is None else float(force)
    return min(depart + delta, plafond)


def poids_temporel(delta_jours: float, force: float | None) -> float:
    """`exp(-Δt / force)`, where Δt is counted from the LAST RECALL and not from the write.

    As in Park et al. (2023, § 4.1), whose recency decays "since the memory was last
    retrieved", and in MemoryBank. A missing `force` — entry written before this lot — falls
    back on the default time constant rather than being worth zero.
    """
    f = float(settings.agent.long_term_retrieval__force_base_jours) if force is None else float(force)
    if f <= 0:
        return 0.0
    return math.exp(-max(0.0, float(delta_jours)) / f)


def est_purgeable(delta_jours: float, force: float | None) -> bool:
    """An EPISODIC entry is purged when its weight drops below the threshold, i.e. ~4.6 × force.

    Thirteen days for an ordinary trip never recalled, ninety-one for a `marquant`:
    hence never within a run. Replaces the per-type thresholds of the cleanup in service, which
    carried away a thirty-one-day salient memory together with the ordinary trips.

    CONCEPTS never go through here: they are not forgotten by the clock (lot 3).
    """
    return poids_temporel(delta_jours, force) < float(settings.agent.memoire__purge_seuil_poids)


# ── Declaration of what is active ────────────────────────────────────────────────


def composantes_sans_source() -> tuple[str, ...]:
    """The components that have, AT THIS INSTANT, nothing to feed them.

    `COMPOSANTES_INACTIVES` lists those that have no source *by default*. Since
    ticket 079, `incident_reseau` leaves it **as soon as a declared shock carries it**: the shock IS
    the source that ticket 071 had left pending. Without a loaded shock, it stays declared
    inactive — a component without a source must keep saying so, otherwise its zero would be
    confused with that of a perfect trip.
    """
    inertes = list(COMPOSANTES_INACTIVES)
    if "incident_reseau" in inertes:
        try:
            from llm.evenements import incident_reseau_a_une_source

            if incident_reseau_a_une_source():
                inertes.remove("incident_reseau")
        except Exception:  # noqa: BLE001 — the absence of the module does not change the declaration
            pass
    return tuple(inertes)


def journal_des_composantes() -> str:
    """Startup line naming the active and inactive components of `I_det`.

    A silent setup cannot tell "the component is zero because the trip
    went well" from "the component is zero because nothing feeds it".
    """
    toutes = {
        "retard": POIDS_RETARD,
        "correspondance_ratee": POIDS_CORRESPONDANCE_RATEE,
        "incident_reseau": POIDS_INCIDENT_RESEAU,
        "mode_contraint": POIDS_MODE_CONTRAINT,
    }
    inertes = set(composantes_sans_source())
    actives = [f"{n} ({p:.2f})" for n, p in toutes.items() if n not in inertes]
    inactives = [f"{n} ({p:.2f})" for n, p in toutes.items() if n in inertes]
    ligne = (
        f"[gravite] composantes de la gravité déterministe — ACTIVES : {', '.join(actives)}"
        f" | INACTIVES (aucune source, contribuent 0) : {', '.join(inactives) or 'aucune'}"
        f" | RETARD_REF = {settings.agent.memoire__retard_ref_s} s"
        f" | seuil de choc = {settings.agent.memoire__importance_choc}"
    )
    logger.info(ligne)
    if inactives:
        logger.warning(
            f"[gravite] {len(inactives)} severity component(s) without a source: severity is "
            f"UNDERESTIMATED as long as GAMA does not produce them. The others are not reweighted."
        )
    return ligne
