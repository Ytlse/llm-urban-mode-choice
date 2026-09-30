"""The agent judges what it has just experienced or read — ticket 100, lot 3, decision D4.

WHY THIS MODULE
---------------
Until now the severity of an event was purely deterministic: a delay, a missed connection, a
network incident. It is objective and costs nothing — but it has nothing to say about a press
article, which inflicts no delay and would therefore be worth **zero**, that is exactly the
value of a perfect trip. An entry at zero lives 2.8 days: its silence would pass for an
absence of effect whereas it would only measure the lifetime it was
given.

Judgement gives weight to what the simulation does not measure. It applies to **both**
regimes (D4), and since decision **D7 of 2026-09-22** the severity of an entry is the agent's
estimate, **and that alone** — in both channels.

WHAT D7 REMOVES, AND THE GUARD THAT REPLACES IT
-----------------------------------------------
`gravite_concept` applied `I = max(I_llm, I_det)` under a « NON NÉGOCIABLE » comment: a model
that judged a thirty-minute breakdown « anodin » could not downgrade it. D7 removes this
protection, and it is not neutral — a 2.8-day memory where the floor imposed fifteen can no
longer be told apart from an event that produced nothing.

What replaces the floor corrects nothing: **the `estimated − deterministic` gap is logged
at every judgement**, and an `[ALARME]` is raised when the agent underestimates by more than one
level. We want to **know**, not to catch up. Correcting silently would restore the floor under
another name, and the campaign would again measure the guard instead of measuring the agent.

⚠ The deterministic term does not disappear from the setup: the delay is still endured, it
shifts the day, constrains the following trips, and enters what the agent tells in the
evening. It only stops weighing on the severity.

THE REFUSAL IS CLEAR-CUT, AND IT DOES NOT REUSE `gravite_jugee`
---------------------------------------------------------------
`llm/gravite.py:gravite_jugee` falls back **silently** to `None` when the model returns an
off-grid level, and that is intended there: an aberrant answer must not lose a whole
reflection carrying five concepts. Here, the object of the call IS the level. An off-grid
answer leaves nothing to save, and a fallback would manufacture an exposure that was not
judged while counting it like the others (059, Q17).

Hence: refusal, `[ALARME]`, and the exposure is declared void. No median value, no fallback
to the deterministic severity, no default level.
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from llm.gravite import ALIAS_NIVEAUX, NIVEAUX, ancres_anglaises, borne_0_1
from settings import settings

CATEGORIE = "evenement_jugement"

# Valences of the reflection schema, taken over as they are (D3: a single scale).
VALENCES: tuple[str, ...] = ("negative", "neutral", "positive")
# The French vocabulary of `MemoryEntry.valence`, where they land.
VALENCE_FR: dict[str, str] = {
    "negative": "negative", "neutral": "neutre", "positive": "positive",
}


class JugementRefuse(ValueError):
    """The model did not answer within the grid. The exposure is void."""


@dataclass(frozen=True)
class Jugement:
    """What the agent says of the event, and the severity that comes out of it."""

    intensite: str          # one of the five levels, canonical French vocabulary
    valence: str            # `negative` | `neutre` | `positive`
    modes: tuple[str, ...]  # the modes it concerns, possibly none
    importance_estimee: float
    # Since D7, `importance_retenue == importance_estimee`. The two fields STAY distinct,
    # and it is not a redundancy: the trace must keep saying what the agent estimated AND
    # what was retained. The day a guard came back, it would show in the gap between the
    # two columns rather than in a changelog.
    importance_retenue: float
    # The gap to the measured fact, logged and never applied. Negative = the agent underestimates.
    ecart_au_fait: float = 0.0


def resoudre_intensite(brut, evenement_id: str, person_id: str, jour: int) -> str:
    """The level returned by the model, mapped to the canonical vocabulary. Raises otherwise.

    Both vocabularies are accepted, English and French: the setup addresses the model in
    English (ticket 074) but the scale was defined and discussed in French, and a model that
    answered in the other language must be understood rather than rejected.
    """
    clef = str(brut or "").strip().lower()
    clef = ALIAS_NIVEAUX.get(clef, clef)
    if clef not in NIVEAUX:
        logger.error(
            f"[ALARME] [evenements] « {evenement_id} »: the model returned the level "
            f"« {brut} », outside the five levels {sorted(ALIAS_NIVEAUX)}. NO fallback is "
            f"applied: the exposure of {person_id} on day {jour} did NOT take place. Do not "
            f"count it in the analysis."
        )
        raise JugementRefuse(f"échelon « {brut} » hors grille")
    return clef


def resoudre_valence(brut) -> str:
    """The valence, mapped to the `MemoryEntry` vocabulary. Neutral by default.

    Unlike the level, a missing valence does not make the call void: the severity holds
    without it, and `neutre` is the value every unqualified entry already carries. A default
    here therefore manufactures nothing — it merely adds nothing.
    """
    clef = str(brut or "").strip().lower()
    if clef not in VALENCES:
        if clef:
            logger.warning(
                f"[evenements] valence « {brut} » outside {VALENCES} — taken as neutral"
            )
        return "neutre"
    return VALENCE_FR[clef]


def resoudre_modes(brut) -> tuple[str, ...]:
    """The affected modes, filtered on the repository hierarchy.

    An unknown mode is DROPPED and counted, never let through: mode vocabulary already cost
    ticket 077 thirty days, when two lists coexisted without anything saying so.
    """
    from llm.axes import mode_canonique

    retenus, ecartes = [], []
    for brut_mode in brut or []:
        canonique = mode_canonique(str(brut_mode))
        if canonique and canonique == str(brut_mode).strip().lower():
            retenus.append(canonique)
        else:
            ecartes.append(str(brut_mode))
    if ecartes:
        logger.warning(
            f"[evenements] mode(s) {ecartes} outside the repository hierarchy — dropped from "
            f"the judgement. The concept will not carry these modes."
        )
    return tuple(dict.fromkeys(retenus))


def charge_utile(person_id: str, perception: str, texte: str) -> dict:
    """The call payload. The anchors come from `gravite.py`, not from the template.

    The `stm_reflection` template copies them, and `gravite.py` notes that "the textual
    anchors live in the template": two sources for one scale, which will diverge the day
    one of them moves. This one has only one — and if the scale changes, the prompt follows
    without anyone having to think about it.
    """
    from urban_mobility_agents.utils.routage import instances_pour

    return {
        "category": CATEGORIE,
        "instances_admises": instances_pour(CATEGORIE),
        "agents": [
            {"agent_id": str(person_id), "perception": perception, "evenement": texte}
        ],
        "parameters": {
            "temperature": 0.2,
            # ⚠ 256 WAS TOO TIGHT, and the symptom did not look like its cause. Reasoning
            # models (`gpt-oss-120b`, `gpt-oss-20b`) spend their reasoning INSIDE this
            # budget: measured on 2026-09-22 on Groq, 240 to 330 reasoning tokens before the
            # first JSON character. At 256, the provider returns « max completion tokens
            # reached before generating a valid document » — or, when the reasoning barely
            # fits, a valid but DEGRADED JSON, where the model takes the first value of each
            # enumeration. That is how 19 calls out of 38 came back empty and the other
            # 11 all answered `anodin`, the first level of the list.
            #
            # 1024 is ample: 290 tokens measured at worst on `gpt-oss-20b`, 66 on `qwen3.8`.
            # The ceiling costs nothing — only emitted tokens are billed and counted in the
            # OTPM. What it buys is that the budget stops being the variable that decides
            # the severity in the agent's place.
            "max_tokens": 1024,
            # The anchors are passed under the ENGLISH label of the output schema. Giving them
            # in French while the enumeration accepts only English amounted to asking the
            # model to choose from a list it was then forbidden to write.
            "ancres": ancres_anglaises(),
        },
    }


async def juger(
    llm_client,
    person_id: str,
    perception: str,
    texte: str,
    gravite_deterministe: float,
    evenement_id: str,
    jour: int,
) -> Jugement:
    """One call, one event. Raises `JugementRefuse` rather than returning an invented value.

    ⚠ No fallback delay, no default value, no separate queue. When quota runs short, the
    ordinary queue makes us wait — and we wait. A judgement returned by default would be a
    manufactured measurement, and this repository has already paid the price.
    """
    reponse = await llm_client.execute(charge_utile(person_id, perception, texte))
    if not reponse or not getattr(reponse, "agents", None):
        logger.error(
            f"[ALARME] [evenements] « {evenement_id} »: no answer to the judgement of "
            f"{person_id} on day {jour}. The exposure was NOT judged."
        )
        raise JugementRefuse("réponse vide")

    rendu = reponse.agents[0]
    intensite = resoudre_intensite(
        getattr(rendu, "severity", None), evenement_id, person_id, jour
    )
    estimee = NIVEAUX[intensite]
    # ── D7, 2026-09-22: the severity is the AGENT's, and that alone ──────────────────────
    # `gravite_concept(estimee, deterministe)` took the maximum of the two. It is no longer
    # called here — it keeps its role for concepts, where the floor of a consumed group
    # stays a memory rule, outside the scope of this ticket.
    retenue = borne_0_1(estimee)
    mesure = float(gravite_deterministe or 0.0)
    ecart = round(retenue - mesure, 4)

    jugement = Jugement(
        intensite=intensite,
        valence=resoudre_valence(getattr(rendu, "valence", None)),
        modes=resoudre_modes(getattr(rendu, "modes", None)),
        importance_estimee=estimee,
        importance_retenue=retenue,
        ecart_au_fait=ecart,
    )
    logger.info(
        f"[evenements] « {evenement_id} » jugé par {person_id} (jour {jour}) : "
        f"{intensite} ({estimee:.2f}), valence {jugement.valence}, modes "
        f"{list(jugement.modes) or 'aucun'} — gravité retenue {retenue:.2f}"
        + (f", fait mesuré {mesure:.2f}, écart {ecart:+.2f}" if mesure else "")
    )
    # The guard that replaces the floor: it SAYS, it does not correct. Correcting silently
    # would restore the floor under another name, and the campaign would again measure the
    # guard instead of measuring the agent.
    seuil = float(getattr(settings.agent, "memoire__ecart_jugement_alarme", 0.30))
    if mesure and ecart <= -seuil:
        from llm.gravite import force_initiale

        logger.error(
            f"[ALARME] [evenements] « {evenement_id} » : {person_id} juge « {intensite} » "
            f"({estimee:.2f}) un fait mesuré à {mesure:.2f} — il SOUS-ESTIME de {-ecart:.2f}, "
            f"plus d'un échelon. La valeur n'est PAS corrigée (décision D7) : le souvenir vivra "
            f"{force_initiale(retenue):.1f} jours au lieu de {force_initiale(mesure):.1f}. "
            f"Si ce cas se répète, l'effet mesuré dépend de la qualité du jugement et non de "
            f"l'événement."
        )
    return jugement
