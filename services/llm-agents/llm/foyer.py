"""What a household member tells the others — ticket 100, batch 4 (design: ticket 078).

THE AGENT CONSOLIDATES ALONE, AND DIES WITH WHAT IT HAS LEARNT
--------------------------------------------------------------
Yet the household has existed in the data since the seal — `household.id` — and it is the
**only social group of the simulation that carries a stable identifier**. On the v6 cohort,
788 agents out of 1,000 live with someone else in the simulation, and these people are not
alike: 223 households out of 287 mix different trip purposes, 133 different
licences.

This module makes it a channel. At the evening consolidation, what the other members have lived and
learnt enters the call **as one more entry**.

« The evening » is a rule, not a figure of speech (2026-09-25): the first consolidation of the
receiver from `memoire__recit_soir_heure` (6 p.m.) on, or before 3 a.m., and only once per
simulated day. Before, the block went at each consolidation — three per day at the median —,
and a receiver who consolidated little found up to eleven summaries from the others. No separate pass, no dedicated
prompt, **not a single extra LLM call**: the extra cost is in input tokens on a
call that already takes place.

TWO THINGS CIRCULATE, AND THEY DO NOT CIRCULATE THE SAME WAY
------------------------------------------------------------
**The evening account** — the summary of the day of each other member (D1, clarified on
2026-09-22). This summary is not written here: it is the reflection the agent itself wrote
in the evening, `a text summarising what happened today`. The account **quotes** it. It is the only reading
compatible with the instruction « no new information »: a summary we composed
would be a rewording, hence a text its bearer never said.

**The beliefs** — the concepts, under the six rules of 078, unchanged.

A SINGLE HOP (D2), AND FOUR GUARDS AGAINST THE LOOP
---------------------------------------------------
The author, on 2026-09-22: « beware of infinite loops within the family; the agent must
recognise information already known ». No guard is enough on its own.

- **G1 — an account is never served twice to the same receiver.** Marker `lu_jusqu_a`, per
  receiver AND per member, set on the timestamp of the last summary QUOTED (not on the time of the
  receiver, which swallowed a summary dated before but arrived after in the queue): what A
  produces after B's turn is not lost, it is read the next evening. The sharing is continuous, with a lag of at most one night, never a
  loss — and the order of the consolidations stays irrelevant, which avoids a barrier in an
  EDF queue sized so as not to have any.
- **G2 — the receiver sees what it already believes**, in the same call: `known_beliefs` is in
  the reflection prompt since batch 3 of 071, with its counters. That is where the agent
  recognises known information, and the vocabulary to say so exists: `confirm` rather
  than `create`.
- **G3 — what is heard never goes out again.** `origine: entendu`, including after
  later confirmation by a trip of the receiver. Its own day can create a distinct `vecu`
  belief, which, for its part, circulates.
- **G4 — the circular rewording detector**: within one household and one basket,
  how many distinct concepts coexist and how many keywords they share. ⚠ It ALERTS,
  it cuts nothing: 071 and 077 both refused to let a similarity measure decide in
  place of a rule.

G3 closes the cycle *A says → B believes → B says → A believes*. It does not prevent A from repeating the same
thing every evening: it is G2 that must absorb it, and G4 that must make it visible if G2 fails.

NOTHING IS WRITTEN BEHIND THE RECEIVER'S BACK
---------------------------------------------
The block is an **entry of the call**, never a write into B's memory. A sentence
heard that resonates with nothing disappears with the call — that is the right default: the memory does not
grow from having listened.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from loguru import logger

from llm.concepts import panier_de
from llm.memory import MemoryType
from settings import settings

# Modes that no ownership lock rules out: they pass through for everyone, here as
# in `eligibilite()`.
MODES_SANS_VERROU: frozenset[str] = frozenset({"walking", "public_transport", "train"})

# Mapping between the vocabulary of the concepts (canonical hierarchy) and that of the
# ownership lock (`vehicle_chain._VEHICLE_MODES`). EXPLICIT and tested: three mode vocabularies
# coexist in this repository, and an unknown word that passed « by default » would make the licence
# lock decorative (ticket 077, batch A).
MODE_VERS_VEHICULE: dict[str, str] = {"car": "car", "cycling": "bike", "motorbike": "bike"}


@dataclass
class Membre:
    """What the household needs to know about an agent, and nothing more."""

    person_id: str
    household_id: str
    nom: str
    age: int | None
    immobile: bool = False

    @property
    def libelle(self) -> str:
        """« Matéo (8) », or « Matéo » when the age is missing. Never a family tie.

        ⚠ EMC² and eqasim give the household membership, the age and the gender — **not the
        parentage**. Saying « your son » would fabricate a datum, and it would fall back into the
        prompts as a fact.
        """
        return f"{self.nom} ({self.age})" if self.age is not None else self.nom


@dataclass
class EtatFoyer:
    """What a receiver has already heard. Persisted at the resume point (ticket 075).

    Without persistence, a hot resume would make the whole household hear again several nights
    already heard — the defect that 075 fixed for the memory, not to be reintroduced through
    the household's door.
    """

    # G1 — key `receveur:membre`, value the simulated timestamp of the last summary of this member QUOTED
    # to this receiver. A resume point from before 2026-09-25 carries `receveur` keys alone
    # (the time of its last consolidation): they still serve as a fallback at reading.
    lu_jusqu_a: dict[str, str] = field(default_factory=dict)
    # The simulated day (boundary at 3 a.m.) on which this receiver has already heard its household this evening.
    soir_servi: dict[str, str] = field(default_factory=dict)
    # R2 — the beliefs already shown to this receiver, and in what form. The key is
    # `(receveur, doc_id)`, the value the fingerprint of the content shown: a REFINED concept changes
    # fingerprint and may go out again, a merely CONFIRMED concept may not.
    croyances_montrees: dict[str, str] = field(default_factory=dict)


_membres: dict[str, Membre] = {}
_par_foyer: dict[str, list[str]] = {}
_etat = EtatFoyer()
_initialise = False


# ── Registre du processus ───────────────────────────────────────────────────────────────────
def initialiser(population) -> int:
    """Indexes the households of the loaded population. Returns the number of multi-member households.

    Without `household_id`, no household: an [ALARME] says so rather than letting a whole run
    unfold with an empty channel and without the slightest symptom.
    """
    global _initialise
    _membres.clear()
    _par_foyer.clear()
    for personne in population:
        foyer = getattr(personne, "household_id", None)
        if not foyer:
            continue
        traits = getattr(getattr(personne, "identity", None), "traits_json", None) or {}
        membre = Membre(
            person_id=str(personne.person_id),
            household_id=str(foyer),
            nom=str(traits.get("name") or personne.person_id),
            age=traits.get("age"),
            immobile=bool(getattr(personne, "immobile", False)),
        )
        _membres[membre.person_id] = membre
        _par_foyer.setdefault(membre.household_id, []).append(membre.person_id)
    _initialise = True

    multi = sum(1 for m in _par_foyer.values() if len(m) > 1)
    total = sum(1 for _ in population)
    if not _membres and total:
        logger.error(
            "[ALARME] [foyer] aucun agent de la population chargée ne porte `household_id` : "
            "le partage au sein du foyer ne touchera PERSONNE, et le run se déroulera sans le "
            "moindre symptôme. Vérifiez que la population porte bien `household.id`."
        )
    else:
        logger.info(
            f"[foyer] {len(_par_foyer)} ménage(s) indexé(s) sur {total} agent(s), dont "
            f"{multi} à plusieurs membres — {sum(len(m) for m in _par_foyer.values() if len(m) > 1)} "
            f"agent(s) ont quelqu'un à qui parler le soir"
        )
    return multi


def reinitialiser() -> None:
    """Forgets the index and the state. Reserved for tests and for the end of a run."""
    global _initialise, _etat
    _membres.clear()
    _par_foyer.clear()
    _etat = EtatFoyer()
    _compteurs_soir.clear()
    _receveurs_en_exces.clear()
    _initialise = False


def etat() -> EtatFoyer:
    return _etat


def charger_etat(brut: dict | None) -> None:
    """Rereads the state from a resume point. An unreadable state does not lose the run."""
    global _etat
    if not brut:
        return
    try:
        _etat = EtatFoyer(
            lu_jusqu_a=dict(brut.get("lu_jusqu_a") or {}),
            croyances_montrees=dict(brut.get("croyances_montrees") or {}),
            soir_servi=dict(brut.get("soir_servi") or {}),
        )
        logger.info(
            f"[foyer] état relu du point de reprise : {len(_etat.lu_jusqu_a)} repère(s) de "
            f"lecture, {len(_etat.croyances_montrees)} croyance(s) déjà montrée(s), "
            f"{len(_etat.soir_servi)} receveur(s) déjà servi(s) un soir"
        )
    except Exception as err:  # noqa: BLE001
        logger.warning(f"[foyer] état de reprise illisible ({err}) — le foyer repart à neuf")


def restaurer_depuis_point(meta: dict | None) -> None:
    """Rereads the household state of a restored resume point (ticket 118, O1).

    To be called AFTER the construction of the scenario: the factory resets the household
    (`reinitialiser`, then `initialiser`). Reread before, the state would be erased at once — that is what
    made each resume start again without any reading marker. On the a13 v5 control, the
    first evening after a resume thus quoted 112 summaries, all those since the first day.

    An empty state is legitimate (point of the first day: nobody has heard anything yet). A point
    WITHOUT the `foyer` key, on a run where sharing is on, is not: it dates from before
    ticket 100, and the resume would make the household hear again everything it has already heard.
    """
    if not meta:
        return
    if "foyer" not in meta:
        if actif():
            logger.error(
                f"[ALARME] [foyer] le point de reprise du jour {meta.get('jour_simule')} ne porte "
                f"pas l'état du foyer : les repères de lecture repartent de zéro, et le premier "
                f"soir fera ré-entendre au foyer tout ce qu'il a déjà entendu."
            )
        return
    brut = meta.get("foyer") or {}
    if not any(brut.values()):
        logger.info(
            f"[foyer] état du foyer vide au point du jour {meta.get('jour_simule')} : aucun "
            f"récit n'avait encore été entendu."
        )
        return
    charger_etat(brut)


def etat_pour_reprise() -> dict:
    return {
        "lu_jusqu_a": dict(_etat.lu_jusqu_a),
        "croyances_montrees": dict(_etat.croyances_montrees),
        "soir_servi": dict(_etat.soir_servi),
    }


def actif() -> bool:
    return bool(getattr(settings.agent, "memoire__partage_foyer_enabled", False))


def foyer_de(person_id: str) -> str | None:
    """The household of this agent, or `None` if it carries none."""
    membre = _membres.get(str(person_id))
    return membre.household_id if membre else None


def autres_membres(person_id: str) -> list[Membre]:
    """The other PRESENT members of the same household. R6.

    A household member absent from the cohort does not exist and tells nothing; an `immobile`
    member decides nothing either, and has nothing to tell.
    """
    moi = _membres.get(str(person_id))
    if moi is None:
        return []
    return [
        _membres[p]
        for p in _par_foyer.get(moi.household_id, [])
        if p != moi.person_id and not _membres[p].immobile
    ]


# ── The evening account ─────────────────────────────────────────────────────────────────────
def _empreinte(texte: str) -> str:
    return hashlib.sha256((texte or "").strip().encode("utf-8")).hexdigest()[:16]


def _reflexions_depuis(long_term_memory, person_id: str, depuis: datetime | None) -> list:
    """The day summaries of this agent later than the marker, from oldest to most recent."""
    try:
        entrees = long_term_memory.user_metadata[str(person_id)]["entries"]
    except (KeyError, TypeError, AttributeError):
        return []
    retenues = [
        e for e in entrees
        if e.memory_type == MemoryType.REFLECTION
        and (depuis is None or e.timestamp > depuis)
        and (e.content or "").strip()
    ]
    return sorted(retenues, key=lambda e: e.timestamp)


def _lire_repere(brut: str | None) -> datetime | None:
    if not brut:
        return None
    try:
        return datetime.fromisoformat(brut)
    except (TypeError, ValueError):
        return None


def _repere(receveur_id: str, membre_id: str) -> datetime | None:
    """The last summary of this member already quoted to this receiver — or the marker of the old form."""
    propre = _lire_repere(_etat.lu_jusqu_a.get(f"{receveur_id}:{membre_id}"))
    return propre if propre is not None else _lire_repere(_etat.lu_jusqu_a.get(receveur_id))


def recit_du_soir(long_term_memory, receveur_id: str, maintenant: datetime) -> list[str]:
    """What the other members have told of their day, since the last time.

    One line per member who has something new, quoting their summaries from oldest to most
    recent. Empty when the household is empty, when the flag is off, or when nobody has
    produced anything new. ⚠ No time rule here: it is `bloc_du_soir` that decides WHEN the
    household speaks; this function only says WHAT it has to say.

    G1 is structural: the marker of each member moves up to the last summary quoted, and a
    reflection already served does not go out again. An account is NEVER copied into the memory of the
    receiver — it only exists in the call, so there is nothing to retell.
    """
    if not actif():
        return []
    receveur_id = str(receveur_id)
    autres = autres_membres(receveur_id)
    if not autres:
        return []

    # ⚠ NO TRUNCATION (author, 2026-09-29): everything a member has said since the last
    # evening heard is quoted, whatever its volume. The former bound (8 members, 8 summaries per
    # member) postponed the surplus to the next evening; on the a13 v5 control it let a resume
    # that had lost its markers pour out 112 summaries in one evening, then postpone from evening to evening
    # for ten days — an account lagging behind the day it tells. A large volume is
    # no longer a flow to spread out: it is the SYMPTOM of an upstream defect (lost markers, receiver
    # that no longer consolidates in the evening), and it is flagged without cutting anything.
    seuil_alerte = int(getattr(settings.agent, "memoire__recit_soir_alerte_par_membre", 8))
    lignes: list[str] = []
    en_exces: dict[str, int] = {}
    for membre in sorted(autres, key=lambda m: m.person_id):
        reflexions = _reflexions_depuis(
            long_term_memory, membre.person_id, _repere(receveur_id, membre.person_id)
        )
        if not reflexions:
            continue
        if len(reflexions) > seuil_alerte:
            en_exces[membre.person_id] = len(reflexions)
        # ⚠ The text is QUOTED, not reworded. « No new information » (author,
        # 2026-09-22): what circulates is what its bearer wrote themselves, in their own words.
        # Rewriting it would bring in information they did not say.
        citations = " Later: ".join(f'"{(e.content or "").strip()}"' for e in reflexions)
        lignes.append(f"- {membre.libelle} told you about their day: {citations}")
        _etat.lu_jusqu_a[f"{receveur_id}:{membre.person_id}"] = reflexions[-1].timestamp.isoformat()
        _compteurs_soir["bilans_cites"] += len(reflexions)

    if en_exces:
        # The [ALARME] fires on a rising edge, per receiver: a receiver that stays in excess
        # several evenings does not flood the journal. Nothing is cut.
        _compteurs_soir["exces"] += 1
        if receveur_id not in _receveurs_en_exces:
            _receveurs_en_exces.add(receveur_id)
            detail = ", ".join(f"{m} : {n}" for m, n in sorted(en_exces.items()))
            logger.error(
                f"[ALARME] [foyer] récit du soir anormalement long pour {receveur_id} le "
                f"{maintenant.isoformat()} : bilans en attente par membre ({detail}), au-delà "
                f"de `memoire__recit_soir_alerte_par_membre` = {seuil_alerte}. Tout est cité, "
                f"rien n'est reporté ; un tel volume signale un défaut en amont — repères de "
                f"lecture perdus à une reprise, ou receveur qui ne consolide plus le soir."
            )
    else:
        _receveurs_en_exces.discard(receveur_id)
    return lignes


# ── The beliefs (R1 to R6 of ticket 078) ────────────────────────────────────────────────────
def _mode_praticable(mode: str | None, traits: dict) -> bool:
    """R4 — the Constance / Jacques case.

    What Jacques learns about the car must **never** become a belief of Constance,
    12 years old, without a licence and without a bike. No permissive fallback: an unknown mode does not pass,
    it is counted and raises an alarm.
    """
    if not mode or mode == "any":
        return True
    if mode in MODES_SANS_VERROU:
        return True
    vehicule = MODE_VERS_VEHICULE.get(mode)
    if vehicule is None:
        logger.warning(
            f"[foyer] mode « {mode} » hors de la table de correspondance — la croyance NE "
            f"TRAVERSE PAS. Un mot inconnu qui passerait par défaut rendrait le verrou de "
            f"possession décoratif (ticket 077, lot A)."
        )
        return False
    if vehicule == "car":
        return bool(traits.get("has_driving_license")) and int(
            traits.get("number_of_cars") or 0
        ) > 0
    return str(traits.get("personal_bike") or "").lower() not in ("", "no bike")


def croyances_partagees(
    long_term_memory, receveur, maintenant: datetime
) -> tuple[list[str], dict[str, int]]:
    """What the other members have learnt and that may pass. Returns (statements, refusals per rule).

    The refusals are counted SEPARATELY, rule by rule: if the channel is empty, it is the
    first thing to look at, and a global counter would not say which of the six closes the
    door.
    """
    # « G3 » is not a rule of ticket 078: it is the anti-loop guard of the single hop
    # (D2). It has been counted HERE with the others since 2026-09-22, because it was
    # not: on the run of the read channel, 130 concepts out of 426 were discarded by it without a
    # single line saying so — the summary added up to 296 out of 426 and nobody saw the hole.
    # A guard that nobody knows whether it bit cannot be checked: that is exactly what
    # the repository has already paid for on the household refusal counters.
    refus = {f"R{i}": 0 for i in range(1, 7)} | {"G3": 0}
    if not actif():
        return [], refus
    autres = autres_membres(str(receveur.person_id))
    if not autres:
        refus["R6"] += 1
        return [], refus

    traits = getattr(getattr(receveur, "identity", None), "traits_json", None) or {}
    seuil = int(getattr(settings.agent, "memoire__partage_foyer_observations_min", 1))
    borne = int(getattr(settings.agent, "memoire__partage_foyer_max_bloc", 12))
    lignes: list[str] = []
    candidats = 0

    for membre in sorted(autres, key=lambda m: m.person_id):
        try:
            entrees = long_term_memory.user_metadata[membre.person_id]["entries"]
        except (KeyError, TypeError, AttributeError):
            continue
        for concept in entrees:
            if concept.memory_type != MemoryType.CONCEPT:
                continue
            candidats += 1
            _compteurs_soir["candidats"] += 1
            # G3 / D2 — what was HEARD never goes out again, even confirmed later.
            if (concept.origine or "vecu") == "entendu":
                refus["G3"] += 1
                continue
            if int(concept.observations or 0) < seuil:
                refus["R1"] += 1
                continue
            if not concept.axe_objet:
                refus["R3"] += 1
                continue
            if not _mode_praticable(concept.axe_objet, traits):
                refus["R4"] += 1
                continue
            contenu = _contenu_lisible(concept)
            if not contenu:
                refus["R3"] += 1
                continue
            cle = f"{receveur.person_id}:{getattr(concept, 'doc_id', '') or contenu[:40]}"
            empreinte = _empreinte(contenu + ("|hs" if not concept.est_servi else ""))
            if _etat.croyances_montrees.get(cle) == empreinte:
                # R2 — never shown twice. A mere CONFIRMATION does not make a concept go out
                # again: otherwise a belief its author confirms every day would be
                # served every evening to the whole family. Only a refinement — which changes the
                # content — or a setting-aside changes the fingerprint.
                refus["R2"] += 1
                continue
            if len(lignes) >= borne:
                continue
            _etat.croyances_montrees[cle] = empreinte
            if not concept.est_servi:
                lignes.append(f'- {membre.libelle} no longer believes: "{contenu}"')
            else:
                lignes.append(
                    f'- {membre.libelle} told you: "{contenu}" — they have seen it '
                    f'{int(concept.observations or 0)} time(s)'
                )

    if candidats and not lignes:
        logger.info(
            f"[foyer] {receveur.person_id} : {candidats} concept(s) examiné(s) chez ses "
            f"co-résidents, aucun ne traverse — refus par règle {refus}"
        )
    return lignes, refus


def _contenu_lisible(concept) -> str:
    """The text of a concept, whether its content is a JSON 5-tuple or a bare sentence."""
    import json

    brut = (concept.content or "").strip()
    if brut.startswith("["):
        try:
            return str(json.loads(brut)[0]).strip()
        except Exception:  # noqa: BLE001
            return brut
    return brut


# ── G4 — le détecteur de reformulation circulaire ───────────────────────────────────────────
_MOTS = re.compile(r"[a-zà-ÿ]{4,}", re.IGNORECASE)


def detecter_reformulation(long_term_memory, household_id: str) -> list[tuple]:
    """How many distinct concepts coexist in one basket, and what they share.

    ⚠ **A reading indicator, never a threshold that acts.** 071 and 077 both
    refused to let a similarity measure decide in place of a rule. This one alerts a
    human; it cuts nothing, and it must never do so.

    Returns a list of `(basket, number of concepts, shared words)` for the baskets where
    several concepts coexist — a family that says the same thing again in four forms can
    then be seen at a glance.
    """
    paniers: dict[tuple, list[str]] = {}
    for person_id in _par_foyer.get(str(household_id), []):
        try:
            entrees = long_term_memory.user_metadata[person_id]["entries"]
        except (KeyError, TypeError, AttributeError):
            continue
        for concept in entrees:
            if concept.memory_type != MemoryType.CONCEPT:
                continue
            paniers.setdefault(
                panier_de(concept.axe_objet, concept.axe_motif), []
            ).append(_contenu_lisible(concept))

    signales = []
    for panier, contenus in sorted(paniers.items(), key=lambda kv: str(kv[0])):
        if len(contenus) < 2:
            continue
        ensembles = [set(m.lower() for m in _MOTS.findall(c)) for c in contenus]
        partages = set.intersection(*ensembles) if ensembles else set()
        if partages:
            signales.append((panier, len(contenus), sorted(partages)))
    if signales:
        logger.warning(
            f"[foyer] reformulation possible dans le ménage {household_id} : "
            + " ; ".join(
                f"panier {p} — {n} concepts partageant {mots}" for p, n, mots in signales
            )
            + ". Indicateur de LECTURE : rien n'est coupé, un humain décide."
        )
    return signales


# Evening counters, cumulated over the run. Ticket 078 § 6 asks for them in this order, and the
# first measure comes BEFORE all the others: if the number of concepts that pass R1 is
# close to zero, the channel is empty and nothing else makes sense to measure. 077 counted 225
# concepts out of 231 left at zero observation — R1 would then leave the channel closed.
_compteurs_soir: Counter = Counter()
# The receivers whose last account exceeded the alert threshold — the [ALARME] fires on entry only.
_receveurs_en_exces: set[str] = set()
# Same day boundary as the measures (`scripts/analysis/mesures/calcul.py`): an account
# heard at 1:30 a.m. belongs to the evening before.
FRONTIERE_JOUR_H = 3


def compteurs() -> dict:
    """What the household has done since the start of the run. Read by the journal, never reset."""
    return dict(_compteurs_soir)


def journaliser_compteurs() -> None:
    """The household summary, logged EVEN AT ZERO.

    A mute counter does not tell « the household had nothing to say » from « the mechanism is not
    running », and it is this confusion that cost ticket 075 thirty days.
    """
    if not actif():
        logger.info("[foyer] partage au sein du foyer ÉTEINT — aucun bloc n'a été servi.")
        return
    c = _compteurs_soir
    logger.info(
        f"[foyer] bilan du run — {c['blocs_servis']} bloc(s) servi(s) à "
        f"{c['receveurs']} consolidation(s), {c['recits']} bilan(s) et {c['croyances']} "
        f"croyance(s) transmis, {c['bilans_cites']} bilan(s) cité(s), {c['exces']} récit(s) "
        f"au-delà du seuil d'alerte (rien n'est tronqué) ; "
        f"{c['hors_soir']} consolidation(s) de jour et {c['deja_servi_ce_soir']} seconde(s) "
        f"consolidation(s) du soir sans bloc. Concepts examinés : "
        f"{c['candidats']}, écartés par règle — "
        + ", ".join(f"R{i}:{c[f'refus_R{i}']}" for i in range(1, 7))
        + f", G3:{c['refus_G3']} (déjà entendus, saut unique)"
    )
    # The count must add up. A gap means that an exit of the loop is counted
    # nowhere, and that is precisely what makes a guard uncheckable.
    compte = c["croyances"] + sum(c[f"refus_R{i}"] for i in range(1, 7)) + c["refus_G3"]
    if c["candidats"] and compte != c["candidats"]:
        logger.error(
            f"[ALARME] [foyer] le compte ne tombe pas juste : {c['candidats']} concept(s) "
            f"examiné(s) pour {compte} issue(s) comptée(s) — {c['candidats'] - compte} "
            f"concept(s) quittent la boucle sans être comptés. Un refus invisible rend la "
            f"règle qui l'a produit invérifiable."
        )
    if c["blocs_servis"] and not c["croyances"]:
        logger.error(
            f"[ALARME] [foyer] {c['blocs_servis']} bloc(s) servi(s) et AUCUNE croyance "
            f"transmise sur tout le run : le canal des croyances est fermé. Regarder R1 "
            f"d'abord ({c['refus_R1']} concepts écartés faute d'ancrage) — c'est la première "
            f"chose à mesurer, et si elle est proche de tout, rien d'autre n'a de sens."
        )


def bloc_du_soir(long_term_memory, receveur, maintenant: datetime) -> str:
    """The « Tonight at home » block, ready to enter the reflection call. Empty if nothing.

    ⚠ This block is an ENTRY of the call, never a write into the receiver's memory. A
    sentence heard that resonates with nothing disappears with the call: the memory does not grow
    from having listened.
    """
    receveur_id = str(receveur.person_id)
    journee = (maintenant - timedelta(hours=FRONTIERE_JOUR_H)).date().isoformat()
    if actif() and autres_membres(receveur_id):
        # In the evening, once. A daytime consolidation or a second consolidation of the same
        # evening examines NOTHING and moves no marker: what would have been said waits for the
        # right time. (An agent alone follows the former path: its R6 refusal is counted.)
        heure_soir = int(getattr(settings.agent, "memoire__recit_soir_heure", 18))
        if FRONTIERE_JOUR_H <= maintenant.hour < heure_soir:
            _compteurs_soir["hors_soir"] += 1
            return ""
        if _etat.soir_servi.get(receveur_id) == journee:
            _compteurs_soir["deja_servi_ce_soir"] += 1
            return ""

    recits = recit_du_soir(long_term_memory, receveur_id, maintenant)
    croyances, refus = croyances_partagees(long_term_memory, receveur, maintenant)
    # ⚠ The refusals were COMPUTED then thrown away — measure no. 1 of 078 § 6, « what is
    # eligible, before everything else », was therefore emitted nowhere. An empty channel would have
    # read as a household with nothing to say to each other.
    _compteurs_soir["receveurs"] += 1
    _compteurs_soir["recits"] += len(recits)
    _compteurs_soir["croyances"] += len(croyances)
    for regle, combien in refus.items():
        _compteurs_soir[f"refus_{regle}"] += combien
    if not recits and not croyances:
        # An evening with nothing new does not close the evening: what another member tells
        # at 9 p.m. will be heard at the receiver's next consolidation.
        return ""
    _etat.soir_servi[receveur_id] = journee
    _compteurs_soir["blocs_servis"] += 1
    lignes = ["Tonight at home"] + recits + croyances
    return "\n".join(lignes)
