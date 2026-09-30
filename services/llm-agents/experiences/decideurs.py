"""The decision-makers (ticket 035, spec 02 D1/D8, spec 03 S10, spec 05 Q2/Q3/Q12).

Four implementations of the same contract `Decideur.choisir(person, ctx, presentees)`:

- `DecideurPasserelle`: the language model, via `LlmAgent` (same presented text as the
  simulation), **pinned** to one model: only the instances serving it are called, a
  response served by another instance is refused (`substitution_refusee`);
- `DecideurDureeMinimale`: deterministic, local heuristic, no quota;
- `DecideurAleatoire`: seeded uniform (floor `exp_00a` of the experiment plan);
- `DecideurRejeu`: serves again the archived responses of an earlier run — it is what makes
  mode equality **verifiable** without the noise of a model (D8, Q6).
"""

from __future__ import annotations

import json
import random
import re

from experiences.decision import (
    ContexteDecision,
    Proposition,
    ReponseDecideur,
    graine_ordre,
)
from experiences.ressources import MoniteurRessources
from loguru import logger
from models import Person
from urban_mobility_agents.candidats import _primary_mode

# `epuise` (real blocker, → stop/wait for a window) is reserved for CONFIRMED quota (429) and
# credit (402). "saturés / indisponibles / timeout" describes a BUSY gateway — transient: we
# retry, we never skip (R1/R2). The "satur" that lived in _RE_QUOTA wrongly classified "Providers
# saturés ou indisponibles" as `epuise` (outage of 2026-09-07, nothing was exhausted).
_RE_QUOTA = re.compile(r"\b(429|quota|rate.?limit)", re.IGNORECASE)
_RE_CREDIT = re.compile(r"\b(402|credit|crédit|insufficient)", re.IGNORECASE)
_RE_OCCUPEE = re.compile(
    r"(satur|indisponible|timeout|occup|unavailable|\b50[234]\b)", re.IGNORECASE
)


def _distribution(poids: list[float], presentees: list[Proposition]) -> dict:
    from mobility_llm.mode_choice import mode_distribution

    return mode_distribution(poids, [p.mode for p in presentees])


def _presente_local(presentees: list[Proposition]) -> dict:
    return {
        "options": [
            {"index": i, "mode": p.mode, "duree_s": p.plan.duration, "source": p.source}
            for i, p in enumerate(presentees)
        ]
    }


class DecideurDureeMinimale:
    nom = "duree_minimale"
    sans_quota = True

    async def choisir(
        self, person: Person, ctx: ContexteDecision, presentees: list[Proposition]
    ) -> ReponseDecideur:
        idx = min(
            range(len(presentees)),
            key=lambda i: (
                (
                    presentees[i].plan.duration
                    if presentees[i].plan.duration is not None
                    else float("inf")
                ),
                presentees[i].code,
            ),
        )
        poids = [1.0 if i == idx else 0.0 for i in range(len(presentees))]
        return ReponseDecideur(
            index=idx,
            fournisseur=self.nom,
            distribution=_distribution(poids, presentees),
            poids=poids,
            reponse_brute=json.dumps({"regle": self.nom, "index": idx}),
            raison="durée minimale",
            presente=_presente_local(presentees),
        )


class DecideurAleatoire:
    sans_quota = True

    def __init__(self, graine: int):
        self.graine = int(graine)
        self.nom = f"aleatoire:{self.graine}"

    async def choisir(
        self, person: Person, ctx: ContexteDecision, presentees: list[Proposition]
    ) -> ReponseDecideur:
        n = len(presentees)
        idx = random.Random(
            graine_ordre(self.graine, person.person_id, ctx.activity_id)
        ).randrange(n)
        poids = [1.0 / n] * n
        return ReponseDecideur(
            index=idx,
            fournisseur=self.nom,
            distribution=_distribution(poids, presentees),
            poids=poids,
            reponse_brute=json.dumps(
                {"regle": "uniforme", "graine": self.graine, "index": idx}
            ),
            raison="tirage uniforme",
            presente=_presente_local(presentees),
        )


class DecideurMajoritaireVoiture:
    """Empirical floor `exp_00b`: the car if it is offered, otherwise a neutral fallback.

    "Empirical prior" of the plan (`config/experience_plan`): reproduces the trivial prior
    "everyone drives". We keep the option whose **main mode** is the car, in the survey's
    sense (`_primary_mode`, EMC² hierarchy) — and **not** in the chain's sense
    (`_vehicle_mode`): a car + train feeder trip is a collective trip, it is not
    "taking the car". Among several car options, the fastest (stable tie-break by
    `code`, like the minimum duration). With no car option at all, deterministic fallback to the
    first presented option — the D7 order is already seeded, so this fallback introduces no
    draw. Local, deterministic, no quota (S10/Q12).

    `decision.decider` guarantees `len(presentees) >= 2` (0 → no solution, 1 → single choice,
    without calling): the fallback index 0 always exists.
    """

    nom = "majoritaire_voiture"
    sans_quota = True

    async def choisir(
        self, person: Person, ctx: ContexteDecision, presentees: list[Proposition]
    ) -> ReponseDecideur:
        voitures = [
            i for i, p in enumerate(presentees) if _primary_mode(p.plan) == "car"
        ]
        if voitures:
            idx = min(
                voitures,
                key=lambda i: (
                    (
                        presentees[i].plan.duration
                        if presentees[i].plan.duration is not None
                        else float("inf")
                    ),
                    presentees[i].code,
                ),
            )
            raison = "voiture disponible (mode principal)"
        else:
            idx = 0
            raison = "aucune option voiture — repli sur la première présentée"
        poids = [1.0 if i == idx else 0.0 for i in range(len(presentees))]
        return ReponseDecideur(
            index=idx,
            fournisseur=self.nom,
            distribution=_distribution(poids, presentees),
            poids=poids,
            reponse_brute=json.dumps({"regle": self.nom, "index": idx}),
            raison=raison,
            presente=_presente_local(presentees),
        )


class DecideurRejeu:
    sans_quota = True

    def __init__(self, execution, nom_source: str | None = None):
        self.execution = execution
        self.nom = f"rejeu:{nom_source or getattr(execution, 'nom', '?')}"

    async def choisir(
        self, person: Person, ctx: ContexteDecision, presentees: list[Proposition]
    ) -> ReponseDecideur:
        trace = self.execution.decision(person.person_id, ctx.activity_id)
        if trace is None or not trace.get("retenue"):
            return ReponseDecideur(
                index=None,
                fournisseur=self.nom,
                erreur="rejeu : aucune décision archivée pour ce déplacement",
            )
        code = trace["retenue"].get("code")
        idx = next((i for i, p in enumerate(presentees) if p.code == code), None)
        if idx is None:
            return ReponseDecideur(
                index=None,
                fournisseur=self.nom,
                erreur=f"rejeu : la proposition archivée {code!r} n'est pas parmi les présentées",
            )
        poids = list(trace.get("poids_presentes") or [])
        if len(poids) != len(presentees):
            poids = [1.0 if i == idx else 0.0 for i in range(len(presentees))]
        j_offre = trace.get("jour_offre") or ctx.jour_offre
        j_tire = trace.get("jour_meteo_tire")
        j_lu = trace.get("jour_meteo_lu")
        s_meteo = trace.get("source_date_meteo")
        if s_meteo is None:
            from urban_mobility_agents.utils.weather_draw import resoudre_meteo_decision

            info_m = resoudre_meteo_decision(
                person_id=person.person_id,
                timestamp=ctx.timestamp,
                graine=ctx.graine_tirage,
                weather_per_agent=True,
            )
            s_meteo = info_m["source_date_meteo"]
            j_tire = info_m["jour_meteo_tire"]
            j_lu = info_m["jour_meteo_lu"]

        return ReponseDecideur(
            index=idx,
            fournisseur=self.nom,
            distribution=dict(trace.get("distribution") or {}),
            poids=poids,
            reponse_brute=trace.get("reponse_brute"),
            raison=trace.get("raison") or "rejeu",
            souvenirs=list(trace.get("souvenirs") or []),
            presente=trace.get("presente"),
            repli_uniforme=trace.get("methode") == "repli_uniforme",
            jour_offre=j_offre,
            jour_meteo_tire=j_tire,
            jour_meteo_lu=j_lu,
            source_date_meteo=s_meteo,
        )


# The RANK of a key, never its name: the log is read, copied and passed on, and the name of an
# instance designates an account. "Passage sur la seconde clé" says all that is needed to act.
_RANGS = (
    "première",
    "seconde",
    "troisième",
    "quatrième",
    "cinquième",
    "sixième",
    "septième",
    "huitième",
    "neuvième",
    "dixième",
)


def rang_en_mots(rang: int) -> str:
    """1 → "première", 2 → "seconde"… beyond ten, "clé n° 11"."""
    return _RANGS[rang - 1] if 1 <= rang <= len(_RANGS) else f"n° {rang}"


class DecideurPasserelle:
    """The remote language model, pinned (Q2), without substitution (Q3)."""

    sans_quota = False

    def __init__(
        self,
        agent,
        modele: str,
        instances: list[str],
        moniteur: MoniteurRessources | None = None,
    ):
        self.agent = agent
        self.modele = modele
        self.instances = list(instances)
        self.moniteur = moniteur
        self.nom = f"passerelle:{modele}"
        self._curseur = (
            0  # kept: no longer used by selection, serial since 2026-09-07
        )
        self._derniere_instance: str | None = None
        self.derniere_erreur_quota: str | None = None

    def _prochaine_instance(self, besoin: int = 1) -> str | None:
        candidates = (
            self.moniteur.instances_disponibles(besoin)
            if self.moniteur is not None
            else list(self.instances)
        )
        if not candidates:
            return None
        # SERIAL, not rotation (decision of 2026-09-07). `instances_disponibles` only keeps
        # those whose DAILY quota is not exhausted: always taking the first one consumes
        # them one after another, instead of starting two buckets of 500 requests in
        # parallel. Per-minute saturation is absorbed downstream (waiting).
        inst = candidates[0]
        if inst != self._derniere_instance:
            if self._derniere_instance is not None:
                # The rank in the model's declared order, not the instance name.
                rang = (
                    self.instances.index(inst) + 1
                    if inst in self.instances
                    else len(self.instances)
                )
                logger.warning(
                    f"[decideur] Passage sur la {rang_en_mots(rang)} clé "
                    f"({rang}/{len(self.instances)}) — la précédente a épuisé son quota du jour ; "
                    f"{len(candidates)} clé(s) encore disponible(s)"
                )
            self._derniere_instance = inst
        return inst

    async def choisir(
        self, person: Person, ctx: ContexteDecision, presentees: list[Proposition]
    ) -> ReponseDecideur:
        from urban_mobility_agents.agents.llm_agent import Context

        instance = self._prochaine_instance()
        if instance is None:
            raison = (
                self.moniteur.raison_epuisement()
                if self.moniteur
                else "aucune instance"
            )
            return ReponseDecideur(
                index=None, fournisseur="", erreur=f"epuise: {raison}"
            )
        options = [p.plan for p in presentees]
        for o in options:
            o.purpose = ctx.purpose
        context = Context(
            person=person,
            timestamp=int(ctx.timestamp),
            activity_id=ctx.activity_id,
            data={"type": "travel_plan"},
        )
        trace: dict = {}
        (
            idx,
            raison,
            fournisseur,
            distribution,
        ) = await self.agent.evaluate_and_choose_travel_plan(
            context=context,
            options=options,
            destination=ctx.purpose,
            departure_time=int(ctx.departure_time),
            anticipation=ctx.anticipation,
            force_provider=instance,
            allowed_providers=set(self.instances),
            trace=trace,
            presentation_figee=True,
        )
        if "substitution_refusee" in trace:
            if self.moniteur is not None:
                self.moniteur.compteurs["substitution_refusee"] += 1
            return ReponseDecideur(
                index=None,
                fournisseur=fournisseur,
                erreur=f"substitution_refusee:{fournisseur}",
                presente={"payload": trace.get("payload")},
                reponse_brute=trace.get("reponse_brute"),
            )
        if not isinstance(idx, int) or idx < 0:
            erreur = str(trace.get("erreur") or raison or "réponse inexploitable")
            reprise_a = trace.get("reprise_a")
            if trace.get("genre_erreur") == "restriction_instances":
                # Ticket 085, lot A. A configuration contradiction — routing restriction
                # incompatible with the pinned instance — is NEITHER a busy gateway NOR an
                # exhausted quota. Putting it in either bucket sends people looking for a quota
                # where there is only one YAML line to fix: that is what cost three hours
                # on 2026-09-16. The type is new, and that is intended.
                #
                # `configuration` is unknown to `runner.py`: it falls into the waiting branch,
                # exactly as before. This lot makes the reason right and fast, it does not
                # change what the client does with it (§ 9 of the ticket, Q1 of
                # `specs/ticket_085/questions.md`).
                erreur = "configuration: " + erreur
            elif trace.get("genre_erreur") == "quota_journalier":
                # The provider QUALIFIED it itself (429 "per day"): we no longer guess
                # from the text. Channel added on 2026-09-08 — the message rephrased by the
                # worker ("Providers saturés ou indisponibles") fell into _RE_OCCUPEE and
                # the run waited indefinitely for a key closed for 7 h. The regexes
                # remain the fallback for a provider that does not fill the field, and the
                # `satur` → busy boundary (decision of 2026-09-07) is not touched.
                if self.moniteur is not None:
                    self.moniteur.compteurs["429"] += 1
                self.derniere_erreur_quota = erreur
                erreur = "epuise: " + erreur
            elif trace.get("genre_erreur") == "surcharge_fournisseur":
                # 2026-09-25 — overload QUALIFIED by the gateway (5xx or per-minute 429 on
                # all admitted instances): transient, never marked exhausted (R2). The bucket is
                # the one the text already put it in, but without depending on its wording.
                erreur = "passerelle_occupee: " + erreur
            elif _RE_CREDIT.search(erreur) or _RE_QUOTA.search(erreur):
                # CONFIRMED quota (429) or credit (402): real blocker → `epuise` (R2).
                if self.moniteur is not None:
                    self.moniteur.compteurs[
                        "402" if _RE_CREDIT.search(erreur) else "429"
                    ] += 1
                self.derniere_erreur_quota = erreur
                erreur = "epuise: " + erreur
            elif _RE_OCCUPEE.search(erreur):
                # Gateway busy / unavailable / timeout: TRANSIENT, never `epuise` (R2).
                erreur = "passerelle_occupee: " + erreur
            return ReponseDecideur(
                index=None,
                fournisseur=fournisseur,
                erreur=erreur,
                presente={"payload": trace.get("payload")},
                reponse_brute=trace.get("reponse_brute"),
                reprise_a=(
                    reprise_a.isoformat()
                    if hasattr(reprise_a, "isoformat")
                    else reprise_a
                ),
                jour_meteo_tire=trace.get("jour_meteo_tire"),
                jour_meteo_lu=trace.get("jour_meteo_lu"),
                source_date_meteo=trace.get("source_date_meteo"),
            )
        return ReponseDecideur(
            index=idx,
            fournisseur=fournisseur,
            distribution=dict(distribution or {}),
            poids=list(trace.get("poids_presentes") or []),
            reponse_brute=trace.get("reponse_brute"),
            raison=raison or "",
            souvenirs=list(trace.get("souvenirs") or []),
            presente={"payload": trace.get("payload")},
            repli_uniforme=bool(trace.get("repli_uniforme")),
            identifiant_lot=trace.get("identifiant_lot"),
            jour_meteo_tire=trace.get("jour_meteo_tire"),
            jour_meteo_lu=trace.get("jour_meteo_lu"),
            source_date_meteo=trace.get("source_date_meteo"),
        )


def construire_decideur(
    spec,
    *,
    agent=None,
    moniteur: MoniteurRessources | None = None,
    instances: list[str] | None = None,
    execution_rejeu=None,
    dossier_echanges=None,
    attente_max_s: int = 120,
    execution=None,
    gabarit=None,
):
    """Build from `DecideurSpec` (experience.py)."""
    if spec.type == "duree_minimale":
        return DecideurDureeMinimale()
    if spec.type == "aleatoire":
        return DecideurAleatoire(spec.graine)
    if spec.type == "majoritaire_voiture":
        return DecideurMajoritaireVoiture()
    if spec.type == "rejeu":
        if execution_rejeu is None:
            from experiences.archive import Execution

            execution_rejeu = Execution.ouvrir(spec.rejeu_de)
        return DecideurRejeu(execution_rejeu, spec.rejeu_de)
    if spec.type == "passerelle":
        if agent is None:
            raise ValueError("a gateway decision-maker requires an LlmAgent")
        return DecideurPasserelle(agent, spec.modele, instances or [], moniteur)
    if spec.type == "antigravity":
        from experiences.decideur_antigravity import DecideurAntigravity

        if agent is None:
            raise ValueError(
                "an antigravity decision-maker requires an LlmAgent (payload construction)"
            )
        return DecideurAntigravity(
            agent=agent,
            modele=spec.modele,
            echanges=dossier_echanges,
            attente_max_s=attente_max_s,
            parametres=spec.parametres,
            execution=execution,
            # Not exposed in experience.yaml: the startup delay follows that of a
            # trip. A channel that served NO decision within `attente_max_s` is not
            # slow, it is dead.
            demarrage_max_s=None,
        )
    if spec.type == "typesafe":
        from experiences.decideur_typesafe import DecideurTypesafe

        if agent is None:
            raise ValueError(
                "a typesafe decision-maker requires an LlmAgent: the presentation served to "
                "Jev comes from the SAME `build_travel_plan_payload` as the LLM arms, it is not "
                "reimplemented"
            )
        return DecideurTypesafe(
            agent=agent,
            modele=spec.modele,
            variante=getattr(gabarit, "variante", None),
            categorie=getattr(gabarit, "categorie", "itinary_multi_agent"),
        )
    if spec.type == "modele":
        # Late import: LightGBM / geopandas are only loaded when deciding by model.
        from experiences.decideur_modele import DecideurModele

        return DecideurModele(artefact=getattr(spec, "artefact", None))
    raise ValueError(f"unknown decision-maker type: {spec.type!r}")


def __getattr__(name: str):
    if name == "DecideurTypesafe":
        from experiences.decideur_typesafe import DecideurTypesafe

        return DecideurTypesafe
    if name == "DecideurAntigravity":
        from experiences.decideur_antigravity import DecideurAntigravity

        return DecideurAntigravity
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DecideurAleatoire",
    "DecideurAntigravity",
    "DecideurDureeMinimale",
    "DecideurMajoritaireVoiture",
    "DecideurPasserelle",
    "DecideurRejeu",
    "DecideurTypesafe",
    "construire_decideur",
]
