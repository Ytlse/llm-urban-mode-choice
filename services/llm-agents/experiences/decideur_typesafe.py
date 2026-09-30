"""The "Jev" decision-maker (TypeSafe) — ticket 096, lot 1, contract `specs/ticket_096/tests.md`.

One more decision-maker under the common contract `Decideur.choisir(person, ctx, presentees)`,
but of a third family: neither a generative language model, nor a tabular model trained on the
survey. Jev is a **zero-shot classifier with typed output** — we send it a `state` and a `Choice`
question, it returns the full distribution over the options and a confidence, without producing
a line of text.

**Nothing of the presentation is rewritten.** The served text comes from
`LlmAgent.build_travel_plan_payload`, the very one that serves the LLM arms: a second
implementation of the persona block would make the two arms diverge silently, and that is exactly
what "equal presented text" forbids. What changes is the FORM of the request: the
options leave the text for `criteria`, and the instruction loses its `[Output instructions]` block
since the `Choice` type replaces it.

Three things this decision-maker does NOT do, and they are decisions:

- it does not **write** a `raison`. Jev does not produce one; a justification synthesised from
  the probabilities would read like a model output in the traces and the memory
  reports;
- it does not **round away** the rounding problem. Jev returns its probabilities to two decimals,
  so a sum of 0.99 or 1.01 is normal (measured at lot 0: ten cases out of 850, all at ±0.01
  exactly). The tolerance is declared, the weights are renormalised, and beyond it this is a
  non-decision;
- it **never falls back silently**. A transport failure is an error (the runner retries);
  a response returned but unusable is an explicit non-decision (archived, counted,
  excluded from the shares). The two cases are distinct and must not be confused: one is
  retried indefinitely, the other never.
"""

from __future__ import annotations

import hashlib
import json
import random
import re

from experiences.decision import (
    ContexteDecision,
    Proposition,
    ReponseDecideur,
    graine_ordre,
)
from loguru import logger
from models import Person

# Pinned version, never an alias: `jev-latest` would make the fingerprint lie at the first
# version change, SILENTLY. The spec refuses the alias rather than resolving it at
# launch — a resolved alias seals a version that the `experience.yaml` does not carry.
RE_VERSION_FIGEE = re.compile(r"^jev-\d+\.\d+\.\d+$")

# The marker that separates, in a `prompts.yaml` variant, the arbitration (which we send) from the
# output instruction (which the `Choice` type replaces: JSON schema, sum to 100, "no markdown").
MARQUEUR_SORTIE = "[Output instructions]"

# Jev returns its probabilities rounded to 2 decimals: with 6 options, the sum can be 0.99
# or 1.01. Measured at lot 0 over 850 calls — ten deviations, all exactly ±0.01, never anything
# else. The tolerance covers rounding and nothing more: a sum of 0.80 is an unusable
# response, not a rounding.
TOLERANCE_SOMME = 0.02


def instructions_servies(texte: str) -> str:
    """The text of a variant stripped of its output block — what is actually sent as
    `instructions`. A separate function: the fingerprint needs it without the decision-maker."""
    coupe = texte.find(MARQUEUR_SORTIE)
    return (texte[:coupe] if coupe > 0 else texte).strip()


def sha_instructions(categorie: str, variante: str | None) -> str | None:
    """sha256 of the text ACTUALLY sent to Jev (R6 of ticket 045, applied to this arm).

    The template fingerprint hashes the WHOLE variant; what we send is derived from it. If
    the stripping rule changed, the template fingerprint would not move and two incomparable
    runs would carry the same signature. This sha does move.
    """
    try:
        from mobility_llm import prompt_manager as get_prompt_manager

        texte = (
            get_prompt_manager().get_system_prompt(
                categorie, variante, verifier_validite=False
            )
            or ""
        )
    except Exception as e:  # noqa: BLE001 — the fingerprint SAYS so rather than disappearing
        logger.warning(
            f"[decideur] prompt système indisponible pour l'empreinte typesafe "
            f"(catégorie {categorie!r}, variante {variante!r}) : {e}"
        )
        return None
    return hashlib.sha256(instructions_servies(texte).encode("utf-8")).hexdigest()


def _etat(agent_bloc: dict) -> str:
    """The persona block of the payload, WITHOUT the options (they go into `criteria`).

    The fields and their order follow `itinary_multi_agent/template.md.j2`. The options are
    removed because a `Choice` already carries them: leaving them in the `state` would present them
    twice, and Jev's documentation announces a drop in accuracy when the `state`
    grows with content that does not serve the decision.
    """
    lignes = [str(agent_bloc.get("perception") or "")]
    dest = agent_bloc.get("destination") or ""
    zone = agent_bloc.get("destination_zone")
    lignes.append(f"Destination: {dest}{f' ({zone})' if zone else ''}")
    if agent_bloc.get("departure_time"):
        lignes.append(f"Departure: {agent_bloc['departure_time']}")
    if agent_bloc.get("context") and agent_bloc["context"] != "None":
        lignes.append(f"Context: {agent_bloc['context']}")
    if agent_bloc.get("day_outlook"):
        lignes.append(f"Weather later: {agent_bloc['day_outlook']}")
    if agent_bloc.get("agenda"):
        lignes.append("Further trips planned today:")
        lignes += [f"  - {l}" for l in agent_bloc["agenda"]]
    if agent_bloc.get("history"):
        lignes.append("History:")
        lignes += [f"  - {h}" for h in agent_bloc["history"]]
    return "\n".join(lignes)


def cle_option(index: int) -> str:
    """`option_<presentation rank>`. Indexed and NOT named by mode: two itineraries
    often share the same mode (two bus variants, two walking variants), and a
    mode-keyed dictionary would overwrite one."""
    return f"option_{index}"


def _criteres(trajectories: list[dict]) -> dict[str, str]:
    out: dict[str, str] = {}
    for i, t in enumerate(trajectories):
        desc = str(t.get("description") or "").replace("\n- ", " · ").replace("\n", " ")
        out[cle_option(i)] = f"Mode {t.get('mode') or 'unknown'}. {desc}".strip()
    return out


class DecideurTypesafe:
    """Jev as decision-maker. No quota, no reserved key, no text generation."""

    # Neither an API key to reserve nor a daily window: 1,200 req/min announced for 2,482
    # calls. The scheduler therefore has nothing to arbitrate (E5).
    sans_quota = True

    def __init__(
        self,
        agent,
        modele: str,
        variante: str | None = None,
        categorie: str = "itinary_multi_agent",
        client=None,
    ):
        if not RE_VERSION_FIGEE.match(str(modele or "")):
            raise ValueError(
                f"typesafe decision-maker: {modele!r} is not a pinned version. "
                "Expected `jev-<major>.<minor>.<patch>` (e.g. `jev-1.13.0`); the aliases "
                "`jev-latest` and `jev-preview` are refused, they would make the fingerprint "
                "lie at the first version change."
            )
        self.agent = agent
        self.modele = str(modele)
        self.variante = variante
        self.categorie = categorie
        self.nom = f"typesafe:{self.modele}"
        self._client = client
        self._instructions: str | None = None

    # ── resources ───────────────────────────────────────────────────────────
    def client(self):
        """**Asynchronous** TypeSafe client, built at the first decision.

        Asynchronous and not synchronous, and that is anything but a detail: the runner advances
        `regroupement.parallelisme` persons abreast behind an `asyncio.Semaphore`. A BLOCKING
        HTTP call inside an `async def` blocks the whole loop — the eight semaphore slots
        become useless and everything serialises on the network. Measured on 2026-09-21: 40
        decisions/minute with the synchronous client, where Jev answers in 300 ms. The result was
        identical, only the time changed; it is exactly the kind of defect that no contract
        test catches and that one pays for in hours.

        Late import: the SDK is only loaded when deciding by Jev. The key follows the repository
        convention (`PROVIDER_KEYS__<instance>`) and is passed EXPLICITLY — letting the SDK
        resolve its own `TYPESAFE_API_KEY` would serve a key lying around in the
        environment without the fingerprint knowing.
        """
        if self._client is None:
            import os

            from typesafe_sdk import AsyncTypeSafeClient

            cle = os.environ.get("PROVIDER_KEYS__typesafeAI")
            if not cle:
                raise RuntimeError(
                    "typesafe decision-maker: PROVIDER_KEYS__typesafeAI missing from the "
                    "environment — the run does not start rather than decide without a model."
                )
            self._client = AsyncTypeSafeClient(api_key=cle)
        return self._client

    def instructions(self) -> str:
        if self._instructions is None:
            from mobility_llm import prompt_manager as get_prompt_manager

            texte = (
                get_prompt_manager().get_system_prompt(
                    self.categorie, self.variante, verifier_validite=False
                )
                or ""
            )
            self._instructions = instructions_servies(texte)
            logger.info(
                f"[decideur] Jev {self.modele} — instruction \"{self.variante or 'active'}\", "
                f"{len(self._instructions)} characters served as `instructions` "
                f"(output block removed: the Choice type replaces it)"
            )
        return self._instructions

    def empreinte(self) -> dict:
        return {
            "type": "typesafe",
            "modele": self.modele,
            "variante": self.variante,
            "instructions_sha256": sha_instructions(self.categorie, self.variante),
        }

    # ── decision ────────────────────────────────────────────────────────────
    async def choisir(
        self, person: Person, ctx: ContexteDecision, presentees: list[Proposition]
    ) -> ReponseDecideur:
        from experiences.decideurs import _distribution, _presente_local
        from urban_mobility_agents.agents.llm_agent import Context

        options = [p.plan for p in presentees]
        for o in options:
            o.purpose = ctx.purpose
        contexte = Context(
            person=person,
            timestamp=int(ctx.timestamp),
            activity_id=ctx.activity_id,
            data={"type": "travel_plan"},
        )
        # The SAME payload builder as the LLM arms (T1): the presentation is not
        # reimplemented, otherwise the two arms drift without anything signalling it.
        # Ticket 111: the guaranteed line (article read, household message) applies to all
        # decision-makers, otherwise this arm would be deprived of what the LLM arms read. Empty
        # outside the setup.
        from llm import evenements as _evenements

        _lignes = await _evenements.lignes_du_jour(
            person.person_id, int(ctx.departure_time or ctx.timestamp)
        )
        payload = await self.agent.build_travel_plan_payload(
            context=contexte,
            options=options,
            destination=ctx.purpose,
            departure_time=int(ctx.departure_time),
            anticipation=ctx.anticipation,
            lignes=_lignes,
        )
        if _lignes:
            _evenements.noter_rendu(
                person.person_id, int(ctx.departure_time or ctx.timestamp), _lignes,
                (payload.get("agents") or [{}])[0].get("history", []),
            )
        bloc = (payload.get("agents") or [{}])[0]
        etat = _etat(bloc)
        criteres = _criteres(bloc.get("trajectories") or [])

        from typesafe_sdk import Choice

        try:
            reponse = await self.client().system_one(
                model=self.modele,
                state=etat,
                questions={"mode": Choice(instructions=self.instructions(), criteria=criteres)},
            )
        except Exception as e:  # noqa: BLE001 — la nature de l'échec décide de la suite
            return self._echec_transport(e, presentees)

        rep = reponse.answers["mode"]
        proba = dict(rep.probabilities or {})
        attendues = list(criteres)

        # R4 — a missing key cannot be recovered: we do not know what Jev would have put.
        manquantes = [c for c in attendues if c not in proba]
        if manquantes:
            return self._non_imputable(
                f"cles_manquantes:{','.join(manquantes[:3])}", presentees
            )

        poids = [float(proba[c]) for c in attendues]
        total = sum(poids)
        # R2/R3 — rounding to two decimals is accepted and renormalised; beyond that, it is no
        # longer rounding and we do not guess what the model meant.
        if abs(total - 1.0) > TOLERANCE_SOMME:
            return self._non_imputable(f"somme_hors_tolerance:{total:.4f}", presentees)
        if total <= 0:
            return self._non_imputable("poids_nuls", presentees)
        poids = [w / total for w in poids]

        idx = _tirer(
            poids, graine_ordre(ctx.graine_tirage, person.person_id, ctx.activity_id)
        )
        return ReponseDecideur(
            index=idx,
            fournisseur=self.nom,
            distribution=_distribution(poids, presentees),
            poids=poids,
            # R8 — the WHOLE response is archived: this is what lets `DecideurRejeu`
            # keep the arm if the service disappears (§ 6.2 of the ticket).
            reponse_brute=json.dumps(
                {
                    "modele": reponse.model,
                    "choice": rep.choice,
                    "probabilities": proba,
                    "confidence": rep.confidence,
                    "input_tokens": getattr(reponse.usage, "input_tokens", None),
                },
                ensure_ascii=False,
            ),
            # R7 — EMPTY. Jev does not write, and a sentence made up from the probabilities
            # would read like a model output.
            raison="",
            presente=_presente_local(presentees),
        )

    # ── failures ────────────────────────────────────────────────────────────
    def _echec_transport(self, e: Exception, presentees: list[Proposition]) -> ReponseDecideur:
        """E1/E2/E3 — a transport failure is an ERROR (retried), never a non-decision.

        Both prefixes already exist in the runner: `passerelle_occupee` waits and
        retries, `configuration` (ticket 085) says there is a configuration line to
        fix and not a quota to wait for. No error kind is added here.
        """
        nom = type(e).__name__
        if nom in ("TypeSafeAuthenticationError", "TypeSafePermissionDeniedError",
                   "TypeSafeNotFoundError", "TypeSafeUnprocessableEntityError",
                   "TypeSafeBadRequestError"):
            prefixe = "configuration"
        else:
            # Rate, overload, timeout, network outage: TRANSIENT. Never `epuise` —
            # this arm has no daily quota window to wait for.
            prefixe = "passerelle_occupee"
        logger.debug(f"[decideur] Jev {prefixe} — {nom}: {e}")
        return ReponseDecideur(
            index=None,
            fournisseur=self.nom,
            erreur=f"{prefixe}: {nom}: {e}",
            presente=self._presente(presentees),
        )

    def _non_imputable(self, raison: str, presentees: list[Proposition]) -> ReponseDecideur:
        """E4 — response returned but unusable: TERMINAL non-decision, never retried."""
        logger.warning(f"[decideur] Jev non-attributable ({raison}) — non-decision")
        return ReponseDecideur(
            index=None,
            fournisseur=self.nom,
            non_imputable=True,
            raison=f"typesafe_non_imputable:{raison}",
            presente=self._presente(presentees),
        )

    @staticmethod
    def _presente(presentees: list[Proposition]) -> dict:
        # Late import, as in `decideur_modele`: `decideurs` imports this module.
        from experiences.decideurs import _presente_local

        return _presente_local(presentees)


def _tirer(poids: list[float], graine: int) -> int:
    """Draw an index proportionally to the weights. IDENTICAL to `DecideurModele._tirer`:
    same seed, same draw — this is what makes two decision-makers comparable trip by
    trip (R6)."""
    r = random.Random(graine).random() * sum(poids)
    cumul = 0.0
    for i, w in enumerate(poids):
        cumul += w
        if r <= cumul:
            return i
    return len(poids) - 1


__all__ = [
    "DecideurTypesafe",
    "cle_option",
    "instructions_servies",
    "sha_instructions",
    "RE_VERSION_FIGEE",
    "TOLERANCE_SOMME",
]
