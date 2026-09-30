"""Fitness of a model to carry an experiment — refusal up front (hygiene spec §6, P3).

`estimer()` already costed the forecast; it refused nothing. Five runs
stopped midway for lack of quota or of a reachable model, and had to be archived without
having measured anything. This module turns the costing into a verdict.

Two levels, and the distinction is deliberate:

- **refusal** — the run CANNOT complete: a single agent would exceed the provider's
  per-request cap (every call would be truncated), or the thinking settings
  would fail with 400. No patience setting would change anything.
- **warning** — it will complete, but slowly: the tokens-per-minute cap bounds the
  throughput below the declared `rpm_limit`. We state the implied duration and let the user decide.

Two units, never to be confused: a **solicitation** is a trip to decide, a
**request** is a provider call, and the gateway merges several agents per call.
Quotas are counted in requests; `experiences/lots.py` provides the divisor and
`estimer()` applies it. Without a measurement, this divisor is 1 and we fall back on the
pre-2026-09-22 assumption: caution cannot decrease.

What this module does NOT know: the limits that `providers.yaml` does not declare. The OTPM
(OUTPUT tokens per minute) of the Groq gateways is the known case — 1,000 measured on
2026-09-08, invisible outside the body of the 429s, absent from the file. A provider can therefore
pass this check and still throttle: that is what P4 (health probe) must fix.
"""

from __future__ import annotations

import math
from typing import Any

# Safety margin on the daily quota: a run that would consume exactly the
# quota down to the token fails on the first retried error.
MARGE_QUOTA = 1.05

# A daily quota below the load is NOT an impossibility: a run resumes
# to the exact trip (S9) and can therefore spread over several quota windows.
# This is measured: runs on `gemini-3.5-flash-lite` completed although the declared
# `rpd_limit` (2 × 500) is below their 2,048 solicitations — either the real limit is
# higher, or the file's RPD accounting is wrong (cf. P4, health probe).
# So we only refuse the absurd: beyond this factor, no reasonable spreading
# makes up the gap (20 requests/day for 2,285 solicitations = 114 days).
FACTEUR_QUOTA_ABSURDE = 10

# Beyond this, we warn about duration: a run of more than 6 h crosses a quota
# renewal window and goes on hold midway.
SEUIL_DUREE_AVERTISSEMENT_S = 6 * 3600


def _int_ou_none(v: Any) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _somme(providers: dict[str, dict], instances: list[str], cle: str) -> int | None:
    """Sum of a limit over the instances that declare it. None if none declares it."""
    vals = [
        _int_ou_none((providers.get(i) or {}).get(cle))
        for i in instances
    ]
    connus = [v for v in vals if v is not None]
    return sum(connus) if connus else None


def verifier(
    *,
    modele: str | None,
    sollicitations: int,
    jetons_entree: int | None,
    jetons_sortie: int | None,
    max_tokens_demande: int | None,
    providers: dict[str, dict],
    instances: list[str],
    reflexion_demandee: int | None = None,
    niveau_demande: str | None = None,
    requetes: int | None = None,
    regroupement: dict | None = None,
) -> tuple[list[str], list[str]]:
    """(refusals, warnings) for a given model against a given load.

    `jetons_entree` / `jetons_sortie`: per solicitation, i.e. per AGENT. `None` when
    no measurement exists — we then do NOT refuse on tokens: refusing on an unknown
    value would amount to blocking any never-measured model, hence any new model.

    `sollicitations` counts TRIPS; `requetes` counts PROVIDER CALLS, which
    group several of them (gateway micro-batching). An `rpd_limit` quota is compared to the
    latter, never to the former: the gateway groups eight per request, and confusing them
    made arms that had ample room cry short quota (ticket 073 § 4). `requetes=None` falls back on
    `sollicitations`, i.e. on the most cautious assumption — no grouping.
    """
    refus: list[str] = []
    avert: list[str] = []
    if not instances:
        return refus, avert  # a missing instance is already refused upstream

    par_agent = (jetons_entree or 0) + (jetons_sortie or 0)
    appels = int(requetes) if requetes else int(sollicitations)
    facteur = (regroupement or {}).get("prudent") or (
        sollicitations / appels if appels else 1
    )
    dit_regroupement = (
        f" (regroupement retenu {float(facteur):.2f} agent(s)/requête ; "
        f"{(regroupement or {}).get('source', 'aucune mesure : une requête par déplacement')})"
    )
    # What the arm will likely cost, next to what we retain to decide. Without a
    # usable archived run, the cautious figure equals the number of trips — stating
    # the expected value next to it avoids reading a quota warning without knowing it is built on
    # the least favourable assumption, and that the derived cap promises eight times less.
    attendu = (regroupement or {}).get("attendu")
    dit_attendu = ""
    if attendu and float(attendu) > float(facteur):
        appels_attendus = max(1, math.ceil(sollicitations / float(attendu)))
        dit_attendu = (
            f" — attendu plutôt ~{appels_attendus} requêtes à {float(attendu):.2f} "
            f"agent(s)/requête, mais aucune mesure ne le garantit encore"
        )
    requis = int(appels * MARGE_QUOTA)

    # 1. Daily quota. Only a warning: a run resumes and can
    # spread over several quota windows. The declarative limits of providers.yaml
    # or the raw trip estimates must never block a legitimate launch.
    rpd = _somme(providers, instances, "rpd_limit")
    if rpd is not None and rpd > 0 and rpd < requis:
        jours = appels / rpd
        avert.append(
            f"quota journalier plus court que la charge pour {modele!r} : {rpd} "
            f"requêtes/jour déclarées contre ~{appels} requêtes pour {sollicitations} "
            f"déplacements{dit_regroupement} (~{jours:.1f} jours) → l'exécution s'étalera sur "
            f"plusieurs fenêtres, avec reprise. Les limites de providers.yaml sont déclaratives "
            f"et connues comme parfois fausses : ce n'est pas un refus{dit_attendu}"
        )

    # 2. Per-request cap — every call would be truncated.
    #
    # Compared to ONE agent, not to the batch: an oversized batch is not a refusal, the gateway
    # shrinks it by itself (`max_tokens_per_request` bounds `batch_max_agents`, cf.
    # `llm_gateway.core.batching.compute_batch_max_agents`). What is structural is the single
    # agent that does not fit in a request, and no patience setting can make up for it.
    if par_agent:
        for i in instances:
            plafond = _int_ou_none((providers.get(i) or {}).get("max_tokens_per_request"))
            if plafond is not None and plafond < par_agent:
                refus.append(
                    f"{i} plafonne à {plafond} jetons par requête, or UNE sollicitation en "
                    f"demande ~{par_agent} (entrée {jetons_entree} + sortie {jetons_sortie}) "
                    f"→ chaque appel serait tronqué, même sans regroupement ; relevez "
                    f"max_tokens_per_request ou changez d'instance"
                )
    if max_tokens_demande:
        for i in instances:
            sortie_max = _int_ou_none((providers.get(i) or {}).get("max_output_tokens"))
            if sortie_max is not None and sortie_max < max_tokens_demande:
                refus.append(
                    f"{i} plafonne la sortie à {sortie_max} jetons, or l'expérience demande "
                    f"max_tokens={max_tokens_demande} → abaissez max_tokens ou changez d'instance"
                )

    # 2 bis-a. Both thinking settings together: the API returns 400.
    if niveau_demande and reflexion_demandee is not None:
        refus.append(
            f"thinking_level={niveau_demande!r} et thinking_budget={reflexion_demandee} "
            "demandés ensemble : l'API les refuse conjointement (400) → n'en garder qu'un, "
            "`thinking_level` étant le réglage courant"
        )

    # 2 bis-b. Thinking level not accepted by the model — 400 on every call.
    if niveau_demande:
        for i in instances:
            connus = (providers.get(i) or {}).get("thinking_levels")
            if connus and niveau_demande not in connus:
                refus.append(
                    f"niveau de réflexion {niveau_demande!r} non accepté par {i} "
                    f"(déclarés : {', '.join(connus)}) → chaque appel partirait en 400"
                )
        if not any((providers.get(i) or {}).get("thinking_levels") for i in instances):
            avert.append(
                f"niveau de réflexion {niveau_demande!r} demandé, mais aucun "
                f"`thinking_levels` n'est déclaré pour {modele!r} : impossible de vérifier "
                f"que le modèle l'accepte"
            )

    # 2 bis. Thinking depth beyond the declared cap — refusal, because the
    # provider would trim it silently and the fingerprint would carry an unapplied budget.
    if reflexion_demandee is not None and reflexion_demandee > 0:
        for i in instances:
            plafond = _int_ou_none((providers.get(i) or {}).get("thinking_budget_max"))
            if plafond is not None and reflexion_demandee > plafond:
                refus.append(
                    f"profondeur de réflexion {reflexion_demandee} au-delà du plafond déclaré "
                    f"{plafond} de {i} → le fournisseur la raboterait sans le signaler et "
                    f"l'empreinte porterait un budget non appliqué ; abaissez thinking_budget "
                    f"ou corrigez thinking_budget_max"
                )
        if not any((providers.get(i) or {}).get("thinking_budget_max") for i in instances):
            avert.append(
                f"profondeur de réflexion {reflexion_demandee} demandée, mais aucun "
                f"`thinking_budget_max` n'est déclaré pour {modele!r} : impossible de vérifier "
                f"qu'elle sera appliquée telle quelle"
            )

    # 3. Throughput — the run completes, but the tokens-per-minute cap bounds the pace.
    rpm = _somme(providers, instances, "rpm_limit")
    tpm = _somme(providers, instances, "tpm_limit")
    if rpm and par_agent:
        rpm_effectif = rpm
        # The tokens of ONE request are those of all the agents it carries. The bound set by
        # the TPM is therefore insensitive to grouping (n·jetons_agent ÷ tpm on both sides);
        # the one set by the RPM is divided accordingly. That is the reason for the distinction.
        jetons_par_appel = max(1, int(par_agent * float(facteur)))
        if tpm:
            rpm_par_tpm = tpm / jetons_par_appel
            if rpm_par_tpm < rpm:
                rpm_effectif = rpm_par_tpm
                avert.append(
                    f"débit bridé par les jetons : tpm cumulé {tpm} ÷ ~{jetons_par_appel} jetons "
                    f"par requête ({par_agent} par agent × {float(facteur):.2f}) = "
                    f"{rpm_par_tpm:.1f} requêtes/min effectives, contre rpm_limit {rpm} déclaré"
                )
        duree = appels / rpm_effectif * 60
        if duree > SEUIL_DUREE_AVERTISSEMENT_S:
            avert.append(
                f"durée estimée ~{duree / 3600:.1f} h à {rpm_effectif:.1f} requêtes/min : "
                f"l'exécution traversera une fenêtre de renouvellement de quota et se mettra "
                f"en attente en cours de route"
            )
    return refus, avert


def verifier_depuis_estimation(
    exp, est: dict, providers: dict[str, dict], instances: list[str]
) -> tuple[list[str], list[str]]:
    """Adapter: reads the output of `experience.estimer()` rather than its inputs."""
    deplacements = (
        (est.get("deplacements") or {}).get("valeur")
        or (est.get("sollicitations") or {}).get("valeur")
        or 0
    )
    j = est.get("jetons") or {}
    par = j.get("par_sollicitation") or {}
    # `prudente` is the figure that decides (cf. `lots.facteurs`): without a measurement it
    # equals the number of trips, so the verdict cannot get more permissive than before the fix.
    appels = (est.get("requetes") or {}).get("prudente")
    return verifier(
        modele=exp.decideur.modele,
        sollicitations=int(deplacements),
        requetes=int(appels) if appels else None,
        regroupement=est.get("regroupement"),
        jetons_entree=_int_ou_none(par.get("entree")),
        jetons_sortie=_int_ou_none(par.get("sortie")),
        max_tokens_demande=_int_ou_none((exp.decideur.parametres or {}).get("max_tokens")),
        providers=providers,
        instances=instances,
        reflexion_demandee=_int_ou_none((exp.decideur.parametres or {}).get("thinking_budget")),
        niveau_demande=(exp.decideur.parametres or {}).get("thinking_level"),
    )


__all__ = ["FACTEUR_QUOTA_ABSURDE",
           "MARGE_QUOTA", "SEUIL_DUREE_AVERTISSEMENT_S", "verifier", "verifier_depuis_estimation"]
