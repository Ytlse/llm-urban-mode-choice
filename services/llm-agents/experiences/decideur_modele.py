"""Statistical-model decision-maker (LightGBM, multinomial logit or KLR) — spec R11..R15.

One more decision-maker, under the same contract as the LLM gateway
(`Decideur.choisir(person, ctx, presentees)`): it does not score in parallel, it
**decides**. An experiment can therefore be rerun (with or without the simulator) by
swapping only the decision-maker, and the model version is sealed by SHA
in the run fingerprint (R12).

Nothing of the prediction is rewritten: `persona_features`, `renormalize`,
`load_policy` and `predict` are imported from `scripts/synthesis/model_on_common_set.py`,
and the encoding comes from the training script. It is the only way not to
introduce a silent shift between training and decision.

**Non-attributable** cases (RG-2, R13) — persona without traits, no predictable mode
on offer, OD outside the zone layer: the decision-maker returns an **explicit
non-decision** (`non_imputable`), never a silent fallback to a mode.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

from experiences.chemins import racine_depot
from experiences.decision import (
    ContexteDecision,
    Proposition,
    ReponseDecideur,
    graine_ordre,
)
from loguru import logger
from models import Person

REPO_ROOT = racine_depot()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.synthesis.model_on_common_set import (
    CANONICAL_TO_CAT,
    PREDICTABLE_CATS,
    STATUS_OK,
    load_policy,
    persona_features,
    predict,
)

# Default artefact paths (same as scripts/synthesis/sources.yaml).
POLICY_DEFAUT = REPO_ROOT / "scripts" / "progedo_logit" / "mode_choice_policy.json"
SPEC_DEFAUT = REPO_ROOT / "scripts" / "progedo_logit" / "feature_spec.json"

# Artefact format → family name, for the trace labels.
#
# **Why a table and not a constant.** Two model families go through this
# decision-maker since ticket 042 — the supervised booster and the strict-parity multinomial
# logit — and they take the SAME decision path. As long as the decision-maker name,
# the fingerprint and the raw response carried a hard-coded "lightgbm", a logit run
# announced itself as LightGBM: the artefact SHA stayed right (R12), but any comparison
# read from the traces designated the wrong model. A wrong label is worse than a missing
# label, because it goes unnoticed.
FAMILLES = {
    "lightgbm_mode_choice_policy": "lightgbm",
    "mnl_mode_choice_policy": "mnl",
    # Ticket 043: kernel logistic regression joins the table with no other change — that is
    # the whole point of a label DERIVED from the format rather than hard-coded.
    "klr_mode_choice_policy": "klr",
    # Ticket 088 § 3.3: the random forest joins the table. It was kept out of it
    # because ticket 043 was modifying these same lines in parallel; it is closed. Its
    # launcher used to register the family FOR THE LIFETIME OF ITS PROCESS and replaced
    # `load_policy` in the decision-maker's namespace — a switch invisible from this
    # file, which made the experiment not replayable through the CLI.
    #
    # Rule R7 of ticket 044 ("a control does not become a referee") is not carried
    # by this table: what keeps the forest out of the oracles' composite score is
    # the arms of the manifest `scripts/synthesis/sources.yaml`, of which `bi_oracle.py` only
    # reads the three oracles lightgbm, mnl and klr — not the absence of a label here. The
    # label stays DERIVED from the artefact format, never hard-coded: that is what
    # prevents a forest run from announcing itself as "lightgbm" in the traces.
    #
    # The forest is refitted at load time (no tree is serialised) and checks that it
    # reproduces the published metrics to within 1e-9. It was estimated under scikit-learn
    # 1.8.0; the `controller` container carries 1.9.1, under which this check fails —
    # that is its job. This experiment is therefore launched from the HOST.
    "rf_mode_choice_policy": "rf",
}


def _sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _resoudre(chemin: str | Path) -> Path:
    """Artefact path, absolute as is, relative resolved from the repository root."""
    p = Path(chemin)
    return p if p.is_absolute() else REPO_ROOT / p


def _page_cat(mode: str) -> str:
    """Mode of a proposal → page category (via the canonical mode)."""
    from mobility_llm.mode_choice import canonical_mode

    return CANONICAL_TO_CAT.get(canonical_mode(mode), "other")


def _purpose_origin(person: Person, activity_id: str) -> str | None:
    """Purpose of the origin activity — the previous one in the persona's cyclic chain.

    Same logic as `model_on_common_set.activity_index`: the origin of the first
    trip is the last activity (cyclic chain). No invented value:
    activity not found → None (the contract will route the variable as missing)."""
    activities = getattr(person.identity, "activities", None) or []
    for i, act in enumerate(activities):
        if getattr(act, "id", None) == activity_id:
            origine = activities[i - 1]
            return getattr(origine, "purpose", None)
    return None


class DecideurModele:
    """LightGBM statistical policy as decision-maker. Deterministic, no quota."""

    sans_quota = True

    def __init__(
        self,
        artefact: str | Path | None = None,
        spec: str | Path | None = None,
        zones: str | Path | None = None,
    ):
        # A relative path is resolved from the repository root, not from the current
        # directory: the `experience.yaml` must designate the same artefact on the host and in
        # the container, and `empreinte()` needs a path relative TO THAT root (R12).
        self.policy_path = _resoudre(artefact) if artefact else POLICY_DEFAUT
        self.spec_path = Path(spec) if spec else SPEC_DEFAUT
        # R14 — refusal to start if a mandatory artefact is missing: no degraded
        # decision, the run does not begin.
        for label, p in (
            ("policy", self.policy_path),
            ("feature_spec", self.spec_path),
        ):
            if not p.exists():
                raise FileNotFoundError(
                    f"Model decision-maker: {label} not found ({p}). "
                    "Train the policy (make policy) before launching a model experiment."
                )
        self._spec = json.loads(self.spec_path.read_text(encoding="utf-8"))
        self._booster, self._artefact = load_policy(self.policy_path, self._spec)
        self.sha256 = _sha256(self.policy_path)
        self._resolver = self._charger_resolver(zones)
        self.format = self._artefact.get("format")
        # A format outside the table is refused rather than named "inconnu": the decision-maker
        # name enters the canonical experiment name and its traces.
        if self.format not in FAMILLES:
            raise ValueError(
                f"Model decision-maker: unrecognised artefact format ({self.format!r}). "
                f"Known formats: {sorted(FAMILLES)}."
            )
        self.famille = FAMILLES[self.format]
        self.nom = f"modele:{self.famille}@{self.sha256[:12]}"
        logger.info(
            f"[decideur] Model {self.famille} loaded ({self.format}) — spec "
            f"v{self._spec.get('spec_version')}, artefact {self.sha256[:12]}, "
            f"zone lookup {'prêt' if self._resolver is not None else 'ABSENT'}"
        )

    def _charger_resolver(self, zones: str | Path | None):
        # R14 — the zone layer is mandatory: the 6 geographic variables
        # (including od_km, the model's first) depend on it. Missing → refusal.
        from mobility_core.zone_resolver import ZoneResolver

        return ZoneResolver.load(
            Path(zones) if zones else None, feature_spec=self.spec_path
        )

    def empreinte(self) -> dict:
        """What seals the model version in the run fingerprint (R12)."""
        return {
            "type": "modele",
            "modele": self.format,
            "famille": self.famille,
            "artefact": str(self.policy_path.relative_to(REPO_ROOT)),
            "sha256": self.sha256,
            "spec_version": self._spec.get("spec_version"),
        }

    def _construire_ligne(
        self, person: Person, ctx: ContexteDecision, offered_predictable: list[str]
    ) -> dict | None:
        """A feature row for the policy, or None if the OD is outside the layer."""
        traits = getattr(person.identity, "traits_json", None) or {}
        origin = ctx.from_location
        dest = ctx.destination
        if origin is None or dest is None:
            return None
        geo = self._resolver.geo_features_many(
            [(float(origin.lat), float(origin.lon))],
            [(float(dest.lat), float(dest.lon))],
        )
        if not geo or geo[0] is None:
            return None  # OD outside the zone layer (R13)
        heure = datetime.fromtimestamp(int(ctx.departure_time), tz=timezone.utc).hour
        ligne = {
            "agent_id": person.person_id,
            "activity_id": ctx.activity_id,
            "offered_predictable": "|".join(offered_predictable),
            "status": STATUS_OK,
            "purpose": ctx.purpose,
            "purpose_origin": _purpose_origin(person, ctx.activity_id),
            "departure_hour": heure,
            **persona_features(traits),
            **geo[0].as_dict(),
        }
        return ligne

    async def choisir(
        self, person: Person, ctx: ContexteDecision, presentees: list[Proposition]
    ) -> ReponseDecideur:
        from experiences.decideurs import _distribution, _presente_local

        traits = getattr(person.identity, "traits_json", None) or {}
        if not traits:
            return self._non_imputable("persona_sans_traits", presentees)

        cats = [_page_cat(p.mode) for p in presentees]
        offered_predictable = sorted({c for c in cats if c in PREDICTABLE_CATS})
        if not offered_predictable:
            return self._non_imputable("offre_sans_mode_predictible", presentees)

        ligne = self._construire_ligne(person, ctx, offered_predictable)
        if ligne is None:
            return self._non_imputable("od_hors_couche_zones", presentees)

        predict([ligne], self._booster, self._spec)
        if ligne.get("status") != STATUS_OK:
            # The model gives no mass to what is offered: no distribution.
            return self._non_imputable("modele_sans_masse_offerte", presentees)

        # Renormalised mass per category → weight per presented proposal, by
        # splitting a category's mass equally among its proposals.
        par_cat = {c: cats.count(c) for c in offered_predictable}
        poids = [
            (ligne.get(f"p_{c}", 0.0) / par_cat[c]) if c in offered_predictable else 0.0
            for c in cats
        ]
        total = sum(poids)
        if total <= 0:
            return self._non_imputable("poids_nuls", presentees)
        poids = [w / total for w in poids]

        idx = self._tirer(
            poids, graine_ordre(ctx.graine_tirage, person.person_id, ctx.activity_id)
        )
        brut = {f"p_{c}": ligne.get(f"p_{c}") for c in PREDICTABLE_CATS}
        brut.update({f"p_raw_{c}": ligne.get(f"p_raw_{c}") for c in PREDICTABLE_CATS})
        return ReponseDecideur(
            index=idx,
            fournisseur=self.nom,
            distribution=_distribution(poids, presentees),
            poids=poids,
            reponse_brute=json.dumps(
                {"modele": self.famille, **brut}, ensure_ascii=False
            ),
            raison="modèle statistique (masse renormalisée sur l'offre)",
            presente=_presente_local(presentees),
        )

    @staticmethod
    def _tirer(poids: list[float], graine: int) -> int:
        """Draw an index proportionally to the weights (deterministic, like the LLM)."""
        r = random.Random(graine).random() * sum(poids)
        cumul = 0.0
        for i, w in enumerate(poids):
            cumul += w
            if r <= cumul:
                return i
        return len(poids) - 1

    def _non_imputable(
        self, raison: str, presentees: list[Proposition]
    ) -> ReponseDecideur:
        from experiences.decideurs import _presente_local

        logger.debug(f"[decideur] model non-attributable ({raison}) — non-decision")
        return ReponseDecideur(
            index=None,
            fournisseur=self.nom,
            non_imputable=True,
            raison=f"modele_non_imputable:{raison}",
            presente=_presente_local(presentees),
        )


__all__ = ["DecideurModele"]
