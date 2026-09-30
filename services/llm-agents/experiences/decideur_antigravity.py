"""Antigravity decision-maker (ticket 035, spec decideur-antigravity v2).

Delegates each trip decision to an Antigravity sub-agent via file-based IPC,
without consuming API quota and without degrading the guarantees of ticket 035.

The model served by Antigravity is declared but not verified (P1/P2):
`modele_verifie: false` is written in the decision-maker fingerprint and in each trace.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger
from models import Person

from experiences.archive import ETAT_EN_ATTENTE_AGENT, ETAT_EN_COURS
from experiences.decision import ContexteDecision, Proposition, ReponseDecideur
from mobility_llm import CATEGORIES, prompt_manager
from mobility_llm.mode_choice import (
    UniformFallback,
    draw_index,
    mode_distribution,
    normalize_option_probabilities,
)
from urban_mobility_agents.agents.llm_agent import Context, _format_distribution

from settings import settings


class DecideurAntigravity:
    """Decision-maker delegating the mode choice to an Antigravity sub-agent via on-disk IPC."""

    sans_quota = True  # runner.py:218 and 973 neutralise the quota block
    modele_verifie = False  # P1: declared model, unverifiable via the IDE runtime

    def __init__(
        self,
        agent,
        modele: str,
        echanges: Path | str | None = None,
        attente_max_s: int = 120,
        parametres: dict | None = None,
        execution=None,
        demarrage_max_s: float | None = None,
    ) -> None:
        self.agent = agent
        self.modele = str(modele)
        self.nom = f"antigravity:{self.modele}"
        self.attente_max_s = int(attente_max_s)
        self.parametres = dict(parametres or {})
        self.execution = execution
        # Cold-start safeguard (§4.5 bis). Measured on 2026-09-15: a channel without a
        # sub-agent leaves the run in `en_attente_agent` — a NON-final state — with 0 decisions
        # out of 3,161 for more than six hours, and the campaign blocked behind it. Past this
        # delay without the SLIGHTEST response, the run stops instead of sleeping all night.
        self.demarrage_max_s = float(
            demarrage_max_s if demarrage_max_s is not None else attente_max_s
        )

        if echanges is None:
            if execution is not None:
                echanges = execution.dossier / "echanges"
            else:
                echanges = Path("data/echanges_antigravity")
        self.echanges = Path(echanges)
        self.dossier_demandes = self.echanges / "demandes"
        self.dossier_demandes_traitees = self.dossier_demandes / "traitees"
        self.dossier_reponses = self.echanges / "reponses"

        self.dossier_demandes_traitees.mkdir(parents=True, exist_ok=True)
        self.dossier_reponses.mkdir(parents=True, exist_ok=True)

        self.compteurs: Counter = Counter()
        self._demandes_en_cours: dict[str, float] = {}
        self._dernier_log_status: float = time.time()
        self._alarme_timeout_levee: bool = False
        self._alarme_rejet_levee: bool = False
        self._alarme_litterale_levee: bool = False
        self._alarme_parametres_levee: bool = False
        self._debut: float = time.time()
        self._premiere_reponse: bool = False
        self._arret_demarrage_demande: bool = False

        # Cache of the category's output schema
        cat = CATEGORIES["itinary_multi_agent"]
        self._schema_sortie: dict[str, Any] = json.loads(
            cat.schema_path.read_text(encoding="utf-8")
        )

        logger.info(
            f"[antigravity] Initialisation — expected model: {self.modele}, "
            f"exchanges: {self.echanges}, max wait: {self.attente_max_s}s, "
            f"max startup: {self.demarrage_max_s:.0f}s, "
            f"parameters passed: {self.parametres or '—'}"
        )

    def _verifier_demarrage(self, now: float) -> None:
        """Stop the run if NO sub-agent has ever responded (§4.5 bis).

        The safeguard only applies to startup: as soon as the first response arrives, it is
        disarmed for good and the original behaviour (state `en_attente_agent`, per-trip
        timeout) takes over again. The stop goes through the STOP file of the run
        directory, an existing mechanism: the runner then closes cleanly in `arretee`, which is
        a FINAL state, so the campaign moves on instead of waiting.
        """
        if self._premiere_reponse or self._arret_demarrage_demande:
            return
        if now - self._debut < self.demarrage_max_s:
            return
        self._arret_demarrage_demande = True
        logger.error(
            f"[ALARME] [antigravity] No sub-agent has responded within "
            f"{self.demarrage_max_s:.0f}s since startup — 0 responses for "
            f"{len(self._demandes_en_cours)} pending request(s). Stopping the run: "
            f"an Antigravity channel without an agent does not unblock by itself. Check that an "
            f"Antigravity session is watching {self.dossier_demandes}."
        )
        if self.execution is None:
            return
        # Late import: `runner` imports `decideurs`, which imports this module. An import at
        # the top of the file would be circular.
        from experiences.runner import FICHIER_STOP

        try:
            (Path(self.execution.dossier) / FICHIER_STOP).write_text(
                "antigravity: aucun sous-agent n'a répondu au démarrage\n", encoding="utf-8"
            )
        except OSError as exc:  # pragma: no cover - dépend du système de fichiers
            logger.error(
                f"[antigravity] STOP file not written ({exc}): the run will not stop "
                "by itself."
            )

    def _verifier_journal_et_alarmes(self, now: float) -> None:
        """Every 30s: status and rising-edge alarms (§4.7)."""
        if now - self._dernier_log_status >= 30.0:
            age_max = (
                max(now - t for t in self._demandes_en_cours.values())
                if self._demandes_en_cours
                else 0.0
            )
            logger.info(
                f"[antigravity] Status — pending: {len(self._demandes_en_cours)}, "
                f"max age: {age_max:.1f}s, served: {self.compteurs['servies']}, "
                f"fallbacks: {self.compteurs['replis']}, rejections: {self.compteurs['rejets']}, "
                f"without literal output: {self.compteurs['sans_sortie_litterale']}, "
                f"without applied parameters: {self.compteurs['sans_parametres_appliques']}"
            )
            self._dernier_log_status = now

            if age_max >= self.attente_max_s and not self._alarme_timeout_levee:
                logger.error(
                    f"[ALARME] [antigravity] No response received for a request for "
                    f"more than {self.attente_max_s}s (age: {age_max:.1f}s)"
                )
                self._alarme_timeout_levee = True

            if self.compteurs["replis"] > 0 and not getattr(self, "_alarme_repli_levee", False):
                logger.error(
                    f"[ALARME] [antigravity] Uniform fallbacks detected ({self.compteurs['replis']}) — "
                    "an Antigravity experiment must keep strictly 0 fallbacks."
                )
                self._alarme_repli_levee = True

            total_traite = (
                self.compteurs["servies"]
                + self.compteurs["rejets"]
                + self.compteurs["timeouts"]
            )
            if total_traite >= 20 and not self._alarme_rejet_levee:
                taux_rejets = self.compteurs["rejets"] / total_traite
                if taux_rejets > 0.01:
                    logger.error(
                        f"[ALARME] [antigravity] High rejection rate "
                        f"({self.compteurs['rejets']}/{total_traite} = {taux_rejets:.1%}) — "
                        "check the wiring of the Antigravity agent"
                    )
                    self._alarme_rejet_levee = True

    async def choisir(
        self, person: Person, ctx: ContexteDecision, presentees: list[Proposition]
    ) -> ReponseDecideur:
        """Contract D1: builds the exact prompt, writes it for the agent, and awaits its response."""
        options = [p.plan for p in presentees]
        for o in options:
            o.purpose = ctx.purpose

        context = Context(
            person=person,
            timestamp=int(ctx.timestamp),
            activity_id=ctx.activity_id,
            data={"type": "travel_plan"},
        )

        # 1. In-process payload construction (pure builder)
        # Ticket 111: the guaranteed line applies to all decision-makers. Empty outside the setup.
        from llm import evenements as _evenements

        _lignes = await _evenements.lignes_du_jour(
            person.person_id, int(ctx.departure_time or ctx.timestamp)
        )
        payload = await self.agent.build_travel_plan_payload(
            context,
            options,
            ctx.purpose,
            int(ctx.departure_time),
            ctx.anticipation,
            lignes=_lignes,
        )
        if _lignes:
            _evenements.noter_rendu(
                person.person_id, int(ctx.departure_time or ctx.timestamp), _lignes,
                payload["agents"][0].get("history", []),
            )

        # 2. Exact in-process PromptEngine rendering (no network, no gateway)
        cat = CATEGORIES["itinary_multi_agent"]
        items = [cat.item_model(**a) for a in payload["agents"]]
        messages = prompt_manager().render(
            "itinary_multi_agent", items, payload["parameters"]
        )

        messages_dict = [{"role": m.role, "content": m.content} for m in messages]
        presente = {"payload": payload, "messages": messages_dict}

        # 3. Atomic write of the IPC request
        cle = f"{person.person_id}__{ctx.activity_id}"
        demande_fichier = f"{cle}.json"
        demande_path = self.dossier_demandes / demande_fichier
        demande_tmp = self.dossier_demandes / f"{cle}.json.tmp"
        reponse_path = self.dossier_reponses / demande_fichier

        demande_corps = {
            "version": 1,
            "person_id": str(person.person_id),
            "activity_id": str(ctx.activity_id),
            "modele_attendu": self.modele,
            "n_options": len(presentees),
            "messages": messages_dict,
            "schema_sortie": self._schema_sortie,
            # REQUESTED sampling settings. They used to be stored without being passed on:
            # the `temperature: 0.0` of experience.yaml never reached the sub-agent, and
            # the archive suggested a greedy decoding that nothing enforced. The sub-agent
            # answers with `parametres_appliques`, which says what it was able to apply.
            "parametres": dict(self.parametres),
        }

        # Clean up a possible earlier orphan response
        reponse_path.unlink(missing_ok=True)

        demande_tmp.write_text(
            json.dumps(demande_corps, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(demande_tmp, demande_path)

        # 4. Active wait for the response (non-blocking polling)
        t_start = time.time()
        self._demandes_en_cours[cle] = t_start
        pas_sondage_s = 0.5
        seuil_attente_agent = self.attente_max_s / 4.0

        try:
            while True:
                now = time.time()
                elapsed = now - t_start

                # Typed state handling ETAT_EN_ATTENTE_AGENT (§4.5)
                if elapsed > seuil_attente_agent and self.execution is not None:
                    etat_courant = self.execution.etat().get("etat")
                    if etat_courant not in (ETAT_EN_ATTENTE_AGENT, "epuisee", "arretee"):
                        self.execution.changer_etat(
                            ETAT_EN_ATTENTE_AGENT,
                            f"antigravity: attente agent > {seuil_attente_agent:.0f}s",
                        )

                self._verifier_journal_et_alarmes(now)
                self._verifier_demarrage(now)

                # Per-trip timeout (§4.5)
                if elapsed >= self.attente_max_s:
                    self.compteurs["timeouts"] += 1
                    return ReponseDecideur(
                        index=None,
                        fournisseur=self.nom,
                        erreur=f"antigravity: pas de réponse en {self.attente_max_s}s",
                        modele_verifie=False,
                        presente=presente,
                    )

                if reponse_path.is_file():
                    try:
                        contenu = reponse_path.read_text(encoding="utf-8")
                        resp = json.loads(contenu)
                    except (json.JSONDecodeError, OSError):
                        # File being written non-atomically: wait for the next round
                        await asyncio.sleep(pas_sondage_s)
                        continue

                    # person_id / activity_id check (§4.4)
                    if (
                        str(resp.get("person_id")) != str(person.person_id)
                        or str(resp.get("activity_id")) != str(ctx.activity_id)
                    ):
                        self.compteurs["rejets"] += 1
                        logger.warning(
                            f"[antigravity] Off-topic response for {cle}: "
                            f"received {resp.get('person_id')}/{resp.get('activity_id')}"
                        )
                        reponse_path.unlink(missing_ok=True)
                        await asyncio.sleep(pas_sondage_s)
                        continue

                    # Declared model vs expected model check (§4.4)
                    modele_declare = resp.get("modele_declare")
                    if modele_declare != self.modele:
                        self.compteurs["rejets"] += 1
                        # Move the request to processed to close the solicitation
                        traitee_path = self.dossier_demandes_traitees / demande_fichier
                        try:
                            os.replace(demande_path, traitee_path)
                        except OSError:
                            pass
                        return ReponseDecideur(
                            index=None,
                            fournisseur=self.nom,
                            erreur=(
                                f"antigravity: modèle déclaré inattendu "
                                f"({modele_declare!r} != {self.modele!r})"
                            ),
                            reponse_brute=resp.get("reponse_brute"),
                            modele_verifie=False,
                            presente=presente,
                        )

                    # First response of the channel: the startup safeguard is disarmed.
                    self._premiere_reponse = True

                    # Valid response received: move the request to processed
                    traitee_path = self.dossier_demandes_traitees / demande_fichier
                    try:
                        os.replace(demande_path, traitee_path)
                    except OSError:
                        pass

                    # Restore the in-progress state if we were waiting for the agent
                    if self.execution is not None:
                        if self.execution.etat().get("etat") == ETAT_EN_ATTENTE_AGENT:
                            self.execution.changer_etat(
                                ETAT_EN_COURS, "antigravity: réponse reçue"
                            )

                    # 5. Interpretation of the verdict (§3.1 step 4, D10)
                    agents = resp.get("agents") or []
                    agent_entry = agents[0] if agents else {}
                    entries = agent_entry.get("probabilities")
                    sent_modes = [
                        t.get("mode") for t in payload["agents"][0]["trajectories"]
                    ]

                    sorted_options = sorted(options, key=lambda p: p.get_code() or "")
                    position_in_sorted = {
                        id(opt): i for i, opt in enumerate(sorted_options)
                    }

                    shuffled_weights = normalize_option_probabilities(
                        entries,
                        len(options),
                        modes=sent_modes,
                        context=f"agent={person.person_id} activity={ctx.activity_id}",
                    )
                    weights_are_fallback = isinstance(
                        shuffled_weights, UniformFallback
                    )
                    if weights_are_fallback:
                        self.compteurs["rejets"] += 1
                        logger.error(
                            f"[ALARME] [antigravity] Unusable response for {person.person_id}/{ctx.activity_id} "
                            f"— uniform fallback REFUSED in Antigravity mode. Unexpected 'probabilities' format."
                        )
                        try:
                            os.replace(traitee_path, demande_path)
                            reponse_path.unlink(missing_ok=True)
                        except OSError:
                            pass
                        await asyncio.sleep(pas_sondage_s)
                        continue

                    weights = [0.0] * len(sorted_options)
                    for opt, w in zip(options, shuffled_weights):
                        weights[position_in_sorted[id(opt)]] += w

                    seed_parts = (
                        ctx.graine_tirage,
                        person.person_id,
                        ctx.activity_id,
                        datetime.fromtimestamp(
                            ctx.timestamp, tz=timezone.utc
                        ).strftime("%Y-%m-%d"),
                    )
                    idx_sorted = draw_index(
                        weights,
                        *seed_parts,
                        min_prob_threshold=settings.agent.mode_choice_truncation_threshold,
                    )
                    chosen_plan = sorted_options[idx_sorted]
                    idx_presentees = options.index(chosen_plan)

                    # Extraction of the justification
                    reasons = [None] * len(sorted_options)
                    if entries:
                        for entry in entries:
                            entry_reason = (
                                entry.get("reason")
                                if isinstance(entry, dict)
                                else getattr(entry, "reason", None)
                            )
                            if not entry_reason:
                                continue
                            try:
                                entry_idx = int(
                                    entry["index"]
                                    if isinstance(entry, dict)
                                    else entry.index
                                )
                            except (TypeError, ValueError, KeyError):
                                continue
                            if 0 <= entry_idx < len(options):
                                opt = options[entry_idx]
                                reasons[position_in_sorted[id(opt)]] = entry_reason

                    agent_reason = agent_entry.get("reason") or ""
                    reason = (
                        (reasons[idx_sorted] if reasons else None)
                        or agent_reason
                        or "Pas de justification fournie."
                    )
                    if "is chosen because it" in reason:
                        reason = f"This plan {reason.split('is chosen because it', 1)[1].strip()}"

                    modes = [opt.mode_label() for opt in sorted_options]
                    distribution = mode_distribution(weights, modes)
                    reason = (
                        f"{reason} [Répartition estimée : "
                        f"{_format_distribution(distribution)} — mode tiré au sort.]"
                    )

                    self.compteurs["servies"] += 1
                    if weights_are_fallback:
                        self.compteurs["replis"] += 1
                    if not resp.get("sortie_litterale"):
                        self.compteurs["sans_sortie_litterale"] += 1
                        if not self._alarme_litterale_levee:
                            logger.warning(
                                "[antigravity] Response without `sortie_litterale` — the trace "
                                "will not show what the model wrote (P8). The field is "
                                "expected verbatim in each response file."
                            )
                            self._alarme_litterale_levee = True

                    # What the sub-agent declares it applied of the requested parameters. A
                    # silent response is NOT "temperature 0": it is archived as a
                    # gap, exactly like `sortie_litterale` (P8, same reason).
                    parametres_appliques = resp.get("parametres_appliques")
                    if parametres_appliques is None and self.parametres:
                        self.compteurs["sans_parametres_appliques"] += 1
                        if not self._alarme_parametres_levee:
                            logger.warning(
                                "[antigravity] Response without `parametres_appliques` although "
                                f"{self.parametres} were requested — the trace will not say if "
                                "the setting was applied. The field is expected in each "
                                "response file; `{}` is a valid response and means "
                                "\"no applicable parameter\"."
                            )
                            self._alarme_parametres_levee = True

                    return ReponseDecideur(
                        index=idx_presentees,
                        fournisseur=self.nom,
                        distribution=dict(distribution or {}),
                        poids=[float(w) for w in shuffled_weights],
                        reponse_brute=(
                            resp.get("reponse_brute")
                            or json.dumps(resp, ensure_ascii=False)
                        ),
                        # P8 — passed on AS IS, never normalised nor folded back onto
                        # `reponse_brute`: a declared gap is better than a copy that would
                        # be taken for the model's output.
                        sortie_litterale=resp.get("sortie_litterale"),
                        parametres_appliques=parametres_appliques,
                        raison=reason,
                        souvenirs=[],
                        presente=presente,
                        repli_uniforme=weights_are_fallback,
                        modele_verifie=False,
                    )

                await asyncio.sleep(pas_sondage_s)
        finally:
            self._demandes_en_cours.pop(cle, None)


__all__ = ["DecideurAntigravity"]
