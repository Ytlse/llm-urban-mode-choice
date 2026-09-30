"""From trip to request: the gateway's grouping factor.

An experiment is counted in **trips** — one trip, one modal decision to make.
A provider quota is counted in **requests**. The two are not the same thing: the
gateway merges several agents into a single call (micro-batching), and the batch key
(`llm_gateway/core/batching.compute_batch_key`) is built on the category, the parameters,
the forced instance and the allowed instances — all constant within an arm. All
the decisions of an experiment therefore share the same batch.

Until 2026-09-22, the estimate assumed "one request per trip" and so overestimated
the quota consumed by a factor of about eight — the benchmark kept in ticket 073 § 4: about
310 requests for a full arm, i.e. roughly 0.3 day of quota. This module provides the
missing divisor, and above all **says where it comes from**:

- the **cap** is derived: the `batch_max_agents` the gateway computes for the target
  instance, bounded by the experiment's parallelism in simulator-less mode (the batch cannot
  hold more agents than there are tasks in flight);
- the **observed factor** is measured on the archived runs of the same template.

The two answer different questions and neither replaces the other: the cap is
a structural bound never reached, the measurement is noisy. What decides a launch
is the **prudent** figure (cf. `facteurs`), never the cap.
"""

from __future__ import annotations

import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from experiences.ressources import FUSEAU_QUOTA_DEFAUT
from loguru import logger

#: A grouping measurement is an average: it is worthless on two requests. We require
#: the archived run to have carried at least this number of FULL batches, otherwise a
#: handful of noise requests can double the ratio. Derived from the cap, not
#: a hard-coded volume: `sollicitations >= LOTS_MINIMUM * plafond`.
LOTS_MINIMUM_POUR_MESURER = 10

#: Reasons for discarding an archived run, so that the rejection is readable, not silent.
MOTIFS = (
    "reglages",  # same template, but other parallelism or other truncation
    "sans_compteur",  # neither a measured delta nor a daily counter
    "trop_courte",  # fewer than LOTS_MINIMUM_POUR_MESURER batches
    "fenetre_traversee",  # the daily counter was reset to zero during the run
    "hors_bornes",  # ratio outside [1, plafond]: the counter does not describe this run
    "run_non_probant",  # resumed, interrupted or quota exhausted: the day's counter lies
)


# ── Derived cap ──────────────────────────────────────────────────────────────


def _plafond_instance(cfg: dict, etat: dict | None) -> tuple[int | None, str]:
    """Batch cap of an instance: the gateway first, the formula second.

    `/health` has published `batch_max_agents` since 2026-09-22: it is the **authoritative**
    value, computed at startup of the container that actually serves the requests. The
    fallback recomputes the same formula from `providers.yaml` — with the defaults of
    `BatchingSettings`, which may differ from the worker's environment. Hence the distinct
    source: a recomputed figure must not pass itself off as a reported one.
    """
    publie = (etat or {}).get("batch_max_agents")
    if publie:
        try:
            return int(publie), "passerelle"
        except (TypeError, ValueError):
            pass
    try:
        from llm_gateway.config.settings import BatchingSettings
        from llm_gateway.core.batching import compute_batch_max_agents
    except ImportError as e:  # package missing: say so, do not guess
        logger.warning(f"[lots] llm_gateway indisponible ({e}) : plafond de lot inconnu")
        return None, "indisponible"
    b = BatchingSettings()
    rpm = cfg.get("rpm_limit")
    if not rpm:
        return None, "indisponible"
    return (
        compute_batch_max_agents(
            tpm_limit=cfg.get("tpm_limit"),
            rpm_limit=int(rpm),
            max_tokens_per_request=cfg.get("max_tokens_per_request"),
            tokens_per_agent=b.assumed_prompt_tokens + b.assumed_output_tokens,
            plafond=b.max_batch_agents,
        ),
        "providers.yaml (formule de la passerelle, défauts de batching)",
    )


def plafond_lot(
    providers: dict[str, dict],
    instances: list[str],
    *,
    parallelisme: int | None = None,
    etat_passerelle: dict[str, dict] | None = None,
) -> tuple[int, str]:
    """(plafond, source): the most agents that one request of this experiment can carry.

    Minimum over the allowed instances — the decision-maker uses them up IN SERIES (one
    forced instance per request, cf. `DecideurPasserelle._prochaine_instance`), so the
    smallest capacity ends up serving, and it is the one that gives the prudent bound.

    `parallelisme` bounds in turn, and this bound explains the measurements: in
    simulator-less mode, `regroupement.parallelisme` persons advance side by side and the
    trips of one person are serial — so there can never be more than
    `parallelisme` tasks waiting to be merged. Passing `None` removes this bound: in
    simulator mode GAMA feeds the queue, and batches of 15 agents are observed there for a
    declared parallelism of 8.
    """
    if not instances:
        return 1, "aucune instance admise : aucun regroupement supposé"
    valeurs, sources = [], set()
    for i in instances:
        v, src = _plafond_instance(providers.get(i) or {}, (etat_passerelle or {}).get(i))
        if v:
            valeurs.append(v)
            sources.add(src)
    if not valeurs:
        return 1, "plafond de lot inconnu : aucun regroupement supposé"
    plafond = min(valeurs)
    source = f"batch_max_agents={plafond} (min sur {len(valeurs)} instance(s), {', '.join(sorted(sources))})"
    if parallelisme and parallelisme < plafond:
        source = (
            f"parallélisme {parallelisme} de l'expérience (sous batch_max_agents={plafond}, "
            f"{', '.join(sorted(sources))})"
        )
        plafond = int(parallelisme)
    return plafond, source


# ── Measurement on the archived runs ─────────────────────────────────────────


def _date_locale(iso: str | None, fuseau: str) -> Any:
    if not iso:
        return None
    try:
        from zoneinfo import ZoneInfo

        tz = ZoneInfo(fuseau)
    except Exception:  # noqa: BLE001 — fuseau inconnu / tzdata absente
        tz = None
    try:
        d = datetime.fromisoformat(str(iso))
    except ValueError:
        return None
    return (d.astimezone(tz) if tz else d).date()


def fenetre_quota_traversee(debut: str | None, fin: str | None, fuseau: str) -> bool:
    """Did the run cross the midnight that resets the daily counters to zero?

    If it did ⇒ `daily_requests` read at closure counts only part of the run, the
    "trips ÷ requests" ratio blows up, and grouping looks better than it is.
    It is the only bias of this measurement that leans towards imprudence: we discard.

    An unreadable bound is treated as a crossing: when in doubt, we discard.
    """
    d1, d2 = _date_locale(debut, fuseau), _date_locale(fin, fuseau)
    if d1 is None or d2 is None:
        return True
    return d1 != d2


def _compteur_journalier_probant(conf: dict, compteurs: dict) -> str | None:
    """Can this DAY counter stand for the cost of THIS run? If not, why.

    Only under three conditions, and forgetting them is costly: read as is on run
    `2026-09-21_17_00_49`, which had exhausted its quota (`key1` at 506 for a limit of 500), it
    shows 2.4 agents/request where the real grouping is about 8 — the retries and
    the day's other runs inflate the denominator.

    - **finished**: a run stopped midway consumed requests that its
      archived solicitations do not reflect;
    - **no resume**: a resume re-serves archived decisions without soliciting anything, and its
      requests add to those of the previous attempt;
    - **quota not exhausted**: at the cap, the provider refused requests that were still
      counted.
    """
    if (conf.get("cloture") or {}).get("etat") != "terminee":
        return "exécution non terminée"
    if conf.get("interruptions"):
        return "exécution interrompue ou reprise"
    for ligne in compteurs.get("quota") or []:
        if not isinstance(ligne, dict):
            continue
        if ligne.get("epuisee"):
            return f"quota épuisé sur {ligne.get('instance')}"
        servies, plafond = ligne.get("requetes_jour"), ligne.get("limite_jour")
        if servies is not None and plafond and int(servies) >= int(plafond):
            return (
                f"{ligne.get('instance')} au plafond du jour "
                f"({servies}/{plafond}) : des refus ont été comptés"
            )
    return None


def _requetes_de_lexecution(compteurs: dict) -> tuple[int | None, bool]:
    """(requests consumed, reliable measurement?) for an archived run.

    Two sources, and they are not equivalent:

    1. `compteurs["requetes"]["delta"]` — difference of the gateway counters between
       opening and closure, written by the runner since 2026-09-22: it is the number
       of requests of THIS run.
    2. `compteurs["quota"][].requetes_jour` — DAILY counter read at closure. It
       aggregates everything the instance served that day: the other runs, the reflections,
       the retries. It therefore overestimates the run's requests and underestimates grouping
       — in the prudent direction, which makes it usable, but never as a measurement.
    """
    r = compteurs.get("requetes")
    if isinstance(r, dict) and r.get("fiable") and r.get("delta"):
        return int(r["delta"]), True
    q = compteurs.get("quota")
    if isinstance(q, list) and q:
        total = sum(int(l.get("requetes_jour") or 0) for l in q if isinstance(l, dict))
        if total > 0:
            return total, False
    return None, False


def mesures_archivees(
    empreinte_gabarit_: str,
    *,
    parallelisme: int | None,
    troncature: bool,
    plafond: int,
    providers: dict[str, dict] | None = None,
    dossier_experiences_: Path | str | None = None,
) -> dict | None:
    """Grouping factors read on the comparable runs already played.

    Comparable = same template, same parallelism, same truncation of the *consideration set*:
    these are the three settings that shift an agent's token cost, hence the size of the
    batches; mixing them would produce an average that describes none of them.

    Returns `{minimum, mediane, n, fiabilite, source, ecartees}` or `None` if nothing is kept.
    """
    from experiences.experience import dossier_experiences, executions_vivantes

    racine = (
        Path(dossier_experiences_) if dossier_experiences_ else dossier_experiences()
    )
    if not racine.is_dir():
        return None
    seuil = LOTS_MINIMUM_POUR_MESURER * max(1, plafond)
    ratios: list[float] = []
    fiables = 0
    ecartees: dict[str, int] = {}
    sources: list[str] = []

    def ecarter(motif: str) -> None:
        ecartees[motif] = ecartees.get(motif, 0) + 1

    for exec_yaml in executions_vivantes(racine):
        try:
            conf = yaml.safe_load(exec_yaml.read_text(encoding="utf-8")) or {}
        except (yaml.YAMLError, OSError):
            continue
        emp = (conf.get("empreintes") or {}).get("gabarit") or {}
        exp = conf.get("experience") or {}
        if emp.get("sha256") != empreinte_gabarit_:
            continue
        par = (exp.get("regroupement") or {}).get("parallelisme")
        if parallelisme is not None and par is not None and int(par) != int(parallelisme):
            ecarter("reglages")
            continue
        if bool(exp.get("troncature_15", False)) != bool(troncature):
            ecarter("reglages")
            continue
        chemin_compteurs = exec_yaml.parent / "compteurs.json"
        if not chemin_compteurs.is_file():
            ecarter("sans_compteur")
            continue
        try:
            compteurs = json.loads(chemin_compteurs.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            ecarter("sans_compteur")
            continue
        sollicitations = int(compteurs.get("sollicitations") or 0)
        requetes, fiable = _requetes_de_lexecution(compteurs)
        if not sollicitations or not requetes:
            ecarter("sans_compteur")
            continue
        if sollicitations < seuil:
            ecarter("trop_courte")
            continue
        if not fiable:
            motif = _compteur_journalier_probant(conf, compteurs)
            if motif:
                logger.debug(f"[lots] {exec_yaml.parent.name} discarded: {motif}")
                ecarter("run_non_probant")
                continue
            # The daily counter only makes sense if the run fits within a single window.
            instances = [
                str(l.get("instance"))
                for l in (compteurs.get("quota") or [])
                if isinstance(l, dict) and l.get("instance")
            ]
            fuseau = next(
                (
                    str(((providers or {}).get(i) or {}).get("quota_reset_tz"))
                    for i in instances
                    if ((providers or {}).get(i) or {}).get("quota_reset_tz")
                ),
                FUSEAU_QUOTA_DEFAUT,
            )
            if fenetre_quota_traversee(
                conf.get("cree_le"), (conf.get("cloture") or {}).get("le"), fuseau
            ):
                ecarter("fenetre_traversee")
                continue
        ratio = sollicitations / requetes
        if ratio < 1 or ratio > plafond:
            # < 1: the counter describes more than this run. > plafond: it describes less
            # than this run (reset to zero, gateway restart). Neither one measures
            # grouping — and the second would lean towards imprudence.
            ecarter("hors_bornes")
            continue
        ratios.append(ratio)
        fiables += 1 if fiable else 0
        sources.append(str(exec_yaml.parent.relative_to(racine)))

    if not ratios:
        if ecartees:
            logger.info(
                f"[lots] no usable archived run for grouping "
                f"({', '.join(f'{m} × {n}' for m, n in sorted(ecartees.items()))})"
            )
        return None
    fiabilite = "mesurée" if fiables == len(ratios) else "minorée (compteur journalier)"
    return {
        "minimum": min(ratios),
        "mediane": statistics.median(ratios),
        "n": len(ratios),
        "fiabilite": fiabilite,
        "ecartees": ecartees,
        "source": (
            f"{len(ratios)} exécution(s) archivée(s) du même gabarit, même parallélisme, "
            f"même troncature — {fiabilite}"
            + (f" ; {sum(ecartees.values())} écartée(s)" if ecartees else "")
        ),
    }


# ── Summary: the three figures, and the one that decides ─────────────────────


def facteurs(
    *,
    providers: dict[str, dict],
    instances: list[str],
    parallelisme: int | None,
    empreinte_gabarit_: str | None = None,
    troncature: bool = False,
    etat_passerelle: dict[str, dict] | None = None,
    dossier_experiences_: Path | str | None = None,
) -> dict:
    """The grouping factors to apply, each with its source.

    Three figures, and only one decides:

    - `plafond` — structural bound, never reached. Used to display a floor of requests,
      **never** to authorise a launch.
    - `attendu` — median of the comparable runs, failing that the cap. It is the figure
      one reads to know what the arm will really cost.
    - `prudent` — minimum of the comparable runs, failing that **1**. It is what feeds
      the quota warning, the quota share and the duration. Without a measurement, it is 1:
      prudence falls back exactly on the behaviour from before this fix, never
      below. An over-optimistic estimate would launch an arm that would not finish; the
      derived cap therefore cannot play this role.
    """
    plafond, source_plafond = plafond_lot(
        providers, instances, parallelisme=parallelisme, etat_passerelle=etat_passerelle
    )
    mesure = (
        mesures_archivees(
            empreinte_gabarit_,
            parallelisme=parallelisme,
            troncature=troncature,
            plafond=plafond,
            providers=providers,
            dossier_experiences_=dossier_experiences_,
        )
        if empreinte_gabarit_
        else None
    )
    if mesure:
        return {
            "plafond": plafond,
            "attendu": round(mesure["mediane"], 2),
            "prudent": round(mesure["minimum"], 2),
            "decide_par": "prudent",
            "fiabilite": mesure["fiabilite"],
            "source": f"{mesure['source']} ; plafond : {source_plafond}",
            "mesures": {k: mesure[k] for k in ("n", "minimum", "mediane", "ecartees")},
        }
    return {
        "plafond": plafond,
        "attendu": float(plafond),
        "prudent": 1.0,
        "decide_par": "prudent",
        "fiabilite": "aucune mesure",
        "source": (
            f"aucune exécution archivée comparable : attendu = plafond dérivé "
            f"({source_plafond}), prudent = 1 requête par déplacement"
        ),
        "mesures": None,
    }


def requetes(deplacements: int, facteur: float) -> int:
    """Number of provider requests for this number of trips, rounded up."""
    if deplacements <= 0:
        return 0
    return max(1, math.ceil(deplacements / max(1.0, float(facteur))))


__all__ = [
    "LOTS_MINIMUM_POUR_MESURER",
    "MOTIFS",
    "facteurs",
    "fenetre_quota_traversee",
    "mesures_archivees",
    "plafond_lot",
    "requetes",
]
