"""Decision-by-decision pairing of TWO runs — the quantification of axis 0 (ticket 073).

Axis 0 of ticket 073 replays an experiment **changing nothing**: same model, same prompt,
same set, same seeds. Any gap between the two outputs then belongs to the provider, and
to it alone — the control `exp_rf_…_c_nosim` establishes it, its two runs being identical to
the hundredth. This tool puts the two runs face to face at the decision level.

Three levels, from the finest to the coarsest:

1. **the masses** — the probability vector the model assigns to the offered options;
2. **the chosen mode** — the option actually drawn from these masses;
3. **the macro** — composite score and modal shares, delegated to `experiences.registre.comparer`.

⚠ **The masses are read in `poids_presentes`, never in `distribution`.** `distribution`
aggregates over the six canonical modes and crushes several options of a same mode: measured on
2026-09-21 on the first 120 decisions of the replicate, it announced 5 "switches at equal
masses" where `poids_presentes` sees only ONE. The other four were an artefact of
the aggregation. The difference is not cosmetic: the ticket classifies a switch at equal
masses as a defect of the SETUP, not of the model. Picking the wrong vector means inventing a bug.

What this tool does NOT say:

- it scores nothing — the composite score stays with the scorer, and level 3 is only a pointer;
- it does not conclude on a partial run. A resumption in progress covers only part
  of the cohort; the figure is then an indication, not a measure, and the tool says so as
  `[ALARME]`. It is exactly the defect of 2026-09-15: a `moves.csv` truncated at 274 lines
  published as if it carried the 3,299 decisions;
- it does not tell two seeds apart. A seed changes the order, the draw and the calendar;
  pairing could not untangle these effects and does not claim to.

Usage:

    make apparier A=<dossier exécution> B=<dossier exécution> [JSON=<fichier>] [TOUT=1]

Usage in a notebook:

    import sys; sys.path.append("..")          # from scripts/analysis/
    from appariement_executions import apparier
    r = apparier(dossier_a, dossier_b)
    r["masses"]["l1_moyenne"]
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger("appariement")

# Trace file read on both sides. One line = one decision.
F_DECISIONS = "decisions.jsonl"

# Below this coverage rate, the pairing no longer bears on the cohort but on a
# sample of circumstance: it comes out as [ALARME] and the report repeats it.
SEUIL_COUVERTURE = 0.80

# Bounds of the histogram of L1 gaps between masses (L1 ∈ [0, 2]). A fat tail on few
# decisions does not read like diffuse noise on all of them: the shape decides.
TRANCHES_L1 = (0.0, 0.05, 0.10, 0.25, 0.50, 1.00, 2.00)

# Two floats from the same computation are not compared bit for bit.
EPSILON = 1e-9


# ─────────────────────────────── reading ────────────────────────────────


def charger(dossier: str | Path) -> dict[tuple[str, str], dict]:
    """The decisions of a run, indexed by `(person_id, activity_id)`.

    It is the pair that identifies a trip: the same individual decides several times
    in their day, and the same trip is found again from one run to the other.
    """
    chemin = Path(dossier) / F_DECISIONS
    if not chemin.exists():
        raise FileNotFoundError(
            f"{chemin} is missing — this directory is not a run, or it "
            "archived no decision."
        )
    par_cle: dict[tuple[str, str], dict] = {}
    doublons = 0
    for ligne in chemin.open(encoding="utf-8"):
        ligne = ligne.strip()
        if not ligne:
            continue
        d = json.loads(ligne)
        cle = (str(d.get("person_id")), str(d.get("activity_id")))
        if cle in par_cle:
            doublons += 1
        par_cle[cle] = d
    if doublons:
        logger.warning(
            f"[appariement] {chemin}: {doublons} (person_id, activity_id) pair(s) "
            "repeated — the last occurrence is authoritative."
        )
    return par_cle


def _codes_offerts(d: dict) -> list[str]:
    """The codes of the options submitted to the decision-maker, in the order presented to it.

    The order matters: `poids_presentes` aligns on it, and `index_presente` points into it.
    """
    return [str(o.get("code")) for o in (d.get("presentees") or [])]


def _masses(d: dict, taille: int) -> list[float]:
    """`poids_presentes` padded to `taille` — never `distribution` (cf. the header)."""
    p = [float(x) for x in (d.get("poids_presentes") or [])]
    return p + [0.0] * max(0, taille - len(p))


def _code_retenu(d: dict) -> Optional[str]:
    return (d.get("retenue") or {}).get("code")


# ──────────────────────────── classification ────────────────────────────


def classer(
    a: dict[tuple[str, str], dict], b: dict[tuple[str, str], dict]
) -> dict[str, list[tuple[str, str]]]:
    """Sorts the common keys into three disjoint populations.

    The classification IS the measure: mixing these populations means attributing to the
    decision-maker a divergence the chain produced upstream, or drowning the signal in decisions
    where the model was never called upon.

    - `cascade_amont`: the offered options differ. The previous trip switched,
      the bike is no longer where it was, the offer changes. Nothing here is attributable to the
      decision-maker ON THIS TRIP — the gap is real but inherited.
    - `choix_unique`: a single option, no call. Control: these decisions must
      coincide, and if they diverge the offer has moved, so the upstream cascade
      was not correctly detected.
    - `appariables`: same options AND decision-maker called on both sides. **The only
      population on which axis 0 is quantified.**
    """
    familles: dict[str, list[tuple[str, str]]] = {
        "cascade_amont": [],
        "choix_unique": [],
        "appariables": [],
        "hors_mesure": [],
        "methode_dissymetrique": [],
    }
    for cle in sorted(set(a) & set(b)):
        da, db = a[cle], b[cle]
        ma, mb = da.get("methode"), db.get("methode")
        if _codes_offerts(da) != _codes_offerts(db):
            familles["cascade_amont"].append(cle)
        elif ma != mb:
            # The decision-maker answered on one side and not the other, for an identical offer.
            # Neither cascade nor single choice: an incident, and hiding it would make it invisible.
            familles["methode_dissymetrique"].append(cle)
        elif ma == "decideur":
            familles["appariables"].append(cle)
        elif ma == "choix_unique":
            familles["choix_unique"].append(cle)
        else:
            # Same method on both sides, but no decision to compare: `inexploitable`
            # (no proposal from the engines), `sans_solution`. Filing them as asymmetric
            # would invent 9 incidents where there are none — noted on 2026-09-21.
            familles["hors_mesure"].append(cle)
    return familles


def _tranche(l1: float) -> str:
    if l1 <= EPSILON:
        return "0"
    for bas, haut in zip(TRANCHES_L1, TRANCHES_L1[1:]):
        if l1 <= haut:
            return f"]{bas:g} ; {haut:g}]"
    return f"> {TRANCHES_L1[-1]:g}"


# ─────────────────────────────── measures ───────────────────────────────


def mesurer(
    a: dict[tuple[str, str], dict],
    b: dict[tuple[str, str], dict],
    cles: Iterable[tuple[str, str]],
) -> dict[str, Any]:
    """Levels 1 and 2 on the `appariables` decisions only.

    Also returns, named and detailed, the switches **at strictly identical masses**:
    with equal masses and equal `graine_tirage`, the draw MUST return the same option. A
    switch there is a defect of the setup, not a dispersion of the model, and it is handled
    as such — so it is listed, it is not summed up as a rate.
    """
    cles = list(cles)
    l1s: list[float] = []
    identiques = argmax_bascule = retenue_bascule = 0
    histogramme: dict[str, int] = {}
    suspectes: list[dict] = []

    for cle in cles:
        da, db = a[cle], b[cle]
        taille = max(len(_codes_offerts(da)), len(da.get("poids_presentes") or []))
        pa, pb = _masses(da, taille), _masses(db, taille)
        l1 = sum(abs(x - y) for x, y in zip(pa, pb))
        l1s.append(l1)
        egales = l1 <= EPSILON
        if egales:
            identiques += 1
        histogramme[_tranche(l1)] = histogramme.get(_tranche(l1), 0) + 1

        if pa and pb and max(range(len(pa)), key=pa.__getitem__) != max(
            range(len(pb)), key=pb.__getitem__
        ):
            argmax_bascule += 1

        ra, rb = _code_retenu(da), _code_retenu(db)
        if ra != rb:
            retenue_bascule += 1
            if egales:
                suspectes.append(
                    {
                        "person_id": cle[0],
                        "activity_id": cle[1],
                        "poids": pa,
                        "options": _codes_offerts(da),
                        "a": {
                            "code": ra,
                            "index_presente": (da.get("retenue") or {}).get(
                                "index_presente"
                            ),
                            "mode": (da.get("retenue") or {}).get("mode"),
                            "identifiant_lot": da.get("identifiant_lot"),
                        },
                        "b": {
                            "code": rb,
                            "index_presente": (db.get("retenue") or {}).get(
                                "index_presente"
                            ),
                            "mode": (db.get("retenue") or {}).get("mode"),
                            "identifiant_lot": db.get("identifiant_lot"),
                        },
                        "graine_tirage": {
                            "a": da.get("graine_tirage"),
                            "b": db.get("graine_tirage"),
                        },
                    }
                )

    n = len(cles)
    return {
        "n": n,
        "masses": {
            "identiques": identiques,
            "part_identiques": (identiques / n) if n else None,
            "l1_moyenne": statistics.mean(l1s) if l1s else None,
            "l1_mediane": statistics.median(l1s) if l1s else None,
            "l1_max": max(l1s) if l1s else None,
            "histogramme": dict(sorted(histogramme.items())),
            "argmax_bascule": argmax_bascule,
            "part_argmax_bascule": (argmax_bascule / n) if n else None,
        },
        "retenue": {
            "bascules": retenue_bascule,
            "taux_bascule": (retenue_bascule / n) if n else None,
            "bascules_a_masses_identiques": len(suspectes),
            "detail_masses_identiques": suspectes,
        },
    }


def _composite(dossier: str | Path) -> Optional[dict]:
    """The composite score of a run, read from its `scores.json`.

    `registre.synthese()` carries the counters and the modal shares, NOT the composite score: it
    comes from the scorer, which writes `scores.json` separately. A partial run has none —
    and that is normal, the scorer refuses to score an incomplete run.
    """
    f = Path(dossier) / "scores.json"
    if not f.exists():
        return None
    return (json.loads(f.read_text(encoding="utf-8")) or {}).get("composite")


def _garde(a: str | Path, b: str | Path, inclure_invalides: bool) -> dict[str, Any]:
    """Level 3 + comparability guard, delegated to the registry (RG-4, P6).

    Pairing two runs that do not differ ONLY by chance does not measure chance:
    `registre.comparer` checks it on the fingerprints, never on the strength of the names, and
    refuses an invalidated or archived run. The module is available only where the package
    `experiences` is (`controller` container); elsewhere, the pairing remains computable
    but the guard does not apply, and the report says so instead of hiding it.
    """
    try:
        from experiences import registre  # type: ignore
    except Exception as exc:  # noqa: BLE001 — paquet absent hors du conteneur
        logger.warning(
            f"[appariement] comparability guard UNAVAILABLE ({exc.__class__.__name__}: "
            f"{exc}) — the fingerprints are not checked. Launch in the "
            "`controller` container (make apparier) to enable it."
        )
        return {
            "disponible": False,
            "comparable": None,
            "differences": None,
            "composite_a": _composite(a),
            "composite_b": _composite(b),
        }

    r = registre.comparer(a, b, inclure_invalides=inclure_invalides)
    return {
        "disponible": True,
        "comparable": r.get("comparable"),
        "differences": r.get("differences"),
        "statuts": r.get("statuts"),
        "composite_a": _composite(a),
        "composite_b": _composite(b),
    }


# ────────────────────────────── orchestration ───────────────────────────


def apparier(
    dossier_a: str | Path,
    dossier_b: str | Path,
    *,
    inclure_invalides: bool = False,
    seuil_couverture: float = SEUIL_COUVERTURE,
) -> dict[str, Any]:
    """Pairs two runs and returns the three levels. Modifies nothing, anywhere."""
    debut = time.monotonic()
    logger.info(f"[appariement] start: A={dossier_a} · B={dossier_b}")

    macro = _garde(dossier_a, dossier_b, inclure_invalides)
    if macro["disponible"] and macro["comparable"] is False and not inclure_invalides:
        raise ValueError(
            "pairing refused: the two executions differ by more than chance — "
            f"{json.dumps(macro['differences'], ensure_ascii=False)}. Pairing different "
            "conditions does not measure the provider's non-determinism. To override "
            "knowingly: --tout."
        )

    a, b = charger(dossier_a), charger(dossier_b)
    familles = classer(a, b)
    communes = set(a) & set(b)
    orphelines_a, orphelines_b = set(a) - communes, set(b) - communes

    # The coverage that matters is NOT `appariables / communes`: single choices make it
    # drop although they are legitimate. It is the share of the larger of the two runs
    # that the pairing reaches — hence what would say a fraction is published for a whole.
    reference = max(len(a), len(b))
    couverture = (len(communes) / reference) if reference else 0.0

    mesures = mesurer(a, b, familles["appariables"])

    duree = time.monotonic() - debut
    logger.info(
        f"[appariement] A {len(a)} decisions · B {len(b)} · common {len(communes)} "
        f"· orphans A {len(orphelines_a)} / B {len(orphelines_b)} · "
        f"pairable {len(familles['appariables'])} · upstream cascade "
        f"{len(familles['cascade_amont'])} · single choice {len(familles['choix_unique'])} "
        f"· out of measure {len(familles['hors_mesure'])} · asymmetric method "
        f"{len(familles['methode_dissymetrique'])}"
    )
    if couverture < seuil_couverture:
        logger.error(
            f"[ALARME] [appariement] couverture {couverture:.1%} < {seuil_couverture:.0%} : "
            f"{len(communes)} décisions appariées pour {reference} archivées du côté le plus "
            "avancé. Le chiffre porte sur un échantillon de circonstance, PAS sur la cohorte "
            "— ne pas le publier tel quel (cf. le moves.csv tronqué du 2026-09-15)."
        )
    if mesures["retenue"]["bascules_a_masses_identiques"]:
        logger.error(
            f"[ALARME] [appariement] "
            f"{mesures['retenue']['bascules_a_masses_identiques']} switch(es) of the chosen mode "
            "at STRICTLY identical masses and equal draw seed: the draw should "
            "be reproducible. It is a defect of the setup, not a dispersion of the model."
        )
    logger.info(
        f"[appariement] finished in {duree:.2f} s — "
        f"{len(familles['appariables'])} decision(s) quantified, "
        f"identical masses {mesures['masses']['identiques']}, "
        f"switches {mesures['retenue']['bascules']}"
    )

    return {
        "a": str(dossier_a),
        "b": str(dossier_b),
        "couverture": {
            "decisions_a": len(a),
            "decisions_b": len(b),
            "communes": len(communes),
            "orphelines_a": len(orphelines_a),
            "orphelines_b": len(orphelines_b),
            "reference": reference,
            "taux": couverture,
            "seuil": seuil_couverture,
            "suffisante": couverture >= seuil_couverture,
        },
        "populations": {k: len(v) for k, v in familles.items()},
        "masses": mesures["masses"],
        "retenue": mesures["retenue"],
        "macro": macro,
        "duree_s": round(duree, 3),
    }


# ─────────────────────────────── rendering ──────────────────────────────


def _pc(x: Optional[float]) -> str:
    return "—" if x is None else f"{x:.1%}"


def _f(x: Optional[float], n: int = 4) -> str:
    return "—" if x is None else f"{x:.{n}f}"


def rendre(r: dict[str, Any]) -> str:
    """The readable report. The order follows the ticket: coverage, populations, 1, 2, 3."""
    c, p, m, ret = r["couverture"], r["populations"], r["masses"], r["retenue"]
    L = [
        "=== APPARIEMENT EXÉCUTION CONTRE EXÉCUTION (axe 0, ticket 073) ===",
        f"A : {r['a']}",
        f"B : {r['b']}",
        "",
        f"Couverture — {c['communes']} décisions communes sur {c['reference']} "
        f"({_pc(c['taux'])}) · seuil {_pc(c['seuil'])} : "
        f"{'suffisante' if c['suffisante'] else '*** INSUFFISANTE — chiffre non publiable ***'}",
        f"  A {c['decisions_a']} · B {c['decisions_b']} · "
        f"orphelines A {c['orphelines_a']} / B {c['orphelines_b']}",
        "",
        "Populations",
        f"  appariables (chiffrées)   : {p['appariables']}",
        f"  cascade amont (écartées)  : {p['cascade_amont']}",
        f"  choix unique (écartées)   : {p['choix_unique']}",
        f"  hors mesure (écartées)    : {p['hors_mesure']}",
        f"  méthode dissymétrique     : {p['methode_dissymetrique']}",
        "",
        "Niveau 1 — masses (poids_presentes)",
        f"  identiques        : {m['identiques']} ({_pc(m['part_identiques'])})",
        f"  L1 moyenne        : {_f(m['l1_moyenne'])}",
        f"  L1 médiane        : {_f(m['l1_mediane'])}",
        f"  L1 max            : {_f(m['l1_max'])}",
        f"  argmax qui bascule: {m['argmax_bascule']} ({_pc(m['part_argmax_bascule'])})",
        "  histogramme L1    : "
        + (
            " · ".join(f"{k} → {v}" for k, v in m["histogramme"].items())
            if m["histogramme"]
            else "—"
        ),
        "",
        "Niveau 2 — option retenue",
        f"  bascules          : {ret['bascules']} ({_pc(ret['taux_bascule'])})",
        f"  dont à masses identiques : {ret['bascules_a_masses_identiques']}",
    ]
    for s in ret["detail_masses_identiques"]:
        L.append(
            f"    ⚠ {s['person_id']} / {s['activity_id']} · poids {s['poids']} · "
            f"A idx {s['a']['index_presente']} ({s['a']['mode']}) → "
            f"B idx {s['b']['index_presente']} ({s['b']['mode']}) · "
            f"graine_tirage {s['graine_tirage']['a']}/{s['graine_tirage']['b']} · "
            f"lots {s['a']['identifiant_lot']} / {s['b']['identifiant_lot']}"
        )
    L += ["", "Niveau 3 — macro"]
    macro = r["macro"]
    if not macro["disponible"]:
        L.append("  garde de comparabilité indisponible (paquet `experiences` absent)")
    else:
        L.append(f"  comparable : {macro['comparable']}")
        if macro["differences"]:
            for d in macro["differences"]:
                L.append(f"    différence : {d}")
    for cote in ("a", "b"):
        comp = macro.get(f"composite_{cote}") or {}
        if comp:
            L.append(
                f"  {cote.upper()} composite {_f(comp.get('emd_jsd'), 4)} · "
                f"hors choix unique {_f(comp.get('emd_jsd_hors_choix_unique'), 4)} · "
                f"L1 {_f(comp.get('l1'), 2)}"
            )
        else:
            L.append(f"  {cote.upper()} composite — (pas de scores.json : run partiel)")
    return "\n".join(L)


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    ap = argparse.ArgumentParser(
        description="Pairs two runs decision by decision (axis 0, ticket 073)."
    )
    ap.add_argument("a", help="directory of the reference run")
    ap.add_argument("b", help="directory of the run to compare")
    ap.add_argument("--json", dest="json_out", help="writes the full report as JSON")
    ap.add_argument(
        "--tout",
        action="store_true",
        help="pairs despite an invalidated, archived or non-comparable run",
    )
    ap.add_argument(
        "--seuil",
        type=float,
        default=SEUIL_COUVERTURE,
        help=f"alarm threshold on coverage (default {SEUIL_COUVERTURE})",
    )
    args = ap.parse_args(argv)

    try:
        r = apparier(
            args.a, args.b, inclure_invalides=args.tout, seuil_couverture=args.seuil
        )
    except (ValueError, FileNotFoundError) as exc:
        logger.error(f"[appariement] {exc}")
        return 2

    print(rendre(r))
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        logger.info(f"[appariement] JSON report written: {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
