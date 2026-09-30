"""Figure D.1 — finite-sample bias of the EMD–JSD composite, measured by subsampling persons.

Appendix D of the short paper. No LLM call and no simulation: the script
re-reads the `moves.csv` of three real executions on the sealed cohort c1, keeps the
decisions of N persons drawn without replacement, and re-scores them with the project's
own scorer (`experiences.score`, formula `v1_reference`). Decisions never change; only
the set of persons scored does.

For each N and each execution, R replicates are drawn with `random.Random(2026)`. The
figure plots the mean excess of the subsample composite over the composite of all
scored persons of that execution, with the 5–95 % band of the replicates.

Run with the service venv (the scorer imports the calibration loss engine):

    services/llm-agents/.venv/bin/python scripts/annexes/fig_D_finite_sample_bias.py

Outputs
    docs/paper/article-court/appendices/images/D1_finite_sample_bias.png
    scripts/annexes/data/fig_D_finite_sample_bias.json   (every number of the figure)

`--redraw` redraws the figure from that JSON without rescoring (a few seconds).
"""

from __future__ import annotations

import json
import logging
import random
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "services" / "llm-agents"))
sys.path.insert(0, str(REPO))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from experiences import formule as F  # noqa: E402
from experiences import score as S  # noqa: E402
from scripts.depot_papiers import exiger_depot_papiers, sortie_papier  # noqa: E402
from scripts.synthesis import frames  # noqa: E402

log = logging.getLogger("fig_D_finite_sample_bias")

EXPERIENCES = REPO / "archive/1_regime_nominal/jeu_1000_PANEL_v6_EN_c/experiences"
SUFFIX = "_jtir_pop-1000_PANEL_v6_jeu-20260316_EN_c"
# (label in the figure, experiment directory, execution) — the executions of Table 1.
ARMS = [
    ("LightGBM", f"exp_lgbm{SUFFIX}_nosim", "2026-09-16_13_40_53"),
    ("jev, expert prompt 32", f"exp_jev-1130_proexp32{SUFFIX}_nosim", "2026-09-21_10_14_52"),
    ("gemini-3.5, expert prompt", f"exp_gemini-35-fl_proexp05{SUFFIX}_t0_nosim",
     "2026-09-16_22_20_25"),
]
SIZES = [40, 50, 60, 81, 100, 150, 200, 300, 400, 600]
REPLICATES = 200
SEED = 2026
# Categorical slots 1-3 of the reference palette (not transport modes: no mode colour).
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a"]

# In the papers repository (PAPER_DIR, ticket 115).
OUT_PNG = sortie_papier("article-court", "appendices", "images", "D1_finite_sample_bias.png")
OUT_JSON = REPO / "scripts/annexes/data/fig_D_finite_sample_bias.json"


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    k = (len(ordered) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def measure_arm(scorer, label: str, exp: str, execution: str, rng: random.Random) -> dict:
    folder = EXPERIENCES / exp / "executions" / execution
    stored = json.loads((folder / "scores.json").read_text(encoding="utf-8"))
    synthese = json.loads((folder / "synthese.json").read_text(encoding="utf-8"))
    cerema = frames.load_cerema(S._resoudre_cerema(synthese))
    rows, _ = S.lire_perimetre(folder, S.EXCLURE_METHODES)
    frame = frames.simulation_frames(rows)["attendu"]
    full = scorer.score(frame, cerema)["emd_jsd"]["composite"]
    expected = stored["composite"]["emd_jsd"]
    if abs(full - expected) > 1e-6:
        log.error("[ALARME] %s: rescored composite %.6f differs from scores.json %.6f (%s)",
                  label, full, expected, folder)
        raise SystemExit(1)
    persons = sorted({r["agent_id"] for r in frame})
    by_person: dict[str, list[dict]] = {}
    for r in frame:
        by_person.setdefault(r["agent_id"], []).append(r)
    log.info("%s: %d persons, %d frame rows, full composite %.3f (matches scores.json)",
             label, len(persons), len(frame), full)

    points = []
    for n in SIZES:
        t0 = time.monotonic()
        values, undefined = [], 0
        for _ in range(REPLICATES):
            keep = rng.sample(persons, n)
            sub = [r for p in keep for r in by_person[p]]
            values.append(scorer.score(sub, cerema)["emd_jsd"]["composite"])
            _, mesure = scorer.primary.compute_detailed(scorer._pd.DataFrame(sub), cerema)
            undefined += bool(mesure.undefined)
        excess = [v - full for v in values]
        points.append({
            "n_persons": n,
            "mean_composite": statistics.fmean(values),
            "mean_excess": statistics.fmean(excess),
            "p05_excess": percentile(excess, 0.05),
            "p95_excess": percentile(excess, 0.95),
            "replicates_with_unmeasured_dimension": undefined,
        })
        log.info("%s N=%d: mean excess %+.2f [%+.2f; %+.2f], %d/%d replicates with an "
                 "unmeasured dimension, %.1f s", label, n, points[-1]["mean_excess"],
                 points[-1]["p05_excess"], points[-1]["p95_excess"], undefined,
                 REPLICATES, time.monotonic() - t0)
    return {"label": label, "experiment": exp, "execution": execution,
            "n_persons_full": len(persons), "full_composite": full, "points": points}


def draw(results: list[dict]) -> None:
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
    for colour, arm in zip(COLOURS, results):
        xs = [p["n_persons"] for p in arm["points"]] + [arm["n_persons_full"]]
        ys = [p["mean_excess"] for p in arm["points"]] + [0.0]
        lo = [p["p05_excess"] for p in arm["points"]] + [0.0]
        hi = [p["p95_excess"] for p in arm["points"]] + [0.0]
        if arm is results[0]:  # one band only: three overlapping bands blur each other
            ax.fill_between(xs, lo, hi, color=colour, alpha=0.15, linewidth=0,
                            label="LightGBM, 5-95% of replicates")
        ax.plot(xs, ys, color=colour, linewidth=2, marker="o", markersize=3.5,
                label=f"{arm['label']} (all {arm['n_persons_full']}: "
                      f"{arm['full_composite']:.2f})")
    ax.axvline(81, color="#8a8984", linewidth=1, linestyle=(0, (3, 3)))
    ax.text(84, ax.get_ylim()[1] * 0.93, "81 persons", color="#52514e", fontsize=8)
    ax.axhline(0, color="#8a8984", linewidth=0.8)
    ax.set_xscale("log")
    ticks = [40, 81, 150, 300, 600, 868]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Persons scored (drawn without replacement, decisions unchanged)")
    ax.set_ylabel("Composite excess over all persons (points)")
    ax.grid(axis="y", color="#e4e3df", linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    exiger_depot_papiers(OUT_PNG)
    if "--redraw" in sys.argv:  # redraw from the saved measurements, no rescoring
        results = json.loads(OUT_JSON.read_text(encoding="utf-8"))["arms"]
        draw(results)
        log.info("success: redrew %s from %s", OUT_PNG,
                 OUT_JSON.relative_to(REPO))
        return
    for noisy in ("experiences", "scripts"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    t0 = time.monotonic()
    log.info("start: %d executions, sizes %s, %d replicates, seed %d",
             len(ARMS), SIZES, REPLICATES, SEED)
    scorer, error = S.scorer_pour(F.charger().reference)
    if scorer is None:
        log.error("[ALARME] loss engine unavailable: %s", error)
        raise SystemExit(1)
    rng = random.Random(SEED)
    results = [measure_arm(scorer, *arm, rng) for arm in ARMS]
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "script": "scripts/annexes/fig_D_finite_sample_bias.py",
        "formula": "v1_reference", "seed": SEED, "replicates": REPLICATES,
        "sampling": "persons without replacement; all scored decisions of a drawn person kept",
        "arms": results,
    }, indent=2), encoding="utf-8")
    draw(results)
    log.info("success: wrote %s and %s in %.0f s", OUT_PNG,
             OUT_JSON.relative_to(REPO), time.monotonic() - t0)


if __name__ == "__main__":
    main()
