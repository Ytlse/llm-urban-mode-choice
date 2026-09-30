"""Paired differences between decision arms, with person-level cluster bootstrap.

Recomputes the paired intervals reported in the paper (Section 6 and Appendix H) and the
per-arm marginal half-widths, from the archived executions alone. No LLM call is made.

Method, identical for every preset:

* scorer: the official reference formula (``experiences.formule``), composite EMD-JSD and
  global L1, against the CEREMA EMC² 2023 reference (``SC.CEREMA_DEPOT``);
* trips read with ``frames.read_moves(..., first_day_only="auto")``, the same reading as
  the platform;
* resampling: persons (clusters) drawn with replacement among the persons common to every
  arm, ``B`` replicates, seed 2026. Each replicate scores every arm on the same draw, so a
  pair difference and a per-arm value come from the same resampled population.

Each arm points to one execution. Presets pin the execution timestamp that the paper used:
taking "the latest execution" silently changes the result as soon as an arm is replayed.

Usage::

    python -m scripts.analysis.paired_intervals --preset chapter6 --out <dir> [-B 2000]
    python -m scripts.analysis.paired_intervals --preset jev_mutations --out <dir>
    python -m scripts.analysis.paired_intervals --preset jev32_vs_tabular --out <dir>

``--experiences`` points to the directory holding the ``exp_*`` folders (default: the
public archive of cohort c1). ``moves.csv`` may be stored gzip-compressed (``moves.csv.gz``).
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import logging
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "services" / "llm-agents"))
sys.path.insert(0, str(REPO_ROOT))

from experiences import formule as F  # noqa: E402
from experiences import score as SC  # noqa: E402
from scripts.synthesis import frames  # noqa: E402

log = logging.getLogger("paired_intervals")

DEFAULT_EXPERIENCES = REPO_ROOT / "archive" / "1_regime_nominal" / "jeu_1000_PANEL_v6_EN_c" / "experiences"
SEED = 2026
SUF = "_jtir_pop-1000_PANEL_v6_jeu-20260316_EN_c"

# arm -> "experience/executions/<timestamp>"
PRESETS: dict[str, dict] = {
    # Section 6 and Appendix H: twelve arms, twenty pairs (run of 2026-09-21, ticket 096 lot 2).
    "chapter6": {
        "arms": {
            "g35_min": f"exp_gemini-35-fl_promin02{SUF}_t0_nosim/executions/2026-09-16_19_05_45",
            "g35_exp": f"exp_gemini-35-fl_proexp05{SUF}_t0_nosim/executions/2026-09-16_22_20_25",
            "g31_min": f"exp_gemini-31-fl_promin02{SUF}_t0_nosim/executions/2026-09-17_00_04_49",
            "g31_exp": f"exp_gemini-31-fl_proexp05{SUF}_t0_nosim/executions/2026-09-17_00_51_26",
            "mis_min": f"exp_mistral-l-25_promin02{SUF}_t0_nosim/executions/2026-09-17_01_13_43",
            "mis_exp": f"exp_mistral-l-25_proexp05{SUF}_t0_nosim/executions/2026-09-17_02_47_58",
            "lgbm": f"exp_lgbm{SUF}_nosim/executions/2026-09-16_13_40_53",
            "klr": f"exp_klr{SUF}_nosim/executions/2026-09-16_13_38_33",
            "rf": f"exp_rf{SUF}_nosim/executions/2026-09-16_15_52_59",
            "mnl": f"exp_mnl{SUF}_nosim/executions/2026-09-16_13_43_16",
            "jev_min": f"exp_jev-1130_promin02{SUF}_nosim/executions/2026-09-21_06_27_05",
            "jev_exp": f"exp_jev-1130_proexp05{SUF}_nosim/executions/2026-09-21_06_25_29",
        },
        "pairs": [
            ("jev_exp", "jev_min"),
            ("jev_exp", "lgbm"), ("jev_exp", "klr"), ("jev_exp", "rf"), ("jev_exp", "mnl"),
            ("jev_exp", "g35_exp"),
            ("g35_exp", "g35_min"), ("g31_exp", "g31_min"), ("mis_exp", "mis_min"),
            ("g35_exp", "lgbm"), ("g35_exp", "klr"), ("g35_exp", "rf"), ("g35_exp", "mnl"),
            ("g31_exp", "lgbm"), ("mis_exp", "lgbm"),
            ("g31_exp", "g35_exp"), ("g31_min", "g35_min"), ("mis_exp", "g35_exp"),
            ("g31_exp", "mis_exp"), ("g31_min", "mis_min"),
        ],
    },
    # Prompt mutations on the Jev arm. T2 - T1 is the noise floor: the same prompt run twice.
    "jev_mutations": {
        "arms": {
            "T1": f"exp_jev-1130_proexp05{SUF}_nosim/executions/2026-09-21_06_25_29",
            "T2": f"exp_jev-1130_proexp05{SUF}_nosim/executions/2026-09-21_09_35_52",
            "A1": f"exp_jev-1130_proexp31{SUF}_nosim",
            "A2": f"exp_jev-1130_proexp32{SUF}_nosim",
            "B1": f"exp_jev-1130_proexp33{SUF}_nosim",
            "B2": f"exp_jev-1130_proexp34{SUF}_nosim",
            "A3": f"exp_jev-1130_proexp35{SUF}_nosim",
            "A4": f"exp_jev-1130_proexp36{SUF}_nosim",
        },
        "pairs": [("T2", "T1"),
                  ("A1", "T1"), ("A2", "T1"), ("A3", "T1"), ("A4", "T1"),
                  ("B1", "T1"), ("B2", "T1"),
                  ("A1", "A2")],
    },
    # The retained mutation (proexp32) against the tabular baselines.
    "jev32_vs_tabular": {
        "arms": {
            "T1": f"exp_jev-1130_proexp05{SUF}_nosim/executions/2026-09-21_06_25_29",
            "jev32": f"exp_jev-1130_proexp32{SUF}_nosim",
            "lgbm": f"exp_lgbm{SUF}_nosim/executions/2026-09-16_13_40_53",
            "klr": f"exp_klr{SUF}_nosim/executions/2026-09-16_13_38_33",
            "rf": f"exp_rf{SUF}_nosim/executions/2026-09-16_15_52_59",
            "mnl": f"exp_mnl{SUF}_nosim/executions/2026-09-16_13_43_16",
            "g35exp": f"exp_gemini-35-fl_proexp05{SUF}_t0_nosim/executions/2026-09-16_22_20_25",
        },
        "pairs": [("jev32", "lgbm"), ("jev32", "klr"), ("jev32", "rf"), ("jev32", "mnl"),
                  ("jev32", "g35exp"), ("jev32", "T1")],
    },
}

METRICS = ("comp", "hcu", "l1")  # composite EMD-JSD, same outside captive users, global L1


def resolve_execution(experiences: Path, spec: str) -> Path:
    """An arm spec is ``experience`` (single execution expected) or ``experience/executions/<ts>``."""
    path = experiences / spec
    if "/executions/" in spec:
        if not path.is_dir():
            raise FileNotFoundError(f"execution not found: {path}")
        return path
    candidates = sorted(p for p in (path / "executions").glob("*") if p.is_dir())
    if not candidates:
        raise FileNotFoundError(f"no execution under {path}/executions")
    if len(candidates) > 1:
        log.warning("%s has %d executions, using the latest (%s): pin it in the preset",
                    spec, len(candidates), candidates[-1].name)
    return candidates[-1]


def moves_file(execution: Path, workdir: Path) -> Path:
    """Return a readable moves.csv, decompressing moves.csv.gz into ``workdir`` if needed."""
    plain = execution / "moves.csv"
    if plain.is_file():
        return plain
    packed = execution / "moves.csv.gz"
    if not packed.is_file():
        raise FileNotFoundError(f"neither moves.csv nor moves.csv.gz in {execution}")
    target = workdir / execution.parent.parent.name / execution.name / "moves.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(packed, "rb") as src, target.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    return target


def summarize(values: list[float]) -> dict:
    v = sorted(values)
    n = len(v)
    mean = sum(v) / n
    sd = (sum((x - mean) ** 2 for x in v) / (n - 1)) ** 0.5 if n > 1 else 0.0
    q = lambda p: v[int(p * (n - 1))]  # noqa: E731 - same quantile rule as the paper runs
    return {"moy": mean, "sd": sd, "lo": q(0.025), "hi": q(0.975),
            "P": sum(1 for x in v if x > 0) / n}


def run(preset: str, experiences: Path, replicates: int, out: Path) -> Path:
    config = PRESETS[preset]
    started = time.time()
    log.info("start preset=%s replicates=%d seed=%d experiences=%s", preset, replicates, SEED, experiences)

    registry = F.charger()
    scorer, _ = SC.scorer_pour(registry.reference)
    cerema = frames.load_cerema(SC.CEREMA_DEPOT)

    def measures(rows: list[dict]) -> tuple[float, float, float]:
        s = scorer.score(frames.simulation_frames(rows)["attendu"], cerema)
        outside = [r for r in rows if r["probas"]]
        so = scorer.score(frames.simulation_frames(outside)["attendu"], cerema)
        return s["emd_jsd"]["composite"], so["emd_jsd"]["composite"], s["l1_composite"]["global"]

    data: dict[str, dict[str, list[dict]]] = {}
    executions: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="paired_moves_") as tmp:
        for arm, spec in config["arms"].items():
            execution = resolve_execution(experiences, spec)
            rows, _ = frames.read_moves(moves_file(execution, Path(tmp)), SC.EXCLURE_METHODES,
                                        first_day_only="auto")
            by_person = collections.defaultdict(list)
            for row in rows:
                by_person[row["agent_id"]].append(row)
            data[arm] = by_person
            executions[arm] = f"{execution.parent.parent.name}/{execution.name}"
            log.info("arm %-8s %5d persons %6d trips (%s)", arm, len(by_person), len(rows), execution.name)

    common = sorted(set.intersection(*(set(v) for v in data.values())))
    if not common:
        log.error("[ALARME] no person common to all arms of preset %s: %s", preset,
                  {a: len(v) for a, v in data.items()})
        raise SystemExit(2)
    dropped = {a: len(v) - len(common) for a, v in data.items() if len(v) != len(common)}
    log.info("%d common persons, %d replicates, %d pairs; persons outside the common set: %s",
             len(common), replicates, len(config["pairs"]), dropped or "none")

    pair_draws = {p: {m: [] for m in METRICS} for p in config["pairs"]}
    arm_draws = {a: {m: [] for m in METRICS} for a in data}
    rng = random.Random(SEED)
    loop_start = time.time()
    for b in range(replicates):
        draw = rng.choices(common, k=len(common))
        cache = {arm: measures([r for p in draw for r in data[arm].get(p, [])]) for arm in data}
        for arm, values in cache.items():
            for i, m in enumerate(METRICS):
                arm_draws[arm][m].append(values[i])
        for x, y in config["pairs"]:
            for i, m in enumerate(METRICS):
                pair_draws[(x, y)][m].append(cache[x][i] - cache[y][i])
        if (b + 1) % 100 == 0 or b + 1 == replicates:
            log.info("  %d/%d replicates, %.0fs", b + 1, replicates, time.time() - loop_start)

    pairs = {f"{x}-{y}": {m: summarize(v) for m, v in pair_draws[(x, y)].items()}
             for x, y in config["pairs"]}
    arms = {}
    for arm, draws in arm_draws.items():
        arms[arm] = {m: summarize(v) for m, v in draws.items()}
        arms[arm]["comp_half_width"] = (arms[arm]["comp"]["hi"] - arms[arm]["comp"]["lo"]) / 2
    for name, r in pairs.items():
        log.info("%-18s comp d=%+.2f CI[%+.2f,%+.2f] | L1 d=%+.1f CI[%+.1f,%+.1f]", name,
                 r["comp"]["moy"], r["comp"]["lo"], r["comp"]["hi"],
                 r["l1"]["moy"], r["l1"]["lo"], r["l1"]["hi"])

    out.mkdir(parents=True, exist_ok=True)
    target = out / f"paired_{preset}_B{replicates}.json"
    target.write_text(json.dumps({
        "preset": preset, "replicates": replicates, "seed": SEED,
        "common_persons": len(common), "executions": executions,
        "pairs": pairs, "arms": arms,
    }, indent=1), encoding="utf-8")
    log.info("done in %.0fs: %d pairs, %d arms written to %s",
             time.time() - started, len(pairs), len(arms), target)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--preset", choices=sorted(PRESETS), required=True)
    parser.add_argument("--experiences", type=Path, default=DEFAULT_EXPERIENCES,
                        help="directory holding the exp_* folders (default: %(default)s)")
    parser.add_argument("-B", "--replicates", type=int, default=2000)
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args.preset, args.experiences, args.replicates, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
