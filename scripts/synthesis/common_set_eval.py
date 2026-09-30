"""Re-evaluates seed and best prompt on the common set (action A3).

    python -m scripts.synthesis.common_set_eval [--dry-run] [--provider …]

**The problem.** Strand 2 of the synthesis page is scored on the *frozen
personas* (``calibration_datasets/v1``), i.e. a subset of an earlier March
run, while strand 1 is scored on the run pinned in
``sources.yaml``. The two columns therefore do not bear on the same population:
comparing them amounts to comparing two measurements made on two substrates.

**What this script does.** It replays the two ends of the pinned lineage —
the seed and the leaf — on a sample **of the pinned run**, under the pinned
evaluation regime, and writes the decisions obtained in the format the page
consumes (``arms.calibration.common_set_eval`` of the manifest).

**What it does not do.** No home-made batch splitting nor any catch-up
loop: the calibration engine's ``Evaluator`` handles it, with the defences
set up by action A10 (comparison of personas sent / decisions returned, re-draw
of the incomplete batch by halves, refusal to cache an eval below the coverage
floor). Re-implementing this splitting would reintroduce exactly the defect
that A10 just fixed — a measurement computed on a sub-population, without
anything flagging it.

Resume: evals are cached in the store, content-addressed. A
replay interrupted by the quota resumes where it stopped; an already
complete replay costs no call.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .sources import REPO_ROOT, import_calibration, load_manifest

# ── Common-set sample — FROZEN rule ──────────────────────────────────────────
#
# Same logic as the frozen sets of ``calibration_datasets/v1`` (cf.
# ``calibration/datasets.py``): assignment **per person**, by a stable sha256 hash
# of the ``agent_id`` — never ``hash()``, salted per interpreter — then
# coverage report on the Cerema strata. All decisions of a
# selected person are kept, none is cut in two.
#
# A single thing changes, and it is deliberate: the hash is **namespaced**. Reusing
# ``sha256(agent_id) % 100 < k`` as is would pick a prefix of the train
# interval ([0, 70)) — the sample would then be made 100 % of personas from the
# split the calibration was optimised on, which would flatter the leaf.
# The namespace prefix decorrelates the draw from the train/val/test split: the
# sample composition mirrors that of the population (≈ 70/15/15).
SAMPLE_NAMESPACE = "common_set_v1"
SAMPLE_MODULUS = 1000

# Frozen threshold. Chosen by increasing sweep: it is the smallest threshold whose
# engine coverage report is **clean**, i.e. all the Cerema
# strata present in the run reach ``COVERAGE_MIN_COUNT`` (5).
# Below it (k=83, 424 decisions), the 70-74 age band is empty: the `age` dimension
# of the composite score would be computed on a truncated support, hence not comparable to
# strand 1. The threshold is FIXED here rather than recomputed at each run — otherwise
# the slightest added data would silently change the sample.
SAMPLE_BUCKET_MAX = 99

# Set name under which evals are cached in the store. Distinct from
# train/val/test on purpose: the page only reads those three for the trajectory
# of prompts (``frames.read_store_history``), so these evals cannot mix
# with the calibration curves.
DATASET_NAME = "common_set_v1"

# The store cache indexes an eval on (node × SET NAME × params) — the run
# is not part of it. Under a fixed name, changing the pinned run thus served again the
# measurement of the previous run, relabelling it with the description of the new one:
# 0 call paid, composite scores unchanged to the hundredth, and a file that claimed
# to describe 383 decisions of the new run while carrying 762 of the old one.
# The set name now carries the fingerprint of the records actually submitted: two
# distinct runs can no longer share a cache entry, and rerunning on the
# same run stays free. The suffix stays outside train/val/test, hence invisible
# to the calibration curves.
CACHE_DIGEST_CHARS = 12

# Batches of 8 personas. At 15 (capacity inferred from the provider), the model returns
# valid JSON but missing personas — measured by A10: 4 batches out of 12 incomplete.
# Does not change the measurement (the splitting does not enter ``eval_params_key``),
# only the number of calls.
DEFAULT_BATCH = 8

SCHEMA = "calibration_on_common_set/v1"

# Columns written for each decision: exactly the scoring frame of the
# page (``frames.decisions_frame``), plus the raw mode returned by the model.
COLUMNS = ["agent_id", "mode", "mode_cat", "weight",
           "genre", "age_cat", "occupation", "motif", "dist_cat"]


def sample_bucket(agent_id: str) -> int:
    """Stable bucket [0, ``SAMPLE_MODULUS``) of a person in the sample."""
    digest = hashlib.sha256(f"{SAMPLE_NAMESPACE}:{agent_id}".encode()).hexdigest()
    return int(digest, 16) % SAMPLE_MODULUS


def in_sample(agent_id: str) -> bool:
    return sample_bucket(agent_id) < SAMPLE_BUCKET_MAX


def sample_rule() -> str:
    return (f'sha256("{SAMPLE_NAMESPACE}:" + agent_id) % {SAMPLE_MODULUS} '
            f'< {SAMPLE_BUCKET_MAX}')


def records_digest(records: list[dict]) -> str:
    """Fingerprint of the records submitted to the model: identities + frozen text.

    It is the ``section`` text that actually goes into the request (persona,
    context, itinerary options): two runs that produced neither the
    same trips nor the same contexts cannot be confused by it. Sorted, hence
    independent of the log reading order.
    """
    h = hashlib.sha256()
    for agent_id, section in sorted((str(r.get("agent_id", "")),
                                     str(r.get("section", ""))) for r in records):
        h.update(agent_id.encode("utf-8"))
        h.update(b"\x00")
        h.update(section.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:CACHE_DIGEST_CHARS]


def cache_dataset(digest: str) -> str:
    """Set name of the eval cache, carrying the sample fingerprint."""
    return f"{DATASET_NAME}@{digest}"


def build_sample(run_dir: Path) -> tuple[list[dict], dict]:
    """Decisions of the pinned run kept in the sample, + description.

    The records are built by the engine itself (``build_decision_records``),
    hence in the exact format the evaluator expects: the run's persona text, its
    weather context, its itinerary options, and the traits joined from
    ``population_*.json``. A section that cannot be attached is an anomaly, not a
    silently lost line — we then refuse to produce the sample.
    """
    from calibration.datasets import coverage_report, split_of
    from calibration.evaluation import parse_option_modes
    from calibration.exchanges import itinerary_entries
    from calibration.metadata import build_decision_records, load_population

    exchanges = run_dir / "llm_exchanges.jsonl"
    candidates = sorted(p for p in run_dir.glob("population_*[0-9].json"))
    if not exchanges.exists() or not candidates:
        raise FileNotFoundError(
            f"Run {run_dir} does not carry the two required sources: "
            f"llm_exchanges.jsonl and population_*.json.")
    entries = itinerary_entries(exchanges)
    # Same cut as strands 1 and 3 (ticket 008, A6.b): the first simulated day,
    # and it alone. This strand does not read moves.csv — it rebuilds its sample
    # from llm_exchanges.jsonl — so the `frames.read_moves` filter would not
    # reach it. Forgetting this second entry point would give the three
    # strands different scopes, without anything flagging it.
    # `sim_day` is already present in each log record (UTC date of
    # `sim_ts`, cf. llm_gateway/telemetry/logger.py), and it is exactly the
    # convention that `frames.simulated_day` applies to the « Temps simulé » column.
    days = {e.get("sim_day") for e in entries if e.get("sim_day")}
    sim_day = min(days) if days else None
    n_entries_run = len(entries)
    if sim_day:
        entries = [e for e in entries if e.get("sim_day") == sim_day]
    traits = load_population(candidates[0])
    records, anomalies = build_decision_records(entries, traits)
    if anomalies:
        causes = Counter(a["cause"] for a in anomalies)
        raise ValueError(f"{len(anomalies)} section(s) not attached to the run "
                         f"({dict(causes)}) — sample refused.")

    kept = [r for r in records if in_sample(r["agent_id"])]
    for rec in kept:
        # Like ``cli.load_records``: option modes are extracted once from the
        # frozen text, and serve as reference to map a drawn index to a mode.
        rec["option_modes"] = parse_option_modes(rec.get("section", ""))
    unparsed = sum(1 for r in kept if not r["option_modes"])
    if unparsed:
        print(f"⚠ [ALARME] {unparsed}/{len(kept)} records without a usable "
              f"option list — the mode will fall back on the LLM label.")

    coverage, warnings = coverage_report({DATASET_NAME: kept})
    # A stratum empty in the whole run cannot be filled by any
    # sampling whatsoever: telling it apart avoids passing off a
    # property of the run as a defect of the draw. The pinned run contains no
    # trip longer than 50 km — nor does the frozen train.
    _cov_run, warnings_run = coverage_report({DATASET_NAME: records})
    empty_in_run = {w.rsplit(":", 1)[0] for w in warnings_run
                    if w.endswith("effectif 0 < 5")}
    warnings = [w + (" (strate vide dans le run entier)"
                     if w.rsplit(":", 1)[0] in empty_in_run else "")
                for w in warnings]
    agents = {r["agent_id"] for r in kept}
    digest = records_digest(kept)
    info = {
        "run": str(run_dir.relative_to(REPO_ROOT)) if run_dir.is_relative_to(REPO_ROOT)
        else str(run_dir),
        "dataset": DATASET_NAME,
        "rule": sample_rule(),
        "namespace": SAMPLE_NAMESPACE,
        "modulus": SAMPLE_MODULUS,
        "bucket_max": SAMPLE_BUCKET_MAX,
        "n_records": len(kept),
        "n_agents": len(agents),
        "n_run_records": len(records),
        "n_run_agents": len({r["agent_id"] for r in records}),
        # Time scope, to compare with that of strands 1 and 3: all three
        # must announce the same day.
        "sim_day": sim_day,
        "n_entries_run": n_entries_run,
        "n_entries_kept": len(entries),
        # Fingerprint of what is actually submitted, and set name it induces
        # in the store cache. Written in the file: the page can thus tell
        # which sample the measurement comes from, and not only which run.
        "records_digest": digest,
        "cache_dataset": cache_dataset(digest),
        # Composition in frozen splits: tells which share of the sample falls
        # in the train the calibration was optimised on. It is not a
        # leak (trips, contexts and dates come from another run),
        # but the reader must be able to judge it.
        "splits": dict(Counter(split_of(a) for a in agents)),
        "coverage": coverage[DATASET_NAME],
        "coverage_warnings": warnings,
    }
    return kept, info


def resolve_prompts(store, leaf: str) -> list[dict]:
    """Seed and leaf of the pinned lineage — the two measured ends.

    The seed is not hard-coded: it is the first node of the chain that
    the page itself displays (``reeval.lineage_chain``, which falls back on the mutation
    edges, without which the lineage loses its seed).
    """
    from calibration.reeval import lineage_chain, resolve_node

    leaf_hash = resolve_node(store, leaf)
    chain = lineage_chain(store, leaf_hash)
    if len(chain) < 2:
        raise ValueError(f"The lineage of {leaf_hash[:8]} has only one node: "
                         f"no seed to compare.")
    out = []
    for role, label, node_hash in (("seed", "Graine", chain[0]),
                                   ("leaf", "Meilleur prompt", chain[-1])):
        row = store.node(node_hash)
        out.append({"role": role, "label": label, "node": node_hash,
                    "short": node_hash[:8],
                    "branch": (row["branch"] if row is not None else "—"),
                    "n_nodes_in_lineage": len(chain)})
    return out


def _rows_from_frame(df) -> list[list]:
    """Decisions DataFrame → compact rows, in ``COLUMNS`` order.

    The evaluator's df carries metadata **per decision** (they come from the
    batch record, not from a per-agent index): a person who makes three trips
    keeps their three purposes and three distances. Writing these rows as is
    avoids the "one purpose per agent" approximation that any recomputation
    from the stored decisions alone suffers.
    """
    rows = []
    for rec in df.to_dict("records"):
        row = []
        for col in COLUMNS:
            value = rec.get(col)
            if value is not None and hasattr(value, "item"):
                value = value.item()
            row.append(value)
        rows.append(row)
    return rows


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="sources manifest (default: sources.yaml)")
    parser.add_argument("--run-config", default="run.yaml",
                        help="calibration engine config (default: run.yaml)")
    parser.add_argument("--out", help="output file (default: the manifest's one)")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH,
                        help=f"personas per request (default: {DEFAULT_BATCH})")
    parser.add_argument("--workers", type=int, default=0,
                        help="requests in flight (default: eval_workers from the config)")
    parser.add_argument("--provider", default=None,
                        help="overrides eval_provider (e.g. google_gemini31_key2: second key, "
                             "separate quota bucket — separate cache key, "
                             "reserved for measurements)")
    parser.add_argument("--dry-run", action="store_true",
                        help="prints the sample and the cost, without any LLM call")
    args = parser.parse_args(argv)

    manifest = load_manifest(args.config)
    repo = manifest.get("arms.calibration.repo", "prompt_calibration")
    calibration, engine_error = import_calibration(repo)
    if calibration is None:
        print(f"[erreur] {engine_error}", file=sys.stderr)
        return 2

    out_path = Path(args.out or manifest.get("arms.calibration.common_set_eval"))
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    run_dir = manifest.path_of("common_set.run")
    if run_dir is None or not run_dir.exists():
        print(f"[erreur] Run not found: {manifest.get('common_set.run')}",
              file=sys.stderr)
        return 2
    pinned = manifest.get("arms.calibration.lineage") or {}
    leaf = pinned.get("leaf")
    if not leaf:
        print("[erreur] No pinned lineage leaf "
              "(arms.calibration.lineage.leaf).", file=sys.stderr)
        return 2

    print(f"Pinned run  : {run_dir.relative_to(REPO_ROOT)}")
    records, info = build_sample(run_dir)
    print(f"Sample      : {info['n_records']} decisions, {info['n_agents']} persons "
          f"(out of {info['n_run_records']} / {info['n_run_agents']} in the run)")
    print(f"Frozen rule : {info['rule']}")
    if info.get("sim_day"):
        print(f"Simulated day: {info['sim_day']} "
              f"({info['n_entries_kept']}/{info['n_entries_run']} log entries) "
              f"— same cut as strands 1 and 3")
    print(f"Fingerprint  : {info['records_digest']} → cache set "
          f"{info['cache_dataset']}")
    print(f"Frozen splits of the selected persons: {info['splits']}")
    if info["coverage_warnings"]:
        print(f"⚠ {len(info['coverage_warnings'])} stratum(a) below the threshold of 5:")
        for w in info["coverage_warnings"]:
            print(f"  - {w}")
    else:
        print("Coverage: complete (all Cerema strata ≥ 5)")

    # ── Calibration engine ───────────────────────────────────────────────────
    # RunConfig carries paths relative to the calibration repository root
    # (frozen sets, store, resources of the sibling repository): we move there, once the
    # output paths are already made absolute.
    engine_root = REPO_ROOT / repo
    cwd = Path.cwd()
    results: list[dict] = []
    os.chdir(engine_root)
    try:
        from calibration.cli import (_fmt_local, _load_dotenv, _resume_after,
                                     build_engine)
        from calibration.evaluation import (EvaluationAborted, InsufficientCoverage,
                                            batches_from_records, normalize_decisions)
        from calibration.models import RunConfig
        from calibration.store import RunStore

        _load_dotenv()
        config = RunConfig.from_yaml(Path(args.run_config))
        if args.provider:
            print(f"  🔑 eval: provider {args.provider} (instead of "
                  f"{config.eval_provider}) — separate quota bucket, separate cache "
                  f"key. Acceptable for a MEASUREMENT (same model queried).")
            config.eval_provider = args.provider
        config.eval_batch_max = args.batch
        if args.workers:
            config.eval_workers = args.workers

        # Quota guard: an active cooldown means the bucket is exhausted. We do not
        # bypass it — we report it and stop.
        guard = RunStore(config.store_path)
        cooldown = guard.get_cooldown()
        guard.close()
        remaining = ((cooldown["resume_after"] - datetime.now(timezone.utc)).total_seconds()
                     if cooldown else 0.0)
        if remaining > 0 and not args.dry_run:
            print(f"⏸️  Cooldown quota still active for {remaining / 3600:.1f} h "
                  f"({cooldown['reason']}) — nothing to do.")
            return 0

        store, evaluator, _mut, _cerema, _seed, _train, _val, _screen = build_engine(
            config, with_mutator=False)
        try:
            prompts = resolve_prompts(store, leaf)
            params_key = config.eval_params_key()
            # EXACT number of calls (not estimated): the engine's batching function
            # gives it, the same one that will be used for the eval.
            n_batches = len(batches_from_records(
                records, config.eval_batch_max,
                prod_option_handling=config.prod_option_handling))
            # The cache set carries the sample fingerprint: an eval is only
            # served if it was paid for on EXACTLY these records.
            dataset_key = info["cache_dataset"]
            to_pay = [p for p in prompts
                      if store.cached_eval(p["node"], dataset_key, params_key) is None]
            print()
            print(f"Eval key   : {params_key}")
            print(f"Prompts    : " + ", ".join(
                f'{p["short"]} ({p["label"]}, branche {p["branch"]})' for p in prompts))
            print(f"To pay     : {len(to_pay)}/{len(prompts)} eval(s) × {n_batches} batch(es) "
                  f"= {len(to_pay) * n_batches} LLM call(s) before re-draws "
                  f"(+~16 % measured by A10 → ≈ {round(len(to_pay) * n_batches * 1.16)}), "
                  f"batches of {config.eval_batch_max} personas, {config.eval_rpm} req/min "
                  f"→ ≳ {len(to_pay) * n_batches / max(1, config.eval_rpm):.0f} min")

            if args.dry_run:
                print("\n--dry-run: no LLM call issued.")
                return 0

            results = []
            for i, prompt in enumerate(prompts, 1):
                blocks = store.node_blocks(prompt["node"])
                if blocks is None:
                    print(f"  [{i}/{len(prompts)}] {prompt['short']} — blocks "
                          f"not found, skipped")
                    continue
                try:
                    result, df = evaluator.evaluate(
                        prompt["node"], blocks, dataset_key, records,
                        desc=f"{DATASET_NAME} {prompt['short']}")
                except EvaluationAborted as exc:
                    # Same behaviour as `calibrate reeval`: we PERSIST the resume
                    # date in the store rather than let the next command
                    # hammer an exhausted API. The Google free tier daily quota
                    # resets at **Pacific** midnight, not at UTC
                    # midnight — and the `retryDelay` returned in the 429 (about thirty
                    # seconds) says nothing at all about when the bucket refills.
                    resume, reason = _resume_after(config, exc)
                    guard = RunStore(config.store_path)
                    guard.set_cooldown(resume, reason)
                    guard.close()
                    print(f"\n⏸️  {exc}\n   Resume allowed {_fmt_local(resume)} "
                          f"({reason}).\n   Rerun the same command after the reset: "
                          f"evals already paid for are served by the cache.")
                    break
                except InsufficientCoverage as exc:
                    # Refusal BEFORE writing: better a missing prompt than a
                    # composite score computed on a sub-population.
                    print(f"  🚨 [ALARME] [{i}/{len(prompts)}] {prompt['short']} — {exc}")
                    continue
                if df.empty:
                    print(f"  ⚠ [{i}/{len(prompts)}] {prompt['short']} — no decision")
                    continue
                covered = len({aid for aid, _m, _w
                               in normalize_decisions(result.decisions)})
                results.append({
                    "schema": SCHEMA,
                    "role": prompt["role"], "label": prompt["label"],
                    "node": prompt["node"], "short": prompt["short"],
                    "branch": prompt["branch"],
                    "regime": {
                        "model": config.eval_model,
                        "policy": "masse de probabilité",
                        "label": f"{config.eval_model} · masse de probabilité",
                        "params_key": params_key,
                        "provider": config.eval_provider,
                        "temperature": config.eval_temp,
                    },
                    "sample": info,
                    "n_decisions": len(result.decisions),
                    "n_agents_covered": covered,
                    "coverage": covered / max(1, info["n_agents"]),
                    "stored_composite": result.scores.composite,
                    "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "columns": COLUMNS,
                    "decisions": _rows_from_frame(df),
                })
                print(f"  [{i}/{len(prompts)}] {prompt['short']} — engine composite "
                      f"{result.scores.composite:.2f} on {covered}/{info['n_agents']} "
                      f"persons")
        finally:
            store.close()
    finally:
        os.chdir(cwd)

    if not results:
        print("\nNo usable result — output file unchanged.")
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for entry in results:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    rel = out_path.relative_to(REPO_ROOT) if out_path.is_relative_to(REPO_ROOT) else out_path
    print(f"\nWritten: {rel} ({len(results)} prompt(s))")
    print("Regenerate the page: make synthesis")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
