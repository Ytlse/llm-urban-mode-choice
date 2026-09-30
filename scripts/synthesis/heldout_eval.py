"""Evaluates the pinned lineage on a frozen set NEVER SEEN by the loop (action A4).

    python -m scripts.synthesis.heldout_eval [--dataset test] [--dry-run] [--provider …]

**The problem.** The whole calibration was optimised *and* measured on ``train``
(and on its ``screen`` subsample). The store held no evaluation
on ``test``: the figure that says what the prompt is worth **outside what it
was optimised on** did not exist. A training composite score is not a
publishable result — it does not tell a prompt that understood the population
from a prompt that memorised 298 personas.

**What this script does.** It replays the nodes of the lineage pinned in
``sources.yaml`` on the requested set (``test`` by default), under the
evaluation regime of ``run.yaml``, and writes the results **into the store**, at
the exact place where ``calibrate reeval`` writes them. The page then reads them through
``frames.read_store_history``, which already accepts ``train``/``val``/``test``: nothing
to copy into an intermediate file.

**What it does not do.** No home-made batch splitting nor any catch-up
loop: the engine's ``Evaluator`` handles it, with the defences of action A10
(comparison of personas sent / decisions returned, re-draw of the incomplete batch by
halves, refusal to cache an eval below the coverage floor). A3
measured 29 truncated batches out of 128 while producing its measurement: batching rewritten
for the occasion would score on a sub-population without anything flagging it.

**What it refuses to do.** Compare the ``train`` composite score with the ``test``
composite score as is. The two sets do not have the same size — 298 persons
versus 66 — and the per-stratum divergences (JSD, EMD) are biased upwards
when sizes are small: A3 measured +5.02 composite points for the
mere reduction from 881 persons to 81, with decisions unchanged. The sample-size control
that neutralises this effect is computed by the page (``build.build_size_control``), without
a single LLM call, by resampling the ``train`` decisions already stored to
the size of ``test``. This script recalls it in its summary, so that
the raw gap is not published.

Resume: evals are content-addressed in the store. A replay interrupted
by the quota resumes where it stopped — the granularity is the **node**, so
a finished node is kept; an interrupted node leaves nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .sources import REPO_ROOT, import_calibration, load_manifest

# Default set. `test` is the only set the loop has never seen: `val` is used
# for early stopping (it therefore influenced the selection of prompts), `screen`
# is a strict subset of `train`.
DEFAULT_DATASET = "test"


def dataset_store_key(split: str, version: str) -> str:
    """Name under which a frozen-set eval is stored in the store.

    Qualified by the version as soon as we leave v1: the store indexes on the
    split name alone, and "test" designates two different sets in v1 and in v2
    (run weather vs weather drawn within the year). v1 keeps the bare name so that
    evals already paid for remain retrievable.
    """
    return split if version in ("", "v1") else f"{split}@{version}"

# Batches of 8 personas — same reason as for action A3: at 15 (capacity inferred
# from the provider), the model returns valid JSON but missing personas. The
# splitting does not enter ``eval_params_key``: it changes the number of calls,
# not the measurement.
DEFAULT_BATCH = 8

# Nodes measured by default: the two ends of the lineage. It is the pair that
# the page contrasts everywhere else (synthesis matrix, common set), and the only one
# whose cost surely fits in a quota remainder. ``--all`` measures the
# whole chain when the bucket allows it.
NODE_CHOICES = ("ends", "all")


def select_nodes(chain: list[str], which: str) -> list[dict]:
    """Nodes to measure in the seed → leaf chain, with their role.

    Kept apart from the CLI to be testable without store or provider. A chain of a
    single node has no ends to contrast: it is an error, not a degraded
    measurement.
    """
    if len(chain) < 2:
        raise ValueError(f"The lineage has only {len(chain)} node(s): "
                         f"no seed to set against the leaf.")
    if which == "ends":
        picked = [(0, chain[0]), (len(chain) - 1, chain[-1])]
    elif which == "all":
        picked = list(enumerate(chain))
    else:
        raise ValueError(f"Unknown selection: {which!r} "
                         f"(expected: {', '.join(NODE_CHOICES)})")
    out = []
    for rank, node_hash in picked:
        if rank == 0:
            role, label = "seed", "Graine"
        elif rank == len(chain) - 1:
            role, label = "leaf", "Meilleur prompt"
        else:
            role, label = "step", f"Étape {rank}"
        out.append({"role": role, "label": label, "node": node_hash,
                    "short": node_hash[:8], "rank": rank,
                    "n_nodes_in_lineage": len(chain)})
    return out


def dataset_profile(dataset_dir: Path, splits=("train", "val", "test", "screen")) -> dict:
    """What the frozen sets really are: sizes, overlap, content.

    Three facts are established **from the evidence** rather than assumed, because they
    change the meaning of the word "generalisation":

    - ``agents_shared_with_train``: if the test shared its persons with the
      train, generalisation would bear on *trips* and not on
      *individuals* — a much weaker claim;
    - ``n_agents``: it is the size, hence the factor that biases the per-stratum
      divergences upwards. A level gap between two sets of different
      sizes proves nothing as long as it is not neutralised;
    - ``with_memory``: the share of records carrying the ``**Historique :**`` section.
      ``calibration.datasets`` removes it from ``val`` and ``test`` (the STM/LTM
      memory of the source run is not reproducible) and keeps it in
      ``train``. The two sets therefore do not present the same input *form* to the
      model, and the reader must know it before attributing a gap to the prompt.
    """
    profile: dict[str, dict] = {}
    agents_by_split: dict[str, set] = {}
    for split in splits:
        path = Path(dataset_dir) / f"{split}.jsonl"
        if not path.exists():
            continue
        records = [json.loads(line)
                   for line in path.read_text(encoding="utf-8").splitlines()
                   if line.strip()]
        agents = {str(r["agent_id"]) for r in records}
        agents_by_split[split] = agents
        with_memory = sum(1 for r in records
                          if "**Historique" in (r.get("section") or ""))
        profile[split] = {
            "n_records": len(records),
            "n_agents": len(agents),
            "with_memory": with_memory,
            "memory_share": with_memory / len(records) if records else None,
        }
    train_agents = agents_by_split.get("train", set())
    for split, entry in profile.items():
        shared = len(agents_by_split[split] & train_agents) if split != "train" else None
        entry["agents_shared_with_train"] = shared
    return profile


def split_rule(dataset_dir: Path) -> Optional[str]:
    """Split rule declared by the frozen sets' manifest."""
    path = Path(dataset_dir) / "manifest.yaml"
    if not path.exists():
        return None
    import yaml
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data.get("split_rule")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="sources manifest (default: sources.yaml)")
    parser.add_argument("--run-config", default="run.yaml",
                        help="calibration engine config (default: run.yaml)")
    parser.add_argument("--dataset", default=DEFAULT_DATASET,
                        help=f"frozen set to measure (default: {DEFAULT_DATASET})")
    parser.add_argument("--nodes", default="ends", choices=NODE_CHOICES,
                        help="ends: seed and leaf (default); all: the whole lineage")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH,
                        help=f"personas per request (default: {DEFAULT_BATCH})")
    parser.add_argument("--workers", type=int, default=0,
                        help="requests in flight (default: eval_workers from the config)")
    parser.add_argument("--provider", default=None,
                        help="overrides eval_provider (e.g. google_gemini31_key2: second key, "
                             "separate quota bucket — separate cache key, "
                             "reserved for measurements)")
    parser.add_argument("--dry-run", action="store_true",
                        help="prints the plan and the exact cost, without any LLM call")
    args = parser.parse_args(argv)

    manifest = load_manifest(args.config)
    repo = manifest.get("arms.calibration.repo", "prompt_calibration")
    calibration, engine_error = import_calibration(repo)
    if calibration is None:
        print(f"[erreur] {engine_error}", file=sys.stderr)
        return 2

    pinned = manifest.get("arms.calibration.lineage") or {}
    leaf = pinned.get("leaf")
    if not leaf:
        print("[erreur] No pinned lineage leaf "
              "(arms.calibration.lineage.leaf).", file=sys.stderr)
        return 2

    dataset_dir = manifest.path_of("arms.calibration.datasets")
    if dataset_dir and dataset_dir.exists():
        rule = split_rule(dataset_dir)
        profile = dataset_profile(dataset_dir)
        print(f"Frozen sets: {dataset_dir.relative_to(REPO_ROOT)}")
        if rule:
            print(f"Split rule: {rule}")
        for split, entry in profile.items():
            shared = entry["agents_shared_with_train"]
            shared_txt = ("—" if shared is None
                          else f"{shared} personne(s) en commun avec le train")
            memory = entry["memory_share"]
            memory_txt = ("historique absent" if not memory
                          else f"historique sur {memory:.0%} des records")
            print(f"  {split:<6} {entry['n_records']:>4} decisions, "
                  f"{entry['n_agents']:>3} persons — {shared_txt}, {memory_txt}")

    engine_root = REPO_ROOT / repo
    cwd = Path.cwd()
    os.chdir(engine_root)
    try:
        from calibration.cli import (_fmt_local, _load_dotenv, _resume_after,
                                     build_engine, load_records)
        from calibration.evaluation import (EvaluationAborted, InsufficientCoverage,
                                            batches_from_records)
        from calibration.models import RunConfig
        from calibration.reeval import lineage_chain, resolve_node
        from calibration.store import RunStore

        _load_dotenv()
        config = RunConfig.from_yaml(Path(args.run_config))

        # ── Align the frozen-set version on the one pinned by the page ───────
        # `run.yaml` is the config of the calibration LOOP: it keeps
        # pointing to the version the campaign was optimised on, and this ticket
        # does not touch it. The page, for its part, pins the version it displays
        # (`arms.calibration.datasets`). Without this alignment, `make heldout-eval`
        # measured v1 while the page announced v2 — and nothing would have
        # flagged it, both versions carrying the same split names.
        if dataset_dir and dataset_dir.exists():
            config.dataset_dir = dataset_dir.parent
            config.dataset_version = dataset_dir.name
            print(f"\n  📂 sets: {config.dataset_version} (pinned by the manifest; "
                  f"run.yaml has {RunConfig.from_yaml(Path(args.run_config)).dataset_version} "
                  f"for the optimisation loop)")

        if args.provider:
            print(f"\n  🔑 eval: provider {args.provider} (instead of "
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
            chain = lineage_chain(store, resolve_node(store, leaf))
            plan = select_nodes(chain, args.nodes)
            records = load_records(config, args.dataset)
            # The store indexes an eval on (node × SET NAME × params): the
            # frozen-set version is not part of it, and the v1 and v2 splits
            # carry the same name. Without qualification, a v1 measurement would be
            # served as is for a v2 request — zero calls, and a
            # figure labelled with the wrong weather regime. v1 keeps the bare name,
            # so that evals already paid for stay readable.
            dataset_key = dataset_store_key(args.dataset, config.dataset_version)
            params_key = config.eval_params_key()
            n_batches = len(batches_from_records(
                records, config.eval_batch_max,
                prod_option_handling=config.prod_option_handling))
            for entry in plan:
                entry["cached"] = store.cached_eval(
                    entry["node"], dataset_key, params_key) is not None
            to_pay = [p for p in plan if not p["cached"]]

            print()
            print(f"Lineage    : {chain[0][:8]} → {chain[-1][:8]} ({len(chain)} nodes) — "
                  f"{len(plan)} measured ({args.nodes})")
            print(f"Set        : {dataset_key} ({len(records)} decisions, "
                  f"{len({r['agent_id'] for r in records})} persons)")
            print(f"Eval key   : {params_key}")
            print(f"To pay     : {len(to_pay)}/{len(plan)} eval(s) × {n_batches} batch(es) "
                  f"= {len(to_pay) * n_batches} LLM call(s) before re-draws "
                  f"(+~16 % measured by A10 → ≈ {round(len(to_pay) * n_batches * 1.16)}), "
                  f"batches of {config.eval_batch_max} personas, {config.eval_rpm} req/min "
                  f"→ ≳ {len(to_pay) * n_batches / max(1, config.eval_rpm):.0f} min")
            for entry in plan:
                print(f"  {'cache' if entry['cached'] else 'à payer':>7}  "
                      f"{entry['short']} — {entry['label']}")

            if args.dry_run:
                print("\n--dry-run: no LLM call issued.")
                return 0

            paid = 0
            for i, entry in enumerate(plan, 1):
                if entry["cached"]:
                    print(f"  [{i}/{len(plan)}] {entry['short']} — cache")
                    continue
                blocks = store.node_blocks(entry["node"])
                if blocks is None:
                    print(f"  [{i}/{len(plan)}] {entry['short']} — blocks not found, "
                          f"skipped")
                    continue
                try:
                    result, _df = evaluator.evaluate(
                        entry["node"], blocks, dataset_key, records,
                        desc=f"{dataset_key} {entry['short']}")
                except EvaluationAborted as exc:
                    # Same behaviour as `calibrate reeval`: we PERSIST the resume
                    # date rather than let the next command hammer an
                    # exhausted API. The Google free tier daily bucket
                    # resets at **Pacific** midnight, not at UTC midnight.
                    resume, reason = _resume_after(config, exc)
                    guard = RunStore(config.store_path)
                    guard.set_cooldown(resume, reason)
                    guard.close()
                    print(f"\n⏸️  {exc}\n   Resume allowed {_fmt_local(resume)} "
                          f"({reason}).\n   Rerun the same command after the reset: "
                          f"nodes already paid for are served by the cache.")
                    return 2
                except InsufficientCoverage as exc:
                    # Refusal BEFORE writing: better a missing node than a
                    # composite score computed on a sub-population.
                    print(f"  🚨 [ALARME] [{i}/{len(plan)}] {entry['short']} — {exc}")
                    continue
                paid += 1
                print(f"  [{i}/{len(plan)}] {entry['short']} — engine composite "
                      f"{result.scores.composite:.2f}")
        finally:
            store.close()
    finally:
        os.chdir(cwd)

    print(f"\nWritten to the store: {paid} eval(s) on « {dataset_key} ».")
    print("⚠ Do not publish the raw train → test gap: the two sets do not have the "
          "same size, and the per-stratum divergences are biased upwards "
          "at small sizes. The sample-size control is computed by the page.")
    print("Regenerate the page: make synthesis")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
