#!/usr/bin/env python3
"""Ablation study: Uniform Replay vs Prioritized Experience Replay (PER).

Trains a DQN agent for each combination of (replay_type × seed), evaluates
against baseline agents, and aggregates results for scientific comparison.

Usage
-----
    # Full experiment
    python scripts/run_replay_ablation.py \\
        --episodes 2000 --seeds 1 2 3 --device auto \\
        --max-moves 400 --eval-games 30 \\
        --out outputs/experiments/replay_ablation

    # Quick smoke test  (~seconds)
    python scripts/run_replay_ablation.py \\
        --episodes 5 --seeds 1 2 --device cpu \\
        --eval-games 2 --out outputs/experiments/test_replay_ablation

    # Dry-run: only print the experiment plan
    python scripts/run_replay_ablation.py --episodes 500 --seeds 1 2 3 --dry-run

    # Estimate total runtime before committing
    python scripts/run_replay_ablation.py \\
        --episodes 500 --seeds 1 2 3 --benchmark-first \\
        --out outputs/experiments/replay_ablation

Output structure
----------------
    <out>/
    ├── experiments_index.csv           ← status of every run
    ├── aggregated_learning_curves.csv  ← all train_log.csv merged
    ├── final_eval_summary.csv          ← all eval summaries merged
    ├── uniform_seed_1/
    │   ├── logs/train_log.csv
    │   ├── models/dqn_latest.pt
    │   └── eval/summary.csv
    └── prioritized_seed_1/
        ├── logs/train_log.csv
        ├── models/dqn_latest.pt
        └── eval/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from corners_rl.evaluation.aggregate_experiments import (
    aggregate_eval_summaries,
    aggregate_learning_curves,
)
from corners_rl.rl.train_dqn import ReplayConfig, SelfPlayTrainer, TrainConfig

log = logging.getLogger(__name__)

REPLAY_TYPES = ["uniform", "prioritized"]

# ── Config builders ───────────────────────────────────────────────────────────

def _replay_config(replay_type: str, episodes: int) -> ReplayConfig:
    """Return a ReplayConfig with sensible defaults scaled to episode count."""
    beta_anneal = max(200, episodes * 30)
    return ReplayConfig(
        type=replay_type,
        alpha=0.6,
        beta_start=0.4,
        beta_end=1.0,
        beta_anneal_steps=beta_anneal,
        priority_epsilon=1e-6,
    )


def _train_config(
    args: argparse.Namespace,
    replay_type: str,
    seed: int,
    run_dir: Path,
) -> TrainConfig:
    """Build a TrainConfig for one experiment cell.

    Hyperparameters scale automatically with ``--episodes`` so the same CLI
    works for smoke tests (5 episodes) and full runs (2000+ episodes).
    ``train_start_size`` is always >= ``batch_size`` to prevent sampling errors.
    """
    n = args.episodes
    # Scale batch_size down for very short smoke runs; cap at 64 for real runs
    batch_size = min(64, max(8, n))
    # Buffer must accumulate at least batch_size transitions before first update
    train_start_size = batch_size
    return TrainConfig(
        episodes=n,
        max_moves=args.max_moves,
        batch_size=batch_size,
        replay_capacity=min(100_000, max(500, n * 50)),
        train_start_size=train_start_size,
        train_every_steps=1,
        target_update_steps=min(1_000, max(20, n * 2)),
        save_every=max(1, n // 5),
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=max(50, n * 20),
        device=args.device,
        seed=seed,
        output_dir=str(run_dir),
        replay=_replay_config(replay_type, n),
    )


# ── Experiment index helpers ──────────────────────────────────────────────────

_INDEX_FIELDS = [
    "replay_type", "seed", "output_dir",
    "final_checkpoint", "status", "runtime_seconds",
]


def _write_index(path: Path, rows: list[dict]) -> None:
    """Overwrite experiments_index.csv with current accumulated rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


# ── DQN loader for eval ───────────────────────────────────────────────────────

def _load_dqn(checkpoint: Path, device: str, seed: int):
    import torch
    from corners_rl.agents.dqn_agent import DQNAgent
    from corners_rl.rl.model import DQNModel
    from corners_rl.utils.seeding import resolve_device

    # Resolve "auto" to an actual device string before passing to torch.load
    resolved = str(resolve_device(device))
    model = DQNModel()
    ckpt = torch.load(checkpoint, map_location=resolved, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    return DQNAgent(model=model, device=resolved, epsilon=0.0, seed=seed)


# ── Evaluation ────────────────────────────────────────────────────────────────

def _run_eval(args: argparse.Namespace, run_dir: Path, checkpoint: str, seed: int) -> Path:
    """Run round-robin tournament; return path to summary.csv."""
    from corners_rl.agents.greedy_agent import GreedyAgent
    from corners_rl.agents.heuristic_agent import HeuristicAgent
    from corners_rl.agents.random_agent import RandomAgent
    from corners_rl.evaluation.tournaments import round_robin_tournament

    agents = [
        RandomAgent(seed=seed),
        GreedyAgent(seed=seed),
        HeuristicAgent(seed=seed),
    ]

    ckpt_path = Path(checkpoint)
    if ckpt_path.exists():
        try:
            agents.append(_load_dqn(ckpt_path, args.device, seed))
            log.info("  DQN loaded from %s", ckpt_path)
        except Exception as exc:
            log.warning("  Could not load DQN checkpoint (%s) — eval without DQN.", exc)
    else:
        log.warning("  Checkpoint not found (%s) — eval without DQN.", ckpt_path)

    _, summary = round_robin_tournament(
        agents,
        games_per_pair=args.eval_games,
        max_moves=args.max_moves,
        seed=seed,
    )

    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    summary_path = eval_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    log.info("  Eval summary saved to %s", summary_path)
    return summary_path


# ── Single-experiment runner ──────────────────────────────────────────────────

def _run_experiment(
    args: argparse.Namespace,
    replay_type: str,
    seed: int,
    out_dir: Path,
    index_rows: list[dict],
    step: int,
    total: int,
) -> bool:
    """Run one (replay_type, seed) cell.  Returns True on success."""
    run_dir = out_dir / f"{replay_type}_seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─'*60}")
    print(f"  [{step}/{total}]  {replay_type.upper()}  seed={seed}")
    print(f"  Output: {run_dir}")
    print(f"{'─'*60}")

    t0 = time.perf_counter()
    status = "pending"
    final_ckpt = ""

    try:
        # ── Training ──────────────────────────────────────────────────────────
        cfg = _train_config(args, replay_type, seed, run_dir)
        log.info("Starting training: %s", cfg)
        trainer = SelfPlayTrainer(cfg)
        trainer.train()
        final_ckpt = str(run_dir / "models" / "dqn_latest.pt")
        print(f"  ✓  Training complete  (checkpoint: {final_ckpt})")

        # ── Evaluation ────────────────────────────────────────────────────────
        _run_eval(args, run_dir, final_ckpt, seed)
        print("  ✓  Evaluation complete")

        status = "completed"

    except Exception as exc:
        status = f"failed: {type(exc).__name__}: {exc}"
        print(f"  ✗  {status}")
        traceback.print_exc()

    runtime = round(time.perf_counter() - t0, 1)
    print(f"  ⏱  {runtime}s")

    index_rows.append({
        "replay_type":       replay_type,
        "seed":              seed,
        "output_dir":        str(run_dir),
        "final_checkpoint":  final_ckpt,
        "status":            status,
        "runtime_seconds":   runtime,
    })
    _write_index(out_dir / "experiments_index.csv", index_rows)

    return status == "completed"


# ── Benchmark ────────────────────────────────────────────────────────────────

def _run_benchmark(args: argparse.Namespace, out_dir: Path) -> float:
    """Run 1 episode of uniform/seed[0] and return seconds-per-episode."""
    seed = args.seeds[0]
    run_dir = out_dir / f"_benchmark_seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    bench_args = argparse.Namespace(**vars(args))
    bench_args.episodes = 1

    print("\n⏱  Benchmark: 1 episode (uniform, seed={})…".format(seed))
    t0 = time.perf_counter()
    try:
        cfg = _train_config(bench_args, "uniform", seed, run_dir)
        SelfPlayTrainer(cfg).train()
    except Exception as exc:
        print(f"  Benchmark failed: {exc}")
        return 0.0

    elapsed = time.perf_counter() - t0
    total_experiments = len(args.seeds) * len(REPLAY_TYPES)
    total_episodes = total_experiments * args.episodes
    estimated = elapsed * total_episodes

    print(f"  1 episode = {elapsed:.2f}s")
    print(f"  Full run estimate: {total_experiments} runs × {args.episodes} ep"
          f"  ≈  {_fmt_time(estimated)}")
    return elapsed


def _fmt_time(seconds: float) -> str:
    if seconds < 120:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}min"
    return f"{seconds/3600:.1f}h"


# ── Aggregation ───────────────────────────────────────────────────────────────

def _aggregate(out_dir: Path, index_rows: list[dict]) -> None:
    """Merge all completed run outputs into aggregated CSVs."""
    completed = [r for r in index_rows if r["status"] == "completed"]
    if not completed:
        print("\n  No completed runs — nothing to aggregate.")
        return

    curve_runs = [
        {
            "replay_type": r["replay_type"],
            "seed":        r["seed"],
            "log_path":    Path(r["output_dir"]) / "logs" / "train_log.csv",
        }
        for r in completed
    ]
    eval_runs = [
        {
            "replay_type": r["replay_type"],
            "seed":        r["seed"],
            "eval_path":   Path(r["output_dir"]) / "eval" / "summary.csv",
        }
        for r in completed
    ]

    curves = aggregate_learning_curves(curve_runs)
    evals  = aggregate_eval_summaries(eval_runs)

    curves_path = out_dir / "aggregated_learning_curves.csv"
    evals_path  = out_dir / "final_eval_summary.csv"

    if not curves.empty:
        curves.to_csv(curves_path, index=False)
        print(f"  ✓  Learning curves  → {curves_path}  ({len(curves)} rows)")
    else:
        print("  ⚠  Learning curves: no data found.")

    if not evals.empty:
        evals.to_csv(evals_path, index=False)
        print(f"  ✓  Eval summary     → {evals_path}  ({len(evals)} rows)")
    else:
        print("  ⚠  Eval summary: no data found.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ablation: Uniform Replay vs PER across multiple seeds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--episodes",  type=int,   default=2000,
                   help="Training episodes per run.")
    p.add_argument("--seeds",     type=int,   nargs="+", default=[1, 2, 3],
                   help="Random seeds to use (one run per seed per replay type).")
    p.add_argument("--device",    type=str,   default="auto",
                   help='Torch device: "cpu", "cuda", "mps", or "auto".')
    p.add_argument("--max-moves", type=int,   default=400, dest="max_moves",
                   help="Step limit per game.")
    p.add_argument("--eval-games",type=int,   default=30,  dest="eval_games",
                   help="Games per pair in post-training tournament.")
    p.add_argument("--out",       type=Path,
                   default=Path("outputs/experiments/replay_ablation"),
                   help="Root output directory for all runs.")
    p.add_argument("--dry-run",   action="store_true", dest="dry_run",
                   help="Print experiment plan without running anything.")
    p.add_argument("--benchmark-first", action="store_true", dest="benchmark_first",
                   help="Run a 1-episode benchmark and print a time estimate before "
                        "starting the full ablation.")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()
    out_dir: Path = args.out

    # ── Build experiment grid ─────────────────────────────────────────────────
    grid = [
        (replay_type, seed)
        for replay_type in REPLAY_TYPES
        for seed in args.seeds
    ]
    total = len(grid)

    # ── Dry-run ───────────────────────────────────────────────────────────────
    if args.dry_run:
        print(f"\n{'='*60}")
        print(f"  DRY RUN — {total} experiments would be launched")
        print(f"{'='*60}")
        print(f"  episodes   : {args.episodes}")
        print(f"  max_moves  : {args.max_moves}")
        print(f"  eval_games : {args.eval_games}")
        print(f"  device     : {args.device}")
        print(f"  output     : {out_dir}")
        print()
        for i, (rt, seed) in enumerate(grid, 1):
            run_dir = out_dir / f"{rt}_seed_{seed}"
            print(f"  [{i:2d}/{total}]  {rt:<15}  seed={seed}  →  {run_dir}")
        print()
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Replay Ablation Study — {total} experiments")
    print(f"{'='*60}")
    print(f"  replay types : {REPLAY_TYPES}")
    print(f"  seeds        : {args.seeds}")
    print(f"  episodes     : {args.episodes}")
    print(f"  max_moves    : {args.max_moves}")
    print(f"  eval_games   : {args.eval_games}")
    print(f"  device       : {args.device}")
    print(f"  output       : {out_dir}")

    # ── Benchmark ─────────────────────────────────────────────────────────────
    if args.benchmark_first:
        _run_benchmark(args, out_dir)
        answer = input("\nContinue with full ablation? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            return

    # ── Run all experiments ───────────────────────────────────────────────────
    index_rows: list[dict] = []
    n_ok = 0
    t_total_start = time.perf_counter()

    for step, (replay_type, seed) in enumerate(grid, 1):
        ok = _run_experiment(args, replay_type, seed, out_dir,
                             index_rows, step, total)
        if ok:
            n_ok += 1

    total_runtime = time.perf_counter() - t_total_start

    # ── Aggregate ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Aggregating results…")
    print(f"{'='*60}")
    _aggregate(out_dir, index_rows)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Ablation complete")
    print(f"  {n_ok}/{total} experiments succeeded")
    print(f"  Total runtime: {_fmt_time(total_runtime)}")
    print(f"  Index: {out_dir / 'experiments_index.csv'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
