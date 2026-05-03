#!/usr/bin/env python3
"""Train and evaluate a forward-only DQN agent.

Mirrors the structure of run_replay_ablation.py but trains a single variant:
DQN with forward-only action masking (pieces may only move toward their target
zone).  Saves results in the same CSV format so they can be merged with the
existing Uniform/PER baseline for comparison.

Usage
-----
    # Full experiment (2 seeds, 1500 episodes — matches baseline)
    python scripts/run_forward_experiment.py \\
        --episodes 1500 --seeds 1 2 --device auto \\
        --out outputs/experiments/forward

    # Quick smoke test
    python scripts/run_forward_experiment.py \\
        --episodes 5 --seeds 1 --device cpu --eval-games 2 \\
        --out outputs/experiments/test_forward

Output structure
----------------
    <out>/
    ├── experiments_index.csv
    ├── aggregated_learning_curves.csv
    ├── final_eval_summary.csv
    └── forward_seed_1/
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

# ── Config ────────────────────────────────────────────────────────────────────

def _train_config(args: argparse.Namespace, seed: int, run_dir: Path) -> TrainConfig:
    n = args.episodes
    batch_size       = min(64, max(8, n))
    train_start_size = batch_size
    imitation_ckpt   = Path("outputs/models/imitation.pt")
    init_ckpt        = str(imitation_ckpt) if imitation_ckpt.exists() else None
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
        init_checkpoint=init_ckpt,
        replay=ReplayConfig(type="uniform"),
        forward_only=True,
    )


# ── DQN loader ────────────────────────────────────────────────────────────────

def _load_dqn(checkpoint: Path, device: str, seed: int):
    import torch
    from corners_rl.agents.dqn_agent import DQNAgent
    from corners_rl.rl.model import DQNModel
    from corners_rl.utils.seeding import resolve_device

    resolved = str(resolve_device(device))
    model = DQNModel()
    ckpt = torch.load(checkpoint, map_location=resolved, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    return DQNAgent(model=model, device=resolved, epsilon=0.0, seed=seed,
                    forward_only=True)


# ── Evaluation ────────────────────────────────────────────────────────────────

def _run_eval(args: argparse.Namespace, run_dir: Path, checkpoint: str,
              seed: int) -> Path:
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
            log.info("  DQN (forward) loaded from %s", ckpt_path)
        except Exception as exc:
            log.warning("  Could not load DQN (%s) — eval without DQN.", exc)
    else:
        log.warning("  Checkpoint not found (%s).", ckpt_path)

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
    log.info("  Eval summary → %s", summary_path)
    return summary_path


# ── Index helpers ─────────────────────────────────────────────────────────────

_INDEX_FIELDS = ["seed", "output_dir", "final_checkpoint", "status",
                 "runtime_seconds"]


def _write_index(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


# ── Single-seed runner ────────────────────────────────────────────────────────

def _run_one(args: argparse.Namespace, seed: int, out_dir: Path,
             index_rows: list[dict], step: int, total: int) -> bool:
    run_dir = out_dir / f"forward_seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─'*60}")
    print(f"  [{step}/{total}]  FORWARD-ONLY  seed={seed}")
    print(f"  Output: {run_dir}")
    print(f"{'─'*60}")

    t0 = time.perf_counter()
    status, final_ckpt = "pending", ""

    try:
        cfg = _train_config(args, seed, run_dir)
        log.info("Config: %s", cfg)
        SelfPlayTrainer(cfg).train()
        final_ckpt = str(run_dir / "models" / "dqn_latest.pt")
        print(f"  ✓  Training done  ({final_ckpt})")

        _run_eval(args, run_dir, final_ckpt, seed)
        print("  ✓  Evaluation done")
        status = "completed"

    except Exception as exc:
        status = f"failed: {type(exc).__name__}: {exc}"
        print(f"  ✗  {status}")
        traceback.print_exc()

    runtime = round(time.perf_counter() - t0, 1)
    print(f"  ⏱  {runtime}s")
    index_rows.append({"seed": seed, "output_dir": str(run_dir),
                        "final_checkpoint": final_ckpt, "status": status,
                        "runtime_seconds": runtime})
    _write_index(out_dir / "experiments_index.csv", index_rows)
    return status == "completed"


# ── Aggregation ───────────────────────────────────────────────────────────────

def _aggregate(out_dir: Path, seeds: list[int]) -> None:
    completed_dirs = [
        out_dir / f"forward_seed_{s}"
        for s in seeds
        if (out_dir / f"forward_seed_{s}" / "models" / "dqn_latest.pt").exists()
    ]

    curve_runs = [
        {"replay_type": "forward", "seed": int(d.name.split("_")[-1]),
         "log_path": d / "logs" / "train_log.csv"}
        for d in completed_dirs
    ]
    eval_runs = [
        {"replay_type": "forward", "seed": int(d.name.split("_")[-1]),
         "eval_path": d / "eval" / "summary.csv"}
        for d in completed_dirs
    ]

    curves = aggregate_learning_curves(curve_runs)
    evals  = aggregate_eval_summaries(eval_runs)

    if not curves.empty:
        p = out_dir / "aggregated_learning_curves.csv"
        curves.to_csv(p, index=False)
        print(f"  ✓  Learning curves → {p}  ({len(curves)} rows)")

    if not evals.empty:
        p = out_dir / "final_eval_summary.csv"
        evals.to_csv(p, index=False)
        print(f"  ✓  Eval summary    → {p}  ({len(evals)} rows)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train & evaluate forward-only DQN agent.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--episodes",   type=int,   default=1500)
    p.add_argument("--seeds",      type=int,   nargs="+", default=[1, 2])
    p.add_argument("--device",     type=str,   default="auto")
    p.add_argument("--max-moves",  type=int,   default=300, dest="max_moves")
    p.add_argument("--eval-games", type=int,   default=100, dest="eval_games")
    p.add_argument("--out",        type=Path,
                   default=Path("outputs/experiments/forward"))
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
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(args.seeds)
    print(f"\n{'='*60}")
    print(f"  Forward-Only DQN Experiment — {total} seed(s)")
    print(f"{'='*60}")
    print(f"  episodes   : {args.episodes}")
    print(f"  seeds      : {args.seeds}")
    print(f"  max_moves  : {args.max_moves}")
    print(f"  eval_games : {args.eval_games}")
    print(f"  device     : {args.device}")
    print(f"  output     : {out_dir}")

    index_rows: list[dict] = []
    n_ok, t0 = 0, time.perf_counter()

    for step, seed in enumerate(args.seeds, 1):
        if _run_one(args, seed, out_dir, index_rows, step, total):
            n_ok += 1

    print(f"\n{'='*60}")
    print("  Aggregating…")
    _aggregate(out_dir, args.seeds)

    print(f"\n  Done: {n_ok}/{total} seeds OK")
    print(f"  Total runtime: {(time.perf_counter()-t0)/60:.1f} min")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
