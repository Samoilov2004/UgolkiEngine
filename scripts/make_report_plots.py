#!/usr/bin/env python3
"""Generate all report figures from training and evaluation logs.

Usage
-----
    python scripts/make_report_plots.py
    python scripts/make_report_plots.py \\
        --train-log   outputs/logs/train_log.csv \\
        --eval-summary outputs/eval/summary.csv \\
        --eval-detailed outputs/eval/detailed_results.csv \\
        --out outputs/figures

If a required input file is missing the corresponding plots are skipped with a
warning rather than crashing.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from corners_rl.visualization.plots import (
    generate_position_heatmap,
    plot_evaluation_summary,
    plot_training_curves,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate report-quality plots from training/eval logs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--train-log",
        type=Path,
        default=Path("outputs/logs/train_log.csv"),
        dest="train_log",
        help="Path to train_log.csv produced by SelfPlayTrainer.",
    )
    p.add_argument(
        "--eval-summary",
        type=Path,
        default=Path("outputs/eval/summary.csv"),
        dest="eval_summary",
        help="Path to tournament summary.csv.",
    )
    p.add_argument(
        "--eval-detailed",
        type=Path,
        default=Path("outputs/eval/detailed_results.csv"),
        dest="eval_detailed",
        help="Path to per-game detailed_results.csv.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/figures"),
        help="Directory where PNG figures are saved.",
    )
    p.add_argument(
        "--heatmap-games",
        type=int,
        default=30,
        dest="heatmap_games",
        help="Number of games to play for the position heatmap.",
    )
    p.add_argument(
        "--heatmap-agents",
        type=str,
        nargs=2,
        default=["greedy", "random"],
        dest="heatmap_agents",
        metavar=("AGENT1", "AGENT2"),
        help='Agent pair for the heatmap (random / greedy / heuristic).',
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    p.add_argument(
        "--roll-short",
        type=int,
        default=10,
        dest="roll_short",
    )
    p.add_argument(
        "--roll-long",
        type=int,
        default=30,
        dest="roll_long",
    )
    return p.parse_args()


def _build_agent(kind: str, seed: int):
    """Return a baseline agent by name string."""
    from corners_rl.agents.greedy_agent import GreedyAgent
    from corners_rl.agents.heuristic_agent import HeuristicAgent
    from corners_rl.agents.random_agent import RandomAgent

    kind = kind.lower()
    if kind == "random":
        return RandomAgent(seed=seed)
    if kind == "greedy":
        return GreedyAgent(seed=seed)
    if kind == "heuristic":
        return HeuristicAgent(seed=seed)
    raise ValueError(f"Unknown agent kind: {kind!r}. Choose random/greedy/heuristic.")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)
    args = parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", out)

    produced: list[str] = []
    skipped:  list[str] = []

    # ── Training curves ───────────────────────────────────────────────────────
    if args.train_log.exists():
        log.info("Plotting training curves from %s …", args.train_log)
        try:
            plot_training_curves(
                args.train_log,
                out,
                roll_short=args.roll_short,
                roll_long=args.roll_long,
            )
            names = [
                "training_epsilon.png",
                "training_moves.png",
                "training_loss.png",
                "training_winners.png",
                "training_rewards.png",
            ]
            produced.extend(names)
        except Exception as exc:
            log.warning("Training-curve plots failed: %s", exc, exc_info=True)
            skipped.append("training_*.png")
    else:
        log.warning("train-log not found (%s) — skipping training plots.", args.train_log)
        skipped.append("training_*.png")

    # ── Evaluation summary ────────────────────────────────────────────────────
    if args.eval_summary.exists():
        log.info("Plotting evaluation summary from %s …", args.eval_summary)
        try:
            detailed = args.eval_detailed if args.eval_detailed.exists() else None
            if args.eval_detailed and not args.eval_detailed.exists():
                log.warning(
                    "eval-detailed not found (%s) — pairwise heatmap skipped.",
                    args.eval_detailed,
                )
            plot_evaluation_summary(args.eval_summary, out, detailed_path=detailed)
            produced.extend(["eval_win_rates.png", "eval_avg_moves.png"])
            if detailed:
                produced.append("eval_pairwise.png")
        except Exception as exc:
            log.warning("Evaluation plots failed: %s", exc, exc_info=True)
            skipped.append("eval_*.png")
    else:
        log.warning(
            "eval-summary not found (%s) — skipping evaluation plots.", args.eval_summary
        )
        skipped.append("eval_*.png")

    # ── Position heatmap ──────────────────────────────────────────────────────
    log.info(
        "Generating position heatmap (%d games, %s vs %s) …",
        args.heatmap_games,
        args.heatmap_agents[0],
        args.heatmap_agents[1],
    )
    try:
        a1 = _build_agent(args.heatmap_agents[0], args.seed)
        a2 = _build_agent(args.heatmap_agents[1], args.seed + 1)
        generate_position_heatmap(
            a1, a2,
            n_games=args.heatmap_games,
            max_moves=500,
            seed=args.seed,
            out_dir=out,
        )
        produced.append("heatmap_positions.png")
    except Exception as exc:
        log.warning("Position heatmap failed: %s", exc, exc_info=True)
        skipped.append("heatmap_positions.png")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 50)
    print("  Report figures summary")
    print("=" * 50)
    if produced:
        print(f"  Saved ({len(produced)} files):")
        for name in produced:
            print(f"    {out / name}")
    if skipped:
        print(f"  Skipped ({len(skipped)}): {', '.join(skipped)}")
    print("=" * 50)


if __name__ == "__main__":
    main()
