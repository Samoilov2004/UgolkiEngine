#!/usr/bin/env python3
"""CLI entry point: evaluate and compare agents via a round-robin tournament.

Usage
-----
    python scripts/evaluate_agents.py
    python scripts/evaluate_agents.py --checkpoint outputs/models/dqn_latest.pt
    python scripts/evaluate_agents.py --games 50 --device cpu --out outputs/eval/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from corners_rl.agents.greedy_agent import GreedyAgent
from corners_rl.agents.heuristic_agent import HeuristicAgent
from corners_rl.agents.random_agent import RandomAgent
from corners_rl.evaluation.tournaments import round_robin_tournament


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate agents via a round-robin tournament.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to a DQN checkpoint (.pt).  Skipped if not found.",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=100,
        help="Games per agent pair (will be rounded to nearest even number).",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=300,
        dest="max_moves",
        help="Move limit per game.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for DQNAgent (cpu / cuda / mps).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/eval"),
        help="Directory to write CSV outputs.",
    )
    parser.add_argument(
        "--forward-only",
        action="store_true",
        default=False,
        dest="forward_only",
        help="Load the DQN checkpoint with forward-only move restriction.",
    )
    return parser.parse_args()


def _build_agents(args: argparse.Namespace) -> list:
    """Assemble the list of agents to include in the tournament."""
    agents = [
        RandomAgent(name="random", seed=args.seed),
        GreedyAgent(name="greedy", seed=args.seed),
        HeuristicAgent(name="heuristic", seed=args.seed),
    ]

    # Optionally add DQN agent from checkpoint
    if args.checkpoint is not None:
        ckpt_path = Path(args.checkpoint)
        if ckpt_path.exists():
            from corners_rl.agents.dqn_agent import DQNAgent

            dqn = DQNAgent.load(
                ckpt_path,
                device=args.device,
                epsilon=0.0,
                forward_only=args.forward_only,
            )
            agents.append(dqn)
            logging.info("Loaded DQNAgent from %s (epsilon=0.0)", ckpt_path)
        else:
            logging.warning(
                "Checkpoint not found: %s — evaluating baseline agents only.",
                ckpt_path,
            )

    return agents


def _print_summary(summary_df) -> None:
    """Pretty-print the summary table to stdout."""
    col_widths = {
        "agent":     12,
        "games":      6,
        "wins":       5,
        "draws":      5,
        "losses":     7,
        "win_rate":  10,
        "draw_rate": 10,
        "avg_moves": 10,
    }

    header_labels = {
        "agent":     "Agent",
        "games":     "Games",
        "wins":      "Wins",
        "draws":     "Draws",
        "losses":    "Losses",
        "win_rate":  "Win Rate",
        "draw_rate": "Draw Rate",
        "avg_moves": "Avg Moves",
    }

    cols = list(col_widths.keys())
    sep = "  "

    # Header
    header = sep.join(
        header_labels[c].ljust(col_widths[c]) for c in cols
    )
    divider = sep.join("-" * col_widths[c] for c in cols)

    print()
    print("=" * len(divider))
    print("  TOURNAMENT RESULTS")
    print("=" * len(divider))
    print(header)
    print(divider)

    for _, row in summary_df.iterrows():
        line = sep.join(
            [
                str(row["agent"]).ljust(col_widths["agent"]),
                str(int(row["games"])).ljust(col_widths["games"]),
                str(int(row["wins"])).ljust(col_widths["wins"]),
                str(int(row["draws"])).ljust(col_widths["draws"]),
                str(int(row["losses"])).ljust(col_widths["losses"]),
                f"{row['win_rate']:.1%}".ljust(col_widths["win_rate"]),
                f"{row['draw_rate']:.1%}".ljust(col_widths["draw_rate"]),
                f"{row['avg_moves']:.1f}".ljust(col_widths["avg_moves"]),
            ]
        )
        print(line)

    print(divider)
    print()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()
    agents = _build_agents(args)

    logging.info(
        "Starting round-robin tournament: %d agents, %d games/pair, seed=%d",
        len(agents),
        args.games,
        args.seed,
    )

    detailed, summary = round_robin_tournament(
        agents,
        games_per_pair=args.games,
        max_moves=args.max_moves,
        seed=args.seed,
    )

    # ── Save outputs ──────────────────────────────────────────────────────────
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    detailed_path = out_dir / "detailed_results.csv"
    summary_path  = out_dir / "summary.csv"

    detailed.to_csv(detailed_path, index=False)
    summary.to_csv(summary_path, index=False)

    logging.info("Detailed results saved to: %s", detailed_path)
    logging.info("Summary saved to:          %s", summary_path)

    _print_summary(summary)


if __name__ == "__main__":
    main()
