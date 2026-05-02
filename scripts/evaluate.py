#!/usr/bin/env python3
"""CLI entry point: evaluate a trained DQN agent against baselines.

Usage
-----
    python scripts/evaluate.py --checkpoint checkpoints/dqn_best.pt \\
                                --episodes 200 [--device cpu]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DQN agent for Corners.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a saved DQNAgent checkpoint.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=200,
        help="Number of evaluation games per opponent.",
    )
    parser.add_argument(
        "--opponents",
        nargs="+",
        default=["random", "greedy"],
        help='Opponents to evaluate against (default: "random" "greedy").',
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help='Torch device (default: "cpu").',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # TODO: load DQNAgent from args.checkpoint
    # TODO: build Evaluator with args.opponents and args.episodes
    # TODO: call evaluator.evaluate(agent)
    # TODO: pretty-print results table
    raise NotImplementedError("Evaluation not yet implemented.")


if __name__ == "__main__":
    main()
