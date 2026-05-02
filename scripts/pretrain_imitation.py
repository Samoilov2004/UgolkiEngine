#!/usr/bin/env python3
"""Imitation-learning pre-training: behavioural cloning from HeuristicAgent.

Usage
-----
    python scripts/pretrain_imitation.py
    python scripts/pretrain_imitation.py \\
        --games 500 --epochs 10 --batch-size 128 \\
        --opponent random --device cpu --seed 42 \\
        --out outputs/models/imitation.pt

The resulting checkpoint can be passed to train_dqn.py via --init-checkpoint
to give the DQN a warm start before self-play fine-tuning.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from corners_rl.agents.heuristic_agent import HeuristicAgent
from corners_rl.rl.imitation import ImitationConfig, generate_imitation_dataset, train_imitation
from corners_rl.rl.model import DQNModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pretrain a DQN model via behavioural cloning from HeuristicAgent.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--games",
        type=int,
        default=500,
        help="Number of expert games to generate training data from.",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs over the imitation dataset.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=128,
        dest="batch_size",
        help="Mini-batch size for training.",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        dest="learning_rate",
        help="Adam learning rate.",
    )
    p.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        dest="val_fraction",
        help="Fraction of data held out for validation.",
    )
    p.add_argument(
        "--grad-clip",
        type=float,
        default=5.0,
        dest="grad_clip",
        help="L2 gradient clipping norm.",
    )
    p.add_argument(
        "--opponent",
        type=str,
        default="random",
        choices=["random", "greedy", "self"],
        help=(
            "Opponent the expert plays against during data collection. "
            '"self" = expert plays both sides.'
        ),
    )
    p.add_argument(
        "--max-moves",
        type=int,
        default=300,
        dest="max_moves",
        help="Step limit per game during data collection.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help='Torch device: "cpu", "cuda", "mps", or "auto".',
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed.",
    )
    p.add_argument(
        "--out",
        type=str,
        default="outputs/models/imitation.pt",
        help="Output path for the pretrained checkpoint (.pt).",
    )
    p.add_argument(
        "--log",
        type=str,
        default="outputs/logs/imitation_log.csv",
        help="Path where the per-epoch CSV log is written.",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)

    args = parse_args()

    log.info(
        "Generating imitation dataset: %d games vs %s opponent, seed=%d",
        args.games, args.opponent, args.seed,
    )
    expert = HeuristicAgent(seed=args.seed)
    dataset = generate_imitation_dataset(
        expert_agent=expert,
        games=args.games,
        max_moves=args.max_moves,
        seed=args.seed,
        opponent=args.opponent,
    )
    log.info("Dataset size: %d samples", len(dataset))

    config = ImitationConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_fraction=args.val_fraction,
        grad_clip=args.grad_clip,
        device=args.device,
        seed=args.seed,
        log_path=args.log,
        out_path=args.out,
    )

    model = DQNModel()
    log.info("Starting imitation training …")
    train_imitation(model, dataset, config)

    log.info("Pre-training complete.  Checkpoint: %s", args.out)
    log.info(
        "To use as a warm start: python scripts/train_dqn.py "
        "--init-checkpoint %s",
        args.out,
    )


if __name__ == "__main__":
    main()
