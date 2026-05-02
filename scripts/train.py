#!/usr/bin/env python3
"""CLI entry point: train a DQN agent on the Corners game.

Usage
-----
    python scripts/train.py --config configs/dqn.yaml [--device cuda]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN agent for Corners.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dqn.yaml"),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Override device (e.g. "cpu", "cuda", "mps").',
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume training from a checkpoint file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Override device if specified on the command line.
    if args.device:
        cfg["training"]["device"] = args.device

    # TODO: set global random seeds (random, numpy, torch)
    # TODO: resolve device (handle "auto" → detect cuda/mps/cpu)
    # TODO: build DQNAgent from cfg["agent"] (resume from args.resume if given)
    # TODO: build Trainer from cfg sections
    # TODO: call trainer.train()
    # TODO: save final plots via plot_training_curves()
    raise NotImplementedError("Training loop not yet implemented.")


if __name__ == "__main__":
    main()
