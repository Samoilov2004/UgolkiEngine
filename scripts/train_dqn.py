#!/usr/bin/env python3
"""CLI entry point: train a DQN agent on the Corners game via self-play.

Usage
-----
    python scripts/train_dqn.py --config configs/dqn.yaml
    python scripts/train_dqn.py --config configs/dqn.yaml --episodes 500 --device cpu
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

# Allow running without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from corners_rl.rl.train_dqn import SelfPlayTrainer, TrainConfig, config_from_dict, ReplayConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a DQN agent on Corners via self-play.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dqn.yaml"),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override the number of training episodes from the config.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Override device (e.g. "cpu", "cuda", "mps", "auto").',
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override global random seed.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        dest="output_dir",
        help="Override output directory for logs and checkpoints.",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default=None,
        dest="init_checkpoint",
        help=(
            "Path to an imitation-learning checkpoint (.pt) for warm-start. "
            "Produced by scripts/pretrain_imitation.py."
        ),
    )
    parser.add_argument(
        "--replay-type",
        type=str,
        default=None,
        dest="replay_type",
        choices=["uniform", "prioritized"],
        help=(
            "Override the replay-buffer strategy from the config. "
            '"uniform" = standard random sampling; '
            '"prioritized" = PER (Schaul et al., 2016).'
        ),
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()

    # Load base config from YAML
    with open(args.config) as f:
        raw = yaml.safe_load(f)

    config: TrainConfig = config_from_dict(raw)

    # Apply CLI overrides
    if args.episodes is not None:
        config.episodes = args.episodes
    if args.device is not None:
        config.device = args.device
    if args.seed is not None:
        config.seed = args.seed
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.init_checkpoint is not None:
        config.init_checkpoint = args.init_checkpoint
    if args.replay_type is not None:
        config.replay = ReplayConfig(
            type=args.replay_type,
            # Preserve all PER parameters from the loaded config; only override type
            alpha=config.replay.alpha,
            beta_start=config.replay.beta_start,
            beta_end=config.replay.beta_end,
            beta_anneal_steps=config.replay.beta_anneal_steps,
            priority_epsilon=config.replay.priority_epsilon,
        )

    logging.info("Training config: %s", config)

    trainer = SelfPlayTrainer(config)
    trainer.train()

    logging.info("Training complete.  Outputs in: %s", config.output_dir)


if __name__ == "__main__":
    main()
