#!/usr/bin/env python3
"""CLI entry point: watch or play a game of Corners.

Usage
-----
    # Watch DQN vs random, render to terminal
    python scripts/play.py --agent1 dqn --agent2 random \\
                           --checkpoint checkpoints/dqn_best.pt

    # Save an animated GIF
    python scripts/play.py --agent1 dqn --agent2 greedy \\
                           --render gif --output game.gif
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play / watch a game of Corners.")
    parser.add_argument(
        "--agent1",
        type=str,
        default="dqn",
        choices=["dqn", "random", "greedy", "human"],
        help="Agent type for Player 1 (default: dqn).",
    )
    parser.add_argument(
        "--agent2",
        type=str,
        default="random",
        choices=["dqn", "random", "greedy", "human"],
        help="Agent type for Player -1 (default: random).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="DQNAgent checkpoint (required when agent1 or agent2 is 'dqn').",
    )
    parser.add_argument(
        "--render",
        type=str,
        default="ansi",
        choices=["ansi", "rgb", "gif"],
        help="Render mode (default: ansi).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("game.gif"),
        help="Output path for GIF render (default: game.gif).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=4,
        help="GIF frames per second (default: 4).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.3,
        help="Seconds to pause between moves in ansi mode (default: 0.3).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible games.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    # TODO: build agent1 and agent2 from args (load DQNAgent if needed)
    # TODO: create Game instance, reset
    # TODO: game loop:
    #       - render board (ansi → print; gif → collect frames)
    #       - select action for current player
    #       - game.step(move)
    #       - if done: show result, break
    # TODO: if render == "gif": save gif via BoardRenderer.save_gif()
    raise NotImplementedError("Play script not yet implemented.")


if __name__ == "__main__":
    main()
