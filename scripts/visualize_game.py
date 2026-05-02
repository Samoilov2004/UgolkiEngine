#!/usr/bin/env python3
"""CLI entry point: play and visualise a game between two agents.

Usage
-----
    python scripts/visualize_game.py --agent1 greedy --agent2 random
    python scripts/visualize_game.py --agent1 dqn --agent2 heuristic \\
        --checkpoint outputs/models/dqn_latest.pt --out outputs/games/game.gif
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

# Allow running without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from corners_rl.agents.greedy_agent import GreedyAgent
from corners_rl.agents.heuristic_agent import HeuristicAgent
from corners_rl.agents.random_agent import RandomAgent
from corners_rl.visualization.animate_game import record_game, save_game_gif
from corners_rl.visualization.board_plot import plot_board

_AGENT_CHOICES = ("random", "greedy", "heuristic", "dqn")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualise a game between two agents and save a GIF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--agent1",
        choices=_AGENT_CHOICES,
        default="greedy",
        help="Agent for Player 1.",
    )
    parser.add_argument(
        "--agent2",
        choices=_AGENT_CHOICES,
        default="random",
        help="Agent for Player −1.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="DQN checkpoint path (required when agent1 or agent2 is 'dqn').",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=300,
        dest="max_moves",
        help="Move limit per game.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Frames per second in the output GIF.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for DQNAgent.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/games/game.gif"),
        help="Output path for the GIF file.",
    )
    return parser.parse_args()


def _build_agent(kind: str, name: str, seed: int, checkpoint: Path | None, device: str):
    """Instantiate the requested agent type."""
    if kind == "random":
        return RandomAgent(name=name, seed=seed)
    if kind == "greedy":
        return GreedyAgent(name=name, seed=seed)
    if kind == "heuristic":
        return HeuristicAgent(name=name, seed=seed)
    if kind == "dqn":
        if checkpoint is None or not Path(checkpoint).exists():
            logging.warning(
                "DQN requested but checkpoint not found (%s). "
                "Falling back to GreedyAgent.",
                checkpoint,
            )
            return GreedyAgent(name=f"{name}(fallback-greedy)", seed=seed)
        from corners_rl.agents.dqn_agent import DQNAgent
        agent = DQNAgent.load(checkpoint, device=device, epsilon=0.0)
        agent._name = name  # type: ignore[attr-defined]
        return agent
    raise ValueError(f"Unknown agent kind: {kind!r}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()

    agent1 = _build_agent(args.agent1, args.agent1, args.seed, args.checkpoint, args.device)
    agent2 = _build_agent(args.agent2, args.agent2, args.seed + 1, args.checkpoint, args.device)

    logging.info(
        "Recording game: %s (P1) vs %s (P−1)  max_moves=%d  seed=%d",
        agent1.name, agent2.name, args.max_moves, args.seed,
    )

    frames = record_game(agent1, agent2, max_moves=args.max_moves, seed=args.seed)

    last_frame = frames[-1]
    winner = last_frame["winner"]
    move_count = last_frame["move_number"]
    if winner == 1:
        outcome = f"{agent1.name} wins"
    elif winner == -1:
        outcome = f"{agent2.name} wins"
    else:
        outcome = "Draw"
    logging.info("Game over after %d moves — %s", move_count, outcome)

    # ── Save GIF ──────────────────────────────────────────────────────────────
    out_path = Path(args.out)
    save_game_gif(
        frames,
        out_path,
        fps=args.fps,
        agent1_name=agent1.name,
        agent2_name=agent2.name,
    )
    logging.info("GIF saved to: %s  (%d frames)", out_path, len(frames))

    # ── Save final board PNG ──────────────────────────────────────────────────
    png_path = out_path.with_suffix(".png")
    result_title = (
        f"Final board — {outcome} — move {move_count}\n"
        f"{agent1.name} (blue) vs {agent2.name} (red)"
    )
    fig, _ = plot_board(
        last_frame["board"],
        title=result_title,
        last_move=last_frame["move"],
        target_zones=True,
    )
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    logging.info("Final board PNG saved to: %s", png_path)


if __name__ == "__main__":
    main()
