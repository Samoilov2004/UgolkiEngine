#!/usr/bin/env python3
"""Benchmark: RandomAgent vs GreedyAgent (and the reverse).

Usage
-----
    python scripts/play_random_vs_greedy.py --games 100 --max-moves 500 --seed 42
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Allow running directly without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from corners_rl.agents.greedy_agent import GreedyAgent
from corners_rl.agents.random_agent import RandomAgent
from corners_rl.game_runner import GameResult, play_game


# ── Result accumulator ────────────────────────────────────────────────────────

@dataclass
class MatchStats:
    """Accumulated statistics for a series of games between two agents."""

    agent1_name: str
    agent2_name: str
    wins_agent1: int = 0
    wins_agent2: int = 0
    draws: int = 0
    total_moves: list[int] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.wins_agent1 + self.wins_agent2 + self.draws

    @property
    def win_rate_agent1(self) -> float:
        return self.wins_agent1 / self.total if self.total else 0.0

    @property
    def win_rate_agent2(self) -> float:
        return self.wins_agent2 / self.total if self.total else 0.0

    @property
    def draw_rate(self) -> float:
        return self.draws / self.total if self.total else 0.0

    @property
    def avg_moves(self) -> float:
        return sum(self.total_moves) / len(self.total_moves) if self.total_moves else 0.0

    def record(self, result: GameResult) -> None:
        """Integrate one game result into the statistics."""
        self.total_moves.append(len(result["moves"]))
        if result["winner"] == 1:
            self.wins_agent1 += 1
        elif result["winner"] == -1:
            self.wins_agent2 += 1
        else:
            self.draws += 1

    def print_summary(self) -> None:
        """Print a formatted summary table to stdout."""
        sep = "─" * 52
        print(sep)
        print(f"  {self.agent1_name} (P1)  vs  {self.agent2_name} (P2)")
        print(sep)
        print(f"  Games played : {self.total}")
        print(
            f"  {self.agent1_name:>12} wins : "
            f"{self.wins_agent1:>4}  ({self.win_rate_agent1:.1%})"
        )
        print(
            f"  {self.agent2_name:>12} wins : "
            f"{self.wins_agent2:>4}  ({self.win_rate_agent2:.1%})"
        )
        print(f"  {'Draws':>12}      : {self.draws:>4}  ({self.draw_rate:.1%})")
        print(f"  Avg game length: {self.avg_moves:.1f} half-moves")
        print(sep)


# ── Match runner ──────────────────────────────────────────────────────────────

def run_match(
    agent1_name: str,
    agent2_name: str,
    n_games: int,
    max_moves: int,
    base_seed: int,
) -> MatchStats:
    """Run *n_games* between the named agents and return statistics.

    Args:
        agent1_name: ``"random"`` or ``"greedy"`` for Player 1.
        agent2_name: ``"random"`` or ``"greedy"`` for Player -1.
        n_games: Number of games to play.
        max_moves: Step limit per game.
        base_seed: Seed used to derive per-game seeds reproducibly.

    Returns:
        Populated :class:`MatchStats` instance.
    """
    import random

    seed_rng = random.Random(base_seed)
    stats = MatchStats(agent1_name=agent1_name, agent2_name=agent2_name)

    def make_agent(name: str, game_seed: int):
        if name == "random":
            return RandomAgent(seed=game_seed)
        return GreedyAgent(seed=game_seed)

    for i in range(n_games):
        game_seed = seed_rng.randint(0, 2**32 - 1)
        a1 = make_agent(agent1_name, game_seed)
        a2 = make_agent(agent2_name, game_seed + 1)
        result = play_game(a1, a2, max_moves=max_moves)
        stats.record(result)

    return stats


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark RandomAgent vs GreedyAgent."
    )
    parser.add_argument(
        "--games",
        type=int,
        default=50,
        help="Number of games per match-up (default: 50).",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=500,
        help="Maximum half-moves per game (default: 500).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master random seed for reproducibility (default: 42).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n = args.games
    max_moves = args.max_moves
    seed = args.seed

    print(f"\nRunning {n} games per match-up  (max_moves={max_moves}, seed={seed})\n")

    # Match 1: Random (P1) vs Greedy (P2)
    stats1 = run_match("random", "greedy", n, max_moves, seed)
    stats1.print_summary()

    print()

    # Match 2: Greedy (P1) vs Random (P2)
    stats2 = run_match("greedy", "random", n, max_moves, seed + 1)
    stats2.print_summary()

    # Aggregate greedy performance
    greedy_wins = stats1.wins_agent2 + stats2.wins_agent1
    random_wins = stats1.wins_agent1 + stats2.wins_agent2
    total = 2 * n
    print(
        f"\n  Overall — Greedy: {greedy_wins}/{total} ({greedy_wins/total:.1%})  "
        f"Random: {random_wins}/{total} ({random_wins/total:.1%})\n"
    )


if __name__ == "__main__":
    main()
