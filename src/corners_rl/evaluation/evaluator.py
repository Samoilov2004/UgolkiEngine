"""Evaluation: head-to-head matches and summary statistics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from corners_rl.agents.base import BaseAgent
from corners_rl.env.game import Game, GameResult


@dataclass
class EvalMetrics:
    """Aggregated results for one evaluation round.

    Attributes:
        opponent_name: Name of the opponent agent.
        wins: Number of wins for the evaluated agent.
        losses: Number of losses.
        draws: Number of draws.
        total: Total games played.
        win_rate: Fraction of games won.
        avg_steps: Average game length (half-moves).
    """

    opponent_name: str
    wins: int = 0
    losses: int = 0
    draws: int = 0
    total: int = 0
    avg_steps: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.total if self.total > 0 else 0.0

    def __repr__(self) -> str:
        return (
            f"EvalMetrics(vs={self.opponent_name!r}, "
            f"W/L/D={self.wins}/{self.losses}/{self.draws}, "
            f"win_rate={self.win_rate:.2%}, avg_steps={self.avg_steps:.1f})"
        )


class Evaluator:
    """Runs tournament-style evaluation between agents.

    Args:
        board_size: Board side length.
        zone_size: Corner zone size.
        episodes_per_opponent: Number of games to play against each opponent.
        opponent_names: List of opponent names (``"random"``, ``"greedy"``).
        max_steps: Episode step limit.
    """

    def __init__(
        self,
        board_size: int = 8,
        zone_size: int = 3,
        episodes_per_opponent: int = 100,
        opponent_names: Optional[list[str]] = None,
        max_steps: int = 500,
    ) -> None:
        self.board_size = board_size
        self.zone_size = zone_size
        self.episodes_per_opponent = episodes_per_opponent
        self.opponent_names: list[str] = opponent_names or ["random", "greedy"]
        self.max_steps = max_steps

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def evaluate(self, agent: BaseAgent) -> list[EvalMetrics]:
        """Evaluate ``agent`` against all configured opponents.

        The agent plays as Player 1.  Each opponent configuration is run for
        ``episodes_per_opponent`` games and an :class:`EvalMetrics` is returned
        for each.

        Args:
            agent: The agent to evaluate (will be put in eval mode if supported).

        Returns:
            List of :class:`EvalMetrics`, one per opponent.
        """
        results: list[EvalMetrics] = []
        for name in self.opponent_names:
            opponent = self._build_opponent(name)
            metrics = self._run_matches(agent, opponent, name)
            results.append(metrics)
        return results

    # ------------------------------------------------------------------ #
    #  Private                                                             #
    # ------------------------------------------------------------------ #

    def _run_matches(
        self, agent: BaseAgent, opponent: BaseAgent, opponent_name: str
    ) -> EvalMetrics:
        """Play ``episodes_per_opponent`` games and aggregate results.

        Args:
            agent: Player 1 agent.
            opponent: Player -1 agent.
            opponent_name: Label for the metrics output.

        Returns:
            Aggregated :class:`EvalMetrics`.
        """
        # TODO: loop over episodes, run games via Game, tally W/L/D and steps.
        raise NotImplementedError

    def _build_opponent(self, name: str) -> BaseAgent:
        """Instantiate a baseline opponent by name.

        Supported names: ``"random"``, ``"greedy"``.
        """
        # TODO: build and return the appropriate agent with player=-1
        raise NotImplementedError
