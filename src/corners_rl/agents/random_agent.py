"""Random baseline agent — selects a uniformly random legal move."""

from __future__ import annotations

import random
from typing import Optional

from corners_rl.agents.base import BaseAgent
from corners_rl.env.board import Board, Move


class RandomAgent(BaseAgent):
    """Baseline that picks uniformly at random from the legal-move list.

    Args:
        player: The player this agent controls.
        seed: Optional random seed for reproducibility.
    """

    def __init__(self, player: int, seed: Optional[int] = None) -> None:
        super().__init__(player, name="random")
        self._rng = random.Random(seed)

    def select_action(self, board: Board, legal_moves: list[Move]) -> Move:
        """Return a uniformly random legal move."""
        # TODO: return self._rng.choice(legal_moves)
        raise NotImplementedError
