"""Random baseline agent."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Optional, Union

from corners_rl.agents.base import BaseAgent

if TYPE_CHECKING:
    from corners_rl.env.corners_env import CornersEnv
    from corners_rl.env.moves import Move


class RandomAgent(BaseAgent):
    """Selects a uniformly random legal move each turn.

    Args:
        name: Agent label (default ``"random"``).
        seed: Integer seed **or** an existing :class:`random.Random` instance.
              Pass ``None`` (default) for a non-deterministic agent.
    """

    def __init__(
        self,
        name: str = "random",
        seed: Optional[Union[int, random.Random]] = None,
    ) -> None:
        super().__init__(name)
        if isinstance(seed, random.Random):
            self._rng = seed
        else:
            self._rng = random.Random(seed)

    def select_move(self, env: "CornersEnv") -> "Move":
        """Return a uniformly random legal move.

        Args:
            env: Current game environment.

        Returns:
            A randomly chosen move from ``env.legal_moves()``.

        Raises:
            ValueError: If there are no legal moves (degenerate state).
        """
        moves = env.legal_moves()
        if not moves:
            raise ValueError(
                f"No legal moves for player {env.current_player}. "
                "Board may be in a degenerate state."
            )
        return self._rng.choice(moves)
