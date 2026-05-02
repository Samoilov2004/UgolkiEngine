"""Self-play opponent pool manager."""

from __future__ import annotations

import copy
import logging
import random
from collections import deque
from pathlib import Path
from typing import Optional

from corners_rl.agents.dqn_agent import DQNAgent
from corners_rl.agents.base import BaseAgent

logger = logging.getLogger(__name__)


class SelfPlayManager:
    """Maintains a pool of past agent checkpoints for self-play training.

    The current opponent is sampled uniformly from the pool.  A new checkpoint
    is admitted to the pool whenever the training agent achieves a win-rate
    above ``win_rate_threshold`` against the current pool.

    Args:
        agent: The agent being trained (Player 1).
        pool_size: Maximum number of past checkpoints to retain.
        swap_interval: Episodes between opponent-swap attempts.
        win_rate_threshold: Minimum win-rate to promote a checkpoint.
    """

    def __init__(
        self,
        agent: DQNAgent,
        pool_size: int = 10,
        swap_interval: int = 500,
        win_rate_threshold: float = 0.55,
    ) -> None:
        self.agent = agent
        self.pool_size = pool_size
        self.swap_interval = swap_interval
        self.win_rate_threshold = win_rate_threshold

        # Pool is a deque of DQNAgent snapshots acting as Player -1.
        self._pool: deque[DQNAgent] = deque(maxlen=pool_size)
        self.current_opponent: Optional[BaseAgent] = None

        # Bootstrap pool with a copy of the initial agent.
        self._add_snapshot()
        self._swap_opponent()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def maybe_swap(self, episode: int) -> None:
        """Possibly swap the current opponent based on episode count.

        Args:
            episode: The current episode number (1-indexed).
        """
        if episode % self.swap_interval == 0:
            self._add_snapshot()
            self._swap_opponent()

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    def _add_snapshot(self) -> None:
        """Deep-copy the current agent and add it to the pool as Player -1."""
        # TODO: deep copy agent, set player = -1, call eval_mode(), append to pool
        raise NotImplementedError

    def _swap_opponent(self) -> None:
        """Randomly select a new opponent from the pool."""
        if self._pool:
            self.current_opponent = random.choice(list(self._pool))
            logger.debug("Opponent swapped to: %s", self.current_opponent)
