"""Uniform experience replay buffer."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class Transition:
    """A single (s, a_idx, r, s', done) experience tuple.

    Attributes:
        state: Board observation before the move, shape ``(3, N, N)``.
        action_idx: Index of the chosen move in the legal-moves list at ``state``.
        reward: Scalar reward.
        next_state: Board observation after the move, shape ``(3, N, N)``.
        done: Whether the episode ended after this transition.
    """

    state: NDArray[np.float32]
    action_idx: int
    reward: float
    next_state: NDArray[np.float32]
    done: bool


class ReplayBuffer:
    """Fixed-capacity circular replay buffer with uniform random sampling.

    Args:
        capacity: Maximum number of transitions to store.
        seed: Optional random seed for reproducible sampling.
    """

    def __init__(self, capacity: int, seed: Optional[int] = None) -> None:
        self.capacity = capacity
        self._buffer: deque[Transition] = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def push(self, transition: Transition) -> None:
        """Add a transition to the buffer (overwrites oldest if full)."""
        self._buffer.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        """Sample a random mini-batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            List of :class:`Transition` objects.

        Raises:
            ValueError: If ``batch_size > len(self)``.
        """
        if batch_size > len(self):
            raise ValueError(
                f"Cannot sample {batch_size} from buffer of size {len(self)}."
            )
        # TODO: return self._rng.sample(list(self._buffer), batch_size)
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:
        return f"ReplayBuffer(size={len(self)}/{self.capacity})"
