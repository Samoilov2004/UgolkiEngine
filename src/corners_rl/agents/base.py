"""Abstract base class for all agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from corners_rl.env.board import Board, Move


class BaseAgent(ABC):
    """Common interface for all agents (baselines and learned).

    All agents must implement :meth:`select_action`.  Training-capable agents
    should also implement :meth:`observe` for receiving transitions.

    Args:
        player: The player this agent controls (``1`` or ``-1``).
        name: Human-readable agent name (used in logs and tournament tables).
    """

    def __init__(self, player: int, name: str = "agent") -> None:
        self.player = player
        self.name = name

    @abstractmethod
    def select_action(self, board: Board, legal_moves: list[Move]) -> Move:
        """Choose a move given the current board and list of legal moves.

        Args:
            board: Current board state (read-only; do not modify).
            legal_moves: Non-empty list of legal moves for this agent.

        Returns:
            The chosen :class:`~corners_rl.env.board.Move`.
        """
        ...

    def observe(
        self,
        board: Board,
        move: Move,
        reward: float,
        next_board: Board,
        done: bool,
    ) -> None:
        """Receive a transition for learning (no-op for non-learning agents).

        Args:
            board: Board state before the move.
            move: The move that was applied.
            reward: Reward received.
            next_board: Board state after the move.
            done: Whether the episode ended.
        """

    def reset(self) -> None:
        """Reset any episode-level internal state (called at episode start)."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(player={self.player}, name={self.name!r})"
