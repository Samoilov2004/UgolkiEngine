"""Abstract base class for all Corners agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from corners_rl.env.corners_env import CornersEnv
    from corners_rl.env.moves import Move


class BaseAgent(ABC):
    """Interface that every agent must implement.

    An agent is a stateless (or episode-stateful) policy: given the current
    environment it returns a legal move.

    The environment is passed directly so the agent can call
    :meth:`~corners_rl.env.corners_env.CornersEnv.legal_moves` and read
    :attr:`~corners_rl.env.corners_env.CornersEnv.current_player` without the
    caller having to forward those values separately.

    Args:
        name: Human-readable label used in logs and results tables.
    """

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        """Human-readable agent name."""
        return self._name

    @abstractmethod
    def select_move(self, env: "CornersEnv") -> "Move":
        """Choose a move for the current player.

        Args:
            env: The live environment.  The agent must **not** call
                 :meth:`~corners_rl.env.corners_env.CornersEnv.step` on it;
                 it should only read state.

        Returns:
            A move that appears in ``env.legal_moves()``.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self._name!r})"
