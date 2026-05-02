"""Utility for running a complete game between two agents."""

from __future__ import annotations

from typing import Optional, TypedDict

import numpy as np
from numpy.typing import NDArray

from corners_rl.agents.base import BaseAgent
from corners_rl.env.corners_env import CornersEnv
from corners_rl.env.moves import Move


class GameResult(TypedDict):
    """Return type of :func:`play_game`."""

    winner: Optional[int]   # PLAYER1, PLAYER2, or None (draw)
    moves: list[Move]       # ordered list of half-moves played
    draw: bool              # True if the game ended by step limit
    final_board: NDArray[np.int8]


def play_game(
    agent1: BaseAgent,
    agent2: BaseAgent,
    max_moves: int = 500,
    seed: Optional[int] = None,
) -> GameResult:
    """Play one complete game between *agent1* (Player 1) and *agent2* (Player -1).

    The game loop calls :meth:`~BaseAgent.select_move` on the active agent each
    turn, then advances the environment via
    :meth:`~CornersEnv.step`.

    Args:
        agent1: Agent that controls Player 1 (moves first).
        agent2: Agent that controls Player -1.
        max_moves: Episode step limit passed to :class:`CornersEnv`.
        seed: When provided, seeds the internal RNG of each agent (if the
              agent exposes a ``_rng`` attribute that is a
              :class:`random.Random` instance) to make the game reproducible.

    Returns:
        A :class:`GameResult` dict with keys:

        * ``winner``      — ``1``, ``-1``, or ``None`` for a draw.
        * ``moves``       — list of all half-moves in order.
        * ``draw``        — ``True`` iff the game ended by the step limit.
        * ``final_board`` — board state at game end.
    """
    import random as _random

    env = CornersEnv(max_moves=max_moves)
    env.reset()

    # Seed agents' RNGs when requested
    if seed is not None:
        master_rng = _random.Random(seed)
        for agent in (agent1, agent2):
            if hasattr(agent, "_rng") and isinstance(agent._rng, _random.Random):
                agent._rng = _random.Random(master_rng.randint(0, 2**32 - 1))

    move_map: dict[int, BaseAgent] = {1: agent1, -1: agent2}
    moves_played: list[Move] = []

    while not env.is_terminal():
        player = env.current_player
        move = move_map[player].select_move(env)
        moves_played.append(move)
        env.step(move)

    return GameResult(
        winner=env.winner,
        moves=moves_played,
        draw=env.winner is None,
        final_board=env.board,
    )
