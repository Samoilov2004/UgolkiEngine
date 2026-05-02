"""Reward shaping for the DQN self-play training loop."""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from corners_rl.agents.greedy_agent import total_distance
from corners_rl.env.moves import Move
from corners_rl.env.rules import PLAYER1, get_target_zone

# ── Reward weights ────────────────────────────────────────────────────────────

W_DISTANCE     =  0.2    # per unit of total-distance improvement
W_ENTER_ZONE   =  2.0    # bonus when a piece enters the target zone
W_LEAVE_ZONE   = -2.0    # penalty when a piece leaves the target zone
W_STEP         = -0.01   # per-step penalty (encourages speed)
W_WIN          = 100.0   # terminal win bonus
W_LOSE         = -100.0  # terminal loss penalty


def state_distance_score(board: NDArray[np.int8], player: int) -> float:
    """Sum of minimum Manhattan distances from each *player* piece to the target zone.

    Lower values are better (0 means all pieces are inside the target zone).

    Args:
        board: Current board state, shape ``(8, 8)``, dtype int8.
        player: ``PLAYER1`` or ``PLAYER2``.

    Returns:
        Non-negative float distance score.
    """
    return float(total_distance(board, player))


def compute_shaped_reward(
    board_before: NDArray[np.int8],
    board_after: NDArray[np.int8],
    player: int,
    move: Move,
    done: bool,
    winner: Optional[int],
) -> float:
    """Compute a shaped scalar reward for *player* after applying *move*.

    Components:

    * ``+0.2 × (dist_before − dist_after)`` — reward proportional to distance
      improvement towards the target zone.
    * ``+2.0`` if the moved piece entered the target zone.
    * ``−2.0`` if the moved piece left the target zone.
    * ``−0.01`` per step (encourages faster wins).
    * ``+100.0`` / ``−100.0`` for win / loss at a terminal state.

    All coordinates are in *real-board* (un-transformed) space.

    Args:
        board_before: Board state before the move.
        board_after: Board state after the move.
        player: The player who made the move (``1`` or ``-1``).
        move: The move that was applied (real-board coordinates).
        done: Whether the episode ended after this move.
        winner: The winning player (``1``, ``-1``, or ``None`` for a draw).

    Returns:
        Scalar shaped reward from *player*'s perspective.
    """
    reward = 0.0
    target_zone = get_target_zone(player)

    # ── distance improvement ──────────────────────────────────────────────────
    dist_before = state_distance_score(board_before, player)
    dist_after  = state_distance_score(board_after,  player)
    reward += W_DISTANCE * (dist_before - dist_after)

    # ── zone entry / exit ─────────────────────────────────────────────────────
    start, end = move[0], move[-1]
    if end in target_zone and start not in target_zone:
        reward += W_ENTER_ZONE
    if start in target_zone and end not in target_zone:
        reward += W_LEAVE_ZONE

    # ── step penalty ──────────────────────────────────────────────────────────
    reward += W_STEP

    # ── terminal ──────────────────────────────────────────────────────────────
    if done:
        if winner == player:
            reward += W_WIN
        elif winner is not None and winner == -player:
            reward += W_LOSE

    return float(reward)
