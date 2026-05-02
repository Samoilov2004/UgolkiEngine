"""Greedy baseline agent — maximises one-step distance improvement."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Optional, Union

from corners_rl.agents.base import BaseAgent
from corners_rl.env.moves import apply_move
from corners_rl.env.rules import get_piece_positions, get_target_zone

if TYPE_CHECKING:
    from corners_rl.env.corners_env import CornersEnv
    from corners_rl.env.moves import Move


# ── Distance helpers ──────────────────────────────────────────────────────────

def _min_dist_to_zone(
    pos: tuple[int, int],
    zone: set[tuple[int, int]],
) -> int:
    """Return the minimum Manhattan distance from *pos* to any cell in *zone*."""
    r, c = pos
    return min(abs(r - tr) + abs(c - tc) for tr, tc in zone)


def total_distance(
    board,  # NDArray[np.int8]
    player: int,
    target_zone: Optional[set[tuple[int, int]]] = None,
) -> int:
    """Sum of minimum Manhattan distances of all *player* pieces to their target zone.

    Lower is better (0 means all pieces are in the target zone).

    Args:
        board: Current board state.
        player: ``PLAYER1`` or ``PLAYER2``.
        target_zone: Pre-computed target zone set (computed if not given).

    Returns:
        Non-negative integer score.
    """
    zone = target_zone if target_zone is not None else get_target_zone(player)
    return sum(
        _min_dist_to_zone(pos, zone)
        for pos in get_piece_positions(board, player)
    )


# ── Agent ─────────────────────────────────────────────────────────────────────

class GreedyAgent(BaseAgent):
    """One-step greedy agent: picks the move that most reduces total distance to goal.

    **Scoring:**  ``improvement = total_distance_before − total_distance_after``

    The move with the highest improvement is chosen; ties are broken uniformly
    at random.

    Args:
        name: Agent label (default ``"greedy"``).
        seed: Integer seed or :class:`random.Random` for tie-breaking.
    """

    def __init__(
        self,
        name: str = "greedy",
        seed: Optional[Union[int, random.Random]] = None,
    ) -> None:
        super().__init__(name)
        if isinstance(seed, random.Random):
            self._rng = seed
        else:
            self._rng = random.Random(seed)

    def select_move(self, env: "CornersEnv") -> "Move":
        """Return the move that maximises single-step distance improvement.

        Args:
            env: Current game environment.

        Returns:
            Best legal move (random tie-break among equally good moves).
        """
        player = env.current_player
        board = env.board
        moves = env.legal_moves()

        target_zone = get_target_zone(player)
        before = total_distance(board, player, target_zone)

        best_improvement: int = -(10 ** 9)
        best_moves: list[Move] = []

        for move in moves:
            new_board = apply_move(board, move, player)
            after = total_distance(new_board, player, target_zone)
            improvement = before - after

            if improvement > best_improvement:
                best_improvement = improvement
                best_moves = [move]
            elif improvement == best_improvement:
                best_moves.append(move)

        return self._rng.choice(best_moves)
