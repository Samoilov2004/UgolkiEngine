"""Heuristic agent — stronger baseline with a multi-factor move score."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Optional, Union

from corners_rl.agents.base import BaseAgent
from corners_rl.agents.greedy_agent import total_distance
from corners_rl.env.moves import Move, apply_move
from corners_rl.env.rules import get_start_zone, get_target_zone

if TYPE_CHECKING:
    from corners_rl.env.corners_env import CornersEnv


# ── Scoring weights (easy to tune) ────────────────────────────────────────────

W_DISTANCE    = 2.0   # reward per unit of total-distance improvement
W_JUMP_CHAIN  = 0.5   # reward per extra hop in a jump chain
W_ENTERS_GOAL = 3.0   # bonus: piece enters the target zone
W_LEAVES_GOAL = -5.0  # penalty: piece leaves the target zone
W_FREES_START = 0.2   # bonus: piece leaves own start zone


class HeuristicAgent(BaseAgent):
    """Multi-factor heuristic agent.

    Score for a move ``m`` played by *player*::

        score = W_DISTANCE   * distance_improvement
              + W_JUMP_CHAIN  * jump_chain_length
              + W_ENTERS_GOAL * enters_target_zone
              + W_LEAVES_GOAL * leaves_target_zone
              + W_FREES_START * frees_own_start_zone

    Where:

    * ``distance_improvement`` — decrease in total Manhattan distance to goal
      (same as the Greedy metric).
    * ``jump_chain_length`` — ``len(move) - 2`` for a chain jump, ``0`` for a
      simple step.  Favours longer jump chains since they cover more ground
      without costing extra turns.
    * ``enters_target_zone`` — ``1`` if the destination is in the target zone
      and the origin was not; else ``0``.
    * ``leaves_target_zone`` — ``1`` if the origin was in the target zone and
      the destination is not; else ``0``.  Heavy penalty because evicting a
      piece from the goal is almost never desirable.
    * ``frees_own_start_zone`` — ``1`` if the origin is in the player's own
      starting zone and the destination is not; else ``0``.  Encourages
      clearing the starting corner, which tends to open jump routes.

    Ties are broken uniformly at random.

    Args:
        name: Agent label (default ``"heuristic"``).
        seed: Integer seed or :class:`random.Random` for tie-breaking.
    """

    def __init__(
        self,
        name: str = "heuristic",
        seed: Optional[Union[int, random.Random]] = None,
    ) -> None:
        super().__init__(name)
        if isinstance(seed, random.Random):
            self._rng = seed
        else:
            self._rng = random.Random(seed)

    def select_move(self, env: "CornersEnv") -> Move:
        """Return the highest-scoring legal move.

        Args:
            env: Current game environment.

        Returns:
            Best-scoring move (random tie-break among equally scored moves).
        """
        player = env.current_player
        board = env.board
        moves = env.legal_moves()

        target_zone = get_target_zone(player)
        start_zone  = get_start_zone(player)
        before_dist = total_distance(board, player, target_zone)

        best_score: float = -(10 ** 9)
        best_moves: list[Move] = []

        for move in moves:
            score = self._score_move(
                move, board, player, target_zone, start_zone, before_dist
            )
            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)

        return self._rng.choice(best_moves)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _score_move(
        self,
        move: Move,
        board,
        player: int,
        target_zone: set[tuple[int, int]],
        start_zone: set[tuple[int, int]],
        before_dist: int,
    ) -> float:
        """Compute the heuristic score for *move*.

        Args:
            move: Candidate move.
            board: Board state before the move.
            player: The player making the move.
            target_zone: Pre-computed target zone for *player*.
            start_zone: Pre-computed start zone for *player*.
            before_dist: Total distance score before the move.

        Returns:
            Scalar heuristic score (higher = better).
        """
        origin = move[0]
        dest   = move[-1]

        # ── distance improvement ──────────────────────────────────────────
        new_board = apply_move(board, move, player)
        after_dist = total_distance(new_board, player, target_zone)
        dist_improvement = before_dist - after_dist

        # ── jump chain length (extra hops beyond the first) ───────────────
        # A simple step has len=2; a single jump has len=2 as well (start+land).
        # A chain of k jumps has len = k+1 (start + k landing cells).
        # bonus = number of jumps - 1 = len(move) - 2  (0 for step or 1-jump)
        jump_chain_length = max(len(move) - 2, 0)

        # ── zone transitions ──────────────────────────────────────────────
        origin_in_target = origin in target_zone
        dest_in_target   = dest   in target_zone

        enters_target_zone = int(dest_in_target and not origin_in_target)
        leaves_target_zone = int(origin_in_target and not dest_in_target)

        # ── frees own start zone ──────────────────────────────────────────
        frees_own_start_zone = int(origin in start_zone and dest not in start_zone)

        return (
            W_DISTANCE   * dist_improvement
            + W_JUMP_CHAIN  * jump_chain_length
            + W_ENTERS_GOAL * enters_target_zone
            + W_LEAVES_GOAL * leaves_target_zone
            + W_FREES_START * frees_own_start_zone
        )
