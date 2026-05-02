"""Greedy baseline agent — picks the move that maximises immediate progress."""

from __future__ import annotations

from corners_rl.agents.base import BaseAgent
from corners_rl.env.board import Board, Move


class GreedyAgent(BaseAgent):
    """Greedy heuristic: choose the move whose destination is closest to the
    goal zone centre (Manhattan distance), with ties broken by piece furthest
    from goal first.

    This baseline is cheap to run and provides a meaningful sanity check for
    the learned agent.

    Args:
        player: The player this agent controls.
    """

    def __init__(self, player: int) -> None:
        super().__init__(player, name="greedy")

    def select_action(self, board: Board, legal_moves: list[Move]) -> Move:
        """Return the greedily best legal move.

        Scoring heuristic (lower = better):
        ``-pieces_already_in_goal_after_move + distance_of_moved_piece_to_goal_centre``

        Args:
            board: Current board state (read-only).
            legal_moves: Non-empty list of legal moves.

        Returns:
            The move with the best (lowest) heuristic score.
        """
        # TODO: for each move in legal_moves, simulate it on a board copy,
        #       compute _score(), return the move with the best score.
        raise NotImplementedError

    def _score(self, board: Board, move: Move) -> float:
        """Compute a heuristic score for ``move`` applied to ``board``.

        Lower score → better move.
        """
        # TODO: compute Manhattan distance from move.end to goal centre,
        #       subtract number of pieces already in goal zone after the move.
        raise NotImplementedError
