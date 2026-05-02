"""Tests for move generation and application (rules.py + moves.py)."""

import numpy as np
import pytest

from corners_rl.env.rules import (
    BOARD_SIZE,
    EMPTY,
    PLAYER1,
    PLAYER2,
    check_winner,
    get_piece_positions,
    get_target_zone,
    initial_board,
    inside_board,
)
from corners_rl.env.moves import (
    Move,
    apply_move,
    get_jump_moves,
    get_legal_moves,
    get_simple_moves,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def start_board() -> np.ndarray:
    """Standard initial board."""
    return initial_board()


def _empty_board() -> np.ndarray:
    """Completely empty board — useful for crafting custom scenarios."""
    return np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)


# ── inside_board ─────────────────────────────────────────────────────────────

class TestInsideBoard:
    def test_corners_are_inside(self) -> None:
        assert inside_board((0, 0))
        assert inside_board((7, 7))
        assert inside_board((0, 7))
        assert inside_board((7, 0))

    def test_centre_is_inside(self) -> None:
        assert inside_board((3, 4))

    def test_negative_row_is_outside(self) -> None:
        assert not inside_board((-1, 0))

    def test_negative_col_is_outside(self) -> None:
        assert not inside_board((0, -1))

    def test_row_too_large(self) -> None:
        assert not inside_board((8, 0))

    def test_col_too_large(self) -> None:
        assert not inside_board((0, 8))


# ── initial_board ─────────────────────────────────────────────────────────────

class TestInitialBoard:
    def test_shape(self, start_board: np.ndarray) -> None:
        assert start_board.shape == (BOARD_SIZE, BOARD_SIZE)

    def test_dtype(self, start_board: np.ndarray) -> None:
        assert start_board.dtype == np.int8

    def test_player1_piece_count(self, start_board: np.ndarray) -> None:
        assert np.sum(start_board == PLAYER1) == 9

    def test_player2_piece_count(self, start_board: np.ndarray) -> None:
        assert np.sum(start_board == PLAYER2) == 9

    def test_player1_in_top_left(self, start_board: np.ndarray) -> None:
        assert np.all(start_board[:3, :3] == PLAYER1)

    def test_player2_in_bottom_right(self, start_board: np.ndarray) -> None:
        assert np.all(start_board[5:, 5:] == PLAYER2)

    def test_rest_is_empty(self, start_board: np.ndarray) -> None:
        # cells outside the two 3×3 zones must be empty
        total_pieces = np.sum(start_board != EMPTY)
        assert total_pieces == 18

    def test_zones_do_not_overlap(self, start_board: np.ndarray) -> None:
        # No cell can hold both players' pieces simultaneously
        assert not np.any((start_board == PLAYER1) & (start_board == PLAYER2))


# ── get_simple_moves ──────────────────────────────────────────────────────────

class TestSimpleMoves:
    def test_corner_piece_has_moves(self, start_board: np.ndarray) -> None:
        # (0, 0) is blocked on 3 sides by own pieces; only (1,0),(0,1),(1,1)
        # are occupied — so NO simple moves from (0,0) on the initial board.
        moves = get_simple_moves(start_board, (0, 0))
        # all 3 neighbours of (0,0) that are inside board are occupied by P1
        # ((-1,-1),(-1,0),(0,-1) are out; (1,0),(0,1),(1,1) all == PLAYER1)
        assert moves == []

    def test_edge_piece_has_moves(self, start_board: np.ndarray) -> None:
        # (0, 2) is the top-right piece of the P1 zone.
        # Neighbours inside board: (0,1), (0,3), (1,1), (1,2), (1,3)
        # (0,3) and (1,3) are empty → 2 simple moves expected
        moves = get_simple_moves(start_board, (0, 2))
        dests = {m[1] for m in moves}
        assert (0, 3) in dests
        assert (1, 3) in dests

    def test_simple_move_length(self, start_board: np.ndarray) -> None:
        for m in get_simple_moves(start_board, (0, 2)):
            assert len(m) == 2

    def test_destination_is_empty(self, start_board: np.ndarray) -> None:
        for m in get_simple_moves(start_board, (0, 2)):
            r, c = m[1]
            assert start_board[r, c] == EMPTY

    def test_isolated_piece_has_all_8_moves(self) -> None:
        board = _empty_board()
        board[4, 4] = PLAYER1
        moves = get_simple_moves(board, (4, 4))
        assert len(moves) == 8
        dests = {m[1] for m in moves}
        expected = {
            (3, 3), (3, 4), (3, 5),
            (4, 3),         (4, 5),
            (5, 3), (5, 4), (5, 5),
        }
        assert dests == expected

    def test_board_boundary_limits_moves(self) -> None:
        board = _empty_board()
        board[0, 0] = PLAYER1
        moves = get_simple_moves(board, (0, 0))
        # Only (0,1), (1,0), (1,1) are in-bounds
        assert len(moves) == 3


# ── get_jump_moves ────────────────────────────────────────────────────────────

class TestJumpMoves:
    def test_single_jump_over_own_piece(self) -> None:
        # P1 at (4,4), own piece at (3,4), empty at (2,4) → can jump up
        board = _empty_board()
        board[4, 4] = PLAYER1
        board[3, 4] = PLAYER1  # hurdle
        moves = get_jump_moves(board, (4, 4))
        ends = {m[-1] for m in moves}
        assert (2, 4) in ends

    def test_single_jump_over_opponent_piece(self) -> None:
        # P1 at (4,4), P2 at (3,3), empty at (2,2) → diagonal jump
        board = _empty_board()
        board[4, 4] = PLAYER1
        board[3, 3] = PLAYER2
        moves = get_jump_moves(board, (4, 4))
        ends = {m[-1] for m in moves}
        assert (2, 2) in ends

    def test_no_jump_over_empty(self) -> None:
        # P1 at (4,4), everything else empty → no jumps
        board = _empty_board()
        board[4, 4] = PLAYER1
        moves = get_jump_moves(board, (4, 4))
        assert moves == []

    def test_no_jump_to_occupied_cell(self) -> None:
        # P1 at (4,4), hurdle at (3,4), blocker at (2,4) → cannot jump
        board = _empty_board()
        board[4, 4] = PLAYER1
        board[3, 4] = PLAYER1
        board[2, 4] = PLAYER2  # landing cell blocked
        moves = get_jump_moves(board, (4, 4))
        ends = {m[-1] for m in moves}
        assert (2, 4) not in ends

    def test_chain_jump_two_hops(self) -> None:
        # P1 at (6,4)
        # hurdle at (5,4) → lands at (4,4)
        # hurdle at (3,4) → lands at (2,4)
        # Expected move: ((6,4), (4,4), (2,4))
        board = _empty_board()
        board[6, 4] = PLAYER1
        board[5, 4] = PLAYER2  # first hurdle
        board[3, 4] = PLAYER2  # second hurdle
        moves = get_jump_moves(board, (6, 4))
        paths = [m for m in moves if len(m) == 3]
        assert any(m == ((6, 4), (4, 4), (2, 4)) for m in paths), (
            f"Expected chain ((6,4),(4,4),(2,4)) in {moves}"
        )

    def test_chain_jump_includes_intermediate_as_valid_move(self) -> None:
        # After the first hop, stopping there must also be a valid move
        board = _empty_board()
        board[6, 4] = PLAYER1
        board[5, 4] = PLAYER2
        board[3, 4] = PLAYER2
        moves = get_jump_moves(board, (6, 4))
        # Single hop: ((6,4), (4,4))
        assert ((6, 4), (4, 4)) in moves

    def test_chain_jump_no_revisit(self) -> None:
        # Layout that would cause an infinite loop if visited cells were not tracked:
        # P1 at (4,4), hurdles at (3,4) and (4,3)
        # Jump up: land at (2,4)
        # Jump left from (2,4): need hurdle at (2,3), land at (2,2) — add that
        # The piece should NOT return to (4,4)
        board = _empty_board()
        board[4, 4] = PLAYER1
        board[3, 4] = PLAYER2  # hurdle for upward jump
        board[2, 3] = PLAYER2  # hurdle for leftward jump from (2,4)
        moves = get_jump_moves(board, (4, 4))
        for m in moves:
            assert m.count((4, 4)) == 1, f"Start cell revisited in path {m}"

    def test_three_hop_chain(self) -> None:
        # Build a straight vertical corridor for 3 hops upward
        # P1 at (6,0), hurdles at (5,0), (3,0), (1,0)
        # lands: (4,0), (2,0), (0,0)
        board = _empty_board()
        board[6, 0] = PLAYER1
        board[5, 0] = PLAYER2
        board[3, 0] = PLAYER2
        board[1, 0] = PLAYER2
        moves = get_jump_moves(board, (6, 0))
        three_hop = ((6, 0), (4, 0), (2, 0), (0, 0))
        assert three_hop in moves, f"Three-hop chain not found in {moves}"

    def test_jump_start_cell_is_first_element(self) -> None:
        board = _empty_board()
        board[4, 4] = PLAYER1
        board[3, 4] = PLAYER2
        moves = get_jump_moves(board, (4, 4))
        for m in moves:
            assert m[0] == (4, 4)


# ── get_legal_moves ───────────────────────────────────────────────────────────

class TestLegalMoves:
    def test_initial_board_has_moves_for_player1(self, start_board: np.ndarray) -> None:
        moves = get_legal_moves(start_board, PLAYER1)
        assert len(moves) > 0

    def test_initial_board_has_moves_for_player2(self, start_board: np.ndarray) -> None:
        moves = get_legal_moves(start_board, PLAYER2)
        assert len(moves) > 0

    def test_all_moves_start_with_own_piece(self, start_board: np.ndarray) -> None:
        for player in (PLAYER1, PLAYER2):
            positions = set(get_piece_positions(start_board, player))
            for m in get_legal_moves(start_board, player):
                assert m[0] in positions, (
                    f"Move {m} does not start on a {player} piece"
                )

    def test_all_move_destinations_are_empty(self, start_board: np.ndarray) -> None:
        for player in (PLAYER1, PLAYER2):
            for m in get_legal_moves(start_board, player):
                r, c = m[-1]
                assert start_board[r, c] == EMPTY, (
                    f"Move {m} ends on non-empty cell ({r},{c})"
                )


# ── apply_move ────────────────────────────────────────────────────────────────

class TestApplyMove:
    def test_piece_moves_from_start_to_end(self, start_board: np.ndarray) -> None:
        move: Move = ((0, 2), (0, 3))  # simple step for P1
        new_board = apply_move(start_board, move, PLAYER1)
        assert new_board[0, 2] == EMPTY
        assert new_board[0, 3] == PLAYER1

    def test_original_board_not_modified(self, start_board: np.ndarray) -> None:
        original_copy = start_board.copy()
        move: Move = ((0, 2), (0, 3))
        apply_move(start_board, move, PLAYER1)
        np.testing.assert_array_equal(start_board, original_copy)

    def test_returns_new_array(self, start_board: np.ndarray) -> None:
        move: Move = ((0, 2), (0, 3))
        new_board = apply_move(start_board, move, PLAYER1)
        assert new_board is not start_board

    def test_jump_move_applied_correctly(self) -> None:
        board = _empty_board()
        board[4, 4] = PLAYER1
        board[3, 4] = PLAYER2
        move: Move = ((4, 4), (2, 4))
        new_board = apply_move(board, move, PLAYER1)
        assert new_board[4, 4] == EMPTY
        assert new_board[2, 4] == PLAYER1
        assert new_board[3, 4] == PLAYER2  # hurdle is not captured

    def test_chain_jump_applied_to_end(self) -> None:
        board = _empty_board()
        board[6, 4] = PLAYER1
        board[5, 4] = PLAYER2
        board[3, 4] = PLAYER2
        move: Move = ((6, 4), (4, 4), (2, 4))
        new_board = apply_move(board, move, PLAYER1)
        assert new_board[6, 4] == EMPTY
        assert new_board[2, 4] == PLAYER1
        # Intermediate landing cell (4,4) is empty (piece passed through)
        assert new_board[4, 4] == EMPTY

    def test_raises_on_wrong_piece(self, start_board: np.ndarray) -> None:
        # Piece at (0,0) belongs to PLAYER1; try applying as PLAYER2
        with pytest.raises(ValueError, match="player"):
            apply_move(start_board, ((0, 0), (1, 0)), PLAYER2)

    def test_raises_on_occupied_destination(self, start_board: np.ndarray) -> None:
        # (0,1) is occupied by PLAYER1 — cannot land there
        with pytest.raises(ValueError):
            apply_move(start_board, ((0, 0), (0, 1)), PLAYER1)

    def test_raises_on_too_short_move(self, start_board: np.ndarray) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            apply_move(start_board, ((0, 0),), PLAYER1)


# ── check_winner ──────────────────────────────────────────────────────────────

class TestCheckWinner:
    def test_no_winner_at_start(self, start_board: np.ndarray) -> None:
        assert check_winner(start_board) is None

    def test_empty_board_has_no_winner(self) -> None:
        assert check_winner(_empty_board()) is None

    def test_player1_wins(self) -> None:
        board = _empty_board()
        # Fill P1's target zone (bottom-right 3×3) with P1 pieces
        for r in range(5, 8):
            for c in range(5, 8):
                board[r, c] = PLAYER1
        assert check_winner(board) == PLAYER1

    def test_player2_wins(self) -> None:
        board = _empty_board()
        # Fill P2's target zone (top-left 3×3) with P2 pieces
        for r in range(3):
            for c in range(3):
                board[r, c] = PLAYER2
        assert check_winner(board) == PLAYER2

    def test_partial_target_zone_is_not_win(self) -> None:
        board = _empty_board()
        # Only 8 out of 9 cells filled for P1
        cells = list(get_target_zone(PLAYER1))
        for r, c in cells[:-1]:
            board[r, c] = PLAYER1
        assert check_winner(board) is None
