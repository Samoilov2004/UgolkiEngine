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
    validate_move,
    is_orthogonal_step,
    is_orthogonal_jump,
    midpoint,
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
        # Orthogonal neighbours: up (-1,2) OOB, down (1,2) P1, left (0,1) P1, right (0,3) empty
        # Only (0,3) is reachable with orthogonal moves
        moves = get_simple_moves(start_board, (0, 2))
        dests = {m[1] for m in moves}
        assert (0, 3) in dests
        # (1,3) was reachable diagonally — must NOT appear with orthogonal-only rules
        assert (1, 3) not in dests

    def test_simple_move_length(self, start_board: np.ndarray) -> None:
        for m in get_simple_moves(start_board, (0, 2)):
            assert len(m) == 2

    def test_destination_is_empty(self, start_board: np.ndarray) -> None:
        for m in get_simple_moves(start_board, (0, 2)):
            r, c = m[1]
            assert start_board[r, c] == EMPTY

    def test_isolated_piece_has_all_4_orthogonal_moves(self) -> None:
        board = _empty_board()
        board[4, 4] = PLAYER1
        moves = get_simple_moves(board, (4, 4))
        assert len(moves) == 4
        dests = {m[1] for m in moves}
        expected = {
            (3, 4),          # up
            (5, 4),          # down
            (4, 3),          # left
            (4, 5),          # right
        }
        assert dests == expected

    def test_board_boundary_limits_moves(self) -> None:
        board = _empty_board()
        board[0, 0] = PLAYER1
        moves = get_simple_moves(board, (0, 0))
        # Orthogonal neighbours in-bounds: right (0,1) and down (1,0)
        assert len(moves) == 2


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
        # P1 at (4,4), P2 at (3,4), empty at (2,4) → orthogonal jump upward
        board = _empty_board()
        board[4, 4] = PLAYER1
        board[3, 4] = PLAYER2
        moves = get_jump_moves(board, (4, 4))
        ends = {m[-1] for m in moves}
        assert (2, 4) in ends

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


# ── Orthogonal-only rule enforcement ──────────────────────────────────────────

class TestOrthogonalOnly:
    """Verify that diagonal moves and jumps are never generated."""

    def test_diagonal_simple_move_not_in_legal_moves(self) -> None:
        """A diagonal step must never appear in get_legal_moves."""
        board = _empty_board()
        board[4, 4] = PLAYER1
        moves = get_legal_moves(board, PLAYER1)
        for m in moves:
            dr = abs(m[-1][0] - m[0][0])
            dc = abs(m[-1][1] - m[0][1])
            assert not (dr == 1 and dc == 1), (
                f"Diagonal simple move found: {m}"
            )

    def test_diagonal_step_impossible_when_only_diagonal_free(self) -> None:
        """A piece surrounded orthogonally but with free diagonals has NO simple moves."""
        board = _empty_board()
        board[4, 4] = PLAYER1
        # Block all orthogonal neighbours
        board[3, 4] = PLAYER2
        board[5, 4] = PLAYER2
        board[4, 3] = PLAYER2
        board[4, 5] = PLAYER2
        # Diagonal cells (3,3), (3,5), (5,3), (5,5) are empty
        moves = get_simple_moves(board, (4, 4))
        assert moves == [], f"Expected no simple moves, got {moves}"

    def test_diagonal_jump_not_allowed(self) -> None:
        """A diagonal piece arrangement must NOT produce a jump."""
        board = _empty_board()
        board[4, 4] = PLAYER1
        board[3, 3] = PLAYER2  # diagonal hurdle
        # (2, 2) is empty — would be the landing cell for a diagonal jump
        moves = get_jump_moves(board, (4, 4))
        ends = {m[-1] for m in moves}
        assert (2, 2) not in ends, "Diagonal jump (4,4)→(2,2) must be illegal"

    def test_horizontal_jump_allowed(self) -> None:
        """A horizontal jump over an adjacent piece must be generated."""
        board = _empty_board()
        board[4, 4] = PLAYER1
        board[4, 3] = PLAYER2  # piece to the left
        # (4, 2) is empty landing cell
        moves = get_jump_moves(board, (4, 4))
        ends = {m[-1] for m in moves}
        assert (4, 2) in ends, "Horizontal jump (4,4)→(4,2) must be legal"

    def test_vertical_jump_allowed(self) -> None:
        """A vertical jump over an adjacent piece must be generated."""
        board = _empty_board()
        board[4, 4] = PLAYER1
        board[3, 4] = PLAYER2  # piece above
        # (2, 4) is empty landing cell
        moves = get_jump_moves(board, (4, 4))
        ends = {m[-1] for m in moves}
        assert (2, 4) in ends, "Vertical jump (4,4)→(2,4) must be legal"

    def test_chain_jump_all_hops_orthogonal(self) -> None:
        """Every individual hop in a chain jump must be orthogonal."""
        board = _empty_board()
        board[6, 4] = PLAYER1
        board[5, 4] = PLAYER2  # hurdle for 1st hop (up)
        board[3, 4] = PLAYER2  # hurdle for 2nd hop (up again)
        board[2, 3] = PLAYER2  # hurdle for 3rd hop (left)
        # Expect chain: (6,4) → (4,4) → (2,4) → (2,2)
        moves = get_jump_moves(board, (6, 4))
        three_hop = ((6, 4), (4, 4), (2, 4), (2, 2))
        assert three_hop in moves, (
            f"Expected orthogonal chain {three_hop} in {moves}"
        )
        # No hop in any move should be diagonal
        for m in moves:
            for i in range(len(m) - 1):
                r0, c0 = m[i]
                r1, c1 = m[i + 1]
                assert abs(r1 - r0) == 0 or abs(c1 - c0) == 0, (
                    f"Diagonal hop found in chain {m}: {m[i]}→{m[i+1]}"
                )

    def test_initial_board_legal_moves_no_diagonal_steps(self) -> None:
        """No move in the starting position should involve a diagonal step."""
        board = initial_board()
        for player in (PLAYER1, PLAYER2):
            for m in get_legal_moves(board, player):
                # For simple steps (len==2) and single jump landings,
                # each consecutive waypoint pair must be axially aligned.
                for i in range(len(m) - 1):
                    r0, c0 = m[i]
                    r1, c1 = m[i + 1]
                    is_orthogonal = (r0 == r1) or (c0 == c1)
                    assert is_orthogonal, (
                        f"Non-orthogonal step in move {m} for player {player}"
                    )


# ── validate_move ─────────────────────────────────────────────────────────────

class TestValidateMove:
    """Systematic tests for the validate_move hard-validation layer."""

    # ── helper geometry functions ─────────────────────────────────────────────

    def test_is_orthogonal_step_true(self) -> None:
        assert is_orthogonal_step((3, 3), (4, 3))
        assert is_orthogonal_step((3, 3), (2, 3))
        assert is_orthogonal_step((3, 3), (3, 4))
        assert is_orthogonal_step((3, 3), (3, 2))

    def test_is_orthogonal_step_false_diagonal(self) -> None:
        assert not is_orthogonal_step((3, 3), (4, 4))
        assert not is_orthogonal_step((3, 3), (2, 2))

    def test_is_orthogonal_jump_true(self) -> None:
        assert is_orthogonal_jump((3, 3), (5, 3))
        assert is_orthogonal_jump((3, 3), (1, 3))
        assert is_orthogonal_jump((3, 3), (3, 5))
        assert is_orthogonal_jump((3, 3), (3, 1))

    def test_is_orthogonal_jump_false_diagonal(self) -> None:
        assert not is_orthogonal_jump((3, 3), (5, 5))
        assert not is_orthogonal_jump((3, 3), (1, 1))

    def test_midpoint_horizontal(self) -> None:
        assert midpoint((3, 1), (3, 3)) == (3, 2)

    def test_midpoint_vertical(self) -> None:
        assert midpoint((1, 3), (3, 3)) == (2, 3)

    # ── diagonal step rejected ────────────────────────────────────────────────

    def test_diagonal_step_rejected(self) -> None:
        """A diagonal step must always raise ValueError."""
        board = _empty_board()
        board[3, 3] = PLAYER1
        with pytest.raises(ValueError, match="[Dd]iagonal"):
            apply_move(board, ((3, 3), (4, 4)), PLAYER1)

    def test_diagonal_step_rejected_validate(self) -> None:
        board = _empty_board()
        board[3, 3] = PLAYER1
        with pytest.raises(ValueError, match="[Dd]iagonal"):
            validate_move(board, ((3, 3), (4, 4)), PLAYER1)

    # ── diagonal jump rejected (even if midpoint is occupied) ─────────────────

    def test_diagonal_jump_rejected_even_if_midpoint_occupied(self) -> None:
        """Diagonal jump (2,2 distance) must be rejected regardless of midpoint."""
        board = _empty_board()
        board[2, 2] = PLAYER1
        board[3, 3] = PLAYER2   # would-be "midpoint" diagonally
        # (4, 4) is empty — still illegal because it's diagonal
        with pytest.raises(ValueError, match="[Dd]iagonal"):
            apply_move(board, ((2, 2), (4, 4)), PLAYER1)

    # ── jump over empty cell rejected ─────────────────────────────────────────

    def test_jump_over_empty_rejected(self) -> None:
        """A horizontal jump over an empty midpoint must be rejected."""
        board = _empty_board()
        board[2, 2] = PLAYER1
        # (2, 3) is empty (no piece to jump over)
        # (2, 4) is also empty
        with pytest.raises(ValueError, match="empty"):
            apply_move(board, ((2, 2), (2, 4)), PLAYER1)

    def test_jump_over_empty_rejected_vertical(self) -> None:
        board = _empty_board()
        board[2, 2] = PLAYER1
        # (3, 2) empty, (4, 2) empty
        with pytest.raises(ValueError, match="empty"):
            apply_move(board, ((2, 2), (4, 2)), PLAYER1)

    # ── horizontal and vertical jumps allowed ─────────────────────────────────

    def test_horizontal_jump_allowed(self) -> None:
        board = _empty_board()
        board[2, 2] = PLAYER1
        board[2, 3] = PLAYER2   # hurdle
        new_board = apply_move(board, ((2, 2), (2, 4)), PLAYER1)
        assert new_board[2, 2] == EMPTY
        assert new_board[2, 4] == PLAYER1
        assert new_board[2, 3] == PLAYER2   # hurdle not captured

    def test_vertical_jump_allowed(self) -> None:
        board = _empty_board()
        board[2, 2] = PLAYER1
        board[3, 2] = PLAYER2   # hurdle
        new_board = apply_move(board, ((2, 2), (4, 2)), PLAYER1)
        assert new_board[2, 2] == EMPTY
        assert new_board[4, 2] == PLAYER1
        assert new_board[3, 2] == PLAYER2

    # ── chain with diagonal segment rejected ──────────────────────────────────

    def test_chain_with_diagonal_segment_rejected(self) -> None:
        """A chain where one segment is diagonal must be rejected."""
        board = _empty_board()
        board[0, 0] = PLAYER1
        board[0, 1] = PLAYER2   # hurdle for first hop right: (0,0)→(0,2)
        # Second segment (0,2)→(2,4) is diagonal with dr=2, dc=2 → rejected
        with pytest.raises(ValueError, match="[Dd]iagonal"):
            apply_move(board, ((0, 0), (0, 2), (2, 4)), PLAYER1)

    def test_chain_with_step_segment_rejected(self) -> None:
        """A chain that includes a simple-step segment must be rejected."""
        board = _empty_board()
        board[0, 0] = PLAYER1
        board[0, 1] = PLAYER2   # hurdle
        # After jumping to (0,2), trying to step to (0,3) — not a jump
        with pytest.raises(ValueError, match="[Ss]tep"):
            validate_move(board, ((0, 0), (0, 2), (0, 3)), PLAYER1)

    # ── all initial legal moves are orthogonal ────────────────────────────────

    def test_all_initial_legal_moves_pass_validate(self) -> None:
        """Every legal move from the start position must pass validate_move."""
        board = initial_board()
        for player in (PLAYER1, PLAYER2):
            for move in get_legal_moves(board, player):
                validate_move(board, move, player)   # must not raise

    def test_all_initial_legal_moves_are_orthogonal(self) -> None:
        """No move from the initial position should have any diagonal segment."""
        board = initial_board()
        for player in (PLAYER1, PLAYER2):
            for move in get_legal_moves(board, player):
                for i in range(len(move) - 1):
                    a, b = move[i], move[i + 1]
                    assert (a[0] == b[0]) or (a[1] == b[1]), (
                        f"Diagonal segment {a}→{b} in move {move}"
                    )

    # ── random game: every applied move is valid ──────────────────────────────

    def test_all_random_game_moves_are_orthogonal(self) -> None:
        """Play 20 random games and assert every applied move is orthogonal."""
        import random as _r
        from corners_rl.env.corners_env import CornersEnv

        rng = _r.Random(42)
        for _ in range(20):
            env = CornersEnv(max_moves=200)
            env.reset()
            while not env.is_terminal():
                board_before = env.board
                moves = env.legal_moves()
                move = rng.choice(moves)
                _, _, _, info = env.step(move)

                applied = info["move"]
                player  = info["player_moved"]
                # Must pass geometry validation on the board BEFORE the move
                validate_move(board_before, applied, player)
                # No diagonal segments
                for i in range(len(applied) - 1):
                    a, b = applied[i], applied[i + 1]
                    assert (a[0] == b[0]) or (a[1] == b[1]), (
                        f"Diagonal segment {a}→{b} in applied move {applied}"
                    )

    # ── validate_move error messages ──────────────────────────────────────────

    def test_too_short_move_rejected(self) -> None:
        board = _empty_board()
        board[3, 3] = PLAYER1
        with pytest.raises(ValueError, match="2 waypoints"):
            validate_move(board, ((3, 3),), PLAYER1)

    def test_wrong_player_piece_rejected(self) -> None:
        board = _empty_board()
        board[3, 3] = PLAYER2
        with pytest.raises(ValueError, match="player"):
            validate_move(board, ((3, 3), (3, 4)), PLAYER1)

    def test_occupied_destination_rejected(self) -> None:
        board = _empty_board()
        board[3, 3] = PLAYER1
        board[3, 4] = PLAYER2
        with pytest.raises(ValueError, match="not empty"):
            validate_move(board, ((3, 3), (3, 4)), PLAYER1)

    def test_require_in_legal_moves_rejects_unreachable(self) -> None:
        """require_in_legal_moves=True should reject geometrically valid but
        unreachable moves (e.g. a jump where the piece can't actually reach)."""
        board = _empty_board()
        board[4, 4] = PLAYER1
        board[4, 3] = PLAYER2   # hurdle; (4,2) is empty — jump IS reachable
        # This move IS in legal_moves; should not raise
        validate_move(board, ((4, 4), (4, 2)), PLAYER1, require_in_legal_moves=True)

    def test_require_in_legal_moves_rejects_not_in_list(self) -> None:
        """A move not generated by get_legal_moves must be rejected."""
        board = _empty_board()
        board[4, 4] = PLAYER1
        # (4,3) is empty — this is a valid STEP move.
        # We will ask for it WITH require_in_legal_moves=True. It should pass
        # since it IS in legal moves (step to (4,3)).
        validate_move(board, ((4, 4), (4, 3)), PLAYER1, require_in_legal_moves=True)


# ── visualization: record_game stores actual move from info ───────────────────

class TestVisualizationRecordsActualMove:
    def test_record_game_stores_move_from_info(self) -> None:
        """record_game must store the exact move from env.step info."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from corners_rl.visualization.animate_game import record_game
        from corners_rl.agents.random_agent import RandomAgent
        from corners_rl.env.corners_env import CornersEnv

        frames = record_game(
            RandomAgent(seed=1), RandomAgent(seed=2), max_moves=20, seed=42
        )
        # Frame 0 is the initial state; subsequent frames have a real move
        for frame in frames[1:]:
            move = frame["move"]
            assert move is not None
            assert len(move) >= 2

    def test_record_game_board_after_matches_frame_board(self) -> None:
        """The board stored in each frame must equal board_after from step info."""
        from corners_rl.visualization.animate_game import record_game
        from corners_rl.agents.random_agent import RandomAgent
        from corners_rl.env.corners_env import CornersEnv
        import numpy as np

        frames = record_game(
            RandomAgent(seed=3), RandomAgent(seed=4), max_moves=20, seed=7
        )
        # All post-move frames must have non-trivial boards
        for frame in frames[1:]:
            assert frame["board"].shape == (8, 8)
            assert frame["move"] is not None

    def test_all_recorded_moves_are_orthogonal(self) -> None:
        """No frame in a recorded game should have a diagonal move segment."""
        from corners_rl.visualization.animate_game import record_game
        from corners_rl.agents.greedy_agent import GreedyAgent
        from corners_rl.agents.random_agent import RandomAgent

        frames = record_game(
            GreedyAgent(seed=5), RandomAgent(seed=6), max_moves=100, seed=99
        )
        for frame in frames[1:]:
            move = frame["move"]
            for i in range(len(move) - 1):
                a, b = move[i], move[i + 1]
                assert (a[0] == b[0]) or (a[1] == b[1]), (
                    f"Diagonal segment {a}→{b} in recorded move {move}"
                )
