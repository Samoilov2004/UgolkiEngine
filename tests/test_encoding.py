"""Tests for src/corners_rl/rl/encoding.py."""

from __future__ import annotations

import numpy as np
import pytest

from corners_rl.env.moves import Move, get_legal_moves
from corners_rl.env.rules import (
    BOARD_SIZE,
    PLAYER1,
    PLAYER2,
    get_start_zone,
    get_target_zone,
    initial_board,
)
from corners_rl.rl.encoding import (
    ACTION_SPACE_SIZE,
    STATE_CHANNELS,
    action_id_to_move,
    decode_action,
    encode_action,
    encode_state,
    inverse_transform_move_for_player,
    inverse_transform_pos_for_player,
    legal_action_mask,
    transform_move_for_player,
    transform_pos_for_player,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _empty_board() -> np.ndarray:
    return np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)


# ── encode_state ──────────────────────────────────────────────────────────────

class TestEncodeState:
    # ---------- shape / dtype ----------

    def test_shape_player1(self) -> None:
        obs = encode_state(initial_board(), PLAYER1)
        assert obs.shape == (STATE_CHANNELS, BOARD_SIZE, BOARD_SIZE)

    def test_shape_player2(self) -> None:
        obs = encode_state(initial_board(), PLAYER2)
        assert obs.shape == (STATE_CHANNELS, BOARD_SIZE, BOARD_SIZE)

    def test_dtype_float32(self) -> None:
        obs = encode_state(initial_board(), PLAYER1)
        assert obs.dtype == np.float32

    def test_values_in_zero_one(self) -> None:
        obs = encode_state(initial_board(), PLAYER1)
        assert obs.min() >= 0.0 and obs.max() <= 1.0

    # ---------- Player 1 (no transform) ----------

    def test_player1_my_pieces_channel0(self) -> None:
        """Channel 0 must show P1's pieces exactly where they are on the board."""
        board = initial_board()
        obs = encode_state(board, PLAYER1)
        expected = (board == PLAYER1).astype(np.float32)
        np.testing.assert_array_equal(obs[0], expected)

    def test_player1_opponent_channel1(self) -> None:
        board = initial_board()
        obs = encode_state(board, PLAYER1)
        expected = (board == PLAYER2).astype(np.float32)
        np.testing.assert_array_equal(obs[1], expected)

    def test_player1_target_zone_channel2_is_bottom_right(self) -> None:
        obs = encode_state(initial_board(), PLAYER1)
        target = get_target_zone(PLAYER1)  # bottom-right 3×3
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                expected = 1.0 if (r, c) in target else 0.0
                assert obs[2, r, c] == expected, (
                    f"channel 2 mismatch at ({r},{c}): "
                    f"got {obs[2, r, c]}, expected {expected}"
                )

    def test_player1_my_pieces_sum(self) -> None:
        obs = encode_state(initial_board(), PLAYER1)
        assert obs[0].sum() == 9.0   # 9 P1 pieces

    def test_player1_opponent_pieces_sum(self) -> None:
        obs = encode_state(initial_board(), PLAYER1)
        assert obs[1].sum() == 9.0   # 9 P2 pieces

    # ---------- Player -1 (rotate 180° + negate) ----------

    def test_player2_my_pieces_channel0(self) -> None:
        """After normalisation P2's pieces must appear in channel 0."""
        board = initial_board()
        obs = encode_state(board, PLAYER2)
        assert obs[0].sum() == 9.0

    def test_player2_my_pieces_top_left(self) -> None:
        """P2's pieces start at bottom-right; after 180° rotation they land
        in the top-left corner of the canonical frame."""
        board = initial_board()
        obs = encode_state(board, PLAYER2)
        # Canonical top-left 3×3 must all be 1 in channel 0
        for r in range(3):
            for c in range(3):
                assert obs[0, r, c] == 1.0, (
                    f"Expected piece at canonical ({r},{c}), got 0."
                )

    def test_player2_no_pieces_outside_start_in_channel0(self) -> None:
        """In the initial board only the starting 3×3 should be lit in ch0."""
        obs = encode_state(initial_board(), PLAYER2)
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                in_start = r < 3 and c < 3
                assert obs[0, r, c] == (1.0 if in_start else 0.0)

    def test_player2_opponent_channel1(self) -> None:
        board = initial_board()
        obs = encode_state(board, PLAYER2)
        assert obs[1].sum() == 9.0   # 9 opponent (P1) pieces in ch1

    def test_player2_target_zone_channel2_bottom_right(self) -> None:
        """After normalisation P2's goal must appear in the bottom-right (canonical)."""
        obs = encode_state(initial_board(), PLAYER2)
        # canonical bottom-right 3×3
        for r in range(5, 8):
            for c in range(5, 8):
                assert obs[2, r, c] == 1.0
        assert obs[2].sum() == 9.0

    def test_player2_channels_disjoint(self) -> None:
        """Channels 0 and 1 must not overlap."""
        obs = encode_state(initial_board(), PLAYER2)
        overlap = (obs[0] == 1) & (obs[1] == 1)
        assert not overlap.any()

    # ---------- does not mutate the original board ----------

    def test_does_not_mutate_board(self) -> None:
        board = initial_board()
        original = board.copy()
        encode_state(board, PLAYER2)
        np.testing.assert_array_equal(board, original)

    # ---------- custom board ----------

    def test_single_piece_player1(self) -> None:
        board = _empty_board()
        board[3, 4] = PLAYER1
        obs = encode_state(board, PLAYER1)
        assert obs[0, 3, 4] == 1.0
        assert obs[0].sum() == 1.0

    def test_single_piece_player2_transforms_correctly(self) -> None:
        """A P2 piece at (3,4) should appear at (4,3) after 180° rotation."""
        board = _empty_board()
        board[3, 4] = PLAYER2
        obs = encode_state(board, PLAYER2)
        canonical_r = BOARD_SIZE - 1 - 3  # 4
        canonical_c = BOARD_SIZE - 1 - 4  # 3
        assert obs[0, canonical_r, canonical_c] == 1.0
        assert obs[0].sum() == 1.0


# ── encode_action / decode_action ─────────────────────────────────────────────

class TestEncodeDecodeAction:
    def test_roundtrip_simple_move(self) -> None:
        move: Move = ((0, 0), (1, 1))
        action_id = encode_action(move)
        from_pos, to_pos = decode_action(action_id)
        assert from_pos == (0, 0)
        assert to_pos == (1, 1)

    def test_roundtrip_chain_jump(self) -> None:
        """Chain jumps should encode using first and last cell only."""
        move: Move = ((0, 0), (2, 2), (4, 4))
        action_id = encode_action(move)
        from_pos, to_pos = decode_action(action_id)
        assert from_pos == (0, 0)
        assert to_pos == (4, 4)

    def test_chain_and_simple_same_endpoints_same_id(self) -> None:
        """Two moves with identical start/end but different paths share one ID."""
        simple: Move = ((0, 0), (4, 4))
        chain: Move  = ((0, 0), (2, 2), (4, 4))
        assert encode_action(simple) == encode_action(chain)

    def test_action_id_range(self) -> None:
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                for dr in range(BOARD_SIZE):
                    for dc in range(BOARD_SIZE):
                        move: Move = ((r, c), (dr, dc))
                        aid = encode_action(move)
                        assert 0 <= aid < ACTION_SPACE_SIZE

    def test_different_moves_different_ids(self) -> None:
        m1: Move = ((0, 0), (1, 1))
        m2: Move = ((0, 0), (1, 2))
        assert encode_action(m1) != encode_action(m2)

    def test_decode_raises_on_negative(self) -> None:
        with pytest.raises(ValueError):
            decode_action(-1)

    def test_decode_raises_on_too_large(self) -> None:
        with pytest.raises(ValueError):
            decode_action(ACTION_SPACE_SIZE)

    def test_all_corners(self) -> None:
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        for f in corners:
            for t in corners:
                move: Move = (f, t)
                fp, tp = decode_action(encode_action(move))
                assert fp == f and tp == t

    def test_decode_is_deterministic(self) -> None:
        aid = encode_action(((3, 5), (6, 2)))
        assert decode_action(aid) == decode_action(aid)


# ── legal_action_mask ─────────────────────────────────────────────────────────

class TestLegalActionMask:
    def test_shape(self) -> None:
        moves = get_legal_moves(initial_board(), PLAYER1)
        mask = legal_action_mask(moves)
        assert mask.shape == (ACTION_SPACE_SIZE,)

    def test_dtype_bool(self) -> None:
        mask = legal_action_mask(get_legal_moves(initial_board(), PLAYER1))
        assert mask.dtype == np.bool_

    def test_legal_moves_are_true(self) -> None:
        board = initial_board()
        moves = get_legal_moves(board, PLAYER1)
        mask = legal_action_mask(moves)
        for m in moves:
            assert mask[encode_action(m)], f"Move {m} not set in mask"

    def test_empty_move_list_all_false(self) -> None:
        mask = legal_action_mask([])
        assert not mask.any()

    def test_number_of_true_le_number_of_moves(self) -> None:
        """Distinct action IDs ≤ number of moves (chain duplicates collapse)."""
        moves = get_legal_moves(initial_board(), PLAYER1)
        mask = legal_action_mask(moves)
        assert mask.sum() <= len(moves)

    def test_illegal_positions_are_false(self) -> None:
        """A self-to-self 'move' must not appear in the mask of real moves."""
        board = initial_board()
        moves = get_legal_moves(board, PLAYER1)
        mask = legal_action_mask(moves)
        # A piece staying in place is never legal
        for r, c in [(0, 0), (1, 1), (2, 2)]:
            stay_id = encode_action(((r, c), (r, c)))
            assert not mask[stay_id]


# ── action_id_to_move ─────────────────────────────────────────────────────────

class TestActionIdToMove:
    def test_returns_legal_move(self) -> None:
        board = initial_board()
        moves = get_legal_moves(board, PLAYER1)
        for move in moves:
            aid = encode_action(move)
            result = action_id_to_move(aid, moves)
            assert result in moves

    def test_returns_longest_chain_on_tie(self) -> None:
        """When short and long paths share (start, end), return the longer one."""
        short: Move = ((0, 0), (4, 4))
        long_:  Move = ((0, 0), (2, 2), (4, 4))
        moves = [short, long_]
        aid = encode_action(short)  # same id for both
        result = action_id_to_move(aid, moves)
        assert result == long_

    def test_raises_on_missing_move(self) -> None:
        board = initial_board()
        moves = get_legal_moves(board, PLAYER1)
        # Encode a completely non-existent move
        fake_id = encode_action(((7, 7), (0, 0)))
        with pytest.raises(ValueError, match="No legal move"):
            action_id_to_move(fake_id, moves)

    def test_result_start_matches_from_pos(self) -> None:
        board = initial_board()
        moves = get_legal_moves(board, PLAYER1)
        move = moves[0]
        aid = encode_action(move)
        result = action_id_to_move(aid, moves)
        assert result[0] == move[0]

    def test_result_end_matches_to_pos(self) -> None:
        board = initial_board()
        moves = get_legal_moves(board, PLAYER1)
        move = moves[0]
        aid = encode_action(move)
        result = action_id_to_move(aid, moves)
        assert result[-1] == move[-1]


# ── transform_pos_for_player ──────────────────────────────────────────────────

class TestTransformPos:
    def test_player1_is_identity(self) -> None:
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                assert transform_pos_for_player((r, c), PLAYER1) == (r, c)

    def test_player2_rotates_180(self) -> None:
        n = BOARD_SIZE - 1
        assert transform_pos_for_player((0, 0), PLAYER2) == (n, n)
        assert transform_pos_for_player((7, 7), PLAYER2) == (0, 0)
        assert transform_pos_for_player((3, 4), PLAYER2) == (n - 3, n - 4)

    def test_transform_and_inverse_are_identity_player1(self) -> None:
        pos = (3, 5)
        assert inverse_transform_pos_for_player(
            transform_pos_for_player(pos, PLAYER1), PLAYER1
        ) == pos

    def test_transform_and_inverse_are_identity_player2(self) -> None:
        pos = (3, 5)
        assert inverse_transform_pos_for_player(
            transform_pos_for_player(pos, PLAYER2), PLAYER2
        ) == pos

    def test_player2_self_inverse(self) -> None:
        """For player -1, transform(transform(p)) == p."""
        pos = (2, 6)
        assert transform_pos_for_player(
            transform_pos_for_player(pos, PLAYER2), PLAYER2
        ) == pos

    def test_start_zone_maps_to_opposite_corner_player2(self) -> None:
        """P2's start zone (bottom-right) → top-left under transform."""
        start = get_start_zone(PLAYER2)  # bottom-right cells
        for r, c in start:
            tr, tc = transform_pos_for_player((r, c), PLAYER2)
            assert tr < 3 and tc < 3, (
                f"({r},{c}) did not map to top-left, got ({tr},{tc})"
            )


# ── transform_move_for_player ─────────────────────────────────────────────────

class TestTransformMove:
    def test_player1_is_identity(self) -> None:
        move: Move = ((0, 0), (2, 2), (4, 4))
        assert transform_move_for_player(move, PLAYER1) == move

    def test_player2_transforms_all_cells(self) -> None:
        move: Move = ((7, 7), (5, 5))
        transformed = transform_move_for_player(move, PLAYER2)
        assert transformed[0] == (0, 0)
        assert transformed[1] == (2, 2)

    def test_player2_chain_jump_all_cells_transformed(self) -> None:
        move: Move = ((6, 6), (4, 4), (2, 2))
        n = BOARD_SIZE - 1
        transformed = transform_move_for_player(move, PLAYER2)
        expected = ((n - 6, n - 6), (n - 4, n - 4), (n - 2, n - 2))
        assert transformed == expected

    def test_transform_inverse_roundtrip_player1(self) -> None:
        move: Move = ((1, 2), (3, 4))
        assert inverse_transform_move_for_player(
            transform_move_for_player(move, PLAYER1), PLAYER1
        ) == move

    def test_transform_inverse_roundtrip_player2(self) -> None:
        move: Move = ((1, 2), (3, 4))
        assert inverse_transform_move_for_player(
            transform_move_for_player(move, PLAYER2), PLAYER2
        ) == move

    def test_transform_inverse_roundtrip_chain_player2(self) -> None:
        move: Move = ((0, 0), (2, 2), (4, 4))
        assert inverse_transform_move_for_player(
            transform_move_for_player(move, PLAYER2), PLAYER2
        ) == move

    def test_length_preserved(self) -> None:
        move: Move = ((0, 0), (2, 2), (4, 4), (6, 6))
        assert len(transform_move_for_player(move, PLAYER2)) == len(move)

    def test_encode_action_consistent_with_transform(self) -> None:
        """Encoding a transformed move must match encoding the original after
        applying the same transform to from/to positions."""
        move: Move = ((5, 5), (7, 7))
        t_move = transform_move_for_player(move, PLAYER2)
        aid_real = encode_action(move)
        aid_canonical = encode_action(t_move)
        # They should differ (different coordinates)
        assert aid_real != aid_canonical
        # But decoding the canonical id and inverse-transforming gives back original
        fp_c, tp_c = decode_action(aid_canonical)
        fp_r = inverse_transform_pos_for_player(fp_c, PLAYER2)
        tp_r = inverse_transform_pos_for_player(tp_c, PLAYER2)
        assert fp_r == move[0]
        assert tp_r == move[-1]


# ── Integration: full encode → select → decode workflow ───────────────────────

class TestWorkflow:
    """End-to-end sanity: encode state, build mask, select action, recover move."""

    def _workflow(self, player: int) -> None:
        import random

        board = initial_board()
        real_moves = get_legal_moves(board, player)

        # 1. Encode state
        obs = encode_state(board, player)
        assert obs.shape == (STATE_CHANNELS, BOARD_SIZE, BOARD_SIZE)

        # 2. Transform moves to canonical frame and build mask
        canonical_moves = [transform_move_for_player(m, player) for m in real_moves]
        mask = legal_action_mask(canonical_moves)
        assert mask.any(), "Mask must have at least one legal action"

        # 3. Simulate DQN choosing a random legal canonical action
        legal_ids = np.where(mask)[0].tolist()
        chosen_id = random.choice(legal_ids)

        # 4. Recover canonical move, inverse-transform to real move
        canonical_move = action_id_to_move(chosen_id, canonical_moves)
        real_move = inverse_transform_move_for_player(canonical_move, player)

        # 5. The recovered move must be in the real legal moves
        # (match by start and end)
        matching = [m for m in real_moves
                    if m[0] == real_move[0] and m[-1] == real_move[-1]]
        assert matching, (
            f"Recovered real move {real_move} not found in legal moves for "
            f"player {player}."
        )

    def test_workflow_player1(self) -> None:
        self._workflow(PLAYER1)

    def test_workflow_player2(self) -> None:
        self._workflow(PLAYER2)
