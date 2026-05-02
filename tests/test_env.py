"""Tests for CornersEnv."""

import numpy as np
import pytest

from corners_rl.env.corners_env import CornersEnv
from corners_rl.env.moves import Move, get_legal_moves
from corners_rl.env.rules import (
    BOARD_SIZE,
    EMPTY,
    PLAYER1,
    PLAYER2,
    get_target_zone,
    initial_board,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def env() -> CornersEnv:
    return CornersEnv(max_moves=500)


@pytest.fixture()
def tiny_env() -> CornersEnv:
    """Environment with max_moves=2 to test the draw condition cheaply."""
    return CornersEnv(max_moves=2)


# ── reset ─────────────────────────────────────────────────────────────────────

class TestReset:
    def test_returns_correct_shape(self, env: CornersEnv) -> None:
        obs = env.reset()
        assert obs.shape == (BOARD_SIZE, BOARD_SIZE)

    def test_initial_player_is_player1(self, env: CornersEnv) -> None:
        env.reset()
        assert env.current_player == PLAYER1

    def test_move_count_is_zero(self, env: CornersEnv) -> None:
        # Play a move, then reset
        env.reset()
        moves = env.legal_moves()
        env.step(moves[0])
        env.reset()
        assert env.move_count == 0

    def test_winner_is_none_after_reset(self, env: CornersEnv) -> None:
        env.reset()
        assert env.winner is None

    def test_is_not_terminal_after_reset(self, env: CornersEnv) -> None:
        env.reset()
        assert not env.is_terminal()

    def test_board_matches_initial_board(self, env: CornersEnv) -> None:
        obs = env.reset()
        np.testing.assert_array_equal(obs, initial_board())

    def test_reset_clears_previous_game_state(self, env: CornersEnv) -> None:
        env.reset()
        # Play a few moves
        for _ in range(5):
            moves = env.legal_moves()
            env.step(moves[0])
        env.reset()
        assert env.move_count == 0
        assert env.current_player == PLAYER1
        assert not env.is_terminal()


# ── legal_moves ───────────────────────────────────────────────────────────────

class TestLegalMoves:
    def test_has_moves_at_start(self, env: CornersEnv) -> None:
        env.reset()
        assert len(env.legal_moves()) > 0

    def test_legal_moves_belong_to_current_player(self, env: CornersEnv) -> None:
        env.reset()
        board = env.board
        player = env.current_player
        for m in env.legal_moves():
            r, c = m[0]
            assert board[r, c] == player

    def test_legal_moves_after_one_step(self, env: CornersEnv) -> None:
        env.reset()
        moves = env.legal_moves()
        env.step(moves[0])
        # Now it's the other player's turn — should also have moves
        assert len(env.legal_moves()) > 0


# ── step ──────────────────────────────────────────────────────────────────────

class TestStep:
    def test_step_returns_four_tuple(self, env: CornersEnv) -> None:
        env.reset()
        result = env.step(env.legal_moves()[0])
        assert len(result) == 4

    def test_observation_shape(self, env: CornersEnv) -> None:
        env.reset()
        obs, *_ = env.step(env.legal_moves()[0])
        assert obs.shape == (BOARD_SIZE, BOARD_SIZE)

    def test_step_switches_player(self, env: CornersEnv) -> None:
        env.reset()
        assert env.current_player == PLAYER1
        env.step(env.legal_moves()[0])
        assert env.current_player == PLAYER2

    def test_step_alternates_player_multiple_times(self, env: CornersEnv) -> None:
        env.reset()
        players = [env.current_player]
        for _ in range(4):
            env.step(env.legal_moves()[0])
            players.append(env.current_player)
        # Should alternate: 1, -1, 1, -1, 1
        assert players == [1, -1, 1, -1, 1]

    def test_step_increments_move_count(self, env: CornersEnv) -> None:
        env.reset()
        for i in range(1, 4):
            env.step(env.legal_moves()[0])
            assert env.move_count == i

    def test_step_board_changes(self, env: CornersEnv) -> None:
        env.reset()
        before = env.board
        obs, *_ = env.step(env.legal_moves()[0])
        # At least one cell must differ
        assert not np.array_equal(before, obs)

    def test_observation_matches_internal_board(self, env: CornersEnv) -> None:
        env.reset()
        obs, *_ = env.step(env.legal_moves()[0])
        np.testing.assert_array_equal(obs, env.board)

    def test_done_false_during_normal_play(self, env: CornersEnv) -> None:
        env.reset()
        _, _, done, _ = env.step(env.legal_moves()[0])
        assert not done

    def test_reward_zero_during_normal_play(self, env: CornersEnv) -> None:
        env.reset()
        _, reward, _, _ = env.step(env.legal_moves()[0])
        assert reward == 0.0

    def test_raises_after_game_over(self, tiny_env: CornersEnv) -> None:
        tiny_env.reset()
        for _ in range(2):
            tiny_env.step(tiny_env.legal_moves()[0])
        assert tiny_env.is_terminal()
        with pytest.raises(RuntimeError, match="already over"):
            tiny_env.step(tiny_env.legal_moves()[0])

    def test_info_contains_expected_keys(self, env: CornersEnv) -> None:
        env.reset()
        _, _, _, info = env.step(env.legal_moves()[0])
        assert "move_count" in info
        assert "winner" in info
        assert "current_player" in info


# ── draw: max_moves ───────────────────────────────────────────────────────────

class TestMaxMoves:
    def test_terminates_after_max_moves(self, tiny_env: CornersEnv) -> None:
        tiny_env.reset()
        for _ in range(2):
            _, _, done, _ = tiny_env.step(tiny_env.legal_moves()[0])
        assert done

    def test_winner_is_none_on_draw(self, tiny_env: CornersEnv) -> None:
        tiny_env.reset()
        for _ in range(2):
            tiny_env.step(tiny_env.legal_moves()[0])
        assert tiny_env.winner is None

    def test_is_terminal_true_on_draw(self, tiny_env: CornersEnv) -> None:
        tiny_env.reset()
        for _ in range(2):
            tiny_env.step(tiny_env.legal_moves()[0])
        assert tiny_env.is_terminal()

    def test_reward_is_zero_on_draw(self, tiny_env: CornersEnv) -> None:
        tiny_env.reset()
        reward = 0.0
        for _ in range(2):
            _, reward, _, _ = tiny_env.step(tiny_env.legal_moves()[0])
        # Last step triggers draw → reward must be 0
        assert reward == 0.0


# ── win detection via step ────────────────────────────────────────────────────

class TestWinDetection:
    def _build_one_move_win_board(self) -> tuple[np.ndarray, Move]:
        """Board where P1 needs exactly one step to win.

        All 9 P1 pieces already fill the bottom-right zone except (7,7).
        One P1 piece sits at (7, 6) [inside the zone — already counted].

        Actually simpler: 8 pieces in target zone, 1 piece one step away.
        """
        board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        # Sort for deterministic ordering; sorted first cell will be (5,5).
        target = sorted(get_target_zone(PLAYER1))  # 9 cells in bottom-right
        # Place 8 pieces in the target zone, leaving target[0] = (5,5) empty.
        for r, c in target[1:]:
            board[r, c] = PLAYER1
        lr, lc = target[0]  # (5, 5) — top-left corner of the zone
        # Place 9th piece one row above: (4, 5) is guaranteed empty
        src_r, src_c = lr - 1, lc
        board[src_r, src_c] = PLAYER1
        winning_move: Move = ((src_r, src_c), (lr, lc))  # (4,5) → (5,5)
        return board, winning_move

    def test_player1_wins(self) -> None:
        board, move = self._build_one_move_win_board()
        env = CornersEnv()
        env.reset()
        # Directly inject the crafted board
        env._board = board
        env._current_player = PLAYER1

        _, reward, done, info = env.step(move)
        assert done
        assert reward == 1.0
        assert info["winner"] == PLAYER1
        assert env.winner == PLAYER1
        assert env.is_terminal()

    def test_player2_wins(self) -> None:
        """Mirror of the P1 win test, for P2 targeting the top-left zone."""
        board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        # Sort for deterministic ordering; sorted last cell will be (2,2).
        target = sorted(get_target_zone(PLAYER2))  # top-left 3×3
        for r, c in target[:-1]:
            board[r, c] = PLAYER2
        lr, lc = target[-1]   # (2, 2) — bottom-right corner of the zone
        # Place 9th piece one row below: (3, 2) is guaranteed empty
        src_r, src_c = lr + 1, lc
        board[src_r, src_c] = PLAYER2
        winning_move: Move = ((src_r, src_c), (lr, lc))

        env = CornersEnv()
        env.reset()
        env._board = board
        env._current_player = PLAYER2

        _, reward, done, info = env.step(winning_move)
        assert done
        assert reward == 1.0
        assert info["winner"] == PLAYER2


# ── clone ─────────────────────────────────────────────────────────────────────

class TestClone:
    def test_clone_is_independent(self, env: CornersEnv) -> None:
        env.reset()
        clone = env.clone()
        # Step in the clone — original must not change
        original_board = env.board.copy()
        clone.step(clone.legal_moves()[0])
        np.testing.assert_array_equal(env.board, original_board)

    def test_clone_step_does_not_affect_original_player(self, env: CornersEnv) -> None:
        env.reset()
        assert env.current_player == PLAYER1
        clone = env.clone()
        clone.step(clone.legal_moves()[0])
        assert env.current_player == PLAYER1  # original unchanged

    def test_clone_copies_move_count(self, env: CornersEnv) -> None:
        env.reset()
        env.step(env.legal_moves()[0])
        clone = env.clone()
        assert clone.move_count == 1

    def test_clone_same_board(self, env: CornersEnv) -> None:
        env.reset()
        clone = env.clone()
        np.testing.assert_array_equal(env.board, clone.board)


# ── board property ────────────────────────────────────────────────────────────

class TestBoardProperty:
    def test_board_is_a_copy(self, env: CornersEnv) -> None:
        env.reset()
        b = env.board
        b[0, 0] = 99  # mutate the returned copy
        assert env.board[0, 0] != 99  # internal state unchanged

    def test_board_dtype(self, env: CornersEnv) -> None:
        env.reset()
        assert env.board.dtype == np.int8


# ── render_ascii ──────────────────────────────────────────────────────────────

class TestRenderAscii:
    def test_returns_string(self, env: CornersEnv) -> None:
        env.reset()
        output = env.render_ascii()
        assert isinstance(output, str)

    def test_contains_player_symbols(self, env: CornersEnv) -> None:
        env.reset()
        output = env.render_ascii()
        assert "O" in output  # P1 pieces
        assert "X" in output  # P2 pieces
