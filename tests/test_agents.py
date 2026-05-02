"""Tests for baseline agents and the play_game runner."""

from __future__ import annotations

import random

import numpy as np
import pytest

from corners_rl.agents.base import BaseAgent
from corners_rl.agents.greedy_agent import GreedyAgent, total_distance
from corners_rl.agents.heuristic_agent import HeuristicAgent
from corners_rl.agents.random_agent import RandomAgent
from corners_rl.env.corners_env import CornersEnv
from corners_rl.env.moves import Move, get_legal_moves
from corners_rl.env.rules import (
    PLAYER1,
    PLAYER2,
    get_target_zone,
    initial_board,
)
from corners_rl.game_runner import play_game


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def env() -> CornersEnv:
    e = CornersEnv(max_moves=500)
    e.reset()
    return e


def _make_agents(seed: int = 0):
    return (
        RandomAgent(seed=seed),
        GreedyAgent(seed=seed),
        HeuristicAgent(seed=seed),
    )


# ── BaseAgent ─────────────────────────────────────────────────────────────────

class TestBaseAgent:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            BaseAgent("x")  # type: ignore[abstract]

    def test_name_property(self) -> None:
        agent = RandomAgent(name="my_random")
        assert agent.name == "my_random"

    def test_repr_contains_name(self) -> None:
        agent = GreedyAgent(name="test_greedy")
        assert "test_greedy" in repr(agent)


# ── RandomAgent ───────────────────────────────────────────────────────────────

class TestRandomAgent:
    def test_returns_legal_move(self, env: CornersEnv) -> None:
        agent = RandomAgent(seed=7)
        move = agent.select_move(env)
        assert move in env.legal_moves()

    def test_different_seeds_can_produce_different_moves(self, env: CornersEnv) -> None:
        """Over many seeds at least two distinct moves should appear."""
        moves_seen: set = set()
        for s in range(20):
            m = RandomAgent(seed=s).select_move(env)
            moves_seen.add(m)
        # The initial position has many legal moves; we expect > 1 distinct choice.
        assert len(moves_seen) > 1

    def test_same_seed_same_move(self, env: CornersEnv) -> None:
        m1 = RandomAgent(seed=42).select_move(env)
        m2 = RandomAgent(seed=42).select_move(env)
        assert m1 == m2

    def test_accepts_random_instance(self, env: CornersEnv) -> None:
        rng = random.Random(99)
        agent = RandomAgent(seed=rng)
        move = agent.select_move(env)
        assert move in env.legal_moves()

    def test_raises_on_no_legal_moves(self) -> None:
        """Agent should raise when there are no moves (manually force empty list)."""
        env = CornersEnv()
        env.reset()
        agent = RandomAgent(seed=0)
        # Monkey-patch legal_moves to return empty
        env.legal_moves = lambda: []  # type: ignore[method-assign]
        with pytest.raises(ValueError, match="No legal moves"):
            agent.select_move(env)


# ── GreedyAgent ───────────────────────────────────────────────────────────────

class TestGreedyAgent:
    def test_returns_legal_move(self, env: CornersEnv) -> None:
        agent = GreedyAgent(seed=0)
        move = agent.select_move(env)
        assert move in env.legal_moves()

    def test_move_does_not_worsen_distance(self, env: CornersEnv) -> None:
        """Greedy must never increase total distance (improvement ≥ 0)."""
        agent = GreedyAgent(seed=0)
        player = env.current_player
        board = env.board
        before = total_distance(board, player)
        move = agent.select_move(env)
        new_board = __import__(
            "corners_rl.env.moves", fromlist=["apply_move"]
        ).apply_move(board, move, player)
        after = total_distance(new_board, player)
        assert after <= before

    def test_greedy_prefers_shorter_distance(self) -> None:
        """On a crafted board, greedy must pick the move that reduces distance most."""
        # Set up: 8 P1 pieces already in target zone (all except (5,5)),
        # one piece at (4,5) — one orthogonal step away from the empty (5,5).
        # With orthogonal-only rules, (4,5)→(5,5) is the unique best move.
        board = np.zeros((8, 8), dtype=np.int8)
        target = sorted(get_target_zone(PLAYER1))
        for r, c in target[1:]:   # fill all except target[0] = (5,5)
            board[r, c] = PLAYER1
        board[4, 5] = PLAYER1     # one step above (5,5) — orthogonal move available

        env = CornersEnv()
        env.reset()
        env._board = board
        env._current_player = PLAYER1

        agent = GreedyAgent(seed=0)
        move = agent.select_move(env)
        # The winning move must end at (5,5) (distance 0 improvement is maximal)
        assert move[-1] == (5, 5)

    def test_tie_breaking_is_random(self, env: CornersEnv) -> None:
        """When multiple moves are equally good, greedy makes varied choices."""
        # Run greedy with many different seeds and collect chosen moves.
        choices = set()
        for s in range(30):
            m = GreedyAgent(seed=s).select_move(env)
            choices.add(m)
        # There should be more than 1 distinct move chosen across seeds.
        assert len(choices) > 1

    def test_same_seed_same_move(self, env: CornersEnv) -> None:
        m1 = GreedyAgent(seed=7).select_move(env)
        m2 = GreedyAgent(seed=7).select_move(env)
        assert m1 == m2


# ── total_distance ────────────────────────────────────────────────────────────

class TestTotalDistance:
    def test_zero_when_all_in_target(self) -> None:
        board = np.zeros((8, 8), dtype=np.int8)
        for r, c in get_target_zone(PLAYER1):
            board[r, c] = PLAYER1
        assert total_distance(board, PLAYER1) == 0

    def test_positive_at_start(self) -> None:
        board = initial_board()
        assert total_distance(board, PLAYER1) > 0
        assert total_distance(board, PLAYER2) > 0

    def test_symmetric_at_start(self) -> None:
        """By symmetry, both players have the same total distance at the start."""
        board = initial_board()
        assert total_distance(board, PLAYER1) == total_distance(board, PLAYER2)


# ── HeuristicAgent ────────────────────────────────────────────────────────────

class TestHeuristicAgent:
    def test_returns_legal_move(self, env: CornersEnv) -> None:
        agent = HeuristicAgent(seed=0)
        move = agent.select_move(env)
        assert move in env.legal_moves()

    def test_penalises_leaving_target_zone(self) -> None:
        """Heuristic should never voluntarily move a piece OUT of the goal zone
        when better alternatives exist."""
        board = np.zeros((8, 8), dtype=np.int8)
        target = sorted(get_target_zone(PLAYER1))
        # 8 pieces in target, plus one piece both in target and with a move outside
        for r, c in target:
            board[r, c] = PLAYER1
        # Add a 9th piece just outside the zone to give a non-target option
        board[4, 4] = PLAYER1

        env = CornersEnv()
        env.reset()
        env._board = board
        env._current_player = PLAYER1

        agent = HeuristicAgent(seed=0)
        move = agent.select_move(env)
        # The chosen move must not start from inside the target zone
        # (because leaving the goal gets a heavy -5 penalty)
        assert move[0] not in get_target_zone(PLAYER1), (
            f"Heuristic chose to move a piece OUT of the goal zone: {move}"
        )

    def test_favours_entering_target_zone(self) -> None:
        """Heuristic should prefer a move that places a piece into the goal zone."""
        board = np.zeros((8, 8), dtype=np.int8)
        target = sorted(get_target_zone(PLAYER1))
        # 8 pieces already in goal
        for r, c in target[1:]:
            board[r, c] = PLAYER1
        # 9th piece one step from the last target cell
        lr, lc = target[0]   # (5, 5)
        board[lr - 1, lc] = PLAYER1   # (4, 5)

        env = CornersEnv()
        env.reset()
        env._board = board
        env._current_player = PLAYER1

        agent = HeuristicAgent(seed=0)
        move = agent.select_move(env)
        assert move[-1] == (lr, lc), (
            f"Expected piece to step into ({lr},{lc}), got move ending at {move[-1]}"
        )

    def test_same_seed_same_move(self, env: CornersEnv) -> None:
        m1 = HeuristicAgent(seed=3).select_move(env)
        m2 = HeuristicAgent(seed=3).select_move(env)
        assert m1 == m2


# ── play_game ─────────────────────────────────────────────────────────────────

class TestPlayGame:
    def test_terminates(self) -> None:
        result = play_game(RandomAgent(seed=1), RandomAgent(seed=2), max_moves=200)
        assert "winner" in result
        assert "moves" in result
        assert "draw" in result
        assert "final_board" in result

    def test_winner_is_valid(self) -> None:
        result = play_game(RandomAgent(seed=5), RandomAgent(seed=6), max_moves=200)
        assert result["winner"] in (1, -1, None)

    def test_draw_consistent_with_winner(self) -> None:
        result = play_game(RandomAgent(seed=7), RandomAgent(seed=8), max_moves=10)
        if result["draw"]:
            assert result["winner"] is None
        else:
            assert result["winner"] in (1, -1)

    def test_moves_list_is_non_empty(self) -> None:
        result = play_game(RandomAgent(seed=9), RandomAgent(seed=10), max_moves=200)
        assert len(result["moves"]) > 0

    def test_moves_respect_step_limit(self) -> None:
        max_m = 20
        result = play_game(RandomAgent(seed=11), RandomAgent(seed=12), max_moves=max_m)
        assert len(result["moves"]) <= max_m

    def test_final_board_shape(self) -> None:
        result = play_game(RandomAgent(seed=13), RandomAgent(seed=14), max_moves=100)
        assert result["final_board"].shape == (8, 8)

    def test_seed_produces_reproducible_game(self) -> None:
        r1 = play_game(RandomAgent(), RandomAgent(), max_moves=100, seed=42)
        r2 = play_game(RandomAgent(), RandomAgent(), max_moves=100, seed=42)
        assert r1["winner"] == r2["winner"]
        assert len(r1["moves"]) == len(r2["moves"])

    def test_greedy_vs_random_terminates(self) -> None:
        result = play_game(
            GreedyAgent(seed=0), RandomAgent(seed=0), max_moves=300
        )
        assert result["winner"] in (1, -1, None)

    def test_heuristic_vs_random_terminates(self) -> None:
        result = play_game(
            HeuristicAgent(seed=0), RandomAgent(seed=0), max_moves=300
        )
        assert result["winner"] in (1, -1, None)


# ── Greedy vs Random win-rate ─────────────────────────────────────────────────

class TestGreedyWinRate:
    """Check that GreedyAgent outperforms RandomAgent.

    With orthogonal-only rules the average game is much longer (total Manhattan
    distance ≈ 72 vs ≈ 36 with 8-direction rules), so max_moves=2000 is used
    to allow games to finish.  The threshold is intentionally lenient (≥ 25 %)
    to avoid flakiness while still confirming greedy is meaningfully better.
    """

    N_GAMES = 50
    SEED = 0

    def _run_series(self, n: int, seed: int) -> dict[str, int]:
        """Play *n* games greedy (P1) vs random (P2), return tallies."""
        import random as _r
        rng = _r.Random(seed)
        wins = {"greedy": 0, "random": 0, "draw": 0}
        for _ in range(n):
            s = rng.randint(0, 2**32 - 1)
            result = play_game(
                GreedyAgent(seed=s), RandomAgent(seed=s + 1), max_moves=2000
            )
            if result["winner"] == 1:
                wins["greedy"] += 1
            elif result["winner"] == -1:
                wins["random"] += 1
            else:
                wins["draw"] += 1
        return wins

    def test_win_rate_computed(self) -> None:
        """Smoke test: win rates sum to 1.0."""
        wins = self._run_series(10, self.SEED)
        total = wins["greedy"] + wins["random"] + wins["draw"]
        assert total == 10

    def test_greedy_wins_at_least_some_games(self) -> None:
        """Greedy must win at least one game in 50 tries."""
        wins = self._run_series(self.N_GAMES, self.SEED)
        assert wins["greedy"] > 0, (
            f"Greedy won 0 out of {self.N_GAMES} games — agent may be broken."
        )

    def test_greedy_win_rate_above_threshold(self) -> None:
        """Greedy win rate should exceed 25 % vs random over 50 games (orthogonal rules)."""
        wins = self._run_series(self.N_GAMES, self.SEED)
        win_rate = wins["greedy"] / self.N_GAMES
        assert win_rate >= 0.25, (
            f"Greedy win rate {win_rate:.1%} is below threshold 25%. "
            f"Results: {wins}"
        )
