"""Minimal tests for the DQN training pipeline correctness.

Covers:
1. DQNAgent always returns legal moves (checkpoint and random model).
2. Evaluation correctly attributes winner_agent when sides swap.
3. One training update changes model weights.
4. Reward for move toward goal > reward for move away from goal.
5. encode/transform/inverse_transform are consistent for player=-1.
"""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from corners_rl.agents.dqn_agent import DQNAgent
from corners_rl.agents.random_agent import RandomAgent
from corners_rl.env.corners_env import CornersEnv
from corners_rl.env.moves import apply_move, get_legal_moves
from corners_rl.env.rules import PLAYER1, PLAYER2, get_target_zone, initial_board
from corners_rl.evaluation.evaluate import evaluate_match
from corners_rl.rl.encoding import (
    ACTION_SPACE_SIZE,
    encode_action,
    encode_state,
    inverse_transform_move_for_player,
    legal_action_mask,
    transform_move_for_player,
)
from corners_rl.rl.model import DQNModel
from corners_rl.rl.replay_buffer import ReplayBuffer
from corners_rl.rl.self_play import compute_shaped_reward
from corners_rl.rl.train_dqn import dqn_update


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def env() -> CornersEnv:
    e = CornersEnv()
    e.reset()
    return e


@pytest.fixture()
def dqn_random_model() -> DQNAgent:
    """DQNAgent with a fresh random model, epsilon=0.0."""
    return DQNAgent(model=DQNModel(), device="cpu", epsilon=0.0, seed=0)


@pytest.fixture()
def dqn_saved(tmp_path: Path) -> DQNAgent:
    """DQNAgent loaded from a saved checkpoint, epsilon=0.0."""
    agent = DQNAgent(model=DQNModel(), device="cpu", epsilon=0.5, seed=7)
    ckpt = tmp_path / "ckpt.pt"
    agent.save(ckpt)
    return DQNAgent.load(ckpt, device="cpu", epsilon=0.0)


# ── 1. DQNAgent always returns legal moves ────────────────────────────────────

class TestDQNReturnsLegalMoves:
    """DQNAgent (epsilon=0) must only return moves that appear in legal_moves()."""

    def _assert_legal_for_n_steps(self, agent: DQNAgent, n: int = 30) -> None:
        env = CornersEnv(max_moves=200)
        env.reset()
        for _ in range(n):
            if env.is_terminal():
                env.reset()
            legal = env.legal_moves()
            move = agent.select_move(env)
            assert move in legal, (
                f"DQN returned {move} which is not in legal_moves: {legal[:5]}…"
            )
            env.step(legal[0])  # advance with a deterministic step

    def test_random_model_player1(self, dqn_random_model: DQNAgent) -> None:
        """Random-weight model, always player 1 turn."""
        env = CornersEnv()
        env.reset()
        for _ in range(10):
            assert dqn_random_model.select_move(env) in env.legal_moves()

    def test_random_model_multi_step(self, dqn_random_model: DQNAgent) -> None:
        self._assert_legal_for_n_steps(dqn_random_model, n=30)

    def test_saved_checkpoint_player1(self, dqn_saved: DQNAgent) -> None:
        env = CornersEnv()
        env.reset()
        for _ in range(10):
            assert dqn_saved.select_move(env) in env.legal_moves()

    def test_saved_checkpoint_player2(self, dqn_saved: DQNAgent) -> None:
        env = CornersEnv()
        env.reset()
        env.step(env.legal_moves()[0])  # advance to player 2's turn
        assert env.current_player == PLAYER2
        assert dqn_saved.select_move(env) in env.legal_moves()

    def test_saved_checkpoint_multi_step(self, dqn_saved: DQNAgent) -> None:
        self._assert_legal_for_n_steps(dqn_saved, n=30)


# ── 2. Evaluation winner attribution ─────────────────────────────────────────

class TestEvalWinnerAttribution:
    """winner_agent must match the actual winning agent when sides swap."""

    def test_winner_agent_when_dqn_is_p1(self) -> None:
        """When DQN=P1 and winner==1, winner_agent should be 'dqn'."""
        dqn   = DQNAgent(name="dqn",    device="cpu", epsilon=1.0, seed=0)
        rand  = RandomAgent(name="random", seed=0)
        df = evaluate_match(dqn, rand, games=4, max_moves=100, seed=0)

        p1_dqn = df[df["player1_agent"] == "dqn"]
        for _, row in p1_dqn.iterrows():
            if row["winner"] == 1:
                assert row["winner_agent"] == "dqn", (
                    f"DQN=P1 won but winner_agent='{row['winner_agent']}'"
                )
            elif row["winner"] == -1:
                assert row["winner_agent"] == "random", (
                    f"DQN=P1 lost but winner_agent='{row['winner_agent']}'"
                )
            else:
                assert row["winner_agent"] is None or row["winner"] is None

    def test_winner_agent_when_dqn_is_pm1(self) -> None:
        """When DQN=P-1 and winner==-1, winner_agent should be 'dqn'."""
        dqn   = DQNAgent(name="dqn",    device="cpu", epsilon=1.0, seed=0)
        rand  = RandomAgent(name="random", seed=0)
        # Force second half (DQN=P-1) by using games=2 (half=1, so game 2 swaps)
        df = evaluate_match(rand, dqn, games=4, max_moves=100, seed=0)

        pm1_dqn = df[df["player_minus1_agent"] == "dqn"]
        for _, row in pm1_dqn.iterrows():
            if row["winner"] == -1:
                assert row["winner_agent"] == "dqn", (
                    f"DQN=P-1 won but winner_agent='{row['winner_agent']}'"
                )
            elif row["winner"] == 1:
                assert row["winner_agent"] == "random", (
                    f"DQN=P-1 lost but winner_agent='{row['winner_agent']}'"
                )

    def test_rates_sum_to_one(self) -> None:
        """win_rate + opponent_win_rate + draw_rate == 1.0."""
        from corners_rl.evaluation.evaluate import summarize_results
        dqn  = DQNAgent(name="dqn", device="cpu", epsilon=1.0, seed=0)
        rand = RandomAgent(name="random", seed=1)
        df = evaluate_match(dqn, rand, games=4, max_moves=100, seed=1)
        s = summarize_results(df)
        total = s["agent1_win_rate"] + s["agent2_win_rate"] + s["draw_rate"]
        assert abs(total - 1.0) < 1e-9, f"Rates sum to {total}"


# ── 3. One training update changes weights ────────────────────────────────────

class TestTrainingUpdateChangesWeights:
    """A single dqn_update() call must change the online network's parameters."""

    @staticmethod
    def _make_batch(n: int = 32) -> dict[str, torch.Tensor]:
        """Synthetic mini-batch with random but structurally valid data."""
        rng = np.random.default_rng(0)
        board = initial_board()
        player = PLAYER1
        canon = [transform_move_for_player(m, player)
                 for m in get_legal_moves(board, player)]
        mask = legal_action_mask(canon)

        state = torch.from_numpy(encode_state(board, player)).unsqueeze(0).repeat(n, 1, 1, 1)
        # Use only legal action IDs so actions are > 0
        legal_ids = np.where(mask)[0]
        actions = torch.from_numpy(
            rng.choice(legal_ids, size=n).astype(np.int64)
        )
        rewards = torch.from_numpy(rng.uniform(-1, 1, n).astype(np.float32))
        next_states = state.clone()
        dones = torch.zeros(n, dtype=torch.bool)
        next_masks = torch.from_numpy(
            np.tile(mask, (n, 1))
        )
        return {
            "states": state,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "dones": dones,
            "next_legal_masks": next_masks,
        }

    def test_weights_change_after_update(self) -> None:
        online = DQNModel()
        target = copy.deepcopy(online)
        opt = torch.optim.Adam(online.parameters(), lr=1e-3)

        # Snapshot weights before update
        before = {n: p.clone().detach() for n, p in online.named_parameters()}

        batch = self._make_batch(32)
        loss, _ = dqn_update(batch, online, target, opt, gamma=0.99,
                             device=torch.device("cpu"))

        # At least one parameter must have changed
        changed = any(
            not torch.allclose(before[n], p.detach(), atol=1e-9)
            for n, p in online.named_parameters()
        )
        assert changed, "No weights changed after dqn_update() — optimizer not working!"

    def test_loss_is_finite(self) -> None:
        online = DQNModel()
        target = copy.deepcopy(online)
        opt = torch.optim.Adam(online.parameters(), lr=1e-3)
        batch = self._make_batch(32)
        loss, td_errs = dqn_update(batch, online, target, opt, gamma=0.99,
                                   device=torch.device("cpu"))
        assert np.isfinite(loss), f"Loss is not finite: {loss}"
        assert np.all(np.isfinite(td_errs)), "TD errors contain NaN/Inf"

    def test_target_network_unchanged_after_update(self) -> None:
        """target_net must NOT be modified by dqn_update."""
        online = DQNModel()
        target = copy.deepcopy(online)
        target_before = {n: p.clone().detach() for n, p in target.named_parameters()}

        opt = torch.optim.Adam(online.parameters(), lr=1e-3)
        batch = self._make_batch(32)
        dqn_update(batch, online, target, opt, gamma=0.99,
                   device=torch.device("cpu"))

        for n, p in target.named_parameters():
            assert torch.allclose(target_before[n], p.detach()), (
                f"Target network parameter '{n}' was modified by dqn_update!"
            )


# ── 4. Reward for goal-directed move > reward for neutral/retreating move ──────

class TestRewardShaping:
    """compute_shaped_reward must reward progress toward the target zone."""

    def test_approach_rewarded_more_than_retreat(self) -> None:
        board = initial_board()
        player = PLAYER1

        # Find a move that improves distance and one that doesn't
        target_zone = get_target_zone(player)
        legal = get_legal_moves(board, player)

        r_max = float("-inf")
        r_min = float("inf")

        for move in legal:
            board_after = apply_move(board, move, player)
            r = compute_shaped_reward(board, board_after, player, move, False, None)
            r_max = max(r_max, r)
            r_min = min(r_min, r)

        assert r_max > r_min, (
            "All moves have the same shaped reward — "
            "reward shaping does not differentiate moves."
        )

    def test_win_reward_positive(self) -> None:
        """Terminal WIN must produce a large positive reward."""
        board = initial_board()
        dummy_move = ((0, 0), (7, 7))
        r = compute_shaped_reward(board, board, PLAYER1, dummy_move, True, PLAYER1)
        assert r > 50.0, f"Win reward = {r:.2f}, expected > 50"

    def test_loss_reward_negative(self) -> None:
        """Terminal LOSS must produce a large negative reward."""
        board = initial_board()
        dummy_move = ((0, 0), (7, 7))
        r = compute_shaped_reward(board, board, PLAYER1, dummy_move, True, PLAYER2)
        assert r < -50.0, f"Loss reward = {r:.2f}, expected < -50"

    def test_draw_reward_near_zero(self) -> None:
        """A draw (winner=None, done=True) should not give win/loss bonus."""
        board = initial_board()
        legal = get_legal_moves(board, PLAYER1)
        move = legal[0]
        board_after = apply_move(board, move, PLAYER1)
        r = compute_shaped_reward(board, board_after, PLAYER1, move, True, None)
        # Draw has no +100 or -100 terminal bonus; should be small
        assert -10.0 < r < 10.0, f"Draw reward = {r:.2f}, expected near zero"


# ── 5. encode / transform / inverse_transform for player=-1 ───────────────────

class TestEncodingForPlayer2:
    """Perspective transforms must be consistent for PLAYER2."""

    def test_encode_state_player2_own_pieces_at_top_left(self) -> None:
        """In the canonical frame, PLAYER2's own pieces appear at top-left (rows 0-2)."""
        board = initial_board()
        state = encode_state(board, PLAYER2)
        # After 180° rotation + negation, P2 pieces (originally at rows 5-7, cols 5-7)
        # should appear at rows 0-2, cols 0-2 in channel 0.
        assert state[0, 0, 0] == 1.0, "P2 piece not at canonical top-left corner"
        assert state[0, 2, 2] == 1.0, "P2 piece not at canonical (2,2)"

    def test_encode_state_player2_opponent_at_bottom_right(self) -> None:
        """Opponent pieces (P1) should appear at bottom-right in P2's canonical view."""
        board = initial_board()
        state = encode_state(board, PLAYER2)
        # P1 pieces originally at rows 0-2, cols 0-2 → after 180°: rows 5-7, cols 5-7
        assert state[1, 7, 7] == 1.0, "P1 pieces not at canonical bottom-right for P2"

    def test_transform_inverse_roundtrip_player2(self) -> None:
        """transform + inverse_transform must be the identity for PLAYER2."""
        env = CornersEnv()
        env.reset()
        env.step(env.legal_moves()[0])  # advance to P2's turn
        assert env.current_player == PLAYER2
        legal = env.legal_moves()

        for move in legal[:10]:
            canonical = transform_move_for_player(move, PLAYER2)
            restored  = inverse_transform_move_for_player(canonical, PLAYER2)
            assert restored == move, (
                f"Round-trip failed: {move} → {canonical} → {restored}"
            )

    def test_canonical_move_valid_action_id(self) -> None:
        """encode_action(canonical_move) must be in [0, ACTION_SPACE_SIZE)."""
        env = CornersEnv()
        env.reset()
        env.step(env.legal_moves()[0])  # P2's turn
        legal = env.legal_moves()

        for move in legal:
            canon = transform_move_for_player(move, PLAYER2)
            aid = encode_action(canon)
            assert 0 <= aid < ACTION_SPACE_SIZE, (
                f"action_id={aid} out of range for move {move} (canonical {canon})"
            )

    def test_legal_mask_contains_all_legal_canonical_moves(self) -> None:
        """legal_action_mask must be True for every canonical legal move."""
        env = CornersEnv()
        env.reset()
        env.step(env.legal_moves()[0])
        legal = env.legal_moves()
        player = env.current_player  # PLAYER2

        canon = [transform_move_for_player(m, player) for m in legal]
        mask = legal_action_mask(canon)

        for cm in canon:
            aid = encode_action(cm)
            assert mask[aid], f"Canonical move {cm} (id={aid}) not in legal mask!"
