"""Tests for DQN model, replay buffer, and DQN agent."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from corners_rl.agents.dqn_agent import DQNAgent
from corners_rl.env.corners_env import CornersEnv
from corners_rl.env.moves import get_legal_moves
from corners_rl.env.rules import (
    BOARD_SIZE,
    PLAYER1,
    PLAYER2,
    initial_board,
)
from corners_rl.rl.encoding import (
    ACTION_SPACE_SIZE,
    STATE_CHANNELS,
    encode_state,
    legal_action_mask,
    transform_move_for_player,
)
from corners_rl.rl.model import DQNModel, masked_argmax
from corners_rl.rl.replay_buffer import ReplayBuffer


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def model() -> DQNModel:
    return DQNModel()


@pytest.fixture()
def env() -> CornersEnv:
    e = CornersEnv()
    e.reset()
    return e


def _make_transition(seed: int = 0) -> dict:
    """Create a synthetic transition for buffer tests."""
    rng = np.random.default_rng(seed)
    board = initial_board()
    player = PLAYER1
    moves = get_legal_moves(board, player)
    canonical = [transform_move_for_player(m, player) for m in moves]
    mask = legal_action_mask(canonical)
    return dict(
        state=encode_state(board, player),
        action=int(rng.integers(0, ACTION_SPACE_SIZE)),
        reward=float(rng.uniform(-1, 1)),
        next_state=encode_state(board, player),
        done=bool(rng.integers(0, 2)),
        next_legal_mask=mask,
    )


# ── DQNModel ──────────────────────────────────────────────────────────────────

class TestDQNModel:
    def test_output_shape_batch1(self, model: DQNModel) -> None:
        x = torch.zeros(1, STATE_CHANNELS, BOARD_SIZE, BOARD_SIZE)
        out = model(x)
        assert out.shape == (1, ACTION_SPACE_SIZE)

    def test_output_shape_batch4(self, model: DQNModel) -> None:
        x = torch.zeros(4, STATE_CHANNELS, BOARD_SIZE, BOARD_SIZE)
        out = model(x)
        assert out.shape == (4, ACTION_SPACE_SIZE)

    def test_output_dtype_float32(self, model: DQNModel) -> None:
        x = torch.zeros(1, STATE_CHANNELS, BOARD_SIZE, BOARD_SIZE)
        out = model(x)
        assert out.dtype == torch.float32

    def test_different_inputs_different_outputs(self, model: DQNModel) -> None:
        x1 = torch.zeros(1, STATE_CHANNELS, BOARD_SIZE, BOARD_SIZE)
        x2 = torch.ones(1, STATE_CHANNELS, BOARD_SIZE, BOARD_SIZE)
        # With default initialisation, the outputs should differ
        assert not torch.allclose(model(x1), model(x2))

    def test_no_nan_in_output(self, model: DQNModel) -> None:
        x = torch.randn(2, STATE_CHANNELS, BOARD_SIZE, BOARD_SIZE)
        out = model(x)
        assert not torch.isnan(out).any()

    def test_gradient_flows(self, model: DQNModel) -> None:
        x = torch.randn(1, STATE_CHANNELS, BOARD_SIZE, BOARD_SIZE)
        out = model(x)
        loss = out.sum()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_action_space_size_matches_encoding(self, model: DQNModel) -> None:
        x = torch.zeros(1, STATE_CHANNELS, BOARD_SIZE, BOARD_SIZE)
        out = model(x)
        assert out.shape[1] == ACTION_SPACE_SIZE


# ── masked_argmax ─────────────────────────────────────────────────────────────

class TestMaskedArgmax:
    def test_picks_highest_legal_q(self) -> None:
        q = torch.zeros(ACTION_SPACE_SIZE)
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
        # Make action 10 legal with highest value, action 5 legal with lower value
        q[10] = 5.0
        q[5]  = 3.0
        mask[10] = True
        mask[5]  = True
        assert masked_argmax(q, mask) == 10

    def test_does_not_pick_illegal_action(self) -> None:
        q = torch.full((ACTION_SPACE_SIZE,), fill_value=0.0)
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
        # Only action 99 is legal; action 0 has the highest raw Q
        q[0]   = 100.0
        q[99]  = 1.0
        mask[99] = True
        assert masked_argmax(q, mask) == 99

    def test_returns_int(self) -> None:
        q = torch.randn(ACTION_SPACE_SIZE)
        mask = np.ones(ACTION_SPACE_SIZE, dtype=np.bool_)
        result = masked_argmax(q, mask)
        assert isinstance(result, int)

    def test_result_in_valid_range(self) -> None:
        q = torch.randn(ACTION_SPACE_SIZE)
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
        mask[:100] = True
        result = masked_argmax(q, mask)
        assert 0 <= result < ACTION_SPACE_SIZE

    def test_raises_on_empty_mask(self) -> None:
        q = torch.randn(ACTION_SPACE_SIZE)
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
        with pytest.raises(ValueError, match="no True entries"):
            masked_argmax(q, mask)

    def test_all_legal_returns_global_argmax(self) -> None:
        q = torch.randn(ACTION_SPACE_SIZE)
        mask = np.ones(ACTION_SPACE_SIZE, dtype=np.bool_)
        assert masked_argmax(q, mask) == int(q.argmax().item())

    def test_single_legal_action(self) -> None:
        q = torch.randn(ACTION_SPACE_SIZE)
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
        mask[42] = True
        assert masked_argmax(q, mask) == 42

    def test_negative_q_values(self) -> None:
        q = torch.full((ACTION_SPACE_SIZE,), fill_value=-10.0)
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
        mask[7]  = True   # value -10
        mask[13] = True   # same value, but 7 < 13 so tie-breaks to 7
        q[13] = -9.0     # 13 is slightly higher → should be picked
        assert masked_argmax(q, mask) == 13


# ── ReplayBuffer ──────────────────────────────────────────────────────────────

class TestReplayBuffer:
    def test_initial_length_zero(self) -> None:
        buf = ReplayBuffer(capacity=100)
        assert len(buf) == 0

    def test_push_increments_size(self) -> None:
        buf = ReplayBuffer(capacity=100)
        t = _make_transition()
        buf.push(**t)
        assert len(buf) == 1

    def test_push_multiple(self) -> None:
        buf = ReplayBuffer(capacity=100)
        for i in range(10):
            buf.push(**_make_transition(seed=i))
        assert len(buf) == 10

    def test_capacity_not_exceeded(self) -> None:
        buf = ReplayBuffer(capacity=5)
        for i in range(20):
            buf.push(**_make_transition(seed=i))
        assert len(buf) == 5

    def test_capacity_property(self) -> None:
        buf = ReplayBuffer(capacity=42)
        assert buf.capacity == 42

    def test_sample_returns_correct_keys(self) -> None:
        buf = ReplayBuffer(capacity=100)
        for i in range(10):
            buf.push(**_make_transition(seed=i))
        batch = buf.sample(5)
        for key in ("states", "actions", "rewards", "next_states",
                    "dones", "next_legal_masks"):
            assert key in batch, f"Missing key: {key}"

    def test_sample_batch_size(self) -> None:
        buf = ReplayBuffer(capacity=100)
        for i in range(20):
            buf.push(**_make_transition(seed=i))
        batch = buf.sample(8)
        assert batch["states"].shape[0] == 8

    def test_sample_states_shape(self) -> None:
        buf = ReplayBuffer(capacity=50)
        for i in range(10):
            buf.push(**_make_transition(seed=i))
        batch = buf.sample(4)
        assert batch["states"].shape == (4, STATE_CHANNELS, BOARD_SIZE, BOARD_SIZE)

    def test_sample_next_states_shape(self) -> None:
        buf = ReplayBuffer(capacity=50)
        for i in range(10):
            buf.push(**_make_transition(seed=i))
        batch = buf.sample(4)
        assert batch["next_states"].shape == (4, STATE_CHANNELS, BOARD_SIZE, BOARD_SIZE)

    def test_sample_masks_shape(self) -> None:
        buf = ReplayBuffer(capacity=50)
        for i in range(10):
            buf.push(**_make_transition(seed=i))
        batch = buf.sample(4)
        assert batch["next_legal_masks"].shape == (4, ACTION_SPACE_SIZE)

    def test_sample_masks_are_bool(self) -> None:
        buf = ReplayBuffer(capacity=50)
        for i in range(10):
            buf.push(**_make_transition(seed=i))
        batch = buf.sample(4)
        assert batch["next_legal_masks"].dtype == torch.bool

    def test_sample_returns_torch_tensors(self) -> None:
        buf = ReplayBuffer(capacity=50)
        for i in range(10):
            buf.push(**_make_transition(seed=i))
        batch = buf.sample(4)
        for key, val in batch.items():
            assert isinstance(val, torch.Tensor), f"{key} is not a Tensor"

    def test_sample_raises_when_buffer_too_small(self) -> None:
        buf = ReplayBuffer(capacity=100)
        buf.push(**_make_transition())
        with pytest.raises(ValueError):
            buf.sample(10)

    def test_mask_roundtrip_after_push_sample(self) -> None:
        """The mask recovered from the buffer must equal the original."""
        buf = ReplayBuffer(capacity=10, seed=0)
        t = _make_transition(seed=0)
        buf.push(**t)
        batch = buf.sample(1)
        recovered = batch["next_legal_masks"][0].numpy()
        np.testing.assert_array_equal(recovered, t["next_legal_mask"])

    def test_reward_values_preserved(self) -> None:
        buf = ReplayBuffer(capacity=10, seed=0)
        t = _make_transition(seed=7)
        buf.push(**t)
        batch = buf.sample(1)
        assert abs(float(batch["rewards"][0]) - t["reward"]) < 1e-6

    def test_done_value_preserved(self) -> None:
        buf = ReplayBuffer(capacity=10)
        for done_val in (True, False):
            t = _make_transition()
            t["done"] = done_val
            buf.push(**t)
        batch = buf.sample(2)
        assert batch["dones"].dtype == torch.bool

    def test_repr(self) -> None:
        buf = ReplayBuffer(capacity=200)
        assert "0/200" in repr(buf)


# ── DQNAgent ──────────────────────────────────────────────────────────────────

class TestDQNAgent:
    # ---------- construction ----------

    def test_default_construction(self) -> None:
        agent = DQNAgent()
        assert isinstance(agent, DQNAgent)

    def test_name_property(self) -> None:
        agent = DQNAgent(name="test_dqn")
        assert agent.name == "test_dqn"

    def test_epsilon_property(self) -> None:
        agent = DQNAgent(epsilon=0.5)
        assert agent.epsilon == pytest.approx(0.5)

    def test_set_epsilon(self) -> None:
        agent = DQNAgent(epsilon=1.0)
        agent.set_epsilon(0.1)
        assert agent.epsilon == pytest.approx(0.1)

    def test_set_epsilon_clamps_below_zero(self) -> None:
        agent = DQNAgent()
        agent.set_epsilon(-0.5)
        assert agent.epsilon == 0.0

    def test_set_epsilon_clamps_above_one(self) -> None:
        agent = DQNAgent()
        agent.set_epsilon(2.0)
        assert agent.epsilon == 1.0

    # ---------- select_move: exploration (epsilon = 1.0) ----------

    def test_select_move_epsilon1_returns_legal_move(self, env: CornersEnv) -> None:
        agent = DQNAgent(epsilon=1.0, seed=0)
        move = agent.select_move(env)
        assert move in env.legal_moves()

    def test_select_move_epsilon1_player2_returns_legal_move(self) -> None:
        env = CornersEnv()
        env.reset()
        # Advance to player 2's turn
        env.step(env.legal_moves()[0])
        assert env.current_player == PLAYER2
        agent = DQNAgent(epsilon=1.0, seed=0)
        move = agent.select_move(env)
        assert move in env.legal_moves()

    def test_select_move_epsilon1_reproducible_with_seed(self, env: CornersEnv) -> None:
        m1 = DQNAgent(epsilon=1.0, seed=77).select_move(env)
        m2 = DQNAgent(epsilon=1.0, seed=77).select_move(env)
        assert m1 == m2

    # ---------- select_move: exploitation (epsilon = 0.0) ----------

    def test_select_move_epsilon0_returns_legal_move(self, env: CornersEnv) -> None:
        agent = DQNAgent(epsilon=0.0, seed=0)
        move = agent.select_move(env)
        assert move in env.legal_moves()

    def test_select_move_epsilon0_player2_returns_legal_move(self) -> None:
        env = CornersEnv()
        env.reset()
        env.step(env.legal_moves()[0])
        assert env.current_player == PLAYER2
        agent = DQNAgent(epsilon=0.0, seed=0)
        move = agent.select_move(env)
        assert move in env.legal_moves()

    def test_select_move_epsilon0_deterministic(self, env: CornersEnv) -> None:
        """Same model + same state → same greedy move."""
        model = DQNModel()
        m1 = DQNAgent(model=model, epsilon=0.0).select_move(env)
        m2 = DQNAgent(model=model, epsilon=0.0).select_move(env)
        assert m1 == m2

    def test_select_move_mid_game(self) -> None:
        """Agent must work at any point in the game, not just the initial state."""
        env = CornersEnv()
        env.reset()
        for _ in range(10):
            env.step(env.legal_moves()[0])
        agent = DQNAgent(epsilon=0.5, seed=0)
        move = agent.select_move(env)
        assert move in env.legal_moves()

    # ---------- save / load ----------

    def test_save_creates_file(self) -> None:
        agent = DQNAgent(epsilon=0.3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.pt"
            agent.save(path)
            assert path.exists()

    def test_load_restores_epsilon(self) -> None:
        agent = DQNAgent(epsilon=0.42)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.pt"
            agent.save(path)
            loaded = DQNAgent.load(path)
            assert loaded.epsilon == pytest.approx(0.42)

    def test_load_epsilon_override(self) -> None:
        agent = DQNAgent(epsilon=0.9)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.pt"
            agent.save(path)
            loaded = DQNAgent.load(path, epsilon=0.0)
            assert loaded.epsilon == 0.0

    def test_load_model_weights_match(self) -> None:
        """Weights must be identical after save → load."""
        agent = DQNAgent(epsilon=0.5)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.pt"
            agent.save(path)
            loaded = DQNAgent.load(path)

        # Compare each parameter
        for (n1, p1), (n2, p2) in zip(
            agent._model.named_parameters(),
            loaded._model.named_parameters(),
        ):
            assert n1 == n2
            assert torch.allclose(p1, p2), f"Parameter {n1} differs after load"

    def test_loaded_agent_selects_legal_move(self, env: CornersEnv) -> None:
        agent = DQNAgent(epsilon=0.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.pt"
            agent.save(path)
            loaded = DQNAgent.load(path)
        move = loaded.select_move(env)
        assert move in env.legal_moves()

    def test_create_subdir_on_save(self) -> None:
        agent = DQNAgent()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "dir" / "ckpt.pt"
            agent.save(path)
            assert path.exists()
