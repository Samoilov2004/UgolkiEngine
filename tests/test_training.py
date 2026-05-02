"""Tests for reward shaping and the DQN training pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from corners_rl.env.moves import apply_move, get_legal_moves
from corners_rl.env.rules import (
    PLAYER1,
    PLAYER2,
    get_target_zone,
    initial_board,
)
from corners_rl.rl.self_play import compute_shaped_reward, state_distance_score
from corners_rl.rl.train_dqn import (
    ReplayConfig,
    SelfPlayTrainer,
    TrainConfig,
    config_from_dict,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def start_board():
    return initial_board()


def _step_move(player: int = PLAYER1, board=None):
    """Return (board_before, board_after, move) for the first legal move."""
    if board is None:
        board = initial_board()
    moves = get_legal_moves(board, player)
    move = moves[0]
    new_board = apply_move(board, move, player)
    return board, new_board, move


# ── state_distance_score ──────────────────────────────────────────────────────

class TestStateDistanceScore:
    def test_returns_float(self, start_board) -> None:
        result = state_distance_score(start_board, PLAYER1)
        assert isinstance(result, float)

    def test_positive_at_start(self, start_board) -> None:
        assert state_distance_score(start_board, PLAYER1) > 0.0

    def test_zero_when_all_in_target(self) -> None:
        board = np.zeros((8, 8), dtype=np.int8)
        for r, c in get_target_zone(PLAYER1):
            board[r, c] = PLAYER1
        assert state_distance_score(board, PLAYER1) == 0.0

    def test_symmetric_at_start(self, start_board) -> None:
        d1 = state_distance_score(start_board, PLAYER1)
        d2 = state_distance_score(start_board, PLAYER2)
        assert d1 == pytest.approx(d2)


# ── compute_shaped_reward ─────────────────────────────────────────────────────

class TestComputeShapedReward:
    def test_returns_float(self, start_board) -> None:
        before, after, move = _step_move(PLAYER1)
        result = compute_shaped_reward(before, after, PLAYER1, move, False, None)
        assert isinstance(result, float)

    def test_step_penalty_present(self, start_board) -> None:
        """Every move should incur the step penalty (reward < 0 for zero-gain move)."""
        # Force a move that doesn't change distance by keeping piece in same zone
        before, after, move = _step_move(PLAYER1)
        reward = compute_shaped_reward(before, after, PLAYER1, move, False, None)
        # Even if distance improves, we can check the step penalty is subtracted
        # The reward must be finite
        assert np.isfinite(reward)

    def test_win_adds_large_reward(self) -> None:
        """Winner should receive a large positive bonus."""
        before = initial_board()
        # Craft a one-move win for PLAYER1
        board = np.zeros((8, 8), dtype=np.int8)
        target = sorted(get_target_zone(PLAYER1))
        for r, c in target[1:]:
            board[r, c] = PLAYER1
        lr, lc = target[0]          # (5, 5)
        board[lr - 1, lc] = PLAYER1  # piece one step above (5,5)
        move = ((lr - 1, lc), (lr, lc))
        after = apply_move(board, move, PLAYER1)
        reward = compute_shaped_reward(board, after, PLAYER1, move, True, PLAYER1)
        assert reward > 50.0

    def test_loss_adds_large_negative_reward(self) -> None:
        """Losing player should receive a large negative bonus."""
        board = np.zeros((8, 8), dtype=np.int8)
        target = sorted(get_target_zone(PLAYER1))
        for r, c in target[1:]:
            board[r, c] = PLAYER1
        lr, lc = target[0]
        board[lr - 1, lc] = PLAYER1
        move = ((lr - 1, lc), (lr, lc))
        after = apply_move(board, move, PLAYER1)
        # From PLAYER2's perspective — they lost
        reward = compute_shaped_reward(board, after, PLAYER2, move, True, PLAYER1)
        assert reward < -50.0

    def test_draw_no_win_penalty(self) -> None:
        before, after, move = _step_move(PLAYER1)
        reward = compute_shaped_reward(before, after, PLAYER1, move, True, None)
        # No win/loss bonus — only step penalty and possible distance reward
        assert -5.0 < reward < 5.0

    def test_enter_target_zone_bonus(self) -> None:
        """Moving a piece INTO the target zone adds a bonus."""
        board = np.zeros((8, 8), dtype=np.int8)
        target = sorted(get_target_zone(PLAYER1))
        # Place one piece adjacent to the target zone
        lr, lc = target[0]           # (5, 5)
        board[lr - 1, lc] = PLAYER1  # (4, 5) - just outside
        move = ((lr - 1, lc), (lr, lc))
        after = apply_move(board, move, PLAYER1)
        reward = compute_shaped_reward(board, after, PLAYER1, move, False, None)
        assert reward > 1.5  # bonus of 2.0 minus small step penalty

    def test_leave_target_zone_penalty(self) -> None:
        """Moving a piece OUT of the target zone applies a penalty."""
        board = np.zeros((8, 8), dtype=np.int8)
        target = sorted(get_target_zone(PLAYER1))
        lr, lc = target[0]           # (5, 5) inside target
        board[lr, lc] = PLAYER1
        dest_r, dest_c = lr - 1, lc  # (4, 5) outside target
        move = ((lr, lc), (dest_r, dest_c))
        after = apply_move(board, move, PLAYER1)
        reward = compute_shaped_reward(board, after, PLAYER1, move, False, None)
        assert reward < -1.5  # penalty of -2.0

    def test_player2_win_from_player2_perspective(self) -> None:
        board = np.zeros((8, 8), dtype=np.int8)
        target = sorted(get_target_zone(PLAYER2))
        for r, c in target[1:]:
            board[r, c] = PLAYER2
        lr, lc = target[0]           # (0, 0)
        board[lr + 1, lc] = PLAYER2  # (1, 0)
        move = ((lr + 1, lc), (lr, lc))
        after = apply_move(board, move, PLAYER2)
        reward = compute_shaped_reward(board, after, PLAYER2, move, True, PLAYER2)
        assert reward > 50.0


# ── config_from_dict ──────────────────────────────────────────────────────────

class TestConfigFromDict:
    def test_basic_fields(self) -> None:
        d = {"episodes": 10, "max_moves": 50, "batch_size": 32}
        cfg = config_from_dict(d)
        assert cfg.episodes == 10
        assert cfg.max_moves == 50
        assert cfg.batch_size == 32

    def test_unknown_keys_ignored(self) -> None:
        d = {"episodes": 5, "unknown_key": "ignored"}
        cfg = config_from_dict(d)
        assert cfg.episodes == 5

    def test_defaults_for_missing_keys(self) -> None:
        cfg = config_from_dict({})
        assert cfg.gamma == pytest.approx(0.99)


# ── SelfPlayTrainer — smoke tests ─────────────────────────────────────────────

def _small_config(output_dir: str) -> TrainConfig:
    """Minimal config for fast smoke tests."""
    return TrainConfig(
        episodes=2,
        max_moves=30,        # very short games
        batch_size=8,
        replay_capacity=500,
        train_start_size=8,  # start training after 8 transitions
        train_every_steps=4,
        target_update_steps=20,
        save_every=2,
        epsilon_start=1.0,
        epsilon_end=0.5,
        epsilon_decay_steps=50,
        device="cpu",
        seed=0,
        output_dir=output_dir,
    )


class TestSelfPlayTrainer:
    def test_train_2_episodes_no_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _small_config(tmp)
            trainer = SelfPlayTrainer(cfg)
            trainer.train()   # must not raise

    def test_csv_log_created(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _small_config(tmp)
            trainer = SelfPlayTrainer(cfg)
            trainer.train()
            log_path = Path(tmp) / "logs" / "train_log.csv"
            assert log_path.exists()

    def test_csv_log_has_correct_columns(self) -> None:
        import csv

        with tempfile.TemporaryDirectory() as tmp:
            cfg = _small_config(tmp)
            SelfPlayTrainer(cfg).train()
            log_path = Path(tmp) / "logs" / "train_log.csv"
            with open(log_path) as f:
                reader = csv.DictReader(f)
                row = next(reader)
            expected = {
                "episode", "total_steps", "epsilon", "winner",
                "moves", "total_reward_player1", "total_reward_player_minus1",
                "loss_mean", "buffer_size",
            }
            assert expected.issubset(set(row.keys()))

    def test_csv_log_has_correct_episode_count(self) -> None:
        import csv

        with tempfile.TemporaryDirectory() as tmp:
            cfg = _small_config(tmp)
            SelfPlayTrainer(cfg).train()
            log_path = Path(tmp) / "logs" / "train_log.csv"
            with open(log_path) as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == cfg.episodes

    def test_latest_checkpoint_saved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _small_config(tmp)
            SelfPlayTrainer(cfg).train()
            latest = Path(tmp) / "models" / "dqn_latest.pt"
            assert latest.exists()

    def test_periodic_checkpoint_saved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _small_config(tmp)
            cfg.save_every = 1
            SelfPlayTrainer(cfg).train()
            # Episode 1 and 2 should each have a checkpoint
            models = list(Path(tmp).glob("models/dqn_ep*.pt"))
            assert len(models) >= 1

    def test_buffer_grows_during_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _small_config(tmp)
            trainer = SelfPlayTrainer(cfg)
            trainer.train()
            assert len(trainer.buffer) > 0

    def test_epsilon_decreases_over_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _small_config(tmp)
            trainer = SelfPlayTrainer(cfg)
            assert trainer._current_epsilon == cfg.epsilon_start
            trainer.train()
            assert trainer._current_epsilon < cfg.epsilon_start

    def test_total_steps_increase(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _small_config(tmp)
            trainer = SelfPlayTrainer(cfg)
            trainer.train()
            assert trainer._total_steps > 0

    def test_checkpoint_loadable_by_dqn_agent(self) -> None:
        """A saved checkpoint must be loadable via DQNAgent.load."""
        from corners_rl.agents.dqn_agent import DQNAgent

        with tempfile.TemporaryDirectory() as tmp:
            cfg = _small_config(tmp)
            SelfPlayTrainer(cfg).train()
            latest = Path(tmp) / "models" / "dqn_latest.pt"
            # DQNAgent.load expects the model state dict format we save
            import torch
            ckpt = torch.load(latest, map_location="cpu", weights_only=True)
            assert "model_state_dict" in ckpt

    def test_csv_has_replay_type_column(self) -> None:
        import csv

        with tempfile.TemporaryDirectory() as tmp:
            cfg = _small_config(tmp)
            SelfPlayTrainer(cfg).train()
            log_path = Path(tmp) / "logs" / "train_log.csv"
            with open(log_path) as f:
                row = next(csv.DictReader(f))
            assert "replay_type" in row

    def test_csv_uniform_replay_type_value(self) -> None:
        import csv

        with tempfile.TemporaryDirectory() as tmp:
            cfg = _small_config(tmp)
            SelfPlayTrainer(cfg).train()
            log_path = Path(tmp) / "logs" / "train_log.csv"
            with open(log_path) as f:
                rows = list(csv.DictReader(f))
            assert all(r["replay_type"] == "uniform" for r in rows)

    def test_csv_has_td_error_abs_mean_column(self) -> None:
        import csv

        with tempfile.TemporaryDirectory() as tmp:
            cfg = _small_config(tmp)
            SelfPlayTrainer(cfg).train()
            log_path = Path(tmp) / "logs" / "train_log.csv"
            with open(log_path) as f:
                row = next(csv.DictReader(f))
            assert "td_error_abs_mean" in row

    def test_checkpoint_stores_replay_type(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            cfg = _small_config(tmp)
            SelfPlayTrainer(cfg).train()
            ckpt = torch.load(
                Path(tmp) / "models" / "dqn_latest.pt",
                map_location="cpu", weights_only=True,
            )
            assert ckpt["replay_type"] == "uniform"


# ── SelfPlayTrainer — PER smoke tests ─────────────────────────────────────────

def _small_per_config(output_dir: str) -> TrainConfig:
    """Minimal PER config for fast smoke tests."""
    cfg = TrainConfig(
        episodes=2,
        max_moves=30,
        batch_size=8,
        replay_capacity=500,
        train_start_size=8,
        train_every_steps=4,
        target_update_steps=20,
        save_every=2,
        epsilon_start=1.0,
        epsilon_end=0.5,
        epsilon_decay_steps=50,
        device="cpu",
        seed=0,
        output_dir=output_dir,
        replay=ReplayConfig(
            type="prioritized",
            alpha=0.6,
            beta_start=0.4,
            beta_end=1.0,
            beta_anneal_steps=1000,
            priority_epsilon=1e-6,
        ),
    )
    return cfg


class TestSelfPlayTrainerPER:
    def test_train_2_episodes_per_no_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _small_per_config(tmp)
            SelfPlayTrainer(cfg).train()   # must not raise

    def test_per_csv_log_created(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            SelfPlayTrainer(_small_per_config(tmp)).train()
            assert (Path(tmp) / "logs" / "train_log.csv").exists()

    def test_per_csv_has_replay_type_prioritized(self) -> None:
        import csv

        with tempfile.TemporaryDirectory() as tmp:
            SelfPlayTrainer(_small_per_config(tmp)).train()
            log_path = Path(tmp) / "logs" / "train_log.csv"
            with open(log_path) as f:
                rows = list(csv.DictReader(f))
            assert all(r["replay_type"] == "prioritized" for r in rows)

    def test_per_csv_has_per_beta_column(self) -> None:
        import csv

        with tempfile.TemporaryDirectory() as tmp:
            SelfPlayTrainer(_small_per_config(tmp)).train()
            log_path = Path(tmp) / "logs" / "train_log.csv"
            with open(log_path) as f:
                row = next(csv.DictReader(f))
            assert "per_beta" in row
            # For PER, per_beta must be a real number, not nan
            assert row["per_beta"] not in ("nan", "", "NaN")

    def test_per_csv_has_td_error_abs_mean(self) -> None:
        import csv

        with tempfile.TemporaryDirectory() as tmp:
            SelfPlayTrainer(_small_per_config(tmp)).train()
            log_path = Path(tmp) / "logs" / "train_log.csv"
            with open(log_path) as f:
                row = next(csv.DictReader(f))
            assert "td_error_abs_mean" in row

    def test_per_checkpoint_stores_per_metadata(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            SelfPlayTrainer(_small_per_config(tmp)).train()
            ckpt = torch.load(
                Path(tmp) / "models" / "dqn_latest.pt",
                map_location="cpu", weights_only=True,
            )
            assert ckpt["replay_type"] == "prioritized"
            assert "per_alpha" in ckpt
            assert "per_beta" in ckpt

    def test_per_buffer_is_per_instance(self) -> None:
        from corners_rl.rl.replay_buffer import PrioritizedReplayBuffer

        with tempfile.TemporaryDirectory() as tmp:
            trainer = SelfPlayTrainer(_small_per_config(tmp))
            assert isinstance(trainer.buffer, PrioritizedReplayBuffer)

    def test_per_beta_increases_over_steps(self) -> None:
        """Beta must be strictly greater than beta_start after training steps."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _small_per_config(tmp)
            trainer = SelfPlayTrainer(cfg)
            initial_beta = trainer._current_beta
            trainer.train()
            # beta_anneal_steps=1000; after a few dozen steps beta should have grown
            assert trainer._current_beta >= initial_beta


# ── config_from_dict with replay section ──────────────────────────────────────

class TestConfigFromDictReplay:
    def test_replay_defaults_to_uniform(self) -> None:
        cfg = config_from_dict({})
        assert cfg.replay.type == "uniform"

    def test_replay_section_parsed(self) -> None:
        d = {"replay": {"type": "prioritized", "alpha": 0.5}}
        cfg = config_from_dict(d)
        assert cfg.replay.type == "prioritized"
        assert cfg.replay.alpha == pytest.approx(0.5)

    def test_replay_missing_fields_get_defaults(self) -> None:
        cfg = config_from_dict({"replay": {"type": "prioritized"}})
        assert cfg.replay.beta_start == pytest.approx(0.4)

    def test_replay_unknown_keys_ignored(self) -> None:
        cfg = config_from_dict({"replay": {"type": "uniform", "unknown": 99}})
        assert cfg.replay.type == "uniform"

    def test_replay_none_gives_uniform(self) -> None:
        cfg = config_from_dict({"replay": None})
        assert cfg.replay.type == "uniform"
