"""DQN self-play training: config, update step, and Trainer."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from corners_rl.agents.dqn_agent import DQNAgent
from corners_rl.env.corners_env import CornersEnv
from corners_rl.rl.encoding import (
    ACTION_SPACE_SIZE,
    BOARD_SIZE,
    STATE_CHANNELS,
    action_id_to_move,
    encode_action,
    encode_state,
    inverse_transform_move_for_player,
    legal_action_mask,
    transform_move_for_player,
)
from corners_rl.rl.model import DQNModel
from corners_rl.rl.replay_buffer import ReplayBuffer
from corners_rl.rl.self_play import compute_shaped_reward
from corners_rl.utils.logging_utils import CSVLogger
from corners_rl.utils.seeding import resolve_device, seed_everything

log = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    """All hyperparameters for one training run.

    Attributes:
        episodes: Total number of self-play episodes to run.
        max_moves: Step limit per episode (draw if exceeded).
        batch_size: Mini-batch size for each gradient update.
        replay_capacity: Maximum replay buffer capacity.
        learning_rate: Adam learning rate.
        gamma: Bellman discount factor.
        target_update_steps: How often (in environment steps) to hard-copy the
            online network weights to the target network.
        train_start_size: Minimum buffer occupancy before training begins.
        train_every_steps: Gradient update every this many environment steps.
        save_every: Save a checkpoint every this many episodes.
        epsilon_start: Initial exploration probability.
        epsilon_end: Minimum exploration probability.
        epsilon_decay_steps: Number of steps over which ε decays linearly.
        device: Torch device string (``"cpu"``, ``"cuda"``, ``"mps"``,
                ``"auto"``).
        seed: Global random seed.
        output_dir: Root directory for logs and model checkpoints.
    """

    episodes: int = 200
    max_moves: int = 300
    batch_size: int = 64
    replay_capacity: int = 50_000
    learning_rate: float = 1e-4
    gamma: float = 0.99
    target_update_steps: int = 1_000
    train_start_size: int = 1_000
    train_every_steps: int = 1
    save_every: int = 50
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20_000
    device: str = "cpu"
    seed: int = 42
    output_dir: str = "outputs"


def config_from_dict(d: dict) -> TrainConfig:
    """Build a :class:`TrainConfig` from a plain dict (e.g. loaded from YAML).

    Unknown keys are silently ignored.

    Args:
        d: Dict of hyperparameter key → value pairs.

    Returns:
        Populated :class:`TrainConfig`.
    """
    fields = {f.name for f in TrainConfig.__dataclass_fields__.values()}
    filtered = {k: v for k, v in d.items() if k in fields}
    return TrainConfig(**filtered)


# ── DQN update ────────────────────────────────────────────────────────────────

def dqn_update(
    batch: dict[str, torch.Tensor],
    online_net: DQNModel,
    target_net: DQNModel,
    optimizer: torch.optim.Optimizer,
    gamma: float,
    device: torch.device,
    grad_clip: float = 10.0,
) -> float:
    """Perform one gradient step on the online network.

    Uses standard DQN targets with legal-action masking on the next state::

        q_selected = Q_online(s)[a]
        next_q_max = max_{a legal} Q_target(s')
        target     = r + γ * next_q_max * (1 − done)
        loss       = MSE(q_selected, target)

    The ``next_legal_masks`` tensor ensures that the bootstrap target only
    considers legal actions in the next state (or zero for terminal states).

    Args:
        batch: Dict of tensors as returned by :meth:`~ReplayBuffer.sample`.
        online_net: The network being trained.
        target_net: The frozen target network.
        optimizer: Optimiser for *online_net*.
        gamma: Discount factor.
        device: Target torch device.
        grad_clip: L2 gradient clipping norm.

    Returns:
        Scalar loss value.
    """
    states           = batch["states"].to(device)
    actions          = batch["actions"].to(device)
    rewards          = batch["rewards"].to(device)
    next_states      = batch["next_states"].to(device)
    dones            = batch["dones"].to(device)
    next_legal_masks = batch["next_legal_masks"].to(device)   # (B, 4096) bool

    # ── Current Q-values ─────────────────────────────────────────────────────
    online_net.train()
    q_all      = online_net(states)                              # (B, 4096)
    q_selected = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)

    # ── Target Q-values ───────────────────────────────────────────────────────
    with torch.no_grad():
        target_net.eval()
        next_q = target_net(next_states)                         # (B, 4096)

        # Mask illegal next actions with -∞ so they cannot be chosen
        next_q = next_q.masked_fill(~next_legal_masks, float("-inf"))
        next_q_max = next_q.max(dim=1).values                   # (B,)

        # Replace -∞ (terminal / all-illegal) with 0 before Bellman backup
        next_q_max = torch.where(
            torch.isinf(next_q_max),
            torch.zeros_like(next_q_max),
            next_q_max,
        )

        targets = rewards + gamma * next_q_max * (~dones).float()

    # ── Gradient update ───────────────────────────────────────────────────────
    loss = F.mse_loss(q_selected, targets)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=grad_clip)
    optimizer.step()

    return float(loss.item())


# ── Self-play Trainer ─────────────────────────────────────────────────────────

class SelfPlayTrainer:
    """Trains a single DQN model against itself.

    Both players share the same :class:`~corners_rl.rl.model.DQNModel`.
    States are encoded from the current player's canonical perspective so the
    network always sees the same board topology regardless of which side it is
    playing.

    Args:
        config: Hyperparameter configuration.
    """

    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self.device = resolve_device(config.device)
        seed_everything(config.seed)

        # ── Networks ──────────────────────────────────────────────────────────
        self.online_net = DQNModel().to(self.device)
        self.target_net = copy.deepcopy(self.online_net).to(self.device)
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(
            self.online_net.parameters(), lr=config.learning_rate
        )

        # ── Agent wrapping the online network ─────────────────────────────────
        self.agent = DQNAgent(
            model=self.online_net,
            device=str(self.device),
            epsilon=config.epsilon_start,
            seed=config.seed,
        )

        # ── Replay buffer ─────────────────────────────────────────────────────
        self.buffer = ReplayBuffer(capacity=config.replay_capacity, seed=config.seed)

        # ── Output paths ──────────────────────────────────────────────────────
        self._out = Path(config.output_dir)
        self._model_dir = self._out / "models"
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self._out / "logs" / "train_log.csv"

        # ── State ─────────────────────────────────────────────────────────────
        self._total_steps: int = 0
        self._current_epsilon: float = config.epsilon_start

    # ── Public API ────────────────────────────────────────────────────────────

    def train(self) -> None:
        """Run the full training loop.

        For each episode:

        1. Play a self-play game, collecting transitions into the buffer.
        2. After every environment step (and once the buffer is warm), perform
           a gradient update.
        3. Periodically hard-copy the online network to the target network.
        4. Log metrics to CSV.
        5. Save checkpoints.
        """
        with CSVLogger(self._log_path) as csv_log:
            for episode in range(1, self.config.episodes + 1):
                stats = self._run_episode()
                row = {
                    "episode":                episode,
                    "total_steps":            self._total_steps,
                    "epsilon":                round(self._current_epsilon, 6),
                    "winner":                 stats["winner"],
                    "moves":                  stats["moves"],
                    "total_reward_player1":   round(stats["total_reward_player1"], 4),
                    "total_reward_player_minus1": round(stats["total_reward_player_minus1"], 4),
                    "loss_mean":              round(stats["loss_mean"], 6),
                    "buffer_size":            len(self.buffer),
                }
                csv_log.log(row)

                if episode % max(self.config.save_every, 1) == 0:
                    self._save_checkpoint(f"ep{episode:06d}")
                    log.info(
                        "ep=%d steps=%d ε=%.4f winner=%s moves=%d "
                        "loss=%.4f buf=%d",
                        episode,
                        self._total_steps,
                        self._current_epsilon,
                        stats["winner"],
                        stats["moves"],
                        stats["loss_mean"],
                        len(self.buffer),
                    )

        self._save_checkpoint("latest")

    # ── Episode ───────────────────────────────────────────────────────────────

    def _run_episode(self) -> dict:
        """Play one self-play game and return episode statistics.

        Returns:
            Dict with keys: ``winner``, ``moves``, ``total_reward_player1``,
            ``total_reward_player_minus1``, ``loss_mean``.
        """
        env = CornersEnv(max_moves=self.config.max_moves)
        env.reset()

        total_rewards: dict[int, float] = {1: 0.0, -1: 0.0}
        losses: list[float] = []

        while not env.is_terminal():
            player = env.current_player
            board_before = env.board
            real_moves = env.legal_moves()

            # ── Canonical frame ───────────────────────────────────────────────
            canonical_moves = [
                transform_move_for_player(m, player) for m in real_moves
            ]
            state_arr = encode_state(board_before, player)

            # ── Select action (epsilon-greedy in canonical frame) ─────────────
            self.agent.set_epsilon(self._current_epsilon)
            real_move = self.agent.select_move(env)
            canonical_move = transform_move_for_player(real_move, player)
            action_id = encode_action(canonical_move)

            # ── Apply move ────────────────────────────────────────────────────
            board_after, _, done, info = env.step(real_move)
            winner = info["winner"]

            # ── Shaped reward (from current player's perspective) ─────────────
            shaped_r = compute_shaped_reward(
                board_before, board_after, player, real_move, done, winner
            )
            total_rewards[player] += shaped_r

            # ── Next-state encoding (from NEXT player's perspective) ──────────
            if done:
                # Terminal: next state / mask won't contribute to targets
                next_state_arr = np.zeros(
                    (STATE_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32
                )
                next_mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
            else:
                next_player = env.current_player
                next_state_arr = encode_state(board_after, next_player)
                next_canonical = [
                    transform_move_for_player(m, next_player)
                    for m in env.legal_moves()
                ]
                next_mask = legal_action_mask(next_canonical)

            # ── Store transition ──────────────────────────────────────────────
            self.buffer.push(
                state=state_arr,
                action=action_id,
                reward=shaped_r,
                next_state=next_state_arr,
                done=done,
                next_legal_mask=next_mask,
            )

            self._total_steps += 1
            self._update_epsilon()

            # ── Learn ─────────────────────────────────────────────────────────
            can_train = (
                len(self.buffer) >= self.config.train_start_size
                and self._total_steps % self.config.train_every_steps == 0
            )
            if can_train:
                loss = dqn_update(
                    self.buffer.sample(self.config.batch_size),
                    self.online_net,
                    self.target_net,
                    self.optimizer,
                    self.config.gamma,
                    self.device,
                )
                losses.append(loss)

            # ── Sync target network ───────────────────────────────────────────
            if self._total_steps % self.config.target_update_steps == 0:
                self.target_net.load_state_dict(self.online_net.state_dict())
                self.target_net.eval()

        return {
            "winner": info["winner"],
            "moves": env.move_count,
            "total_reward_player1": total_rewards[1],
            "total_reward_player_minus1": total_rewards[-1],
            "loss_mean": float(np.mean(losses)) if losses else 0.0,
        }

    # ── Epsilon schedule ──────────────────────────────────────────────────────

    def _update_epsilon(self) -> None:
        """Linear epsilon decay from ``epsilon_start`` to ``epsilon_end``."""
        cfg = self.config
        if cfg.epsilon_decay_steps <= 0:
            self._current_epsilon = cfg.epsilon_end
            return
        frac = min(self._total_steps / cfg.epsilon_decay_steps, 1.0)
        self._current_epsilon = (
            cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)
        )

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def _save_checkpoint(self, label: str) -> None:
        """Save the online network and training state.

        Args:
            label: A string appended to the filename
                   (e.g. ``"ep000050"`` or ``"latest"``).
        """
        path = self._model_dir / f"dqn_{label}.pt"
        torch.save(
            {
                "model_state_dict":  self.online_net.state_dict(),
                "optimizer_state":   self.optimizer.state_dict(),
                "total_steps":       self._total_steps,
                "epsilon":           self._current_epsilon,
            },
            path,
        )
