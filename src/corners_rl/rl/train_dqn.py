"""DQN self-play training: config, update step, and Trainer.

Supports two replay strategies controlled by the ``replay`` section in
the YAML config (or the ``TrainConfig.replay`` field):

* **Uniform** (default) — standard random sampling; loss = SmoothL1.
* **Prioritized** (PER) — sampling proportional to TD error; loss is
  IS-weight-corrected SmoothL1; priorities are updated after each
  gradient step.

References
----------
Schaul et al., "Prioritized Experience Replay", ICLR 2016.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from corners_rl.agents.dqn_agent import DQNAgent
from corners_rl.env.corners_env import CornersEnv
from corners_rl.env.moves import filter_forward_moves
from corners_rl.rl.encoding import (
    ACTION_SPACE_SIZE,
    BOARD_SIZE,
    STATE_CHANNELS,
    encode_action,
    encode_state,
    legal_action_mask,
    transform_move_for_player,
)
from corners_rl.rl.model import DQNModel
from corners_rl.rl.replay_buffer import PrioritizedReplayBuffer, ReplayBuffer
from corners_rl.rl.self_play import compute_shaped_reward
from corners_rl.utils.logging_utils import CSVLogger
from corners_rl.utils.seeding import resolve_device, seed_everything

log = logging.getLogger(__name__)

# Type alias for either buffer variant
_AnyBuffer = Union[ReplayBuffer, PrioritizedReplayBuffer]


# ── Replay configuration ──────────────────────────────────────────────────────

@dataclass
class ReplayConfig:
    """Configuration for the experience-replay strategy.

    Attributes:
        type: ``"uniform"`` or ``"prioritized"``.
        alpha: PER prioritisation exponent (0 = uniform, 1 = full priority).
        beta_start: Initial IS-correction exponent (annealed to ``beta_end``).
        beta_end: Final IS-correction exponent (usually 1.0).
        beta_anneal_steps: Environment steps over which beta is annealed.
        priority_epsilon: Floor added to every |TD error| before raising to
                          ``alpha``, ensuring no transition has zero priority.
    """

    type:               str   = "uniform"
    alpha:              float = 0.6
    beta_start:         float = 0.4
    beta_end:           float = 1.0
    beta_anneal_steps:  int   = 500_000
    priority_epsilon:   float = 1e-6


# ── Training configuration ────────────────────────────────────────────────────

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
        init_checkpoint: Optional path to an imitation-learning ``.pt`` file
                         for warm-start (produced by pretrain_imitation.py).
        replay: Replay-buffer strategy configuration.
    """

    episodes:             int   = 200
    max_moves:            int   = 300
    batch_size:           int   = 64
    replay_capacity:      int   = 50_000
    learning_rate:        float = 1e-4
    gamma:                float = 0.99
    target_update_steps:  int   = 1_000
    train_start_size:     int   = 1_000
    train_every_steps:    int   = 1
    save_every:           int   = 50
    epsilon_start:        float = 1.0
    epsilon_end:          float = 0.05
    epsilon_decay_steps:  int   = 20_000
    device:               str   = "cpu"
    seed:                 int   = 42
    output_dir:           str   = "outputs"
    init_checkpoint:      Optional[str] = None
    replay:               ReplayConfig = field(default_factory=ReplayConfig)
    forward_only:         bool = False


def config_from_dict(d: dict) -> TrainConfig:
    """Build a :class:`TrainConfig` from a plain dict (e.g. loaded from YAML).

    Handles the optional nested ``replay`` sub-dict.  Unknown top-level and
    nested keys are silently ignored.

    Args:
        d: Dict of hyperparameter key → value pairs.

    Returns:
        Populated :class:`TrainConfig`.
    """
    top_fields = {
        f.name for f in TrainConfig.__dataclass_fields__.values()
        if f.name != "replay"
    }
    filtered = {k: v for k, v in d.items() if k in top_fields}
    cfg = TrainConfig(**filtered)

    if isinstance(d.get("replay"), dict):
        replay_fields = {f.name for f in ReplayConfig.__dataclass_fields__.values()}
        r = {k: v for k, v in d["replay"].items() if k in replay_fields}
        cfg.replay = ReplayConfig(**r)

    return cfg


# ── DQN update ────────────────────────────────────────────────────────────────

def dqn_update(
    batch: dict[str, torch.Tensor],
    online_net: DQNModel,
    target_net: DQNModel,
    optimizer: torch.optim.Optimizer,
    gamma: float,
    device: torch.device,
    grad_clip: float = 10.0,
) -> tuple[float, np.ndarray]:
    """Perform one gradient step on the online network.

    Computes DQN targets with legal-action masking::

        q_selected  = Q_online(s)[a]
        next_q_max  = max_{a legal} Q_target(s')
        target      = r + γ * next_q_max * (1 − done)
        td_error    = target − q_selected
        per_loss    = SmoothL1(q_selected, target)   [no reduction]
        loss        = mean(IS_weights * per_loss)

    When the batch contains no ``"weights"`` key (uniform replay), IS weights
    default to all-ones, recovering standard DQN.

    Args:
        batch: Transition dict as returned by a replay buffer ``sample``.
               May optionally contain ``"weights"`` (IS weights from PER) and
               ``"indices"`` (buffer indices — handled externally by the
               caller for priority updates).
        online_net: The network being trained.
        target_net: The frozen target network.
        optimizer: Optimiser for *online_net*.
        gamma: Discount factor.
        device: Target torch device.
        grad_clip: L2 gradient clipping norm.

    Returns:
        Tuple ``(loss_scalar, td_errors_numpy)`` where ``td_errors_numpy``
        is a 1-D float32 array of per-sample TD errors ``(target − q)``.
    """
    states           = batch["states"].to(device)
    actions          = batch["actions"].to(device)
    rewards          = batch["rewards"].to(device)
    next_states      = batch["next_states"].to(device)
    dones            = batch["dones"].to(device)
    next_legal_masks = batch["next_legal_masks"].to(device)   # (B, 4096) bool

    # IS weights: all-ones for uniform, provided by PER buffer
    if "weights" in batch:
        is_weights = batch["weights"].to(device)              # (B,) float32
    else:
        is_weights = torch.ones(states.shape[0], device=device)

    # ── Current Q-values ─────────────────────────────────────────────────────
    online_net.train()
    q_all      = online_net(states)                               # (B, 4096)
    q_selected = q_all.gather(1, actions.unsqueeze(1)).squeeze(1) # (B,)

    # ── Target Q-values (no gradient) ────────────────────────────────────────
    with torch.no_grad():
        target_net.eval()
        next_q = target_net(next_states)                         # (B, 4096)
        next_q = next_q.masked_fill(~next_legal_masks, float("-inf"))
        next_q_max = next_q.max(dim=1).values                   # (B,)
        next_q_max = torch.where(
            torch.isinf(next_q_max),
            torch.zeros_like(next_q_max),
            next_q_max,
        )
        # ── Negamax bootstrapping (zero-sum self-play) ────────────────────────
        # next_state is encoded from the OPPONENT's perspective because after
        # player P moves, it becomes the opponent's turn.  The Q-network applied
        # to next_state therefore gives the opponent's best expected value.
        # In a zero-sum game: V(s for P) = −V(s for opponent), so we SUBTRACT
        # the opponent's bootstrap value rather than adding it.
        targets = rewards - gamma * next_q_max * (~dones).float()

    # ── TD errors (used for PER priority updates) ─────────────────────────────
    td_errors = (targets - q_selected).detach().cpu().numpy().astype(np.float32)

    # ── IS-weighted SmoothL1 loss ─────────────────────────────────────────────
    per_sample_loss = F.smooth_l1_loss(q_selected, targets, reduction="none")  # (B,)
    loss = (is_weights * per_sample_loss).mean()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=grad_clip)
    optimizer.step()

    return float(loss.item()), td_errors


# ── Self-play Trainer ─────────────────────────────────────────────────────────

class SelfPlayTrainer:
    """Trains a single DQN model against itself.

    Both players share the same :class:`~corners_rl.rl.model.DQNModel`.
    States are encoded from the current player's canonical perspective so the
    network always sees the same board topology regardless of which side it is
    playing.

    Supports both **Uniform** and **Prioritized** (PER) replay buffers,
    controlled by ``config.replay.type``.

    Args:
        config: Hyperparameter configuration.
    """

    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self.device = resolve_device(config.device)
        seed_everything(config.seed)

        self._is_per: bool = config.replay.type == "prioritized"

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
            forward_only=config.forward_only,
        )

        # ── Load imitation pre-training checkpoint (warm start) ───────────────
        if config.init_checkpoint:
            ckpt_path = Path(config.init_checkpoint)
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location=self.device)
                self.online_net.load_state_dict(ckpt["model_state_dict"])
                self.target_net.load_state_dict(ckpt["model_state_dict"])
                log.info("Loaded imitation checkpoint from %s", ckpt_path)
            else:
                log.warning(
                    "init_checkpoint not found: %s — starting from scratch.",
                    ckpt_path,
                )

        # ── Replay buffer ─────────────────────────────────────────────────────
        if self._is_per:
            self.buffer: _AnyBuffer = PrioritizedReplayBuffer(
                capacity=config.replay_capacity,
                alpha=config.replay.alpha,
                priority_epsilon=config.replay.priority_epsilon,
                seed=config.seed,
            )
            log.info(
                "Using PrioritizedReplayBuffer (α=%.2f, β %.2f→%.2f over %d steps)",
                config.replay.alpha,
                config.replay.beta_start,
                config.replay.beta_end,
                config.replay.beta_anneal_steps,
            )
        else:
            self.buffer = ReplayBuffer(
                capacity=config.replay_capacity, seed=config.seed
            )
            log.info("Using uniform ReplayBuffer")

        # ── Output paths ──────────────────────────────────────────────────────
        self._out = Path(config.output_dir)
        self._model_dir = self._out / "models"
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self._out / "logs" / "train_log.csv"

        # ── State ─────────────────────────────────────────────────────────────
        self._total_steps:   int   = 0
        self._current_epsilon: float = config.epsilon_start
        self._current_beta:  float = config.replay.beta_start

    # ── Public API ────────────────────────────────────────────────────────────

    def train(self) -> None:
        """Run the full training loop.

        For each episode:

        1. Play a self-play game, collecting transitions into the buffer.
        2. After every environment step (and once the buffer is warm), perform
           a gradient update (with priority updates for PER).
        3. Periodically hard-copy the online network to the target network.
        4. Log metrics to CSV (including PER-specific columns).
        5. Save checkpoints.
        """
        with CSVLogger(self._log_path) as csv_log:
            for episode in range(1, self.config.episodes + 1):
                stats = self._run_episode(episode)

                # ── Priority / PER stats ──────────────────────────────────────
                if self._is_per:
                    pstats = self.buffer.stats()     # type: ignore[union-attr]
                    p_mean = round(pstats["priority_mean"], 6)
                    p_max  = round(pstats["priority_max"],  6)
                    p_std  = round(pstats["priority_std"],  6)
                    per_alpha = self.config.replay.alpha
                    per_beta  = round(self._current_beta, 6)
                else:
                    p_mean = p_max = p_std = float("nan")
                    per_alpha = float("nan")
                    per_beta  = float("nan")

                row = {
                    "episode":                       episode,
                    "total_steps":                   self._total_steps,
                    "epsilon":                       round(self._current_epsilon, 6),
                    "winner":                        stats["winner"],
                    "moves":                         stats["moves"],
                    "total_reward_player1":          round(stats["total_reward_player1"], 4),
                    "total_reward_player_minus1":    round(stats["total_reward_player_minus1"], 4),
                    "loss_mean":                     round(stats["loss_mean"], 6),
                    "buffer_size":                   len(self.buffer),
                    # ── Replay metadata ───────────────────────────────────────
                    "replay_type":                   self.config.replay.type,
                    "per_alpha":                     per_alpha,
                    "per_beta":                      per_beta,
                    "priority_mean":                 p_mean,
                    "priority_max":                  p_max,
                    "priority_std":                  p_std,
                    # ── TD-error statistics ───────────────────────────────────
                    "td_error_mean":                 round(stats["td_error_mean"], 6),
                    "td_error_abs_mean":             round(stats["td_error_abs_mean"], 6),
                    "td_error_std":                  round(stats["td_error_std"], 6),
                }
                csv_log.log(row)

                if episode % max(self.config.save_every, 1) == 0:
                    self._save_checkpoint(f"ep{episode:06d}", episode)
                    log.info(
                        "ep=%d steps=%d ε=%.4f β=%.4f winner=%s moves=%d "
                        "loss=%.4f |δ|=%.4f buf=%d",
                        episode,
                        self._total_steps,
                        self._current_epsilon,
                        self._current_beta if self._is_per else 0.0,
                        stats["winner"],
                        stats["moves"],
                        stats["loss_mean"],
                        stats["td_error_abs_mean"],
                        len(self.buffer),
                    )

        self._save_checkpoint("latest", self.config.episodes)

    # ── Episode ───────────────────────────────────────────────────────────────

    def _run_episode(self, episode: int) -> dict:
        """Play one self-play game and return episode statistics.

        Returns:
            Dict with keys: ``winner``, ``moves``, ``total_reward_player1``,
            ``total_reward_player_minus1``, ``loss_mean``,
            ``td_error_mean``, ``td_error_abs_mean``, ``td_error_std``.
        """
        env = CornersEnv(max_moves=self.config.max_moves)
        env.reset()

        total_rewards: dict[int, float] = {1: 0.0, -1: 0.0}
        losses:        list[float]      = []
        td_errors_ep:  list[np.ndarray] = []

        while not env.is_terminal():
            player       = env.current_player
            board_before = env.board
            all_moves    = env.legal_moves()
            real_moves   = (
                filter_forward_moves(all_moves, player)
                if self.config.forward_only
                else all_moves
            )

            # ── Canonical frame ───────────────────────────────────────────────
            canonical_moves = [
                transform_move_for_player(m, player) for m in real_moves
            ]
            state_arr = encode_state(board_before, player)

            # ── Select action (epsilon-greedy) ────────────────────────────────
            self.agent.set_epsilon(self._current_epsilon)
            real_move      = self.agent.select_move(env)
            canonical_move = transform_move_for_player(real_move, player)
            action_id      = encode_action(canonical_move)

            # ── Apply move ────────────────────────────────────────────────────
            board_after, _, done, info = env.step(real_move)

            # ── Shaped reward ─────────────────────────────────────────────────
            shaped_r = compute_shaped_reward(
                board_before, board_after, player, real_move, done, info["winner"]
            )
            total_rewards[player] += shaped_r

            # ── Next-state encoding ───────────────────────────────────────────
            if done:
                next_state_arr = np.zeros(
                    (STATE_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32
                )
                next_mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
            else:
                next_player = env.current_player
                next_state_arr = encode_state(board_after, next_player)
                next_raw = env.legal_moves()
                next_filtered = (
                    filter_forward_moves(next_raw, next_player)
                    if self.config.forward_only
                    else next_raw
                )
                next_canonical = [
                    transform_move_for_player(m, next_player)
                    for m in next_filtered
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
            if self._is_per:
                self._update_beta()

            # ── Learn ─────────────────────────────────────────────────────────
            can_train = (
                len(self.buffer) >= self.config.train_start_size
                and self._total_steps % self.config.train_every_steps == 0
            )
            if can_train:
                if self._is_per:
                    batch = self.buffer.sample(          # type: ignore[union-attr]
                        self.config.batch_size, beta=self._current_beta
                    )
                else:
                    batch = self.buffer.sample(self.config.batch_size)

                loss, td_errs = dqn_update(
                    batch,
                    self.online_net,
                    self.target_net,
                    self.optimizer,
                    self.config.gamma,
                    self.device,
                )
                losses.append(loss)
                td_errors_ep.append(td_errs)

                if self._is_per:
                    self.buffer.update_priorities(          # type: ignore[union-attr]
                        batch["indices"], td_errs
                    )

            # ── Sync target network ───────────────────────────────────────────
            if self._total_steps % self.config.target_update_steps == 0:
                self.target_net.load_state_dict(self.online_net.state_dict())
                self.target_net.eval()

        # ── Aggregate TD-error stats ──────────────────────────────────────────
        if td_errors_ep:
            all_td = np.concatenate(td_errors_ep)
            td_error_mean     = float(np.mean(all_td))
            td_error_abs_mean = float(np.mean(np.abs(all_td)))
            td_error_std      = float(np.std(all_td))
        else:
            td_error_mean = td_error_abs_mean = td_error_std = float("nan")

        return {
            "winner":                   info["winner"],
            "moves":                    env.move_count,
            "total_reward_player1":     total_rewards[1],
            "total_reward_player_minus1": total_rewards[-1],
            "loss_mean":                float(np.mean(losses)) if losses else 0.0,
            "td_error_mean":            td_error_mean,
            "td_error_abs_mean":        td_error_abs_mean,
            "td_error_std":             td_error_std,
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

    # ── Beta schedule (PER) ───────────────────────────────────────────────────

    def _update_beta(self) -> None:
        """Linear beta annealing from ``beta_start`` to ``beta_end``."""
        rc = self.config.replay
        if rc.beta_anneal_steps <= 0:
            self._current_beta = rc.beta_end
            return
        frac = min(self._total_steps / rc.beta_anneal_steps, 1.0)
        self._current_beta = rc.beta_start + frac * (rc.beta_end - rc.beta_start)

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def _save_checkpoint(self, label: str, episode: int = 0) -> None:
        """Save the online network and training state.

        Args:
            label: String appended to the filename
                   (e.g. ``"ep000050"`` or ``"latest"``).
            episode: Current episode number (stored in the checkpoint).
        """
        path = self._model_dir / f"dqn_{label}.pt"
        payload = {
            "model_state_dict": self.online_net.state_dict(),
            "optimizer_state":  self.optimizer.state_dict(),
            "total_steps":      self._total_steps,
            "epsilon":          self._current_epsilon,
            "episode":          episode,
            # Replay metadata
            "replay_type":      self.config.replay.type,
            "per_alpha":        self.config.replay.alpha,
            "per_beta":         self._current_beta,
            "per_beta_start":   self.config.replay.beta_start,
            "per_beta_end":     self.config.replay.beta_end,
            "per_beta_anneal_steps": self.config.replay.beta_anneal_steps,
            "priority_epsilon": self.config.replay.priority_epsilon,
        }
        torch.save(payload, path)
