"""Imitation learning (behavioural cloning) pre-training for the DQN model.

The expert agent (typically :class:`~corners_rl.agents.heuristic_agent.HeuristicAgent`)
plays games against a configurable opponent.  Every (state, action) pair observed
is stored as a supervised training example.  The DQN model is then trained to
predict the expert's actions via cross-entropy loss, giving it a warm start
before self-play fine-tuning.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch.utils.data import DataLoader, TensorDataset

from corners_rl.agents.base import BaseAgent
from corners_rl.env.corners_env import CornersEnv
from corners_rl.rl.encoding import (
    ACTION_SPACE_SIZE,
    encode_action,
    encode_state,
    legal_action_mask,
    transform_move_for_player,
)
from corners_rl.rl.model import DQNModel
from corners_rl.utils.logging_utils import CSVLogger
from corners_rl.utils.seeding import resolve_device, seed_everything

log = logging.getLogger(__name__)


# ── Dataset generation ────────────────────────────────────────────────────────

@dataclass
class ImitationSample:
    """One supervised training example from an expert game."""

    state:        NDArray[np.float32]   # (STATE_CHANNELS, 8, 8)
    action_id:    int                   # expert action in canonical frame
    legal_mask:   NDArray[np.bool_]     # (ACTION_SPACE_SIZE,) bool


@dataclass
class ImitationDataset:
    """Collection of imitation samples with conversion utilities."""

    samples: list[ImitationSample] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.samples)

    def add(self, sample: ImitationSample) -> None:
        self.samples.append(sample)

    def to_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(states, actions, masks)`` as float/long/bool tensors."""
        states  = torch.from_numpy(
            np.stack([s.state for s in self.samples], axis=0)
        )                                              # (N, C, 8, 8)  float32
        actions = torch.tensor(
            [s.action_id for s in self.samples], dtype=torch.long
        )                                              # (N,)           int64
        masks   = torch.from_numpy(
            np.stack([s.legal_mask for s in self.samples], axis=0)
        )                                              # (N, 4096)      bool
        return states, actions, masks


def generate_imitation_dataset(
    expert_agent: BaseAgent,
    games: int = 500,
    max_moves: int = 300,
    seed: int = 42,
    opponent: str = "random",
) -> ImitationDataset:
    """Play *games* games with *expert_agent* and record every (state, action) pair.

    The expert always plays as the current player; states are encoded from the
    current player's canonical perspective (same convention as DQN training).

    Args:
        expert_agent: The agent whose policy is being imitated (e.g.
                      :class:`~corners_rl.agents.heuristic_agent.HeuristicAgent`).
        games: Number of games to play.
        max_moves: Step limit per game.
        seed: Base random seed.
        opponent: Opponent type — ``"random"``, ``"greedy"``, or
                  ``"self"`` (expert plays both sides).

    Returns:
        :class:`ImitationDataset` populated with one sample per expert move.
    """
    from corners_rl.agents.greedy_agent import GreedyAgent
    from corners_rl.agents.random_agent import RandomAgent

    seed_everything(seed)
    rng = random.Random(seed)

    dataset = ImitationDataset()
    total_samples = 0

    for game_idx in range(games):
        game_seed = rng.randint(0, 2**32 - 1)

        # Build opponent for this game
        if opponent == "self":
            opp = expert_agent          # same object, same weights — both sides
        elif opponent == "greedy":
            opp = GreedyAgent(seed=game_seed)
        else:
            opp = RandomAgent(seed=game_seed)

        # Seed agent RNGs
        for ag in {expert_agent, opp}:
            if hasattr(ag, "_rng") and isinstance(ag._rng, random.Random):
                ag._rng = random.Random(rng.randint(0, 2**32 - 1))

        env = CornersEnv(max_moves=max_moves)
        env.reset()

        while not env.is_terminal():
            player    = env.current_player
            board     = env.board
            real_moves = env.legal_moves()

            if not real_moves:
                break

            # Expert always selects the move
            expert_move = expert_agent.select_move(env)

            # Encode in canonical frame
            canonical_moves = [
                transform_move_for_player(m, player) for m in real_moves
            ]
            canonical_expert = transform_move_for_player(expert_move, player)
            action_id = encode_action(canonical_expert)

            state_arr = encode_state(board, player)
            mask      = legal_action_mask(canonical_moves)

            dataset.add(ImitationSample(
                state=state_arr,
                action_id=action_id,
                legal_mask=mask,
            ))
            total_samples += 1

            # Advance environment (both sides use expert for data collection;
            # the opponent only matters when expert plays one side)
            env.step(expert_move)

    log.info(
        "Dataset generated: %d games → %d samples", games, total_samples
    )
    return dataset


# ── Training ──────────────────────────────────────────────────────────────────

@dataclass
class ImitationConfig:
    """Hyperparameters for imitation-learning pre-training."""

    epochs:        int   = 5
    batch_size:    int   = 128
    learning_rate: float = 1e-3
    val_fraction:  float = 0.1
    grad_clip:     float = 5.0
    device:        str   = "cpu"
    seed:          int   = 42
    log_path:      str   = "outputs/logs/imitation_log.csv"
    out_path:      str   = "outputs/models/imitation.pt"


def train_imitation(
    model: DQNModel,
    dataset: ImitationDataset,
    config: ImitationConfig,
) -> None:
    """Train *model* via supervised cross-entropy on *dataset*.

    Loss is masked-softmax cross-entropy: illegal actions are set to ``-∞``
    before computing the softmax, so the model is only penalised over the
    legal action distribution.

    Args:
        model: The :class:`~corners_rl.rl.model.DQNModel` to train in-place.
        dataset: Expert demonstrations from :func:`generate_imitation_dataset`.
        config: Training hyperparameters.
    """
    seed_everything(config.seed)
    device = resolve_device(config.device)
    model = model.to(device)

    states, actions, masks = dataset.to_tensors()
    n = len(states)
    n_val  = max(1, int(n * config.val_fraction))
    n_train = n - n_val

    # Shuffle once and split
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(config.seed))
    tr_idx = perm[:n_train]
    val_idx = perm[n_train:]

    train_ds = TensorDataset(states[tr_idx], actions[tr_idx], masks[tr_idx])
    val_ds   = TensorDataset(states[val_idx], actions[val_idx], masks[val_idx])

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=config.batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    log_path = Path(config.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(
        "Imitation training: %d train / %d val samples, %d epochs",
        n_train, n_val, config.epochs,
    )

    with CSVLogger(log_path) as csv_log:
        for epoch in range(1, config.epochs + 1):
            # ── Train ─────────────────────────────────────────────────────────
            model.train()
            train_loss_sum = 0.0
            train_steps    = 0

            for s_batch, a_batch, m_batch in train_loader:
                s_batch = s_batch.to(device)
                a_batch = a_batch.to(device)
                m_batch = m_batch.to(device)

                logits = model(s_batch)                          # (B, 4096)
                # Mask illegal actions to -inf before cross-entropy
                logits_masked = logits.masked_fill(~m_batch, float("-inf"))
                loss = F.cross_entropy(logits_masked, a_batch)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=config.grad_clip
                )
                optimizer.step()

                train_loss_sum += float(loss.item())
                train_steps    += 1

            train_loss = train_loss_sum / max(train_steps, 1)

            # ── Validate ──────────────────────────────────────────────────────
            model.eval()
            val_loss_sum = 0.0
            val_correct  = 0
            val_total    = 0

            with torch.no_grad():
                for s_batch, a_batch, m_batch in val_loader:
                    s_batch = s_batch.to(device)
                    a_batch = a_batch.to(device)
                    m_batch = m_batch.to(device)

                    logits = model(s_batch)
                    logits_masked = logits.masked_fill(~m_batch, float("-inf"))
                    loss = F.cross_entropy(logits_masked, a_batch)
                    val_loss_sum += float(loss.item())

                    preds = logits_masked.argmax(dim=1)
                    val_correct += (preds == a_batch).sum().item()
                    val_total   += a_batch.size(0)

            val_loss = val_loss_sum / max(len(val_loader), 1)
            val_acc  = val_correct / max(val_total, 1)

            log.info(
                "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_acc=%.2f%%",
                epoch, config.epochs, train_loss, val_loss, val_acc * 100,
            )
            csv_log.log({
                "epoch":      epoch,
                "train_loss": round(train_loss, 6),
                "val_loss":   round(val_loss,   6),
                "val_acc":    round(val_acc,     6),
            })

    # ── Save checkpoint ───────────────────────────────────────────────────────
    out = Path(config.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epsilon":          0.0,   # greedy by default after pretraining
            "name":             "dqn_imitation",
        },
        out,
    )
    log.info("Imitation checkpoint saved to %s", out)
