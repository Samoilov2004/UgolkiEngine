"""DQN agent with ε-greedy exploration and experience replay."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import numpy as np
from numpy.typing import NDArray

from corners_rl.agents.base import BaseAgent
from corners_rl.env.board import Board, Move
from corners_rl.models.dqn_net import DQNNetwork
from corners_rl.training.replay_buffer import ReplayBuffer, Transition


class DQNAgent(BaseAgent):
    """Double DQN agent with ε-greedy exploration.

    Architecture overview:
    * **Online network** — updated every training step.
    * **Target network** — hard-copied from online net every
      ``target_update_freq`` steps.
    * **Replay buffer** — uniform random sampling.

    The agent encodes the board as a ``(3, N, N)`` float tensor fed to a
    convolutional or MLP policy network.  Each legal move is scored by running a
    forward pass on the *resulting* board state (afterstate DQN).

    Args:
        player: The player this agent controls.
        board_size: Board side length.
        hidden_dim: Hidden layer width for the network.
        num_layers: Number of hidden layers.
        gamma: Discount factor.
        lr: Learning rate for Adam optimiser.
        epsilon_start: Initial ε for ε-greedy exploration.
        epsilon_end: Minimum ε.
        epsilon_decay_steps: Number of steps over which ε decays linearly.
        target_update_freq: Steps between target-network hard updates.
        buffer_size: Maximum replay buffer capacity.
        batch_size: Mini-batch size for gradient steps.
        double_dqn: Whether to use Double DQN target computation.
        device: Torch device string (``"cpu"``, ``"cuda"``, ``"mps"``).
    """

    def __init__(
        self,
        player: int,
        board_size: int = 8,
        hidden_dim: int = 256,
        num_layers: int = 4,
        gamma: float = 0.99,
        lr: float = 1e-4,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 200_000,
        target_update_freq: int = 1000,
        buffer_size: int = 100_000,
        batch_size: int = 256,
        double_dqn: bool = True,
        device: str = "cpu",
    ) -> None:
        super().__init__(player, name="dqn")
        self.board_size = board_size
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.double_dqn = double_dqn
        self.device = torch.device(device)

        self._steps: int = 0

        # Networks
        self.online_net = DQNNetwork(
            board_size=board_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        ).to(self.device)
        self.target_net = DQNNetwork(
            board_size=board_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        ).to(self.device)
        self._sync_target()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)

    # ------------------------------------------------------------------ #
    #  BaseAgent interface                                                 #
    # ------------------------------------------------------------------ #

    def select_action(self, board: Board, legal_moves: list[Move]) -> Move:
        """ε-greedy move selection.

        With probability ε choose a random move; otherwise choose the move
        that maximises the Q-value of the resulting afterstate.

        Args:
            board: Current board (before the move).
            legal_moves: Non-empty list of legal moves.

        Returns:
            The selected :class:`~corners_rl.env.board.Move`.
        """
        if random.random() < self._epsilon():
            return random.choice(legal_moves)
        # TODO: compute afterstate tensor for each legal move, batch-forward
        #       through online_net, return argmax move.
        raise NotImplementedError

    def observe(
        self,
        board: Board,
        move: Move,
        reward: float,
        next_board: Board,
        done: bool,
    ) -> None:
        """Store transition in replay buffer and trigger a learning step."""
        # TODO: encode board → tensor, store Transition in replay_buffer
        # TODO: if replay_buffer is large enough, call _learn()
        # TODO: if _steps % target_update_freq == 0, call _sync_target()
        # TODO: increment _steps
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    #  Learning                                                            #
    # ------------------------------------------------------------------ #

    def _learn(self) -> float:
        """Sample a mini-batch and perform one gradient step.

        Returns:
            The scalar TD loss value (for logging).
        """
        # TODO: sample batch from replay_buffer
        # TODO: compute target Q-values (Double DQN or plain DQN)
        # TODO: compute online Q-values for taken actions
        # TODO: MSE / Huber loss, backprop, optimizer.step()
        raise NotImplementedError

    def _sync_target(self) -> None:
        """Hard-copy online network weights into target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    def _epsilon(self) -> float:
        """Current ε, linearly decayed from ``epsilon_start`` to ``epsilon_end``."""
        progress = min(self._steps / max(self.epsilon_decay_steps, 1), 1.0)
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def save(self, path: Path | str) -> None:
        """Save agent state to a checkpoint file.

        Saves network weights, optimiser state, and training counters.
        """
        # TODO: torch.save({...}, path)
        raise NotImplementedError

    @classmethod
    def load(cls, path: Path | str, player: int, device: str = "cpu") -> "DQNAgent":
        """Load an agent from a checkpoint file.

        Args:
            path: Path to the checkpoint.
            player: Which player the loaded agent will control.
            device: Target device.

        Returns:
            A fully initialised :class:`DQNAgent`.
        """
        # TODO: load checkpoint, reconstruct agent, load state dicts.
        raise NotImplementedError

    def eval_mode(self) -> None:
        """Switch to evaluation mode (ε = 0, no exploration)."""
        self.epsilon_start = 0.0
        self.epsilon_end = 0.0
        self.online_net.eval()
