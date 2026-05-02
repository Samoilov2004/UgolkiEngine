"""Main training loop: self-play episodes, logging, checkpointing."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from corners_rl.agents.dqn_agent import DQNAgent
from corners_rl.env.game import Game
from corners_rl.evaluation.evaluator import Evaluator
from corners_rl.training.self_play import SelfPlayManager
from corners_rl.visualization.plots import TrainingLogger

logger = logging.getLogger(__name__)


class Trainer:
    """Orchestrates DQN training with self-play.

    Responsibilities:
    * Run self-play episodes, collecting transitions for the agent.
    * Periodically evaluate against baseline agents.
    * Save checkpoints and update the self-play opponent pool.
    * Log metrics to :class:`~corners_rl.visualization.plots.TrainingLogger`.

    Args:
        agent: The DQN agent to train (controls Player 1).
        game_cfg: Dict with board/game configuration (board_size, zone_size, …).
        training_cfg: Dict with training configuration (total_episodes, …).
        self_play_cfg: Dict with self-play configuration (pool_size, …).
        eval_cfg: Dict with evaluation configuration.
        checkpoint_dir: Directory to save model checkpoints.
        device: Torch device string.
    """

    def __init__(
        self,
        agent: DQNAgent,
        game_cfg: dict,
        training_cfg: dict,
        self_play_cfg: dict,
        eval_cfg: dict,
        checkpoint_dir: Path = Path("checkpoints"),
        device: str = "cpu",
    ) -> None:
        self.agent = agent
        self.game_cfg = game_cfg
        self.training_cfg = training_cfg
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.self_play_manager = SelfPlayManager(
            agent=agent,
            pool_size=self_play_cfg.get("pool_size", 10),
            swap_interval=self_play_cfg.get("swap_interval", 500),
            win_rate_threshold=self_play_cfg.get("win_rate_threshold", 0.55),
        )
        self.evaluator = Evaluator(
            board_size=game_cfg.get("board_size", 8),
            zone_size=game_cfg.get("zone_size", 3),
            episodes_per_opponent=eval_cfg.get("episodes_per_opponent", 100),
            opponent_names=eval_cfg.get("opponents", ["random", "greedy"]),
        )
        self.training_logger = TrainingLogger()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def train(self) -> None:
        """Run the full training loop.

        Episode loop:
        1. Reset game and agents.
        2. Play one episode to completion (both players alternate).
        3. Feed transitions to :meth:`~DQNAgent.observe`.
        4. Every ``checkpoint_interval`` episodes: save checkpoint.
        5. Every ``eval_interval`` episodes: run evaluation.
        6. Every ``swap_interval`` episodes: swap self-play opponent.
        """
        total_episodes = self.training_cfg.get("total_episodes", 50_000)
        checkpoint_interval = self.training_cfg.get("checkpoint_interval", 1_000)
        log_interval = self.training_cfg.get("log_interval", 100)
        eval_interval = self.training_cfg.get("evaluation", {}).get("interval", 1_000)

        for episode in range(1, total_episodes + 1):
            result = self._run_episode()
            self.training_logger.record(episode, result)

            if episode % log_interval == 0:
                self._log_progress(episode)

            if episode % checkpoint_interval == 0:
                self._save_checkpoint(episode)

            if episode % eval_interval == 0:
                self._evaluate(episode)

            self.self_play_manager.maybe_swap(episode)

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _run_episode(self) -> dict:
        """Play a single self-play episode and return episode statistics.

        Returns:
            Dict with keys: ``result``, ``steps``, ``loss`` (mean TD loss).
        """
        # TODO: create Game instance, reset agents
        # TODO: alternate turns: agent.select_action → game.step → agent.observe
        # TODO: let opponent (from self_play_manager.current_opponent) play P2 moves
        # TODO: collect and return episode stats
        raise NotImplementedError

    def _save_checkpoint(self, episode: int) -> None:
        """Save the agent's current state to disk."""
        path = self.checkpoint_dir / f"dqn_ep{episode:06d}.pt"
        self.agent.save(path)
        logger.info("Checkpoint saved: %s", path)

    def _evaluate(self, episode: int) -> None:
        """Run evaluation and log results."""
        metrics = self.evaluator.evaluate(self.agent)
        self.training_logger.record_eval(episode, metrics)
        logger.info("Eval @ ep %d: %s", episode, metrics)

    def _log_progress(self, episode: int) -> None:
        """Print a one-line progress summary."""
        # TODO: pull recent stats from training_logger and print
        raise NotImplementedError
