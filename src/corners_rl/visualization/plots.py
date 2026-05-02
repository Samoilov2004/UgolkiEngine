"""Training-curve logging and plot generation."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Optional

import pandas as pd


class TrainingLogger:
    """Accumulates per-episode training metrics and evaluation results.

    Designed to be queried at the end of training for plot generation or
    exported as a CSV.

    Attributes:
        episode_data: List of dicts, one per episode.
        eval_data: List of dicts, one per evaluation round.
    """

    def __init__(self) -> None:
        self.episode_data: list[dict] = []
        self.eval_data: list[dict] = []

    def record(self, episode: int, stats: dict) -> None:
        """Store per-episode statistics.

        Args:
            episode: Episode number.
            stats: Dict with keys such as ``result``, ``steps``, ``loss``, etc.
        """
        self.episode_data.append({"episode": episode, **stats})

    def record_eval(self, episode: int, metrics_list: list) -> None:
        """Store evaluation results.

        Args:
            episode: Episode at which evaluation was performed.
            metrics_list: List of :class:`~corners_rl.evaluation.evaluator.EvalMetrics`.
        """
        for m in metrics_list:
            self.eval_data.append(
                {
                    "episode": episode,
                    "opponent": m.opponent_name,
                    "win_rate": m.win_rate,
                    "avg_steps": m.avg_steps,
                }
            )

    def to_dataframe(self) -> pd.DataFrame:
        """Return all episode data as a pandas DataFrame."""
        return pd.DataFrame(self.episode_data)

    def to_eval_dataframe(self) -> pd.DataFrame:
        """Return all evaluation data as a pandas DataFrame."""
        return pd.DataFrame(self.eval_data)

    def save_csv(self, path: Path | str) -> None:
        """Save episode data to a CSV file."""
        self.to_dataframe().to_csv(path, index=False)


def plot_training_curves(
    logger: TrainingLogger,
    save_dir: Path | str = Path("plots"),
    smoothing_window: int = 100,
) -> None:
    """Generate and save standard training plots.

    Plots produced:
    * Win-rate over episodes (vs each evaluation opponent).
    * Episode length over time (smoothed).
    * TD loss over training steps (smoothed).

    Args:
        logger: Populated :class:`TrainingLogger` instance.
        save_dir: Directory to save PNG files.
        smoothing_window: Rolling-average window for smoothing curves.
    """
    # TODO: use seaborn / matplotlib to produce and save the plots.
    raise NotImplementedError
