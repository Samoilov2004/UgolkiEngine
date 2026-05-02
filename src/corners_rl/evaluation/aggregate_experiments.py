"""Utilities for aggregating results from multi-seed / multi-config experiments.

Typical usage (inside the ablation script)::

    runs = [
        {"replay_type": "uniform",    "seed": 1,
         "log_path":  Path(".../.../train_log.csv"),
         "eval_path": Path(".../.../eval/summary.csv")},
        ...
    ]
    curves  = aggregate_learning_curves(runs)
    evals   = aggregate_eval_summaries(runs)
    curves.to_csv("aggregated_learning_curves.csv", index=False)
    evals.to_csv("final_eval_summary.csv",          index=False)
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def aggregate_learning_curves(runs: list[dict]) -> pd.DataFrame:
    """Concatenate train_log.csv files from multiple runs, tagging each row.

    Reads the CSV at ``run["log_path"]`` for every run dict that has a
    ``replay_type`` and ``seed`` key.  Rows from missing / unreadable files
    are silently skipped with a warning.

    Args:
        runs: List of run descriptor dicts, each with at least:
              ``"replay_type"`` (str), ``"seed"`` (int),
              ``"log_path"`` (str or Path).

    Returns:
        Single :class:`pandas.DataFrame` with all rows from all train logs plus
        ``replay_type`` and ``seed`` columns prepended.  Empty DataFrame if no
        file could be read.
    """
    frames: list[pd.DataFrame] = []
    for run in runs:
        path = Path(run["log_path"])
        if not path.exists():
            log.warning("Learning-curve log not found, skipping: %s", path)
            continue
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            log.warning("Could not read %s: %s", path, exc)
            continue
        # Always tag with the run's canonical values (overwrite if already present)
        df["replay_type"] = run["replay_type"]
        df["seed"]        = run["seed"]
        # Move tagging columns to the front for readability
        front = ["replay_type", "seed"]
        rest  = [c for c in df.columns if c not in front]
        frames.append(df[front + rest])

    if not frames:
        log.warning("aggregate_learning_curves: no data found in any run.")
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def aggregate_eval_summaries(runs: list[dict]) -> pd.DataFrame:
    """Concatenate eval/summary.csv files from multiple runs, tagging each row.

    Args:
        runs: List of run descriptor dicts, each with at least:
              ``"replay_type"`` (str), ``"seed"`` (int),
              ``"eval_path"`` (str or Path — path to the eval ``summary.csv``).

    Returns:
        Single :class:`pandas.DataFrame` with all agent-level summary rows plus
        ``replay_type`` and ``seed`` columns prepended.  Empty DataFrame if no
        file could be read.
    """
    frames: list[pd.DataFrame] = []
    for run in runs:
        path = Path(run["eval_path"])
        if not path.exists():
            log.warning("Eval summary not found, skipping: %s", path)
            continue
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            log.warning("Could not read %s: %s", path, exc)
            continue
        df["replay_type"] = run["replay_type"]
        df["seed"]        = run["seed"]
        front = ["replay_type", "seed"]
        rest  = [c for c in df.columns if c not in front]
        frames.append(df[front + rest])

    if not frames:
        log.warning("aggregate_eval_summaries: no data found in any run.")
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)
