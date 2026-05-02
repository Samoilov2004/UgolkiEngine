"""Round-robin tournament between a collection of agents."""

from __future__ import annotations

from itertools import combinations
from typing import Optional

import pandas as pd

from corners_rl.agents.base import BaseAgent
from corners_rl.evaluation.evaluate import evaluate_match, summarize_results


def round_robin_tournament(
    agents: list[BaseAgent],
    games_per_pair: int = 100,
    max_moves: int = 300,
    seed: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a round-robin tournament where every agent plays every other agent.

    Each pair plays ``games_per_pair`` games (with symmetric side-swapping via
    :func:`~corners_rl.evaluation.evaluate.evaluate_match`).

    Args:
        agents: List of agents to compete.  Must have at least two entries.
        games_per_pair: Games to play for each unique pair.
        max_moves: Move limit per game, forwarded to :func:`evaluate_match`.
        seed: Base seed; each pair gets a derived seed to keep results
              reproducible while avoiding seed reuse across pairs.

    Returns:
        A tuple ``(detailed_results, summary_table)``:

        * **detailed_results** — :class:`pandas.DataFrame` with one row per
          game (all pairs concatenated).  Columns match :func:`evaluate_match`.
        * **summary_table** — :class:`pandas.DataFrame` with one row per agent
          containing aggregated win/draw statistics across all matches:

          ``agent``, ``games``, ``wins``, ``draws``, ``losses``,
          ``win_rate``, ``draw_rate``, ``avg_moves``.

    Raises:
        ValueError: If fewer than two agents are provided.
    """
    if len(agents) < 2:
        raise ValueError("round_robin_tournament requires at least 2 agents.")

    all_frames: list[pd.DataFrame] = []
    pair_seed_offset = 0

    for agent_a, agent_b in combinations(agents, 2):
        pair_seed = None if seed is None else seed + pair_seed_offset
        df = evaluate_match(
            agent_a,
            agent_b,
            games=games_per_pair,
            max_moves=max_moves,
            seed=pair_seed,
        )
        all_frames.append(df)
        pair_seed_offset += games_per_pair  # shift seed window for next pair

    detailed_results = (
        pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    )

    # ── Build per-agent summary ───────────────────────────────────────────────
    summary_rows: list[dict] = []
    for agent in agents:
        name = agent.name
        # All rows where this agent participated
        mask = (
            (detailed_results["agent1_name"] == name)
            | (detailed_results["agent2_name"] == name)
        )
        agent_df = detailed_results[mask].copy()

        total_games = len(agent_df)
        wins   = (agent_df["winner_agent"] == name).sum()
        draws  = agent_df["draw"].sum()
        losses = total_games - wins - draws

        summary_rows.append(
            {
                "agent": name,
                "games": total_games,
                "wins": int(wins),
                "draws": int(draws),
                "losses": int(losses),
                "win_rate": float(wins / total_games) if total_games > 0 else 0.0,
                "draw_rate": float(draws / total_games) if total_games > 0 else 0.0,
                "avg_moves": float(agent_df["moves"].mean()) if total_games > 0 else 0.0,
            }
        )

    summary_table = pd.DataFrame(summary_rows).sort_values(
        "win_rate", ascending=False
    ).reset_index(drop=True)

    return detailed_results, summary_table
