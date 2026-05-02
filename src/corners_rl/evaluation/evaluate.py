"""Head-to-head match evaluation and result summarisation."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from corners_rl.agents.base import BaseAgent
from corners_rl.game_runner import play_game


def evaluate_match(
    agent1: BaseAgent,
    agent2: BaseAgent,
    games: int = 100,
    max_moves: int = 300,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Play *games* games between *agent1* and *agent2* with side-swapping.

    The first half of games agent1 plays as Player 1 and agent2 as Player -1.
    The second half the sides are swapped so that any board-side advantage
    is balanced out.

    Args:
        agent1: First agent.
        agent2: Second agent.
        games: Total number of games to play.  Rounded up to the nearest even
               number so both halves are equal.
        max_moves: Step limit per game.
        seed: Base random seed; each game gets a derived seed for reproducibility.

    Returns:
        :class:`pandas.DataFrame` with one row per game and columns:

        ``game_id``, ``agent1_name``, ``agent2_name``,
        ``player1_agent``, ``player_minus1_agent``,
        ``winner``, ``winner_agent``, ``draw``, ``moves``.
    """
    # Ensure an even number of games for symmetric side assignment
    if games % 2 != 0:
        games += 1

    half = games // 2
    rows: list[dict] = []

    for i in range(games):
        game_id = i + 1
        game_seed = None if seed is None else seed + i

        # First half: agent1=P1, agent2=P-1;  second half: swap
        if i < half:
            p1_agent, pm1_agent = agent1, agent2
        else:
            p1_agent, pm1_agent = agent2, agent1

        result = play_game(p1_agent, pm1_agent, max_moves=max_moves, seed=game_seed)
        winner_val = result["winner"]  # 1, -1, or None

        # Translate winner player-id → agent name
        if winner_val is None:
            winner_agent = None
        elif winner_val == 1:
            winner_agent = p1_agent.name
        else:
            winner_agent = pm1_agent.name

        rows.append(
            {
                "game_id": game_id,
                "agent1_name": agent1.name,
                "agent2_name": agent2.name,
                "player1_agent": p1_agent.name,
                "player_minus1_agent": pm1_agent.name,
                "winner": winner_val,
                "winner_agent": winner_agent,
                "draw": result["draw"],
                "moves": len(result["moves"]),
            }
        )

    return pd.DataFrame(rows)


def summarize_results(df: pd.DataFrame) -> dict:
    """Aggregate per-game results into summary statistics.

    Expects a DataFrame produced by :func:`evaluate_match`.

    Args:
        df: Per-game results DataFrame.

    Returns:
        Dictionary with keys:

        * ``games``            — total games.
        * ``agent1_win_rate``  — fraction of games won by agent1.
        * ``agent2_win_rate``  — fraction of games won by agent2.
        * ``draw_rate``        — fraction of draws.
        * ``avg_moves``        — mean game length.
        * ``median_moves``     — median game length.
    """
    if df.empty:
        return {
            "games": 0,
            "agent1_win_rate": 0.0,
            "agent2_win_rate": 0.0,
            "draw_rate": 0.0,
            "avg_moves": 0.0,
            "median_moves": 0.0,
        }

    n = len(df)
    agent1_name = df["agent1_name"].iloc[0]
    agent2_name = df["agent2_name"].iloc[0]

    agent1_wins = (df["winner_agent"] == agent1_name).sum()
    agent2_wins = (df["winner_agent"] == agent2_name).sum()
    draws = df["draw"].sum()

    return {
        "games": n,
        "agent1_win_rate": float(agent1_wins / n),
        "agent2_win_rate": float(agent2_wins / n),
        "draw_rate": float(draws / n),
        "avg_moves": float(df["moves"].mean()),
        "median_moves": float(df["moves"].median()),
    }
