"""Report-quality plots for training curves, evaluation results, and board heatmaps."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

log = logging.getLogger(__name__)

# ── Global style ──────────────────────────────────────────────────────────────

_STYLE       = "whitegrid"
_CONTEXT     = "talk"        # larger fonts — good for slides/reports
_DPI         = 200
_PALETTE     = "tab10"

_C_P1        = "#4A90D9"     # blue  — Player 1
_C_P2        = "#E05252"     # red   — Player -1
_C_WIN       = "#2ECC71"     # green
_C_LOSS      = "#E74C3C"     # red
_C_DRAW      = "#95A5A6"     # grey
_C_ACCENT    = "#F39C12"     # orange accent

_ROLL_SHORT  = 10
_ROLL_LONG   = 30


def _apply_style() -> None:
    sns.set_theme(style=_STYLE, context=_CONTEXT, palette=_PALETTE)
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "savefig.facecolor": "white",
        "savefig.bbox":      "tight",
        "savefig.dpi":       _DPI,
    })


def _save(fig: plt.Figure, path: Path, tight: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)


def _rolling(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=1).mean()


# ── Training curves ───────────────────────────────────────────────────────────

def plot_training_curves(
    train_log_path: str | Path,
    out_dir: str | Path,
    roll_short: int = _ROLL_SHORT,
    roll_long: int = _ROLL_LONG,
) -> None:
    """Read *train_log.csv* and save five training-curve plots.

    Plots produced:

    * ``training_epsilon.png``  — ε-greedy schedule.
    * ``training_moves.png``    — moves per episode + rolling mean.
    * ``training_loss.png``     — mean TD loss + rolling mean.
    * ``training_winners.png``  — rolling win / draw distribution (stacked area).
    * ``training_rewards.png``  — rolling mean total reward for each player.

    Args:
        train_log_path: Path to the CSV produced by ``SelfPlayTrainer``.
        out_dir: Directory where PNGs are written.
        roll_short: Short rolling-average window for raw data overlay.
        roll_long: Long rolling-average window for trend line.
    """
    _apply_style()
    out = Path(out_dir)
    path = Path(train_log_path)

    df = pd.read_csv(path)

    ep = df["episode"]

    # ── 1. Epsilon ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ep, df["epsilon"], color=_C_ACCENT, linewidth=2, label="ε (epsilon)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Exploration rate ε")
    ax.set_title("Epsilon-Greedy Decay Schedule")
    ax.set_ylim(-0.02, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(framealpha=0.8)
    _save(fig, out / "training_epsilon.png")

    # ── 2. Moves per episode ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ep, df["moves"], color="#BDC3C7", linewidth=0.8,
            alpha=0.5, label="Raw")
    ax.plot(ep, _rolling(df["moves"], roll_short), color=_C_P1, linewidth=1.5,
            alpha=0.8, label=f"Rolling mean ({roll_short})")
    ax.plot(ep, _rolling(df["moves"], roll_long), color=_C_P2, linewidth=2.2,
            label=f"Rolling mean ({roll_long})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Half-moves")
    ax.set_title("Game Length per Episode")
    ax.legend(framealpha=0.8)
    _save(fig, out / "training_moves.png")

    # ── 3. TD Loss ────────────────────────────────────────────────────────────
    loss_col = df["loss_mean"].replace(0, np.nan)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ep, loss_col, color="#BDC3C7", linewidth=0.6, alpha=0.4, label="Raw")
    ax.plot(ep, _rolling(loss_col, roll_short), color=_C_P1, linewidth=1.5,
            alpha=0.8, label=f"Rolling mean ({roll_short})")
    ax.plot(ep, _rolling(loss_col, roll_long), color=_C_P2, linewidth=2.2,
            label=f"Rolling mean ({roll_long})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("MSE loss")
    ax.set_title("Mean TD Loss per Episode")
    ax.legend(framealpha=0.8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ax.set_yscale("log")
    _save(fig, out / "training_loss.png")

    # ── 4. Win / draw distribution ────────────────────────────────────────────
    w = _roll_winner_fracs(df, roll_long)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.stackplot(
        ep,
        w["p1_win"], w["draw"], w["p2_win"],
        labels=["P1 wins", "Draw", "P-1 wins"],
        colors=[_C_P1, _C_DRAW, _C_P2],
        alpha=0.75,
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Fraction (rolling {roll_long})")
    ax.set_title("Win / Draw Distribution over Training")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(loc="upper right", framealpha=0.8)
    _save(fig, out / "training_winners.png")

    # ── 5. Rewards ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    for col, color, label in (
        ("total_reward_player1",        _C_P1, "Player 1"),
        ("total_reward_player_minus1",  _C_P2, "Player -1"),
    ):
        if col not in df.columns:
            continue
        ax.plot(ep, _rolling(df[col], roll_long), color=color,
                linewidth=2, label=label)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Total shaped reward (rolling {roll_long})")
    ax.set_title("Total Reward per Episode")
    ax.legend(framealpha=0.8)
    _save(fig, out / "training_rewards.png")


def _roll_winner_fracs(df: pd.DataFrame, window: int) -> dict[str, pd.Series]:
    """Compute rolling fraction of P1 wins, draws, and P-1 wins."""
    p1   = (df["winner"] == 1).astype(float)
    pm1  = (df["winner"] == -1).astype(float)
    draw = df["winner"].isna().astype(float)

    roll = lambda s: s.rolling(window, min_periods=1).mean()  # noqa: E731
    p1_r   = roll(p1)
    pm1_r  = roll(pm1)
    draw_r = roll(draw)

    # Normalise so fractions sum to 1 (they always should, but float drift)
    total = p1_r + pm1_r + draw_r
    total = total.replace(0, 1)
    return {
        "p1_win":  p1_r  / total,
        "draw":    draw_r / total,
        "p2_win":  pm1_r / total,
    }


# ── Evaluation plots ──────────────────────────────────────────────────────────

def plot_evaluation_summary(
    summary_path: str | Path,
    out_dir: str | Path,
    detailed_path: Optional[str | Path] = None,
) -> None:
    """Plot per-agent win / draw rates and average game lengths.

    Plots produced:

    * ``eval_win_rates.png``  — grouped bar chart: win rate + draw rate.
    * ``eval_avg_moves.png``  — bar chart of average game length per agent.

    Args:
        summary_path: Path to ``summary.csv`` from the tournament.
        out_dir: Output directory.
        detailed_path: Optional path to ``detailed_results.csv``.  When
                       provided, an additional pair-wise win-rate heatmap
                       ``eval_pairwise.png`` is saved.
    """
    _apply_style()
    out = Path(out_dir)

    df = pd.read_csv(summary_path)
    df = df.sort_values("win_rate", ascending=False).reset_index(drop=True)

    agents = df["agent"].tolist()
    x = np.arange(len(agents))
    width = 0.35

    # ── Win/draw rates bar chart ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(8, len(agents) * 2), 5))
    bars_win  = ax.bar(x - width / 2, df["win_rate"],  width,
                       color=_C_WIN,  label="Win rate",  zorder=3)
    bars_draw = ax.bar(x + width / 2, df["draw_rate"], width,
                       color=_C_DRAW, label="Draw rate", zorder=3)

    # Value labels
    for bar in list(bars_win) + list(bars_draw):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.01,
            f"{h:.0%}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([a.capitalize() for a in agents])
    ax.set_ylabel("Rate")
    ax.set_title("Agent Win & Draw Rates (Round-Robin Tournament)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylim(0, min(1.0, df["win_rate"].max() * 1.4 + 0.15))
    ax.legend(framealpha=0.8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    _save(fig, out / "eval_win_rates.png")

    # ── Average game length ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(8, len(agents) * 2), 5))
    bars = ax.bar(agents, df["avg_moves"], color=_C_P1, zorder=3)
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 1,
            f"{h:.0f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
    ax.set_xlabel("Agent")
    ax.set_ylabel("Average half-moves")
    ax.set_title("Average Game Length per Agent")
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    _save(fig, out / "eval_avg_moves.png")

    # ── Pairwise win-rate heatmap (optional) ──────────────────────────────────
    if detailed_path is not None:
        _plot_pairwise_heatmap(pd.read_csv(detailed_path), out)


def _plot_pairwise_heatmap(det: pd.DataFrame, out: Path) -> None:
    """Save a pairwise win-rate heatmap from detailed game records."""
    all_agents = sorted(
        set(det["agent1_name"].tolist() + det["agent2_name"].tolist())
    )
    n = len(all_agents)
    matrix = np.full((n, n), np.nan)

    for i, a in enumerate(all_agents):
        for j, b in enumerate(all_agents):
            if a == b:
                continue
            mask = (
                ((det["agent1_name"] == a) & (det["agent2_name"] == b)) |
                ((det["agent1_name"] == b) & (det["agent2_name"] == a))
            )
            sub = det[mask]
            if len(sub) == 0:
                continue
            wins_a = (sub["winner_agent"] == a).sum()
            matrix[i, j] = wins_a / len(sub)

    fig, ax = plt.subplots(figsize=(max(6, n + 2), max(5, n + 1)))
    labels = [a.capitalize() for a in all_agents]
    mask_nan = np.isnan(matrix)
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".0%",
        xticklabels=labels,
        yticklabels=labels,
        cmap="RdYlGn",
        vmin=0, vmax=1,
        linewidths=0.5,
        mask=mask_nan,
        ax=ax,
        cbar_kws={"label": "Win rate (row agent vs col agent)"},
    )
    ax.set_title("Pairwise Win Rates")
    ax.set_xlabel("Opponent")
    ax.set_ylabel("Agent")
    _save(fig, out / "eval_pairwise.png")


# ── Board position heatmap ────────────────────────────────────────────────────

def generate_position_heatmap(
    agent1,
    agent2,
    n_games: int = 50,
    max_moves: int = 300,
    seed: int = 42,
    out_dir: str | Path = "outputs/figures",
    filename: str = "heatmap_positions.png",
) -> None:
    """Play *n_games* games and plot a heatmap of final piece positions.

    Two sub-plots are produced side by side: one for each player's pieces,
    showing how often each board cell was occupied at the END of a game.

    Args:
        agent1: Agent for Player 1.
        agent2: Agent for Player -1.
        n_games: Number of games to record.
        max_moves: Step limit per game.
        seed: Base random seed.
        out_dir: Output directory.
        filename: Output filename.
    """
    import random as _r

    from corners_rl.env.corners_env import CornersEnv
    from corners_rl.env.rules import PLAYER1, PLAYER2

    _apply_style()
    out = Path(out_dir)
    rng = _r.Random(seed)

    heat1 = np.zeros((8, 8), dtype=np.float64)
    heat2 = np.zeros((8, 8), dtype=np.float64)

    for _ in range(n_games):
        game_seed = rng.randint(0, 2**32 - 1)
        env = CornersEnv(max_moves=max_moves)
        env.reset()
        for ag in (agent1, agent2):
            if hasattr(ag, "_rng"):
                import random as _rand
                ag._rng = _rand.Random(game_seed)
        agent_map = {PLAYER1: agent1, PLAYER2: agent2}
        while not env.is_terminal():
            env.step(agent_map[env.current_player].select_move(env))
        board = env.board
        heat1 += (board == PLAYER1).astype(np.float64)
        heat2 += (board == PLAYER2).astype(np.float64)

    heat1 /= n_games
    heat2 /= n_games

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    cbar_kw = {"shrink": 0.85, "label": "Occupancy frequency"}

    for ax, heat, title, cmap in (
        (axes[0], heat1, f"Player 1 ({agent1.name})",  "Blues"),
        (axes[1], heat2, f"Player -1 ({agent2.name})", "Reds"),
    ):
        sns.heatmap(
            heat,
            ax=ax,
            cmap=cmap,
            vmin=0, vmax=1,
            linewidths=0.3,
            linecolor="#cccccc",
            annot=True,
            fmt=".2f",
            annot_kws={"size": 7},
            cbar_kws=cbar_kw,
        )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        ax.set_xticklabels(range(8))
        ax.set_yticklabels(range(8), rotation=0)

    fig.suptitle(
        f"Final Piece Positions Heatmap\n"
        f"({n_games} games, {agent1.name} vs {agent2.name})",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    _save(fig, out / filename)


# ── Legacy TrainingLogger (kept for backward compat) ─────────────────────────

class TrainingLogger:
    """Accumulates per-episode training metrics and evaluation results."""

    def __init__(self) -> None:
        self.episode_data: list[dict] = []
        self.eval_data: list[dict] = []

    def record(self, episode: int, stats: dict) -> None:
        self.episode_data.append({"episode": episode, **stats})

    def record_eval(self, episode: int, metrics_list: list) -> None:
        for m in metrics_list:
            self.eval_data.append({
                "episode":  episode,
                "opponent": m.opponent_name,
                "win_rate": m.win_rate,
                "avg_steps": m.avg_steps,
            })

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.episode_data)

    def to_eval_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.eval_data)

    def save_csv(self, path: Path | str) -> None:
        self.to_dataframe().to_csv(path, index=False)


def plot_training_curves_from_logger(
    logger: TrainingLogger,
    save_dir: Path | str = Path("plots"),
    smoothing_window: int = 100,
) -> None:
    """Generate training plots from a :class:`TrainingLogger` instance.

    Thin wrapper around :func:`plot_training_curves` — saves the logger data
    to a temporary CSV and calls the main plotting function.
    """
    import tempfile, os

    save_dir = Path(save_dir)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as f:
        tmp = f.name
        logger.to_dataframe().to_csv(tmp, index=False)
    try:
        plot_training_curves(tmp, save_dir, roll_long=smoothing_window)
    finally:
        os.unlink(tmp)
