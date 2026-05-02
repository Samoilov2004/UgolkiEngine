"""Publication-ready plots for Uniform vs Prioritized Experience Replay ablation.

Data layout expected
--------------------
``curves`` DataFrame  (from aggregated_learning_curves.csv):
    columns: ``replay_type``, ``seed``, ``episode``, ``total_steps``,
             ``winner``, ``moves``, ``loss_mean``, ``td_error_abs_mean``,
             ``per_beta``, ``priority_mean``, ``priority_max``,
             ``priority_std``, …

``evals`` DataFrame  (from final_eval_summary.csv):
    columns: ``replay_type``, ``seed``,
             ``agent``, ``games``, ``wins``, ``draws``, ``losses``,
             ``win_rate``, ``draw_rate``, ``avg_moves``

Each function returns ``True`` on success and ``False`` when data is
unavailable (a ``logging.WARNING`` is emitted in that case).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

log = logging.getLogger(__name__)

# ── Global style ──────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi":        200,
    "savefig.dpi":       200,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "legend.fontsize":   10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
})

PALETTE: dict[str, str] = {
    "uniform":     "#2196F3",   # blue
    "prioritized": "#FF5722",   # deep orange
}
FIGSIZE    = (8.0, 4.8)
FIGSIZE_SQ = (6.0, 5.0)
DPI        = 200

# ── Internal helpers ──────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", path)


def _has_col(df: pd.DataFrame, col: str, ctx: str) -> bool:
    """Return True if column exists; emit warning otherwise."""
    if col not in df.columns:
        log.warning("[%s] Column '%s' not found — skipping.", ctx, col)
        return False
    return True


def _smoothing_window(n_points: int, frac: float = 0.05, minimum: int = 5) -> int:
    """Adaptive smoothing window: 5% of total points, at least ``minimum``."""
    return max(minimum, int(n_points * frac))


def _smooth(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=1, center=True).mean()


def _mean_std_by_episode(
    df: pd.DataFrame,
    y_col: str,
    x_col: str = "episode",
) -> dict[str, tuple[pd.Series, pd.Series]]:
    """Per-replay_type mean ± std grouped on ``x_col``."""
    result: dict[str, tuple[pd.Series, pd.Series]] = {}
    for rt, grp in df.groupby("replay_type"):
        g = grp.groupby(x_col)[y_col]
        mean = g.mean()
        std  = g.std().fillna(0.0)
        result[str(rt)] = (mean, std)
    return result


def _draw_curve(
    ax: plt.Axes,
    x: pd.Index,
    mean: pd.Series,
    std: pd.Series,
    label: str,
    color: str,
    window: int,
) -> None:
    """Draw a smoothed mean line with shaded ±std band."""
    y_s   = _smooth(mean, window)
    std_s = _smooth(std,  window)
    ax.plot(x, y_s, label=label, color=color, linewidth=1.8)
    ax.fill_between(x, y_s - std_s, y_s + std_s, alpha=0.20, color=color)


# ── Helpers for eval-summary plots ───────────────────────────────────────────

def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _grouped_bar(
    ax: plt.Axes,
    grouped: pd.DataFrame,
    x_col: str,
    y_col: str,
    err_col: str,
    replay_types: list[str],
    width: float = 0.35,
) -> None:
    """Generic grouped-bar helper."""
    categories = sorted(grouped[x_col].unique())
    x = np.arange(len(categories))
    n = len(replay_types)
    for i, rt in enumerate(replay_types):
        sub    = grouped[grouped["replay_type"] == rt]
        means  = []
        errors = []
        for cat in categories:
            row = sub[sub[x_col] == cat]
            means.append(float(row[y_col].iloc[0])  if len(row) else 0.0)
            errors.append(float(row[err_col].iloc[0]) if len(row) else 0.0)
        offset = (i - (n - 1) / 2) * width
        ax.bar(x + offset, means, width,
               label=rt.capitalize(),
               color=PALETTE.get(rt, "#999"),
               alpha=0.85,
               yerr=errors, capsize=5, error_kw={"linewidth": 1.2})
    ax.set_xticks(x)
    ax.set_xticklabels(
        [str(c).replace("_", " ").title() for c in categories],
        rotation=20, ha="right",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 1–3  Learning-curve win rates vs specific opponents
# ═══════════════════════════════════════════════════════════════════════════════

def plot_learning_curve_win_rate(
    curves: pd.DataFrame,
    opponent: str,
    out_path: Path,
    smoothing_frac: float = 0.05,
) -> bool:
    """Plot training-episode win-rate curve for one opponent type.

    Falls back to smoothed self-play win rate (Player 1) when the
    ``eval_win_rate_<opponent>`` column is absent from *curves*.

    Args:
        curves: Aggregated learning-curves DataFrame.
        opponent: One of ``"random"``, ``"greedy"``, ``"heuristic"``.
        out_path: Destination path.
        smoothing_frac: Rolling-window size as fraction of total episodes.

    Returns:
        ``True`` on success, ``False`` if data is unavailable.
    """
    col = f"eval_win_rate_{opponent}"

    # ── Primary: dedicated eval column ───────────────────────────────────────
    if col in curves.columns:
        curves_clean = curves.dropna(subset=[col])
        if curves_clean.empty:
            log.warning("[lc_win_rate_%s] Column '%s' is all-NaN.", opponent, col)
            return False

        data   = _mean_std_by_episode(curves_clean, col)
        n_pts  = max(len(v[0]) for v in data.values()) if data else 1
        window = _smoothing_window(n_pts, smoothing_frac)

        fig, ax = plt.subplots(figsize=FIGSIZE)
        for rt, (mean, std) in sorted(data.items()):
            _draw_curve(ax, mean.index, mean, std,
                        rt.capitalize(), PALETTE.get(rt, "#666"), window)

        ax.set_xlabel("Episode")
        ax.set_ylabel(f"Win Rate vs {opponent.capitalize()}")
        ax.set_title(f"Learning Curve: Win Rate vs {opponent.capitalize()}")
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        plt.tight_layout()
        _save(fig, out_path)
        return True

    # ── Fallback: smoothed self-play win rate ────────────────────────────────
    log.warning(
        "[lc_win_rate_%s] Column '%s' not found — "
        "falling back to smoothed self-play win rate (Player 1).",
        opponent, col,
    )
    if "winner" not in curves.columns:
        log.warning("[lc_win_rate_%s] 'winner' column also absent — skipping.", opponent)
        return False

    df = curves.copy()
    df["p1_win"] = (df["winner"] == 1).astype(float)
    data   = _mean_std_by_episode(df, "p1_win")
    n_pts  = max(len(v[0]) for v in data.values()) if data else 1
    window = _smoothing_window(n_pts, smoothing_frac)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for rt, (mean, std) in sorted(data.items()):
        _draw_curve(ax, mean.index, mean, std,
                    rt.capitalize(), PALETTE.get(rt, "#666"), window)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Self-Play Win Rate (Player 1)")
    ax.set_title(
        f"Learning Curve: Self-Play Win Rate\n"
        f"(proxy — eval_win_rate_{opponent} not available)"
    )
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    plt.tight_layout()
    _save(fig, out_path)
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# 4  Sample efficiency
# ═══════════════════════════════════════════════════════════════════════════════

def plot_sample_efficiency(
    curves: pd.DataFrame,
    out_path: Path,
    threshold: float = 0.60,
    opponent: str = "random",
) -> bool:
    """Bar plot: first episode at which win rate ≥ threshold.

    If ``eval_win_rate_<opponent>`` is absent, falls back to
    smoothed self-play win rate.

    Args:
        curves: Aggregated learning-curves DataFrame.
        out_path: Destination path.
        threshold: Win-rate threshold to detect (default 0.60).
        opponent: Opponent name used in column lookup.

    Returns:
        ``True`` on success.
    """
    col = f"eval_win_rate_{opponent}"
    using_proxy = False

    if col not in curves.columns:
        log.warning(
            "[sample_efficiency] '%s' not found — using self-play win rate proxy.", col
        )
        if "winner" not in curves.columns:
            log.warning("[sample_efficiency] 'winner' absent too — skipping.")
            return False
        curves = curves.copy()
        curves["p1_win"] = (curves["winner"] == 1).astype(float)
        col = "p1_win"
        using_proxy = True

    # Compute smoothed per-seed curves, then find first crossing
    results: dict[str, list[Optional[int]]] = {}
    for (rt, seed), grp in curves.groupby(["replay_type", "seed"]):
        grp_s = grp.sort_values("episode")
        window = _smoothing_window(len(grp_s))
        smoothed = _smooth(grp_s[col].reset_index(drop=True), window)
        hit = smoothed[smoothed >= threshold]
        ep = int(grp_s["episode"].iloc[hit.index[0]]) if len(hit) > 0 else None
        results.setdefault(str(rt), []).append(ep)

    bar_data = []
    for rt in sorted(results):
        vals    = results[rt]
        reached = [v for v in vals if v is not None]
        missed  = len(vals) - len(reached)
        bar_data.append({
            "replay_type":  rt,
            "mean":         float(np.mean(reached)) if reached else None,
            "std":          float(np.std(reached, ddof=1)) if len(reached) > 1 else 0.0,
            "missed":       missed,
            "n_seeds":      len(vals),
        })

    fig, ax = plt.subplots(figsize=FIGSIZE_SQ)
    x = np.arange(len(bar_data))

    for i, row in enumerate(bar_data):
        color = PALETTE.get(row["replay_type"], "#999")
        if row["mean"] is not None:
            ax.bar(i, row["mean"], color=color, alpha=0.85,
                   yerr=row["std"], capsize=7,
                   error_kw={"linewidth": 1.2})
            ax.text(i, row["mean"] + row["std"] + ax.get_ylim()[1] * 0.02,
                    f"{row['mean']:.0f}", ha="center", va="bottom", fontsize=10)
        else:
            ax.bar(i, 0, color=color, alpha=0.25)
            ax.text(i, ax.get_ylim()[1] * 0.05,
                    "Not reached", ha="center", va="bottom",
                    fontsize=9, color="#C62828", weight="bold")
        if row["missed"] > 0:
            ax.text(i, -ax.get_ylim()[1] * 0.08,
                    f"({row['missed']}/{row['n_seeds']} missed)",
                    ha="center", va="top", fontsize=8, color="#B71C1C")

    ax.set_xticks(x)
    ax.set_xticklabels([r["replay_type"].capitalize() for r in bar_data])
    ax.set_ylabel("Episodes to Reach Threshold")
    proxy_note = "\n(self-play proxy)" if using_proxy else f" vs {opponent.capitalize()}"
    ax.set_title(f"Sample Efficiency (threshold = {threshold}{proxy_note})")
    plt.tight_layout()
    _save(fig, out_path)
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# 5  Final evaluation comparison
# ═══════════════════════════════════════════════════════════════════════════════

def plot_final_eval_comparison(evals: pd.DataFrame, out_path: Path) -> bool:
    """Grouped bar: DQN win rate vs each baseline per replay strategy.

    Filters eval summary to ``agent == "dqn"`` rows, groups by
    ``replay_type``, uses ``win_rate`` ± std across seeds as error bars.
    If no per-opponent breakdown exists, shows aggregate win rate.

    Args:
        evals: Aggregated eval-summary DataFrame.
        out_path: Destination path.

    Returns:
        ``True`` on success.
    """
    win_col = _find_col(evals, ["win_rate", "win_fraction", "wins"])
    if win_col is None:
        log.warning("[final_eval] No win-rate column found in evals. Available: %s",
                    list(evals.columns))
        return False

    # Filter to DQN rows only
    agent_col = _find_col(evals, ["agent", "agent_name", "name"])
    if agent_col is None:
        log.warning("[final_eval] No agent-name column found.")
        return False

    dqn_mask = evals[agent_col].astype(str).str.lower().str.contains("dqn")
    dqn_df   = evals[dqn_mask]

    if dqn_df.empty:
        log.warning("[final_eval] No DQN rows found in eval summary.")
        return False

    grouped = (
        dqn_df.groupby("replay_type")[win_col]
        .agg(mean="mean", std="std")
        .reset_index()
    )
    grouped["std"] = grouped["std"].fillna(0.0)

    replay_types = sorted(grouped["replay_type"].unique())

    fig, ax = plt.subplots(figsize=FIGSIZE_SQ)
    x = np.arange(len(replay_types))

    for i, rt in enumerate(replay_types):
        row   = grouped[grouped["replay_type"] == rt]
        mean_ = float(row["mean"].iloc[0]) if len(row) else 0.0
        std_  = float(row["std"].iloc[0])  if len(row) else 0.0
        ax.bar(i, mean_, color=PALETTE.get(rt, "#999"), alpha=0.85,
               yerr=std_, capsize=8, error_kw={"linewidth": 1.4})
        ax.text(i, mean_ + std_ + 0.02,
                f"{mean_:.3f}", ha="center", va="bottom", fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels([rt.capitalize() for rt in replay_types])
    ax.set_ylabel("Win Rate (DQN, all opponents)")
    ax.set_ylim(0, min(1.15, grouped["mean"].max() + grouped["std"].max() + 0.25))
    ax.set_title("Final Evaluation: DQN Win Rate by Replay Strategy\n"
                 "(mean ± std across seeds, vs all tournament opponents)")
    plt.tight_layout()
    _save(fig, out_path)
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# 6  Draw rate comparison
# ═══════════════════════════════════════════════════════════════════════════════

def plot_draw_rate_comparison(evals: pd.DataFrame, out_path: Path) -> bool:
    """Grouped bar: DQN draw rate per replay strategy.

    Args:
        evals: Aggregated eval-summary DataFrame.
        out_path: Destination path.

    Returns:
        ``True`` on success.
    """
    draw_col = _find_col(evals, ["draw_rate", "draw_fraction", "draws"])
    if draw_col is None:
        log.warning("[draw_rate] No draw-rate column found. Available: %s",
                    list(evals.columns))
        return False

    agent_col = _find_col(evals, ["agent", "agent_name", "name"])
    if agent_col is None:
        log.warning("[draw_rate] No agent-name column found.")
        return False

    dqn_mask = evals[agent_col].astype(str).str.lower().str.contains("dqn")
    dqn_df   = evals[dqn_mask]

    if dqn_df.empty:
        log.warning("[draw_rate] No DQN rows found in eval summary.")
        return False

    grouped = (
        dqn_df.groupby("replay_type")[draw_col]
        .agg(mean="mean", std="std")
        .reset_index()
    )
    grouped["std"] = grouped["std"].fillna(0.0)

    replay_types = sorted(grouped["replay_type"].unique())
    fig, ax = plt.subplots(figsize=FIGSIZE_SQ)
    x = np.arange(len(replay_types))

    for i, rt in enumerate(replay_types):
        row   = grouped[grouped["replay_type"] == rt]
        mean_ = float(row["mean"].iloc[0]) if len(row) else 0.0
        std_  = float(row["std"].iloc[0])  if len(row) else 0.0
        ax.bar(i, mean_, color=PALETTE.get(rt, "#999"), alpha=0.85,
               yerr=std_, capsize=8, error_kw={"linewidth": 1.4})
        ax.text(i, mean_ + std_ + 0.005,
                f"{mean_:.3f}", ha="center", va="bottom", fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels([rt.capitalize() for rt in replay_types])
    ax.set_ylabel("Draw Rate (DQN, all opponents)")
    ax.set_ylim(0, max(0.3, grouped["mean"].max() + grouped["std"].max() + 0.1))
    ax.set_title("Final Evaluation: DQN Draw Rate by Replay Strategy\n"
                 "(mean ± std across seeds)")
    plt.tight_layout()
    _save(fig, out_path)
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# 7  Average game length
# ═══════════════════════════════════════════════════════════════════════════════

def plot_avg_moves_comparison(
    evals: Optional[pd.DataFrame],
    curves: Optional[pd.DataFrame],
    out_path: Path,
    smoothing_frac: float = 0.05,
) -> bool:
    """Average game length — from eval summary if available, else training log.

    Args:
        evals: Aggregated eval-summary DataFrame (may be ``None``).
        curves: Aggregated training-curves DataFrame (fallback, may be ``None``).
        out_path: Destination path.

    Returns:
        ``True`` on success.
    """
    # ── Primary: eval summary ────────────────────────────────────────────────
    if evals is not None:
        moves_col = _find_col(evals, ["avg_moves", "mean_moves", "avg_game_length"])
        agent_col = _find_col(evals, ["agent", "agent_name", "name"])

        if moves_col and agent_col:
            dqn_mask = evals[agent_col].astype(str).str.lower().str.contains("dqn")
            dqn_df   = evals[dqn_mask]

            if not dqn_df.empty:
                grouped = (
                    dqn_df.groupby("replay_type")[moves_col]
                    .agg(mean="mean", std="std")
                    .reset_index()
                )
                grouped["std"] = grouped["std"].fillna(0.0)

                replay_types = sorted(grouped["replay_type"].unique())
                fig, ax = plt.subplots(figsize=FIGSIZE_SQ)
                x = np.arange(len(replay_types))

                for i, rt in enumerate(replay_types):
                    row   = grouped[grouped["replay_type"] == rt]
                    mean_ = float(row["mean"].iloc[0]) if len(row) else 0.0
                    std_  = float(row["std"].iloc[0])  if len(row) else 0.0
                    ax.bar(i, mean_, color=PALETTE.get(rt, "#999"), alpha=0.85,
                           yerr=std_, capsize=8, error_kw={"linewidth": 1.4})
                    ax.text(i, mean_ + std_ + 0.5,
                            f"{mean_:.1f}", ha="center", va="bottom", fontsize=11)

                ax.set_xticks(x)
                ax.set_xticklabels([rt.capitalize() for rt in replay_types])
                ax.set_ylabel("Average Moves per Game")
                ax.set_title("Average Game Length: DQN by Replay Strategy\n"
                             "(mean ± std across seeds, tournament games)")
                plt.tight_layout()
                _save(fig, out_path)
                return True

    # ── Fallback: training-log self-play game length ─────────────────────────
    if curves is not None and "moves" in curves.columns:
        log.warning("[avg_moves] Falling back to self-play moves from training log.")
        data   = _mean_std_by_episode(curves, "moves")
        n_pts  = max(len(v[0]) for v in data.values()) if data else 1
        window = _smoothing_window(n_pts, smoothing_frac)

        fig, ax = plt.subplots(figsize=FIGSIZE)
        for rt, (mean, std) in sorted(data.items()):
            _draw_curve(ax, mean.index, mean, std,
                        rt.capitalize(), PALETTE.get(rt, "#666"), window)

        ax.set_xlabel("Episode")
        ax.set_ylabel("Moves per Episode (Self-Play)")
        ax.set_title("Average Game Length over Training")
        ax.legend()
        plt.tight_layout()
        _save(fig, out_path)
        return True

    log.warning("[avg_moves] No moves data found in either evals or curves.")
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# 8  TD-error dynamics
# ═══════════════════════════════════════════════════════════════════════════════

def plot_td_error_dynamics(
    curves: pd.DataFrame,
    out_path: Path,
    smoothing_frac: float = 0.05,
) -> bool:
    """Line plot of |TD error| mean over training episodes.

    Args:
        curves: Aggregated learning-curves DataFrame.
        out_path: Destination path.

    Returns:
        ``True`` on success.
    """
    col = "td_error_abs_mean"
    if not _has_col(curves, col, "td_error_dynamics"):
        return False

    clean  = curves.dropna(subset=[col])
    data   = _mean_std_by_episode(clean, col)
    n_pts  = max(len(v[0]) for v in data.values()) if data else 1
    window = _smoothing_window(n_pts, smoothing_frac)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for rt, (mean, std) in sorted(data.items()):
        _draw_curve(ax, mean.index, mean, std,
                    rt.capitalize(), PALETTE.get(rt, "#666"), window)

    ax.set_xlabel("Episode")
    ax.set_ylabel("|TD Error| Mean")
    ax.set_title("TD Error Dynamics over Training")
    ax.legend()
    plt.tight_layout()
    _save(fig, out_path)
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# 9  Priority dynamics (PER only)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_priority_dynamics(
    curves: pd.DataFrame,
    out_path: Path,
    smoothing_frac: float = 0.05,
) -> bool:
    """Line plot of priority_mean / priority_max / priority_std (PER only).

    Args:
        curves: Aggregated learning-curves DataFrame.
        out_path: Destination path.

    Returns:
        ``True`` on success.
    """
    per_df = curves[curves["replay_type"] == "prioritized"]
    if per_df.empty:
        log.warning("[priority_dynamics] No 'prioritized' rows in curves.")
        return False

    metrics = {
        "priority_mean": ("#1565C0", "Mean"),
        "priority_max":  ("#B71C1C", "Max"),
        "priority_std":  ("#2E7D32", "Std"),
    }
    available = {k: v for k, v in metrics.items() if k in per_df.columns}
    if not available:
        log.warning("[priority_dynamics] No priority_* columns found.")
        return False

    n_pts  = per_df["episode"].nunique()
    window = _smoothing_window(n_pts, smoothing_frac)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for col, (color, label) in available.items():
        grouped = per_df.groupby("episode")[col].mean().dropna()
        if grouped.empty:
            continue
        y_s = _smooth(grouped, window)
        ax.plot(grouped.index, y_s, label=label, color=color, linewidth=1.8)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Priority Value")
    ax.set_title("Replay Priority Dynamics (PER only)")
    ax.legend()
    plt.tight_layout()
    _save(fig, out_path)
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# 10  Beta annealing schedule
# ═══════════════════════════════════════════════════════════════════════════════

def plot_beta_schedule(curves: pd.DataFrame, out_path: Path) -> bool:
    """Line plot of PER beta (IS-correction exponent) per episode.

    Shows one thin line per seed, plus a thick mean line.

    Args:
        curves: Aggregated learning-curves DataFrame.
        out_path: Destination path.

    Returns:
        ``True`` on success.
    """
    col = "per_beta"
    if not _has_col(curves, col, "beta_schedule"):
        return False

    per_df = curves[curves["replay_type"] == "prioritized"]
    if per_df.empty:
        log.warning("[beta_schedule] No 'prioritized' rows.")
        return False

    fig, ax = plt.subplots(figsize=FIGSIZE)

    seeds = sorted(per_df["seed"].unique())
    for seed in seeds:
        sub = per_df[per_df["seed"] == seed].sort_values("episode")
        beta = pd.to_numeric(sub[col], errors="coerce")
        valid = beta.notna()
        ax.plot(sub["episode"][valid].values, beta[valid].values,
                alpha=0.35, linewidth=1.0,
                color=PALETTE["prioritized"],
                label=f"Seed {seed}" if len(seeds) <= 6 else None)

    # Mean across seeds
    mean_beta = (
        per_df.groupby("episode")[col]
        .apply(lambda s: pd.to_numeric(s, errors="coerce").mean())
    )
    valid = mean_beta.notna()
    ax.plot(mean_beta.index[valid], mean_beta[valid],
            color=PALETTE["prioritized"], linewidth=2.5,
            label="Mean (PER)", zorder=5)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Beta (IS-weight exponent)")
    ax.set_title("PER Beta Annealing Schedule")
    if len(seeds) <= 6:
        ax.legend(loc="lower right")
    else:
        ax.legend(["Mean (PER)"], loc="lower right")
    plt.tight_layout()
    _save(fig, out_path)
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# Summary tables
# ═══════════════════════════════════════════════════════════════════════════════

def build_summary_tables(evals: pd.DataFrame) -> tuple[str, str]:
    """Build LaTeX and Markdown summary tables from eval summary.

    Groups by ``(replay_type, agent)`` and computes ``mean ± std`` across
    seeds for each numeric metric.

    Args:
        evals: Aggregated eval-summary DataFrame.

    Returns:
        Tuple ``(latex_str, markdown_str)``.  Both are empty strings on
        failure (a warning is logged).
    """
    agent_col = _find_col(evals, ["agent", "agent_name", "name"])
    if agent_col is None:
        log.warning("[summary_table] No agent column — cannot build table.")
        return "", ""

    metric_cols = [c for c in ["win_rate", "draw_rate", "avg_moves"]
                   if c in evals.columns]
    if not metric_cols:
        log.warning("[summary_table] No metric columns found.")
        return "", ""

    rows = []
    for (rt, agent), grp in evals.groupby(["replay_type", agent_col]):
        row: dict = {
            "Replay Type": str(rt).capitalize(),
            "Agent":       str(agent),
        }
        for m in metric_cols:
            vals = pd.to_numeric(grp[m], errors="coerce").dropna()
            if len(vals) > 1:
                row[m.replace("_", " ").title()] = f"{vals.mean():.3f} ± {vals.std(ddof=1):.3f}"
            elif len(vals) == 1:
                row[m.replace("_", " ").title()] = f"{vals.iloc[0]:.3f}"
            else:
                row[m.replace("_", " ").title()] = "—"
        rows.append(row)

    if not rows:
        log.warning("[summary_table] No rows assembled.")
        return "", ""

    df_table = pd.DataFrame(rows).sort_values(
        ["Replay Type", "Agent"]
    ).reset_index(drop=True)

    # ── Markdown ─────────────────────────────────────────────────────────────
    try:
        md = df_table.to_markdown(index=False)
    except ImportError:
        # tabulate not installed — manual fallback
        header = "| " + " | ".join(df_table.columns) + " |"
        sep    = "| " + " | ".join(["---"] * len(df_table.columns)) + " |"
        body   = "\n".join(
            "| " + " | ".join(str(v) for v in row) + " |"
            for row in df_table.itertuples(index=False)
        )
        md = "\n".join([header, sep, body])

    # ── LaTeX ────────────────────────────────────────────────────────────────
    latex = df_table.to_latex(
        index=False,
        escape=True,
        caption=(
            "Final tournament evaluation results (mean $\\pm$ std across seeds). "
            "Win Rate and Draw Rate are fractions of total games played."
        ),
        label="tab:replay_ablation",
        column_format="l" * len(df_table.columns),
    )

    return latex, md
