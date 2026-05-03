#!/usr/bin/env python3
"""Generate publication-quality figures for the LaTeX report.

All labels in Russian. Three figures:
  fig1_td_error.png        — TD-error dynamics (learning curves)
  fig2_winrate_barplot.png — Main bar chart: win rate Uniform vs PER per opponent
  fig3_seed_stability.png  — Per-seed win rate scatter (stability)
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).parent.parent

# ── Palette & style ───────────────────────────────────────────────────────────

UNIFORM_COLOR = "#2166AC"
PER_COLOR     = "#D6604D"

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "axes.grid.axis":    "y",
    "grid.alpha":        0.35,
    "grid.linestyle":    "--",
})

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — TD-error dynamics
# ─────────────────────────────────────────────────────────────────────────────

def plot_td_error(out_path: Path) -> None:
    csv = ROOT / "outputs/experiments/main/aggregated_learning_curves.csv"
    df  = pd.read_csv(csv)

    # Keep td_error_abs_mean; drop NaN
    df = df[["replay_type", "seed", "episode", "td_error_abs_mean"]].dropna()

    window = 30   # rolling average width

    fig, ax = plt.subplots(figsize=(7, 2.7))

    for replay, label, color in [
        ("uniform",     "Равномерная выборка",        UNIFORM_COLOR),
        ("prioritized", "Приоритизированная выборка",  PER_COLOR),
    ]:
        sub = df[df["replay_type"] == replay]
        seeds = sub["seed"].unique()

        # Per-seed smoothed series
        smoothed = []
        for seed in seeds:
            s = sub[sub["seed"] == seed].sort_values("episode")["td_error_abs_mean"]
            smoothed.append(s.rolling(window, min_periods=1).mean().values)

        min_len = min(len(s) for s in smoothed)
        mat = np.array([s[:min_len] for s in smoothed])
        x   = sub[sub["seed"] == seeds[0]].sort_values("episode")["episode"].values[:min_len]

        mean = mat.mean(axis=0)
        # 95% CI via percentile across seeds (or ±1.96·sem when n≥2)
        if len(seeds) >= 2:
            sem  = mat.std(axis=0, ddof=1) / np.sqrt(len(seeds))
            lo, hi = mean - 1.96 * sem, mean + 1.96 * sem
        else:
            lo, hi = mean, mean

        ax.plot(x, mean, color=color, linewidth=2.2, label=label)
        ax.fill_between(x, lo, hi, color=color, alpha=0.18)

    ax.set_xlabel("Эпизод обучения", fontsize=11)
    ax.set_ylabel("Средняя абсолютная TD-ошибка", fontsize=11)
    ax.set_title(
        "Динамика TD-ошибки в процессе self-play обучения",
        fontsize=12, pad=8,
    )
    ax.legend(fontsize=10, loc="upper right")
    ax.yaxis.grid(True)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Win rate barplot (main result)
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(vals: np.ndarray, n_boot: int = 10_000,
                 rng: np.random.Generator | None = None):
    if rng is None:
        rng = np.random.default_rng(42)
    boot = np.array([
        rng.choice(vals, size=len(vals), replace=True).mean()
        for _ in range(n_boot)
    ])
    return (float(vals.mean()),
            float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)))


def plot_winrate_barplot(out_path: Path) -> None:
    csv = ROOT / "outputs/eval_forward_masking/per_pair_results.csv"
    df  = pd.read_csv(csv)

    # Opponent label mapping
    opp_labels = {
        "random":    "Случайный",
        "greedy":    "Жадный",
        "heuristic": "Эвристический",
    }

    opponents = ["random", "greedy", "heuristic"]
    groups    = opponents + ["overall"]  # 4 x-groups

    rng = np.random.default_rng(42)

    # Build (replay, opponent/overall) → (mean, lo, hi)
    stats: dict[tuple, tuple] = {}
    for replay in ("uniform", "prioritized"):
        sub = df[df["replay_type"] == replay]
        for opp in opponents:
            vals = sub[sub["opponent"] == opp]["win_rate"].values
            stats[(replay, opp)] = bootstrap_ci(vals, rng=rng)
        # overall — pool all games
        all_wr = sub["win_rate"].values
        stats[(replay, "overall")] = bootstrap_ci(all_wr, rng=rng)

    fig, ax = plt.subplots(figsize=(8, 3.6))

    x   = np.arange(len(groups))
    w   = 0.32
    gap = 0.05

    methods = [
        ("uniform",     "Равномерная выборка",        UNIFORM_COLOR),
        ("prioritized", "Приоритизированная выборка",  PER_COLOR),
    ]

    for i, (replay, label, color) in enumerate(methods):
        offset = (i - 0.5) * (w + gap)
        means, errs_lo, errs_hi = [], [], []
        for grp in groups:
            mean, lo, hi = stats[(replay, grp)]
            means.append(mean)
            errs_lo.append(mean - lo)
            errs_hi.append(hi - mean)

        bars = ax.bar(
            x + offset, means, w,
            color=color, alpha=0.85,
            yerr=[errs_lo, errs_hi],
            capsize=7,
            error_kw={"linewidth": 1.5, "ecolor": "#222"},
            label=label, zorder=3,
        )

        for xi, (xi_pos, mean, elo, ehi) in enumerate(
                zip(x + offset, means, errs_lo, errs_hi)):
            ax.text(xi_pos, mean + ehi + 0.012,
                    f"{mean:.0%}",
                    ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color=color)

    x_labels = [opp_labels[o] for o in opponents] + ["Суммарно"]
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.set_ylabel("Доля побед", fontsize=11)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylim(0, 0.88)
    ax.set_title(
        "Доля побед DQN-агентов по типу соперника\n"
        "(планки — 95% bootstrap доверительный интервал)",
        fontsize=12, pad=8,
    )
    ax.legend(fontsize=10, loc="upper left")

    n = int(df["n_games"].iloc[0])
    ax.text(0.5, -0.12,
            f"1 000 партий на пару · 2 инициализации · 10 000 ресэмплов bootstrap",
            transform=ax.transAxes, ha="center", fontsize=8.5, color="#555")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Seed stability
# ─────────────────────────────────────────────────────────────────────────────

def plot_seed_stability(out_path: Path) -> None:
    csv = ROOT / "outputs/eval_forward_masking/per_pair_results.csv"
    df  = pd.read_csv(csv)

    # Per-seed mean win rate across all opponents
    per_seed = (
        df.groupby(["replay_type", "seed"])["win_rate"]
        .mean()
        .reset_index()
        .rename(columns={"win_rate": "mean_wr"})
    )

    fig, ax = plt.subplots(figsize=(5, 4.5))

    positions = {"uniform": 0, "prioritized": 1}
    x_labels  = ["Равномерная\nвыборка", "Приоритизированная\nвыборка"]
    colors    = {"uniform": UNIFORM_COLOR, "prioritized": PER_COLOR}
    jitter    = {"uniform": [-0.08, 0.08], "prioritized": [-0.08, 0.08]}

    for replay in ("uniform", "prioritized"):
        sub   = per_seed[per_seed["replay_type"] == replay].sort_values("seed")
        xbase = positions[replay]
        color = colors[replay]
        vals  = sub["mean_wr"].values
        seeds = sub["seed"].values

        # Mean line
        mean = vals.mean()
        ax.hlines(mean, xbase - 0.25, xbase + 0.25,
                  colors=color, linewidths=2.5, zorder=3)

        # 95% CI band (simple sem-based for n=2)
        if len(vals) >= 2:
            sem = vals.std(ddof=1) / np.sqrt(len(vals))
            lo, hi = mean - 1.96 * sem, mean + 1.96 * sem
            ax.fill_between([xbase - 0.25, xbase + 0.25],
                            [lo, lo], [hi, hi],
                            color=color, alpha=0.15, zorder=2)

        # Per-seed dots
        xs = np.array(jitter[replay][:len(vals)]) + xbase
        ax.scatter(xs, vals,
                   s=90, color=color, zorder=5,
                   edgecolors="white", linewidths=1.0)
        for xi, (xp, v, s) in enumerate(zip(xs, vals, seeds)):
            ax.text(xp + 0.05, v, f"seed {s}",
                    va="center", fontsize=8.5, color="#444")

        ax.text(xbase, mean + 0.008, f"{mean:.1%}",
                ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=color)

    ax.set_xticks(list(positions.values()))
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.set_ylabel("Доля побед (среднее по противникам)", fontsize=11)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0.25, 0.75)
    ax.set_title(
        "Разброс качества политик\nмежду независимыми инициализациями",
        fontsize=12, pad=8,
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {out_path}")


# ─────────────────────────────────────────────────────────────────────────────

def plot_combined(out_path: Path) -> None:
    """Two-panel figure: left = TD error, right = win rate barplot."""
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(11, 3.8))
    gs  = GridSpec(1, 2, figure=fig, width_ratios=[1, 1.55], wspace=0.35)
    ax_td  = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1])

    # ── Panel A: TD error ─────────────────────────────────────────────────────
    csv = ROOT / "outputs/experiments/main/aggregated_learning_curves.csv"
    df  = pd.read_csv(csv)
    df  = df[["replay_type", "seed", "episode", "td_error_abs_mean"]].dropna()
    window = 30

    for replay, label, color in [
        ("uniform",     "Равномерная",        UNIFORM_COLOR),
        ("prioritized", "Приоритизированная",  PER_COLOR),
    ]:
        sub   = df[df["replay_type"] == replay]
        seeds = sub["seed"].unique()
        smoothed = []
        for seed in seeds:
            s = sub[sub["seed"] == seed].sort_values("episode")["td_error_abs_mean"]
            smoothed.append(s.rolling(window, min_periods=1).mean().values)
        min_len = min(len(s) for s in smoothed)
        mat = np.array([s[:min_len] for s in smoothed])
        x   = sub[sub["seed"] == seeds[0]].sort_values("episode")["episode"].values[:min_len]
        mean = mat.mean(axis=0)
        if len(seeds) >= 2:
            sem  = mat.std(axis=0, ddof=1) / np.sqrt(len(seeds))
            lo, hi = mean - 1.96 * sem, mean + 1.96 * sem
        else:
            lo, hi = mean, mean
        ax_td.plot(x, mean, color=color, linewidth=2.0, label=label)
        ax_td.fill_between(x, lo, hi, color=color, alpha=0.18)

    ax_td.set_xlabel("Эпизод обучения", fontsize=10)
    ax_td.set_ylabel("Ср. абс. TD-ошибка", fontsize=10)
    ax_td.set_title("(а) Динамика TD-ошибки", fontsize=11, pad=5)
    ax_td.legend(fontsize=9, loc="upper right")
    ax_td.tick_params(labelsize=9)

    # ── Panel B: ranking — win rate EXCLUDING draws (wins / (wins+losses)) ───
    # Only show: DQN Uniform, DQN PER, Heuristic (sorted descending)
    csv2 = ROOT / "outputs/eval_forward_masking/per_pair_results.csv"
    df2  = pd.read_csv(csv2)

    rng = np.random.default_rng(42)

    def non_draw_outcomes(sub_df: pd.DataFrame) -> np.ndarray:
        """Binary array: 1=win, 0=loss; draws excluded entirely."""
        parts = []
        for _, row in sub_df.iterrows():
            n      = int(row["n_games"])
            n_win  = round(float(row["win_rate"]) * n)
            n_draw = round(float(row["draw_rate"]) * n)
            n_loss = max(0, n - n_win - n_draw)
            parts.append(np.array([1] * n_win + [0] * n_loss))
        return np.concatenate(parts) if parts else np.array([])

    agent_outcomes: dict[str, np.ndarray] = {}

    for replay, key in [("uniform", "DQN\nРавномерная"), ("prioritized", "DQN\nPER")]:
        sub = df2[df2["replay_type"] == replay]
        agent_outcomes[key] = non_draw_outcomes(sub)

    # Baseline agents from opponent perspective (excluding draws)
    for opp, key in [("heuristic", "Эвристический"), ("random", "Случайный")]:
        sub_b = df2[df2["opponent"] == opp]
        parts_b = []
        for row in sub_b.itertuples(index=False):
            n        = int(row.n_games)
            n_win_dq = round(float(row.win_rate) * n)
            n_draw   = round(float(row.draw_rate) * n)
            n_win_b  = max(0, n - n_win_dq - n_draw)
            n_loss_b = n_win_dq
            parts_b.append(np.array([1] * n_win_b + [0] * n_loss_b))
        agent_outcomes[key] = np.concatenate(parts_b)

    agent_stats = {k: bootstrap_ci(v, rng=rng) for k, v in agent_outcomes.items()}

    sorted_agents = sorted(agent_stats.items(), key=lambda kv: kv[1][0], reverse=True)

    bar_colors = {
        "DQN\nРавномерная": UNIFORM_COLOR,
        "DQN\nPER":         PER_COLOR,
        "Эвристический":    "#8C7AA9",
        "Случайный":        "#AAAAAA",
    }

    y     = np.arange(len(sorted_agents))
    means = [v[0] for _, v in sorted_agents]
    lo_e  = [v[0] - v[1] for _, v in sorted_agents]
    hi_e  = [v[2] - v[0] for _, v in sorted_agents]
    names = [k for k, _ in sorted_agents]
    cols  = [bar_colors[k] for k in names]

    ax_bar.barh(y, means, 0.50, xerr=[lo_e, hi_e],
                color=cols, alpha=0.88,
                capsize=6, error_kw={"linewidth": 1.5, "ecolor": "#222"},
                zorder=3)

    for yi, (m, eh, name) in enumerate(zip(means, hi_e, names)):
        ax_bar.text(m + eh + 0.008, yi, f"{m:.1%}",
                    va="center", fontsize=9, fontweight="bold",
                    color=bar_colors[name])

    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(names, fontsize=10)
    ax_bar.set_xlabel("Доля побед (без учёта ничьих)", fontsize=10)
    ax_bar.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax_bar.set_xlim(0, 0.92)
    ax_bar.set_title("(б) Рейтинг стратегий\n(побед / (побед + поражений), 95% bootstrap ДИ)",
                     fontsize=10.5, pad=5)
    ax_bar.grid(axis="x", alpha=0.35, linestyle="--")
    ax_bar.grid(axis="y", alpha=0)
    ax_bar.tick_params(labelsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {out_path}")


def plot_outcomes_page3(out_path: Path) -> None:
    """Page-3 figure: outcome distribution + avg game length per opponent."""
    csv = ROOT / "outputs/eval_forward_masking/per_pair_results.csv"
    df  = pd.read_csv(csv)

    opp_order  = ["random", "heuristic", "greedy"]
    opp_labels = {"random": "Случайный", "heuristic": "Эвристический",
                  "greedy": "Жадный"}
    strategies = [
        ("uniform",     "Равномерная", UNIFORM_COLOR),
        ("prioritized", "PER",         PER_COLOR),
    ]

    # Pool across seeds: mean per (replay, opponent)
    pooled = (
        df.groupby(["replay_type", "opponent"])
          .agg(win_rate=("win_rate", "mean"),
               draw_rate=("draw_rate", "mean"),
               avg_moves=("avg_moves", "mean"))
          .reset_index()
    )
    pooled["loss_rate"] = (1 - pooled["win_rate"] - pooled["draw_rate"]).clip(lower=0)

    fig, (ax_stack, ax_moves) = plt.subplots(
        1, 2, figsize=(11, 3.8),
        gridspec_kw={"width_ratios": [1.6, 1], "wspace": 0.38},
    )

    # ── Left: stacked outcome bars ────────────────────────────────────────────
    n_opp = len(opp_order)
    x     = np.arange(n_opp)
    w     = 0.30
    gap   = 0.04

    win_color  = {"uniform": UNIFORM_COLOR, "prioritized": PER_COLOR}
    draw_color = {"uniform": "#A0BFDF",     "prioritized": "#F0B9A8"}
    loss_color = "#CCCCCC"

    for i, (replay, label, _) in enumerate(strategies):
        offset = (i - 0.5) * (w + gap)
        for oi, opp in enumerate(opp_order):
            row = pooled[(pooled["replay_type"] == replay) & (pooled["opponent"] == opp)]
            if row.empty:
                continue
            wr = float(row["win_rate"])
            dr = float(row["draw_rate"])
            lr = float(row["loss_rate"])

            b_win  = ax_stack.bar(oi + offset, wr,      w, bottom=0,    color=win_color[replay],  alpha=0.88, zorder=3)
            b_draw = ax_stack.bar(oi + offset, dr,      w, bottom=wr,   color=draw_color[replay], alpha=0.80, zorder=3)
            b_loss = ax_stack.bar(oi + offset, lr,      w, bottom=wr+dr, color=loss_color,         alpha=0.65, zorder=3)

    ax_stack.set_xticks(x)
    ax_stack.set_xticklabels([opp_labels[o] for o in opp_order], fontsize=10.5)
    ax_stack.set_ylabel("Доля партий", fontsize=10)
    ax_stack.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax_stack.set_ylim(0, 1.05)
    ax_stack.set_title("(а) Структура исходов по типу соперника\n(победа DQN / ничья / победа соперника)",
                       fontsize=11, pad=5)

    # Custom legend
    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor=UNIFORM_COLOR, alpha=0.88, label="Победа — Равномерная"),
        Patch(facecolor=PER_COLOR,     alpha=0.88, label="Победа — PER"),
        Patch(facecolor="#A0BFDF",     alpha=0.80, label="Ничья — Равномерная"),
        Patch(facecolor="#F0B9A8",     alpha=0.80, label="Ничья — PER"),
        Patch(facecolor=loss_color,    alpha=0.65, label="Победа соперника"),
    ]
    ax_stack.legend(handles=legend_els, fontsize=8, loc="upper right",
                    ncol=1, framealpha=0.9)
    ax_stack.tick_params(labelsize=9)

    # ── Right: average game length ────────────────────────────────────────────
    x2 = np.arange(n_opp)
    for replay, label, color in strategies:
        vals = []
        for opp in opp_order:
            row = pooled[(pooled["replay_type"] == replay) & (pooled["opponent"] == opp)]
            vals.append(float(row["avg_moves"]) if not row.empty else 0)
        ax_moves.plot(x2, vals, "o-", color=color, linewidth=2.0,
                      markersize=7, markeredgecolor="white",
                      markeredgewidth=1.0, label=label, zorder=3)
        for xi, v in zip(x2, vals):
            ax_moves.text(xi, v + 4, f"{v:.0f}", ha="center", va="bottom",
                          fontsize=8.5, color=color, fontweight="bold")

    # max_moves reference line
    ax_moves.axhline(300, color="#888", linewidth=1.2, linestyle="--", zorder=2)
    ax_moves.text(n_opp - 0.05, 302, "лимит 300", ha="right", va="bottom",
                  fontsize=8, color="#888")

    ax_moves.set_xticks(x2)
    ax_moves.set_xticklabels([opp_labels[o] for o in opp_order], fontsize=10.5)
    ax_moves.set_ylabel("Средняя длина партии (ходов)", fontsize=10)
    ax_moves.set_ylim(50, 340)
    ax_moves.set_title("(б) Средняя длина партии\nпо типу соперника",
                       fontsize=11, pad=5)
    ax_moves.legend(fontsize=9, loc="upper left")
    ax_moves.tick_params(labelsize=9)
    ax_moves.grid(axis="y", alpha=0.35, linestyle="--")
    ax_moves.grid(axis="x", alpha=0)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {out_path}")


def main():
    out = ROOT / "report/figures"
    print("\nGenerating publication figures…")
    plot_td_error(out / "fig1_td_error.png")
    plot_winrate_barplot(out / "fig2_winrate_barplot.png")
    plot_seed_stability(out / "fig3_seed_stability.png")
    plot_combined(out / "fig_combined.png")
    plot_outcomes_page3(out / "fig_outcomes.png")
    print(f"\nAll figures saved to {out}/\n")


if __name__ == "__main__":
    main()
