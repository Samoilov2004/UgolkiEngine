#!/usr/bin/env python3
"""Generate synthetic figures for the alpha-ablation hypothesis paper.

Simulates realistic DQN training curves and win-rate results for
PER with alpha in {0.3, 0.6, 0.9} vs Uniform Replay baseline.
All numbers are plausible but SYNTHETIC — for layout preview only.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
OUT  = ROOT / "report" / "hypothesis_figures"
OUT.mkdir(parents=True, exist_ok=True)

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

PALETTE = {
    "uniform": "#2166AC",
    "per_03":  "#74C476",   # light green — mild prioritization
    "per_06":  "#D6604D",   # red-orange  — current
    "per_09":  "#7B2D8B",   # purple      — aggressive
}
LABELS = {
    "uniform": "Равномерная",
    "per_03":  r"PER $\alpha{=}0{,}3$",
    "per_06":  r"PER $\alpha{=}0{,}6$",
    "per_09":  r"PER $\alpha{=}0{,}9$",
}

# ── Synthetic TD-error generator ──────────────────────────────────────────────

def td_curve(episodes: np.ndarray, peak: float, peak_ep: int,
             floor: float, decay: float, noise: float,
             rng: np.random.Generator) -> np.ndarray:
    """Exponential rise + decay, ending near `floor`, with Gaussian noise."""
    rise  = np.exp(-((episodes - peak_ep) ** 2) / (2 * (peak_ep * 0.4) ** 2))
    decay_part = floor + (peak - floor) * np.exp(-decay * np.maximum(episodes - peak_ep, 0))
    curve = np.where(episodes <= peak_ep,
                     floor + (peak - floor) * (episodes / peak_ep) ** 1.4,
                     decay_part)
    curve += rng.normal(0, noise * peak, size=len(episodes))
    return np.clip(curve, 0, None)


def smooth(arr: np.ndarray, w: int = 30) -> np.ndarray:
    import pandas as pd
    return pd.Series(arr).rolling(w, min_periods=1).mean().values


# ── Figure 1: TD-error dynamics (4 strategies) ───────────────────────────────

def plot_td_ablation(out_path: Path) -> None:
    episodes = np.arange(1, 1501)
    rng = np.random.default_rng(42)

    # Tuned so curves look like real RL training
    configs = {
        "uniform": dict(peak=3.8,  peak_ep=120, floor=0.6, decay=0.008, noise=0.08),
        "per_03":  dict(peak=7.2,  peak_ep=180, floor=0.5, decay=0.007, noise=0.09),
        "per_06":  dict(peak=12.7, peak_ep=280, floor=0.7, decay=0.006, noise=0.10),
        "per_09":  dict(peak=21.4, peak_ep=420, floor=1.4, decay=0.004, noise=0.13),
    }

    fig, ax = plt.subplots(figsize=(8, 3.6))

    for key, cfg in configs.items():
        seed1 = td_curve(episodes, **cfg, rng=rng)
        seed2 = td_curve(episodes, **cfg, rng=rng) * rng.uniform(0.88, 1.12)
        m1, m2 = smooth(seed1), smooth(seed2)
        mean = (m1 + m2) / 2
        sem  = np.abs(m1 - m2) / 2 * 1.96
        ax.plot(episodes, mean, color=PALETTE[key], linewidth=2.1,
                label=LABELS[key])
        ax.fill_between(episodes, mean - sem, mean + sem,
                        color=PALETTE[key], alpha=0.15)

    ax.set_xlabel("Эпизод обучения", fontsize=11)
    ax.set_ylabel("Средняя абсолютная TD-ошибка", fontsize=11)
    ax.set_title("Динамика TD-ошибки при различных значениях α\n"
                 "(скользящее среднее w=30, полоса — 95% ДИ по инициализациям)",
                 fontsize=11, pad=6)
    ax.legend(fontsize=10, loc="upper right")
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {out_path}")


# ── Figure 2: Win-rate ranking (combined: TD left, ranking right) ─────────────

def plot_combined_hypothesis(out_path: Path) -> None:
    from matplotlib.gridspec import GridSpec

    # Hypothetical win rates (excl. draws): higher alpha → worse at this budget
    # Uniform is our reference (71.8% real). Others synthesised plausibly.
    win_stats = {
        #             mean    ci_lo  ci_hi   (wins / (wins+losses))
        "uniform": (0.718,  0.705, 0.731),
        "per_03":  (0.694,  0.680, 0.708),   # mild PER ≈ near-uniform
        "per_06":  (0.609,  0.595, 0.623),   # real result
        "per_09":  (0.531,  0.516, 0.546),   # too aggressive → unstable
    }

    episodes = np.arange(1, 1501)
    rng = np.random.default_rng(42)

    td_configs = {
        "uniform": dict(peak=3.8,  peak_ep=120, floor=0.6, decay=0.008, noise=0.08),
        "per_03":  dict(peak=7.2,  peak_ep=180, floor=0.5, decay=0.007, noise=0.09),
        "per_06":  dict(peak=12.7, peak_ep=280, floor=0.7, decay=0.006, noise=0.10),
        "per_09":  dict(peak=21.4, peak_ep=420, floor=1.4, decay=0.004, noise=0.13),
    }

    fig = plt.figure(figsize=(11, 3.8))
    gs  = GridSpec(1, 2, figure=fig,
                   width_ratios=[1.4, 1], wspace=0.36)
    ax_td  = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1])

    # ── Left: TD error ────────────────────────────────────────────────────────
    for key, cfg in td_configs.items():
        s1 = td_curve(episodes, **cfg, rng=rng)
        s2 = td_curve(episodes, **cfg, rng=rng) * rng.uniform(0.88, 1.12)
        m1, m2 = smooth(s1), smooth(s2)
        mean = (m1 + m2) / 2
        sem  = np.abs(m1 - m2) / 2 * 1.96
        ax_td.plot(episodes, mean, color=PALETTE[key], linewidth=2.0,
                   label=LABELS[key])
        ax_td.fill_between(episodes, mean - sem, mean + sem,
                           color=PALETTE[key], alpha=0.15)

    ax_td.set_xlabel("Эпизод обучения", fontsize=10)
    ax_td.set_ylabel("Ср. абс. TD-ошибка", fontsize=10)
    ax_td.set_title("(а) Динамика TD-ошибки", fontsize=11, pad=5)
    ax_td.legend(fontsize=8.5, loc="upper right")
    ax_td.tick_params(labelsize=9)
    ax_td.set_ylim(bottom=0)

    # ── Right: horizontal ranking bar ─────────────────────────────────────────
    sorted_items = sorted(win_stats.items(), key=lambda kv: kv[1][0], reverse=True)
    y     = np.arange(len(sorted_items))
    means = [v[0] for _, v in sorted_items]
    lo_e  = [v[0] - v[1] for _, v in sorted_items]
    hi_e  = [v[2] - v[0] for _, v in sorted_items]
    names = [LABELS[k] for k, _ in sorted_items]
    cols  = [PALETTE[k] for k, _ in sorted_items]

    ax_bar.barh(y, means, 0.50, xerr=[lo_e, hi_e],
                color=cols, alpha=0.88,
                capsize=6, error_kw={"linewidth": 1.5, "ecolor": "#222"},
                zorder=3)
    for yi, (m, eh, c) in enumerate(zip(means, hi_e, cols)):
        ax_bar.text(m + eh + 0.008, yi, f"{m:.1%}",
                    va="center", fontsize=9, fontweight="bold", color=c)

    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(names, fontsize=9.5)
    ax_bar.set_xlabel("Доля побед (без учёта ничьих)", fontsize=10)
    ax_bar.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax_bar.set_xlim(0, 0.88)
    ax_bar.set_title("(б) Рейтинг стратегий\n(95% bootstrap ДИ, гипотетические данные)",
                     fontsize=10.5, pad=5)
    ax_bar.grid(axis="x", alpha=0.35, linestyle="--")
    ax_bar.grid(axis="y", alpha=0)
    ax_bar.tick_params(labelsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {out_path}")


if __name__ == "__main__":
    print("\nGenerating hypothesis figures…")
    plot_td_ablation(OUT / "hyp_td_ablation.png")
    plot_combined_hypothesis(OUT / "hyp_combined.png")
    print(f"\nSaved to {OUT}/\n")
