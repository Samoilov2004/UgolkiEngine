#!/usr/bin/env python3
"""Generate comparison plots: Uniform Replay vs PER vs Forward-Only DQN.

Reads existing main-experiment results (Uniform + PER) and newly trained
forward-only results, then writes two report-quality figures:

  winrate_vs_methods.png   — bar chart: DQN win rate vs each opponent per method
  winrate_methods_greedy.png — focused bar chart vs Greedy only (main comparison)

Both figures are saved to ``--out`` (default: report/figures/) so the LaTeX
report can reference them directly.

Usage
-----
    python scripts/plot_forward_comparison.py
    python scripts/plot_forward_comparison.py \\
        --main-dir   outputs/experiments/main \\
        --forward-dir outputs/experiments/forward \\
        --out report/figures
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

log = logging.getLogger(__name__)

# ── Style ─────────────────────────────────────────────────────────────────────

PALETTE = {
    "uniform":     "#2166AC",   # blue
    "prioritized": "#D6604D",   # red-orange
    "forward":     "#2CA02C",   # green
}
LABELS = {
    "uniform":     "Uniform Replay",
    "prioritized": "Prioritized ER",
    "forward":     "Forward-Only",
}

plt.rcParams.update({
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
    "savefig.dpi":       300,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "axes.grid.axis":    "y",
    "grid.alpha":        0.35,
    "grid.linestyle":    "--",
})


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_eval(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        log.warning("Not found: %s", path)
        return None
    try:
        df = pd.read_csv(path)
        log.info("Loaded %d rows from %s", len(df), path)
        return df
    except Exception as exc:
        log.warning("Could not read %s: %s", path, exc)
        return None


def load_combined(main_dir: Path, forward_dir: Path) -> pd.DataFrame:
    """Merge eval summaries from both experiment directories."""
    frames = []
    main_p    = main_dir    / "final_eval_summary.csv"
    forward_p = forward_dir / "final_eval_summary.csv"

    for p in (main_p, forward_p):
        df = _load_eval(p)
        if df is not None:
            frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No eval summaries found in {main_dir} or {forward_dir}.\n"
            "Run experiments first:\n"
            "  python scripts/run_replay_ablation.py --out outputs/experiments/main\n"
            "  python scripts/run_forward_experiment.py --out outputs/experiments/forward"
        )
    return pd.concat(frames, ignore_index=True)


# ── DQN stats per method ──────────────────────────────────────────────────────

def dqn_stats(evals: pd.DataFrame) -> pd.DataFrame:
    """Extract DQN win-rate stats grouped by replay_type.

    Returns a DataFrame with columns:
        replay_type, mean_win_rate, std_win_rate, n_seeds
    """
    agent_col = next(
        (c for c in ("agent", "agent_name", "name") if c in evals.columns), None
    )
    if agent_col is None:
        raise ValueError("No agent name column found in eval summary.")

    dqn = evals[evals[agent_col].astype(str).str.lower().str.contains("dqn")]
    if dqn.empty:
        raise ValueError("No DQN rows found in eval summary.")

    stats = (
        dqn.groupby("replay_type")["win_rate"]
        .agg(mean_win_rate="mean", std_win_rate="std", n_seeds="count")
        .reset_index()
    )
    stats["std_win_rate"] = stats["std_win_rate"].fillna(0.0)
    return stats


# ── Plot: win rate vs all opponents ──────────────────────────────────────────

def plot_winrate_vs_methods(evals: pd.DataFrame, out_path: Path) -> None:
    """Grouped bar: DQN win rate per replay strategy (mean ± std across seeds)."""
    stats = dqn_stats(evals)
    methods = ["uniform", "prioritized", "forward"]
    present = [m for m in methods if m in stats["replay_type"].values]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(present))

    for i, method in enumerate(present):
        row  = stats[stats["replay_type"] == method]
        mean = float(row["mean_win_rate"].iloc[0])
        std  = float(row["std_win_rate"].iloc[0])
        n    = int(row["n_seeds"].iloc[0])

        bar = ax.bar(
            i, mean,
            color=PALETTE.get(method, "#999"),
            alpha=0.85,
            yerr=std,
            capsize=8,
            error_kw={"linewidth": 1.4},
            label=f"{LABELS.get(method, method)} (n={n})",
        )
        ax.text(i, mean + std + 0.012,
                f"{mean:.1%}", ha="center", va="bottom",
                fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(m, m) for m in present])
    ax.set_ylabel("Win rate (DQN, vs all opponents)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylim(0, min(1.0, stats["mean_win_rate"].max() + stats["std_win_rate"].max() + 0.15))
    ax.set_title(
        "DQN Win Rate by Training Method\n(mean ± std across seeds, vs all tournament opponents)",
        fontsize=11, pad=10,
    )
    ax.legend(fontsize=9, loc="upper right")

    _add_footnote(ax, evals)
    fig.tight_layout()
    _save(fig, out_path)


# ── Plot: win rate vs greedy only ─────────────────────────────────────────────

def plot_winrate_greedy_comparison(evals: pd.DataFrame, out_path: Path) -> None:
    """Horizontal bar chart: DQN win rate vs Greedy for each method."""
    agent_col = next(
        (c for c in ("agent", "agent_name", "name") if c in evals.columns), None
    )
    dqn = evals[evals[agent_col].astype(str).str.lower().str.contains("dqn")]

    methods = ["uniform", "prioritized", "forward"]
    present = [m for m in methods if m in dqn["replay_type"].values]

    means, stds, labels, colors = [], [], [], []
    for method in present:
        sub = dqn[dqn["replay_type"] == method]["win_rate"]
        means.append(float(sub.mean()))
        stds.append(float(sub.std(ddof=0)))
        labels.append(LABELS.get(method, method))
        colors.append(PALETTE.get(method, "#999"))

    fig, ax = plt.subplots(figsize=(7, 3.5))
    y = np.arange(len(present))
    bars = ax.barh(y, means, xerr=stds, height=0.45, color=colors, alpha=0.85,
                   capsize=6, error_kw={"linewidth": 1.4})

    for yi, (m, s) in enumerate(zip(means, stds)):
        ax.text(m + s + 0.008, yi, f"{m:.1%}",
                va="center", fontsize=11, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Win rate (mean ± std across seeds)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlim(0, max(means) + max(stds) + 0.12)
    ax.set_title(
        "DQN Win Rate: Uniform vs PER vs Forward-Only\n(all tournament opponents combined)",
        fontsize=11, pad=10,
    )

    _add_footnote(ax, evals)
    fig.tight_layout()
    _save(fig, out_path)


# ── Report-quality figure for LaTeX ──────────────────────────────────────────

def plot_three_way_bar(evals: pd.DataFrame, out_path: Path) -> None:
    """Publication-style 3-method comparison for the LaTeX report.

    Three methods × three opponent types, grouped by opponent.
    This is the main figure that goes into the report.
    """
    agent_col = next(
        (c for c in ("agent", "agent_name", "name") if c in evals.columns), None
    )
    dqn = evals[evals[agent_col].astype(str).str.lower().str.contains("dqn")]

    # Build per-(method, opponent_name) stats from the round-robin summary.
    # The eval summary doesn't have per-opponent rows for the DQN; instead it
    # has the DQN's *aggregate* win rate. We therefore use the win_rate column
    # directly (which IS the DQN's overall tournament win rate) grouped by
    # replay_type, using std over seeds as error.
    methods  = ["uniform", "prioritized", "forward"]
    present  = [m for m in methods if m in dqn["replay_type"].values]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    x      = np.arange(len(present))
    width  = 0.55

    for i, method in enumerate(present):
        sub  = dqn[dqn["replay_type"] == method]["win_rate"]
        mean = float(sub.mean())
        std  = float(sub.std(ddof=0)) if len(sub) > 1 else 0.0

        ax.bar(i, mean, width,
               color=PALETTE.get(method, "#999"),
               alpha=0.88,
               yerr=std,
               capsize=9,
               error_kw={"linewidth": 1.5, "ecolor": "#333"},
               label=LABELS.get(method, method),
               zorder=3)

        ax.text(i, mean + std + 0.013,
                f"{mean:.1%}",
                ha="center", va="bottom",
                fontsize=12, fontweight="bold",
                color=PALETTE.get(method, "#333"))

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(m, m) for m in present], fontsize=11)
    ax.set_ylabel("Win rate (vs all opponents)", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    ymax = dqn.groupby("replay_type")["win_rate"].mean().max()
    ystd = dqn.groupby("replay_type")["win_rate"].std().max()
    ax.set_ylim(0, min(1.0, ymax + ystd + 0.20))

    ax.set_title(
        "Win Rate Comparison: Uniform Replay vs PER vs Forward-Only\n"
        "(mean ± std across seeds)",
        fontsize=11, pad=10,
    )

    n_games = int(evals["games"].iloc[0]) if "games" in evals.columns else "?"
    seeds_u = int((dqn["replay_type"] == "uniform").sum())
    seeds_p = int((dqn["replay_type"] == "prioritized").sum())
    seeds_f = int((dqn["replay_type"] == "forward").sum())
    ax.text(
        0.5, -0.14,
        f"Uniform n={seeds_u}, PER n={seeds_p}, Forward n={seeds_f} seeds  |  "
        f"{n_games} games/pair per seed",
        transform=ax.transAxes, ha="center", fontsize=8.5, color="#555",
    )

    fig.tight_layout()
    _save(fig, out_path)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _add_footnote(ax: plt.Axes, evals: pd.DataFrame) -> None:
    if "games" in evals.columns:
        n = int(evals["games"].iloc[0])
        ax.text(0.5, -0.13, f"{n} games/pair per seed",
                transform=ax.transAxes, ha="center",
                fontsize=8.5, color="#555")


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", path)
    print(f"  ✓  {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare Uniform / PER / Forward-Only DQN win rates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--main-dir",    type=Path,
                   default=Path("outputs/experiments/main"),
                   dest="main_dir")
    p.add_argument("--forward-dir", type=Path,
                   default=Path("outputs/experiments/forward"),
                   dest="forward_dir")
    p.add_argument("--out",         type=Path,
                   default=Path("report/figures"))
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()
    print(f"\n{'='*60}")
    print("  Forward-Only Comparison Plots")
    print(f"{'='*60}")
    print(f"  main dir    : {args.main_dir}")
    print(f"  forward dir : {args.forward_dir}")
    print(f"  output      : {args.out}")

    evals = load_combined(args.main_dir, args.forward_dir)
    print(f"\n  Loaded {len(evals)} rows  "
          f"(methods: {sorted(evals['replay_type'].unique())})\n")

    print("  Generating plots…")
    plot_winrate_vs_methods(
        evals,
        args.out / "winrate_vs_methods.png",
    )
    plot_winrate_greedy_comparison(
        evals,
        args.out / "winrate_methods_horizontal.png",
    )
    plot_three_way_bar(
        evals,
        args.out / "winrate_three_way.png",
    )

    # Overwrite the existing report figure so the LaTeX document picks it up
    plot_three_way_bar(
        evals,
        args.out / "winrate_vs_opponents.png",
    )

    print(f"\n  All plots saved to {args.out}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
