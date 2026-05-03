#!/usr/bin/env python3
"""
eval_all_seeds.py — evaluate all 5 training seeds (Uniform + PER).

Runs ONLY evaluation on existing checkpoints.
Generates:
  - per-seed bootstrap win rates
  - Figure 1: seed variance boxplot/strip
  - Figure 3 extension: draw rate + game length per replay type

Usage
-----
    PYTHONPATH=src python scripts/eval_all_seeds.py \
        --games 500 \
        --device auto \
        --out outputs/eval_all_seeds
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    stream=sys.stderr,
)
log = logging.getLogger(__name__)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

UNIFORM_COLOR     = "#2166AC"
PRIORITIZED_COLOR = "#D6604D"

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})

# ── checkpoint registry ───────────────────────────────────────────────────────

def _build_checkpoint_map() -> dict[str, dict]:
    """Return available checkpoints: {run_key: {replay_type, seed, path}}"""
    registry = {}
    # seeds 1-2: main experiment
    for replay in ("uniform", "prioritized"):
        for seed in (1, 2):
            key  = f"{replay}_s{seed}"
            path = Path(f"outputs/experiments/main/{replay}_seed_{seed}/models/dqn_latest.pt")
            if path.exists():
                registry[key] = {"replay_type": replay, "seed": seed, "path": path}
    # seeds 3-5: extra seeds
    for replay in ("uniform", "prioritized"):
        for seed in (3, 4, 5):
            key  = f"{replay}_s{seed}"
            path = Path(f"outputs/experiments/extra_seeds/{replay}_seed_{seed}/models/dqn_latest.pt")
            if path.exists():
                registry[key] = {"replay_type": replay, "seed": seed, "path": path}
    return registry


# ── bootstrap ─────────────────────────────────────────────────────────────────

def bootstrap_ci(outcomes: np.ndarray, n: int = 5_000, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    if len(outcomes) == 0:
        return 0.0, 0.0, 0.0
    boot = np.array([rng.choice(outcomes, size=len(outcomes), replace=True).mean()
                     for _ in range(n)])
    return float(outcomes.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


# ── evaluation ───────────────────────────────────────────────────────────────

def run_all_seed_eval(
    ckpt_map: dict,
    games: int,
    max_moves: int,
    device: str,
) -> pd.DataFrame:
    from corners_rl.agents.dqn_agent import DQNAgent
    from corners_rl.agents.random_agent import RandomAgent
    from corners_rl.agents.greedy_agent import GreedyAgent
    from corners_rl.agents.heuristic_agent import HeuristicAgent
    from corners_rl.evaluation.evaluate import evaluate_match

    baselines = {
        "random":    RandomAgent(name="random",   seed=0),
        "greedy":    GreedyAgent(name="greedy"),
        "heuristic": HeuristicAgent(name="heuristic"),
    }

    rng  = np.random.default_rng(7)
    rows = []
    total = len(ckpt_map) * len(baselines)
    done  = 0

    for run_key, info in sorted(ckpt_map.items()):
        replay_type = info["replay_type"]
        seed        = info["seed"]
        path        = info["path"]

        agent = DQNAgent.load(path, device=device, epsilon=0.0)
        agent._name = run_key
        agent.name  = run_key
        log.info("Loaded %s", run_key)

        for base_key, base_agent in baselines.items():
            done += 1
            t0 = time.time()
            log.info("  [%d/%d]  %s vs %s  (%d games)…",
                     done, total, run_key, base_key, games)
            df = evaluate_match(agent, base_agent, games=games,
                                max_moves=max_moves, seed=done * 100)
            wins  = (df["winner_agent"] == run_key).values.astype(int)
            draws = df["draw"].values.astype(int)
            wr, lo, hi = bootstrap_ci(wins, rng=rng)
            dr = float(draws.mean())

            rows.append({
                "run_key":     run_key,
                "replay_type": replay_type,
                "seed":        seed,
                "baseline":    base_key,
                "n_games":     games,
                "win_rate":    wr,
                "win_ci_lo":   lo,
                "win_ci_hi":   hi,
                "draw_rate":   dr,
                "avg_moves":   float(df["moves"].mean()),
            })
            log.info("    win=%.1f%% [%.1f,%.1f]  %.1fs",
                     wr*100, lo*100, hi*100, time.time()-t0)

    return pd.DataFrame(rows)


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_seed_variance(df: pd.DataFrame, out_dir: Path) -> None:
    """Figure 1: Boxplot/strip of per-seed overall win rate for Uniform vs PER."""
    # compute per-seed overall win rate (mean across baselines)
    per_seed = (
        df.groupby(["run_key", "replay_type", "seed"])["win_rate"]
        .mean()
        .reset_index()
        .rename(columns={"win_rate": "overall_wr"})
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    for xi, (replay, color, label) in enumerate([
        ("uniform",     UNIFORM_COLOR,     "Uniform Replay"),
        ("prioritized", PRIORITIZED_COLOR, "Prioritized ER"),
    ]):
        sub = per_seed[per_seed["replay_type"] == replay]["overall_wr"].values
        if len(sub) == 0:
            continue
        # box
        bp = ax.boxplot(
            sub,
            positions=[xi],
            widths=0.35,
            patch_artist=True,
            boxprops=dict(facecolor=color, alpha=0.45, linewidth=1.2),
            medianprops=dict(color="#111111", linewidth=2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            flierprops=dict(marker="o", markersize=5, markerfacecolor=color),
            zorder=3,
        )
        # strip (individual seeds)
        jitter = np.random.default_rng(xi).uniform(-0.07, 0.07, len(sub))
        ax.scatter(
            np.full(len(sub), xi) + jitter, sub,
            color=color, s=55, zorder=5, alpha=0.9,
            edgecolors="white", linewidths=0.6,
        )
        # annotate each seed
        seeds = per_seed[per_seed["replay_type"] == replay]["seed"].values
        for j, (s, v) in enumerate(zip(seeds, sub)):
            ax.annotate(
                f"s{s}",
                xy=(xi + jitter[j], v),
                xytext=(8, 0),
                textcoords="offset points",
                fontsize=8, color="#444444", va="center",
            )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Uniform Replay", "Prioritized ER"], fontsize=11)
    ax.set_ylabel("Overall win rate (mean across baselines)", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("Win Rate Distribution Across Training Seeds\n(each point = one seed; box = IQR, line = median)",
                 fontsize=11, pad=10)

    n_uniform = (per_seed["replay_type"] == "uniform").sum()
    n_per     = (per_seed["replay_type"] == "prioritized").sum()
    ax.text(0.5, -0.13,
            f"Uniform: n={n_uniform} seeds  |  PER: n={n_per} seeds  |  "
            f"{df['n_games'].iloc[0]} games/pair per seed",
            transform=ax.transAxes, ha="center", fontsize=8.5, color="#555555")

    fig.tight_layout()
    for fmt in ("png", "pdf"):
        fig.savefig(out_dir / f"seed_variance.{fmt}", bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved: seed_variance.png / .pdf")


def plot_game_length_by_replay(df: pd.DataFrame, out_dir: Path) -> None:
    """Figure 3: Average game length and draw rate per replay type × baseline."""
    baselines = ["random", "greedy", "heuristic"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    x   = np.arange(len(baselines))
    w   = 0.32

    for i, (replay, color, label) in enumerate([
        ("uniform",     UNIFORM_COLOR,     "Uniform"),
        ("prioritized", PRIORITIZED_COLOR, "PER"),
    ]):
        sub = df[df["replay_type"] == replay]
        moves_vals = [sub[sub["baseline"]==b]["avg_moves"].mean() for b in baselines]
        draw_vals  = [sub[sub["baseline"]==b]["draw_rate"].mean()  for b in baselines]
        offset = (i - 0.5) * w
        ax1.bar(x + offset, moves_vals, w, color=color, alpha=0.85, label=label,
                edgecolor="white", linewidth=0.6)
        ax2.bar(x + offset, draw_vals,  w, color=color, alpha=0.85, label=label,
                edgecolor="white", linewidth=0.6, hatch="//" if replay=="prioritized" else "")

    for ax, title, ylabel in [
        (ax1, "Average Game Length per Opponent",   "Avg moves"),
        (ax2, "Draw Rate per Opponent",              "Draw rate"),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels([b.capitalize() for b in baselines], fontsize=10.5)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10.5)
        ax.legend(fontsize=10)

    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    fig.tight_layout()
    for fmt in ("png", "pdf"):
        fig.savefig(out_dir / f"game_length_drawrate.{fmt}", bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved: game_length_drawrate.png / .pdf")


# ── statistical tests ─────────────────────────────────────────────────────────

def print_stat_summary(df: pd.DataFrame, stats_path: Path) -> None:
    per_seed = (
        df.groupby(["run_key", "replay_type", "seed"])["win_rate"]
        .mean()
        .reset_index()
        .rename(columns={"win_rate": "overall_wr"})
    )
    uni = per_seed[per_seed["replay_type"] == "uniform"]["overall_wr"].values
    per = per_seed[per_seed["replay_type"] == "prioritized"]["overall_wr"].values

    rng = np.random.default_rng(0)
    n_boot = 20_000
    diff_dist = np.array([
        rng.choice(uni, replace=True).mean() - rng.choice(per, replace=True).mean()
        for _ in range(n_boot)
    ])
    diff_mean = float(diff_dist.mean())
    diff_lo   = float(np.percentile(diff_dist, 2.5))
    diff_hi   = float(np.percentile(diff_dist, 97.5))
    ci_zero   = diff_lo <= 0 <= diff_hi

    result = {
        "uniform_mean":    float(uni.mean()),
        "uniform_seeds":   uni.tolist(),
        "per_mean":        float(per.mean()),
        "per_seeds":       per.tolist(),
        "diff_mean":       diff_mean,
        "diff_ci_lo":      diff_lo,
        "diff_ci_hi":      diff_hi,
        "ci_contains_zero": ci_zero,
        "n_seeds_uniform": int(len(uni)),
        "n_seeds_per":     int(len(per)),
    }
    with open(stats_path, "w") as f:
        json.dump(result, f, indent=2)

    sep = "=" * 62
    print(f"\n{sep}")
    print("  PER-SEED VARIANCE SUMMARY")
    print(sep)
    print(f"\n  Uniform Replay  (n={len(uni)} seeds)")
    print(f"    seeds: {', '.join(f'{v:.1%}' for v in sorted(uni))}")
    print(f"    mean  = {uni.mean():.1%}   std = {uni.std():.1%}")
    print(f"\n  Prioritized ER  (n={len(per)} seeds)")
    print(f"    seeds: {', '.join(f'{v:.1%}' for v in sorted(per))}")
    print(f"    mean  = {per.mean():.1%}   std = {per.std():.1%}")
    print(f"\n  Bootstrap diff CI (Uniform − PER):")
    print(f"    {diff_mean:+.1%}  [{diff_lo:+.1%}, {diff_hi:+.1%}]")
    if ci_zero:
        print("    ⚠  CI contains 0 — superiority NOT confirmed across seeds")
    else:
        print("    ✓  CI excludes 0 — Uniform significantly higher across seeds")
    print(f"\n{sep}\n")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--games",     type=int,  default=500)
    p.add_argument("--max-moves", type=int,  default=300, dest="max_moves")
    p.add_argument("--device",    type=str,  default="auto")
    p.add_argument("--out",       type=Path, default=Path("outputs/eval_all_seeds"))
    p.add_argument("--skip-eval", action="store_true", dest="skip_eval")
    args = p.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    ckpt_map = _build_checkpoint_map()
    log.info("Found %d checkpoints: %s", len(ckpt_map), sorted(ckpt_map))

    if not ckpt_map:
        log.error("No checkpoints found — check outputs/experiments/ directories")
        sys.exit(1)

    csv_path = out_dir / "per_seed_eval.csv"
    if args.skip_eval and csv_path.exists():
        log.info("Loading existing eval from %s", csv_path)
        df = pd.read_csv(csv_path)
    else:
        # resolve device
        device = args.device
        if device == "auto":
            try:
                import torch
                device = "mps" if torch.backends.mps.is_available() else "cpu"
            except Exception:
                device = "cpu"
        df = run_all_seed_eval(ckpt_map, games=args.games,
                               max_moves=args.max_moves, device=device)
        df.to_csv(csv_path, index=False)
        log.info("Eval saved → %s", csv_path)

    plot_seed_variance(df, fig_dir)
    plot_game_length_by_replay(df, fig_dir)
    print_stat_summary(df, out_dir / "seed_variance_stats.json")

    print(f"  Figures → {fig_dir}/")
    print(f"  CSV     → {csv_path}")


if __name__ == "__main__":
    main()
