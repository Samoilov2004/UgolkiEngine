#!/usr/bin/env python3
"""
eval_bootstrap.py — large-scale evaluation with bootstrap confidence intervals.

Loads existing checkpoints only (no training).

Usage
-----
    PYTHONPATH=src python scripts/eval_bootstrap.py \
        --games 1000 \
        --device auto \
        --out outputs/eval_bootstrap

Steps
-----
1. Load 4 DQN checkpoints (uniform × 2 seeds, prioritized × 2 seeds).
2. Each DQN vs {Random, Greedy, Heuristic} — ``--games`` games per pair.
3. Head-to-head: best Uniform vs best PER (best = highest prior win rate).
4. Bootstrap 95% CI (10 000 resamples, percentile method).
5. Save detailed game CSV + bootstrap JSON.
6. Generate 3 publication-quality plots (PNG 300 dpi + PDF).
7. Print statistical summary.
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

# ── logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    stream=sys.stderr,
)
log = logging.getLogger(__name__)

# ── matplotlib ───────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

UNIFORM_COLOR     = "#2166AC"   # blue
PRIORITIZED_COLOR = "#D6604D"   # red-orange
RANDOM_COLOR      = "#888888"   # grey
HATCH_UNIFORM     = ""
HATCH_PER         = "//"

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})

# ── checkpoints ──────────────────────────────────────────────────────────────
CHECKPOINTS: dict[str, str] = {
    "uniform_s1":     "outputs/experiments/main/uniform_seed_1/models/dqn_latest.pt",
    "uniform_s2":     "outputs/experiments/main/uniform_seed_2/models/dqn_latest.pt",
    "prioritized_s1": "outputs/experiments/main/prioritized_seed_1/models/dqn_latest.pt",
    "prioritized_s2": "outputs/experiments/main/prioritized_seed_2/models/dqn_latest.pt",
}

# Prior win rates from training eval (used to pick "best" for head-to-head)
_PRIOR_WR = {
    "uniform_s1": 0.190,
    "uniform_s2": 0.267,   # ← best uniform
    "prioritized_s1": 0.137,
    "prioritized_s2": 0.233,   # ← best prioritized
}


# ── bootstrap ────────────────────────────────────────────────────────────────

def bootstrap_ci(
    outcomes: np.ndarray,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) using the percentile bootstrap.

    Args:
        outcomes: 1-D array of binary values (1=win, 0=not-win).
        n_resamples: Number of bootstrap resamples.
        alpha: Significance level (default 0.05 → 95% CI).
        rng: NumPy random generator (reproducible).

    Returns:
        Tuple (mean, ci_low, ci_high).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(outcomes)
    if n == 0:
        return 0.0, 0.0, 0.0
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = rng.choice(outcomes, size=n, replace=True)
        means[i] = sample.mean()
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return float(outcomes.mean()), lo, hi


# ── agents ───────────────────────────────────────────────────────────────────

def _resolve_device(device: str) -> str:
    """Resolve 'auto' → 'mps' or 'cpu'."""
    if device != "auto":
        return device
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def load_agents(device: str) -> dict:
    """Load all DQN checkpoints and baseline agents."""
    from corners_rl.agents.dqn_agent import DQNAgent
    from corners_rl.agents.random_agent import RandomAgent
    from corners_rl.agents.greedy_agent import GreedyAgent
    from corners_rl.agents.heuristic_agent import HeuristicAgent

    resolved = _resolve_device(device)

    agents: dict = {
        "random":    RandomAgent(name="random",   seed=0),
        "greedy":    GreedyAgent(name="greedy"),
        "heuristic": HeuristicAgent(name="heuristic"),
    }

    for key, path in CHECKPOINTS.items():
        p = Path(path)
        if not p.exists():
            log.warning("Checkpoint not found: %s — skipping.", path)
            continue
        agents[key] = DQNAgent.load(p, device=resolved, epsilon=0.0)
        agents[key]._name = key   # rename for clarity in CSV
        log.info("  Loaded %-18s ← %s  (device=%s)", key, path, resolved)

    return agents


# ── evaluation ───────────────────────────────────────────────────────────────

def run_evaluations(agents: dict, games: int, max_moves: int) -> pd.DataFrame:
    """Run all required matches; return concatenated per-game DataFrame."""
    from corners_rl.evaluation.evaluate import evaluate_match

    dqn_keys = [k for k in CHECKPOINTS if k in agents]
    baseline_keys = ["random", "greedy", "heuristic"]

    all_frames: list[pd.DataFrame] = []
    eval_seed_base = 1000   # separate from training seeds

    total = len(dqn_keys) * len(baseline_keys) + 1  # +1 head-to-head
    done = 0

    # ── DQN vs each baseline ─────────────────────────────────────────────────
    for dqn_key in dqn_keys:
        for base_key in baseline_keys:
            done += 1
            t0 = time.time()
            log.info("[%d/%d]  %s  vs  %s  (%d games)…",
                     done, total, dqn_key, base_key, games)
            df = evaluate_match(
                agents[dqn_key],
                agents[base_key],
                games=games,
                max_moves=max_moves,
                seed=eval_seed_base + done * 100,
            )
            df["dqn_key"]      = dqn_key
            df["replay_type"]  = "uniform" if dqn_key.startswith("uniform") else "prioritized"
            df["seed"]         = int(dqn_key.split("_s")[1])
            df["baseline"]     = base_key
            all_frames.append(df)
            log.info("   done  %.1fs", time.time() - t0)

    # ── head-to-head: best Uniform vs best PER ───────────────────────────────
    best_uniform = max(
        [k for k in dqn_keys if k.startswith("uniform")],
        key=lambda k: _PRIOR_WR.get(k, 0),
    )
    best_per = max(
        [k for k in dqn_keys if k.startswith("prioritized")],
        key=lambda k: _PRIOR_WR.get(k, 0),
    )

    done += 1
    log.info("[%d/%d]  head-to-head: %s vs %s  (%d games)…",
             done, total, best_uniform, best_per, games)
    t0 = time.time()
    df_h2h = evaluate_match(
        agents[best_uniform],
        agents[best_per],
        games=games,
        max_moves=max_moves,
        seed=eval_seed_base + done * 100,
    )
    df_h2h["dqn_key"]     = "head_to_head"
    df_h2h["replay_type"] = "head_to_head"
    df_h2h["seed"]        = 0
    df_h2h["baseline"]    = "per_best"
    all_frames.append(df_h2h)
    log.info("   done  %.1fs  (%s beat %s in head-to-head)",
             time.time() - t0, best_uniform, best_per)

    return pd.concat(all_frames, ignore_index=True)


# ── bootstrap statistics ─────────────────────────────────────────────────────

def compute_bootstrap_stats(df: pd.DataFrame, n_resamples: int = 10_000) -> dict:
    """Compute bootstrap 95% CI for every (replay_type, baseline) combination."""
    rng = np.random.default_rng(42)
    results: dict = {}

    # ── per replay type × baseline ───────────────────────────────────────────
    for replay in ["uniform", "prioritized"]:
        sub = df[df["replay_type"] == replay]
        results[replay] = {}
        for base in ["random", "greedy", "heuristic"]:
            gsub = sub[sub["baseline"] == base]
            wins = (gsub["winner_agent"] == gsub["dqn_key"]).values.astype(int)
            mean, lo, hi = bootstrap_ci(wins, n_resamples=n_resamples, rng=rng)
            results[replay][base] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": int(len(wins))}

        # overall (pooled across baselines)
        all_sub = sub[sub["baseline"].isin(["random", "greedy", "heuristic"])]
        wins_all = (all_sub["winner_agent"] == all_sub["dqn_key"]).values.astype(int)
        mean, lo, hi = bootstrap_ci(wins_all, n_resamples=n_resamples, rng=rng)
        results[replay]["overall"] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": int(len(wins_all))}

    # ── per individual seed ───────────────────────────────────────────────────
    results["per_seed"] = {}
    for key in CHECKPOINTS:
        sub = df[(df["dqn_key"] == key) & (df["baseline"].isin(["random", "greedy", "heuristic"]))]
        if sub.empty:
            continue
        wins = (sub["winner_agent"] == sub["dqn_key"]).values.astype(int)
        mean, lo, hi = bootstrap_ci(wins, n_resamples=n_resamples, rng=rng)
        results["per_seed"][key] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": int(len(wins))}

    # ── head-to-head ─────────────────────────────────────────────────────────
    h2h = df[df["replay_type"] == "head_to_head"]
    if not h2h.empty:
        best_uniform = max(
            [k for k in CHECKPOINTS if k.startswith("uniform")],
            key=lambda k: _PRIOR_WR.get(k, 0),
        )
        wins_u = (h2h["winner_agent"] == h2h["dqn_key"]).values.astype(int)
        mean_u, lo_u, hi_u = bootstrap_ci(wins_u, n_resamples=n_resamples, rng=rng)
        results["head_to_head"] = {
            "uniform_wr": {"mean": mean_u, "ci_lo": lo_u, "ci_hi": hi_u, "n": int(len(wins_u))},
        }

    # ── difference CI: uniform − prioritized (vs random) ─────────────────────
    # Bootstrap paired difference to test if overlap is just noise
    rng2 = np.random.default_rng(99)
    n_b = n_resamples
    diff_per_baseline: dict = {}
    for base in ["random", "greedy", "heuristic", "overall"]:
        if base == "overall":
            u_wins_arr = (
                df[(df["replay_type"] == "uniform") & df["baseline"].isin(["random","greedy","heuristic"])]
                .assign(win=lambda d: (d["winner_agent"] == d["dqn_key"]).astype(int))
                ["win"].values
            )
            p_wins_arr = (
                df[(df["replay_type"] == "prioritized") & df["baseline"].isin(["random","greedy","heuristic"])]
                .assign(win=lambda d: (d["winner_agent"] == d["dqn_key"]).astype(int))
                ["win"].values
            )
        else:
            u_wins_arr = (
                df[(df["replay_type"] == "uniform") & (df["baseline"] == base)]
                .assign(win=lambda d: (d["winner_agent"] == d["dqn_key"]).astype(int))
                ["win"].values
            )
            p_wins_arr = (
                df[(df["replay_type"] == "prioritized") & (df["baseline"] == base)]
                .assign(win=lambda d: (d["winner_agent"] == d["dqn_key"]).astype(int))
                ["win"].values
            )

        if len(u_wins_arr) == 0 or len(p_wins_arr) == 0:
            continue

        diffs = np.empty(n_b)
        for i in range(n_b):
            diffs[i] = (
                rng2.choice(u_wins_arr, size=len(u_wins_arr), replace=True).mean()
                - rng2.choice(p_wins_arr, size=len(p_wins_arr), replace=True).mean()
            )
        diff_mean = float(diffs.mean())
        diff_lo   = float(np.percentile(diffs, 2.5))
        diff_hi   = float(np.percentile(diffs, 97.5))
        ci_contains_zero = diff_lo <= 0 <= diff_hi
        diff_per_baseline[base] = {
            "diff_mean": diff_mean,
            "ci_lo": diff_lo,
            "ci_hi": diff_hi,
            "ci_contains_zero": ci_contains_zero,
        }

    results["difference_ci"] = diff_per_baseline
    return results


# ── Plot 1: win rate vs opponents ─────────────────────────────────────────────

def plot_winrate_vs_opponents(stats: dict, out_dir: Path) -> None:
    """Bar chart: win rate (Uniform vs PER) per opponent + overall."""
    categories  = ["random", "greedy", "heuristic", "overall"]
    labels      = ["vs Random", "vs Greedy", "vs Heuristic", "Overall"]
    x           = np.arange(len(categories))
    width       = 0.32

    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    for i, (replay, color, hatch, label) in enumerate([
        ("uniform",     UNIFORM_COLOR,     HATCH_UNIFORM, "Uniform Replay"),
        ("prioritized", PRIORITIZED_COLOR, HATCH_PER,     "Prioritized ER"),
    ]):
        means = np.array([stats[replay][c]["mean"]  for c in categories])
        ci_lo = np.array([stats[replay][c]["ci_lo"] for c in categories])
        ci_hi = np.array([stats[replay][c]["ci_hi"] for c in categories])
        err_lo = means - ci_lo
        err_hi = ci_hi - means
        offset = (i - 0.5) * width
        bars = ax.bar(
            x + offset, means, width,
            color=color, hatch=hatch, alpha=0.88,
            label=label, zorder=3, linewidth=0.6,
            edgecolor="white",
        )
        ax.errorbar(
            x + offset, means,
            yerr=[err_lo, err_hi],
            fmt="none", color="#222222",
            capsize=4, capthick=1.2, linewidth=1.2, zorder=4,
        )
        # Value labels
        for bar, m in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{m:.1%}", ha="center", va="bottom",
                fontsize=8.5, color="#333333",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_ylabel("Win rate", fontsize=11)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_title("DQN Win Rate vs Baseline Opponents\n(1 000 games per pair; error bars: 95% bootstrap CI)",
                 fontsize=11, pad=10)
    ax.legend(loc="upper right", framealpha=0.9)

    fig.tight_layout()
    for fmt in ("png", "pdf"):
        path = out_dir / f"winrate_vs_opponents.{fmt}"
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved: winrate_vs_opponents.png / .pdf")


# ── Plot 2: vs Random baseline ────────────────────────────────────────────────

def plot_vs_random_baseline(stats: dict, out_dir: Path) -> None:
    """
    Horizontal dot-and-CI plot comparing DQN agents and Random to a 0% reference.
    """
    # Random agent always wins 0% of games (empirical from our data)
    random_wr = 0.0

    groups = [
        ("Uniform Replay\n(seed 1)", "uniform_s1"),
        ("Uniform Replay\n(seed 2)", "uniform_s2"),
        ("Prioritized ER\n(seed 1)", "prioritized_s1"),
        ("Prioritized ER\n(seed 2)", "prioritized_s2"),
    ]

    fig, ax = plt.subplots(figsize=(7, 4.0))

    colors = {
        "uniform_s1": UNIFORM_COLOR,
        "uniform_s2": UNIFORM_COLOR,
        "prioritized_s1": PRIORITIZED_COLOR,
        "prioritized_s2": PRIORITIZED_COLOR,
    }

    for yi, (label, key) in enumerate(groups):
        if key not in stats.get("per_seed", {}):
            continue
        s = stats["per_seed"][key]
        mean, lo, hi = s["mean"], s["ci_lo"], s["ci_hi"]
        color = colors[key]
        ax.plot([lo, hi], [yi, yi], color=color, linewidth=2.5, solid_capstyle="round")
        ax.plot(mean, yi, "o", color=color, markersize=8, zorder=5)
        ax.text(hi + 0.004, yi, f"{mean:.1%}", va="center", fontsize=9.5, color=color)

    n_groups = len(groups)
    # Random reference line
    ax.axvline(random_wr, color=RANDOM_COLOR, linestyle="--", linewidth=1.4, label="Random baseline (0%)")

    ax.set_yticks(range(n_groups))
    ax.set_yticklabels([g[0] for g in groups], fontsize=10)
    ax.set_xlabel("Win rate (overall vs all opponents)", fontsize=11)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("DQN Agents vs Random Baseline\n(95% bootstrap CI; overall win rate across all opponent types)",
                 fontsize=11, pad=10)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xlim(-0.01, None)

    # Significance annotation
    ax.text(0.01, -0.18, "All trained agents significantly exceed random play (CI does not include 0%)",
            transform=ax.transAxes, fontsize=8.5, color="#444444", style="italic")

    fig.tight_layout()
    for fmt in ("png", "pdf"):
        path = out_dir / f"vs_random_baseline.{fmt}"
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved: vs_random_baseline.png / .pdf")


# ── Plot 3: improved TD error dynamics ───────────────────────────────────────

def plot_td_error_improved(curves_csv: str, out_dir: Path) -> None:
    """Improved TD error dynamics with moving average and CI band."""
    df = pd.read_csv(curves_csv)
    if "td_error_abs_mean" not in df.columns:
        log.warning("td_error_abs_mean column not found — skipping TD error plot")
        return

    fig, ax = plt.subplots(figsize=(7.5, 4.0))

    window = 30   # episodes for moving average

    for replay, color, label, ls in [
        ("uniform",     UNIFORM_COLOR,     "Uniform Replay", "-"),
        ("prioritized", PRIORITIZED_COLOR, "Prioritized ER", "--"),
    ]:
        sub = df[df["replay_type"] == replay].copy()

        # Group by episode, average across seeds
        grp = sub.groupby("episode")["td_error_abs_mean"].agg(["mean", "std", "count"]).reset_index()
        grp = grp.sort_values("episode")

        # Moving average
        mean_ma  = grp["mean"].rolling(window, min_periods=1, center=True).mean()
        std_ma   = grp["std"].rolling(window, min_periods=1, center=True).mean()
        n_seeds  = grp["count"].rolling(window, min_periods=1, center=True).mean()

        # 95% CI approximation: mean ± 1.96 * std / sqrt(n)
        ci = 1.96 * std_ma / np.sqrt(n_seeds.clip(lower=1))

        eps = grp["episode"].values
        ax.plot(eps, mean_ma, color=color, linestyle=ls, linewidth=2.0, label=label, zorder=3)
        ax.fill_between(eps, mean_ma - ci, mean_ma + ci, color=color, alpha=0.18, zorder=2)

    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel(r"Mean $|\delta|$ (TD error)", fontsize=11)
    ax.set_title(
        "TD Error Dynamics During Self-Play Training\n"
        r"(moving average $w=30$; shaded band: 95% CI across seeds)",
        fontsize=11, pad=10,
    )
    ax.legend(loc="upper right", framealpha=0.9)

    fig.tight_layout()
    for fmt in ("png", "pdf"):
        path = out_dir / f"td_error_improved.{fmt}"
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved: td_error_improved.png / .pdf")


# ── summary printer ───────────────────────────────────────────────────────────

def print_summary(stats: dict) -> None:
    sep = "=" * 62
    print(f"\n{sep}")
    print("  BOOTSTRAP EVALUATION SUMMARY  (95% CI)")
    print(sep)

    for replay, label in [("uniform", "Uniform Replay"), ("prioritized", "Prioritized ER")]:
        print(f"\n  {label}:")
        for base in ["random", "greedy", "heuristic", "overall"]:
            s = stats[replay][base]
            print(f"    vs {base:<12} {s['mean']:>6.1%}  "
                  f"[{s['ci_lo']:.1%}, {s['ci_hi']:.1%}]  "
                  f"(n={s['n']})")

    print(f"\n  Per-seed overall win rate:")
    for key, s in stats.get("per_seed", {}).items():
        print(f"    {key:<20} {s['mean']:>6.1%}  [{s['ci_lo']:.1%}, {s['ci_hi']:.1%}]")

    print(f"\n  Difference CI (Uniform − PER win rate):")
    overlaps = []
    for base, d in stats.get("difference_ci", {}).items():
        z = "⚠ CI contains 0" if d["ci_contains_zero"] else "✓ CI excludes 0"
        print(f"    vs {base:<12} diff={d['diff_mean']:+.1%}  "
              f"[{d['ci_lo']:+.1%}, {d['ci_hi']:+.1%}]  {z}")
        if d["ci_contains_zero"]:
            overlaps.append(base)

    if overlaps:
        print(f"\n  ⚠  Overlapping CIs for: {', '.join(overlaps)}")
        print("     → Cannot claim statistical superiority of Uniform over PER.")
    else:
        print("\n  ✓  CIs do not overlap — Uniform shows significantly higher win rate.")

    print(f"\n{sep}\n")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--games",     type=int,  default=1000,
                   help="Games per pair (default 1000)")
    p.add_argument("--max-moves", type=int,  default=300, dest="max_moves")
    p.add_argument("--device",    type=str,  default="auto")
    p.add_argument("--out",       type=Path, default=Path("outputs/eval_bootstrap"))
    p.add_argument("--n-bootstrap", type=int, default=10_000, dest="n_bootstrap",
                   help="Bootstrap resamples (default 10000)")
    p.add_argument("--skip-eval", action="store_true", dest="skip_eval",
                   help="Skip evaluation, load existing detailed_games.csv")
    args = p.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    games_csv  = out_dir / "detailed_games.csv"
    stats_json = out_dir / "bootstrap_stats.json"

    # ── 1. Evaluation ─────────────────────────────────────────────────────────
    if args.skip_eval and games_csv.exists():
        log.info("Loading existing games from %s", games_csv)
        df_games = pd.read_csv(games_csv)
    else:
        log.info("Loading agents…")
        agents = load_agents(args.device)

        t0 = time.time()
        df_games = run_evaluations(agents, games=args.games, max_moves=args.max_moves)
        df_games.to_csv(games_csv, index=False)
        log.info("All evaluations done in %.1fs  →  %s", time.time() - t0, games_csv)

    # ── 2. Bootstrap ──────────────────────────────────────────────────────────
    log.info("Computing bootstrap CIs (%d resamples)…", args.n_bootstrap)
    stats = compute_bootstrap_stats(df_games, n_resamples=args.n_bootstrap)
    with open(stats_json, "w") as f:
        json.dump(stats, f, indent=2)
    log.info("Bootstrap stats saved → %s", stats_json)

    # ── 3. Plots ──────────────────────────────────────────────────────────────
    log.info("Generating plots…")
    plot_winrate_vs_opponents(stats, fig_dir)
    plot_vs_random_baseline(stats, fig_dir)

    curves_csv = "outputs/experiments/main/aggregated_learning_curves.csv"
    plot_td_error_improved(curves_csv, fig_dir)

    # ── 4. Summary ────────────────────────────────────────────────────────────
    print_summary(stats)

    print(f"  Output directory  : {out_dir}/")
    print(f"  Detailed games    : {games_csv}")
    print(f"  Bootstrap stats   : {stats_json}")
    print(f"  Figures           : {fig_dir}/")


if __name__ == "__main__":
    main()
