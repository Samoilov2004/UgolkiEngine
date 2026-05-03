#!/usr/bin/env python3
"""Evaluate existing DQN checkpoints with forward-only inference masking.

Loads the 4 already-trained models (Uniform × 2 seeds, PER × 2 seeds) and
applies forward-only action restriction at *inference time only* — no
retraining.  This tests whether the two training strategies produce policies
that differ in quality when the inference-time constraint is applied.

Steps
-----
1. Load 4 checkpoints with forward_only=True.
2. Each model vs {Greedy, Heuristic, Random} — ``--games`` games per pair
   (side-swapped for fairness).
3. Bootstrap 95% CI for each model × opponent combination.
4. Bootstrap CI of the *difference* (Uniform+FW − PER+FW).
5. Generate a publication-quality figure with CI bars.
6. Print statistical summary and save JSON stats.

Usage
-----
    python scripts/eval_forward_masking.py
    python scripts/eval_forward_masking.py --games 1000 --device auto
    python scripts/eval_forward_masking.py --games 200 --device cpu  # quick
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

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ── Style ─────────────────────────────────────────────────────────────────────

UNIFORM_COLOR     = "#2166AC"
PRIORITIZED_COLOR = "#D6604D"

plt.rcParams.update({
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

# ── Checkpoints ───────────────────────────────────────────────────────────────

CHECKPOINTS: dict[str, dict] = {
    "uniform_s1":     {"replay": "uniform",     "seed": 1,
                       "path": "outputs/experiments/main/uniform_seed_1/models/dqn_latest.pt"},
    "uniform_s2":     {"replay": "uniform",     "seed": 2,
                       "path": "outputs/experiments/main/uniform_seed_2/models/dqn_latest.pt"},
    "prioritized_s1": {"replay": "prioritized", "seed": 1,
                       "path": "outputs/experiments/main/prioritized_seed_1/models/dqn_latest.pt"},
    "prioritized_s2": {"replay": "prioritized", "seed": 2,
                       "path": "outputs/experiments/main/prioritized_seed_2/models/dqn_latest.pt"},
}

# ── Bootstrap ─────────────────────────────────────────────────────────────────

def bootstrap_ci(
    outcomes: np.ndarray,
    n_boot: int = 10_000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Return (mean, ci_lo, ci_hi) via percentile bootstrap."""
    if rng is None:
        rng = np.random.default_rng(0)
    if len(outcomes) == 0:
        return 0.0, 0.0, 0.0
    boot = np.array([
        rng.choice(outcomes, size=len(outcomes), replace=True).mean()
        for _ in range(n_boot)
    ])
    return (float(outcomes.mean()),
            float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)))


def bootstrap_diff_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 10_000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Bootstrap CI of mean(a) − mean(b)."""
    if rng is None:
        rng = np.random.default_rng(1)
    diffs = np.array([
        rng.choice(a, size=len(a), replace=True).mean()
        - rng.choice(b, size=len(b), replace=True).mean()
        for _ in range(n_boot)
    ])
    return (float(diffs.mean()),
            float(np.percentile(diffs, 2.5)),
            float(np.percentile(diffs, 97.5)))


# ── Loading ───────────────────────────────────────────────────────────────────

def load_agent(key: str, info: dict, device: str):
    from corners_rl.agents.dqn_agent import DQNAgent
    path = Path(info["path"])
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    agent = DQNAgent.load(path, device=device, epsilon=0.0, forward_only=True)
    agent._name = key
    return agent


# ── Evaluation ────────────────────────────────────────────────────────────────

def run_eval(
    checkpoints: dict,
    games: int,
    max_moves: int,
    device: str,
) -> pd.DataFrame:
    from corners_rl.agents.greedy_agent import GreedyAgent
    from corners_rl.agents.heuristic_agent import HeuristicAgent
    from corners_rl.agents.random_agent import RandomAgent
    from corners_rl.evaluation.evaluate import evaluate_match

    baselines = {
        "greedy":    GreedyAgent(name="greedy"),
        "heuristic": HeuristicAgent(name="heuristic"),
        "random":    RandomAgent(name="random", seed=0),
    }

    rng   = np.random.default_rng(42)
    rows  = []
    total = len(checkpoints) * len(baselines)
    done  = 0

    for key, info in sorted(checkpoints.items()):
        try:
            agent = load_agent(key, info, device)
        except FileNotFoundError as exc:
            log.warning("Skipping %s: %s", key, exc)
            continue

        log.info("Loaded %s (replay=%s, seed=%d, forward_only=True)",
                 key, info["replay"], info["seed"])

        for base_name, base_agent in baselines.items():
            done += 1
            t0 = time.time()
            log.info("  [%d/%d]  %s vs %s  (%d games)…",
                     done, total, key, base_name, games)

            df = evaluate_match(agent, base_agent,
                                games=games, max_moves=max_moves,
                                seed=done * 100)

            wins  = (df["winner_agent"] == key).values.astype(float)
            draws = df["draw"].values.astype(float)
            mean, lo, hi = bootstrap_ci(wins, rng=rng)

            rows.append({
                "run_key":     key,
                "replay_type": info["replay"],
                "seed":        info["seed"],
                "opponent":    base_name,
                "n_games":     len(df),
                "win_rate":    mean,
                "ci_lo":       lo,
                "ci_hi":       hi,
                "draw_rate":   float(draws.mean()),
                "avg_moves":   float(df["moves"].mean()),
                # raw outcomes stored for pooled CI later
                "_wins_raw":   list(wins),
            })
            log.info("    win=%.1f%% [%.1f, %.1f]  %.1fs",
                     mean*100, lo*100, hi*100, time.time()-t0)

    return pd.DataFrame(rows)


# ── Stats ─────────────────────────────────────────────────────────────────────

def compute_group_stats(df: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Bootstrap stats per replay_type (pooled across seeds and opponents)."""
    result = {}
    for replay in ("uniform", "prioritized"):
        sub = df[df["replay_type"] == replay]
        all_wins = np.concatenate([np.array(r) for r in sub["_wins_raw"]])
        mean, lo, hi = bootstrap_ci(all_wins, rng=rng)
        # Per-seed overall win rate
        per_seed = (
            sub.groupby("seed")["win_rate"].mean().values
        )
        result[replay] = {
            "mean": mean, "ci_lo": lo, "ci_hi": hi,
            "per_seed": per_seed.tolist(),
        }

    # Difference CI
    uni_wins  = np.concatenate([
        np.array(r) for r in df[df["replay_type"]=="uniform"]["_wins_raw"]
    ])
    per_wins  = np.concatenate([
        np.array(r) for r in df[df["replay_type"]=="prioritized"]["_wins_raw"]
    ])
    diff, d_lo, d_hi = bootstrap_diff_ci(uni_wins, per_wins, rng=rng)
    ci_contains_zero = d_lo <= 0 <= d_hi

    result["diff"] = {
        "mean": diff, "ci_lo": d_lo, "ci_hi": d_hi,
        "ci_contains_zero": ci_contains_zero,
    }
    return result


def compute_opponent_stats(df: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Bootstrap stats per replay_type × opponent."""
    result = {}
    for (replay, opp), grp in df.groupby(["replay_type", "opponent"]):
        all_wins = np.concatenate([np.array(r) for r in grp["_wins_raw"]])
        mean, lo, hi = bootstrap_ci(all_wins, rng=rng)
        result[f"{replay}_{opp}"] = {
            "replay_type": replay, "opponent": opp,
            "mean": mean, "ci_lo": lo, "ci_hi": hi,
        }
    return result


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_main(df: pd.DataFrame, group_stats: dict,
              opp_stats: dict, out_path: Path) -> None:
    """Main publication figure: win rate with 95% CI, Uniform vs PER + FW.

    Left panel: overall (pooled over all opponents).
    Right panel: breakdown by opponent type.
    """
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(11, 5),
        gridspec_kw={"width_ratios": [1, 1.6]},
    )

    # ── Left: overall comparison ──────────────────────────────────────────────
    methods = [("uniform", "Uniform + Forward", UNIFORM_COLOR),
               ("prioritized", "PER + Forward", PRIORITIZED_COLOR)]
    x = np.arange(2)

    for i, (replay, label, color) in enumerate(methods):
        st = group_stats[replay]
        mean, lo, hi = st["mean"], st["ci_lo"], st["ci_hi"]
        err_lo = mean - lo
        err_hi = hi - mean

        ax_left.bar(i, mean, 0.5,
                    color=color, alpha=0.85,
                    yerr=[[err_lo], [err_hi]],
                    capsize=10, error_kw={"linewidth": 1.6, "ecolor": "#222"},
                    zorder=3, label=label)

        # per-seed dots
        seeds = st["per_seed"]
        jitter = np.array([-0.07, 0.07]) if len(seeds) == 2 else np.zeros(len(seeds))
        ax_left.scatter(np.full(len(seeds), i) + jitter, seeds,
                        color=color, s=60, zorder=5,
                        edgecolors="white", linewidths=0.8)

        ax_left.text(i, hi + 0.015, f"{mean:.1%}",
                     ha="center", va="bottom",
                     fontsize=12, fontweight="bold", color=color)

    # difference annotation
    d = group_stats["diff"]
    sign = "+" if d["mean"] >= 0 else ""
    color_d = "#1a1a1a" if not d["ci_contains_zero"] else "#888"
    ax_left.text(0.5, 0.04,
                 f"Δ = {sign}{d['mean']:.1%}  [{d['ci_lo']:+.1%}, {d['ci_hi']:+.1%}]",
                 transform=ax_left.transAxes, ha="center", fontsize=9.5,
                 color=color_d,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white",
                           ec="#cccccc", alpha=0.9))
    if not d["ci_contains_zero"]:
        ax_left.text(0.5, -0.01,
                     "★ статистически значимо (0 ∉ CI)",
                     transform=ax_left.transAxes, ha="center",
                     fontsize=8.5, color="#2a7a2a")

    ax_left.set_xticks(x)
    ax_left.set_xticklabels(["Uniform\n+ Forward", "PER\n+ Forward"], fontsize=11)
    ax_left.set_ylabel("Win rate (все противники)", fontsize=11)
    ax_left.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax_left.set_ylim(0, 1.0)
    ax_left.set_title("Суммарный win rate\n(95% bootstrap CI)", fontsize=11, pad=8)

    # ── Right: by opponent ────────────────────────────────────────────────────
    opponents = ["random", "greedy", "heuristic"]
    x_opp = np.arange(len(opponents))
    w = 0.32

    for i, (replay, label, color) in enumerate(methods):
        means_, lo_, hi_ = [], [], []
        for opp in opponents:
            key = f"{replay}_{opp}"
            if key in opp_stats:
                st = opp_stats[key]
                means_.append(st["mean"])
                lo_.append(st["mean"] - st["ci_lo"])
                hi_.append(st["ci_hi"] - st["mean"])
            else:
                means_.append(0); lo_.append(0); hi_.append(0)

        offset = (i - 0.5) * w
        ax_right.bar(x_opp + offset, means_, w,
                     color=color, alpha=0.85, label=label,
                     yerr=[lo_, hi_],
                     capsize=6, error_kw={"linewidth": 1.3},
                     zorder=3)

    ax_right.set_xticks(x_opp)
    ax_right.set_xticklabels(
        [o.capitalize() for o in opponents], fontsize=11)
    ax_right.set_ylabel("Win rate", fontsize=11)
    ax_right.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax_right.set_ylim(0, 1.0)
    ax_right.set_title("Win rate по типу противника\n(95% bootstrap CI)", fontsize=11, pad=8)
    ax_right.legend(fontsize=10, loc="upper right")

    n_games = int(df["n_games"].iloc[0]) if len(df) > 0 else "?"
    fig.text(0.5, -0.03,
             f"Forward-Only masking применён на инференсе к уже обученным моделям  |  "
             f"{n_games} партий/пару  |  95% bootstrap CI (10 000 ресэмплов)",
             ha="center", fontsize=8.5, color="#555")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", out_path)
    print(f"  ✓  {out_path}")


# ── Summary printer ───────────────────────────────────────────────────────────

def print_summary(group_stats: dict, opp_stats: dict) -> None:
    sep = "=" * 65
    print(f"\n{sep}")
    print("  FORWARD MASKING EVAL — STATISTICAL SUMMARY")
    print(sep)

    for replay, label in [("uniform", "Uniform + Forward-Only"),
                           ("prioritized", "PER + Forward-Only")]:
        st = group_stats[replay]
        seeds = ", ".join(f"{v:.1%}" for v in sorted(st["per_seed"]))
        print(f"\n  {label}")
        print(f"    Seeds      : {seeds}")
        print(f"    Mean       : {st['mean']:.1%}")
        print(f"    95% CI     : [{st['ci_lo']:.1%}, {st['ci_hi']:.1%}]")

        print(f"    By opponent:")
        for opp in ("random", "greedy", "heuristic"):
            k = f"{replay}_{opp}"
            if k in opp_stats:
                s = opp_stats[k]
                print(f"      vs {opp:<10}: {s['mean']:.1%}  "
                      f"[{s['ci_lo']:.1%}, {s['ci_hi']:.1%}]")

    d = group_stats["diff"]
    print(f"\n  Bootstrap diff CI (Uniform − PER):")
    print(f"    {d['mean']:+.1%}  [{d['ci_lo']:+.1%}, {d['ci_hi']:+.1%}]")
    if d["ci_contains_zero"]:
        print("    ⚠  CI contains 0 — различие НЕ подтверждено")
    else:
        print("    ✓  CI excludes 0 — различие статистически значимо")

    print(f"\n{sep}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Eval existing DQN checkpoints with forward-only masking.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--games",     type=int, default=1000,
                   help="Games per (model, opponent) pair.")
    p.add_argument("--max-moves", type=int, default=300, dest="max_moves")
    p.add_argument("--device",    type=str, default="auto")
    p.add_argument("--out",       type=Path,
                   default=Path("outputs/eval_forward_masking"))
    p.add_argument("--skip-eval", action="store_true", dest="skip_eval",
                   help="Re-use existing per_game_results.csv if present.")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Resolve device
    device = args.device
    if device == "auto":
        try:
            import torch
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        except Exception:
            device = "cpu"

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    csv_path = out_dir / "per_pair_results.csv"

    print(f"\n{'='*65}")
    print("  Forward-Only Masking Evaluation")
    print(f"{'='*65}")
    print(f"  device    : {device}")
    print(f"  games     : {args.games}/pair")
    print(f"  output    : {out_dir}")

    # ── Evaluation ────────────────────────────────────────────────────────────
    if args.skip_eval and csv_path.exists():
        log.info("Loading cached results from %s", csv_path)
        df = pd.read_csv(csv_path)
        # Reconstruct _wins_raw from win_rate + n_games (approximate)
        # For exact stats we need the raw column — use skip_eval only for plotting
        df["_wins_raw"] = df.apply(
            lambda r: list(np.array([1]*round(r["win_rate"]*r["n_games"])
                                    + [0]*(r["n_games"] - round(r["win_rate"]*r["n_games"])))),
            axis=1,
        )
    else:
        df = run_eval(CHECKPOINTS, games=args.games,
                      max_moves=args.max_moves, device=device)
        # Save without the raw list column
        df.drop(columns=["_wins_raw"]).to_csv(csv_path, index=False)
        log.info("Per-pair results saved → %s", csv_path)

    rng = np.random.default_rng(0)
    group_stats = compute_group_stats(df, rng)
    opp_stats   = compute_opponent_stats(df, rng)

    # ── Save stats JSON ───────────────────────────────────────────────────────
    # Make JSON-serialisable (remove numpy types)
    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, np.ndarray)):
            return [float(x) for x in obj]
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return obj

    stats_path = out_dir / "bootstrap_stats.json"
    with open(stats_path, "w") as f:
        json.dump({"group": _clean(group_stats),
                   "by_opponent": _clean(opp_stats)}, f, indent=2)
    print(f"  ✓  {stats_path}")

    # ── Plot: main figure ─────────────────────────────────────────────────────
    print("\n  Generating plots…")
    fig_path = fig_dir / "winrate_forward_masking.png"
    plot_main(df, group_stats, opp_stats, fig_path)

    # Copy to report/figures so LaTeX picks it up
    import shutil
    report_fig = Path("report/figures/winrate_forward_masking.png")
    report_fig.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(fig_path, report_fig)
    print(f"  ✓  {report_fig}  (copied for LaTeX)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print_summary(group_stats, opp_stats)


if __name__ == "__main__":
    main()
