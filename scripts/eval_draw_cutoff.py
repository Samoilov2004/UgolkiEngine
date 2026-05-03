#!/usr/bin/env python3
"""
eval_draw_cutoff.py — ablation over draw cutoff (max_moves).

Tests whether the draw cutoff setting penalises one replay strategy.
Uses ONLY existing trained checkpoints — no training.

Usage
-----
    PYTHONPATH=src python scripts/eval_draw_cutoff.py \
        --max-moves-list 300 450 600 \
        --games 1000 \
        --device auto \
        --out outputs/eval_draw_cutoff
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import NamedTuple

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

UNIFORM_COLOR     = "#2166AC"
PRIORITIZED_COLOR = "#D6604D"
BASELINE_COLORS   = {"random": "#888888", "greedy": "#4DAC26", "heuristic": "#762A83"}

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

# ── best checkpoint per type ──────────────────────────────────────────────────
BEST_CHECKPOINTS = {
    "uniform":     "outputs/experiments/main/uniform_seed_2/models/dqn_latest.pt",
    "prioritized": "outputs/experiments/main/prioritized_seed_2/models/dqn_latest.pt",
}


# ── Q-value diagnostics wrapper ───────────────────────────────────────────────

class QStats(NamedTuple):
    entropy: float
    q_mean_abs: float
    n_legal: int


class MonitoredDQNAgent:
    """Thin wrapper around DQNAgent that records Q-value statistics per move."""

    def __init__(self, base_agent):
        self._agent = base_agent
        self._name  = base_agent.name
        self.name   = base_agent.name   # plain attr for evaluate_match
        self.q_stats: list[QStats] = []

    def select_move(self, env):
        import torch
        from corners_rl.rl.encoding import (
            encode_state, legal_action_mask, transform_move_for_player,
        )

        player       = env.current_player
        real_moves   = env.legal_moves()
        canon_moves  = [transform_move_for_player(m, player) for m in real_moves]

        # Always exploit (epsilon = 0 assumed)
        state_arr = encode_state(env.board, player)
        state_t   = (
            torch.from_numpy(state_arr)
            .unsqueeze(0)
            .to(self._agent._device)
        )
        mask = legal_action_mask(canon_moves)

        self._agent._model.eval()
        with torch.no_grad():
            q_values = self._agent._model(state_t).squeeze(0)

        legal_q = q_values[mask].cpu().float().numpy()

        # softmax entropy over legal Q-values
        q_shifted = legal_q - legal_q.max()
        exp_q     = np.exp(q_shifted)
        probs     = exp_q / exp_q.sum()
        entropy   = -float((probs * np.log(probs + 1e-10)).sum())

        self.q_stats.append(QStats(
            entropy=entropy,
            q_mean_abs=float(np.abs(legal_q).mean()),
            n_legal=len(real_moves),
        ))

        return self._agent.select_move(env)

    # forward unknown attribute access to base agent
    def __getattr__(self, attr):
        if attr in ("_agent", "_name", "name", "q_stats"):
            raise AttributeError(attr)
        return getattr(self._agent, attr)


# ── bootstrap ─────────────────────────────────────────────────────────────────

def bootstrap_ci(outcomes: np.ndarray, n: int = 10_000, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    sz   = len(outcomes)
    boot = np.array([rng.choice(outcomes, size=sz, replace=True).mean()
                     for _ in range(n)])
    return float(outcomes.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


# ── evaluation loop ───────────────────────────────────────────────────────────

def run_cutoff_ablation(agents: dict, games: int, max_moves_list: list[int]) -> pd.DataFrame:
    from corners_rl.evaluation.evaluate import evaluate_match

    matchups = [
        ("uniform",     "random"),
        ("uniform",     "greedy"),
        ("uniform",     "heuristic"),
        ("prioritized", "random"),
        ("prioritized", "greedy"),
        ("prioritized", "heuristic"),
        ("uniform",     "prioritized"),
    ]

    rng   = np.random.default_rng(0)
    rows  = []
    total = len(max_moves_list) * len(matchups)
    done  = 0

    for mm in max_moves_list:
        for a1_key, a2_key in matchups:
            done += 1
            t0 = time.time()
            log.info("[%d/%d]  max_moves=%d  %s vs %s  (%d games)…",
                     done, total, mm, a1_key, a2_key, games)

            a1 = agents[a1_key]
            a2 = agents[a2_key]

            df = evaluate_match(a1, a2, games=games, max_moves=mm, seed=done * 100 + mm)

            # win outcomes for a1
            a1_name = a1.name
            wins    = (df["winner_agent"] == a1_name).values.astype(int)
            draws   = df["draw"].values.astype(int)
            mean_wr, lo_wr, hi_wr = bootstrap_ci(wins, rng=rng)
            mean_dr, lo_dr, hi_dr = bootstrap_ci(draws, rng=rng)
            avg_moves              = float(df["moves"].mean())

            # Q-stats (if MonitoredDQNAgent)
            q_entropy, q_mag = float("nan"), float("nan")
            if isinstance(a1, MonitoredDQNAgent) and a1.q_stats:
                q_entropy = float(np.mean([s.entropy   for s in a1.q_stats]))
                q_mag     = float(np.mean([s.q_mean_abs for s in a1.q_stats]))
                a1.q_stats.clear()

            rows.append({
                "max_moves":    mm,
                "agent1":       a1_key,
                "agent2":       a2_key,
                "n_games":      games,
                "win_rate":     mean_wr,
                "win_ci_lo":    lo_wr,
                "win_ci_hi":    hi_wr,
                "draw_rate":    mean_dr,
                "draw_ci_lo":   lo_dr,
                "draw_ci_hi":   hi_dr,
                "avg_moves":    avg_moves,
                "q_entropy":    q_entropy,
                "q_mag_abs":    q_mag,
            })
            log.info("   win=%.1f%% [%.1f,%.1f]  draw=%.1f%%  moves=%.0f  %.1fs",
                     mean_wr*100, lo_wr*100, hi_wr*100, mean_dr*100, avg_moves, time.time()-t0)

    return pd.DataFrame(rows)


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_draw_cutoff_sensitivity(df: pd.DataFrame, out_dir: Path) -> None:
    """Figure 2: Win rate vs max_moves for Uniform and PER against each baseline."""
    baselines = ["random", "greedy", "heuristic"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.0), sharey=False)

    for ax, base in zip(axes, baselines):
        for replay, color, marker, ls in [
            ("uniform",     UNIFORM_COLOR,     "o", "-"),
            ("prioritized", PRIORITIZED_COLOR, "s", "--"),
        ]:
            sub = df[(df["agent1"] == replay) & (df["agent2"] == base)].sort_values("max_moves")
            if sub.empty:
                continue
            x    = sub["max_moves"].values
            y    = sub["win_rate"].values
            lo   = sub["win_ci_lo"].values
            hi   = sub["win_ci_hi"].values
            label = "Uniform" if replay == "uniform" else "PER"
            ax.plot(x, y, color=color, marker=marker, linestyle=ls,
                    linewidth=2, markersize=7, label=label, zorder=3)
            ax.fill_between(x, lo, hi, color=color, alpha=0.18, zorder=2)

        ax.set_title(f"vs {base.capitalize()}", fontsize=11)
        ax.set_xlabel("max_moves", fontsize=10)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        if ax is axes[0]:
            ax.set_ylabel("Win rate", fontsize=10)
        ax.legend(fontsize=9, framealpha=0.85)
        ax.set_xticks(df["max_moves"].unique())

    fig.suptitle("Win Rate Sensitivity to Draw Cutoff (max_moves)\n"
                 "(shaded: 95% bootstrap CI)", fontsize=11.5, y=1.02)
    fig.tight_layout()
    for fmt in ("png", "pdf"):
        fig.savefig(out_dir / f"draw_cutoff_sensitivity.{fmt}", bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved: draw_cutoff_sensitivity.png / .pdf")


def plot_game_length_draws(df: pd.DataFrame, out_dir: Path) -> None:
    """Figure 3: Game length and draw rate across max_moves values."""
    baselines = ["random", "greedy", "heuristic"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.0))

    for replay, color, ls, label in [
        ("uniform",     UNIFORM_COLOR,     "-",  "Uniform"),
        ("prioritized", PRIORITIZED_COLOR, "--", "PER"),
    ]:
        sub = df[df["agent1"] == replay].copy()
        # average across baselines
        grp = sub.groupby("max_moves")[["avg_moves", "draw_rate"]].mean().reset_index()
        grp = grp.sort_values("max_moves")
        ax1.plot(grp["max_moves"], grp["avg_moves"],  color=color, linestyle=ls, linewidth=2, marker="o", markersize=7, label=label)
        ax2.plot(grp["max_moves"], grp["draw_rate"],  color=color, linestyle=ls, linewidth=2, marker="o", markersize=7, label=label)

    for ax, title, ylabel in [
        (ax1, "Average Game Length vs Draw Cutoff", "Avg moves per game"),
        (ax2, "Draw Rate vs Draw Cutoff",            "Draw rate"),
    ]:
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("max_moves", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(df["max_moves"].unique())
        ax.legend(fontsize=9)

    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    fig.tight_layout()
    for fmt in ("png", "pdf"):
        fig.savefig(out_dir / f"game_length_draws.{fmt}", bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved: game_length_draws.png / .pdf")


def plot_q_diagnostics(df: pd.DataFrame, out_dir: Path) -> None:
    """Q-value entropy and magnitude across max_moves values."""
    if df["q_entropy"].isna().all():
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))
    for replay, color, ls, label in [
        ("uniform",     UNIFORM_COLOR,     "-",  "Uniform"),
        ("prioritized", PRIORITIZED_COLOR, "--", "PER"),
    ]:
        sub = df[df["agent1"] == replay].dropna(subset=["q_entropy"])
        grp = sub.groupby("max_moves")[["q_entropy", "q_mag_abs"]].mean().reset_index().sort_values("max_moves")
        ax1.plot(grp["max_moves"], grp["q_entropy"], color=color, linestyle=ls, linewidth=2, marker="o", markersize=7, label=label)
        ax2.plot(grp["max_moves"], grp["q_mag_abs"], color=color, linestyle=ls, linewidth=2, marker="o", markersize=7, label=label)

    ax1.set_title("Q-value softmax entropy\n(higher = more uncertain policy)", fontsize=10.5)
    ax2.set_title("Mean |Q-value| over legal actions\n(higher = stronger preference)", fontsize=10.5)
    for ax in (ax1, ax2):
        ax.set_xlabel("max_moves", fontsize=10)
        ax.set_xticks(df["max_moves"].unique())
        ax.legend(fontsize=9)
    fig.tight_layout()
    for fmt in ("png", "pdf"):
        fig.savefig(out_dir / f"q_diagnostics.{fmt}", bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved: q_diagnostics.png / .pdf")


# ── helpers ───────────────────────────────────────────────────────────────────

def _resolve_device(device: str) -> str:
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
    from corners_rl.agents.dqn_agent import DQNAgent
    from corners_rl.agents.random_agent import RandomAgent
    from corners_rl.agents.greedy_agent import GreedyAgent
    from corners_rl.agents.heuristic_agent import HeuristicAgent

    resolved = _resolve_device(device)
    agents: dict = {
        "random":    RandomAgent(name="random",   seed=1),
        "greedy":    GreedyAgent(name="greedy"),
        "heuristic": HeuristicAgent(name="heuristic"),
    }
    for key, path in BEST_CHECKPOINTS.items():
        p = Path(path)
        if not p.exists():
            log.error("Checkpoint missing: %s", path)
            sys.exit(1)
        base_agent = DQNAgent.load(p, device=resolved, epsilon=0.0)
        base_agent._name = key
        agents[key] = MonitoredDQNAgent(base_agent)
        log.info("  Loaded %s (%s)", key, resolved)
    return agents


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-moves-list", type=int, nargs="+",
                   default=[300, 450, 600], dest="max_moves_list")
    p.add_argument("--games",     type=int,  default=1000)
    p.add_argument("--device",    type=str,  default="auto")
    p.add_argument("--out",       type=Path, default=Path("outputs/eval_draw_cutoff"))
    args = p.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    log.info("Draw cutoff ablation: max_moves=%s  games/pair=%d", args.max_moves_list, args.games)
    agents = load_agents(args.device)

    df = run_cutoff_ablation(agents, games=args.games, max_moves_list=args.max_moves_list)
    csv_path = out_dir / "draw_cutoff_results.csv"
    df.to_csv(csv_path, index=False)
    log.info("Results saved → %s", csv_path)

    plot_draw_cutoff_sensitivity(df, fig_dir)
    plot_game_length_draws(df, fig_dir)
    plot_q_diagnostics(df, fig_dir)

    # Summary
    print("\n" + "="*62)
    print("  DRAW CUTOFF SENSITIVITY SUMMARY")
    print("="*62)
    for mm in args.max_moves_list:
        sub = df[df["max_moves"] == mm]
        print(f"\n  max_moves = {mm}")
        for _, row in sub.iterrows():
            print(f"    {row['agent1']:<14} vs {row['agent2']:<14} "
                  f"win={row['win_rate']:.1%} [{row['win_ci_lo']:.1%},{row['win_ci_hi']:.1%}]  "
                  f"draw={row['draw_rate']:.1%}  moves={row['avg_moves']:.0f}")
    print("="*62 + "\n")


if __name__ == "__main__":
    main()
