#!/usr/bin/env python3
"""Full experiment pipeline: baseline → (imitation) → self-play → evaluate → visualise → plots.

Usage
-----
    # Minimal
    python scripts/run_full_experiment.py

    # With imitation pre-training
    python scripts/run_full_experiment.py \\
        --episodes 200 --eval-games 50 --imitation --imitation-games 200 \\
        --device cpu --seed 42 --out outputs

All stages write to sub-directories under --out.  If a stage fails the error is
printed and subsequent stages that depend on its output are skipped gracefully.
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

# Allow running without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _header(step: int, total: int, label: str) -> None:
    print(f"\n{'='*60}")
    print(f"  [{step}/{total}] {label}")
    print(f"{'='*60}")


def _ok(label: str) -> None:
    print(f"  ✓  {label}")


def _warn(label: str) -> None:
    print(f"  ⚠  {label}")


def _fail(label: str, exc: Exception) -> None:
    print(f"  ✗  {label}")
    print(f"     {type(exc).__name__}: {exc}")
    log.debug("Full traceback:", exc_info=True)


def _build_baseline_agents(seed: int):
    from corners_rl.agents.greedy_agent import GreedyAgent
    from corners_rl.agents.heuristic_agent import HeuristicAgent
    from corners_rl.agents.random_agent import RandomAgent

    return [
        RandomAgent(seed=seed),
        GreedyAgent(seed=seed),
        HeuristicAgent(seed=seed),
    ]


def _build_dqn_agent(checkpoint: Path, device: str, seed: int):
    """Load DQNAgent from *checkpoint*.  Returns None if unavailable."""
    from corners_rl.agents.dqn_agent import DQNAgent
    from corners_rl.rl.model import DQNModel
    import torch

    model = DQNModel()
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return DQNAgent(model=model, device=device, epsilon=0.0, seed=seed)


# ── Stages ────────────────────────────────────────────────────────────────────

def stage_baseline_eval(
    out: Path, eval_games: int, seed: int, max_moves: int
) -> Path | None:
    """Round-robin tournament among Random / Greedy / Heuristic."""
    from corners_rl.evaluation.tournaments import round_robin_tournament

    agents = _build_baseline_agents(seed)
    detailed, summary = round_robin_tournament(
        agents, games_per_pair=eval_games, max_moves=max_moves, seed=seed
    )

    eval_dir = out / "eval_baseline"
    eval_dir.mkdir(parents=True, exist_ok=True)
    detailed.to_csv(eval_dir / "detailed_results.csv", index=False)
    summary.to_csv(eval_dir / "summary.csv", index=False)

    print("\n  Baseline results:")
    print(summary.to_string(index=False))
    _ok(f"Saved to {eval_dir}/")
    return eval_dir


def stage_imitation(
    out: Path, games: int, epochs: int, device: str, seed: int, max_moves: int
) -> Path | None:
    """Behavioural cloning from HeuristicAgent."""
    from corners_rl.agents.heuristic_agent import HeuristicAgent
    from corners_rl.rl.imitation import (
        ImitationConfig,
        generate_imitation_dataset,
        train_imitation,
    )
    from corners_rl.rl.model import DQNModel

    ckpt_path = out / "models" / "imitation.pt"
    log_path  = out / "logs"   / "imitation_log.csv"

    expert  = HeuristicAgent(seed=seed)
    dataset = generate_imitation_dataset(
        expert_agent=expert,
        games=games,
        max_moves=max_moves,
        seed=seed,
        opponent="random",
    )
    _ok(f"Dataset: {len(dataset)} samples from {games} games")

    config = ImitationConfig(
        epochs=epochs,
        batch_size=128,
        learning_rate=1e-3,
        device=device,
        seed=seed,
        log_path=str(log_path),
        out_path=str(ckpt_path),
    )
    model = DQNModel()
    train_imitation(model, dataset, config)
    _ok(f"Checkpoint saved to {ckpt_path}")
    return ckpt_path


def stage_train_dqn(
    out: Path,
    episodes: int,
    device: str,
    seed: int,
    max_moves: int,
    init_checkpoint: Path | None,
) -> Path | None:
    """DQN self-play training."""
    from corners_rl.rl.train_dqn import SelfPlayTrainer, TrainConfig

    config = TrainConfig(
        episodes=episodes,
        max_moves=max_moves,
        device=device,
        seed=seed,
        output_dir=str(out),
        init_checkpoint=str(init_checkpoint) if init_checkpoint else None,
    )
    trainer = SelfPlayTrainer(config)
    trainer.train()

    latest = out / "models" / "dqn_latest.pt"
    _ok(f"Training complete — checkpoint at {latest}")
    return latest if latest.exists() else None


def stage_full_eval(
    out: Path,
    dqn_checkpoint: Path | None,
    eval_games: int,
    device: str,
    seed: int,
    max_moves: int,
) -> Path | None:
    """Round-robin tournament including DQN (if checkpoint exists)."""
    from corners_rl.evaluation.tournaments import round_robin_tournament

    agents = _build_baseline_agents(seed)

    if dqn_checkpoint and dqn_checkpoint.exists():
        try:
            dqn = _build_dqn_agent(dqn_checkpoint, device, seed)
            agents.append(dqn)
            _ok(f"DQN agent loaded from {dqn_checkpoint}")
        except Exception as exc:
            _warn(f"Could not load DQN checkpoint ({exc}); running without it.")
    else:
        _warn("DQN checkpoint not found; tournament runs without DQN.")

    detailed, summary = round_robin_tournament(
        agents, games_per_pair=eval_games, max_moves=max_moves, seed=seed
    )

    eval_dir = out / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    detailed.to_csv(eval_dir / "detailed_results.csv", index=False)
    summary.to_csv(eval_dir / "summary.csv", index=False)

    print("\n  Full evaluation results:")
    print(summary.to_string(index=False))
    _ok(f"Saved to {eval_dir}/")
    return eval_dir


def stage_visualise(
    out: Path,
    dqn_checkpoint: Path | None,
    device: str,
    seed: int,
    max_moves: int,
) -> None:
    """Record a GIF of a game (DQN vs Heuristic if available, else Greedy vs Random)."""
    from corners_rl.agents.greedy_agent import GreedyAgent
    from corners_rl.agents.heuristic_agent import HeuristicAgent
    from corners_rl.agents.random_agent import RandomAgent
    from corners_rl.visualization.animate_game import record_game, save_game_gif

    if dqn_checkpoint and dqn_checkpoint.exists():
        try:
            agent1 = _build_dqn_agent(dqn_checkpoint, device, seed)
            agent2 = HeuristicAgent(seed=seed)
            label  = "dqn_vs_heuristic"
        except Exception as exc:
            _warn(f"DQN load failed ({exc}); falling back to greedy vs random.")
            agent1 = GreedyAgent(seed=seed)
            agent2 = RandomAgent(seed=seed)
            label  = "greedy_vs_random"
    else:
        agent1 = GreedyAgent(seed=seed)
        agent2 = RandomAgent(seed=seed)
        label  = "greedy_vs_random"

    frames = record_game(agent1, agent2, max_moves=max_moves, seed=seed)

    gif_dir = out / "figures"
    gif_dir.mkdir(parents=True, exist_ok=True)
    gif_path = gif_dir / f"game_{label}.gif"
    save_game_gif(frames, gif_path, fps=2,
                  agent1_name=agent1.name, agent2_name=agent2.name)
    _ok(f"GIF saved to {gif_path}  ({len(frames)} frames)")


def stage_report_plots(
    out: Path,
    train_log: Path,
    eval_dir: Path | None,
) -> None:
    """Generate all report-quality PNG plots."""
    from corners_rl.agents.greedy_agent import GreedyAgent
    from corners_rl.agents.random_agent import RandomAgent
    from corners_rl.visualization.plots import (
        generate_position_heatmap,
        plot_evaluation_summary,
        plot_training_curves,
    )

    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Training curves
    if train_log.exists():
        try:
            plot_training_curves(train_log, fig_dir)
            _ok("Training curves saved")
        except Exception as exc:
            _warn(f"Training curves skipped: {exc}")
    else:
        _warn(f"No train log at {train_log} — training curves skipped")

    # Evaluation plots
    if eval_dir:
        summary_csv  = eval_dir / "summary.csv"
        detailed_csv = eval_dir / "detailed_results.csv"
        if summary_csv.exists():
            try:
                plot_evaluation_summary(
                    summary_csv, fig_dir,
                    detailed_path=detailed_csv if detailed_csv.exists() else None,
                )
                _ok("Evaluation plots saved")
            except Exception as exc:
                _warn(f"Evaluation plots skipped: {exc}")
        else:
            _warn("No summary.csv — evaluation plots skipped")

    # Position heatmap
    try:
        generate_position_heatmap(
            GreedyAgent(seed=42), RandomAgent(seed=43),
            n_games=30, seed=42, out_dir=fig_dir,
        )
        _ok("Position heatmap saved")
    except Exception as exc:
        _warn(f"Position heatmap skipped: {exc}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the full Corners RL experiment pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--episodes",        type=int,   default=200,
                   help="DQN self-play training episodes.")
    p.add_argument("--max-moves",       type=int,   default=300, dest="max_moves",
                   help="Step limit per game.")
    p.add_argument("--eval-games",      type=int,   default=50,  dest="eval_games",
                   help="Games per pair in each tournament.")
    p.add_argument("--imitation",       action="store_true",
                   help="Run imitation pre-training before self-play.")
    p.add_argument("--imitation-games", type=int,   default=200, dest="imitation_games",
                   help="Expert games for imitation dataset.")
    p.add_argument("--imitation-epochs",type=int,   default=5,   dest="imitation_epochs",
                   help="Training epochs for imitation learning.")
    p.add_argument("--device",          type=str,   default="cpu",
                   help='Torch device: "cpu", "cuda", "mps", or "auto".')
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--out",             type=Path,  default=Path("outputs"),
                   help="Root output directory.")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    n_stages = 5 + (1 if args.imitation else 0)
    step = 0

    def next_step(label: str) -> None:
        nonlocal step
        step += 1
        _header(step, n_stages, label)

    dqn_checkpoint: Path | None = None
    eval_dir:       Path | None = None
    imitation_ckpt: Path | None = None

    # ── 1. Baseline evaluation ────────────────────────────────────────────────
    next_step("Baseline evaluation (Random / Greedy / Heuristic)")
    try:
        stage_baseline_eval(out, args.eval_games, args.seed, args.max_moves)
    except Exception as exc:
        _fail("Baseline evaluation failed", exc)
        traceback.print_exc()

    # ── 2. Imitation pre-training (optional) ──────────────────────────────────
    if args.imitation:
        next_step("Imitation pre-training (behavioural cloning from HeuristicAgent)")
        try:
            imitation_ckpt = stage_imitation(
                out,
                games=args.imitation_games,
                epochs=args.imitation_epochs,
                device=args.device,
                seed=args.seed,
                max_moves=args.max_moves,
            )
        except Exception as exc:
            _fail("Imitation pre-training failed — self-play will start from scratch", exc)
            traceback.print_exc()

    # ── 3. DQN self-play training ─────────────────────────────────────────────
    next_step("DQN self-play training")
    try:
        dqn_checkpoint = stage_train_dqn(
            out,
            episodes=args.episodes,
            device=args.device,
            seed=args.seed,
            max_moves=args.max_moves,
            init_checkpoint=imitation_ckpt,
        )
    except Exception as exc:
        _fail("DQN training failed — subsequent stages will run without DQN", exc)
        traceback.print_exc()

    # ── 4. Full evaluation (all agents including DQN) ─────────────────────────
    next_step("Full evaluation tournament")
    try:
        eval_dir = stage_full_eval(
            out,
            dqn_checkpoint=dqn_checkpoint,
            eval_games=args.eval_games,
            device=args.device,
            seed=args.seed,
            max_moves=args.max_moves,
        )
    except Exception as exc:
        _fail("Full evaluation failed", exc)
        traceback.print_exc()

    # ── 5. Game visualisation (GIF) ───────────────────────────────────────────
    next_step("Game visualisation (GIF)")
    try:
        stage_visualise(
            out,
            dqn_checkpoint=dqn_checkpoint,
            device=args.device,
            seed=args.seed,
            max_moves=args.max_moves,
        )
    except Exception as exc:
        _fail("Visualisation failed", exc)
        traceback.print_exc()

    # ── 6. Report plots ───────────────────────────────────────────────────────
    next_step("Report plots")
    try:
        stage_report_plots(
            out,
            train_log=out / "logs" / "train_log.csv",
            eval_dir=eval_dir,
        )
    except Exception as exc:
        _fail("Report plots failed", exc)
        traceback.print_exc()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Experiment complete!  All outputs under:")
    print(f"    {out.resolve()}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
