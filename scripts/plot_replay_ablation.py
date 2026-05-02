#!/usr/bin/env python3
"""Generate all comparison plots for the Uniform vs PER ablation study.

Reads ``aggregated_learning_curves.csv`` and ``final_eval_summary.csv``
from the experiment directory and writes 10 figures plus two summary
tables (LaTeX + Markdown) to the output directory.

Usage
-----
    python scripts/plot_replay_ablation.py \\
        --experiment-dir outputs/experiments/replay_ablation \\
        --out outputs/experiments/replay_ablation/figures

    # Custom threshold for sample-efficiency plot
    python scripts/plot_replay_ablation.py \\
        --experiment-dir outputs/experiments/replay_ablation \\
        --out outputs/figures \\
        --threshold 0.5 --smoothing 0.08
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from corners_rl.visualization.replay_ablation_plots import (
    build_summary_tables,
    plot_avg_moves_comparison,
    plot_beta_schedule,
    plot_draw_rate_comparison,
    plot_final_eval_comparison,
    plot_learning_curve_win_rate,
    plot_priority_dynamics,
    plot_sample_efficiency,
    plot_td_error_dynamics,
)

log = logging.getLogger(__name__)


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_csv(path: Path, label: str) -> pd.DataFrame | None:
    """Load a CSV file; return None and log a warning on failure."""
    if not path.exists():
        log.warning("File not found: %s (%s)", path, label)
        return None
    try:
        df = pd.read_csv(path)
        log.info("Loaded %s: %d rows, %d columns — %s",
                 label, len(df), len(df.columns), path)
        return df
    except Exception as exc:
        log.warning("Could not read %s (%s): %s", path, label, exc)
        return None


# ── Result reporter ───────────────────────────────────────────────────────────

def _report(results: dict[str, bool]) -> None:
    ok  = [k for k, v in results.items() if v]
    bad = [k for k, v in results.items() if not v]
    print(f"\n  Plots generated  : {len(ok)}/{len(results)}")
    for name in ok:
        print(f"    ✓  {name}")
    if bad:
        print(f"\n  Skipped (data unavailable):")
        for name in bad:
            print(f"    ·  {name}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot Uniform vs PER ablation results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--experiment-dir", "-e",
        type=Path,
        default=Path("outputs/experiments/replay_ablation"),
        dest="experiment_dir",
        help="Directory containing aggregated_learning_curves.csv "
             "and final_eval_summary.csv.",
    )
    p.add_argument(
        "--out", "-o",
        type=Path,
        default=None,
        help="Output directory for figures and tables.  "
             "Defaults to <experiment-dir>/figures.",
    )
    p.add_argument(
        "--curves-csv",
        type=Path,
        default=None,
        dest="curves_csv",
        help="Override path to aggregated_learning_curves.csv.",
    )
    p.add_argument(
        "--evals-csv",
        type=Path,
        default=None,
        dest="evals_csv",
        help="Override path to final_eval_summary.csv.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.60,
        help="Win-rate threshold for sample-efficiency plot.",
    )
    p.add_argument(
        "--smoothing",
        type=float,
        default=0.05,
        dest="smoothing_frac",
        help="Smoothing window as fraction of total episode count (0–1).",
    )
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()

    # ── Resolve paths ─────────────────────────────────────────────────────────
    exp_dir   = args.experiment_dir
    out_dir   = args.out if args.out is not None else exp_dir / "figures"
    curves_p  = args.curves_csv or exp_dir / "aggregated_learning_curves.csv"
    evals_p   = args.evals_csv  or exp_dir / "final_eval_summary.csv"

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("  Replay Ablation — Plot Generator")
    print(f"{'='*60}")
    print(f"  experiment dir : {exp_dir}")
    print(f"  output dir     : {out_dir}")
    print(f"  curves CSV     : {curves_p}")
    print(f"  evals CSV      : {evals_p}")
    print(f"  threshold      : {args.threshold}")
    print(f"  smoothing frac : {args.smoothing_frac}")

    # ── Load data ─────────────────────────────────────────────────────────────
    curves = _load_csv(curves_p, "learning curves")
    evals  = _load_csv(evals_p,  "eval summary")

    if curves is None and evals is None:
        print("\n  ERROR: No input data found.  "
              "Run the ablation study first:\n"
              "    python scripts/run_replay_ablation.py --episodes 2000 ...\n")
        sys.exit(1)

    # ── Generate plots ────────────────────────────────────────────────────────
    print(f"\n  Generating plots…")
    sm = args.smoothing_frac
    results: dict[str, bool] = {}

    # 1–3: Learning-curve win rates
    for opp in ("random", "greedy", "heuristic"):
        key  = f"learning_curve_win_rate_{opp}"
        path = out_dir / f"learning_curve_win_rate_{opp}.png"
        if curves is not None:
            results[key] = plot_learning_curve_win_rate(
                curves, opp, path, smoothing_frac=sm
            )
        else:
            log.warning("[%s] No curves data — skipping.", key)
            results[key] = False

    # 4: Sample efficiency
    if curves is not None:
        results["sample_efficiency"] = plot_sample_efficiency(
            curves,
            out_dir / "sample_efficiency_threshold.png",
            threshold=args.threshold,
        )
    else:
        log.warning("[sample_efficiency] No curves data.")
        results["sample_efficiency"] = False

    # 5: Final eval comparison
    if evals is not None:
        results["final_eval_comparison"] = plot_final_eval_comparison(
            evals, out_dir / "final_eval_comparison.png"
        )
    else:
        log.warning("[final_eval_comparison] No evals data.")
        results["final_eval_comparison"] = False

    # 6: Draw rate comparison
    if evals is not None:
        results["draw_rate_comparison"] = plot_draw_rate_comparison(
            evals, out_dir / "draw_rate_comparison.png"
        )
    else:
        log.warning("[draw_rate_comparison] No evals data.")
        results["draw_rate_comparison"] = False

    # 7: Average moves comparison
    results["avg_moves_comparison"] = plot_avg_moves_comparison(
        evals, curves, out_dir / "avg_moves_comparison.png",
        smoothing_frac=sm,
    )

    # 8: TD error dynamics
    if curves is not None:
        results["td_error_dynamics"] = plot_td_error_dynamics(
            curves, out_dir / "td_error_dynamics.png", smoothing_frac=sm
        )
    else:
        log.warning("[td_error_dynamics] No curves data.")
        results["td_error_dynamics"] = False

    # 9: Priority dynamics
    if curves is not None:
        results["priority_dynamics"] = plot_priority_dynamics(
            curves, out_dir / "priority_dynamics.png", smoothing_frac=sm
        )
    else:
        log.warning("[priority_dynamics] No curves data.")
        results["priority_dynamics"] = False

    # 10: Beta schedule
    if curves is not None:
        results["beta_schedule"] = plot_beta_schedule(
            curves, out_dir / "beta_schedule.png"
        )
    else:
        log.warning("[beta_schedule] No curves data.")
        results["beta_schedule"] = False

    # ── Summary tables ────────────────────────────────────────────────────────
    if evals is not None:
        print("\n  Building summary tables…")
        latex, md = build_summary_tables(evals)

        if latex:
            latex_path = out_dir / "summary_latex_table.txt"
            latex_path.write_text(latex, encoding="utf-8")
            print(f"  ✓  LaTeX table  → {latex_path}")
        else:
            print("  ·  LaTeX table: skipped (insufficient data)")

        if md:
            md_path = out_dir / "summary_markdown_table.md"
            md_path.write_text(md, encoding="utf-8")
            print(f"  ✓  Markdown table → {md_path}")
        else:
            print("  ·  Markdown table: skipped (insufficient data)")
    else:
        print("\n  Summary tables: skipped (no eval data)")

    # ── Final report ──────────────────────────────────────────────────────────
    _report(results)
    print(f"\n  Output directory: {out_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
