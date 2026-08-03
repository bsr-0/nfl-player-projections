#!/usr/bin/env python3
"""
CLI runner for the leakage-free time-series backtester.

Usage:
    python scripts/run_ts_backtest.py                     # Backtest latest season, Ridge α=1.0
    python scripts/run_ts_backtest.py --season 2024       # Backtest specific season
    python scripts/run_ts_backtest.py --model gbm         # Use GBM instead of Ridge
    python scripts/run_ts_backtest.py --model ensemble    # Use production ensemble stack
    python scripts/run_ts_backtest.py --positions QB RB    # Only backtest QB and RB
    python scripts/run_ts_backtest.py --alpha 0.3         # Override Ridge α (default 1.0)
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import RIDGE_DEFAULT_ALPHA
from src.evaluation.ts_backtester import run_ts_backtest


def main():
    parser = argparse.ArgumentParser(
        description="Run leakage-free expanding-window time-series backtest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--season", "-s",
        type=int,
        default=None,
        help="Season to backtest (default: latest available)",
    )
    parser.add_argument(
        "--model", "-m",
        choices=["ridge", "gbm", "ensemble"],
        default="ridge",
        help="Model type: ridge (fast), gbm (higher fidelity), or ensemble (production stack)",
    )
    parser.add_argument(
        "--positions", "-p",
        nargs="+",
        default=None,
        help="Positions to backtest (default: QB RB WR TE)",
    )
    parser.add_argument(
        "--alpha", "-a",
        nargs="+",
        default=[str(RIDGE_DEFAULT_ALPHA)],
        help=(
            "Ridge regularization strength. Pass a single float for uniform "
            f"regularization (default: {RIDGE_DEFAULT_ALPHA}) or a list of "
            "POS=VALUE pairs for per-position tuning (e.g. '--alpha QB=10000 "
            "RB=1 TE=1 WR=1'). Unspecified positions fall back to 1.0. "
            "Ignored when --model is gbm or ensemble."
        ),
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output",
    )
    parser.add_argument(
        "--payout-multiplier",
        type=float,
        default=None,
        help=(
            "Cash-H2H payout multiplier for ROI calculation "
            "(default: config.DECISION_QUALITY; 1.8 ≈ DFS cash H2H after ~20%% rake)."
        ),
    )
    parser.add_argument(
        "--no-decision-quality",
        action="store_true",
        help="Skip cash-H2H win-rate / ROI reporting (legacy MAE/RMSE/R² only).",
    )
    parser.add_argument(
        "--emit-inactive-predictions",
        action="store_true",
        help=(
            "Also emit predictions for players in the season's cumulative-"
            "active pool who didn't play this week (phantom test rows). "
            "Enables the fully-symmetric prospective replay; see "
            "docs/SYMMETRIC_PROSPECTIVE_REPLAY.md and "
            "council-transcript-20260423-051434.md."
        ),
    )
    parser.add_argument(
        "--target-mode",
        choices=["fp", "util", "component"],
        default="fp",
        help=(
            "Target variable for training. 'fp' = predict fantasy points "
            "directly (default), 'util' = predict utilization score then "
            "convert to FP, 'component' = predict stat components "
            "(passing_yards, rushing_tds, etc.) and assemble FP via PPR weights."
        ),
    )

    parser.add_argument(
        "--qb-model",
        choices=["gbm", "ridge"],
        default=None,
        help=(
            "Override model type for QB only. Use 'gbm' to run GBM for QB "
            "while keeping the default model for other positions."
        ),
    )

    args = parser.parse_args()

    # --alpha is either a single float or a list of POS=VALUE pairs.
    alpha_tokens = args.alpha if isinstance(args.alpha, list) else [args.alpha]
    if len(alpha_tokens) == 1 and "=" not in str(alpha_tokens[0]):
        ridge_alpha = float(alpha_tokens[0])
    else:
        ridge_alpha = {}
        for tok in alpha_tokens:
            if "=" not in str(tok):
                parser.error(
                    f"--alpha token {tok!r} is not a POS=VALUE pair; mix of "
                    f"scalar and per-position forms is not allowed."
                )
            pos, val = str(tok).split("=", 1)
            ridge_alpha[pos.strip().upper()] = float(val)

    print("=" * 60)
    print("NFL Time-Series Backtester (Expanding Window)")
    print("=" * 60)
    print(f"  Season: {args.season or 'auto (latest)'}")
    print(f"  Model: {args.model}")
    if args.model == "ridge":
        print(f"  Ridge alpha: {ridge_alpha}")
    print(f"  Positions: {args.positions or 'all'}")
    print(f"  Target mode: {args.target_mode}")
    if args.qb_model:
        print(f"  QB model override: {args.qb_model}")
    print()

    pred_df, results = run_ts_backtest(
        season=args.season,
        model_type=args.model,
        positions=args.positions,
        verbose=not args.quiet,
        ridge_alpha=ridge_alpha,
        payout_multiplier=args.payout_multiplier,
        report_decision_quality=not args.no_decision_quality,
        emit_inactive_predictions=args.emit_inactive_predictions,
        target_mode=args.target_mode,
        qb_model=args.qb_model,
    )

    print(f"\nDone. {len(pred_df)} predictions generated.")
    if results.get("metrics"):
        m = results["metrics"]
        print(f"  MAE:  {m.get('mae', 'N/A')}")
        print(f"  RMSE: {m.get('rmse', 'N/A')}")
        print(f"  R²:   {m.get('r2', 'N/A')}")

    _print_decision_quality(results.get("decision_quality") or {})


def _print_decision_quality(dq: dict) -> None:
    """Pretty-print the cash-H2H decision-quality block.

    Silently returns when the block is absent (disabled run or pre-wiring
    results) or carries an error (e.g. insufficient complete weeks).
    """
    if not dq or dq.get("error"):
        return

    payout = dq.get("payout_multiplier", 1.8)
    break_even = dq.get("break_even_win_rate")
    n_weeks = dq.get("n_weeks", 0)

    header = f"\nDecision Quality (payout = {payout}×"
    if break_even is not None:
        header += f", break-even = {break_even * 100:.1f}%"
    header += f", n_weeks = {n_weeks})"
    print(header)

    tiers = [
        ("Oracle",      dq.get("vs_oracle") or {}),
        ("Hindsight",   dq.get("vs_hindsight") or {}),
        ("Replacement", dq.get("vs_replacement") or {}),
    ]
    print(f"  {'':<14}{'Win rate':>10}{'Record':>12}{'Binomial p':>14}{'ROI':>10}{'Avg margin':>14}")
    for name, t in tiers:
        if not t:
            continue
        wr = t.get("win_rate", 0) * 100
        record = f"{t.get('wins', 0)}-{t.get('losses', 0)}"
        p_val = t.get("p_value", float("nan"))
        roi = t.get("roi", 0) * 100
        margin = t.get("avg_margin", 0)
        print(
            f"  {name:<14}{wr:>9.1f}%{record:>12}{p_val:>14.4f}{roi:>+9.1f}%{margin:>+14.2f}"
        )

    weekly = dq.get("weekly_results") or []
    if weekly:
        marks = "".join(
            ("✓" if w.get("won_vs_hindsight") else "✗") for w in weekly
        )
        cum = weekly[-1].get("cumulative_win_rate_vs_hindsight", 0) * 100
        print(f"  Weekly vs hindsight: {marks}  ({cum:.1f}% cumulative)")


if __name__ == "__main__":
    main()
