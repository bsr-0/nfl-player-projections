#!/usr/bin/env python3
"""
Directional ADP-lift gate: sweep snake_draft_sim across ranking modes
and draft slots.  For each (season, ranking_mode), run ModelBot at
every draft slot 1..12 and report the distribution of ModelBot's
finish rank among the 12 teams, plus actual-points lift vs the
11 ADPBot mean.

Answers the question: "does the model have directional pre-draft
signal before we invest in Step 4 calibration?"

Usage:
    python scripts/draft_sim_mode_sweep.py --seasons 2024 2025
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

spec = importlib.util.spec_from_file_location(
    "snake_draft_sim",
    PROJECT_ROOT / "scripts" / "snake_draft_sim.py",
)
sd = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sd
spec.loader.exec_module(sd)

da_spec = importlib.util.spec_from_file_location(
    "draft_advisor",
    PROJECT_ROOT / "scripts" / "draft_advisor.py",
)
da = importlib.util.module_from_spec(da_spec)
sys.modules[da_spec.name] = da
da_spec.loader.exec_module(da)


def find_predictions_csv(season: int) -> Path | None:
    matches = sorted(
        (PROJECT_ROOT / "data" / "backtest_results").glob(
            f"ts_backtest_{season}_*_predictions.csv"
        )
    )
    return matches[-1] if matches else None


def run_slot_sweep(
    season: int,
    ranking: str,
    csv_path: Path | None,
    vona_picker=None,
    projector_path: Path | None = None,
) -> Dict:
    adp = sd.load_adp_board(season)
    scrape_date = adp["scrape_date"].iloc[0]
    if ranking == "preseason_ml":
        projections = sd.load_preseason_projections(
            season,
            adp_df=adp,
            projection_mode="ml",
            actuals_season=season,
            projector_path=projector_path,
        )
    elif ranking == "preseason_ppg17":
        projections = sd.load_preseason_projections(
            season,
            adp_df=adp,
            projection_mode="ppg17",
            actuals_season=season,
        )
    else:
        if csv_path is None:
            raise FileNotFoundError(
                f"No backtest predictions CSV available for season={season}"
            )
        projections = sd.load_model_projections(
            csv_path, ranking=ranking, season=season
        )
    board = sd.build_draft_board(adp, projections)

    ranks: List[int] = []
    model_totals: List[float] = []
    adp_totals: List[float] = []
    wins_vs_adp_mean = 0
    for slot in range(1, sd.TEAMS + 1):
        teams = sd.run_draft(
            board, model_slot=slot, model_pick_fn=vona_picker,
        )
        summary = sd.summarize(teams)
        ranks.append(summary["model_rank_of_12"])
        model_totals.append(summary["model_actual_total"])
        adp_totals.append(summary["adp_mean_actual_total"])
        if summary["model_actual_total"] > summary["adp_mean_actual_total"]:
            wins_vs_adp_mean += 1
    return {
        "season": season,
        "scrape_date": scrape_date,
        "ranking": ranking,
        "slot_ranks": ranks,
        "mean_rank": round(statistics.mean(ranks), 2),
        "best_rank": min(ranks),
        "worst_rank": max(ranks),
        "mean_model_actual": round(statistics.mean(model_totals), 1),
        "mean_adp_actual": round(statistics.mean(adp_totals), 1),
        "lift_vs_adp_mean": round(
            statistics.mean(model_totals) - statistics.mean(adp_totals), 1
        ),
        "wins_vs_adp_mean_slots": wins_vs_adp_mean,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", type=int, default=[2024, 2025])
    ap.add_argument("--modes", nargs="+",
                    default=["season_sum", "preseason_ml", "preseason_ppg17",
                             "week1", "prior_season", "vona_advisor",
                             "vona_week1"])
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    results: List[Dict] = []
    for season in args.seasons:
        csv_path = find_predictions_csv(season)
        for mode in args.modes:
            picker = None
            ranking = mode
            if mode.startswith("vona_"):
                if csv_path is None:
                    print(
                        f"No predictions CSV for {season}, skipping {mode}.",
                        file=sys.stderr,
                    )
                    continue
                # VONABot picks by dynamic VONA instead of raw ranking.
                # vona_advisor = season_sum projections (hindsight)
                # vona_week1   = week1 projections (honest pre-draft)
                ranking = mode.replace("vona_", "").replace(
                    "advisor", "season_sum")
                adp = sd.load_adp_board(season)
                picker = da.make_vona_picker(adp)
            elif mode not in {"preseason_ml", "preseason_ppg17"} and csv_path is None:
                print(
                    f"No predictions CSV for {season}, skipping {mode}.",
                    file=sys.stderr,
                )
                continue
            r = run_slot_sweep(season, ranking, csv_path,
                               vona_picker=picker)
            r["ranking"] = mode  # label as vona_*, not underlying ranking
            results.append(r)
            print(
                f"  {season}  {mode:>12}  "
                f"mean_rank={r['mean_rank']:.2f}  "
                f"best={r['best_rank']}  "
                f"worst={r['worst_rank']}  "
                f"mean_model_pts={r['mean_model_actual']:.1f}  "
                f"mean_adp_pts={r['mean_adp_actual']:.1f}  "
                f"lift={r['lift_vs_adp_mean']:+.1f}  "
                f"beats_adp_mean_in={r['wins_vs_adp_mean_slots']}/{sd.TEAMS}_slots"
            )
        print()

    print("Legend:")
    print(
        f"  mean_rank:  average ModelBot finish rank (1=best, {sd.TEAMS}=worst)"
        f" across the {sd.TEAMS} draft slots.  Pure-luck null is "
        f"{(sd.TEAMS + 1) / 2:.1f}."
    )
    print(
        f"  lift:       mean_model_pts - mean_adp_pts (avg over {sd.TEAMS} slots)."
    )
    print("  beats_adp_mean_in: # of draft slots where ModelBot's roster"
          f" actual-total exceeds the {sd.TEAMS - 1} ADPBot mean.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2))
        print(f"\nJSON written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
