#!/usr/bin/env python3
"""Audit historical and market-divergence behavior for the preseason projector.

Usage:
    python scripts/audit_preseason_projector.py --season 2026
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.preseason_projector import PreseasonProjector


def residual_report() -> dict:
    proj, pairs = PreseasonProjector.train(seasons=list(range(2018, 2026)))
    prepared = proj._prepare_feature_frame(pairs)
    report: dict[str, dict] = {
        "selected_variant": proj.get_selection_report().get("selected_variant"),
        "selection_report": proj.get_selection_report(),
        "upstream_audit": proj.get_upstream_audit_report(),
        "positions": {},
    }

    for pos in ["QB", "RB", "WR", "TE"]:
        pos_df = prepared[prepared["position"] == pos].copy()
        if pos_df.empty or pos not in proj.feature_names:
            continue
        details = proj.predict_with_details(pos_df, pos)
        pos_df = pos_df.copy()
        for column in details.columns:
            if column in pos_df.columns:
                pos_df[column] = details[column].to_numpy()
            else:
                pos_df[column] = details[column].to_numpy()
        pos_df["error"] = pos_df["pred"] - pos_df["season_total"]
        pos_df["base_error"] = pos_df["base_pred"] - pos_df["season_total"]
        pos_df["market_delta"] = pos_df["pred"] - pos_df["market_anchor"]
        correlations = {}
        for col in [
            "age",
            "ppg",
            "games_played",
            "years_from_peak",
            "veteran_flag",
            "post_peak_ppg",
            "confidence_score",
            "targets_pg" if pos in {"RB", "WR", "TE"} else "passing_yards_pg",
        ]:
            if col in pos_df.columns:
                correlations[col] = round(float(pos_df[[col, "error"]].corr().iloc[0, 1]), 4)

        report["positions"][pos] = {
            "n": int(len(pos_df)),
            "top_overpreds": (
                pos_df.sort_values("error", ascending=False)
                .head(10)[
                    [
                        c
                        for c in [
                            "player_id",
                            "player_name",
                            "prior_season",
                            "curr_season",
                            "age",
                            "ppg",
                            "games_played",
                            "confidence_score",
                            "market_anchor",
                            "base_pred",
                            "pred",
                            "season_total",
                            "error",
                        ]
                        if c in pos_df.columns
                    ]
                ]
                .round(3)
                .to_dict(orient="records")
            ),
            "error_correlations": correlations,
            "summary": proj.get_upstream_audit_report().get("by_position", {}).get(pos, {}),
            "calibrator_audit": proj.get_upstream_audit_report()
            .get("calibrator_audits", {})
            .get(pos, {}),
        }
    return report


def board_report(season: int) -> dict:
    board_path = PROJECT_ROOT / "docs" / "data" / "board.json"
    calibration_path = PROJECT_ROOT / "docs" / "data" / "calibration.json"
    board = json.loads(board_path.read_text())
    calibration = json.loads(calibration_path.read_text()) if calibration_path.exists() else {}

    unresolved = []
    for player in board:
        market_proj = player.get("marketProj") or 0
        if market_proj <= 0:
            continue
        flags = player.get("calibrationFlags", {})
        final_divergence = abs(player["proj"] - market_proj) / max(market_proj, 1.0)
        raw_divergence = abs(player["rawProj"] - market_proj) / max(market_proj, 1.0)
        unresolved.append(
            {
                "name": player["n"],
                "position": player["p"],
                "ecr": player["ecr"],
                "mr": player["mr"],
                "age": player.get("age"),
                "usage": player.get("usage"),
                "market_source": player["calibrationFlags"]["marketSource"],
                "raw_divergence": round(raw_divergence, 4),
                "final_divergence": round(final_divergence, 4),
                "unresolved": bool(flags.get("unresolvedDisplayDivergence")),
            }
        )

    unresolved_rows = [row for row in unresolved if row["unresolved"]]
    displayable = [row for row in unresolved if row["ecr"] <= 150 or row["mr"] <= 150]
    displayable_unresolved = [row for row in displayable if row["unresolved"]]

    return {
        "season": season,
        "calibration_summary": calibration.get("summary", {}),
        "market_source_counts_all": dict(Counter(row["market_source"] for row in unresolved)),
        "market_source_counts_unresolved": dict(Counter(row["market_source"] for row in unresolved_rows)),
        "displayable_count": len(displayable),
        "displayable_unresolved_count": len(displayable_unresolved),
        "top_unresolved": sorted(
            unresolved_rows,
            key=lambda row: row["final_divergence"],
            reverse=True,
        )[:25],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "docs" / "data" / "preseason_projector_audit.json",
    )
    args = parser.parse_args()

    report = {
        "residual_report": residual_report(),
        "board_report": board_report(args.season),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
