#!/usr/bin/env python3
"""Train-only affine calibration for the participation-to-PPR diagnostic."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr


def score(g: pd.DataFrame, pred_col: str) -> dict:
    g = g.dropna(subset=["fantasy_points", pred_col])
    if g.empty:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "r2": np.nan,
                "spearman": np.nan, "bias": np.nan, "actual_mean": np.nan,
                "prediction_mean": np.nan}
    y, p = g.fantasy_points.to_numpy(), g[pred_col].to_numpy()
    return {
        "n": int(len(g)),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(mean_squared_error(y, p) ** 0.5),
        "r2": float(r2_score(y, p)) if len(g) > 1 else np.nan,
        "spearman": float(spearmanr(y, p).correlation) if len(g) > 1 else np.nan,
        "bias": float(np.mean(p - y)),
        "actual_mean": float(np.mean(y)),
        "prediction_mean": float(np.mean(p)),
    }


def main() -> int:
    root = Path("data/experiments/participation_system/phase3/participation_to_ppr")
    rows = pd.read_csv(root / "matched_inputs.csv")
    rates = pd.read_csv(root / "conversion_rates.csv").rename(columns={"test_season": "season"})
    rates = rates[["season", "position", "prior_ppr_per_snap", "prior_team_snaps_per_game"]]
    rows = rows.merge(rates, on=["season", "position"], how="left", validate="many_to_one")
    rows["raw_prediction"] = (
        rows.prediction_share * rows.prior_ppr_per_snap * rows.prior_team_snaps_per_game
    )
    rows["affine_prediction"] = np.nan
    coefficient_rows, metric_rows = [], []
    seasons = sorted(rows.season.unique())
    for season in seasons:
        for position, test in rows[rows.season.eq(season)].groupby("position"):
            test = test.copy()
            prior_counts = rows[rows.season.lt(season)].groupby("player_id").size()
            test["prior_player_weeks"] = test.player_id.map(prior_counts).fillna(0).astype(int)
            train = rows[(rows.season < season) & rows.position.eq(position)].dropna(
                subset=["raw_prediction", "fantasy_points"]
            )
            if len(train) < 20 or test.empty:
                continue
            model = LinearRegression().fit(
                train[["raw_prediction"]], train["fantasy_points"]
            )
            test["affine_prediction"] = model.predict(test[["raw_prediction"]])
            rows.loc[test.index, "affine_prediction"] = test["affine_prediction"]
            coefficient_rows.append({
                "test_season": int(season),
                "position": position,
                "n_train": int(len(train)),
                "intercept": float(model.intercept_),
                "slope": float(model.coef_[0]),
            })
            for subgroup, group in [
                ("all", test),
                ("sparse", test[test.prior_player_weeks < 3]),
                ("established", test[test.prior_player_weeks >= 3]),
            ]:
                for method, col in [("raw", "raw_prediction"), ("affine", "affine_prediction")]:
                    metric_rows.append({
                        "season": int(season), "position": position,
                        "subgroup": subgroup, "method": method,
                        **score(group, col),
                    })
    metrics = pd.DataFrame(metric_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    metrics.to_csv(root / "affine_metrics.csv", index=False)
    coefficients.to_csv(root / "affine_coefficients.csv", index=False)
    rows.to_csv(root / "affine_predictions.csv", index=False)
    report = {
        "diagnostic": "train_only_affine_calibration_of_participation_to_ppr",
        "fit": "fantasy_points ~ intercept + slope * raw_participation_conversion",
        "fit_rule": "position-specific, using only seasons strictly before each test season",
        "promotion_gate": "none",
        "n_test_rows": int(rows.affine_prediction.notna().sum()),
    }
    (root / "affine_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(metrics[metrics.subgroup.eq("all")].groupby(["position", "method"]).agg(
        n=("n", "sum"), mae=("mae", "mean"), rmse=("rmse", "mean"),
        r2=("r2", "mean"), bias=("bias", "mean"), spearman=("spearman", "mean")
    ).round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
