#!/usr/bin/env python3
"""Does the game-outcome model (win prob / margin / total) help WEEKLY
player projections?

Test: add per-team-week features derived from the game-outcome models
(`src/models/game_outcome/`) to the single-week PPR model and measure OOS
MAE on the same folds Phase 6c uses (FINAL_CONFIG architecture/window/
weighting per position, DEFAULT_VALIDATION_SEASONS as held-out seasons).

Leakage discipline for the game-outcome predictions: the committed
artifacts under data/models/ are trained on 2006-2025 and would have SEEN
every validation season, so they are not used. Instead a walk-forward
out-of-fold table is built here -- for every season S, the logistic
(win/loss) and ridge (margin, total) arms are fit on completed games from
seasons strictly < S and predict every game of S. Player rows from season
S therefore only ever carry game-outcome predictions produced without
season S, in train as well as test.

Per-team-week features (team = the player's team):
    go_win_prob            P(team wins)
    go_pred_margin         predicted team margin (points)
    go_pred_total          predicted game total
    go_implied_team_total  (go_pred_total + go_pred_margin) / 2
and "residual vs the market" variants, since the game-outcome models take
spread_line/total_line as inputs and mostly reproduce them:
    go_margin_vs_market    go_pred_margin - (-spread)   (spread: negative = favoured)
    go_total_vs_market     go_pred_total - game_total

Arms (each fit fresh on the same X rows, differing only in columns):
    baseline      CAUSAL_FEATURES[pos]            (already includes spread /
                                                   implied_team_total / game_total)
    go_raw        baseline + the 4 raw go_* columns
    go_residual   baseline + the 2 *_vs_market columns
    go_all        baseline + all 6

Usage:
    python scripts/ablate_game_outcome_features.py
    python scripts/ablate_game_outcome_features.py --positions RB WR --seasons 2024 2025
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.models.game_outcome.features import build_game_outcome_rows, build_margin_total_rows, feature_columns
from src.models.game_outcome.models import GameMarginRidgeModel, GameOutcomeLogisticModel

logger = logging.getLogger(__name__)

ID_COLS = ["season", "week", "home_team", "away_team"]
GO_RAW = ["go_win_prob", "go_pred_margin", "go_pred_total", "go_implied_team_total"]
GO_RESIDUAL = ["go_margin_vs_market", "go_total_vs_market"]
ARMS: Dict[str, List[str]] = {
    "baseline": [],
    "go_raw": GO_RAW,
    "go_residual": GO_RESIDUAL,
    "go_all": GO_RAW + GO_RESIDUAL,
}
DEFAULT_OUTPUT = Path("data/experiments/game_outcome_feature_ablation.csv")
OOF_CACHE = Path("data/experiments/game_outcome_oof_predictions.csv")


def build_oof_game_predictions(
    min_predict_season: int = 2007,
    ridge_alpha_margin: Optional[float] = None,
    ridge_alpha_total: Optional[float] = None,
    logistic_C: Optional[float] = None,
) -> pd.DataFrame:
    """Walk-forward out-of-fold game predictions, one row per completed
    non-tie game from `min_predict_season` on: go_home_win_prob,
    go_home_margin, go_game_total. Season S is predicted by models fit on
    seasons < S only.

    Hyperparameters default to the Optuna-tuned values recorded in the
    committed artifacts' metadata (data/models/game_*_model_metadata.json)
    when present, else the config defaults -- same arms the site serves.
    """
    import json
    from config.settings import MODELS_DIR

    def _tuned(name: str, arm: str, key: str):
        p = MODELS_DIR / f"{name}_model_metadata.json"
        if p.exists():
            return json.loads(p.read_text()).get("tuned_params", {}).get(arm, {}).get(key)
        return None

    if ridge_alpha_margin is None:
        ridge_alpha_margin = _tuned("game_margin", "ridge", "alpha")
    if ridge_alpha_total is None:
        ridge_alpha_total = _tuned("game_total", "ridge", "alpha")
    if logistic_C is None:
        logistic_C = _tuned("game_outcome", "logistic", "C")

    wl = build_game_outcome_rows()
    mt = build_margin_total_rows()
    frame = mt.merge(wl[ID_COLS + ["home_win"]], on=ID_COLS, how="inner")
    feat_cols = feature_columns(frame)
    seasons = sorted(int(s) for s in frame["season"].unique())

    out = []
    for s in seasons:
        if s < min_predict_season:
            continue
        train = frame[frame["season"] < s]
        test = frame[frame["season"] == s]
        if train.empty or test.empty:
            continue
        X_tr, X_te = train[feat_cols], test[feat_cols]
        clf = GameOutcomeLogisticModel(C=logistic_C).fit(X_tr, train["home_win"].to_numpy())
        margin = GameMarginRidgeModel(alpha=ridge_alpha_margin).fit(X_tr, train["home_margin"].to_numpy())
        total = GameMarginRidgeModel(alpha=ridge_alpha_total).fit(X_tr, train["game_total"].to_numpy())
        pred = test[ID_COLS].copy()
        pred["go_home_win_prob"] = clf.predict_proba(X_te)[:, 1]
        pred["go_home_margin"] = margin.predict(X_te)
        pred["go_game_total"] = total.predict(X_te)
        # Observed labels kept only for the sanity report below, never joined to players.
        pred["_home_win"] = test["home_win"].to_numpy()
        pred["_home_margin"] = test["home_margin"].to_numpy()
        pred["_game_total"] = test["game_total"].to_numpy()
        pred["_spread_line"] = test["spread_line"].to_numpy()
        pred["_total_line"] = test["total_line"].to_numpy()
        out.append(pred)
        print(f"  game-outcome OOF: season {s} <- trained on {int(train.season.min())}-{s - 1} "
              f"({len(train)} games), predicted {len(test)}", flush=True)
    return pd.concat(out, ignore_index=True)


def oof_sanity_report(oof: pd.DataFrame) -> None:
    """Confirms the OOF predictions behave like the committed backtest
    (accuracy ~ Vegas, margin/total MAE ~ market) so a broken join or fit
    can't masquerade as 'no signal'."""
    acc = ((oof.go_home_win_prob > 0.5).astype(int) == oof._home_win).mean()
    vegas_acc = ((oof._spread_line > 0).astype(int) == oof._home_win)[oof._spread_line != 0].mean()
    mae_m = (oof.go_home_margin - oof._home_margin).abs().mean()
    mae_m_mkt = (oof._spread_line - oof._home_margin).abs().mean()
    mae_t = (oof.go_game_total - oof._game_total).abs().mean()
    mae_t_mkt = (oof._total_line - oof._game_total).abs().mean()
    print(f"\nOOF sanity ({int(oof.season.min())}-{int(oof.season.max())}, n={len(oof)}): "
          f"win acc {acc:.3f} (vegas {vegas_acc:.3f}); margin MAE {mae_m:.2f} (market {mae_m_mkt:.2f}); "
          f"total MAE {mae_t:.2f} (market {mae_t_mkt:.2f})")
    print(f"  corr(go_home_margin, spread_line) = {oof[['go_home_margin', '_spread_line']].corr().iloc[0, 1]:.3f}; "
          f"corr(go_game_total, total_line) = {oof[['go_game_total', '_total_line']].corr().iloc[0, 1]:.3f}")


def to_team_week(oof: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, week, team) with the team's-perspective features."""
    home = pd.DataFrame({
        "season": oof.season, "week": oof.week, "team": oof.home_team,
        "go_win_prob": oof.go_home_win_prob,
        "go_pred_margin": oof.go_home_margin,
        "go_pred_total": oof.go_game_total,
    })
    away = pd.DataFrame({
        "season": oof.season, "week": oof.week, "team": oof.away_team,
        "go_win_prob": 1.0 - oof.go_home_win_prob,
        "go_pred_margin": -oof.go_home_margin,
        "go_pred_total": oof.go_game_total,
    })
    tw = pd.concat([home, away], ignore_index=True)
    tw["go_implied_team_total"] = (tw.go_pred_total + tw.go_pred_margin) / 2
    dup = tw.duplicated(["season", "week", "team"]).sum()
    if dup:
        raise ValueError(f"{dup} duplicate (season, week, team) rows in game-outcome table")
    return tw


def attach_go_features(df: pd.DataFrame, tw: pd.DataFrame, label: str) -> pd.DataFrame:
    """Left-join the team-week table onto player rows and derive the
    market-residual columns from the player frame's own Vegas features
    (spread: negative = favoured, so expected margin = -spread)."""
    out = df.merge(tw, on=["season", "week", "team"], how="left")
    if len(out) != len(df):
        raise ValueError("game-outcome join changed row count")
    out.index = df.index
    out["go_margin_vs_market"] = out["go_pred_margin"] + out["spread"]
    out["go_total_vs_market"] = out["go_pred_total"] - out["game_total"]
    cov = out["go_win_prob"].notna().mean()
    print(f"    {label}: game-outcome features joined on {cov:.1%} of {len(out)} rows")
    return out


def run(positions: Optional[Sequence[str]], seasons: Sequence[int], output_path: Path,
        arms: Dict[str, List[str]] = ARMS) -> pd.DataFrame:
    from config.settings import CAUSAL_FEATURES, POSITIONS
    from src.models.single_week_ppr.evaluate import (
        DEFAULT_VALIDATION_SEASONS, FoldFailureTracker, _append_row_to_csv,
        _architectures_for_fold, _build_feature_matrices, compute_metrics, run_fold,
    )
    from src.models.single_week_ppr.final_config import FINAL_CONFIG
    from src.models.single_week_ppr.windows import compute_recency_weights, window_to_season_list
    from src.utils.database import DatabaseManager
    from src.utils.leakage import filter_feature_columns

    seasons = list(seasons) if seasons else list(DEFAULT_VALIDATION_SEASONS)
    positions = list(positions) if positions else list(POSITIONS)

    if OOF_CACHE.exists():
        oof = pd.read_csv(OOF_CACHE)
        print(f"Loaded cached game-outcome OOF predictions from {OOF_CACHE} ({len(oof)} games)")
    else:
        print("Building walk-forward game-outcome OOF predictions...")
        oof = build_oof_game_predictions()
        OOF_CACHE.parent.mkdir(parents=True, exist_ok=True)
        oof.to_csv(OOF_CACHE, index=False)
    oof_sanity_report(oof)
    tw = to_team_week(oof)

    rows: List[dict] = []
    tracker = FoldFailureTracker("game-outcome feature ablation")
    for position in positions:
        cfg = FINAL_CONFIG[position]
        available = sorted(
            DatabaseManager().get_all_players_for_training(position=position)["season"].dropna().unique().tolist()
        )
        for season in seasons:
            print(f"\n=== {position} / test_season={season} / {cfg} ===", flush=True)
            train_seasons = window_to_season_list(cfg["window"], season, available)
            if not train_seasons:
                logger.warning("Skipping %s/%s: no training seasons", position, season)
                continue
            try:
                train_df, test_df, _, _ = run_fold(
                    position, season, False, train_seasons_override=train_seasons,
                    fit_existing_models=False,
                )
            except Exception as e:
                tracker.record(position, season, e)
                continue

            pos_train = train_df[train_df["position"] == position].reset_index(drop=True)
            pos_test = test_df[test_df["position"] == position]
            if len(pos_test) < 20:
                logger.warning("Skipping %s/%s: only %d test rows", position, season, len(pos_test))
                continue
            pos_train = attach_go_features(pos_train, tw, "train")
            pos_test = attach_go_features(pos_test, tw, "test")

            base_cols = filter_feature_columns(CAUSAL_FEATURES.get(position, []))
            base_cols = [c for c in base_cols if c in pos_train.columns and c in pos_test.columns]
            sample_weight = compute_recency_weights(pos_train["season"], cfg["weighting"])

            for arm, extra in arms.items():
                cols = base_cols + [c for c in extra if c not in base_cols]
                X_train, y_train, X_test, y_test = _build_feature_matrices(pos_train, pos_test, cols)
                model = _architectures_for_fold()[cfg["architecture"]]
                model.fit(X_train, y_train, sample_weight=sample_weight)
                pred = pd.Series(model.predict(X_test), index=X_test.index)
                row = {"position": position, "season": season, "model": cfg["architecture"],
                       "arm": arm, "n_features": len(cols), **compute_metrics(y_test, pred)}
                rows.append(row)
                _append_row_to_csv(row, output_path)
                print(f"    {arm:12s} n_feat={len(cols):3d} mae={row['mae']:.4f} rmse={row['rmse']:.4f} "
                      f"r2={row['r2']:.4f} spearman={row['spearman']:.4f}", flush=True)

    result = pd.DataFrame(rows)
    tracker.report(output_path)
    return result


def summarize(result: pd.DataFrame) -> None:
    if result.empty:
        print("No results.")
        return
    print("\n" + "=" * 72)
    print("MAE by position / arm (mean across seasons) and delta vs baseline")
    print("=" * 72)
    piv = result.pivot_table(index="position", columns="arm", values="mae", aggfunc="mean")
    cols = [a for a in ARMS if a in piv.columns]
    piv = piv[cols]
    delta = piv.sub(piv["baseline"], axis=0).drop(columns="baseline").add_prefix("d_")
    print(pd.concat([piv, delta], axis=1).round(4).to_string())
    print("\nPer-season MAE deltas vs baseline (negative = game-outcome features helped):")
    per = result.pivot_table(index=["position", "season"], columns="arm", values="mae")
    print(per[cols].sub(per["baseline"], axis=0).drop(columns="baseline").round(4).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--positions", nargs="+", default=None)
    parser.add_argument("--seasons", nargs="+", type=int, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rebuild-oof", action="store_true", help="ignore the cached OOF table")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    if args.rebuild_oof and OOF_CACHE.exists():
        OOF_CACHE.unlink()
    if args.output.exists():
        args.output.unlink()
    result = run(args.positions, args.seasons, args.output)
    summarize(result)
    print(f"\nRow-level results: {args.output}")


if __name__ == "__main__":
    main()
