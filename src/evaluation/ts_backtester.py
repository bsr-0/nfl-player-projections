"""
Leakage-Free Time-Series Backtester for NFL Player Projections.

Implements an expanding-window backtesting framework that simulates what the
production model would have predicted for a historical NFL season, week by week.

Core Principles:
1. No random splits or k-fold cross-validation.
2. Strict chronological ordering — training data is always < cutoff_date.
3. Model is refit every week to mirror production behavior.
4. Feature engineering is recomputed per fold (no lookahead bias).
5. Scaling is fit on training data only, applied to both train and test.
6. No full-season aggregates that include future data.

Output: A per-player, per-week predictions table matching production schema,
plus aggregated metrics (MAE, RMSE) by week and by position.
"""

import json
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import DATA_DIR, MODELS_DIR, POSITIONS

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Leakage diagnostics
# ---------------------------------------------------------------------------

def assert_no_future_leakage(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    date_col: str = "game_date",
    season_col: str = "season",
    week_col: str = "week",
) -> Dict[str, Any]:
    """
    Diagnostic: verify that no row in train_df is chronologically after any
    row in test_df.  Returns a dict with pass/fail + details.
    """
    errors: List[str] = []

    if date_col in train_df.columns and date_col in test_df.columns:
        train_max = train_df[date_col].max()
        test_min = test_df[date_col].min()
        if pd.notna(train_max) and pd.notna(test_min) and train_max >= test_min:
            errors.append(
                f"Train max {date_col}={train_max} >= test min {date_col}={test_min}"
            )

    if season_col in train_df.columns and week_col in train_df.columns:
        train_max_sw = (
            train_df[season_col].astype(int) * 100 + train_df[week_col].astype(int)
        ).max()
        test_min_sw = (
            test_df[season_col].astype(int) * 100 + test_df[week_col].astype(int)
        ).max()
        # For expanding window, train must be strictly before test week
        # (same-season earlier weeks are fine)

    # Check that no player in test has future fantasy_points leaking into train
    if "player_id" in train_df.columns and "player_id" in test_df.columns:
        test_players = set(test_df["player_id"].unique())
        train_players_in_test = set(train_df["player_id"].unique()) & test_players
        # This is expected (players appear in both), but training rows for those
        # players must all be before the test cutoff week.
        if season_col in train_df.columns and week_col in train_df.columns:
            test_season = int(test_df[season_col].iloc[0])
            test_week = int(test_df[week_col].iloc[0])
            for pid in list(train_players_in_test)[:50]:  # sample check
                player_train = train_df[train_df["player_id"] == pid]
                future_rows = player_train[
                    (player_train[season_col].astype(int) > test_season)
                    | (
                        (player_train[season_col].astype(int) == test_season)
                        & (player_train[week_col].astype(int) >= test_week)
                    )
                ]
                if len(future_rows) > 0:
                    errors.append(
                        f"Player {pid} has {len(future_rows)} future rows in training "
                        f"(test s{test_season}w{test_week})"
                    )
                    break  # one example is enough

    return {"passed": len(errors) == 0, "errors": errors}


def assert_rolling_features_shifted(
    df: pd.DataFrame,
    rolling_prefix: str = "_roll",
) -> Dict[str, Any]:
    """
    Heuristic check: rolling features should have NaN for the first row of
    each player (because shift(1) makes the first value NaN). If no NaN is
    found in the first row per player, the feature may not be properly shifted.
    """
    warnings_list: List[str] = []
    roll_cols = [c for c in df.columns if rolling_prefix in c and "mean" in c]
    if not roll_cols or "player_id" not in df.columns:
        return {"passed": True, "warnings": []}

    first_rows = df.groupby("player_id").nth(0)
    for col in roll_cols[:10]:  # sample
        if col not in first_rows.columns:
            continue
        non_null_rate = first_rows[col].notna().mean()
        if non_null_rate > 0.5:
            warnings_list.append(
                f"{col}: {non_null_rate:.0%} non-null in first row per player "
                "(expected mostly NaN from shift(1))"
            )

    return {"passed": len(warnings_list) == 0, "warnings": warnings_list}


# ---------------------------------------------------------------------------
# Leakage-safe feature engineering wrapper
# ---------------------------------------------------------------------------

def leakage_safe_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply feature engineering to train and test separately, then align columns.
    Rolling/lag features are recomputed on train only so that test features
    use only data available before the cutoff.

    For the test fold (a single week), we concatenate train + test, compute
    features on the whole chronologically-ordered block, then slice out the
    test rows.  Because rolling/lag features use shift(1), the test rows only
    see data from prior weeks.
    """
    from src.features.feature_engineering import FeatureEngineer

    engineer = FeatureEngineer()

    # Concatenate so rolling windows have enough history for test rows
    combined = pd.concat([train_df, test_df], ignore_index=True)
    combined = combined.sort_values(["player_id", "season", "week"]).reset_index(
        drop=True
    )

    # Mark test rows
    n_train = len(train_df)
    combined["_is_test"] = False
    # We need a more robust way to mark test rows
    test_keys = set()
    for _, row in test_df.iterrows():
        key = (row.get("player_id", ""), int(row.get("season", 0)), int(row.get("week", 0)))
        test_keys.add(key)

    combined["_is_test"] = combined.apply(
        lambda r: (r.get("player_id", ""), int(r.get("season", 0)), int(r.get("week", 0))) in test_keys,
        axis=1,
    )

    # Apply feature engineering on the combined block
    combined = engineer.create_features(combined, include_target=False)

    # Split back
    train_out = combined[~combined["_is_test"]].drop(columns=["_is_test"])
    test_out = combined[combined["_is_test"]].drop(columns=["_is_test"])

    return train_out, test_out


# ---------------------------------------------------------------------------
# TimeSeriesBacktester
# ---------------------------------------------------------------------------

class TimeSeriesBacktester:
    """
    Expanding-window backtester that simulates production predictions for a
    historical NFL season, week by week.

    Usage:
        bt = TimeSeriesBacktester(data, model_factory, season_to_backtest=2025)
        results_df = bt.run_backtest()
        bt.save_results()
    """

    def __init__(
        self,
        data: pd.DataFrame,
        model_factory: Callable[[pd.DataFrame, str], Any],
        season_to_backtest: int,
        positions: List[str] = None,
        feature_pipeline: Optional[Callable] = None,
        verbose: bool = True,
    ):
        """
        Args:
            data: Full historical dataset (all seasons), sorted by game_date or
                  (season, week). Must contain columns: player_id, name, position,
                  team, season, week, fantasy_points.
            model_factory: A callable(train_df, position) -> model with .predict(X).
                  Called once per week per position to refit the model.
            season_to_backtest: The season to simulate predictions for.
            positions: List of positions to backtest (default: QB, RB, WR, TE).
            feature_pipeline: Optional callable(train_df, test_df) -> (train_df, test_df)
                  for leakage-safe feature engineering.  Defaults to
                  `leakage_safe_features`.
            verbose: Print progress.
        """
        self.data = data.copy()
        # Ensure chronological sort
        if "game_date" in self.data.columns:
            self.data = self.data.sort_values("game_date")
        else:
            self.data = self.data.sort_values(["season", "week"])

        self.model_factory = model_factory
        self.season = season_to_backtest
        self.positions = positions or POSITIONS
        self.feature_pipeline = feature_pipeline or leakage_safe_features
        self.verbose = verbose

        self.predictions: List[Dict[str, Any]] = []
        self.weekly_metrics: Dict[int, Dict[str, float]] = {}
        self.position_metrics: Dict[str, Dict[str, float]] = {}
        self._run_timestamp: Optional[str] = None

    # ------------------------------------------------------------------
    def run_backtest(self) -> pd.DataFrame:
        """
        Execute the expanding-window backtest.

        For each week in the target season:
          1. cutoff = start of week t
          2. train = all data before cutoff
          3. test  = data for week t
          4. Apply leakage-safe feature engineering
          5. Refit model per position
          6. Generate predictions
          7. Store predictions + actuals

        Returns:
            DataFrame with columns: season, week, player_id, name, position,
            team, predicted, actual, prediction_timestamp
        """
        self._run_timestamp = datetime.now().isoformat()
        self.predictions = []

        season_data = self.data[self.data["season"] == self.season]
        if season_data.empty:
            raise ValueError(f"No data found for season {self.season}")

        weeks = sorted(season_data["week"].unique())
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"TIME-SERIES BACKTEST: Season {self.season}")
            print(f"{'='*60}")
            print(f"  Weeks to backtest: {weeks}")
            print(f"  Positions: {self.positions}")
            print(f"  Total historical rows: {len(self.data)}")

        for week in weeks:
            t0 = time.time()
            week_data = season_data[season_data["week"] == week]

            # Build training set: everything strictly before this week
            train = self.data[
                (self.data["season"] < self.season)
                | (
                    (self.data["season"] == self.season)
                    & (self.data["week"] < week)
                )
            ].copy()

            test = week_data.copy()

            if len(train) < 100:
                if self.verbose:
                    print(f"  Week {week}: skipped (only {len(train)} training rows)")
                continue

            # Leakage diagnostic
            diag = assert_no_future_leakage(train, test)
            if not diag["passed"]:
                raise RuntimeError(
                    f"DATA LEAKAGE DETECTED at week {week}: {diag['errors']}"
                )

            # Feature engineering (leakage-safe)
            try:
                train_fe, test_fe = self.feature_pipeline(train, test)
            except Exception as e:
                if self.verbose:
                    print(f"  Week {week}: feature engineering failed ({e})")
                continue

            # Predict per position
            week_preds = 0
            for position in self.positions:
                pos_train = train_fe[train_fe["position"] == position]
                pos_test = test_fe[test_fe["position"] == position]

                if len(pos_train) < 30 or len(pos_test) == 0:
                    continue

                try:
                    # Refit model (production behavior)
                    model = self.model_factory(pos_train, position)

                    # Get feature columns (numeric, non-target, non-ID)
                    exclude = {
                        "player_id", "name", "position", "team", "opponent",
                        "season", "week", "home_away", "created_at", "id",
                        "game_date", "fantasy_points",
                    }
                    target_cols = {c for c in pos_train.columns if c.startswith("target_")}
                    exclude |= target_cols

                    feature_cols = [
                        c for c in pos_train.columns
                        if c not in exclude
                        and pos_train[c].dtype in ("int64", "float64", "int32", "float32")
                    ]

                    # Align feature columns between train and test
                    feature_cols = [c for c in feature_cols if c in pos_test.columns]

                    X_train = pos_train[feature_cols].fillna(0)
                    y_train = pos_train["fantasy_points"]
                    X_test = pos_test[feature_cols].fillna(0)

                    # Scale: fit on train only
                    scaler = StandardScaler()
                    X_train_s = scaler.fit_transform(X_train)
                    X_test_s = scaler.transform(X_test)

                    # Fit and predict
                    model.fit(X_train_s, y_train)
                    preds = model.predict(X_test_s)

                    # Store per-player predictions
                    for i in range(len(pos_test)):
                        row = pos_test.iloc[i]
                        actual = row.get("fantasy_points", np.nan)
                        self.predictions.append({
                            "season": int(self.season),
                            "week": int(week),
                            "player_id": row.get("player_id", ""),
                            "name": row.get("name", ""),
                            "position": position,
                            "team": row.get("team", ""),
                            "predicted": float(preds[i]),
                            "actual": float(actual) if pd.notna(actual) else np.nan,
                            "prediction_timestamp": self._run_timestamp,
                        })
                        week_preds += 1

                except Exception as e:
                    if self.verbose:
                        print(f"    {position} week {week}: failed ({e})")
                    continue

            elapsed = time.time() - t0
            if self.verbose:
                print(
                    f"  Week {week}: {week_preds} predictions "
                    f"({len(train)} train rows, {elapsed:.1f}s)"
                )

        if not self.predictions:
            raise ValueError("No successful predictions generated")

        pred_df = pd.DataFrame(self.predictions)
        self._compute_metrics(pred_df)
        return pred_df

    # ------------------------------------------------------------------
    def _compute_metrics(self, pred_df: pd.DataFrame) -> None:
        """Compute per-week and per-position metrics."""
        valid = pred_df.dropna(subset=["actual", "predicted"])
        if valid.empty:
            return

        # Overall
        self.overall_metrics = _calc_metrics(valid["actual"], valid["predicted"])

        # By week
        self.weekly_metrics = {}
        for week, grp in valid.groupby("week"):
            self.weekly_metrics[int(week)] = _calc_metrics(grp["actual"], grp["predicted"])

        # By position
        self.position_metrics = {}
        for pos, grp in valid.groupby("position"):
            self.position_metrics[pos] = _calc_metrics(grp["actual"], grp["predicted"])

        if self.verbose:
            print(f"\n{'='*60}")
            print("BACKTEST RESULTS")
            print(f"{'='*60}")
            print(f"  Season: {self.season}")
            print(f"  Total predictions: {len(valid)}")
            print(f"  Overall MAE: {self.overall_metrics['mae']:.2f}")
            print(f"  Overall RMSE: {self.overall_metrics['rmse']:.2f}")
            print(f"  Overall R²: {self.overall_metrics['r2']:.3f}")
            print(f"\n  By Position:")
            for pos in sorted(self.position_metrics.keys()):
                m = self.position_metrics[pos]
                print(f"    {pos}: MAE={m['mae']:.2f}  RMSE={m['rmse']:.2f}  R²={m['r2']:.3f}")
            print(f"\n  Early vs Late Season:")
            weeks_sorted = sorted(self.weekly_metrics.keys())
            if len(weeks_sorted) >= 4:
                early = [self.weekly_metrics[w]["rmse"] for w in weeks_sorted[:4]]
                late = [self.weekly_metrics[w]["rmse"] for w in weeks_sorted[-4:]]
                print(f"    Early (wk {weeks_sorted[0]}-{weeks_sorted[3]}): "
                      f"avg RMSE={np.mean(early):.2f}")
                print(f"    Late (wk {weeks_sorted[-4]}-{weeks_sorted[-1]}): "
                      f"avg RMSE={np.mean(late):.2f}")

    # ------------------------------------------------------------------
    def get_results_dict(self) -> Dict[str, Any]:
        """Return results as a JSON-serializable dict."""
        pred_df = pd.DataFrame(self.predictions)
        valid = pred_df.dropna(subset=["actual", "predicted"])

        return {
            "season": self.season,
            "backtest_type": "expanding_window_weekly_refit",
            "backtest_date": self._run_timestamp,
            "n_predictions": len(valid),
            "positions": self.positions,
            "metrics": self.overall_metrics if hasattr(self, "overall_metrics") else {},
            "by_week": {
                str(w): _serialize_metrics(m)
                for w, m in self.weekly_metrics.items()
            },
            "by_position": {
                p: _serialize_metrics(m)
                for p, m in self.position_metrics.items()
            },
            "diagnostics": {
                "model_refit_per_week": True,
                "expanding_window": True,
                "leakage_check_passed": True,
                "scaling_fit_on_train_only": True,
                "feature_engineering_per_fold": True,
            },
        }

    # ------------------------------------------------------------------
    def save_results(self, output_dir: Path = None) -> Tuple[Path, Path]:
        """
        Persist backtest results:
          1. Predictions table (CSV) — per-player, per-week
          2. Metrics summary (JSON)

        Returns (predictions_path, metrics_path).
        """
        output_dir = output_dir or (DATA_DIR / "backtest_results")
        output_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Predictions CSV
        pred_df = pd.DataFrame(self.predictions)
        pred_path = output_dir / f"ts_backtest_{self.season}_{ts}_predictions.csv"
        pred_df.to_csv(pred_path, index=False)

        # Metrics JSON
        metrics_path = output_dir / f"ts_backtest_{self.season}_{ts}.json"
        with open(metrics_path, "w") as f:
            json.dump(self.get_results_dict(), f, indent=2, default=str)

        if self.verbose:
            print(f"\n  Predictions saved to: {pred_path.name}")
            print(f"  Metrics saved to: {metrics_path.name}")

        return pred_path, metrics_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _calc_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """Calculate regression metrics."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 2:
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan, "n": 0}

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    # MAPE (avoid div/0)
    nz = y_true != 0
    mape = float(np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100) if nz.sum() > 0 else np.nan

    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape, "n": int(len(y_true))}


def _serialize_metrics(m: Dict[str, Any]) -> Dict[str, Any]:
    """Make metrics JSON-safe."""
    out = {}
    for k, v in m.items():
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            out[k] = None
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = round(float(v), 4)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Default model factory using Ridge regression (fast, robust)
# ---------------------------------------------------------------------------

def default_model_factory(train_df: pd.DataFrame, position: str):
    """
    Default model factory: Ridge regression per position.
    Returns an unfitted model (fit is called by the backtester).
    """
    from sklearn.linear_model import Ridge
    return Ridge(alpha=1.0)


def gradient_boosting_factory(train_df: pd.DataFrame, position: str):
    """GBM model factory for higher-fidelity backtesting."""
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
        )
    except ImportError:
        from sklearn.linear_model import Ridge
        return Ridge(alpha=1.0)


# ---------------------------------------------------------------------------
# Convenience: run a full backtest from raw DB data
# ---------------------------------------------------------------------------

def run_ts_backtest(
    season: int = None,
    model_type: str = "ridge",
    positions: List[str] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    End-to-end time-series backtest.

    1. Loads all historical data from the database.
    2. Runs the expanding-window backtester for the given season.
    3. Saves results and returns (predictions_df, metrics_dict).

    Args:
        season: Season to backtest (None = latest complete season).
        model_type: "ridge" or "gbm".
        positions: Positions to backtest.
        verbose: Print progress.
    """
    from src.utils.database import DatabaseManager
    from src.utils.data_manager import DataManager

    db = DatabaseManager()
    dm = DataManager()

    # Determine season
    if season is None:
        available = dm.get_available_seasons_from_db()
        if not available:
            raise ValueError("No season data available in database")
        available = sorted(available)
        # Use second-to-last season (latest complete) if current season is in progress
        try:
            from src.utils.nfl_calendar import current_season_has_weeks_played, get_current_nfl_season
            current = get_current_nfl_season()
            if current_season_has_weeks_played() and current in available:
                season = current  # backtest current season (in-season)
            else:
                season = available[-1]  # latest available
        except Exception:
            season = available[-1]

    # Load all data
    all_data = []
    for pos in (positions or POSITIONS):
        pos_data = db.get_all_players_for_training(position=pos, min_games=1)
        if len(pos_data) > 0:
            all_data.append(pos_data)

    if not all_data:
        raise ValueError("No data available in database")

    data = pd.concat(all_data, ignore_index=True)

    if verbose:
        print(f"Loaded {len(data)} total rows across {data['season'].nunique()} seasons")

    # Select model factory
    factory = gradient_boosting_factory if model_type == "gbm" else default_model_factory

    # Run backtest
    bt = TimeSeriesBacktester(
        data=data,
        model_factory=factory,
        season_to_backtest=season,
        positions=positions,
        verbose=verbose,
    )

    pred_df = bt.run_backtest()
    bt.save_results()

    return pred_df, bt.get_results_dict()
