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

from config.settings import DATA_DIR, DECISION_QUALITY, MODELS_DIR, POSITIONS, RIDGE_DEFAULT_ALPHA
from src.utils.leakage import filter_feature_columns

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
    Apply feature engineering to train and test separately to prevent leakage.

    Train features are computed from train data only.
    Test features are computed from the combined (train+test) block so rolling
    windows have enough history, then only test rows are extracted.

    Why separate passes? Computing features on a combined block and then
    extracting both halves lets test-period rows influence train-period
    rolling/positional statistics (train/test contamination). The production
    training code in feature_preparation._apply_with_temporal_context uses
    the same two-pass pattern implemented here.
    """
    from src.features.feature_engineering import FeatureEngineer

    engineer = FeatureEngineer()

    # Step 1: Train features from train data only (no test contamination)
    train_out = engineer.create_features(train_df.copy(), include_target=False)

    # Step 2: Test features from combined block (test rows see train history)
    if test_df.empty:
        return train_out, test_df.copy()

    combined = pd.concat([train_df, test_df], ignore_index=True)
    combined = combined.sort_values(["player_id", "season", "week"]).reset_index(
        drop=True
    )

    # Mark test rows
    test_keys = set()
    for _, row in test_df.iterrows():
        key = (row.get("player_id", ""), int(row.get("season", 0)), int(row.get("week", 0)))
        test_keys.add(key)

    combined["_is_test"] = combined.apply(
        lambda r: (r.get("player_id", ""), int(r.get("season", 0)), int(r.get("week", 0))) in test_keys,
        axis=1,
    )

    combined = engineer.create_features(combined, include_target=False)

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
        payout_multiplier: Optional[float] = None,
        report_decision_quality: bool = True,
        target_mode: str = "fp",
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
            target_mode: Target variable for training.  "fp" = fantasy points
                  (default), "util" = utilization score with in-fold conversion
                  back to FP, "component" = predict stat components separately
                  and assemble FP via PPR weights.
        """
        self.data = data.copy()
        # Ensure chronological sort
        if "game_date" in self.data.columns:
            self.data = self.data.sort_values("game_date")
        else:
            self.data = self.data.sort_values(["season", "week"])

        if target_mode not in ("fp", "util", "component"):
            raise ValueError(f"target_mode must be 'fp', 'util', or 'component', got {target_mode!r}")
        self.target_mode = target_mode
        self.model_factory = model_factory
        self.season = season_to_backtest
        self.positions = positions or POSITIONS
        self.feature_pipeline = feature_pipeline or leakage_safe_features
        self.verbose = verbose

        self.predictions: List[Dict[str, Any]] = []
        self.weekly_metrics: Dict[int, Dict[str, float]] = {}
        self.position_metrics: Dict[str, Dict[str, float]] = {}
        self._run_timestamp: Optional[str] = None
        self._position_model_factories: Dict[str, Callable] = {}

        self.payout_multiplier = (
            payout_multiplier
            if payout_multiplier is not None
            else float(DECISION_QUALITY.get("payout_multiplier", 1.8))
        )
        self.report_decision_quality = report_decision_quality
        # Emit predictions for every cumulative-active-pool player each
        # week, not just players who played.  When True, phantom test
        # rows are injected for (player, season, week) tuples where the
        # player had appeared earlier in the season but didn't play
        # this week.  Their feature vectors use their last-known
        # player-level values with this-week team/opponent/Vegas
        # context.  Actuals are NaN on phantom rows.  Enables the
        # fully-symmetric prospective replay per the 2026-04-23
        # council (`council-transcript-20260423-051434.md` step 1).
        self.emit_inactive_predictions = False

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
            print(f"  Target mode: {self.target_mode}")
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

            # Optional: inject phantom test rows for inactive cumulative-
            # active-pool players.  See docstring on
            # ``emit_inactive_predictions``.  Each phantom row carries the
            # player's last-known player-level feature values with this-
            # week team + opponent (looked up from the schedule table)
            # and ``fantasy_points=NaN`` so feature engineering and
            # downstream scoring treat them as "didn't play."
            if self.emit_inactive_predictions:
                # Explicit False on active rows so the concat doesn't
                # fill NaN (which is truthy and would flip everything
                # to phantom downstream).
                test["_phantom"] = False
                phantoms = self._build_phantom_test_rows(train, test, week)
                if phantoms is not None and not phantoms.empty:
                    test = pd.concat([test, phantoms], ignore_index=True)

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
                    # Use per-position model override if configured
                    if position in self._position_model_factories:
                        model = self._position_model_factories[position](pos_train, position)
                    else:
                        model = self.model_factory(pos_train, position)

                    # Get feature columns: causal mode uses predefined lists;
                    # full mode auto-detects numeric non-target columns.
                    # If causal mode yields zero features (e.g., synthetic data
                    # without causal columns), fall back to auto-detection.
                    from config.settings import FEATURE_MODE, CAUSAL_FEATURES
                    if FEATURE_MODE == "causal":
                        causal_cols = CAUSAL_FEATURES.get(position, [])
                        feature_cols = [c for c in causal_cols
                                        if c in pos_train.columns and c in pos_test.columns]

                    if FEATURE_MODE != "causal" or not feature_cols:
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

                        # Remove leakage-prone columns (match training pipeline)
                        feature_cols = filter_feature_columns(feature_cols)

                        # Align feature columns between train and test
                        feature_cols = [c for c in feature_cols if c in pos_test.columns]

                    X_train = pos_train[feature_cols].fillna(0)
                    X_test = pos_test[feature_cols].fillna(0)

                    # Scale: fit on train only
                    scaler = StandardScaler()
                    X_train_s = scaler.fit_transform(X_train)
                    X_test_s = scaler.transform(X_test)

                    # Fit and predict — branch on target mode
                    if self.target_mode == "util":
                        preds = self._fit_predict_util(
                            model, pos_train, X_train_s, X_test_s, position
                        )
                    elif self.target_mode == "component":
                        preds = self._fit_predict_component(
                            pos_train, X_train_s, X_test_s, position
                        )
                    else:
                        y_train = pos_train["fantasy_points"]
                        model.fit(X_train_s, y_train)
                        preds = model.predict(X_test_s)

                    # Store per-player predictions
                    for i in range(len(pos_test)):
                        row = pos_test.iloc[i]
                        actual = row.get("fantasy_points", np.nan)
                        # NaN-safe phantom check — bool(NaN) is True, so
                        # guard explicitly against NaN and non-bool values.
                        phantom_raw = row.get("_phantom", False)
                        is_phantom = (phantom_raw is True) or (
                            isinstance(phantom_raw, (bool, np.bool_)) and bool(phantom_raw)
                        )
                        self.predictions.append({
                            "season": int(self.season),
                            "week": int(week),
                            "player_id": row.get("player_id", ""),
                            "name": row.get("name", ""),
                            "position": position,
                            "team": row.get("team", ""),
                            "predicted": float(preds[i]),
                            "actual": float(actual) if pd.notna(actual) else np.nan,
                            "is_active": 0 if is_phantom else 1,
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
    def _build_phantom_test_rows(
        self, train: pd.DataFrame, test: pd.DataFrame, week: int
    ) -> Optional[pd.DataFrame]:
        """Build phantom rows for players who were in the season's
        cumulative-active pool through week N-1 but didn't play week N.

        Each phantom carries the player's most recent pre-week-N row
        from ``train`` (all player-level fields frozen at last-known)
        with ``week`` updated to the target week, ``fantasy_points`` set
        to NaN (sentinel for "didn't play"), and ``opponent`` filled in
        from the ``schedule`` table for (team, season, week).  The rest
        of the feature engineering cascades naturally — team-level
        Vegas / opp-defense / injury fields get this-week values from
        their respective merges; rolling player stats remain as of the
        player's last appearance (which is the closest approximation to
        what a production pipeline would see if the player sits
        week N).
        """
        if train.empty:
            return None
        season_train = train[train["season"] == self.season]
        if season_train.empty:
            return None

        cumulative_ids = set(season_train["player_id"].dropna().unique())
        active_ids = set(test["player_id"].dropna().unique()) if not test.empty else set()
        inactive_ids = cumulative_ids - active_ids
        if not inactive_ids:
            return None

        season_train_sorted = season_train.sort_values(
            ["player_id", "week"], kind="stable"
        )
        # Most recent pre-week-N row per inactive player.  Using
        # groupby(...).tail(1) preserves dtype better than iloc[-1] in a
        # loop.
        last_rows = (
            season_train_sorted[season_train_sorted["player_id"].isin(inactive_ids)]
            .groupby("player_id", sort=False, as_index=False)
            .tail(1)
            .copy()
        )
        if last_rows.empty:
            return None

        # Look up (team, opponent) for the target week from the
        # schedule table.  Fall back to last-known opponent if the
        # schedule row isn't available.
        schedule_map: Dict[Tuple[str, int, int], str] = {}
        try:
            from src.utils.database import DatabaseManager
            db = DatabaseManager()
            with db._get_connection() as conn:
                for row in conn.execute(
                    "SELECT home_team, away_team FROM schedule "
                    "WHERE season = ? AND week = ?",
                    (int(self.season), int(week)),
                ):
                    home, away = row[0], row[1]
                    if home:
                        schedule_map[(home, int(self.season), int(week))] = away or ""
                    if away:
                        schedule_map[(away, int(self.season), int(week))] = home or ""
        except Exception:
            pass

        last_rows["week"] = int(week)
        last_rows["fantasy_points"] = np.nan
        if "opponent" in last_rows.columns and "team" in last_rows.columns:
            last_rows["opponent"] = [
                schedule_map.get((t, int(self.season), int(week)), opp if opp else "")
                for t, opp in zip(last_rows["team"].astype(str), last_rows["opponent"].astype(str))
            ]
        # Stamp so downstream prediction storage can flag these.
        last_rows["_phantom"] = True

        return last_rows

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

        # Drawdown analysis: longest streak of poor rank accuracy (Directive V7 §10)
        self.drawdown_metrics = self._compute_drawdown(valid)

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
            if self.drawdown_metrics.get("max_drawdown_weeks", 0) > 0:
                print(f"\n  Drawdown: max {self.drawdown_metrics['max_drawdown_weeks']} "
                      f"consecutive weeks with rank accuracy < 50%")

    # ------------------------------------------------------------------
    @staticmethod
    def _compute_drawdown(valid_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute drawdown metrics: longest streak of poor rank accuracy per week.

        Rank accuracy per week = fraction of top-10 actual players that appear
        in the top-10 predicted players.  A "bad" week has rank accuracy < 50%.
        """
        if valid_df.empty or "week" not in valid_df.columns:
            return {"max_drawdown_weeks": 0, "weekly_rank_accuracy": []}

        weekly_rank_acc = []
        for week, grp in sorted(valid_df.groupby("week")):
            if len(grp) < 10:
                continue
            top_n = min(10, len(grp) // 2)
            actual_top = set(grp.nlargest(top_n, "actual").index)
            pred_top = set(grp.nlargest(top_n, "predicted").index)
            overlap = len(actual_top & pred_top) / top_n
            weekly_rank_acc.append({"week": int(week), "rank_accuracy": round(overlap, 3)})

        # Longest streak below 50%
        max_streak = 0
        current_streak = 0
        for entry in weekly_rank_acc:
            if entry["rank_accuracy"] < 0.5:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        return {
            "max_drawdown_weeks": max_streak,
            "weekly_rank_accuracy": weekly_rank_acc,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def scenario_sensitivity(
        pred_df: pd.DataFrame,
        noise_levels: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Compute scenario sensitivity analysis.

        Per Directive V7 Section 10: measure scenario sensitivity under
        optimistic, base, and pessimistic assumptions.

        Simulates varying levels of prediction noise and reports how
        metrics degrade under each scenario.

        Args:
            pred_df: Predictions DataFrame with 'actual' and 'predicted'.
            noise_levels: Dict of scenario_name -> noise_std_factor.

        Returns:
            Dict of scenario_name -> metrics.
        """
        if noise_levels is None:
            noise_levels = {
                "optimistic": 0.0,   # No additional noise
                "base": 0.5,         # Moderate noise
                "pessimistic": 1.0,  # High noise
            }

        valid = pred_df.dropna(subset=["actual", "predicted"])
        if valid.empty:
            return {}

        actuals = valid["actual"].values
        preds = valid["predicted"].values
        base_std = float(np.std(actuals - preds))

        results = {}
        rng = np.random.RandomState(42)
        for scenario, factor in noise_levels.items():
            if factor == 0:
                noisy_preds = preds
            else:
                noise = rng.normal(0, base_std * factor, len(preds))
                noisy_preds = preds + noise

            results[scenario] = _calc_metrics(
                pd.Series(actuals), pd.Series(noisy_preds)
            )
        return results

    @staticmethod
    def friction_analysis(
        pred_df: pd.DataFrame,
        data_delay_weeks: int = 0,
        roster_lock_penalty_pct: float = 0.0,
    ) -> Dict[str, Any]:
        """Model friction terms relevant to fantasy football.

        Per Directive V7 Section 10: include friction terms relevant
        to the domain (roster lock timing, data availability delays).

        Args:
            pred_df: Predictions DataFrame.
            data_delay_weeks: Simulated weeks of data delay.
            roster_lock_penalty_pct: Percentage penalty for late roster decisions.

        Returns:
            Dict with friction-adjusted metrics.
        """
        valid = pred_df.dropna(subset=["actual", "predicted"])
        if valid.empty:
            return {}

        base_metrics = _calc_metrics(valid["actual"], valid["predicted"])

        # Simulate data delay: predictions become less accurate
        # as they rely on older information
        delay_degradation = 1.0 + (data_delay_weeks * 0.05)
        adjusted_rmse = base_metrics.get("rmse", 0) * delay_degradation

        # Roster lock penalty: late decisions lead to suboptimal lineups
        lock_penalty = base_metrics.get("rmse", 0) * (roster_lock_penalty_pct / 100)

        return {
            "base_rmse": base_metrics.get("rmse"),
            "data_delay_weeks": data_delay_weeks,
            "delay_adjusted_rmse": round(adjusted_rmse, 3),
            "roster_lock_penalty_pct": roster_lock_penalty_pct,
            "friction_adjusted_rmse": round(adjusted_rmse + lock_penalty, 3),
            "total_friction_impact_pct": round(
                ((adjusted_rmse + lock_penalty) / max(base_metrics.get("rmse", 1), 0.01) - 1) * 100, 1
            ),
        }

    # ------------------------------------------------------------------
    def compute_baseline_comparison(self) -> Dict[str, Any]:
        """Compare model predictions against naive baselines.

        Runs trailing-average, season-average, and expert-consensus
        baselines on the same test data used by the backtest. Returns
        per-baseline metrics so every report contextualizes R² against
        what simple methods would achieve.
        """
        from src.evaluation.baselines import (
            trailing_average_baseline,
            season_average_baseline,
            expert_consensus_baseline,
        )

        pred_df = pd.DataFrame(self.predictions)
        if pred_df.empty or "actual" not in pred_df.columns:
            return {}

        valid = pred_df.dropna(subset=["actual", "predicted"])
        if valid.empty:
            return {}

        # Build a player-week lookup from backtest data for baselines
        season_data = self.data[self.data["season"] <= self.season].copy()
        if "fantasy_points" not in season_data.columns:
            return {}

        baselines = {}

        # Map predictions back to the full dataset for baseline computation
        baseline_runners = {
            "trailing_avg_3w": lambda df: trailing_average_baseline(df, n_weeks=3),
            "season_avg": lambda df: season_average_baseline(df),
            "blended_heuristic": lambda df: expert_consensus_baseline(df),
        }

        for name, runner in baseline_runners.items():
            try:
                season_data_sorted = season_data.sort_values(
                    ["player_id", "season", "week"]
                ).reset_index(drop=True)
                baseline_preds = runner(season_data_sorted)

                # Extract baseline predictions for the test season only
                test_mask = season_data_sorted["season"] == self.season
                test_baseline = baseline_preds[test_mask]
                test_actual = season_data_sorted.loc[test_mask, "fantasy_points"]

                # Align and drop NaN
                combined = pd.DataFrame({
                    "actual": test_actual.values,
                    "baseline_pred": test_baseline.values,
                })
                combined = combined.dropna()

                if len(combined) < 10:
                    continue

                bl_metrics = _calc_metrics(combined["actual"], combined["baseline_pred"])
                baselines[name] = _serialize_metrics(bl_metrics)
            except Exception as e:
                baselines[name] = {"error": str(e)}

        # Add model metrics for direct comparison
        model_metrics = _calc_metrics(valid["actual"], valid["predicted"])
        baselines["model"] = _serialize_metrics(model_metrics)

        # Flag whether model beats each baseline
        model_rmse = model_metrics.get("rmse", float("inf"))
        for name in list(baselines.keys()):
            if name == "model":
                continue
            bl = baselines[name]
            if isinstance(bl, dict) and "rmse" in bl and bl["rmse"] is not None:
                baselines[name]["model_beats_baseline"] = model_rmse < bl["rmse"]

        return baselines

    # ------------------------------------------------------------------
    def _compute_decision_quality(self, pred_df: pd.DataFrame) -> Dict[str, Any]:
        """Run cash-H2H lineup decision quality on walk-forward predictions.

        Reuses ``ModelBacktester.backtest_lineup_decisions`` so the three
        opponent tiers, binomial test, and ROI formula stay in one place.
        Returns an empty dict when disabled or when the required columns
        aren't present, so callers can splice the result in unconditionally.
        """
        if not self.report_decision_quality or pred_df is None or pred_df.empty:
            return {}
        required = {"predicted", "actual", "position", "week"}
        if not required.issubset(pred_df.columns):
            return {}
        try:
            from src.evaluation.backtester import ModelBacktester
        except Exception:
            return {}

        df = pred_df.rename(
            columns={"predicted": "predicted_points", "actual": "fantasy_points"}
        )
        return ModelBacktester().backtest_lineup_decisions(
            df,
            pred_col="predicted_points",
            actual_col="fantasy_points",
            position_col="position",
            payout_multiplier=self.payout_multiplier,
        )

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Target-mode helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_proxy_utilization(df: pd.DataFrame, position: str) -> pd.Series:
        """Compute a lightweight utilization proxy from raw stats.

        Uses the same share-based logic as the full UtilizationScoreCalculator
        but without requiring team-level aggregates or percentile bounds.
        Returns a 0-100 score per row.
        """
        util = pd.Series(0.0, index=df.index)
        if position == "QB":
            for col, w in [("passing_attempts", 0.4), ("rushing_attempts", 0.3),
                           ("passing_tds", 0.15), ("rushing_tds", 0.15)]:
                if col in df.columns:
                    s = df[col].fillna(0)
                    if s.max() > 0:
                        util += w * 100 * s / max(s.quantile(0.99), 1)
        elif position == "RB":
            for col, w in [("rushing_attempts", 0.35), ("targets", 0.25),
                           ("rushing_yards", 0.2), ("receptions", 0.2)]:
                if col in df.columns:
                    s = df[col].fillna(0)
                    if s.max() > 0:
                        util += w * 100 * s / max(s.quantile(0.99), 1)
        elif position in ("WR", "TE"):
            for col, w in [("targets", 0.35), ("receptions", 0.25),
                           ("receiving_yards", 0.25), ("receiving_tds", 0.15)]:
                if col in df.columns:
                    s = df[col].fillna(0)
                    if s.max() > 0:
                        util += w * 100 * s / max(s.quantile(0.99), 1)
        return util.clip(0, 100)

    def _fit_predict_util(self, model, pos_train, X_train_s, X_test_s, position):
        """Train on utilization proxy, convert predictions back to FP."""
        from sklearn.linear_model import Ridge as _Ridge

        util_col = "utilization_score"
        if util_col not in pos_train.columns or pos_train[util_col].notna().sum() < 30:
            # Compute proxy utilization from raw stats
            pos_train = pos_train.copy()
            pos_train[util_col] = self._compute_proxy_utilization(pos_train, position)

        if pos_train[util_col].std() < 1e-6:
            # Fall back to FP if utilization is constant
            y_train = pos_train["fantasy_points"]
            model.fit(X_train_s, y_train)
            return model.predict(X_test_s)

        y_train = pos_train[util_col]
        model.fit(X_train_s, y_train)
        util_preds = model.predict(X_test_s)

        # In-fold converter: simple Ridge from util → FP on training data
        train_util = pos_train[[util_col]].fillna(0).values
        train_fp = pos_train["fantasy_points"].fillna(0).values
        converter = _Ridge(alpha=1000)
        converter.fit(train_util, train_fp)
        return converter.predict(util_preds.reshape(-1, 1))

    def _fit_predict_component(self, pos_train, X_train_s, X_test_s, position):
        """Train separate Ridge per stat component, assemble FP via PPR weights."""
        from sklearn.linear_model import Ridge as _Ridge
        from config.settings import COMPONENT_TARGETS, PPR_SCORING_WEIGHTS

        components = COMPONENT_TARGETS.get(position, [])
        preds = np.zeros(X_test_s.shape[0])

        for comp in components:
            if comp not in pos_train.columns:
                continue
            y_comp = pos_train[comp].fillna(0)
            if y_comp.std() < 1e-6:
                continue
            comp_model = _Ridge(alpha=RIDGE_DEFAULT_ALPHA)
            comp_model.fit(X_train_s, y_comp)
            comp_preds = comp_model.predict(X_test_s)
            # Stats can't be negative (except interceptions)
            if comp != "interceptions":
                comp_preds = np.maximum(comp_preds, 0)
            weight = PPR_SCORING_WEIGHTS.get(comp, 0)
            preds += comp_preds * weight

        return preds

    # ------------------------------------------------------------------
    def get_results_dict(self) -> Dict[str, Any]:
        """Return results as a JSON-serializable dict."""
        pred_df = pd.DataFrame(self.predictions)
        valid = pred_df.dropna(subset=["actual", "predicted"])

        # Compute scenario sensitivity (Directive V7 §10)
        scenarios = self.scenario_sensitivity(pred_df)
        friction = self.friction_analysis(pred_df)

        # Week-by-week performance path (Directive V7 §10)
        weekly_path = []
        for w in sorted(self.weekly_metrics.keys()):
            m = self.weekly_metrics[w]
            weekly_path.append({
                "week": w,
                "rmse": round(m.get("rmse", 0), 3),
                "mae": round(m.get("mae", 0), 3),
                "r2": round(m.get("r2", 0), 4),
            })

        return {
            "season": self.season,
            "target_mode": self.target_mode,
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
            "drawdown": self.drawdown_metrics if hasattr(self, "drawdown_metrics") else {},
            "weekly_performance_path": weekly_path,
            "scenario_sensitivity": {
                k: _serialize_metrics(v) for k, v in scenarios.items()
            } if scenarios else {},
            "friction_analysis": friction,
            "baselines": self.compute_baseline_comparison(),
            "decision_quality": self._compute_decision_quality(valid),
            "diagnostics": {
                "model_refit_per_week": True,
                "expanding_window": True,
                "leakage_check_passed": True,
                "leakage_filter_applied": True,
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
        results = self.get_results_dict()
        with open(metrics_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Per-week lineup decision-quality CSV — the council's stated
        # success signal for this workstream.  Skipped when decision quality
        # is disabled or no complete weeks were produced.
        lineup_weekly_path: Optional[Path] = None
        weekly = (results.get("decision_quality") or {}).get("weekly_results")
        if weekly:
            lineup_weekly_path = (
                output_dir / f"ts_backtest_{self.season}_{ts}_lineup_weekly.csv"
            )
            pd.DataFrame(weekly).to_csv(lineup_weekly_path, index=False)

        if self.verbose:
            print(f"\n  Predictions saved to: {pred_path.name}")
            print(f"  Metrics saved to: {metrics_path.name}")
            if lineup_weekly_path is not None:
                print(f"  Lineup weekly saved to: {lineup_weekly_path.name}")

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

def default_model_factory(train_df: pd.DataFrame, position: str, alpha=1.0):
    """
    Default model factory: Ridge regression per position.
    Returns an unfitted model (fit is called by the backtester).

    ``alpha`` is the Ridge regularization strength.  Bind it via
    ``functools.partial`` when constructing a factory for the backtester,
    which calls factories with only ``(train_df, position)``.

    ``alpha`` accepts either a scalar float (uniform regularization) or a
    ``dict[str, float]`` keyed by position code for per-position tuning.
    For the dict form, positions missing from the dict fall back to 1.0
    (the scikit-learn default) — this keeps the dict form opt-in per
    position and matches the long-standing default behavior.
    """
    from sklearn.linear_model import Ridge
    if isinstance(alpha, dict):
        resolved = float(alpha.get(position, 1.0))
    else:
        resolved = float(alpha)
    return Ridge(alpha=resolved)


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
# Production ensemble model factory
# ---------------------------------------------------------------------------

class _EnsembleModelWrapper:
    """Adapts PositionModel to the sklearn fit/predict interface the backtester expects.

    The backtester passes pre-scaled numpy arrays.  PositionModel does its own
    internal scaling, so the double-scaling is harmless (StandardScaler on
    already-standardised data is ~identity).  Hyperparameter tuning is skipped
    for speed — defaults are used, which still exercise the full XGBoost +
    LightGBM + RF + Ridge stacked ensemble with Huber loss.
    """

    def __init__(self, position: str, feature_names: list):
        from src.models.position_models import PositionModel
        self._position = position
        self._feature_names = feature_names
        self._model: "PositionModel" = None

    def fit(self, X: np.ndarray, y) -> "_EnsembleModelWrapper":
        from src.models.position_models import PositionModel
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_df = pd.DataFrame(X, columns=self._feature_names)
            y_s = pd.Series(np.asarray(y, dtype=np.float64), name="fantasy_points")
            self._model = PositionModel(self._position, n_weeks=1)
            self._model.fit(X_df, y_s, tune_hyperparameters=False)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_df = pd.DataFrame(X, columns=self._feature_names)
        return self._model.predict(X_df)


def ensemble_model_factory(train_df: pd.DataFrame, position: str):
    """Production ensemble factory: XGBoost + LightGBM + RF + Ridge stacked.

    Returns an unfitted wrapper whose fit/predict calls go through the full
    PositionModel pipeline (Huber loss, OOF stacking, isotonic calibration).
    Hyperparameter tuning is disabled for backtest speed.
    """
    from config.settings import FEATURE_MODE, CAUSAL_FEATURES
    if FEATURE_MODE == "causal":
        feature_names = [c for c in CAUSAL_FEATURES.get(position, [])
                         if c in train_df.columns]
    else:
        exclude = {
            "player_id", "name", "position", "team", "opponent",
            "season", "week", "home_away", "created_at", "id",
            "game_date", "fantasy_points",
        }
        target_cols = {c for c in train_df.columns if c.startswith("target_")}
        exclude |= target_cols
        feature_names = [
            c for c in train_df.columns
            if c not in exclude
            and train_df[c].dtype in ("int64", "float64", "int32", "float32")
        ]
        feature_names = filter_feature_columns(feature_names)

    return _EnsembleModelWrapper(position, feature_names)


# ---------------------------------------------------------------------------
# Convenience: run a full backtest from raw DB data
# ---------------------------------------------------------------------------

def run_ts_backtest(
    season: int = None,
    model_type: str = "ridge",
    positions: List[str] = None,
    verbose: bool = True,
    ridge_alpha=None,
    payout_multiplier: Optional[float] = None,
    report_decision_quality: bool = True,
    emit_inactive_predictions: bool = False,
    target_mode: str = "fp",
    qb_model: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    End-to-end time-series backtest.

    1. Loads all historical data from the database.
    2. Runs the expanding-window backtester for the given season.
    3. Saves results and returns (predictions_df, metrics_dict).

    Args:
        season: Season to backtest (None = latest complete season).
        model_type: "ridge", "gbm", or "ensemble".
        positions: Positions to backtest.
        verbose: Print progress.
        ridge_alpha: Ridge regularization strength (only used when
            model_type="ridge").  Accepts either a scalar float for
            uniform regularization, or a ``dict[str, float]`` keyed by
            position for per-position tuning.  Positions missing from
            the dict fall back to 1.0.  Default: ``RIDGE_DEFAULT_ALPHA``
            from ``config.settings`` (10_000 as of 2026-04-20).  Other
            model types ignore this parameter.
    """
    if ridge_alpha is None:
        ridge_alpha = RIDGE_DEFAULT_ALPHA
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
    if model_type == "ensemble":
        factory = ensemble_model_factory
    elif model_type == "gbm":
        factory = gradient_boosting_factory
    else:
        from functools import partial
        factory = partial(default_model_factory, alpha=ridge_alpha)

    # Run backtest
    bt = TimeSeriesBacktester(
        data=data,
        model_factory=factory,
        season_to_backtest=season,
        positions=positions,
        verbose=verbose,
        payout_multiplier=payout_multiplier,
        report_decision_quality=report_decision_quality,
        target_mode=target_mode,
    )
    bt.emit_inactive_predictions = bool(emit_inactive_predictions)

    # Per-position model overrides
    if qb_model:
        if qb_model == "gbm":
            bt._position_model_factories["QB"] = gradient_boosting_factory
        elif qb_model == "ridge":
            from functools import partial as _partial
            bt._position_model_factories["QB"] = _partial(default_model_factory, alpha=ridge_alpha)

    pred_df = bt.run_backtest()
    bt.save_results()

    return pred_df, bt.get_results_dict()
