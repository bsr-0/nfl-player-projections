"""Phase 9 (next_focus.md): Monte Carlo season-total PPR quantiles
(P25/P50/P75/P90).

Explicitly not "sum independent weekly quantiles" (next_focus.md's own
stated concern) -- simulates whole-season paths and takes quantiles of the
simulated season-total distribution, which naturally differs from summing
per-week quantiles even under independence, and additionally models real
week-to-week performance correlation via block-bootstrap (see below).

Two uncertainty sources, kept deliberately separate and independently
testable:
  - Games-played: independent per-week Bernoulli(availability_rate) draw
    for every synthetic (no-data) week, reusing
    `season_projection.estimate_availability_rate` as-is.
  - Weekly performance, conditional on playing: block-bootstrap --  within
    each maximal contiguous run of "played" synthetic weeks in a simulated
    path, sample ONE random donor (player_id, season)'s real observed
    residual sequence (actual - predicted, from strictly-prior-season
    Phase 4 OOS folds) and add a matching-length contiguous sub-block to
    that run's point predictions, preserving real hot/cold-streak
    correlation instead of drawing each week's noise independently.

"Player role" uncertainty is not a separate mechanism: real observed
residuals already embed role-driven variance (a committee back's
touches-share swings show up as large residuals in the historical record
the donor pool is built from).

No look-ahead in the residual pool: simulating season S only ever draws
from OOS folds with `season < S` (expanding window, matching this
session's walk-forward discipline everywhere else) -- NOT pooling
2023-2025 indiscriminately, per explicit user direction. Real/inferred
weeks (nflverse_stats / inferred_snap_verified_zero /
inferred_pbp_confirmed_zero) contribute their known actual value
identically to every simulated path -- P(plays)=1 is already known, same
reasoning as season_projection.py's point-estimate redesign.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_N_SIMULATIONS = 2000
DEFAULT_SEED = 42


def build_residual_donor_pools(
    csv_paths: Sequence[Path],
    position: str,
    architecture: str,
    before_season: int,
) -> Dict[Tuple[str, int], List[Tuple[int, float]]]:
    """Loads/concatenates the given Phase-4-style row-level CSVs, filters to
    `position`, `model == architecture` (the FINAL_CONFIG choice for that
    position), and `season < before_season` (strict -- no look-ahead), then
    groups into week-ordered per-(player, season) residual sequences.

    Returns {(player, season): [(week, actual_ppr - prediction), ...]}
    sorted by week -- these are the block-bootstrap donors.
    """
    frames = []
    for path in csv_paths:
        if not Path(path).exists():
            continue
        df = pd.read_csv(path)
        frames.append(df)
    if not frames:
        return {}

    all_df = pd.concat(frames, ignore_index=True)
    sub = all_df[
        (all_df["position"] == position)
        & (all_df["model"] == architecture)
        & (all_df["season"] < before_season)
    ].copy()
    if sub.empty:
        return {}

    sub["residual"] = sub["actual_ppr"] - sub["prediction"]
    pools: Dict[Tuple[str, int], List[Tuple[int, float]]] = {}
    for (player, season), g in sub.groupby(["player", "season"]):
        g_sorted = g.sort_values("week")
        pools[(player, season)] = list(zip(g_sorted["week"].astype(int), g_sorted["residual"].astype(float)))
    return pools


def _sample_block(
    donor_pool: Dict[Tuple[str, int], List[Tuple[int, float]]],
    length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Samples a random contiguous residual sub-block of exactly `length`
    from a random donor with a long-enough sequence. Falls back to
    with-replacement single-residual draws (still from the real pool, just
    losing the correlation-preservation benefit) if no donor is long
    enough -- keeps simulation robust to a thin pool rather than raising.
    """
    candidates = [seq for seq in donor_pool.values() if len(seq) >= length]
    if not candidates:
        all_residuals = [r for seq in donor_pool.values() for _, r in seq]
        if not all_residuals:
            return np.zeros(length)
        return rng.choice(all_residuals, size=length, replace=True)

    donor = candidates[rng.integers(0, len(candidates))]
    max_start = len(donor) - length
    start = rng.integers(0, max_start + 1)
    return np.array([r for _, r in donor[start:start + length]], dtype=float)


def simulate_player_season(
    week_predictions: List[dict],
    availability_rate: float,
    donor_pool: Dict[Tuple[str, int], List[Tuple[int, float]]],
    n_sims: int = DEFAULT_N_SIMULATIONS,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Returns an array of `n_sims` simulated season-total draws for one
    player, given that player's per-week predictions (from
    `season_projection.compute_player_week_predictions`).

    Real/inferred weeks contribute their known actual value to every path
    unconditionally. Synthetic weeks are simulated per-path: independent
    Bernoulli(availability_rate) play/no-play, then block-bootstrap
    residuals added within each maximal contiguous run of "played"
    synthetic weeks in that path.
    """
    rng = rng or np.random.default_rng(DEFAULT_SEED)

    known_total = sum(wp["actual_value"] for wp in week_predictions if wp["is_real"])
    synthetic = [wp for wp in week_predictions if not wp["is_real"]]

    if not synthetic:
        return np.full(n_sims, known_total, dtype=float)

    synth_preds = np.array([wp["point_prediction"] for wp in synthetic], dtype=float)
    n_synth = len(synth_preds)

    plays = rng.random((n_sims, n_synth)) < availability_rate

    totals = np.full(n_sims, known_total, dtype=float)
    for sim_idx in range(n_sims):
        play_mask = plays[sim_idx]
        # Walk the synthetic weeks, finding maximal contiguous "played" runs.
        i = 0
        sim_total = 0.0
        while i < n_synth:
            if not play_mask[i]:
                i += 1
                continue
            j = i
            while j < n_synth and play_mask[j]:
                j += 1
            run_len = j - i
            residual_block = _sample_block(donor_pool, run_len, rng)
            sim_total += float(np.sum(synth_preds[i:j] + residual_block))
            i = j
        totals[sim_idx] += sim_total

    return totals


def run_season_simulation(
    positions: Optional[Sequence[str]] = None,
    seasons: Optional[Sequence[int]] = None,
    n_sims: int = DEFAULT_N_SIMULATIONS,
    residual_csv_paths: Optional[Sequence[Path]] = None,
    output_path=None,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """For each position's FINAL_CONFIG fold: fit once (identical to
    season_projection.run_season_projection), then Monte Carlo simulate
    each player's season total instead of computing a single point
    estimate, recording P25/P50/P75/P90 of the simulated distribution.
    """
    from config.settings import CAUSAL_FEATURES, POSITIONS
    from src.features.feature_engineering import PositionFeatureEngineer
    from src.models.single_week_ppr.evaluate import (
        DEFAULT_VALIDATION_SEASONS, run_fold, _architectures_for_fold, _append_df_to_csv,
        FoldFailureTracker,
    )
    from src.models.single_week_ppr.final_config import FINAL_CONFIG
    from src.models.single_week_ppr.season_projection import (
        possible_weeks_for_player, estimate_availability_rate, compute_player_week_predictions,
        REGULAR_SEASON_MAX_WEEK, WeekSkipTracker,
    )
    from src.models.single_week_ppr.windows import window_to_season_list
    from src.utils.database import DatabaseManager
    from src.utils.leakage import filter_feature_columns

    seasons = seasons or DEFAULT_VALIDATION_SEASONS
    output_path = output_path or Path("data/experiments/phase9_season_simulation.csv")
    positions = list(positions) if positions else POSITIONS
    residual_csv_paths = residual_csv_paths or [
        Path("data/experiments/phase4_row_level_predictions_v2_corrected.csv"),
        Path("data/experiments/phase4_row_level_predictions_v2_corrected_2020_2022.csv"),
    ]
    all_rows: List[dict] = []
    db = DatabaseManager()
    rng = np.random.default_rng(seed)

    tracker = FoldFailureTracker("Phase 9 (season simulation)")
    week_skips = WeekSkipTracker("Phase 9 (season simulation)")
    for position in positions:
        cfg = FINAL_CONFIG[position]
        feature_engineer = PositionFeatureEngineer(position)
        available_seasons = sorted(
            db.get_all_players_for_training(position=position)["season"].dropna().unique().tolist()
        )

        for season in seasons:
            print(f"\n=== SEASON SIMULATION {position} / season={season} / {cfg} / n_sims={n_sims} ===")
            train_seasons = window_to_season_list(cfg["window"], season, available_seasons)
            if not train_seasons:
                logger.warning("Skipping %s/%s: no training seasons available", position, season)
                continue
            try:
                train_df, test_df, _, _ = run_fold(
                    position, season, False, train_seasons_override=train_seasons,
                )
            except Exception as e:
                tracker.record(position, season, e)
                continue

            pos_train = train_df[train_df["position"] == position].reset_index(drop=True)
            pos_test = test_df[test_df["position"] == position].copy()
            if len(pos_test) < 20:
                logger.warning("Skipping %s/%s: only %d test rows", position, season, len(pos_test))
                continue

            feature_cols = filter_feature_columns(CAUSAL_FEATURES.get(position, []))
            feature_cols = [c for c in feature_cols if c in pos_train.columns and c in pos_test.columns]
            if not feature_cols:
                logger.warning("Skipping %s/%s: no CAUSAL_FEATURES columns present", position, season)
                continue

            model = _architectures_for_fold()[cfg["architecture"]]
            X_train = pos_train[feature_cols]
            y_train = pos_train["fantasy_points"]
            model.fit(X_train, y_train)

            full_history = pd.concat([pos_train, pos_test], ignore_index=True)

            donor_pool = build_residual_donor_pools(
                residual_csv_paths, position, cfg["architecture"], before_season=season,
            )
            print(f"  Residual donor pool: {len(donor_pool)} player-seasons "
                  f"(strictly before {season})")

            for player_id, g_all in pos_test.groupby("player_id"):
                # Regular season only -- see season_projection.py's matching
                # comment for why playoff-week rows (week > 18) must be
                # excluded before computing a "season" total.
                g = g_all[g_all["week"] <= REGULAR_SEASON_MAX_WEEK]
                if g.empty:
                    continue
                g_by_week = {int(w): sub for w, sub in g.groupby(g["week"].astype(int))}
                real_weeks = set(g_by_week.keys())
                real_team_by_week = {int(w): sub["team"].iloc[0] for w, sub in g_by_week.items()}
                possible_weeks, team_by_week = possible_weeks_for_player(
                    db, real_team_by_week, season, player_id=player_id,
                    skip_tracker=week_skips)
                if not possible_weeks:
                    continue

                availability_rate = estimate_availability_rate(player_id, position, season, db)

                week_predictions = compute_player_week_predictions(
                    player_id, g_by_week, real_weeks, team_by_week, possible_weeks, model,
                    feature_cols, full_history, db, feature_engineer, season,
                    skip_tracker=week_skips,
                )
                if not week_predictions:
                    continue

                sim_totals = simulate_player_season(
                    week_predictions, availability_rate, donor_pool, n_sims=n_sims, rng=rng,
                )

                actual_total = float(g["fantasy_points"].sum())
                result_row = {
                    "player": player_id, "position": position, "season": season, "team": team,
                    "possible_weeks": len(possible_weeks),
                    "weeks_predicted": len(week_predictions),
                    "games_actually_played": len(real_weeks),
                    "availability_rate": availability_rate,
                    "donor_pool_size": len(donor_pool),
                    "p25": float(np.percentile(sim_totals, 25)),
                    "p50": float(np.percentile(sim_totals, 50)),
                    "p75": float(np.percentile(sim_totals, 75)),
                    "p90": float(np.percentile(sim_totals, 90)),
                    "sim_mean": float(np.mean(sim_totals)),
                    "sim_std": float(np.std(sim_totals)),
                    "actual_season_total": actual_total,
                }
                all_rows.append(result_row)
                _append_df_to_csv(pd.DataFrame([result_row]), output_path)

    result = pd.DataFrame(all_rows)
    print(f"\n{len(result)} rows appended to {output_path}")
    tracker.report(output_path)
    week_skips.report(output_path)
    return result
