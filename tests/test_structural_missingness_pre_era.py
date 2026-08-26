"""Structural missingness -- a source that does not exist yet -- must reach
the model as NaN, not as a fabricated measurement.

Two eras are covered, both found by auditing the Phase 7 cold-start arm:

    snap_counts starts in 2013; player_weekly_stats carried a literal 0.0
    before that, so every player in 2006-2012 looked like he took 0% of his
    team's snaps.

    NGS starts in 2016; ngs_* columns were fillna(0.0)'d, so every receiver
    before 2016 had 0.0 yards of separation.

Both passed an `IS NOT NULL` coverage audit at 100%. Neither failed anything.

The policy-registry case has its own test below because the mechanism was a
substring collision, not a missing exemption: 'epa' is inside
'ngs_avg_sEPAration', so that column resolved to the pbp_advanced group and
was median-filled while its sibling ngs_avg_cushion was preserved.
"""
import numpy as np
import pandas as pd

from src.features.feature_policy_registry import FeaturePolicyRegistry
from src.features.utilization_score import (
    SNAP_DATA_START_SEASON,
    SNAP_ROLL3_COL,
    apply_snap_imputation,
)


class TestPolicyGroupResolution:
    def test_ngs_separation_resolves_to_ngs_not_pbp(self):
        """'epa' is a substring of 'separation'. Declaration order must not
        decide which policy owns the column."""
        r = FeaturePolicyRegistry.from_config()
        assert r.resolve_group_for_feature("ngs_avg_separation") == "ngs"
        assert r.resolve_group_for_feature("ngs_avg_separation_roll3_mean") == "ngs"

    def test_ngs_siblings_agree(self):
        """Cushion and separation are the same family from the same table;
        they must not land in different policy groups."""
        r = FeaturePolicyRegistry.from_config()
        groups = {
            r.resolve_group_for_feature(f"ngs_{m}_roll3_mean")
            for m in ("avg_separation", "avg_cushion", "avg_intended_air_yards")
        }
        assert groups == {"ngs"}

    def test_genuine_pbp_columns_unaffected(self):
        r = FeaturePolicyRegistry.from_config()
        for col in ("pass_epa", "rush_epa_per_play", "recv_epa_per_target"):
            assert r.resolve_group_for_feature(col) == "pbp_advanced"


class TestPreserveStrategy:
    def test_preserve_leaves_nan_and_adds_no_imputed_indicator(self):
        r = FeaturePolicyRegistry.from_config()
        assert r.policies["ngs"].numeric_strategy == "preserve", (
            "NGS is structurally absent before 2016; filling it invents a measurement"
        )
        df = pd.DataFrame({
            "ngs_avg_separation_roll3_mean": [np.nan, np.nan, 2.5, 3.0],
            "ngs_avg_cushion_roll3_mean": [np.nan, 1.0, 2.0, 3.0],
        })
        r.apply(df, context="test")
        assert df["ngs_avg_separation_roll3_mean"].isna().sum() == 2
        assert df["ngs_avg_cushion_roll3_mean"].isna().sum() == 1
        assert "ngs_avg_separation_roll3_mean_imputed" not in df.columns

    def test_preserve_still_reports_rates(self):
        """Preserving the value must not blind the monitoring that reports it."""
        r = FeaturePolicyRegistry.from_config()
        df = pd.DataFrame({"ngs_avg_separation_roll3_mean": [np.nan, np.nan, 2.5, 3.0]})
        res = r.apply(df, context="test")
        assert res.rates["ngs_avg_separation_roll3_mean"] == 0.5


class TestSnapImputationEraExemption:
    def _frame(self, seasons):
        return pd.DataFrame({
            "season": seasons,
            "position": ["WR"] * len(seasons),
            SNAP_ROLL3_COL: [np.nan] * len(seasons),
        })

    def test_pre_2013_rows_are_not_imputed(self):
        df = self._frame([2010, 2012, 2013, 2015])
        out = apply_snap_imputation(df, {("WR", "pre2018"): 55.0, ("__global__", "all"): 50.0})
        got = out[SNAP_ROLL3_COL]
        assert got.iloc[0] != got.iloc[0], "2010 must stay NaN -- no snap data exists"
        assert got.iloc[1] != got.iloc[1], "2012 must stay NaN -- no snap data exists"
        assert got.iloc[2] == 55.0, "2013+ imputation behaviour must be unchanged"
        assert got.iloc[3] == 55.0

    def test_boundary_matches_the_source_table(self):
        assert SNAP_DATA_START_SEASON == 2013


class TestRegularSeasonBoundary:
    """The regular season was 17 weeks through 2020 and 18 from 2021, so a
    flat cap of 18 counted the wild-card round as regular season for every
    pre-2021 fold."""

    def test_boundary_by_era(self):
        from src.models.single_week_ppr.season_projection import regular_season_max_week
        for s in (2006, 2015, 2019, 2020):
            assert regular_season_max_week(s) == 17, s
        for s in (2021, 2024, 2025):
            assert regular_season_max_week(s) == 18, s

    def test_boundary_matches_the_schedule(self):
        """Derived from the schedule rather than asserted: a full slate is
        13+ games, a playoff round is 2-6."""
        import sqlite3
        from config.settings import DB_PATH
        from src.models.single_week_ppr.season_projection import regular_season_max_week
        c = sqlite3.connect(str(DB_PATH))
        try:
            for season in (2015, 2018, 2020, 2021, 2024):
                last = regular_season_max_week(season)
                n_last = c.execute(
                    "SELECT COUNT(*) FROM schedule WHERE season=? AND week=?",
                    (season, last)).fetchone()[0]
                n_next = c.execute(
                    "SELECT COUNT(*) FROM schedule WHERE season=? AND week=?",
                    (season, last + 1)).fetchone()[0]
                assert n_last >= 13, f"{season} wk{last} has {n_last} games, not a full slate"
                assert n_next <= 6, f"{season} wk{last+1} has {n_next} games, not a playoff round"
        finally:
            c.close()

    def test_playoff_week_excluded_from_possible_weeks(self):
        """A 2015 playoff team must get 16 game-weeks, same as a team that
        missed the playoffs -- previously the playoff team got 17."""
        from src.models.single_week_ppr.season_projection import possible_weeks_for_team
        from src.utils.database import DatabaseManager
        db = DatabaseManager()
        playoff = possible_weeks_for_team(db, "NE", 2015)     # made the playoffs
        missed = possible_weeks_for_team(db, "CLE", 2015)     # did not
        assert len(playoff) == len(missed) == 16
        assert max(playoff) <= 17


class TestFumblesLostPopulated:
    """2025 shipped with 0 fumbles lost league-wide because the PBP fallback
    path does not produce the column and the schema helper defaulted it to 0.
    fantasy_points is computed from it at -2 apiece, so the target was
    overstated in the season used as both projection target and newest fold."""

    def test_every_season_has_fumbles(self):
        import sqlite3
        from config.settings import DB_PATH
        c = sqlite3.connect(str(DB_PATH))
        try:
            rows = c.execute(
                "SELECT season, SUM(fumbles_lost) FROM player_weekly_stats "
                "GROUP BY season HAVING SUM(fumbles_lost) < 100").fetchall()
        finally:
            c.close()
        assert not rows, f"seasons with implausibly few fumbles lost: {rows}"

    def test_fantasy_points_reconstructs_from_components(self):
        """Catches a fumbles fix applied to the column but not to the target."""
        import sqlite3
        from config.settings import DB_PATH
        c = sqlite3.connect(str(DB_PATH))
        try:
            d = pd.read_sql(
                "SELECT season, passing_yards, passing_tds, interceptions, "
                "rushing_yards, rushing_tds, receptions, receiving_yards, "
                "receiving_tds, fumbles_lost, two_point_conversions, fantasy_points "
                "FROM player_weekly_stats WHERE season >= 2019", c)
        finally:
            c.close()
        ppr = (d.passing_yards / 25 + d.passing_tds * 4 - d.interceptions * 2
               + d.rushing_yards / 10 + d.rushing_tds * 6
               + d.receptions + d.receiving_yards / 10 + d.receiving_tds * 6
               - d.fumbles_lost * 2 + d.two_point_conversions * 2)
        worst = d.assign(diff=(d.fantasy_points - ppr).abs()).groupby("season")["diff"].max()
        assert (worst < 0.05).all(), f"PPR reconstruction drifts: {worst[worst >= 0.05].to_dict()}"


class TestEraStartConstants:
    """Each constant must match the season its source table actually begins,
    read from the DB rather than trusted. A boundary that drifts from its
    source silently re-creates the bug it was added to fix."""

    def test_boundaries_match_source_tables(self):
        import sqlite3
        from config.settings import DB_PATH
        from src.features.feature_engineering import (
            WEEKLY_PFR_START_SEASON, SEASONAL_PFR_START_SEASON,
            INJURY_DATA_START_SEASON, DEPTH_CHART_START_SEASON,
        )
        c = sqlite3.connect(str(DB_PATH))
        try:
            def first(table):
                return int(next(c.execute(f"SELECT MIN(season) FROM {table}"))[0])
            assert WEEKLY_PFR_START_SEASON == first("weekly_pfr")
            # seasonal_pfr is shifted +1 to become PRIOR-season features
            assert SEASONAL_PFR_START_SEASON == first("seasonal_pfr") + 1
            assert INJURY_DATA_START_SEASON == first("player_injuries")
            assert DEPTH_CHART_START_SEASON == first("depth_charts")
        finally:
            c.close()

    def test_masked_columns_are_exempt_from_the_median_imputer(self):
        """Masking without exempting is inert -- _impute_missing puts the
        constant straight back. Every masked column must be in the exempt set."""
        from src.features.feature_engineering import _STRUCTURALLY_MISSING
        for col in ("qb_pressure_pct_roll3_mean", "recv_drop_pct_roll3_mean",
                    "team_sack_rate_allowed_roll3_mean", "qb_bad_throw_pct_prior",
                    "qb_pocket_time_prior", "injury_score", "depth_chart_rank"):
            assert col in _STRUCTURALLY_MISSING, col


class TestPlayCountDenominators:
    """pass_plays/rush_plays/recv_targets were populated only for 2025, and the
    fallback tested the WHOLE frame, so one 2025 row silently zeroed every
    other season's per-play EPA."""

    def test_fallback_is_per_row_not_per_frame(self):
        """Built from real rows: _create_base_features needs the full weekly
        schema, and a hand-rolled fixture silently diverges from it."""
        import sqlite3
        from config.settings import DB_PATH
        from src.features.feature_engineering import FeatureEngineer

        c = sqlite3.connect(str(DB_PATH))
        try:
            df = pd.read_sql(
                "SELECT * FROM player_weekly_stats "
                "WHERE season IN (2015, 2025) AND passing_attempts > 5 "
                "GROUP BY season LIMIT 2", c)
        finally:
            c.close()
        if len(df) < 2:
            pytest.skip("need one 2015 and one 2025 passing row")

        # Reproduce the pre-fix state for the older row only.
        df.loc[df.season == 2015, "pass_plays"] = 0
        df.loc[:, "pass_epa"] = 10.0
        df.loc[:, "passing_attempts"] = 20

        # The invariant is frame-independence: the 2015 row must compute the
        # same whether or not a populated 2025 row shares the frame. Comparing
        # the two rows to each other would only compare their denominators.
        both = FeatureEngineer()._create_base_features(df.copy())
        alone = FeatureEngineer()._create_base_features(
            df[df.season == 2015].copy())

        v_both = both.loc[both.season == 2015, "pass_epa_per_play"].to_numpy()[0]
        v_alone = alone["pass_epa_per_play"].to_numpy()[0]
        assert v_both == v_alone, (
            "a populated 2025 row changed how 2015 was computed "
            f"({v_both} with it, {v_alone} without)"
        )
        assert v_both != 0, "2015 must not collapse to a fabricated 0.0"

    def test_db_play_counts_are_populated_for_every_season(self):
        import sqlite3
        from config.settings import DB_PATH
        c = sqlite3.connect(str(DB_PATH))
        try:
            bad = c.execute("""
                SELECT season FROM player_weekly_stats
                GROUP BY season
                HAVING SUM(pass_plays = passing_attempts) <> COUNT(*)
                    OR SUM(rush_plays = rushing_attempts) <> COUNT(*)
                    OR SUM(recv_targets = targets) <> COUNT(*)
            """).fetchall()
        finally:
            c.close()
        assert not bad, f"play-count columns disagree with box score in: {bad}"


class TestSnapAccelEraMask:
    """snap_share_accel needed THREE separate exemptions before the era mask
    survived: the mask itself, the utilization policy group's exclude list,
    and _STRUCTURALLY_MISSING (whose median is ~0.0, so the fill was invisible
    -- it put back exactly the value being removed)."""

    def test_pre_era_rows_are_nan_not_flat(self):
        from src.features.feature_engineering import FeatureEngineer
        df = pd.DataFrame({
            "player_id": ["p1"] * 8,
            "season": [2011] * 4 + [2015] * 4,
            "week": [1, 2, 3, 4, 1, 2, 3, 4],
            "snap_share_pct": [np.nan] * 4 + [50.0, 55.0, 60.0, 52.0],
        })
        out = FeatureEngineer()._create_causal_rolling_features(df)
        pre = out[out.season < SNAP_DATA_START_SEASON]["snap_share_accel"]
        post = out[out.season >= SNAP_DATA_START_SEASON]["snap_share_accel"]
        assert pre.isna().all(), "pre-2013 must not assert flat usage"
        assert post.notna().all(), "2013+ keeps its 0.0 'too few prior weeks' default"

    def test_exempt_from_both_downstream_fillers(self):
        from src.features.feature_engineering import _STRUCTURALLY_MISSING
        from src.features.feature_policy_registry import FeaturePolicyRegistry
        assert "snap_share_accel" in _STRUCTURALLY_MISSING
        r = FeaturePolicyRegistry.from_config()
        group = r.resolve_group_for_feature("snap_share_accel")
        assert "snap_share_accel" in r.policies[group].exclude
