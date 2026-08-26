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
