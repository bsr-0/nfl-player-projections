#!/usr/bin/env python
"""Pre-registration gate: prove variants A and B differ MECHANICALLY.

The first A/B was silently invalid -- `_create_base_features` recomputed
snap_share_pct through safe_divide, so both arms saw identical values and
the only difference was a `known` column that could take just two values.
The measured effect belonged to a has-prior-history indicator, not to snap
missingness (GAPS.md 2026-08-19 correction).

This runs BEFORE the 8 backtest runs and aborts if the arms are not
distinguishable on the real feature frames. Exit 0 = proceed, 1 = abort.

Checks, on real 2018-2023 train / 2024 test frames built through the actual
leakage_safe_features path:

    A: snap_share_pct_roll3_known takes {0, 1} only
    B: snap_share_pct_roll3_known takes graded values (0, 1/3, 2/3, 1)
    B: snap_share_pct_roll3_mean differs from A on a nonzero row count

Usage:
    python scripts/check_snap_ab_arms_differ.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

TRAIN_SEASONS = (2018, 2023)
TEST_SEASON = 2024
POSITIONS = ("RB", "WR", "TE")


def build(mode: str):
    """Feature frames under one variant's upstream mode."""
    from src.features import utilization_score as us
    from src.evaluation.ts_backtester import leakage_safe_features
    from src.utils.database import DatabaseManager

    us.set_snap_missingness_mode("zero" if mode == "A" else "preserve")

    db = DatabaseManager()
    data = pd.concat(
        [db.get_all_players_for_training(position=p, min_games=1) for p in POSITIONS],
        ignore_index=True,
    )
    train = data[data.season.between(*TRAIN_SEASONS)].copy()
    test = data[data.season == TEST_SEASON].copy()
    train_out, test_out = leakage_safe_features(train, test)

    if mode == "B":
        values = us.fit_snap_imputation(train_out)
        train_out = us.apply_snap_imputation(train_out, values)
        test_out = us.apply_snap_imputation(test_out, values)
    return train_out, test_out


def main() -> int:
    from src.features import utilization_score as us

    original = us.SNAP_MISSINGNESS_MODE
    try:
        a_train, a_test = build("A")
        b_train, b_test = build("B")
    finally:
        us.set_snap_missingness_mode(original)

    roll, known = us.SNAP_ROLL3_COL, us.SNAP_KNOWN_COL
    failures = []

    a_vals = sorted(np.round(a_test[known].dropna().unique(), 3))
    b_vals = sorted(np.round(b_test[known].dropna().unique(), 3))
    print(f"A {known}: {a_vals}")
    print(f"B {known}: {b_vals}")
    if len(a_vals) > 2:
        failures.append(f"A's indicator is graded ({a_vals}); A should be binary")
    if len(b_vals) <= 2:
        failures.append(
            f"B's indicator is not graded ({b_vals}); upstream missingness is "
            "being destroyed before the rolling feature -- the arms are not "
            "testing snap missingness"
        )

    for name, fa, fb in (("test", a_test, b_test), ("train", a_train, b_train)):
        left = pd.to_numeric(fa[roll], errors="coerce").fillna(-999)
        right = pd.to_numeric(fb[roll], errors="coerce").fillna(-999)
        n_diff = int((~np.isclose(left, right, atol=1e-9)).sum())
        print(f"{name}: {roll} differs on {n_diff:,} / {len(fa):,} rows")
        if name == "test" and n_diff == 0:
            failures.append(
                f"{roll} is identical between arms on the {name} frame; "
                "the treatment has no effect on the feature matrix"
            )

    n_known_diff = int((a_test[known] != b_test[known]).sum())
    print(f"test: {known} differs on {n_known_diff:,} rows")
    if n_known_diff == 0:
        failures.append(f"{known} is identical between arms")

    if failures:
        print("\nABORT -- the arms are not mechanically distinguishable:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nPROCEED -- arms differ in the intended way.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
