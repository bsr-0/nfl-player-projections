"""Training LABELS must never be imputed.

`FeatureEngineer._impute_missing` walks every numeric column and median-fills
it. The horizon targets are numeric columns living in the same frame, so they
were filled like any predictor. A NaN target means "this row has no future to
predict" -- the season ended, or the forward window ran off the end of the
data. Every training path guards on `~y_dict[1].isna()`; the fill defeated that
guard by replacing the drop with a fabricated label.

Measured 2018-2025 (30,065 rows) before the fix:

    target_1w    90.6% -> 100% coverage   ( 9.4% of labels invented)
    target_4w    73.8% -> 100%            (26.2% invented)
    target_18w    9.3% -> 100%            (90.7% invented)

The 18w column was the tell: six values covered 90% of it, and the quartiles
were identical across all four positions (p25 127.55 / p75 134.20 for QB and
TE alike), which real per-player targets do not do.

The naming makes this easy to get wrong in the other direction: `target_1w` is
a label, `target_share_rolling_3` is a legitimate feature. Both tests below
matter -- one that labels survive, one that the target-share family does not.
"""
import numpy as np
import pandas as pd

from src.features.feature_engineering import FeatureEngineer
from src.utils.leakage import is_label_column


LABEL_COLS = ["target_1w", "target_4w", "target_18w", "target_util_1w"]
FEATURE_COLS_NAMED_TARGET = [
    "target_share",
    "target_share_pct",
    "target_share_rolling_3",
    "target_share_norm",
]


def _frame_with_missing_labels() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame({
        "player_id": ["p1"] * 30 + ["p2"] * 30,
        "position": ["RB"] * 30 + ["QB"] * 30,
        "season": [2022] * 60,
        "week": list(range(1, 31)) * 2,
        "fantasy_points": rng.uniform(0, 25, n),
    })
    # Labels: trailing rows NaN, which is what a forward window produces.
    for col in LABEL_COLS:
        vals = rng.uniform(0, 100, n)
        vals[-12:] = np.nan
        df[col] = vals
    # Features that merely LOOK like labels; these must still be filled.
    for col in FEATURE_COLS_NAMED_TARGET:
        vals = rng.uniform(0, 1, n)
        vals[:5] = np.nan
        df[col] = vals
    return df


def test_is_label_column_distinguishes_labels_from_target_share():
    for col in LABEL_COLS:
        assert is_label_column(col), f"{col} is a label"
    for col in FEATURE_COLS_NAMED_TARGET:
        assert not is_label_column(col), f"{col} is a feature, not a label"


def test_impute_missing_leaves_labels_untouched():
    df = _frame_with_missing_labels()
    before = {c: df[c].isna().sum() for c in LABEL_COLS}
    out = FeatureEngineer()._impute_missing(df.copy())

    for col in LABEL_COLS:
        assert out[col].isna().sum() == before[col], (
            f"{col} was imputed: {before[col]} NaN before, "
            f"{out[col].isna().sum()} after. A filled label is a fabricated "
            f"training row, not a recovered one."
        )


def test_impute_missing_still_fills_target_share_features():
    df = _frame_with_missing_labels()
    out = FeatureEngineer()._impute_missing(df.copy())

    for col in FEATURE_COLS_NAMED_TARGET:
        assert out[col].isna().sum() == 0, (
            f"{col} is a feature and must still be imputed; a blanket "
            f"'startswith(target_)' exemption would wrongly spare it."
        )


def test_label_coverage_is_not_total_after_engineering():
    """Coverage of exactly 100% is the signature of the bug.

    A forward-looking target cannot be defined on the final rows of a
    player-season, so full coverage means something filled it.
    """
    df = _frame_with_missing_labels()
    out = FeatureEngineer()._impute_missing(df.copy())
    for col in LABEL_COLS:
        assert out[col].notna().mean() < 1.0, (
            f"{col} reached 100% coverage, which a forward window cannot "
            f"produce."
        )
