"""Production-training population definitions, keyed on observed participation.

The production model estimates E[PPR | the player participated], so a row is
only eligible as a training target when we actually observed participation.
The evidence available for that varies by era, and the whole point of this
module is that the eras must not silently masquerade as each other:

    2013-present   offense_snaps > 0                  quality 2 (snap-confirmed)
    2006-2012      PPR > 0 (or PBP-confirmed)         quality 1 (inferred)
    neither        not a production observation       quality 0 (excluded)

2013 is not "where modern football starts" -- it is the first season with PFR
snap coverage (scripts/build_complete_player_game_panel.py SNAP_COUNT_MIN_SEASON),
i.e. the start of the high-confidence participation-label regime.

The quality-1 proxy is deliberately conservative in one direction: PPR > 0
almost certainly means the player participated, while PPR = 0 is ambiguous
(played and produced nothing, or never dressed). So it costs us legitimate
zero-point games from 2006-2012 rather than admitting thousands of
inactive/IR/practice-squad weeks. False exclusion, not false inclusion.

Note what is NOT lost by excluding a row here: features are engineered over
the full panel before this filter runs, so 2006-2012 still feeds career
history, aging curves and rolling windows even under regime A.

Regimes (compared in scripts/run_population_regime_experiment.py):
    A  clean modern    quality == 2
    B  extended        quality >= 1
    C  extended+flag   quality >= 1, with participation_quality as a feature
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

# First season with PFR game-level snap data. Must track
# scripts/build_complete_player_game_panel.py:SNAP_COUNT_MIN_SEASON.
SNAP_LABEL_MIN_SEASON = 2013

# First season nflverse charts the intended receiver on INCOMPLETE passes.
# Before 2009 it charts essentially only completions -- 2008 play-by-play
# names a receiver on 69 incompletions, 2009 on 6,731 -- so `targets`
# degenerates into a count of receptions and everything built on it is
# wrong at source, not in this pipeline (GAPS.md 2026-08-20):
#
#     catch rate       2006-2008 reads 99.7%, vs ~61% from 2009
#     recv_success_rate 0.847 vs 0.578 (+46%)
#     recv_epa          ~4x inflated, because incompletions carry the
#                       negative EPA and were never recorded
#
# Same shape as SNAP_LABEL_MIN_SEASON: not "old football is different" but
# "this is where the measurement regime starts". QB is exempt -- its
# features depend on passing and NGS, not on target charting.
RECEIVING_CHARTING_MIN_SEASON = 2009
RECEIVING_DEPENDENT_POSITIONS = frozenset({"RB", "WR", "TE"})

QUALITY_UNOBSERVED = 0
QUALITY_INFERRED = 1
QUALITY_SNAP_CONFIRMED = 2

QUALITY_COLUMN = "participation_quality"

REGIMES = ("A_clean_modern", "B_extended", "C_extended_flagged")

# Snap buckets for the participation-vs-production diagnostic.
SNAP_BUCKET_EDGES = [0, 5, 15, 30, 50, np.inf]
SNAP_BUCKET_LABELS = ["0-5", "5-15", "15-30", "30-50", "50+"]


def label_participation(df: pd.DataFrame) -> pd.Series:
    """Per-row participation-evidence quality (0/1/2). See module docstring.

    A pre-2013 row is quality 1 on PPR > 0 alone; the panel's tiny
    `inferred_pbp_confirmed_zero` tier is also quality 1, because those rows
    have direct play-by-play evidence of participation even though they
    scored zero -- excluding them would contradict the contract on exactly
    the rows it was written to protect.

    A snap-era row with a NULL snap_count is quality 0, not 1. Those are
    stat rows the snap feed failed to match (~1.7K of 119K); admitting them
    on the PPR proxy would let the weaker label leak into the regime whose
    entire claim is that its labels are snap-confirmed.
    """
    season = pd.to_numeric(df["season"], errors="coerce")
    snaps = pd.to_numeric(df.get("snap_count"), errors="coerce")
    points = pd.to_numeric(df.get("fantasy_points"), errors="coerce")
    source = df["data_source"] if "data_source" in df.columns else pd.Series(index=df.index, dtype="object")

    quality = pd.Series(QUALITY_UNOBSERVED, index=df.index, dtype="int64")

    snap_era = season >= SNAP_LABEL_MIN_SEASON
    quality[snap_era & (snaps > 0)] = QUALITY_SNAP_CONFIRMED
    quality[~snap_era & (points > 0)] = QUALITY_INFERRED
    quality[~snap_era & (source == "inferred_pbp_confirmed_zero")] = QUALITY_INFERRED
    return quality


def with_participation_quality(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[QUALITY_COLUMN] = label_participation(out)
    return out


def receiving_floor_mask(df: pd.DataFrame) -> pd.Series:
    """False for receiving-dependent rows below the target-charting floor.

    A pre-2009 WR row still has a valid PPR target (receptions, yards and
    touchdowns all come from the box score and are correct). What is broken
    is its usage features. Keeping the row with plausible-looking but wrong
    target counts is the exact failure this project spent 2026-08-20
    removing, so the row goes.

    QB rows are unaffected and stay.
    """
    season = pd.to_numeric(df["season"], errors="coerce")
    position = df["position"] if "position" in df.columns else pd.Series(index=df.index, dtype=object)
    receiving_dependent = position.isin(RECEIVING_DEPENDENT_POSITIONS)
    return ~(receiving_dependent & (season < RECEIVING_CHARTING_MIN_SEASON))


def apply_regime(df: pd.DataFrame, regime: str, apply_receiving_floor: bool = True) -> pd.DataFrame:
    """Filters `df` to the training population for `regime`.

    Adds `participation_quality` if absent. C returns the same rows as B --
    the arms differ only in whether the column is handed to the model, which
    is `regime_feature_columns`' job.

    `apply_receiving_floor` drops RB/WR/TE rows before 2009, where target
    charting does not exist. Off only for measuring what the floor costs.
    """
    if regime not in REGIMES:
        raise ValueError(f"Unknown regime: {regime!r} (expected one of {REGIMES})")
    if QUALITY_COLUMN not in df.columns:
        df = with_participation_quality(df)
    if apply_receiving_floor:
        df = df[receiving_floor_mask(df)]
    if regime == "A_clean_modern":
        return df[df[QUALITY_COLUMN] == QUALITY_SNAP_CONFIRMED]
    return df[df[QUALITY_COLUMN] >= QUALITY_INFERRED]


def regime_feature_columns(feature_cols: List[str], regime: str) -> List[str]:
    """Only regime C exposes the label-quality flag to the model.

    At test time every row is quality 2, so C cannot use the flag to cheat:
    it can only learn a correction for the historical regime and then apply
    the modern branch when predicting.
    """
    if regime == "C_extended_flagged" and QUALITY_COLUMN not in feature_cols:
        return list(feature_cols) + [QUALITY_COLUMN]
    return list(feature_cols)


def evaluation_population(df: pd.DataFrame) -> pd.DataFrame:
    """The held-out rows every regime is scored on: snap-confirmed only.

    Held identical across arms so the comparison is a training-population
    contrast and nothing else.
    """
    if QUALITY_COLUMN not in df.columns:
        df = with_participation_quality(df)
    return df[df[QUALITY_COLUMN] == QUALITY_SNAP_CONFIRMED]


def snap_bucket(snaps: pd.Series) -> pd.Series:
    """Buckets offensive snaps for the participation/production diagnostic.

    Right-closed so a 5-snap game lands in "0-5" and a 50-snap game in
    "30-50", matching how the bucket names read.
    """
    return pd.cut(pd.to_numeric(snaps, errors="coerce"),
                  bins=SNAP_BUCKET_EDGES, labels=SNAP_BUCKET_LABELS, right=True)


def tenure_bucket(df: pd.DataFrame) -> pd.Series:
    """Seasons of experience, from `first_season` where present.

    Falls back to NaN rather than guessing, so a missing career start shows
    up as an unclassified bucket instead of silently becoming a rookie.
    """
    season = pd.to_numeric(df["season"], errors="coerce")
    first = pd.to_numeric(df.get("first_season"), errors="coerce")
    tenure = season - first
    tenure = tenure.where(tenure >= 0)
    return pd.cut(tenure, bins=[-0.5, 0.5, 2.5, 5.5, np.inf],
                  labels=["rookie", "1-2y", "3-5y", "6y+"])


def age_bucket(df: pd.DataFrame) -> pd.Series:
    age = pd.to_numeric(df.get("age"), errors="coerce")
    return pd.cut(age, bins=[-np.inf, 23, 26, 29, np.inf],
                  labels=["<=23", "24-26", "27-29", "30+"])
