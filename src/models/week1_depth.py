"""Week-1 depth-chart rank as a cold-start signal.

The rejected-alternatives round (GAPS.md 2026-09-05) ended by pointing at the
slice rather than the pie: for a player with no NFL history, week 1 is mostly a
question of whether he is on the field, and neither team-room output nor
opponent strength answers it. Depth-chart rank answers it directly.

    week-1 PPR by rank, cold-start players 2013-2025 (n=517 of 527 charted)
        rank 1   9.45      QB1 15.40   RB1 10.72   TE1 6.05   WR1 7.59
        rank 2   3.97      QB2  4.07   RB2  5.37   TE2 2.73   WR2 3.91
        rank 3   2.62      QB3  0.10   RB3  3.16   TE3 0.76   WR3 4.03

TIMING IS THE WHOLE QUESTION, AND IT HAS A KNOWN ANSWER FOR THE BOARD.
GAPS.md already rejected a depth-chart override for the JULY/AUGUST draft
board: `backfill_depth_charts_2025.py` keeps one snapshot per week, the last
one strictly before kickoff, so the stored "week 1" chart is a early-September
snapshot that does not exist when the board is generated. Training on it and
serving nothing is train/serve skew.

That objection is about WHEN THE BOARD RUNS, not about the signal. A projection
produced in the days before week 1 -- which is exactly when the league report
runs -- can legally read the week-1 chart. So this module is only usable by an
in-season consumer, and `depth_charts` must actually contain the target
season before anything here can serve it.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from config.settings import DB_PATH

POSITIONS = ("QB", "RB", "WR", "TE")

# Shrinkage toward the position mean by cell size, the same device and scale
# the availability prior uses for its draft buckets.
BUCKET_K = 10.0


def week1_depth_ranks(seasons=None, con=None) -> pd.DataFrame:
    """Best (lowest) week-1 depth-chart rank per player and season.

    MIN across the rows a player has: a back listed RB2 and KR3 is an RB2.
    """
    owns = con is None
    con = sqlite3.connect(str(DB_PATH)) if owns else con
    try:
        dc = pd.read_sql(
            "SELECT season, gsis_id AS player_id, "
            "       MIN(CAST(depth_team AS INTEGER)) AS depth_rank "
            "FROM depth_charts WHERE week = 1 AND gsis_id IS NOT NULL "
            "AND depth_team IS NOT NULL GROUP BY season, gsis_id", con)
    finally:
        if owns:
            con.close()
    dc["season"] = dc["season"].astype(int)
    if seasons is not None:
        dc = dc[dc["season"].isin(list(seasons))]
    return dc


class RankPrior:
    """Mean week-1 points for a (position, rank) cell, shrunk by cell size.

    Deliberately not a regression. The whole claim is that rank is a strong,
    low-dimensional signal, and twelve cells fitted on a few hundred rows is
    the most that sample supports; anything richer would be fitting the
    seasons rather than the ranks.
    """

    def __init__(self, bucket_k: float = BUCKET_K):
        self.bucket_k = bucket_k
        self.cells = {}
        self.position_means = {}
        self.overall = 0.0

    def fit(self, rows: pd.DataFrame):
        rows = rows.dropna(subset=["actual"])
        self.overall = float(rows["actual"].mean())
        for pos, g in rows.groupby("position"):
            self.position_means[pos] = float(g["actual"].mean())
            for rank, cell in g.dropna(subset=["depth_rank"]).groupby("depth_rank"):
                w = len(cell) / (len(cell) + self.bucket_k)
                self.cells[(pos, int(rank))] = (
                    w * float(cell["actual"].mean())
                    + (1 - w) * self.position_means[pos])
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        out = []
        for pos, rank in zip(rows["position"], rows.get("depth_rank",
                                                        pd.Series(dtype=float))):
            fallback = self.position_means.get(pos, self.overall)
            if pd.isna(rank):
                # Unlisted is not rank 3. It is no information, so the
                # position mean stands in rather than a guess at the worst.
                out.append(fallback)
            else:
                out.append(self.cells.get((pos, int(rank)), fallback))
        return np.asarray(out, dtype=float)
