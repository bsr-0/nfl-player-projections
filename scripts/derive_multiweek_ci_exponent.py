#!/usr/bin/env python3
"""
Derive a real, per-position exponent for scaling a single-week prediction
std into an n-week-total std (GAPS.md: "n_weeks**0.4 multi-week CI-scaling
exponent" open item).

The exponent only depends on the autocorrelation structure of real weekly
fantasy-point time series -- it does not require model predictions or new
backtest infrastructure. For n independent weeks, Var(sum) = n * Var(week),
i.e. std(n) = std(1) * n**0.5. Autocorrelated weeks (hot/cold streaks, role
changes) push the true exponent above 0.5. This script measures that
directly from player_weekly_stats.

Method: for each position and a range of window sizes N, compute the
realized forward-looking N-week rolling sum of fantasy_points per
player-season (mirrors feature_preparation.py's _forward_window, read-only
reuse of the same logic), take its std across all valid player-season-week
rows, and fit std(N) = std(1) * N**alpha via log-log regression across N.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DB_PATH, POSITIONS

WINDOWS = [1, 2, 4, 8, 13, 18]


def _forward_sum(series: pd.Series, window: int) -> pd.Series:
    """Forward-looking rolling sum over the next `window` games, strict
    75%-of-window min_periods (mirrors feature_preparation.py's sum-target
    logic, kept in sync deliberately since both describe the same
    quantity: a real forward multi-week FP total)."""
    shifted = series.shift(-1) if window > 1 else series
    rev = shifted.iloc[::-1]
    min_p = 1 if window <= 1 else max(int(np.ceil(window * 0.75)), 2)
    return rev.rolling(window=window, min_periods=min_p).sum().iloc[::-1]


def load_weekly_fp() -> pd.DataFrame:
    import sqlite3
    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql(
        """
        SELECT pws.player_id, pws.season, pws.week, pws.fantasy_points, p.position
        FROM player_weekly_stats pws
        JOIN players p ON p.player_id = pws.player_id
        WHERE pws.fantasy_points IS NOT NULL
          AND p.position IN ('QB','RB','WR','TE')
        """,
        conn,
    )
    conn.close()
    return df


def fit_exponent(std_by_window: dict) -> tuple:
    """Fit std(N) = std(1) * N**alpha via log-log OLS. Returns (alpha, r2)."""
    windows = sorted(std_by_window.keys())
    if len(windows) < 2:
        return None, None
    x = np.log(np.array(windows, dtype=float))
    y = np.log(np.array([std_by_window[w] for w in windows], dtype=float))
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return None, None
    x, y = x[valid], y[valid]
    alpha, intercept = np.polyfit(x, y, 1)
    y_pred = alpha * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(alpha), float(r2)


def derive_position_exponents(df: pd.DataFrame) -> dict:
    results = {}
    for position in POSITIONS:
        pos_df = df[df["position"] == position].copy()
        pos_df = pos_df.sort_values(["player_id", "season", "week"])
        std_by_window = {}
        n_by_window = {}
        for window in WINDOWS:
            sums = pos_df.groupby(["player_id", "season"])["fantasy_points"].transform(
                lambda x, w=window: _forward_sum(x, w)
            )
            valid = sums.dropna()
            if len(valid) < 30:
                continue
            std_by_window[window] = float(valid.std())
            n_by_window[window] = int(len(valid))
        alpha, r2 = fit_exponent(std_by_window)
        results[position] = {
            "alpha": alpha,
            "r2": r2,
            "std_by_window": std_by_window,
            "n_by_window": n_by_window,
        }
    return results


def main():
    print("Loading player_weekly_stats.fantasy_points joined to players.position...")
    df = load_weekly_fp()
    print(f"  {len(df)} weekly rows, seasons {df['season'].min()}-{df['season'].max()}")

    results = derive_position_exponents(df)

    print("\n" + "=" * 70)
    print("  FITTED n_weeks EXPONENT (std(N) = std(1) * N**alpha)")
    print("=" * 70)
    for position, r in results.items():
        print(f"\n{position}:")
        for w in WINDOWS:
            if w in r["std_by_window"]:
                print(f"    N={w:2d}: std={r['std_by_window'][w]:6.2f}  n={r['n_by_window'][w]}")
        if r["alpha"] is not None:
            print(f"  -> alpha={r['alpha']:.3f}  (R2={r['r2']:.4f})  "
                  f"[naive i.i.d. sqrt-law would be 0.500]")
        else:
            print("  -> insufficient data to fit")

    return results


if __name__ == "__main__":
    main()
