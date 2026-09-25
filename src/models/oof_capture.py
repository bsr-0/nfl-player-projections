"""Row-level out-of-fold predictions from walk-forward validation.

No persisted row-level prediction artifact existed for the served weekly
model. `position_models.py` computes OOF predictions internally for
meta-learner stacking and isotonic calibration but keeps only aggregate
metrics; `train.py --walk-forward` reported four numbers per position and
discarded everything else. Two separate pieces of work were blocked on this:

  * Segment evaluation -- by position x tier, and returning players vs
    cold-start players with no prior starts. Aggregate MAE cannot tell you
    whether a change helps starters and hurts cold-start players equally.
  * The correlation layer (docs/GAME_SIMULATION_CORRELATION_PLAN.md, Phase
    2), whose stated design fits same-game residual covariance "from
    out-of-fold player predictions" and so had nothing to fit on.

Each walk-forward fold trains on seasons 1..N-1 and predicts season N, so a
fold's test-set predictions are genuinely out-of-fold for that season.
Concatenating the folds gives an OOF panel covering the validated seasons,
with no row ever predicted by a model that saw it.

The leakage guarantee is enforced, not assumed: `capture_fold_rows` raises if
a fold's test season appears among its training seasons, or if the frame
carries rows from any other season. A silently contaminated panel would make
every downstream segment number look better than it is.

## 2026-09-25 methodology review

An adversarial review (one lens completed before the reviewing session hit a
rate limit; its claims were independently re-verified against
`data/nfl_data.db` before acting on them -- the composition percentages,
zero-rates, and skew below all replicated within rounding) found the first
version of this module had five real problems, now fixed in the functions
below:

1. `is_cold_start` (panel-relative first appearance) is not primarily a
   novelty signal: of ~930 panel-defined cold-start rows, 65% are week<=2
   (vs 12% of returning rows) and 63% fall in the 2023 fold, the shortest
   training window. For QB specifically, 73% of the "cold-start" cell is
   2023 opening-day starters -- established veterans, not new players. In
   plain terms: cold-start-in-the-panel mostly measures "week 1 of the
   oldest fold," not "a player's first NFL start." `week_bucket` now makes
   that visible in the default segmentation, and
   `add_career_experience_segments` builds the metric the docstring always
   claimed to be describing.
2. `segment_report` reported point estimates with no uncertainty on rows
   that are not independent (same player across weeks, same game across
   players). It now takes an optional `cluster` column and reports a
   cluster bootstrap CI, so a segment difference that is really just an
   underpowered or non-independent sample stops looking like a verdict.
3. No artifact recorded which run produced it, so a second walk-forward run
   silently overwrote the first with nothing to compare against --
   `write_run_panel` stamps provenance and writes an immutable per-run copy,
   the same discipline commit ca4b5e1 added for model artifacts one commit
   before this module was written.
4. Rows dropped for lacking a prediction were logged, not persisted, so two
   panels could differ in which rows they cover with no way to see it.
   `fold_coverage` records offered/captured/dropped per (fold, position).
5. MAE/RMSE/bias on raw fantasy points understate how zero-inflated the
   outcome is (TE is 39% near-zero, WR 26%, skew up to 1.9), and the zero
   rate itself varies by the segments being compared -- so a segment MAE gap
   can be a composition difference, not a skill difference.
   `segment_report` now also reports `zero_rate`, `mae_positive` (MAE
   conditional on a nonzero actual), and Spearman rank correlation.

What was NOT added: formal multiple-comparison correction (Holm/BH) across
the segment cells, or a minimum-detectable-effect column. The CI half-width
serves the same purpose a reader needs most -- a wide interval says "this
cell can't support a conclusion" -- documented here rather than automated,
because the right correction depends on which cells are primary vs
exploratory for a given comparison, which this module cannot know.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Identity, context, and the (predicted, actual) pair. `opponent` and `team`
# are kept because the correlation layer groups residuals by (season, week,
# team) to build same-game vectors.
IDENTITY_COLUMNS = ("player_id", "season", "week", "team", "opponent", "position")
PREDICTION_COLUMN = "predicted_points"
ACTUAL_COLUMN = "actual_for_backtest"

OOF_PANEL_FILENAME = "walk_forward_oof_predictions.parquet"
# A row this near zero is a DNP/inactive/non-factor, not "the model predicted
# ~0 and was right" vs "~0 and was wrong" in any meaningful sense -- matches
# the fantasy-points near-zero floor used elsewhere in this repo's cache gate.
ZERO_FLOOR = 0.05

# A player is a career cold-start if they have no player_weekly_stats row at
# any (season, week) strictly before their first row in the OOF panel. Kept
# separate from IDENTITY_COLUMNS since it requires a DB round-trip that
# build_panel (deliberately DB-independent, for testability) does not do.
CAREER_LOOKBACK_TABLE = "player_weekly_stats"


class OOFLeakageError(RuntimeError):
    """A fold's captured rows are not out-of-fold."""


def capture_fold_rows(
    test_data: pd.DataFrame,
    *,
    train_seasons: Sequence[int],
    test_season: int,
) -> pd.DataFrame:
    """Extract one fold's out-of-fold rows.

    Raises OOFLeakageError if the fold's test season is also a training
    season, or if `test_data` contains rows outside that season -- either
    would mean the "held-out" predictions were partly in-sample.
    """
    if test_season in set(int(s) for s in train_seasons):
        raise OOFLeakageError(
            f"test season {test_season} is also a training season; these "
            f"predictions are in-sample, not out-of-fold")

    present = set(pd.to_numeric(test_data["season"], errors="coerce").dropna().astype(int))
    if present - {int(test_season)}:
        raise OOFLeakageError(
            f"fold for {test_season} carries rows from {sorted(present)}; "
            "a fold's test frame must hold only its held-out season")

    missing = [c for c in IDENTITY_COLUMNS if c not in test_data.columns]
    if missing:
        raise ValueError(f"test_data is missing identity columns: {missing}")
    for column in (PREDICTION_COLUMN, ACTUAL_COLUMN):
        if column not in test_data.columns:
            raise ValueError(f"test_data is missing {column!r}")

    rows = test_data.loc[:, [*IDENTITY_COLUMNS, PREDICTION_COLUMN, ACTUAL_COLUMN]].copy()
    rows = rows.rename(columns={ACTUAL_COLUMN: "actual_points"})

    # A row the model could not score is not evidence about the model. Drop
    # it here rather than letting a NaN-filled zero masquerade as a
    # prediction downstream -- the same silent-zero-fill defect this repo
    # has hit before in reconstruct_partial_fantasy_points.
    before = len(rows)
    rows = rows[rows[PREDICTION_COLUMN].notna() & rows["actual_points"].notna()]
    dropped = before - len(rows)
    if dropped:
        logger.info("fold %s: dropped %d/%d rows lacking a prediction or actual",
                    test_season, dropped, before)

    rows["train_seasons"] = ",".join(str(int(s)) for s in sorted(train_seasons))
    rows["n_train_seasons"] = len(set(int(s) for s in train_seasons))
    rows["residual"] = rows[PREDICTION_COLUMN] - rows["actual_points"]
    return rows.reset_index(drop=True)


def fold_coverage(
    test_data: pd.DataFrame,
    captured: pd.DataFrame,
    *,
    test_season: int,
) -> pd.DataFrame:
    """Per-position offered vs captured row counts for one fold.

    `test_data` is the fold's full test frame BEFORE `capture_fold_rows`
    filters it (still holding every position, including ones a fold skipped
    for having too few test rows, or that failed to score). Comparing it
    against what `capture_fold_rows` actually returned is the only way to
    see a coverage difference between two panels: two arms with different
    per-position success rates can otherwise produce different row sets with
    nothing in either artifact recording that they differ.
    """
    offered = test_data.groupby("position").size().rename("n_offered")
    if captured.empty:
        # An empty Series' default index has no name, which drops "position"
        # from the concat result below (pandas keeps a named index only when
        # every piece agrees on the name) -- pin it explicitly.
        got = pd.Series(dtype="int64", name="n_captured", index=pd.Index([], name="position"))
    else:
        got = captured.groupby("position").size().rename("n_captured")
    coverage = pd.concat([offered, got], axis=1).fillna(0).astype(int).reset_index()
    coverage["test_season"] = int(test_season)
    coverage["n_dropped"] = coverage["n_offered"] - coverage["n_captured"]
    return coverage[["test_season", "position", "n_offered", "n_captured", "n_dropped"]]


def add_experience_segments(panel: pd.DataFrame) -> pd.DataFrame:
    """Label each row by the player's prior experience *within the panel*.

    `prior_weeks_in_panel` counts that player's earlier appearances in the
    OOF panel, ordered by (season, week), computed with a strict shift so a
    row never counts itself. `is_cold_start` marks a player's first
    appearance in the panel.

    `week_bucket` ('early' for week<=2, else 'mid_late') is included because
    `is_cold_start` is heavily confounded with early-season weeks (see module
    docstring): a segment table grouped only by `is_cold_start` hides that
    confound, one grouped by `(is_cold_start, week_bucket)` does not.

    This is deliberately panel-relative, not career-relative: it answers "how
    much had this model seen of this player by this point in the validation",
    which is a real and useful question, but NOT the same question as "has
    this player started an NFL game before". Use
    `add_career_experience_segments` for that one; do not conflate them.
    """
    out = panel.sort_values(["player_id", "season", "week"]).copy()
    out["prior_weeks_in_panel"] = out.groupby("player_id").cumcount()
    out["is_cold_start"] = out["prior_weeks_in_panel"] == 0
    out["week_bucket"] = np.where(out["week"] <= 2, "early", "mid_late")
    return out.reset_index(drop=True)


def add_career_experience_segments(
    panel: pd.DataFrame,
    *,
    db_path=None,
) -> pd.DataFrame:
    """Best-effort career-relative cold-start, using full DB history.

    `is_cold_start` (panel-relative) mislabels a returning veteran as
    "cold-start" whenever their first panel row happens to be the panel's
    earliest season for them -- see the module docstring's measured example
    (73% of the QB cold-start cell was 2023 opening-day starters). This
    queries `player_weekly_stats` for any row strictly before a player's
    first panel appearance and sets `is_cold_start_career` from that instead.

    Fails open: if the DB is unreachable, `is_cold_start_career` is set to
    NaN with a warning rather than raising, because this is a reporting
    augmentation, not a leakage guarantee -- unlike `capture_fold_rows`,
    getting this wrong degrades a summary, it does not corrupt training.
    """
    out = panel.copy()
    out["is_cold_start_career"] = np.nan
    if out.empty:
        return out

    try:
        import sqlite3
        from src.utils.database import DatabaseManager

        db = DatabaseManager(db_path) if db_path else DatabaseManager()
        conn = sqlite3.connect(f"file:{db.db_path}?mode=ro", uri=True)
        history = pd.read_sql(
            f"SELECT player_id, season, week FROM {CAREER_LOOKBACK_TABLE}", conn)
    except Exception as e:  # noqa: BLE001 -- reporting augmentation, fail open
        logger.warning("career-relative cold-start unavailable (%s); "
                       "is_cold_start_career left as NaN", e)
        return out

    if history.empty:
        return out

    first_panel_row = (
        out.sort_values(["player_id", "season", "week"])
        .groupby("player_id")[["season", "week"]].first()
    )
    history = history.merge(first_panel_row, on="player_id", suffixes=("", "_first"))
    prior_exists = (
        (history["season"] < history["season_first"])
        | ((history["season"] == history["season_first"]) & (history["week"] < history["week_first"]))
    )
    has_prior_career_row = (
        history.loc[prior_exists, "player_id"].drop_duplicates()
    )
    cold_start_career = ~out["player_id"].isin(has_prior_career_row)
    # Only a player's own panel-first row can be a career cold start; a
    # returning player's later panel rows are never candidates regardless of
    # career history, matching the panel-relative field's own scope.
    out["is_cold_start_career"] = out["is_cold_start"] & cold_start_career
    return out


def build_panel(fold_rows: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate captured folds into one OOF panel."""
    frames = [f for f in fold_rows if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame(columns=[*IDENTITY_COLUMNS, PREDICTION_COLUMN,
                                     "actual_points", "residual"])
    panel = pd.concat(frames, ignore_index=True)

    duplicated = panel.duplicated(subset=["player_id", "season", "week"], keep=False)
    if duplicated.any():
        # Two folds predicting the same player-week means overlapping test
        # seasons: the panel would double-count those rows and silently
        # reweight every aggregate computed from it.
        offenders = panel.loc[duplicated, ["player_id", "season", "week"]].head(5)
        raise OOFLeakageError(
            f"{int(duplicated.sum())} player-weeks appear in more than one "
            f"fold; test seasons must not overlap. First few:\n{offenders}")

    return add_experience_segments(panel)


def write_panel(panel: pd.DataFrame, path: Path) -> Path:
    """Persist the panel, atomically, to a fixed path.

    Kept for callers that only need "the latest panel" (there is exactly one
    on disk, previous runs are gone). For anything that compares two runs,
    use `write_run_panel` instead -- this function cannot support that, by
    construction.
    """
    from src.utils.atomic_io import atomic_write_parquet
    return atomic_write_parquet(panel, path)


RUN_PANEL_KEEP = 5


def write_run_panel(
    panel: pd.DataFrame,
    panel_dir: Path,
    *,
    coverage: Optional[pd.DataFrame] = None,
    label: str = "default",
    keep: int = RUN_PANEL_KEEP,
) -> dict:
    """Persist an immutable, provenance-stamped copy, plus a convenience pointer.

    A panel written only to a fixed filename cannot answer "which run
    produced this" and a second run silently destroys the first -- exactly
    the failure commit ca4b5e1 fixed for model artifacts, one commit before
    this module was written without the same discipline. This writes:

      data/experiments/oof_panels/<run_id>/panel.parquet     (immutable)
      data/experiments/oof_panels/<run_id>/manifest.json      (provenance)
      data/experiments/oof_panels/<run_id>/coverage.json      (if provided)
      data/experiments/<OOF_PANEL_FILENAME>                   (latest pointer, overwritten)

    The flat pointer is kept for convenience only -- anything that needs to
    compare two runs (scripts/compare_oof_panels.py) must use the per-run
    directory, never the pointer, since the pointer is by definition
    whichever run wrote last.

    Retention defaults to 5 run directories (oldest pruned), matching
    src/utils/model_rollback.py's pattern -- generous relative to the ~500MB
    per model snapshot there, since a panel is typically single-digit MB.
    """
    from src.utils.atomic_io import atomic_write_json, atomic_write_parquet
    from src.models.position_models import _git_commit

    panel_dir = Path(panel_dir)
    root = panel_dir / "oof_panels"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    commit = _git_commit() or "nogit"
    run_id = f"{stamp}_{commit}_{label}"
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    panel_path = atomic_write_parquet(panel, run_dir / "panel.parquet")

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "label": label,
        "n_rows": int(len(panel)),
        "n_players": int(panel["player_id"].nunique()) if "player_id" in panel.columns and not panel.empty else 0,
        "seasons": sorted(int(s) for s in panel["season"].unique()) if "season" in panel.columns and not panel.empty else [],
        "positions": sorted(str(p) for p in panel["position"].unique()) if "position" in panel.columns and not panel.empty else [],
    }
    manifest_path = atomic_write_json(manifest, run_dir / "manifest.json")

    coverage_path = None
    if coverage is not None and not coverage.empty:
        coverage_path = atomic_write_json(coverage.to_dict(orient="records"), run_dir / "coverage.json")

    latest_path = atomic_write_parquet(panel, panel_dir / OOF_PANEL_FILENAME)

    _prune_old_run_panels(root, keep=keep)

    return {
        "run_id": run_id, "run_dir": run_dir, "panel_path": panel_path,
        "manifest_path": manifest_path, "coverage_path": coverage_path,
        "latest_path": latest_path,
    }


def _prune_old_run_panels(root: Path, *, keep: int) -> None:
    if not root.is_dir():
        return
    import shutil
    runs = sorted(d for d in root.iterdir() if d.is_dir())
    for stale in runs[:-keep] if keep > 0 else runs:
        shutil.rmtree(stale, ignore_errors=True)


def cluster_bootstrap_ci(
    values: np.ndarray,
    clusters: np.ndarray,
    *,
    statistic,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple:
    """Bootstrap a CI by resampling CLUSTERS, not rows.

    Rows here are not independent draws -- many rows per player, several per
    same-game team-week -- so a per-row bootstrap understates the standard
    error. Resampling whole clusters (with replacement) and recomputing the
    statistic on each resample is the standard correction; it is approximate
    (a single clustering dimension, not the fully crossed player x game
    structure) but is a large improvement over ignoring clustering entirely.
    """
    unique = np.unique(clusters)
    if len(unique) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    by_cluster = {c: values[clusters == c] for c in unique}
    stats = np.empty(n_boot)
    for i in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        pooled = np.concatenate([by_cluster[c] for c in sampled])
        stats[i] = statistic(pooled)
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def segment_report(
    panel: pd.DataFrame,
    *,
    by: Sequence[str] = ("position",),
    cluster: Optional[str] = None,
    n_boot: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """MAE / RMSE / bias / n by segment, with the context needed to trust it.

    Bias (mean signed residual) is reported alongside the error magnitudes
    because a change can leave MAE flat while shifting the whole
    distribution -- which is exactly what aggregate reporting hides.

    `zero_rate` and `mae_positive` (MAE conditional on a nonzero actual) are
    reported because raw MAE on a heavily zero-inflated, right-skewed outcome
    (TE is ~39% near-zero, WR ~26%, skew up to ~1.9) can move because the
    zero fraction differs between segments, not because model skill does --
    exactly the risk when comparing e.g. cold-start (roster-churn-heavy,
    different zero rate) against returning players.

    `spearman` (rank correlation, via the existing
    src.evaluation.metrics.spearman_rank_correlation) is reported because it
    is closer to the decision this model is used for -- who to start -- than
    a magnitude-of-error metric is.

    Pass `cluster` (a column present in `panel`, e.g. "player_id") to add a
    cluster-bootstrap 95% CI on MAE per cell. Rows are not independent draws;
    a naive per-row CI understates uncertainty and can make noise look like a
    real segment difference. No CI is computed when `cluster` is omitted --
    the columns are absent rather than silently wrong.
    """
    if panel.empty:
        columns = [*by, "n", "mae", "rmse", "bias", "zero_rate", "mae_positive", "spearman"]
        if cluster:
            columns += ["mae_ci_lo", "mae_ci_hi"]
        return pd.DataFrame(columns=columns)

    from src.evaluation.metrics import spearman_rank_correlation

    def _spearman(frame: pd.DataFrame) -> float:
        if len(frame) < 2:
            return float("nan")
        return float(spearman_rank_correlation(
            frame["actual_points"].to_numpy(), frame[PREDICTION_COLUMN].to_numpy()))

    def _mae_positive(frame: pd.DataFrame) -> float:
        positive = frame.loc[frame["actual_points"] > ZERO_FLOOR, "residual"]
        return float(np.abs(positive).mean()) if len(positive) else float("nan")

    rows = []
    for key, group in panel.groupby(list(by), dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        residual = group["residual"].to_numpy()
        entry = dict(zip(by, key))
        entry.update({
            "n": int(len(group)),
            "mae": float(np.abs(residual).mean()),
            "rmse": float(np.sqrt(np.mean(np.square(residual)))),
            "bias": float(np.mean(residual)),
            "zero_rate": float((group["actual_points"] <= ZERO_FLOOR).mean()),
            "mae_positive": _mae_positive(group),
            "spearman": _spearman(group),
        })
        if cluster:
            lo, hi = cluster_bootstrap_ci(
                np.abs(residual), group[cluster].to_numpy(),
                statistic=np.mean, n_boot=n_boot, seed=seed)
            entry["mae_ci_lo"], entry["mae_ci_hi"] = lo, hi
        rows.append(entry)

    report = pd.DataFrame(rows)
    return report.sort_values(list(by)).reset_index(drop=True)
