"""Phase 3: leakage-safe test of Phase 2 usage predictions in weekly PPR.

This is an experiment harness, not serving code.  It compares Phase 7's
FINAL_CONFIG architecture on an *identical, Phase-2-OOF-matched population*:

    baseline:       existing causal PPR features
    p_only:         baseline + predicted participation probability
    p_plus_expected: baseline + probability + expected snap share

The harness deliberately does not alter ``CAUSAL_FEATURES`` or FINAL_CONFIG.
It cannot accidentally promote a result into production.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from src.models.single_week_ppr.evaluate import compute_metrics

KEY = ["player_id", "season", "week"]
PROVENANCE = ["phase2_test_season", "phase2_train_max_season"]
P2_MODEL = "hist_gbm"
ARMS = {
    "baseline": [],
    "p_only": ["p2_participation_probability"],
    "p_plus_expected": ["p2_participation_probability", "p2_expected_snap_share"],
}


def validate_phase2_oof(oof: pd.DataFrame, model: str = P2_MODEL) -> pd.DataFrame:
    """Validate OOF identity, temporal provenance and probability contract."""
    required = set(KEY + PROVENANCE + ["model", "participation_probability", "expected_snap_share"])
    missing = required - set(oof.columns)
    if missing:
        raise ValueError(f"Phase 2 OOF file missing required columns: {sorted(missing)}")
    out = oof[oof["model"].eq(model)].copy()
    if out.empty:
        raise ValueError(f"Phase 2 OOF file has no rows for selected model {model!r}")
    if out.duplicated(KEY).any():
        raise ValueError("Phase 2 OOF has duplicate player/season/week rows for selected model")
    numeric = out[["participation_probability", "expected_snap_share", *PROVENANCE]].apply(
        pd.to_numeric, errors="coerce"
    )
    if numeric.isna().any().any():
        raise ValueError("Phase 2 OOF has null/non-numeric prediction or provenance values")
    for col in ("participation_probability", "expected_snap_share"):
        if not numeric[col].between(0, 1).all():
            raise ValueError(f"Phase 2 OOF {col} lies outside [0, 1]")
    if not numeric["phase2_train_max_season"].lt(out["season"]).all():
        raise ValueError("Phase 2 OOF row was trained on its own/future season")
    if not numeric["phase2_test_season"].eq(out["season"]).all():
        raise ValueError("Phase 2 OOF test-season provenance does not match row season")
    out["p2_participation_probability"] = numeric["participation_probability"]
    out["p2_expected_snap_share"] = numeric["expected_snap_share"]
    return out[KEY + ["p2_participation_probability", "p2_expected_snap_share", *PROVENANCE]]


def validate_phase2_manifest(oof_path: Path) -> dict:
    """Require canonical Phase 2 output, never the raw-snap smoke artifact."""
    path = Path(oof_path).parent / "phase2_manifest.json"
    if not path.exists():
        raise ValueError(f"missing Phase 2 manifest beside OOF file: {path}")
    try:
        manifest = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid Phase 2 manifest: {path}") from exc
    if manifest.get("system") != "participation_opportunity" or manifest.get("phase") != 2:
        raise ValueError("Phase 2 manifest identifies the wrong system or phase")
    if manifest.get("panel_source") != "canonical_player_weeks":
        raise ValueError("Phase 3 requires canonical_player_weeks, not a raw-snap smoke artifact")
    if manifest.get("oof_file") != Path(oof_path).name:
        raise ValueError("Phase 2 manifest OOF filename does not match requested artifact")
    if set(PROVENANCE) - set(manifest.get("oof_provenance_columns", [])):
        raise ValueError("Phase 2 manifest does not declare required OOF provenance")
    return manifest


def attach_phase2_oof(ppr: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    """Inner-join one PPR population to valid Phase 2 OOF predictions.

    An inner join is intentional.  Phase 2 begins in the snap-observed era;
    allowing a missing-value imputer to keep older PPR rows would make the two
    arms train on different information regimes.  The caller must use this
    exact matched frame for *both* arms.
    """
    if ppr.duplicated(KEY).any():
        raise ValueError("PPR frame has duplicate player/season/week rows")
    matched = ppr.merge(oof, on=KEY, how="inner", validate="one_to_one")
    if matched.empty:
        raise ValueError("no PPR rows match the Phase 2 OOF population")
    if not matched["phase2_train_max_season"].lt(matched["season"]).all():
        raise ValueError("joined Phase 2 prediction violates temporal provenance")
    return matched


def paired_bootstrap_mae_delta(rows: pd.DataFrame, n_bootstrap: int = 2000, seed: int = 42) -> dict:
    """Player-clustered CI for augmented minus baseline absolute error.

    Weekly rows for one player are correlated; resampling individual rows would
    overstate precision.  A player is therefore the bootstrap unit.
    """
    required = {"actual", "baseline_prediction", "augmented_prediction"}
    if not required.issubset(rows):
        raise ValueError(f"row-level comparison missing: {sorted(required - set(rows))}")
    # Preserve player identity so player-level, rather than row-level,
    # resampling is actually used when the caller provides it.
    columns = list(required)
    if "player_id" in rows.columns:
        columns.append("player_id")
    clean = rows[columns].dropna()
    if len(clean) < 2:
        return {"n": len(clean), "mae_delta": np.nan, "ci95_low": np.nan, "ci95_high": np.nan}
    delta = np.abs(clean.augmented_prediction - clean.actual).to_numpy() - np.abs(clean.baseline_prediction - clean.actual).to_numpy()
    clusters = clean["player_id"].to_numpy() if "player_id" in clean else np.arange(len(clean))
    unique, inverse = np.unique(clusters, return_inverse=True)
    rng = np.random.default_rng(seed)
    sampled_clusters = rng.integers(0, len(unique), size=(n_bootstrap, len(unique)))
    means = np.empty(n_bootstrap)
    for i, draw in enumerate(sampled_clusters):
        row_mask = np.isin(inverse, draw)
        # Repeated sampled players must retain their multiplicity.
        counts = np.bincount(draw, minlength=len(unique))[inverse]
        means[i] = np.average(delta[row_mask], weights=counts[row_mask])
    return {"n": int(len(delta)), "mae_delta": float(delta.mean()),
            "ci95_low": float(np.quantile(means, .025)), "ci95_high": float(np.quantile(means, .975))}


def _feature_matrix(df: pd.DataFrame, features: Sequence[str]) -> tuple[pd.DataFrame, pd.Series]:
    usable = df.dropna(subset=["fantasy_points", *features]).copy()
    return usable[list(features)], usable["fantasy_points"]


def run_participation_integration(
    oof_path: Path,
    positions: Sequence[str] | None = None,
    seasons: Sequence[int] = (2023, 2024, 2025),
    output_dir: Path = Path("data/experiments/phase3_participation"),
    n_bootstrap: int = 2000,
    phase2_model: str = P2_MODEL,
    fold_loader: Callable | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run matched-population Phase 7 vs Phase 7+Phase-2 OOF comparison.

    ``fold_loader`` is injectable only for tests.  Its normal implementation
    is the existing Phase 7 ``run_fold``; no second feature-engineering path is
    introduced here.
    """
    from config.settings import CAUSAL_FEATURES, POSITIONS
    from src.models.single_week_ppr.evaluate import _architectures_for_fold, run_fold
    from src.models.single_week_ppr.final_config import FINAL_CONFIG
    from src.models.single_week_ppr.windows import compute_recency_weights, window_to_season_list
    from src.utils.database import DatabaseManager
    from src.utils.leakage import filter_feature_columns

    output_dir.mkdir(parents=True, exist_ok=True)
    phase2_manifest = validate_phase2_manifest(oof_path)
    oof = validate_phase2_oof(pd.read_csv(oof_path), model=phase2_model)
    positions = list(positions) if positions is not None else list(POSITIONS)
    fold_loader = fold_loader or run_fold
    row_frames, summary_rows, failures = [], [], []

    for position in positions:
        cfg = FINAL_CONFIG[position]
        available = sorted(DatabaseManager().get_all_players_for_training(position=position)["season"].dropna().unique())
        for season in seasons:
            train_seasons = window_to_season_list(cfg["window"], season, available)
            if not train_seasons:
                failures.append({"position": position, "season": season, "error": "no training seasons"})
                continue
            try:
                train_df, test_df, _, _ = fold_loader(position, season, False, train_seasons_override=train_seasons)
                train = attach_phase2_oof(train_df[train_df.position.eq(position)].copy(), oof)
                test = attach_phase2_oof(test_df[test_df.position.eq(position)].copy(), oof)
                base_features = filter_feature_columns(CAUSAL_FEATURES.get(position, []))
                base_features = [c for c in base_features if c in train and c in test]
                if not base_features:
                    raise ValueError("no shared Phase 7 causal features")
                weights = compute_recency_weights(train["season"], cfg["weighting"])
                for arm, additions in ARMS.items():
                    features = base_features + additions
                    X_train, y_train = _feature_matrix(train, features)
                    X_test, y_test = _feature_matrix(test, features)
                    # Both arms must score the same target player-weeks.
                    if arm == "baseline":
                        shared_test_keys = test.loc[X_test.index, KEY].copy()
                    else:
                        keys = test.loc[X_test.index, KEY]
                        keep = pd.MultiIndex.from_frame(keys).isin(pd.MultiIndex.from_frame(shared_test_keys))
                        keep_index = keys.index[keep]
                        X_test, y_test = X_test.loc[keep_index], y_test.loc[keep_index]
                    if len(X_train) < 20 or len(X_test) < 20:
                        raise ValueError(f"insufficient matched rows for {arm}: train={len(X_train)}, test={len(X_test)}")
                    model = _architectures_for_fold()[cfg["architecture"]]
                    model.fit(X_train, y_train, sample_weight=weights.reindex(X_train.index))
                    pred = pd.Series(model.predict(X_test), index=X_test.index)
                    metrics = compute_metrics(y_test, pred)
                    summary_rows.append({"position": position, "season": season, "arm": arm,
                                         "n_train_matched": len(X_train), "n_test_matched": len(X_test), **metrics})
                    frame = test.loc[X_test.index, KEY].copy()
                    frame["position"], frame["arm"], frame["actual"], frame["prediction"] = position, arm, y_test, pred
                    row_frames.append(frame)
            except Exception as exc:
                failures.append({"position": position, "season": season, "error_type": type(exc).__name__, "error": str(exc)})

    rows = pd.concat(row_frames, ignore_index=True) if row_frames else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    paired = []
    if not rows.empty:
        baseline = rows[rows.arm.eq("baseline")].rename(columns={"prediction": "baseline_prediction"})
        for arm in ("p_only", "p_plus_expected"):
            augmented = rows[rows.arm.eq(arm)].rename(columns={"prediction": "augmented_prediction"})
            joined = baseline.merge(
                augmented[KEY + ["position", "augmented_prediction"]],
                on=KEY + ["position"], how="inner", validate="one_to_one",
            )
            for (position, season), group in joined.groupby(["position", "season"]):
                result = paired_bootstrap_mae_delta(group, n_bootstrap=n_bootstrap)
                paired.append({"position": position, "season": season, "arm": arm, **result})
            aggregate = paired_bootstrap_mae_delta(joined, n_bootstrap=n_bootstrap)
            paired.append({"position": "ALL", "season": "ALL", "arm": arm, **aggregate})
    report = {"system": "participation_opportunity", "phase": 3,
              "phase2_manifest": phase2_manifest, "selected_phase2_model": phase2_model,
              "arms": ARMS, "failures": failures, "all_requested_folds_completed": not failures,
              "selection_rule": "No production change. A future adoption decision requires complete folds, a negative aggregate paired-bootstrap CI, and review by position.",
              "paired_bootstrap": paired}
    rows.to_csv(output_dir / "row_predictions.csv", index=False)
    summary.to_csv(output_dir / "fold_metrics.csv", index=False)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return rows, summary, report
