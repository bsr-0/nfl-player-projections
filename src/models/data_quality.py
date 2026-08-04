"""Data quality check and diagnostics functions for NFL model training."""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _report_missingness(df: pd.DataFrame, label: str, threshold: float = 0.05) -> Dict[str, float]:
    """Report feature missingness and return columns above threshold."""
    if df.empty:
        print(f"  {label}: empty dataset for missingness check")
        return {}
    miss = df.isna().mean()
    high = miss[miss > threshold].sort_values(ascending=False)
    if len(high) == 0:
        print(f"  {label}: no columns above {threshold:.0%} missingness")
        return {}
    print(f"  {label}: {len(high)} columns above {threshold:.0%} missingness")
    for col, pct in high.head(15).items():
        print(f"    - {col}: {pct:.1%}")
    if len(high) > 15:
        print(f"    ... {len(high) - 15} more")
    return high.to_dict()


def _validate_critical_missingness(df: pd.DataFrame, label: str, threshold: float = 0.05) -> None:
    """
    Validate critical columns after preprocessing.
    Raise on severe quality issues that can invalidate training labels/features.
    """
    critical = [c for c in ["player_id", "position", "season", "week", "fantasy_points", "utilization_score"] if c in df.columns]
    bad = {}
    for col in critical:
        pct = float(df[col].isna().mean())
        if pct > threshold:
            bad[col] = pct
    if bad:
        details = ", ".join(f"{k}={v:.1%}" for k, v in sorted(bad.items()))
        raise ValueError(f"{label}: critical missingness exceeds {threshold:.0%}: {details}")


def _report_outliers_3sigma(df: pd.DataFrame, label: str, cols: list) -> Dict[str, Dict[str, float]]:
    """Diagnostics only: report >3 std outliers without dropping data."""
    if df.empty:
        return {}
    print(f"  {label}: >3σ outlier diagnostics")
    out = {}
    for col in cols:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        s = s[np.isfinite(s)]
        if len(s) < 30:
            continue
        mu, sigma = float(s.mean()), float(s.std(ddof=0))
        if sigma <= 0:
            continue
        n_out = int(((s - mu).abs() > 3 * sigma).sum())
        pct = 100.0 * n_out / max(len(s), 1)
        print(f"    - {col}: {n_out}/{len(s)} ({pct:.2f}%)")
        out[col] = {
            "n_outliers": n_out,
            "n_samples": int(len(s)),
            "pct_outliers": round(pct, 3),
        }
    return out


def _write_json_artifact(path: Path, payload: Dict[str, Any], label: str) -> None:
    """Best-effort JSON artifact writer for training diagnostics."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"  Wrote {label}: {path.name}")
    except Exception as e:
        print(f"  {label} write skipped: {e}")


def _check_distribution_shift(train_df: pd.DataFrame, test_df: pd.DataFrame,
                              max_features: int = 50) -> Dict[str, Any]:
    """Detect train/test distribution shift via adversarial validation.

    Trains a classifier to distinguish train from test rows.  If the AUC is
    much above 0.5, train and test distributions differ significantly, which
    means model performance on test may be unreliable.

    Returns a dict with the adversarial AUC, per-feature KS statistics, and
    an overall warning flag.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import roc_auc_score

    exclude = {"player_id", "name", "position", "team", "season", "week",
               "fantasy_points", "target", "opponent", "home_away",
               "created_at", "updated_at", "id", "birth_date", "college",
               "game_id", "game_time", "actual_for_backtest",
               "predicted_points", "predicted_utilization"}

    shared_cols = sorted(set(train_df.columns) & set(test_df.columns) - exclude)
    numeric_cols = [c for c in shared_cols
                    if train_df[c].dtype in ("int64", "float64", "int32", "float32")
                    and not c.startswith("target_")]
    if len(numeric_cols) < 5:
        return {"adversarial_auc": None, "warning": "too_few_features"}

    # Sample to keep fast
    n_sample = min(5000, len(train_df), len(test_df) * 3)
    tr = train_df[numeric_cols].sample(n=min(n_sample, len(train_df)), random_state=42).fillna(0)
    te = test_df[numeric_cols].sample(n=min(n_sample, len(test_df)), random_state=42).fillna(0)
    X = pd.concat([tr, te], ignore_index=True)
    y = np.array([0] * len(tr) + [1] * len(te))

    # Use only top max_features by variance to keep fast
    if X.shape[1] > max_features:
        variances = X.var()
        top_cols = variances.nlargest(max_features).index.tolist()
        X = X[top_cols]

    clf = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
    try:
        scores = cross_val_score(clf, X, y, cv=3, scoring="roc_auc")
        auc = float(np.mean(scores))
    except Exception:
        auc = None

    # Feature importance from adversarial classifier for diagnostics
    top_discriminative = []
    if auc is not None and auc > 0.55:
        try:
            clf.fit(X, y)
            importances = pd.Series(clf.feature_importances_, index=X.columns)
            top_discriminative = importances.nlargest(10)
            print(f"  Top discriminative features: {list(top_discriminative.index)}")
        except Exception:
            pass

    # Per-feature KS test for top shifted features
    ks_results = []
    try:
        from scipy.stats import ks_2samp
        for col in numeric_cols[:max_features]:
            tr_vals = train_df[col].dropna().values
            te_vals = test_df[col].dropna().values
            if len(tr_vals) < 10 or len(te_vals) < 10:
                continue
            stat, pval = ks_2samp(tr_vals, te_vals)
            if pval < 0.01:
                ks_results.append({"feature": col, "ks_stat": round(stat, 3), "p_value": round(pval, 6)})
        ks_results.sort(key=lambda x: -x["ks_stat"])
    except ImportError:
        pass

    result = {
        "adversarial_auc": round(auc, 4) if auc is not None else None,
        "shift_detected": auc is not None and auc > 0.85,
        "top_shifted_features": ks_results[:10],
        "top_discriminative_features": (
            [{"feature": f, "importance": round(float(v), 4)}
             for f, v in top_discriminative.items()]
            if len(top_discriminative) > 0 else []
        ),
    }

    if result["shift_detected"]:
        print(f"\n  *** WARNING: Train/test distribution shift detected (adversarial AUC={auc:.3f}) ***")
        print(f"  Top shifted features: {[f['feature'] for f in ks_results[:5]]}")
        print("  This may indicate temporal concept drift or feature leakage.")
    else:
        print(f"  Distribution shift check: adversarial AUC={auc:.3f} (OK)" if auc else
              "  Distribution shift check: skipped (insufficient features)")

    return result


def _report_train_serve_feature_parity(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Check train/test feature schema parity before model fitting."""
    excluded_prefix = ("target_",)
    excluded_cols = {"fantasy_points", "predicted_points", "predicted_utilization"}
    train_feats = {c for c in train_df.columns if c not in excluded_cols and not c.startswith(excluded_prefix)}
    test_feats = {c for c in test_df.columns if c not in excluded_cols and not c.startswith(excluded_prefix)}
    missing_in_test = sorted(train_feats - test_feats)
    unseen_in_test = sorted(test_feats - train_feats)
    print("  Train/serve feature parity check:")
    print(f"    - train feature count: {len(train_feats)}")
    print(f"    - test feature count: {len(test_feats)}")
    print(f"    - train-only features: {len(missing_in_test)}")
    print(f"    - test-only features: {len(unseen_in_test)}")
    if missing_in_test:
        print("    - sample train-only:", missing_in_test[:8])
    if unseen_in_test:
        print("    - sample test-only:", unseen_in_test[:8])
