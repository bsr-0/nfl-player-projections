"""
Explainability: SHAP values, top-10 features per position, optional partial dependence.

Per requirements: SHAP for feature importance, top-10 per position, partial dependence for key features.
"""
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import MODELS_DIR, POSITIONS

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


def get_top10_feature_importance_per_position(
    position_models: Dict[str, Any],
    output_path: Optional[Path] = None,
) -> Dict[str, List[Dict[str, float]]]:
    """
    Extract top-10 feature importance per position from trained models.
    position_models: dict position -> MultiWeekModel or PositionModel.
    Returns and optionally saves to JSON.
    """
    top10 = {}
    for pos in POSITIONS:
        model = position_models.get(pos)
        if model is None:
            continue
        base = getattr(model, "models", None)
        if base is None:
            base = {1: model}
        pos_model = base.get(1) or (list(base.values())[0] if base else None)
        if pos_model is None or not getattr(pos_model, "is_fitted", False):
            continue
        try:
            imp = pos_model.get_feature_importance()
            if imp is None or imp.empty:
                continue
            combined = imp.get("combined", imp.get("feature", pd.Series()))
            if combined is None or not len(combined):
                continue
            if isinstance(imp, pd.DataFrame) and "feature" in imp.columns:
                names = imp["feature"].tolist()
                values = imp["combined"].values if "combined" in imp.columns else imp.iloc[:, 1].values
            else:
                names = list(imp.index)
                values = imp.values
            order = np.argsort(values)[::-1][:10]
            top10[pos] = [{"feature": names[i], "importance": float(values[i])} for i in order]
        except Exception:
            continue
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(top10, f, indent=2)
    return top10


def explain_with_shap(
    model: Any,
    X: pd.DataFrame,
    feature_names: List[str],
    n_samples: int = 100,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Compute SHAP values for tree-based model (RF/XGB). Returns dict with
    'values' (SHAP array) and 'base_value' when available.
    """
    if not HAS_SHAP:
        return None
    X = X[feature_names].fillna(0)
    if len(X) > n_samples:
        X = X.sample(n=n_samples, random_state=42)
    try:
        tree_models = []
        if hasattr(model, "models"):
            for m in model.models.values():
                if hasattr(m, "predict") and hasattr(m, "feature_importances_"):
                    tree_models.append(m)
        elif hasattr(model, "feature_importances_"):
            tree_models = [model]
        if not tree_models:
            return None
        explainer = shap.TreeExplainer(tree_models[0], X)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        return {"values": shap_values, "base_value": explainer.expected_value}
    except Exception:
        return None


def partial_dependence_1d(
    model: Any,
    X: pd.DataFrame,
    feature: str,
    grid_resolution: int = 20,
) -> Optional[tuple]:
    """One-dimensional partial dependence: (grid_values, predictions)."""
    if feature not in X.columns:
        return None
    grid = np.linspace(X[feature].min(), X[feature].max(), grid_resolution)
    preds = []
    for v in grid:
        Xv = X.copy()
        Xv[feature] = v
        try:
            p = model.predict(Xv)
            preds.append(np.mean(p))
        except Exception:
            preds.append(np.nan)
    return grid, np.array(preds)


def partial_dependence_plots(
    model: Any,
    X: pd.DataFrame,
    features: Optional[List[str]] = None,
    top_n: int = 5,
    grid_resolution: int = 20,
    output_path: Optional[Path] = None,
) -> Dict[str, Dict[str, list]]:
    """
    Generate partial dependence data for top-N features per requirements Section VI.B.
    
    If features is None, uses top_n most important features from model.
    Returns dict feature -> {"grid": [...], "predictions": [...]}.
    Optionally saves to JSON for frontend consumption.
    """
    if features is None:
        # Try to get top features from model importance
        try:
            if hasattr(model, "get_feature_importance"):
                imp = model.get_feature_importance()
                if isinstance(imp, pd.DataFrame) and "feature" in imp.columns:
                    features = imp.nlargest(top_n, "combined")["feature"].tolist()
                elif isinstance(imp, pd.DataFrame):
                    features = imp.nlargest(top_n, imp.columns[-1]).index.tolist()
            if not features and hasattr(model, "feature_names"):
                features = list(model.feature_names)[:top_n]
        except Exception:
            pass
    if not features:
        return {}
    
    results = {}
    for feat in features:
        pd_result = partial_dependence_1d(model, X, feat, grid_resolution)
        if pd_result is not None:
            grid, preds = pd_result
            results[feat] = {
                "grid": [float(v) for v in grid],
                "predictions": [float(v) for v in preds],
            }
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
    
    return results
