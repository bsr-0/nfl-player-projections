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
    Compute weighted-average SHAP values across all tree-based models in the
    ensemble (RF + XGB), using ensemble weights when available.

    Returns dict with 'values' (SHAP array) and 'base_value'.
    """
    if not HAS_SHAP:
        return None
    Xf = X[feature_names].fillna(0)
    if len(Xf) > n_samples:
        Xf = Xf.sample(n=n_samples, random_state=42)
    try:
        tree_models = []
        model_keys = []
        if hasattr(model, "models"):
            for k, m in model.models.items():
                if hasattr(m, "predict") and hasattr(m, "feature_importances_"):
                    tree_models.append(m)
                    model_keys.append(k)
        elif hasattr(model, "feature_importances_"):
            tree_models = [model]
            model_keys = [None]
        if not tree_models:
            return None

        # Compute SHAP per tree model and produce weighted average
        ensemble_weights = getattr(model, "ensemble_weights", {})
        all_shap = []
        all_base = []
        weight_sum = 0.0
        for i, tm in enumerate(tree_models):
            explainer = shap.TreeExplainer(tm, Xf)
            sv = explainer.shap_values(Xf)
            if isinstance(sv, list):
                sv = sv[0]
            bv = explainer.expected_value
            if isinstance(bv, (list, np.ndarray)):
                bv = bv[0]
            w = ensemble_weights.get(model_keys[i], 1.0 / len(tree_models))
            all_shap.append(sv * w)
            all_base.append(bv * w)
            weight_sum += w

        if weight_sum > 0 and weight_sum != 1.0:
            combined_shap = sum(all_shap) / weight_sum
            combined_base = sum(all_base) / weight_sum
        else:
            combined_shap = sum(all_shap)
            combined_base = sum(all_base)
        return {"values": combined_shap, "base_value": combined_base}
    except Exception:
        return None


def explain_individual_prediction(
    model: Any,
    X_single: pd.DataFrame,
    feature_names: List[str],
    top_n: int = 10,
) -> Optional[Dict[str, Any]]:
    """
    Explain a single player's prediction using SHAP values.

    Per requirements Section VI.B: individual prediction explanations.

    Args:
        model: Trained ensemble model (PositionModel or similar).
        X_single: Single-row DataFrame with the player's features.
        feature_names: List of feature column names.
        top_n: Number of top contributing features to return.

    Returns:
        Dict with 'prediction', 'base_value', 'top_positive' (features
        pushing prediction up), 'top_negative' (pushing down), and
        'all_contributions' (sorted list of {feature, contribution}).
    """
    if not HAS_SHAP:
        return None
    Xf = X_single[feature_names].fillna(0)
    try:
        # Find the primary tree model for SHAP
        tree_models = []
        if hasattr(model, "models"):
            for m in model.models.values():
                if hasattr(m, "predict") and hasattr(m, "feature_importances_"):
                    tree_models.append(m)
        elif hasattr(model, "feature_importances_"):
            tree_models = [model]
        if not tree_models:
            return None

        # Use the first tree for individual explanation
        explainer = shap.TreeExplainer(tree_models[0])
        sv = explainer.shap_values(Xf)
        if isinstance(sv, list):
            sv = sv[0]
        base = explainer.expected_value
        if isinstance(base, (list, np.ndarray)):
            base = float(base[0])

        contribs = sv[0] if sv.ndim > 1 else sv
        # Sort contributions
        order = np.argsort(np.abs(contribs))[::-1]
        all_contribs = [
            {"feature": feature_names[i], "contribution": float(contribs[i]),
             "value": float(Xf.iloc[0, i])}
            for i in order
        ]
        top_pos = [c for c in all_contribs if c["contribution"] > 0][:top_n]
        top_neg = [c for c in all_contribs if c["contribution"] < 0][:top_n]

        prediction = float(base + np.sum(contribs))
        return {
            "prediction": prediction,
            "base_value": float(base),
            "top_positive": top_pos,
            "top_negative": top_neg,
            "all_contributions": all_contribs[:top_n * 2],
        }
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
