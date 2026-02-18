"""Dimensionality reduction and feature selection for NFL prediction models."""
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict
from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    SelectKBest, f_regression, mutual_info_regression,
    RFE, VarianceThreshold
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import MODELS_DIR


class DimensionalityReducer:
    """
    Handles dimensionality reduction and feature selection.
    
    Combines multiple techniques:
    1. Variance threshold (remove low-variance features)
    2. Correlation filtering (remove highly correlated features)
    3. Recursive Feature Elimination (RFE)
    4. PCA for remaining correlated features
    5. Feature importance from tree models
    """
    
    def __init__(self, 
                 variance_threshold: float = 0.01,
                 correlation_threshold: float = 0.95,
                 n_features_to_select: int = None,
                 pca_variance_ratio: float = 0.95):
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.n_features_to_select = n_features_to_select
        self.pca_variance_ratio = pca_variance_ratio
        
        # Fitted components
        self.scaler = StandardScaler()
        self.variance_selector = None
        self.selected_features = []
        self.removed_correlated = []
        self.pca = None
        self.pca_features = []
        self.rfe_selector = None
        self.feature_importances = {}
        
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'DimensionalityReducer':
        """
        Fit the dimensionality reduction pipeline.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            self
        """
        X = X.copy()

        # Handle missing values - store medians for use in transform()
        self.train_medians_ = X.median()
        X = X.fillna(self.train_medians_)

        # Replace infinities
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Step 1: Remove low variance features
        X, low_var_removed = self._remove_low_variance(X)
        print(f"Removed {len(low_var_removed)} low-variance features")
        
        # Step 2: Remove highly correlated features
        X, corr_removed = self._remove_correlated(X)
        self.removed_correlated = corr_removed
        print(f"Removed {len(corr_removed)} highly correlated features")
        
        # Step 3: Scale features
        self.scaler.fit(X)
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Step 4: Feature importance ranking
        self.feature_importances = self._calculate_feature_importance(X_scaled, y)
        
        # Step 5: Select top features using RFE if n_features specified
        if self.n_features_to_select and self.n_features_to_select < len(X.columns):
            X_scaled, rfe_selected = self._apply_rfe(X_scaled, y)
            self.selected_features = rfe_selected
            print(f"Selected {len(rfe_selected)} features via RFE")
        else:
            self.selected_features = list(X_scaled.columns)
        
        # Step 6: Apply PCA to remaining correlated feature groups
        self._fit_pca_on_correlated_groups(X_scaled, y)
        
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted pipeline.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Transformed DataFrame
        """
        if not self.is_fitted:
            raise ValueError("DimensionalityReducer must be fitted before transform")
        
        X = X.copy()

        # Handle missing values using training medians (no test-data leakage)
        if hasattr(self, 'train_medians_'):
            X = X.fillna(self.train_medians_)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Select only features that were kept
        available_features = [f for f in self.selected_features if f in X.columns]
        X = X[available_features]
        
        # Scale
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Apply PCA if fitted
        if self.pca is not None and self.pca_features:
            pca_cols = [c for c in self.pca_features if c in X_scaled.columns]
            if pca_cols:
                pca_transformed = self.pca.transform(X_scaled[pca_cols])
                pca_df = pd.DataFrame(
                    pca_transformed,
                    columns=[f"pca_{i}" for i in range(pca_transformed.shape[1])],
                    index=X_scaled.index
                )
                
                # Replace PCA features with components
                X_scaled = X_scaled.drop(columns=pca_cols)
                X_scaled = pd.concat([X_scaled, pca_df], axis=1)
        
        return X_scaled
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X)
    
    def _remove_low_variance(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Remove features with variance below threshold."""
        self.variance_selector = VarianceThreshold(threshold=self.variance_threshold)
        
        # Fit on numeric columns only
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X_numeric = X[numeric_cols]
        
        self.variance_selector.fit(X_numeric)
        
        # Get mask of selected features
        mask = self.variance_selector.get_support()
        removed = [col for col, keep in zip(numeric_cols, mask) if not keep]
        kept = [col for col, keep in zip(numeric_cols, mask) if keep]
        
        return X[kept], removed
    
    def _remove_correlated(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Remove highly correlated features, keeping the one with higher variance."""
        corr_matrix = X.corr().abs()
        
        # Get upper triangle
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features with correlation above threshold
        to_remove = set()
        for col in upper.columns:
            correlated = upper.index[upper[col] > self.correlation_threshold].tolist()
            if correlated:
                # Keep the feature with higher variance
                variances = X[[col] + correlated].var()
                to_remove.update(variances.drop(variances.idxmax()).index.tolist())
        
        kept_cols = [c for c in X.columns if c not in to_remove]
        return X[kept_cols], list(to_remove)
    
    def _calculate_feature_importance(self, X: pd.DataFrame, 
                                       y: pd.Series) -> Dict[str, float]:
        """Calculate feature importance using multiple methods."""
        importances = {}
        
        # Method 1: Mutual Information
        mi_scores = mutual_info_regression(X, y, random_state=42)
        mi_scores = mi_scores / mi_scores.max() if mi_scores.max() > 0 else mi_scores
        
        # Method 2: Random Forest importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
        rf.fit(X, y)
        rf_scores = rf.feature_importances_
        rf_scores = rf_scores / rf_scores.max() if rf_scores.max() > 0 else rf_scores
        
        # Method 3: F-regression scores
        f_scores, _ = f_regression(X, y)
        f_scores = np.nan_to_num(f_scores)
        f_scores = f_scores / f_scores.max() if f_scores.max() > 0 else f_scores
        
        # Combine scores (weighted average)
        for i, col in enumerate(X.columns):
            importances[col] = (
                0.4 * rf_scores[i] +
                0.3 * mi_scores[i] +
                0.3 * f_scores[i]
            )
        
        return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
    
    def _apply_rfe(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, List[str]]:
        """Apply Recursive Feature Elimination."""
        estimator = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)
        
        n_features = min(self.n_features_to_select, len(X.columns))
        
        self.rfe_selector = RFE(
            estimator=estimator,
            n_features_to_select=n_features,
            step=0.1
        )
        
        self.rfe_selector.fit(X, y)
        
        selected = X.columns[self.rfe_selector.support_].tolist()
        return X[selected], selected
    
    def _fit_pca_on_correlated_groups(self, X: pd.DataFrame, y: pd.Series):
        """Apply PCA to groups of correlated features."""
        # Find groups of moderately correlated features (0.7-0.95)
        corr_matrix = X.corr().abs()
        
        # Find feature groups
        groups = []
        used = set()
        
        for col in corr_matrix.columns:
            if col in used:
                continue
            
            # Find correlated features
            correlated = corr_matrix.index[
                (corr_matrix[col] > 0.7) & (corr_matrix[col] < self.correlation_threshold)
            ].tolist()
            
            if len(correlated) >= 3:  # Only apply PCA to groups of 3+
                group = [c for c in correlated if c not in used]
                if len(group) >= 3:
                    groups.append(group)
                    used.update(group)
        
        # Apply PCA to largest group if exists
        if groups:
            largest_group = max(groups, key=len)
            self.pca_features = largest_group
            
            self.pca = PCA(n_components=self.pca_variance_ratio, random_state=42)
            self.pca.fit(X[largest_group])
            
            print(f"Applied PCA to {len(largest_group)} features, "
                  f"reduced to {self.pca.n_components_} components")
    
    def get_top_features(self, n: int = 20) -> List[Tuple[str, float]]:
        """Get top N most important features."""
        return list(self.feature_importances.items())[:n]
    
    def save(self, filepath: Path = None):
        """Save fitted reducer to disk."""
        filepath = filepath or MODELS_DIR / "dimensionality_reducer.joblib"
        joblib.dump(self, filepath)
        print(f"Saved DimensionalityReducer to {filepath}")
    
    @classmethod
    def load(cls, filepath: Path = None) -> 'DimensionalityReducer':
        """Load fitted reducer from disk."""
        filepath = filepath or MODELS_DIR / "dimensionality_reducer.joblib"
        return joblib.load(filepath)


class PositionDimensionalityReducer:
    """Position-specific dimensionality reduction."""
    
    def __init__(self, position: str, **kwargs):
        self.position = position
        self.reducer = DimensionalityReducer(**kwargs)
        
        # Position-specific feature preferences
        self.position_key_features = {
            "QB": [
                "passing_yards", "passing_tds", "interceptions", "rushing_yards",
                "completion_pct", "yards_per_attempt", "utilization_score"
            ],
            "RB": [
                "rushing_yards", "rushing_attempts", "rushing_tds", "receptions",
                "targets", "total_touches", "utilization_score", "snap_share"
            ],
            "WR": [
                "receiving_yards", "receptions", "targets", "receiving_tds",
                "catch_rate", "yards_per_target", "utilization_score"
            ],
            "TE": [
                "receiving_yards", "receptions", "targets", "receiving_tds",
                "catch_rate", "utilization_score", "snap_share"
            ],
        }
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'PositionDimensionalityReducer':
        """Fit with position-specific considerations."""
        # Ensure key features are preserved
        key_features = self.position_key_features.get(self.position, [])
        
        # Boost importance of key features by adding them to selection
        self.reducer.fit(X, y)
        
        # Ensure key features are in selected features
        for feature in key_features:
            matching = [f for f in X.columns if feature in f.lower()]
            for match in matching:
                if match not in self.reducer.selected_features:
                    self.reducer.selected_features.append(match)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features."""
        return self.reducer.transform(X)
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform."""
        self.fit(X, y)
        return self.transform(X)
    
    def save(self, filepath: Path = None):
        """Save reducer."""
        filepath = filepath or MODELS_DIR / f"dim_reducer_{self.position.lower()}.joblib"
        joblib.dump(self, filepath)
    
    @classmethod
    def load(cls, position: str, filepath: Path = None) -> 'PositionDimensionalityReducer':
        """Load reducer."""
        filepath = filepath or MODELS_DIR / f"dim_reducer_{position.lower()}.joblib"
        return joblib.load(filepath)


def compute_vif(X: pd.DataFrame) -> Dict[str, float]:
    """
    Compute Variance Inflation Factor for each feature using R² from OLS.
    VIF = 1 / (1 - R²) where R² is from regressing each feature on the others.
    VIF > 5-10 indicates concerning multicollinearity.
    """
    from sklearn.linear_model import LinearRegression
    
    X_clean = X.copy().fillna(0).replace([np.inf, -np.inf], 0)
    cols = list(X_clean.columns)
    vif_data = {}
    for col in cols:
        try:
            other = [c for c in cols if c != col]
            X_oth = X_clean[other].values
            y_col = X_clean[col].values
            lr = LinearRegression().fit(X_oth, y_col)
            r2 = lr.score(X_oth, y_col)
            vif = 1 / (1 - r2) if r2 < 1 else np.inf
            vif_data[col] = float(vif) if np.isfinite(vif) else np.inf
        except Exception:
            vif_data[col] = np.nan
    return vif_data


def prune_by_vif(X: pd.DataFrame, threshold: float = 10.0,
                 max_iterations: int = 50) -> Tuple[pd.DataFrame, List[str]]:
    """Iteratively drop the highest-VIF feature until all VIF <= threshold.

    Returns:
        Tuple of (pruned DataFrame, list of removed column names).
    """
    removed = []
    X_work = X.copy()
    for _ in range(max_iterations):
        if X_work.shape[1] <= 2:
            break
        vif = compute_vif(X_work)
        max_col = max(vif, key=lambda k: vif[k] if np.isfinite(vif[k]) else -1)
        max_val = vif[max_col]
        if not np.isfinite(max_val) or max_val <= threshold:
            break
        X_work = X_work.drop(columns=[max_col])
        removed.append(max_col)
    return X_work, removed


def adaptive_feature_count(n_samples: int, default: int = 50) -> int:
    """Compute feature count adapted to sample size.

    Returns min(default, max(20, int(sqrt(n_samples)))).
    Small positions (TE ~2000 rows) get ~45 features.
    Large positions (WR ~5000+ rows) get 50 (capped at default).
    """
    return min(default, max(20, int(np.sqrt(n_samples))))


def select_features_simple(X: pd.DataFrame, y: pd.Series,
                           n_features: int = 50,
                           correlation_threshold: float = 0.92) -> Tuple[pd.DataFrame, List[str]]:
    """
    Lightweight feature selection: remove correlated features, prune by VIF,
    then keep top N by mutual information importance.

    Fits on X, y. Returns (X with selected columns only, list of selected feature names).
    """
    from config.settings import MODEL_CONFIG
    X = X.copy().fillna(0).replace([np.inf, -np.inf], np.nan).fillna(0)

    # Remove highly correlated features
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_remove = set()
    for col in upper.columns:
        correlated = upper.index[upper[col] > correlation_threshold].tolist()
        if correlated:
            variances = X[[col] + correlated].var()
            to_remove.update(variances.drop(variances.idxmax()).index.tolist())

    kept = [c for c in X.columns if c not in to_remove]
    X_reduced = X[kept]

    # VIF pruning: iteratively remove highest-VIF features
    vif_thresh = MODEL_CONFIG.get("vif_threshold", 10.0)
    if len(kept) > 5:
        X_reduced, vif_removed = prune_by_vif(X_reduced, threshold=vif_thresh)
        kept = [c for c in kept if c not in vif_removed]

    if len(kept) <= n_features:
        return X_reduced, kept

    # Rank by mutual information
    mi_scores = mutual_info_regression(X_reduced, y, random_state=42)
    ranks = np.argsort(-mi_scores)
    selected = [kept[i] for i in ranks[:n_features]]

    return X[selected], selected


def select_features_for_position(X: pd.DataFrame, y: pd.Series, 
                                  position: str,
                                  n_features: int = 30) -> Tuple[pd.DataFrame, List[str]]:
    """
    Convenience function to select features for a position.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        position: Player position (QB, RB, WR, TE)
        n_features: Number of features to select
        
    Returns:
        Tuple of (transformed DataFrame, selected feature names)
    """
    reducer = PositionDimensionalityReducer(
        position=position,
        n_features_to_select=n_features
    )
    
    X_reduced = reducer.fit_transform(X, y)
    
    return X_reduced, reducer.reducer.selected_features
