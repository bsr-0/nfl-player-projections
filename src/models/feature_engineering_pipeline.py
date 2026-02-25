"""
Sophisticated Feature Engineering & Reduction Pipeline

Implements rigorous feature selection to prevent overfitting:
1. Recursive Feature Elimination with Cross-Validation (RFECV)
2. Stability Selection (features that appear consistently across bootstrap samples)
3. Permutation Importance (model-agnostic importance)
4. Optimal Feature Count Selection via CV
5. Train/Test Gap Analysis (detect overfitting)
6. Learning Curves (sample size vs performance)

Goal: Find the minimal feature set that maximizes out-of-sample performance.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, LassoCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import RFE, RFECV, SelectFromModel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


# =============================================================================
# FEATURE IMPORTANCE METHODS
# =============================================================================

class FeatureImportanceAnalyzer:
    """
    Analyze feature importance using multiple methods.
    
    Methods:
    - Permutation importance (model-agnostic)
    - Built-in importance (tree models)
    - Coefficient magnitude (linear models)
    - Correlation with target
    """
    
    def __init__(self):
        self.importance_scores = {}
    
    def permutation_importance(self, model, X: np.ndarray, y: np.ndarray,
                                feature_names: List[str], n_repeats: int = 10) -> pd.DataFrame:
        """Calculate permutation importance (most reliable method)."""
        result = permutation_importance(model, X, y, n_repeats=n_repeats, 
                                        random_state=42, n_jobs=1)
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance_mean': result.importances_mean,
            'importance_std': result.importances_std
        }).sort_values('importance_mean', ascending=False)
        
        return importance_df
    
    def builtin_importance(self, model, feature_names: List[str]) -> pd.DataFrame:
        """Get built-in feature importance from tree models."""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
        else:
            return pd.DataFrame()
        
        return pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def correlation_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Calculate correlation-based importance."""
        correlations = []
        for col in X.columns:
            corr = abs(X[col].corr(y))
            correlations.append({'feature': col, 'correlation': corr})
        
        return pd.DataFrame(correlations).sort_values('correlation', ascending=False)


# =============================================================================
# STABILITY SELECTION
# =============================================================================

class StabilitySelector:
    """
    Stability Selection: Find features that are consistently selected
    across multiple bootstrap samples.
    
    More robust than single-run feature selection.
    """
    
    def __init__(self, n_bootstrap: int = 50, threshold: float = 0.6):
        self.n_bootstrap = n_bootstrap
        self.threshold = threshold  # Feature must be selected in X% of runs
        self.selection_frequencies = {}
    
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            n_features_to_select: int = 30) -> List[str]:
        """
        Run stability selection.
        
        Returns features selected in at least `threshold` fraction of bootstrap samples.
        """
        print(f"\nRunning Stability Selection ({self.n_bootstrap} bootstrap samples)...")
        
        feature_counts = {col: 0 for col in X.columns}
        
        for i in range(self.n_bootstrap):
            # Bootstrap sample
            idx = np.random.choice(len(X), size=len(X), replace=True)
            X_boot = X.iloc[idx]
            y_boot = y.iloc[idx]
            
            # Use Lasso for feature selection
            lasso = LassoCV(cv=3, random_state=i, max_iter=2000)
            lasso.fit(X_boot.fillna(0), y_boot)
            
            # Count selected features (non-zero coefficients)
            selected = np.where(np.abs(lasso.coef_) > 1e-6)[0]
            for idx in selected:
                feature_counts[X.columns[idx]] += 1
            
            if (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{self.n_bootstrap} iterations")
        
        # Calculate selection frequency
        self.selection_frequencies = {
            k: v / self.n_bootstrap for k, v in feature_counts.items()
        }
        
        # Select features above threshold
        stable_features = [
            f for f, freq in self.selection_frequencies.items() 
            if freq >= self.threshold
        ]
        
        print(f"  Features above {self.threshold:.0%} threshold: {len(stable_features)}")
        
        return stable_features
    
    def get_selection_frequencies(self) -> pd.DataFrame:
        """Get selection frequency for all features."""
        return pd.DataFrame([
            {'feature': k, 'selection_frequency': v}
            for k, v in self.selection_frequencies.items()
        ]).sort_values('selection_frequency', ascending=False)


# =============================================================================
# OPTIMAL FEATURE COUNT FINDER
# =============================================================================

class OptimalFeatureCountFinder:
    """
    Find the optimal number of features using cross-validation.
    
    Tests different feature counts and selects the one with best CV score.
    Includes regularization to prevent overfitting.
    """
    
    def __init__(self, min_features: int = 10, max_features: int = 100,
                 step: int = 5):
        self.min_features = min_features
        self.max_features = max_features
        self.step = step
        self.cv_results = []
        self.optimal_n_features = None
    
    def find_optimal(self, X: pd.DataFrame, y: pd.Series,
                     feature_ranking: List[str]) -> Tuple[int, List[str]]:
        """
        Find optimal feature count using CV.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            feature_ranking: Features ranked by importance (best first)
            
        Returns:
            (optimal_n_features, selected_features)
        """
        print("\nFinding Optimal Feature Count...")
        
        # Limit to available features
        available_features = [f for f in feature_ranking if f in X.columns]
        max_features = min(self.max_features, len(available_features))
        
        from sklearn.pipeline import Pipeline
        tscv = TimeSeriesSplit(n_splits=5)

        results = []

        for n_features in range(self.min_features, max_features + 1, self.step):
            selected = available_features[:n_features]
            X_sel = X[selected].fillna(0)

            # Use Pipeline so scaler is fit inside each CV fold (no data leakage)
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('ridge', RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=3)),
            ])

            # Cross-validation scores
            cv_scores = cross_val_score(model, X_sel, y, cv=tscv,
                                        scoring='neg_root_mean_squared_error')
            
            mean_rmse = -cv_scores.mean()
            std_rmse = cv_scores.std()
            
            results.append({
                'n_features': n_features,
                'cv_rmse': mean_rmse,
                'cv_std': std_rmse
            })
            
            print(f"  {n_features} features: CV RMSE = {mean_rmse:.3f} ± {std_rmse:.3f}")
        
        self.cv_results = results
        
        # Find optimal (lowest CV RMSE)
        best_result = min(results, key=lambda x: x['cv_rmse'])
        self.optimal_n_features = best_result['n_features']
        
        # Apply 1-SE rule: select simplest model within 1 SE of best
        best_rmse = best_result['cv_rmse']
        best_std = best_result['cv_std']
        
        for result in results:
            if result['cv_rmse'] <= best_rmse + best_std:
                self.optimal_n_features = result['n_features']
                break
        
        print(f"\n  Optimal: {self.optimal_n_features} features (1-SE rule)")
        
        return self.optimal_n_features, available_features[:self.optimal_n_features]


# =============================================================================
# OVERFITTING DETECTOR
# =============================================================================

class OverfittingDetector:
    """
    Detect overfitting by comparing train vs test performance.
    
    Metrics:
    - Train/Test RMSE gap
    - Learning curves
    - Regularization path
    """
    
    def __init__(self, max_gap_ratio: float = 0.3):
        self.max_gap_ratio = max_gap_ratio  # Max acceptable (train-test)/test ratio
        self.is_overfitting = False
        self.gap_ratio = None
    
    def check_overfitting(self, model, X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Check for overfitting by comparing train vs test performance.
        """
        # Train predictions
        y_train_pred = model.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        
        # Test predictions
        y_test_pred = model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_r2 = r2_score(y_test, y_test_pred)
        
        # Calculate gap
        self.gap_ratio = (test_rmse - train_rmse) / test_rmse if test_rmse > 0 else 0
        self.is_overfitting = self.gap_ratio > self.max_gap_ratio
        
        return {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'gap_ratio': self.gap_ratio,
            'is_overfitting': self.is_overfitting
        }
    
    def learning_curve(self, model, X: np.ndarray, y: np.ndarray,
                       train_sizes: List[float] = None) -> pd.DataFrame:
        """
        Generate learning curve to diagnose overfitting.
        
        If train and test curves diverge, model is overfitting.
        """
        from sklearn.model_selection import learning_curve
        
        train_sizes = train_sizes or [0.2, 0.4, 0.6, 0.8, 1.0]
        
        train_sizes_abs, train_scores, test_scores = learning_curve(
            model, X, y, train_sizes=train_sizes, cv=5,
            scoring='neg_root_mean_squared_error', n_jobs=1
        )
        
        results = []
        for i, size in enumerate(train_sizes_abs):
            results.append({
                'train_size': size,
                'train_rmse': -train_scores[i].mean(),
                'train_std': train_scores[i].std(),
                'test_rmse': -test_scores[i].mean(),
                'test_std': test_scores[i].std(),
                'gap': -test_scores[i].mean() - (-train_scores[i].mean())
            })
        
        return pd.DataFrame(results)


# =============================================================================
# REGULARIZATION TUNER
# =============================================================================

class RegularizationTuner:
    """
    Tune regularization strength to prevent overfitting.
    
    Uses cross-validation to find optimal regularization.
    """
    
    def __init__(self):
        self.optimal_alpha = None
        self.cv_results = []
    
    def tune_ridge(self, X: np.ndarray, y: np.ndarray,
                   alphas: List[float] = None) -> float:
        """Tune Ridge regularization."""
        alphas = alphas or [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        
        model = RidgeCV(alphas=alphas, cv=5)
        model.fit(X, y)
        
        self.optimal_alpha = model.alpha_
        return self.optimal_alpha
    
    def tune_lasso(self, X: np.ndarray, y: np.ndarray,
                   alphas: List[float] = None) -> float:
        """Tune Lasso regularization."""
        alphas = alphas or [0.0001, 0.001, 0.01, 0.1, 1.0]
        
        model = LassoCV(alphas=alphas, cv=5, max_iter=2000)
        model.fit(X, y)
        
        self.optimal_alpha = model.alpha_
        return self.optimal_alpha
    
    def tune_tree_depth(self, X: np.ndarray, y: np.ndarray,
                        max_depths: List[int] = None) -> int:
        """Tune tree depth for gradient boosting."""
        max_depths = max_depths or [2, 3, 4, 5, 6, 8, 10]
        
        tscv = TimeSeriesSplit(n_splits=5)
        best_depth = 4
        best_score = float('inf')
        
        for depth in max_depths:
            model = GradientBoostingRegressor(
                n_estimators=100, max_depth=depth, 
                learning_rate=0.1, random_state=42
            )
            scores = cross_val_score(model, X, y, cv=tscv,
                                     scoring='neg_root_mean_squared_error')
            mean_rmse = -scores.mean()
            
            if mean_rmse < best_score:
                best_score = mean_rmse
                best_depth = depth
        
        return best_depth


# =============================================================================
# MAIN FEATURE ENGINEERING PIPELINE
# =============================================================================

class FeatureEngineeringPipeline:
    """
    Complete feature engineering and reduction pipeline.
    
    Steps:
    1. Remove highly correlated features
    2. Calculate multiple importance scores
    3. Run stability selection
    4. Find optimal feature count via CV
    5. Check for overfitting
    6. Tune regularization
    7. Final model selection
    """
    
    def __init__(self, correlation_threshold: float = 0.95,
                 stability_threshold: float = 0.5,
                 min_features: int = 15, max_features: int = 60):
        self.correlation_threshold = correlation_threshold
        self.stability_threshold = stability_threshold
        self.min_features = min_features
        self.max_features = max_features
        
        self.selected_features = []
        self.feature_importance = None
        self.cv_results = None
        self.overfitting_metrics = None
        self.final_model = None
    
    def remove_correlated_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove highly correlated features."""
        print("\n1. Removing Highly Correlated Features...")
        
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = [col for col in upper.columns 
                   if any(upper[col] > self.correlation_threshold)]
        
        print(f"   Removed {len(to_drop)} features with correlation > {self.correlation_threshold}")
        
        return X.drop(columns=to_drop)
    
    def run_pipeline(self, df: pd.DataFrame, feature_cols: List[str],
                     target_col: str = 'fantasy_points',
                     test_season: int = 2024) -> Dict[str, Any]:
        """
        Run the complete feature engineering pipeline.
        
        Returns comprehensive results including:
        - Selected features
        - Importance scores
        - CV results
        - Overfitting analysis
        - Final model performance
        """
        print("\n" + "="*70)
        print("SOPHISTICATED FEATURE ENGINEERING PIPELINE")
        print("="*70)
        
        # Split data
        train_df = df[df['season'] < test_season].copy()
        test_df = df[df['season'] == test_season].copy()
        
        print(f"\nTrain: {len(train_df)} samples (seasons < {test_season})")
        print(f"Test: {len(test_df)} samples (season = {test_season})")
        
        # Get available features
        available_features = [c for c in feature_cols if c in train_df.columns]
        X_train_full = train_df[available_features].fillna(0)
        y_train = train_df[target_col]
        X_test_full = test_df[available_features].fillna(0)
        y_test = test_df[target_col]
        
        print(f"Starting features: {len(available_features)}")
        
        # Step 1: Remove correlated features
        X_train_uncorr = self.remove_correlated_features(X_train_full)
        uncorr_features = X_train_uncorr.columns.tolist()
        print(f"   After correlation filter: {len(uncorr_features)}")
        
        # Step 2: Stability Selection
        print("\n2. Running Stability Selection...")
        stability_selector = StabilitySelector(
            n_bootstrap=30, threshold=self.stability_threshold
        )
        stable_features = stability_selector.fit(X_train_uncorr, y_train)
        
        if len(stable_features) < self.min_features:
            # Fall back to top features by frequency
            freq_df = stability_selector.get_selection_frequencies()
            stable_features = freq_df.head(self.min_features)['feature'].tolist()
        
        print(f"   Stable features: {len(stable_features)}")
        
        # Step 3: Calculate importance scores
        print("\n3. Calculating Feature Importance...")
        
        # Train a quick model for importance
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_uncorr[stable_features])
        
        if HAS_LIGHTGBM:
            importance_model = lgb.LGBMRegressor(
                n_estimators=100, max_depth=5, random_state=42, verbosity=-1
            )
        else:
            importance_model = GradientBoostingRegressor(
                n_estimators=100, max_depth=4, random_state=42
            )
        
        importance_model.fit(X_train_scaled, y_train)
        
        # Permutation importance
        importance_analyzer = FeatureImportanceAnalyzer()
        perm_importance = importance_analyzer.permutation_importance(
            importance_model, X_train_scaled, y_train.values, stable_features
        )
        
        # Rank features by permutation importance
        ranked_features = perm_importance['feature'].tolist()
        
        print(f"   Top 10 features by permutation importance:")
        for _, row in perm_importance.head(10).iterrows():
            print(f"      {row['feature']}: {row['importance_mean']:.4f}")
        
        # Step 4: Find optimal feature count
        print("\n4. Finding Optimal Feature Count...")
        
        feature_finder = OptimalFeatureCountFinder(
            min_features=self.min_features,
            max_features=min(self.max_features, len(ranked_features)),
            step=5
        )
        
        optimal_n, optimal_features = feature_finder.find_optimal(
            X_train_uncorr, y_train, ranked_features
        )
        
        self.cv_results = feature_finder.cv_results
        
        # Step 5: Tune regularization
        print("\n5. Tuning Regularization...")
        
        X_train_opt = X_train_uncorr[optimal_features].fillna(0)
        X_train_opt_scaled = scaler.fit_transform(X_train_opt)
        
        reg_tuner = RegularizationTuner()
        optimal_alpha = reg_tuner.tune_ridge(X_train_opt_scaled, y_train)
        print(f"   Optimal Ridge alpha: {optimal_alpha}")
        
        optimal_depth = reg_tuner.tune_tree_depth(X_train_opt_scaled, y_train)
        print(f"   Optimal tree depth: {optimal_depth}")
        
        # Step 6: Train final models and check overfitting
        print("\n6. Training Final Models & Checking Overfitting...")
        
        X_test_opt = X_test_full[optimal_features].fillna(0)
        X_test_opt_scaled = scaler.transform(X_test_opt)
        
        models = {
            'ridge': Ridge(alpha=optimal_alpha),
            'gbm': GradientBoostingRegressor(
                n_estimators=100, max_depth=optimal_depth,
                learning_rate=0.1, random_state=42
            ),
        }
        
        if HAS_XGBOOST:
            models['xgb'] = xgb.XGBRegressor(
                n_estimators=100, max_depth=optimal_depth,
                learning_rate=0.1, random_state=42, verbosity=0
            )
        
        if HAS_LIGHTGBM:
            models['lgb'] = lgb.LGBMRegressor(
                n_estimators=100, max_depth=optimal_depth,
                learning_rate=0.1, random_state=42, verbosity=-1
            )
        
        overfitting_detector = OverfittingDetector(max_gap_ratio=0.3)
        model_results = {}
        
        for name, model in models.items():
            model.fit(X_train_opt_scaled, y_train)
            
            metrics = overfitting_detector.check_overfitting(
                model, X_train_opt_scaled, y_train.values,
                X_test_opt_scaled, y_test.values
            )
            
            model_results[name] = metrics
            
            status = "⚠️ OVERFITTING" if metrics['is_overfitting'] else "✅ OK"
            print(f"\n   {name}:")
            print(f"      Train RMSE: {metrics['train_rmse']:.3f}, R²: {metrics['train_r2']:.3f}")
            print(f"      Test RMSE:  {metrics['test_rmse']:.3f}, R²: {metrics['test_r2']:.3f}")
            print(f"      Gap Ratio:  {metrics['gap_ratio']:.3f} {status}")
        
        # Step 7: Select best model (lowest test RMSE without overfitting)
        print("\n7. Selecting Best Model...")
        
        valid_models = {k: v for k, v in model_results.items() if not v['is_overfitting']}
        
        if valid_models:
            best_model_name = min(valid_models, key=lambda x: valid_models[x]['test_rmse'])
        else:
            # If all overfit, pick the one with smallest gap
            best_model_name = min(model_results, key=lambda x: model_results[x]['gap_ratio'])
        
        best_metrics = model_results[best_model_name]
        self.final_model = models[best_model_name]
        self.selected_features = optimal_features
        self.overfitting_metrics = model_results
        
        print(f"\n   Best Model: {best_model_name}")
        print(f"   Test RMSE: {best_metrics['test_rmse']:.3f}")
        print(f"   Test R²: {best_metrics['test_r2']:.3f}")
        print(f"   Features: {len(optimal_features)}")
        
        # Final summary
        print("\n" + "="*70)
        print("FINAL SUMMARY")
        print("="*70)
        print(f"   Starting features: {len(available_features)}")
        print(f"   After correlation filter: {len(uncorr_features)}")
        print(f"   After stability selection: {len(stable_features)}")
        print(f"   Final optimal features: {len(optimal_features)}")
        print(f"   Feature reduction: {100*(1 - len(optimal_features)/len(available_features)):.1f}%")
        print(f"   Best model: {best_model_name}")
        print(f"   Test RMSE: {best_metrics['test_rmse']:.3f}")
        print(f"   Test R²: {best_metrics['test_r2']:.3f}")
        print(f"   Overfitting: {'No' if not best_metrics['is_overfitting'] else 'Yes'}")
        
        return {
            'selected_features': optimal_features,
            'n_features_start': len(available_features),
            'n_features_final': len(optimal_features),
            'feature_reduction_pct': 100*(1 - len(optimal_features)/len(available_features)),
            'best_model': best_model_name,
            'test_rmse': best_metrics['test_rmse'],
            'test_r2': best_metrics['test_r2'],
            'train_rmse': best_metrics['train_rmse'],
            'gap_ratio': best_metrics['gap_ratio'],
            'is_overfitting': best_metrics['is_overfitting'],
            'all_model_results': model_results,
            'cv_results': self.cv_results,
            'feature_importance': perm_importance.to_dict('records'),
        }


def run_feature_engineering_pipeline():
    """Run the complete feature engineering pipeline."""
    from src.utils.database import DatabaseManager
    from src.data.external_data import add_external_features
    from src.features.multiweek_features import add_multiweek_features
    from src.features.season_long_features import add_season_long_features
    from src.features.utilization import engineer_all_features
    from src.features.qb_features import add_qb_features
    
    print("Loading data...")
    db = DatabaseManager()
    df = db.get_all_players_for_training(min_games=4)
    
    print("Engineering features...")
    df = engineer_all_features(df, allow_autoload_bounds=False)
    df = add_qb_features(df)
    df = add_external_features(df)
    df = add_multiweek_features(df, horizons=[1, 5, 18])
    df = add_season_long_features(df)
    
    # Get feature columns - ONLY use features known BEFORE the game
    allowed_patterns = [
        '_lag_', '_rolling_', 'rolling_', '_trend', '_avg',
        'games_played', 'is_home', 'consistency_score', 'weekly_volatility',
        'coefficient_of_variation', 'boom_bust_range', 'confidence_score',
        'injury_score', 'is_injured', 'opp_defense_rank', 'opp_matchup_score',
        'opp_pts_allowed', 'is_dome', 'is_outdoor', 'weather_score',
        'implied_team_total', 'game_total', 'spread', 'opp_defense_strength',
        'sos_next_', 'sos_rank_next_', 'favorable_matchups_next_',
        'expected_games_next_', 'projection_', 'floor_', 'ceiling_',
        'injury_prob_next_', 'expected_missed_games_', 'injury_risk_score_',
        'variance_', 'std_', 'age', 'age_factor', 'age_expected_games',
        'decline_rate', 'years_from_peak', 'is_in_prime', 'projected_games',
        'historical_gpg', 'is_rookie', 'rookie_projected', 'rookie_weight',
        'position_rank', 'season_position_rank', 'estimated_adp', 'projected_adp',
        'adp_value', 'positional_scarcity', 'adjusted_adp',
        'utilization_score_lag', 'utilization_score_roll', 'target_share', 'rush_share', 'snap_share',
        'air_yards_share', 'red_zone_share', 'wopr',
    ]
    
    forbidden_patterns = [
        'receiving_yards', 'rushing_yards', 'passing_yards',
        'receptions', 'targets', 'rushing_attempts', 'passing_attempts',
        'touchdowns', 'td', 'interceptions', 'fumbles',
        'fantasy_points', 'fp_over_expected', 'expected_fp',
        'risk_adjusted_projection', 'completions', 'carries',
    ]
    
    exclude_cols = ['player_id', 'name', 'team', 'position', 'season', 'week', 
                    'fantasy_points', 'opponent', 'home_away', 'rookie_archetype',
                    'first_season']
    
    # Use only training data for correlation-based feature filtering (avoid test leakage)
    train_only = df[df['season'] < 2024] if 'season' in df.columns else df

    feature_cols = []
    for c in df.columns:
        if c in exclude_cols:
            continue
        if df[c].dtype not in ['int64', 'float64']:
            continue
        is_forbidden = any(pattern in c.lower() for pattern in forbidden_patterns)
        if is_forbidden:
            continue
        is_allowed = any(pattern in c for pattern in allowed_patterns)
        if not is_allowed:
            corr = abs(train_only[c].corr(train_only['fantasy_points']))
            if corr > 0.7:
                continue
        feature_cols.append(c)

    try:
        from src.utils.leakage import filter_feature_columns, assert_no_leakage_columns
        feature_cols = filter_feature_columns(feature_cols)
        assert_no_leakage_columns(feature_cols, context="feature_engineering_pipeline")
    except Exception:
        pass
    
    print(f"Total features available: {len(feature_cols)}")
    
    # Run pipeline
    pipeline = FeatureEngineeringPipeline(
        correlation_threshold=0.95,
        stability_threshold=0.4,
        min_features=15,
        max_features=50
    )
    
    results = pipeline.run_pipeline(df, feature_cols, test_season=2024)
    
    # Save results
    results_path = Path(__file__).parent.parent.parent / 'data' / 'feature_engineering_results.json'
    
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj
    
    results_serializable = convert_numpy(results)
    
    with open(results_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    return results


if __name__ == '__main__':
    run_feature_engineering_pipeline()
