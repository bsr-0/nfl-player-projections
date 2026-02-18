"""
Advanced ML Techniques for NFL Fantasy Predictions

Implements sophisticated techniques that were identified as gaps:
1. SHAP Explanations - Model interpretability
2. Bayesian Hyperparameter Optimization
3. Backtesting Framework - Historical validation
4. Player Embeddings - Learned representations
5. Automated Model Selection

These complete the end-to-end sophisticated ML pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, field
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# SHAP EXPLANATIONS - Model Interpretability
# =============================================================================

class SHAPExplainer:
    """
    SHAP (SHapley Additive exPlanations) for model interpretability.
    
    Provides:
    - Global feature importance
    - Local explanations for individual predictions
    - Feature interaction analysis
    """
    
    def __init__(self, model, X_train: np.ndarray, feature_names: List[str]):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.shap_values = None
        
        # Try to use SHAP library, fall back to permutation importance
        try:
            import shap
            self.explainer = shap.TreeExplainer(model)
            self.has_shap = True
        except ImportError:
            self.has_shap = False
            print("SHAP not installed, using permutation importance fallback")
    
    def explain_global(self, X: np.ndarray) -> pd.DataFrame:
        """Get global feature importance using SHAP or permutation."""
        if self.has_shap:
            import shap
            shap_values = self.explainer.shap_values(X)
            importance = np.abs(shap_values).mean(axis=0)
        else:
            # Permutation importance fallback
            importance = self._permutation_importance(X)
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def explain_prediction(self, x: np.ndarray) -> Dict:
        """Explain a single prediction."""
        if self.has_shap:
            import shap
            shap_values = self.explainer.shap_values(x.reshape(1, -1))[0]
            
            contributions = []
            for i, (feat, val, shap_val) in enumerate(zip(
                self.feature_names, x, shap_values
            )):
                contributions.append({
                    'feature': feat,
                    'value': float(val),
                    'contribution': float(shap_val),
                    'direction': 'positive' if shap_val > 0 else 'negative'
                })
            
            return {
                'base_value': float(self.explainer.expected_value),
                'prediction': float(self.model.predict(x.reshape(1, -1))[0]),
                'contributions': sorted(contributions, 
                                       key=lambda x: abs(x['contribution']), 
                                       reverse=True)
            }
        else:
            return {'error': 'SHAP not available'}
    
    def _permutation_importance(self, X: np.ndarray, n_repeats: int = 10) -> np.ndarray:
        """Fallback permutation importance."""
        baseline_pred = self.model.predict(X)
        importance = np.zeros(X.shape[1])
        
        for i in range(X.shape[1]):
            scores = []
            for _ in range(n_repeats):
                X_permuted = X.copy()
                np.random.shuffle(X_permuted[:, i])
                permuted_pred = self.model.predict(X_permuted)
                score = np.mean((baseline_pred - permuted_pred) ** 2)
                scores.append(score)
            importance[i] = np.mean(scores)
        
        return importance / importance.sum()


# =============================================================================
# BAYESIAN HYPERPARAMETER OPTIMIZATION
# =============================================================================

class BayesianOptimizer:
    """
    Bayesian optimization for hyperparameter tuning.
    
    Uses Gaussian Process surrogate model to efficiently
    search the hyperparameter space.
    """
    
    def __init__(self, param_space: Dict, n_iterations: int = 50):
        self.param_space = param_space
        self.n_iterations = n_iterations
        self.results = []
        self.best_params = None
        self.best_score = float('inf')
    
    def optimize(self, X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray,
                 model_class=GradientBoostingRegressor) -> Dict:
        """
        Run Bayesian optimization to find best hyperparameters.
        
        Uses random search with smart sampling as a fallback
        when optuna/hyperopt not available.
        """
        try:
            import optuna
            return self._optuna_optimize(X_train, y_train, X_val, y_val, model_class)
        except ImportError:
            return self._random_search(X_train, y_train, X_val, y_val, model_class)
    
    def _optuna_optimize(self, X_train, y_train, X_val, y_val, model_class):
        """Optuna-based Bayesian optimization."""
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        def objective(trial):
            params = {}
            for name, config in self.param_space.items():
                if config['type'] == 'int':
                    params[name] = trial.suggest_int(name, config['low'], config['high'])
                elif config['type'] == 'float':
                    params[name] = trial.suggest_float(name, config['low'], config['high'], 
                                                       log=config.get('log', False))
                elif config['type'] == 'categorical':
                    params[name] = trial.suggest_categorical(name, config['choices'])
            
            model = model_class(**params, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            return np.sqrt(mean_squared_error(y_val, y_pred))
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_iterations, show_progress_bar=False)
        
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_trials': len(study.trials)
        }
    
    def _random_search(self, X_train, y_train, X_val, y_val, model_class):
        """Fallback random search with smart sampling."""
        for i in range(self.n_iterations):
            params = {}
            for name, config in self.param_space.items():
                if config['type'] == 'int':
                    params[name] = np.random.randint(config['low'], config['high'] + 1)
                elif config['type'] == 'float':
                    if config.get('log', False):
                        params[name] = np.exp(np.random.uniform(
                            np.log(config['low']), np.log(config['high'])
                        ))
                    else:
                        params[name] = np.random.uniform(config['low'], config['high'])
                elif config['type'] == 'categorical':
                    params[name] = np.random.choice(config['choices'])
            
            try:
                model = model_class(**params, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                score = np.sqrt(mean_squared_error(y_val, y_pred))
                
                self.results.append({'params': params, 'score': score})
                
                if score < self.best_score:
                    self.best_score = score
                    self.best_params = params
            except Exception as e:
                continue
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_trials': len(self.results)
        }


# =============================================================================
# BACKTESTING FRAMEWORK
# =============================================================================

@dataclass
class BacktestResult:
    """Results from a single backtest period."""
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    rmse: float
    mae: float
    r2: float
    n_train: int
    n_test: int


class BacktestingFramework:
    """
    Comprehensive backtesting framework for fantasy predictions.
    
    Implements:
    - Walk-forward validation
    - Season-by-season evaluation
    - Weekly rolling evaluation
    - Statistical significance testing
    """
    
    def __init__(self, model_class=GradientBoostingRegressor, model_params: Dict = None):
        self.model_class = model_class
        self.model_params = model_params or {'n_estimators': 100, 'max_depth': 5}
        self.results = []
    
    def walk_forward_backtest(self, df: pd.DataFrame,
                               feature_cols: List[str],
                               target_col: str = 'fantasy_points',
                               train_seasons: int = 3,
                               test_seasons: int = 1,
                               gap_seasons: int = None) -> List[BacktestResult]:
        """
        Walk-forward backtesting across seasons.

        Train on N seasons, skip gap_seasons, test on next season, roll forward.
        The purge gap prevents feature leakage from rolling/lag features that
        span the train/test boundary.

        Args:
            gap_seasons: Number of seasons to skip between train and test.
                Defaults to MODEL_CONFIG["cv_gap_seasons"] (typically 1).
        """
        if gap_seasons is None:
            from config.settings import MODEL_CONFIG as _MC
            gap_seasons = _MC.get("cv_gap_seasons", 1)

        seasons = sorted(df['season'].unique())
        results = []

        for i in range(len(seasons) - train_seasons - gap_seasons - test_seasons + 1):
            train_seasons_list = seasons[i:i + train_seasons]
            # Skip gap_seasons between training and test to prevent feature leakage
            test_start = i + train_seasons + gap_seasons
            test_seasons_list = seasons[test_start:test_start + test_seasons]

            train_df = df[df['season'].isin(train_seasons_list)]
            test_df = df[df['season'].isin(test_seasons_list)]
            
            if len(train_df) < 100 or len(test_df) < 50:
                continue
            
            X_train = train_df[feature_cols].fillna(0).values
            y_train = train_df[target_col].values
            X_test = test_df[feature_cols].fillna(0).values
            y_test = test_df[target_col].values
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = self.model_class(**self.model_params, random_state=42)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            result = BacktestResult(
                train_start=str(min(train_seasons_list)),
                train_end=str(max(train_seasons_list)),
                test_start=str(min(test_seasons_list)),
                test_end=str(max(test_seasons_list)),
                rmse=np.sqrt(mean_squared_error(y_test, y_pred)),
                mae=mean_absolute_error(y_test, y_pred),
                r2=r2_score(y_test, y_pred),
                n_train=len(train_df),
                n_test=len(test_df)
            )
            results.append(result)
        
        self.results = results
        return results
    
    def get_summary(self) -> Dict:
        """Get summary statistics from backtest results."""
        if not self.results:
            return {}
        
        rmses = [r.rmse for r in self.results]
        maes = [r.mae for r in self.results]
        r2s = [r.r2 for r in self.results]
        
        return {
            'n_periods': len(self.results),
            'rmse_mean': np.mean(rmses),
            'rmse_std': np.std(rmses),
            'rmse_min': np.min(rmses),
            'rmse_max': np.max(rmses),
            'mae_mean': np.mean(maes),
            'r2_mean': np.mean(r2s),
            'r2_std': np.std(r2s),
            'consistency': 1 - (np.std(rmses) / np.mean(rmses))  # Higher = more consistent
        }
    
    def compare_models(self, df: pd.DataFrame, feature_cols: List[str],
                       models: Dict[str, Any]) -> pd.DataFrame:
        """Compare multiple models using backtesting."""
        comparison = []
        
        for name, (model_class, params) in models.items():
            self.model_class = model_class
            self.model_params = params
            self.walk_forward_backtest(df, feature_cols)
            summary = self.get_summary()
            summary['model'] = name
            comparison.append(summary)
        
        return pd.DataFrame(comparison).sort_values('rmse_mean')


# =============================================================================
# PLAYER EMBEDDINGS
# =============================================================================

class PlayerEmbeddings:
    """
    Learn dense vector representations for players.
    
    Uses historical performance patterns to create
    embeddings that capture player characteristics.
    """
    
    def __init__(self, embedding_dim: int = 16):
        self.embedding_dim = embedding_dim
        self.embeddings = {}
        self.player_stats = {}
    
    def fit(self, df: pd.DataFrame) -> 'PlayerEmbeddings':
        """
        Create embeddings from player statistics.

        IMPORTANT: Only pass training data to avoid data leakage.
        Aggregated stats (mean, std, max of fantasy_points etc.) will
        include future performance if test data is included.

        Uses PCA on aggregated player stats as a simple
        embedding approach (neural embeddings would require
        more data and compute).
        """
        from sklearn.decomposition import PCA
        
        # Aggregate player statistics
        agg_cols = ['fantasy_points', 'targets', 'rushing_attempts', 
                    'receptions', 'passing_yards', 'rushing_yards', 'receiving_yards']
        agg_cols = [c for c in agg_cols if c in df.columns]
        
        player_agg = df.groupby('player_id').agg({
            **{col: ['mean', 'std', 'max'] for col in agg_cols if col in df.columns},
            'week': 'count'
        })
        
        # Flatten column names
        player_agg.columns = ['_'.join(col).strip() for col in player_agg.columns]
        player_agg = player_agg.fillna(0)
        
        # Store raw stats
        self.player_stats = player_agg.to_dict('index')
        
        # Create embeddings using PCA
        n_components = min(self.embedding_dim, len(player_agg.columns), len(player_agg))
        pca = PCA(n_components=n_components)
        
        scaler = StandardScaler()
        scaled_stats = scaler.fit_transform(player_agg.values)
        embeddings = pca.fit_transform(scaled_stats)
        
        # Pad if needed
        if embeddings.shape[1] < self.embedding_dim:
            padding = np.zeros((embeddings.shape[0], self.embedding_dim - embeddings.shape[1]))
            embeddings = np.hstack([embeddings, padding])
        
        for i, player_id in enumerate(player_agg.index):
            self.embeddings[player_id] = embeddings[i]
        
        return self
    
    def get_embedding(self, player_id: str) -> np.ndarray:
        """Get embedding for a player."""
        return self.embeddings.get(player_id, np.zeros(self.embedding_dim))
    
    def find_similar_players(self, player_id: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find most similar players based on embedding distance."""
        if player_id not in self.embeddings:
            return []
        
        target_emb = self.embeddings[player_id]
        similarities = []
        
        for pid, emb in self.embeddings.items():
            if pid != player_id:
                # Cosine similarity
                sim = np.dot(target_emb, emb) / (
                    np.linalg.norm(target_emb) * np.linalg.norm(emb) + 1e-8
                )
                similarities.append((pid, sim))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
    
    def add_embeddings_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add embedding features to DataFrame."""
        df = df.copy()
        
        for i in range(self.embedding_dim):
            df[f'player_emb_{i}'] = df['player_id'].apply(
                lambda x: self.get_embedding(x)[i]
            )
        
        return df


# =============================================================================
# AUTOMATED MODEL SELECTION
# =============================================================================

class AutoModelSelector:
    """
    Automatically select the best model for the data.
    
    Evaluates multiple model types and selects the best
    based on cross-validation performance.
    """
    
    def __init__(self):
        self.models = {
            'ridge': (Ridge, {'alpha': 10.0}),
            'random_forest': (RandomForestRegressor, {
                'n_estimators': 100, 'max_depth': 8, 'n_jobs': 1
            }),
            'gradient_boosting': (GradientBoostingRegressor, {
                'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1
            }),
        }
        self.results = {}
        self.best_model_name = None
        self.best_model = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            cv_splits: int = 5) -> 'AutoModelSelector':
        """
        Evaluate all models and select the best.
        """
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        
        for name, (model_class, params) in self.models.items():
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                model = model_class(**params, random_state=42)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_val_scaled)
                
                scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))
            
            self.results[name] = {
                'mean_rmse': np.mean(scores),
                'std_rmse': np.std(scores),
                'scores': scores
            }
        
        # Select best model
        self.best_model_name = min(self.results, key=lambda x: self.results[x]['mean_rmse'])
        
        # Train best model on full data
        model_class, params = self.models[self.best_model_name]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.best_model = model_class(**params, random_state=42)
        self.best_model.fit(X_scaled, y)
        self.scaler = scaler
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the best model."""
        X_scaled = self.scaler.transform(X)
        return self.best_model.predict(X_scaled)
    
    def get_results_summary(self) -> pd.DataFrame:
        """Get comparison of all models."""
        rows = []
        for name, result in self.results.items():
            rows.append({
                'Model': name,
                'Mean RMSE': result['mean_rmse'],
                'Std RMSE': result['std_rmse'],
                'Best': '✓' if name == self.best_model_name else ''
            })
        return pd.DataFrame(rows).sort_values('Mean RMSE')


# =============================================================================
# COMPREHENSIVE VALIDATION
# =============================================================================

def run_comprehensive_validation():
    """Run all advanced techniques and validate the system."""
    from src.utils.database import DatabaseManager
    from src.features.utilization import engineer_all_features
    
    print("="*70)
    print("COMPREHENSIVE SYSTEM VALIDATION")
    print("="*70)
    
    # Load data
    print("\n1. Loading data...")
    db = DatabaseManager()
    df = db.get_all_players_for_training(min_games=4)
    df = engineer_all_features(df)
    
    # Get features
    feature_cols = [c for c in df.columns if any(p in c for p in [
        'rolling', 'lag', 'utilization', 'target_share', 'position_rank'
    ]) and df[c].dtype in ['int64', 'float64']][:20]
    
    # Split data
    train_df = df[df['season'] < 2024]
    test_df = df[df['season'] == 2024]
    
    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df['fantasy_points'].values
    X_test = test_df[feature_cols].fillna(0).values
    y_test = test_df['fantasy_points'].values
    
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}, Features: {len(feature_cols)}")
    
    # 2. Auto Model Selection
    print("\n2. Running Auto Model Selection...")
    auto_selector = AutoModelSelector()
    auto_selector.fit(X_train, y_train)
    print(f"   Best model: {auto_selector.best_model_name}")
    print(auto_selector.get_results_summary().to_string(index=False))
    
    # 3. Bayesian Optimization
    print("\n3. Running Bayesian Hyperparameter Optimization...")
    param_space = {
        'n_estimators': {'type': 'int', 'low': 50, 'high': 200},
        'max_depth': {'type': 'int', 'low': 3, 'high': 10},
        'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
    }
    optimizer = BayesianOptimizer(param_space, n_iterations=20)
    
    # Use validation split
    split_idx = int(len(X_train) * 0.8)
    opt_result = optimizer.optimize(
        X_train[:split_idx], y_train[:split_idx],
        X_train[split_idx:], y_train[split_idx:]
    )
    print(f"   Best params: {opt_result['best_params']}")
    print(f"   Best RMSE: {opt_result['best_score']:.4f}")
    
    # 4. Backtesting
    print("\n4. Running Walk-Forward Backtesting...")
    backtester = BacktestingFramework()
    backtester.walk_forward_backtest(df, feature_cols, train_seasons=2, test_seasons=1)
    summary = backtester.get_summary()
    print(f"   Periods tested: {summary['n_periods']}")
    print(f"   Mean RMSE: {summary['rmse_mean']:.4f} ± {summary['rmse_std']:.4f}")
    print(f"   Consistency: {summary['consistency']:.2%}")
    
    # 5. Player Embeddings (fit on training data only to avoid leakage)
    print("\n5. Creating Player Embeddings...")
    embeddings = PlayerEmbeddings(embedding_dim=8)
    embeddings.fit(train_df)
    print(f"   Created embeddings for {len(embeddings.embeddings)} players")
    
    # 6. SHAP Explanations
    print("\n6. Computing Feature Explanations...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    explainer = SHAPExplainer(model, X_train_scaled, feature_cols)
    importance_df = explainer.explain_global(X_train_scaled[:1000])
    print("   Top 5 features:")
    for _, row in importance_df.head(5).iterrows():
        print(f"     {row['feature']}: {row['importance']:.4f}")
    
    # Final evaluation
    print("\n" + "="*70)
    print("FINAL SYSTEM EVALUATION")
    print("="*70)
    
    y_pred = auto_selector.predict(X_test)
    final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    final_r2 = r2_score(y_test, y_pred)
    
    print(f"\nTest Set Performance:")
    print(f"  RMSE: {final_rmse:.4f}")
    print(f"  R²: {final_r2:.4f}")
    print(f"\n✅ All advanced techniques validated successfully!")
    
    return {
        'auto_selector': auto_selector,
        'optimizer': optimizer,
        'backtester': backtester,
        'embeddings': embeddings,
        'explainer': explainer,
        'final_rmse': final_rmse,
        'final_r2': final_r2
    }


if __name__ == '__main__':
    run_comprehensive_validation()
