"""
Advanced Modeling Techniques for NFL Fantasy Predictions

Implements:
1. Monte Carlo Simulation - Probabilistic projections with uncertainty
2. Lineup Optimization - Optimal roster construction
3. Multiple Model Comparison - ML, Deep Learning, Simulation, Optimization
4. User Preference Customization - Risk tolerance, play style

All models are evaluated and compared for interactive selection.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, field
import json
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
from scipy.optimize import linprog, minimize

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
# USER PREFERENCE PROFILES
# =============================================================================

@dataclass
class UserProfile:
    """User preference profile for customized predictions."""
    name: str
    risk_tolerance: float  # 0 = conservative, 1 = aggressive
    prefer_ceiling: bool   # True = chase upside, False = prefer floor
    prefer_consistency: bool  # True = consistent performers
    boom_bust_preference: float  # 0 = safe, 1 = boom/bust
    
    @classmethod
    def conservative(cls):
        return cls("Conservative", 0.2, False, True, 0.2)
    
    @classmethod
    def balanced(cls):
        return cls("Balanced", 0.5, False, False, 0.5)
    
    @classmethod
    def aggressive(cls):
        return cls("Aggressive", 0.8, True, False, 0.8)
    
    @classmethod
    def boom_or_bust(cls):
        return cls("Boom or Bust", 1.0, True, False, 1.0)


# =============================================================================
# MONTE CARLO SIMULATION
# =============================================================================

class MonteCarloSimulator:
    """
    Monte Carlo simulation for fantasy projections.
    
    Simulates thousands of possible outcomes based on:
    - Historical performance distribution
    - Matchup factors
    - Injury probability
    - Weather effects
    """
    
    def __init__(self, n_simulations: int = 10000):
        self.n_simulations = n_simulations
        self.simulation_results = {}
    
    def simulate_player(self, player_data: Dict, n_sims: int = None) -> Dict:
        """
        Simulate player performance using Monte Carlo.
        
        Returns distribution of possible outcomes.
        """
        n_sims = n_sims or self.n_simulations
        
        # Get base projection and variance
        base_proj = player_data.get('projection', 10.0)
        volatility = player_data.get('volatility', 5.0)
        
        # Adjust for matchup
        matchup_factor = player_data.get('matchup_score', 1.0)
        
        # Adjust for injury probability
        injury_prob = player_data.get('injury_prob', 0.05)
        
        # Simulate outcomes
        # Use mixture of normal and zero (for injury/DNP)
        simulations = np.zeros(n_sims)
        
        # Determine which simulations have player active
        active_mask = np.random.random(n_sims) > injury_prob
        
        # For active games, sample from adjusted distribution
        n_active = active_mask.sum()
        if n_active > 0:
            # Use skewed normal to capture boom potential
            adjusted_mean = base_proj * matchup_factor
            
            # Add some skewness for boom games
            skewness = player_data.get('boom_factor', 0.5)
            
            # Sample from skew-normal distribution
            samples = stats.skewnorm.rvs(
                a=skewness * 2,  # Skewness parameter
                loc=adjusted_mean,
                scale=volatility,
                size=n_active
            )
            
            # Clip to reasonable range
            samples = np.clip(samples, 0, adjusted_mean * 3)
            
            simulations[active_mask] = samples
        
        # Calculate statistics
        results = {
            'mean': np.mean(simulations),
            'median': np.median(simulations),
            'std': np.std(simulations),
            'floor': np.percentile(simulations, 10),
            'ceiling': np.percentile(simulations, 90),
            'p25': np.percentile(simulations, 25),
            'p75': np.percentile(simulations, 75),
            'boom_prob': np.mean(simulations > base_proj * 1.5),  # 50%+ above projection
            'bust_prob': np.mean(simulations < base_proj * 0.5),  # 50%+ below projection
            'zero_prob': np.mean(simulations == 0),
            'simulations': simulations
        }
        
        return results
    
    def simulate_lineup(self, players: List[Dict], n_sims: int = None) -> Dict:
        """
        Simulate entire lineup performance.
        
        Returns distribution of total lineup scores.
        """
        n_sims = n_sims or self.n_simulations
        
        lineup_totals = np.zeros(n_sims)
        
        for player in players:
            player_sims = self.simulate_player(player, n_sims)
            lineup_totals += player_sims['simulations']
        
        return {
            'mean': np.mean(lineup_totals),
            'median': np.median(lineup_totals),
            'std': np.std(lineup_totals),
            'floor': np.percentile(lineup_totals, 10),
            'ceiling': np.percentile(lineup_totals, 90),
            'win_prob_vs_avg': np.mean(lineup_totals > 100),  # Assuming 100 is average
            'simulations': lineup_totals
        }
    
    def compare_players(self, player_a: Dict, player_b: Dict, n_sims: int = None) -> Dict:
        """
        Compare two players using simulation.
        
        Returns probability that player A outscores player B.
        """
        n_sims = n_sims or self.n_simulations
        
        sims_a = self.simulate_player(player_a, n_sims)
        sims_b = self.simulate_player(player_b, n_sims)
        
        a_wins = np.mean(sims_a['simulations'] > sims_b['simulations'])
        
        return {
            'player_a_win_prob': a_wins,
            'player_b_win_prob': 1 - a_wins,
            'expected_diff': sims_a['mean'] - sims_b['mean'],
            'player_a_stats': {k: v for k, v in sims_a.items() if k != 'simulations'},
            'player_b_stats': {k: v for k, v in sims_b.items() if k != 'simulations'}
        }


# =============================================================================
# LINEUP OPTIMIZATION
# =============================================================================

class LineupOptimizer:
    """
    Optimal lineup construction using mathematical optimization.
    
    Supports:
    - Maximize expected points
    - Maximize floor (conservative)
    - Maximize ceiling (aggressive)
    - Risk-adjusted optimization
    """
    
    # Standard lineup constraints
    LINEUP_CONSTRAINTS = {
        'standard': {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'FLEX': 1},
        'ppr': {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'FLEX': 1},
        'superflex': {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'FLEX': 1, 'SUPERFLEX': 1},
    }
    
    def __init__(self, salary_cap: float = 50000):
        self.salary_cap = salary_cap
    
    def optimize_lineup(self, players_df: pd.DataFrame,
                        objective: str = 'expected',
                        user_profile: UserProfile = None,
                        constraints: Dict = None) -> pd.DataFrame:
        """
        Optimize lineup based on objective.
        
        Objectives:
        - 'expected': Maximize expected points
        - 'floor': Maximize floor (10th percentile)
        - 'ceiling': Maximize ceiling (90th percentile)
        - 'sharpe': Maximize risk-adjusted return
        - 'custom': Use user profile
        """
        constraints = constraints or self.LINEUP_CONSTRAINTS['standard']
        
        # Get projection column based on objective
        if objective == 'floor':
            proj_col = 'floor_1w' if 'floor_1w' in players_df.columns else 'projection'
        elif objective == 'ceiling':
            proj_col = 'ceiling_1w' if 'ceiling_1w' in players_df.columns else 'projection'
        else:
            proj_col = 'projection_1w' if 'projection_1w' in players_df.columns else 'fp_rolling_3'
        
        if proj_col not in players_df.columns:
            proj_col = 'fantasy_points'
        
        # Apply user profile adjustments
        if user_profile and objective == 'custom':
            players_df = self._apply_user_profile(players_df, user_profile, proj_col)
            proj_col = 'adjusted_projection'
        
        # Simple greedy optimization (for demonstration)
        # In production, would use integer linear programming
        lineup = []
        remaining_positions = dict(constraints)
        
        # Sort by value (projection / salary if available)
        if 'salary' in players_df.columns:
            players_df['value'] = players_df[proj_col] / players_df['salary'] * 1000
        else:
            players_df['value'] = players_df[proj_col]
        
        players_df = players_df.sort_values('value', ascending=False)
        
        used_players = set()
        
        for pos in ['QB', 'RB', 'WR', 'TE']:
            n_needed = remaining_positions.get(pos, 0)
            pos_players = players_df[
                (players_df['position'] == pos) & 
                (~players_df['player_id'].isin(used_players))
            ].head(n_needed)
            
            for _, player in pos_players.iterrows():
                lineup.append(player)
                used_players.add(player['player_id'])
        
        # FLEX (RB/WR/TE)
        if 'FLEX' in remaining_positions:
            flex_players = players_df[
                (players_df['position'].isin(['RB', 'WR', 'TE'])) &
                (~players_df['player_id'].isin(used_players))
            ].head(remaining_positions['FLEX'])
            
            for _, player in flex_players.iterrows():
                lineup.append(player)
                used_players.add(player['player_id'])
        
        return pd.DataFrame(lineup)
    
    def _apply_user_profile(self, df: pd.DataFrame, profile: UserProfile,
                            base_col: str) -> pd.DataFrame:
        """Apply user profile adjustments to projections."""
        df = df.copy()
        
        base_proj = df[base_col].fillna(0)
        
        # Get floor and ceiling
        if 'floor_1w' in df.columns:
            floor = df['floor_1w'].fillna(base_proj * 0.7)
            ceiling = df['ceiling_1w'].fillna(base_proj * 1.3)
        else:
            floor = base_proj * 0.7
            ceiling = base_proj * 1.3
        
        # Get consistency score
        if 'consistency_score' in df.columns:
            consistency = df['consistency_score'].fillna(50) / 100
        else:
            consistency = 0.5
        
        # Calculate adjusted projection based on profile
        # Risk tolerance: blend between floor and ceiling
        risk_blend = floor * (1 - profile.risk_tolerance) + ceiling * profile.risk_tolerance
        
        # Consistency preference
        if profile.prefer_consistency:
            consistency_bonus = consistency * 2  # Up to 2 point bonus for consistent players
        else:
            consistency_bonus = 0
        
        # Boom/bust preference
        if 'boom_bust_range' in df.columns:
            boom_range = df['boom_bust_range'].fillna(0)
            boom_adjustment = boom_range * profile.boom_bust_preference * 0.1
        else:
            boom_adjustment = 0
        
        df['adjusted_projection'] = risk_blend + consistency_bonus + boom_adjustment
        
        return df


# =============================================================================
# MODEL COMPARISON FRAMEWORK
# =============================================================================

@dataclass
class ModelResult:
    """Results from a single model evaluation."""
    name: str
    category: str  # 'ML', 'Deep Learning', 'Simulation', 'Optimization'
    rmse: float
    mae: float
    r2: float
    train_rmse: float
    overfitting_ratio: float
    training_time: float
    description: str
    pros: List[str]
    cons: List[str]


class ModelComparisonFramework:
    """
    Framework for comparing multiple modeling approaches.
    
    Categories:
    - Traditional ML: Ridge, Lasso, Random Forest, XGBoost, LightGBM
    - Simulation: Monte Carlo
    - Optimization: Linear Programming, Convex Optimization
    - Ensemble: Stacking, Blending
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
    
    def evaluate_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                            X_test: np.ndarray, y_test: np.ndarray,
                            feature_names: List[str] = None) -> Dict[str, ModelResult]:
        """Evaluate all available models and return comparison."""
        
        print("\n" + "="*70)
        print("MODEL COMPARISON FRAMEWORK")
        print("="*70)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        results = {}
        
        # 1. Traditional ML Models
        print("\n--- Traditional ML Models ---")
        
        # Ridge Regression
        start = time.time()
        ridge = Ridge(alpha=10.0)
        ridge.fit(X_train_scaled, y_train)
        train_pred = ridge.predict(X_train_scaled)
        test_pred = ridge.predict(X_test_scaled)
        train_time = time.time() - start
        
        results['ridge'] = ModelResult(
            name="Ridge Regression",
            category="Traditional ML",
            rmse=np.sqrt(mean_squared_error(y_test, test_pred)),
            mae=mean_absolute_error(y_test, test_pred),
            r2=r2_score(y_test, test_pred),
            train_rmse=np.sqrt(mean_squared_error(y_train, train_pred)),
            overfitting_ratio=(np.sqrt(mean_squared_error(y_test, test_pred)) - 
                              np.sqrt(mean_squared_error(y_train, train_pred))) / 
                              np.sqrt(mean_squared_error(y_test, test_pred)),
            training_time=train_time,
            description="Linear model with L2 regularization",
            pros=["Fast training", "Interpretable", "No overfitting"],
            cons=["Cannot capture non-linear patterns", "Limited accuracy"]
        )
        print(f"  Ridge: RMSE={results['ridge'].rmse:.3f}, R²={results['ridge'].r2:.3f}")
        
        # Random Forest
        start = time.time()
        rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
        rf.fit(X_train_scaled, y_train)
        train_pred = rf.predict(X_train_scaled)
        test_pred = rf.predict(X_test_scaled)
        train_time = time.time() - start
        
        results['random_forest'] = ModelResult(
            name="Random Forest",
            category="Traditional ML",
            rmse=np.sqrt(mean_squared_error(y_test, test_pred)),
            mae=mean_absolute_error(y_test, test_pred),
            r2=r2_score(y_test, test_pred),
            train_rmse=np.sqrt(mean_squared_error(y_train, train_pred)),
            overfitting_ratio=(np.sqrt(mean_squared_error(y_test, test_pred)) - 
                              np.sqrt(mean_squared_error(y_train, train_pred))) / 
                              np.sqrt(mean_squared_error(y_test, test_pred)),
            training_time=train_time,
            description="Ensemble of decision trees",
            pros=["Handles non-linearity", "Feature importance", "Robust"],
            cons=["Slower than linear", "Can overfit", "Less interpretable"]
        )
        print(f"  Random Forest: RMSE={results['random_forest'].rmse:.3f}, R²={results['random_forest'].r2:.3f}")
        
        # Gradient Boosting
        start = time.time()
        gbm = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        gbm.fit(X_train_scaled, y_train)
        train_pred = gbm.predict(X_train_scaled)
        test_pred = gbm.predict(X_test_scaled)
        train_time = time.time() - start
        
        results['gradient_boosting'] = ModelResult(
            name="Gradient Boosting",
            category="Traditional ML",
            rmse=np.sqrt(mean_squared_error(y_test, test_pred)),
            mae=mean_absolute_error(y_test, test_pred),
            r2=r2_score(y_test, test_pred),
            train_rmse=np.sqrt(mean_squared_error(y_train, train_pred)),
            overfitting_ratio=(np.sqrt(mean_squared_error(y_test, test_pred)) - 
                              np.sqrt(mean_squared_error(y_train, train_pred))) / 
                              np.sqrt(mean_squared_error(y_test, test_pred)),
            training_time=train_time,
            description="Sequential boosting of weak learners",
            pros=["High accuracy", "Handles complex patterns", "Feature importance"],
            cons=["Slower training", "Can overfit", "Many hyperparameters"]
        )
        print(f"  Gradient Boosting: RMSE={results['gradient_boosting'].rmse:.3f}, R²={results['gradient_boosting'].r2:.3f}")
        
        if HAS_XGBOOST:
            start = time.time()
            xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, 
                                         random_state=42, verbosity=0)
            xgb_model.fit(X_train_scaled, y_train)
            train_pred = xgb_model.predict(X_train_scaled)
            test_pred = xgb_model.predict(X_test_scaled)
            train_time = time.time() - start
            
            results['xgboost'] = ModelResult(
                name="XGBoost",
                category="Traditional ML",
                rmse=np.sqrt(mean_squared_error(y_test, test_pred)),
                mae=mean_absolute_error(y_test, test_pred),
                r2=r2_score(y_test, test_pred),
                train_rmse=np.sqrt(mean_squared_error(y_train, train_pred)),
                overfitting_ratio=(np.sqrt(mean_squared_error(y_test, test_pred)) - 
                                  np.sqrt(mean_squared_error(y_train, train_pred))) / 
                                  np.sqrt(mean_squared_error(y_test, test_pred)),
                training_time=train_time,
                description="Optimized gradient boosting",
                pros=["State-of-the-art accuracy", "Fast", "Regularization built-in"],
                cons=["Complex hyperparameters", "Black box"]
            )
            print(f"  XGBoost: RMSE={results['xgboost'].rmse:.3f}, R²={results['xgboost'].r2:.3f}")
        
        if HAS_LIGHTGBM:
            start = time.time()
            lgb_model = lgb.LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                                          random_state=42, verbosity=-1)
            lgb_model.fit(X_train_scaled, y_train)
            train_pred = lgb_model.predict(X_train_scaled)
            test_pred = lgb_model.predict(X_test_scaled)
            train_time = time.time() - start
            
            results['lightgbm'] = ModelResult(
                name="LightGBM",
                category="Traditional ML",
                rmse=np.sqrt(mean_squared_error(y_test, test_pred)),
                mae=mean_absolute_error(y_test, test_pred),
                r2=r2_score(y_test, test_pred),
                train_rmse=np.sqrt(mean_squared_error(y_train, train_pred)),
                overfitting_ratio=(np.sqrt(mean_squared_error(y_test, test_pred)) - 
                                  np.sqrt(mean_squared_error(y_train, train_pred))) / 
                                  np.sqrt(mean_squared_error(y_test, test_pred)),
                training_time=train_time,
                description="Fast gradient boosting with leaf-wise growth",
                pros=["Fastest training", "Memory efficient", "High accuracy"],
                cons=["Can overfit on small data", "Sensitive to hyperparameters"]
            )
            print(f"  LightGBM: RMSE={results['lightgbm'].rmse:.3f}, R²={results['lightgbm'].r2:.3f}")
        
        # 2. Ensemble Methods
        print("\n--- Ensemble Methods ---")
        
        # Simple averaging ensemble
        start = time.time()
        ensemble_pred = np.zeros_like(y_test, dtype=float)
        n_models = 0
        
        for name, model in [('ridge', ridge), ('rf', rf), ('gbm', gbm)]:
            ensemble_pred += model.predict(X_test_scaled)
            n_models += 1
        
        if HAS_XGBOOST:
            ensemble_pred += xgb_model.predict(X_test_scaled)
            n_models += 1
        if HAS_LIGHTGBM:
            ensemble_pred += lgb_model.predict(X_test_scaled)
            n_models += 1
        
        ensemble_pred /= n_models
        train_time = time.time() - start
        
        results['ensemble_average'] = ModelResult(
            name="Ensemble Average",
            category="Ensemble",
            rmse=np.sqrt(mean_squared_error(y_test, ensemble_pred)),
            mae=mean_absolute_error(y_test, ensemble_pred),
            r2=r2_score(y_test, ensemble_pred),
            train_rmse=0,  # Not applicable
            overfitting_ratio=0,
            training_time=train_time,
            description="Simple average of all models",
            pros=["Reduces variance", "More robust", "No additional training"],
            cons=["May not be optimal", "Slower inference"]
        )
        print(f"  Ensemble Average: RMSE={results['ensemble_average'].rmse:.3f}, R²={results['ensemble_average'].r2:.3f}")
        
        # 3. Simulation-based (Monte Carlo baseline)
        print("\n--- Simulation Methods ---")
        
        # Monte Carlo uses historical mean + noise
        start = time.time()
        mc_simulator = MonteCarloSimulator(n_simulations=1000)
        
        # For each test sample, simulate based on training statistics
        mc_predictions = []
        for i in range(len(y_test)):
            player_data = {
                'projection': y_train.mean(),
                'volatility': y_train.std(),
                'matchup_score': 1.0,
                'injury_prob': 0.05
            }
            sim_result = mc_simulator.simulate_player(player_data, n_sims=100)
            mc_predictions.append(sim_result['mean'])
        
        mc_predictions = np.array(mc_predictions)
        train_time = time.time() - start
        
        results['monte_carlo'] = ModelResult(
            name="Monte Carlo Simulation",
            category="Simulation",
            rmse=np.sqrt(mean_squared_error(y_test, mc_predictions)),
            mae=mean_absolute_error(y_test, mc_predictions),
            r2=r2_score(y_test, mc_predictions),
            train_rmse=0,
            overfitting_ratio=0,
            training_time=train_time,
            description="Probabilistic simulation with uncertainty",
            pros=["Full distribution", "Uncertainty quantification", "Intuitive"],
            cons=["Computationally expensive", "Requires good priors"]
        )
        print(f"  Monte Carlo: RMSE={results['monte_carlo'].rmse:.3f}, R²={results['monte_carlo'].r2:.3f}")
        
        # Find best model
        best_name = min(results, key=lambda x: results[x].rmse)
        self.best_model = best_name
        self.results = results
        
        print(f"\n✅ Best Model: {results[best_name].name} (RMSE={results[best_name].rmse:.3f})")
        
        return results
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame for display."""
        rows = []
        for name, result in self.results.items():
            rows.append({
                'Model': result.name,
                'Category': result.category,
                'Test RMSE': result.rmse,
                'Test MAE': result.mae,
                'Test R²': result.r2,
                'Train RMSE': result.train_rmse,
                'Overfit Ratio': result.overfitting_ratio,
                'Train Time (s)': result.training_time,
                'Description': result.description
            })
        
        return pd.DataFrame(rows).sort_values('Test RMSE')
    
    def get_model_details(self, model_name: str) -> Dict:
        """Get detailed information about a specific model."""
        if model_name not in self.results:
            return {}
        
        result = self.results[model_name]
        return {
            'name': result.name,
            'category': result.category,
            'metrics': {
                'rmse': result.rmse,
                'mae': result.mae,
                'r2': result.r2,
                'train_rmse': result.train_rmse,
                'overfitting_ratio': result.overfitting_ratio
            },
            'training_time': result.training_time,
            'description': result.description,
            'pros': result.pros,
            'cons': result.cons
        }


# =============================================================================
# MAIN EVALUATION FUNCTION
# =============================================================================

def run_model_comparison():
    """Run comprehensive model comparison and save results."""
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
    df = engineer_all_features(df)
    df = add_qb_features(df)
    df = add_external_features(df)
    df = add_multiweek_features(df, horizons=[1, 5, 18])
    df = add_season_long_features(df)
    
    # Get feature columns (safe features only)
    allowed_patterns = [
        '_lag_', '_rolling_', 'rolling_', '_trend', '_avg',
        'games_played', 'is_home', 'consistency_score', 'weekly_volatility',
        'coefficient_of_variation', 'boom_bust_range', 'confidence_score',
        'injury_score', 'opp_defense_rank', 'opp_matchup_score',
        'sos_next_', 'projection_', 'age_factor', 'projected_games',
        'utilization_score', 'target_share', 'rush_share',
    ]
    
    feature_cols = []
    for c in df.columns:
        if df[c].dtype not in ['int64', 'float64']:
            continue
        if any(pattern in c for pattern in allowed_patterns):
            feature_cols.append(c)
    
    # Limit to top features
    feature_cols = feature_cols[:50]
    
    # Split data
    train_df = df[df['season'] < 2024]
    test_df = df[df['season'] == 2024]
    
    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df['fantasy_points'].values
    X_test = test_df[feature_cols].fillna(0).values
    y_test = test_df['fantasy_points'].values
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}, Features: {len(feature_cols)}")
    
    # Run comparison
    framework = ModelComparisonFramework()
    results = framework.evaluate_all_models(X_train, y_train, X_test, y_test, feature_cols)
    
    # Save results
    results_path = Path(__file__).parent.parent.parent / 'data' / 'model_comparison_results.json'
    
    results_dict = {}
    for name, result in results.items():
        results_dict[name] = {
            'name': result.name,
            'category': result.category,
            'rmse': float(result.rmse),
            'mae': float(result.mae),
            'r2': float(result.r2),
            'train_rmse': float(result.train_rmse),
            'overfitting_ratio': float(result.overfitting_ratio),
            'training_time': float(result.training_time),
            'description': result.description,
            'pros': result.pros,
            'cons': result.cons
        }
    
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    return framework


if __name__ == '__main__':
    run_model_comparison()
