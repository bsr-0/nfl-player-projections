"""
Bayesian Hierarchical Models for NFL Player Prediction

This module implements Bayesian hierarchical models that provide:
- Proper uncertainty quantification (epistemic + aleatoric)
- Shrinkage for rookies and players with limited data
- Player-specific random effects that capture individual tendencies
- Posterior updates as the season progresses

Mathematical Formulation:
    y_{i,t} ~ Normal(μ_{i,t}, σ²_{position})
    μ_{i,t} = α_{position} + β × X_{i,t} + γ_i
    γ_i ~ Normal(0, τ²)  # Player random effect
    
Where:
    - y_{i,t}: Fantasy points for player i in week t
    - α_{position}: Position-specific intercept
    - β: Feature coefficients (shared across players)
    - γ_i: Player-specific random effect (shrunk toward 0)
    - τ²: Between-player variance

References:
    - Efron & Morris (1977): James-Stein estimation
    - Glickman & Stern (1998): Bayesian hierarchical models for sports
    - Gelman et al. (2013): Bayesian Data Analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import POSITIONS, MODELS_DIR

logger = logging.getLogger(__name__)

# Try to import PyMC for full Bayesian inference
try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    logger.info(
        "PyMC not installed; using empirical Bayes approximation. "
        "Install pymc+arviz for full Bayesian inference."
    )


class BayesianPlayerModel:
    """
    Bayesian hierarchical model for player performance prediction.
    
    Uses player random effects to capture individual tendencies while
    shrinking estimates toward position means for players with limited data.
    
    This implements the James-Stein shrinkage estimator when PyMC is not
    available, and full MCMC inference when it is.
    """
    
    def __init__(
        self, 
        position: str,
        use_full_bayes: bool = True,
        n_samples: int = 2000,
        n_chains: int = 4
    ):
        """
        Initialize Bayesian player model.
        
        Args:
            position: Player position (QB, RB, WR, TE)
            use_full_bayes: If True and PyMC available, use MCMC. 
                           Otherwise use empirical Bayes approximation.
            n_samples: Number of MCMC samples per chain
            n_chains: Number of MCMC chains
        """
        self.position = position
        self.use_full_bayes = use_full_bayes and PYMC_AVAILABLE
        self.n_samples = n_samples
        self.n_chains = n_chains
        
        # Model components
        self.player_effects: Dict[str, float] = {}
        self.player_effect_std: Dict[str, float] = {}
        self.global_mean: float = 0.0
        self.global_std: float = 1.0
        self.between_player_std: float = 1.0  # τ
        self.within_player_std: float = 5.0   # σ (position-specific)
        self.feature_coefs: Dict[str, float] = {}
        
        # Fitted model artifacts
        self.trace = None
        self.is_fitted = False
        
        # Position-specific priors for within-player std
        self._position_std_priors = {
            'QB': {'mean': 6.0, 'std': 2.0},
            'RB': {'mean': 5.5, 'std': 2.5},
            'WR': {'mean': 5.0, 'std': 2.0},
            'TE': {'mean': 4.0, 'std': 1.5},
        }
    
    def fit(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        player_ids: pd.Series,
        feature_names: List[str] = None
    ) -> 'BayesianPlayerModel':
        """
        Fit the Bayesian hierarchical model.
        
        Args:
            X: Feature DataFrame
            y: Target variable (fantasy points)
            player_ids: Player ID for each observation
            feature_names: Names of features to use
            
        Returns:
            self
        """
        feature_names = feature_names or list(X.columns)
        
        # Calculate global statistics
        self.global_mean = y.mean()
        self.global_std = y.std()
        
        if self.use_full_bayes:
            self._fit_pymc(X, y, player_ids, feature_names)
        else:
            self._fit_empirical_bayes(X, y, player_ids, feature_names)
        
        self.is_fitted = True
        return self
    
    def _fit_empirical_bayes(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        player_ids: pd.Series,
        feature_names: List[str]
    ):
        """
        Fit using empirical Bayes / James-Stein shrinkage.
        
        This is a fast approximation when PyMC is not available.
        Uses the James-Stein estimator for player random effects.
        """
        print(f"Fitting {self.position} model with empirical Bayes...")
        
        # Step 1: Fit simple linear model for feature coefficients
        from sklearn.linear_model import Ridge
        
        ridge = Ridge(alpha=1.0)
        ridge.fit(X[feature_names], y)
        self.feature_coefs = dict(zip(feature_names, ridge.coef_))
        
        # Step 2: Calculate residuals
        y_pred_fixed = ridge.predict(X[feature_names])
        residuals = y - y_pred_fixed
        
        # Step 3: Calculate player-specific effects with shrinkage
        player_df = pd.DataFrame({
            'player_id': player_ids,
            'residual': residuals
        })
        
        player_stats = player_df.groupby('player_id')['residual'].agg(['mean', 'count', 'std'])
        player_stats.columns = ['raw_effect', 'n_games', 'player_std']
        
        # Estimate between-player variance (τ²)
        # Using method of moments
        overall_var = player_stats['raw_effect'].var()
        avg_within_var = (player_stats['player_std'] ** 2).mean()
        avg_n = player_stats['n_games'].mean()
        
        # τ² = var(player_means) - σ²/n
        self.between_player_std = np.sqrt(max(0, overall_var - avg_within_var / avg_n))
        
        # Within-player std (σ) - use position-specific prior
        prior = self._position_std_priors.get(self.position, {'mean': 5.0, 'std': 2.0})
        empirical_std = player_stats['player_std'].mean()
        self.within_player_std = (prior['mean'] + empirical_std) / 2  # Compromise
        
        # Step 4: Apply James-Stein shrinkage to player effects
        for player_id, row in player_stats.iterrows():
            n = row['n_games']
            raw_effect = row['raw_effect']
            
            # Shrinkage factor: B = τ² / (τ² + σ²/n)
            # Closer to 1 = trust the data more
            # Closer to 0 = shrink toward global mean
            if self.between_player_std > 0:
                shrinkage = (self.between_player_std ** 2) / (
                    self.between_player_std ** 2 + 
                    self.within_player_std ** 2 / n
                )
            else:
                shrinkage = 0
            
            # Shrunken estimate
            shrunken_effect = shrinkage * raw_effect
            
            # Posterior standard deviation
            posterior_var = (
                (1 / (1 / self.between_player_std ** 2 + n / self.within_player_std ** 2))
                if self.between_player_std > 0 else self.within_player_std ** 2 / n
            )
            
            self.player_effects[player_id] = shrunken_effect
            self.player_effect_std[player_id] = np.sqrt(max(posterior_var, 0.1))
        
        print(f"  Fitted {len(self.player_effects)} player effects")
        print(f"  Between-player std (τ): {self.between_player_std:.2f}")
        print(f"  Within-player std (σ): {self.within_player_std:.2f}")
    
    def _fit_pymc(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        player_ids: pd.Series,
        feature_names: List[str]
    ):
        """
        Fit using full Bayesian inference with PyMC.
        
        This provides proper posterior distributions for all parameters.
        """
        print(f"Fitting {self.position} model with PyMC (MCMC)...")
        
        # Encode player IDs as integers
        unique_players = player_ids.unique()
        player_idx = pd.Categorical(player_ids, categories=unique_players).codes
        n_players = len(unique_players)
        
        # Standardize features
        X_np = X[feature_names].values
        X_mean = X_np.mean(axis=0)
        X_std = X_np.std(axis=0) + 1e-8
        X_standardized = (X_np - X_mean) / X_std
        
        y_np = y.values
        
        # Get position-specific prior
        prior = self._position_std_priors.get(self.position, {'mean': 5.0, 'std': 2.0})
        
        with pm.Model() as hierarchical_model:
            # Hyperpriors
            tau = pm.HalfNormal('tau', sigma=prior['std'])  # Between-player std
            sigma = pm.HalfNormal('sigma', sigma=prior['mean'])  # Within-player std
            
            # Global intercept
            alpha = pm.Normal('alpha', mu=self.global_mean, sigma=5)
            
            # Feature coefficients (shared across players)
            beta = pm.Normal('beta', mu=0, sigma=1, shape=len(feature_names))
            
            # Player random effects
            gamma = pm.Normal('gamma', mu=0, sigma=tau, shape=n_players)
            
            # Expected value
            mu = alpha + pm.math.dot(X_standardized, beta) + gamma[player_idx]
            
            # Likelihood
            y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_np)
            
            # Inference
            self.trace = pm.sample(
                self.n_samples, 
                chains=self.n_chains,
                return_inferencedata=True,
                progressbar=True
            )
        
        # Extract posterior means
        posterior = self.trace.posterior
        
        self.between_player_std = float(posterior['tau'].mean())
        self.within_player_std = float(posterior['sigma'].mean())
        
        # Extract feature coefficients (unstandardize)
        beta_means = posterior['beta'].mean(dim=['chain', 'draw']).values
        for i, name in enumerate(feature_names):
            self.feature_coefs[name] = float(beta_means[i] / X_std[i])
        
        # Extract player effects
        gamma_means = posterior['gamma'].mean(dim=['chain', 'draw']).values
        gamma_stds = posterior['gamma'].std(dim=['chain', 'draw']).values
        
        for i, player_id in enumerate(unique_players):
            self.player_effects[player_id] = float(gamma_means[i])
            self.player_effect_std[player_id] = float(gamma_stds[i])
        
        print(f"  MCMC complete: {self.n_samples} samples × {self.n_chains} chains")
        print(f"  Between-player std (τ): {self.between_player_std:.2f}")
        print(f"  Within-player std (σ): {self.within_player_std:.2f}")
    
    def predict(
        self, 
        X: pd.DataFrame, 
        player_ids: pd.Series
    ) -> np.ndarray:
        """
        Make point predictions.
        
        Args:
            X: Feature DataFrame
            player_ids: Player IDs for each row
            
        Returns:
            Array of predicted fantasy points
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = np.zeros(len(X))
        
        # Calculate fixed effects contribution
        for feat, coef in self.feature_coefs.items():
            if feat in X.columns:
                predictions += X[feat].values * coef
        
        # Add global mean
        predictions += self.global_mean
        
        # Add player random effects
        for i, player_id in enumerate(player_ids):
            if player_id in self.player_effects:
                predictions[i] += self.player_effects[player_id]
            # If unknown player, effect is 0 (shrunk to position mean)
        
        return predictions
    
    def predict_with_uncertainty(
        self, 
        X: pd.DataFrame, 
        player_ids: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimates.
        
        Returns both point predictions and standard deviations that
        combine epistemic (model) and aleatoric (inherent) uncertainty.
        
        Args:
            X: Feature DataFrame
            player_ids: Player IDs for each row
            
        Returns:
            Tuple of (predictions, standard_deviations)
        """
        predictions = self.predict(X, player_ids)
        
        # Calculate uncertainty for each prediction
        uncertainties = np.zeros(len(X))
        
        for i, player_id in enumerate(player_ids):
            # Aleatoric uncertainty (inherent variability)
            aleatoric_var = self.within_player_std ** 2
            
            # Epistemic uncertainty (uncertainty about player effect)
            if player_id in self.player_effect_std:
                epistemic_var = self.player_effect_std[player_id] ** 2
            else:
                # Unknown player: use between-player variance as prior
                epistemic_var = self.between_player_std ** 2
            
            # Total uncertainty (combined)
            total_var = aleatoric_var + epistemic_var
            uncertainties[i] = np.sqrt(total_var)
        
        return predictions, uncertainties
    
    def predict_distribution(
        self, 
        X: pd.DataFrame, 
        player_ids: pd.Series,
        n_samples: int = 1000
    ) -> np.ndarray:
        """
        Sample from the predictive distribution.
        
        This provides full distributional predictions, useful for:
        - DFS optimization with uncertainty
        - Risk analysis
        - Scenario modeling
        
        Args:
            X: Feature DataFrame
            player_ids: Player IDs for each row
            n_samples: Number of samples to draw
            
        Returns:
            Array of shape (n_samples, n_observations)
        """
        mean, std = self.predict_with_uncertainty(X, player_ids)
        
        # Draw samples from Gaussian predictive distribution
        samples = np.random.normal(
            loc=mean.reshape(1, -1),
            scale=std.reshape(1, -1),
            size=(n_samples, len(X))
        )
        
        # Fantasy points can't be negative
        samples = np.maximum(samples, 0)
        
        return samples
    
    def get_player_shrinkage(self, player_id: str) -> Dict:
        """
        Get shrinkage information for a specific player.
        
        Returns the degree to which this player's estimate has been
        shrunk toward the position mean.
        
        Args:
            player_id: Player identifier
            
        Returns:
            Dict with shrinkage information
        """
        if player_id not in self.player_effects:
            return {
                'player_id': player_id,
                'effect': 0.0,
                'effect_std': self.between_player_std,
                'shrinkage': 1.0,  # Fully shrunk (unknown player)
                'interpretation': 'Unknown player - using position average'
            }
        
        effect = self.player_effects[player_id]
        effect_std = self.player_effect_std[player_id]
        
        # Shrinkage = 1 - (posterior_std / prior_std)²
        # Higher shrinkage = more regularization toward mean
        prior_std = self.between_player_std
        if prior_std > 0:
            shrinkage = 1 - (effect_std / prior_std) ** 2
        else:
            shrinkage = 0
        
        if shrinkage > 0.7:
            interpretation = 'High shrinkage - limited data, estimates uncertain'
        elif shrinkage > 0.3:
            interpretation = 'Moderate shrinkage - reasonable confidence'
        else:
            interpretation = 'Low shrinkage - strong individual signal'
        
        return {
            'player_id': player_id,
            'effect': effect,
            'effect_std': effect_std,
            'shrinkage': shrinkage,
            'interpretation': interpretation
        }
    
    def get_rookie_prior(self) -> Dict:
        """
        Get the prior distribution for rookie players.
        
        Rookies have no historical data, so their predictions are based
        entirely on the position-level prior.
        
        Returns:
            Dict with prior parameters
        """
        return {
            'position': self.position,
            'mean': self.global_mean,
            'std': np.sqrt(self.between_player_std ** 2 + self.within_player_std ** 2),
            'interpretation': (
                f'Rookie {self.position}s are predicted at position average '
                f'({self.global_mean:.1f} pts) with high uncertainty.'
            )
        }
    
    def save(self, filepath: Path = None):
        """Save model to disk."""
        import joblib
        
        filepath = filepath or MODELS_DIR / f"bayesian_{self.position.lower()}.joblib"
        
        model_data = {
            'position': self.position,
            'player_effects': self.player_effects,
            'player_effect_std': self.player_effect_std,
            'global_mean': self.global_mean,
            'global_std': self.global_std,
            'between_player_std': self.between_player_std,
            'within_player_std': self.within_player_std,
            'feature_coefs': self.feature_coefs,
            'is_fitted': self.is_fitted,
        }
        
        joblib.dump(model_data, filepath)
        print(f"Saved Bayesian model to {filepath}")
    
    @classmethod
    def load(cls, position: str, filepath: Path = None) -> 'BayesianPlayerModel':
        """Load model from disk."""
        import joblib
        
        filepath = filepath or MODELS_DIR / f"bayesian_{position.lower()}.joblib"
        model_data = joblib.load(filepath)
        
        model = cls(position=position)
        model.player_effects = model_data['player_effects']
        model.player_effect_std = model_data['player_effect_std']
        model.global_mean = model_data['global_mean']
        model.global_std = model_data['global_std']
        model.between_player_std = model_data['between_player_std']
        model.within_player_std = model_data['within_player_std']
        model.feature_coefs = model_data['feature_coefs']
        model.is_fitted = model_data['is_fitted']
        
        return model


class BayesianEnsemble:
    """
    Ensemble of Bayesian models for all positions.
    
    Coordinates prediction across positions and provides
    unified uncertainty quantification.
    """
    
    def __init__(self, use_full_bayes: bool = True):
        """
        Initialize Bayesian ensemble.
        
        Args:
            use_full_bayes: Whether to use full MCMC (if PyMC available)
        """
        self.models: Dict[str, BayesianPlayerModel] = {}
        self.use_full_bayes = use_full_bayes
        self.is_fitted = False
    
    def fit(
        self, 
        data: pd.DataFrame,
        target_col: str = 'fantasy_points',
        feature_cols: List[str] = None
    ) -> 'BayesianEnsemble':
        """
        Fit models for all positions.
        
        Args:
            data: DataFrame with player data including 'position' column
            target_col: Name of target variable column
            feature_cols: List of feature column names
            
        Returns:
            self
        """
        for position in POSITIONS:
            pos_data = data[data['position'] == position]
            
            if len(pos_data) < 100:
                print(f"Skipping {position}: insufficient data ({len(pos_data)} samples)")
                continue
            
            model = BayesianPlayerModel(
                position=position,
                use_full_bayes=self.use_full_bayes
            )
            
            # Get features (exclude identifiers and target)
            if feature_cols is None:
                exclude_cols = ['player_id', 'name', 'position', 'team',
                               'season', 'week', target_col, 'opponent',
                               'home_away', 'created_at', 'updated_at', 'id',
                               'birth_date', 'college', 'game_id', 'game_time',
                               'player_name', 'gsis_id']
                feature_cols_pos = [c for c in pos_data.columns 
                                   if c not in exclude_cols and not c.startswith('target_')]
            else:
                feature_cols_pos = feature_cols
            
            X = pos_data[feature_cols_pos].fillna(0)
            y = pos_data[target_col]
            player_ids = pos_data['player_id']
            
            model.fit(X, y, player_ids, feature_cols_pos)
            self.models[position] = model
        
        self.is_fitted = True
        return self
    
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions for all players.
        
        Args:
            data: DataFrame with player data
            
        Returns:
            DataFrame with predictions added
        """
        result = data.copy()
        result['predicted_points'] = np.nan
        result['prediction_std'] = np.nan
        
        for position, model in self.models.items():
            mask = result['position'] == position
            if not mask.any():
                continue
            
            pos_data = result[mask]
            
            # Get features used by this model
            feature_cols = list(model.feature_coefs.keys())
            available_cols = [c for c in feature_cols if c in pos_data.columns]
            
            if not available_cols:
                continue
            
            X = pos_data[available_cols].fillna(0)
            player_ids = pos_data['player_id']
            
            preds, stds = model.predict_with_uncertainty(X, player_ids)
            
            result.loc[mask, 'predicted_points'] = preds
            result.loc[mask, 'prediction_std'] = stds
        
        return result
    
    def save_all(self):
        """Save all position models."""
        for position, model in self.models.items():
            model.save()
    
    def load_all(self):
        """Load all position models."""
        for position in POSITIONS:
            try:
                self.models[position] = BayesianPlayerModel.load(position)
            except FileNotFoundError:
                print(f"No saved model found for {position}")
        
        self.is_fitted = len(self.models) > 0


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    # Example usage with synthetic data
    np.random.seed(42)
    
    # Generate synthetic player data
    n_players = 50
    n_weeks = 10
    
    data = []
    for player_id in range(n_players):
        player_effect = np.random.normal(0, 3)  # Player-specific tendency
        
        for week in range(n_weeks):
            data.append({
                'player_id': f'player_{player_id}',
                'position': np.random.choice(['RB', 'WR']),
                'season': 2024,
                'week': week + 1,
                'utilization_score': np.random.uniform(40, 80),
                'snap_share': np.random.uniform(0.3, 0.9),
                'target_share': np.random.uniform(0.05, 0.25),
                'fantasy_points': 10 + player_effect + np.random.normal(0, 5)
            })
    
    df = pd.DataFrame(data)
    
    print("=" * 60)
    print("BAYESIAN HIERARCHICAL MODEL DEMO")
    print("=" * 60)
    
    # Fit Bayesian model for RB
    rb_model = BayesianPlayerModel('RB', use_full_bayes=False)
    
    rb_data = df[df['position'] == 'RB']
    feature_cols = ['utilization_score', 'snap_share', 'target_share']
    
    rb_model.fit(
        X=rb_data[feature_cols],
        y=rb_data['fantasy_points'],
        player_ids=rb_data['player_id'],
        feature_names=feature_cols
    )
    
    # Make predictions
    preds, stds = rb_model.predict_with_uncertainty(
        rb_data[feature_cols].head(5),
        rb_data['player_id'].head(5)
    )
    
    print("\nSample predictions:")
    for i in range(5):
        print(f"  Player {rb_data['player_id'].iloc[i]}: "
              f"{preds[i]:.1f} ± {stds[i]:.1f} pts")
    
    # Show shrinkage for a player
    sample_player = rb_data['player_id'].iloc[0]
    shrinkage_info = rb_model.get_player_shrinkage(sample_player)
    print(f"\nShrinkage for {sample_player}:")
    print(f"  Effect: {shrinkage_info['effect']:.2f}")
    print(f"  Shrinkage: {shrinkage_info['shrinkage']:.1%}")
    print(f"  Interpretation: {shrinkage_info['interpretation']}")
    
    # Rookie prior
    rookie_prior = rb_model.get_rookie_prior()
    print(f"\nRookie prior:")
    print(f"  {rookie_prior['interpretation']}")
    
    print("\n" + "=" * 60)
    print("✅ Bayesian model implementation complete!")
    print("=" * 60)
