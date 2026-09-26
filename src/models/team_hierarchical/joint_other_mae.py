"""Joint share allocation fitted to a smooth player-week absolute error.

The grouped softmax and explicit unrepresented-player bucket are inherited
from the likelihood model. Only the training objective changes. In particular,
the bucket is a conservation constraint, not an additional scored player row.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.models.team_hierarchical.joint_other import JointOtherShareModel


class JointOtherMAEModel(JointOtherShareModel):
    def __init__(self, target: str, *, epsilon: float = 0.001,
                 penalty: float = 0.00001, smooth_width: float = 0.002,
                 maxiter: int = 300):
        super().__init__(target, epsilon=epsilon, penalty=penalty, maxiter=maxiter)
        if not (np.isfinite(smooth_width) and smooth_width > 0):
            raise ValueError("smooth_width must be finite and positive")
        self.smooth_width = smooth_width

    def _objective(self, prepared, y: np.ndarray, other_y: np.ndarray,
                   coef: np.ndarray):
        group, player_x, other_x, _, _ = prepared
        n_rows = len(y)
        n_groups = len(other_x)
        split = player_x.shape[1]
        player_p, other_p, _, _ = self._predict_components(prepared, coef)
        residual = player_p - y
        smooth = np.sqrt(residual * residual + self.smooth_width * self.smooth_width)
        other_residual = other_p - other_y
        other_smooth = np.sqrt(other_residual * other_residual
                               + self.smooth_width * self.smooth_width)
        # The omitted-player bucket is an actual team share. Its mean absolute
        # error receives the same weight as mean player error so fitting cannot
        # improve sparse rows by dumping represented volume into that bucket.
        loss = float((smooth - self.smooth_width).mean()
                     + (other_smooth - self.smooth_width).mean())
        derivative = residual / smooth / n_rows
        other_derivative = other_residual / other_smooth / n_groups
        group_derivative = np.bincount(group, weights=derivative * player_p,
                                       minlength=n_groups) + other_derivative * other_p
        utility_player = player_p * (derivative - group_derivative[group])
        utility_other = other_p * (other_derivative - group_derivative)
        regularized = coef.copy()
        regularized[0] = 0.0
        regularized[split] = 0.0
        gradient = np.r_[player_x.T @ utility_player,
                         other_x.T @ utility_other] + self.penalty * regularized
        return loss + self.penalty * float(regularized @ regularized) / 2, gradient

    def fit(self, frame: pd.DataFrame):
        self._coef = None
        self._validate(frame, training=True)
        team_total = pd.to_numeric(frame[f"team_{self.target}"], errors="raise")
        active = frame.loc[team_total > 0].reset_index(drop=True)
        if active.empty:
            raise ValueError("no positive-volume team-weeks in training")
        group, n_groups = self._groups(active)
        y = active[self.label].to_numpy(float)
        mass = np.bincount(group, weights=y, minlength=n_groups)
        if (mass > 1 + 1e-8).any():
            raise ValueError("represented player shares exceed one in a training team-week")
        other_y = np.maximum(1.0 - mass, 0.0)
        prepared = self._prepare(active, fit=True)
        n_params = prepared[1].shape[1] + prepared[2].shape[1]
        result = minimize(lambda coef: self._objective(prepared, y, other_y, coef),
                          np.zeros(n_params), method="L-BFGS-B", jac=True,
                          options={"maxiter": self.maxiter, "ftol": 1e-11})
        self.fit_diagnostics_ = {
            "converged": bool(result.success), "message": str(result.message),
            "iterations": int(result.nit), "objective": float(result.fun),
            "n_player_rows": len(active), "n_team_weeks": n_groups,
            "zero_total_rows_excluded_from_fit": int(len(frame) - len(active)),
            "n_parameters": n_params, "epsilon": self.epsilon,
            "penalty": self.penalty, "smooth_width": self.smooth_width,
            "objective_definition": "mean_player_plus_mean_other_smooth_absolute_share_error",
        }
        if not result.success or not np.isfinite(result.x).all():
            raise RuntimeError(f"joint MAE allocation did not converge: {self.fit_diagnostics_}")
        self._coef = result.x
        return self
