"""Grouped share allocation with an explicit unrepresented-player bucket.

Player utilities are adjusted jointly within each team-week. The softmax
includes one extra bucket for volume assigned to players outside the capped
roster. All features and offsets use lagged information; current-week shares
are read only as training labels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.models.team_allocation.features import VOLUME_COLS

GROUP = ["team", "season", "week"]


class JointOtherShareModel:
    def __init__(self, target: str, *, epsilon: float = 0.001, penalty: float = 0.001,
                 maxiter: int = 200):
        if target not in VOLUME_COLS:
            raise ValueError(f"unsupported share target: {target}")
        if not (np.isfinite(epsilon) and 0 < epsilon < 1):
            raise ValueError("epsilon must be finite in (0, 1)")
        if not (np.isfinite(penalty) and penalty >= 0):
            raise ValueError("penalty must be finite and nonnegative")
        if type(maxiter) is not int or maxiter < 1:
            raise ValueError("maxiter must be a positive integer")
        self.target, self.epsilon, self.penalty, self.maxiter = target, epsilon, penalty, maxiter
        self.fit_diagnostics_: dict = {}
        self._coef: np.ndarray | None = None

    @property
    def label(self):
        return f"share_of_team_{self.target}"

    @property
    def prior(self):
        return f"{self.label}_roll3"

    @property
    def numeric_cols(self):
        return [f"{self.label}_s2d", "roster_snap_share_s2d", "depth_chart_rank", "is_cold_start"]

    def _validate(self, frame: pd.DataFrame, *, training: bool):
        required = set(GROUP + ["player_id", "slot", self.prior, *self.numeric_cols])
        if training:
            required.update([self.label, f"team_{self.target}"])
        missing = required - set(frame)
        if frame.empty or frame.columns.duplicated().any() or missing:
            raise ValueError(f"empty or malformed share panel; missing {sorted(missing)}")
        key = ["player_id", "season", "week", "team"]
        if frame[key].isna().any().any() or frame.duplicated(key).any():
            raise ValueError("null or duplicate player/team/week identity")
        if frame["slot"].isna().any() or not frame.slot.map(lambda v: isinstance(v, str) and bool(v.strip())).all():
            raise ValueError("invalid roster slot")
        for col in [self.prior, *self.numeric_cols]:
            values = pd.to_numeric(frame[col], errors="raise").to_numpy(float)
            if np.isinf(values).any():
                raise ValueError(f"nonfinite feature: {col}")
        prior = pd.to_numeric(frame[self.prior], errors="raise").to_numpy(float)
        if ((prior[np.isfinite(prior)] < 0) | (prior[np.isfinite(prior)] > 1)).any():
            raise ValueError("lagged shares must be in [0, 1]")
        if training:
            y = pd.to_numeric(frame[self.label], errors="raise").to_numpy(float)
            total = pd.to_numeric(frame[f"team_{self.target}"], errors="raise").to_numpy(float)
            if not np.isfinite(y).all() or ((y < 0) | (y > 1)).any():
                raise ValueError("training share labels must be finite in [0, 1]")
            if not np.isfinite(total).all() or (total < 0).any():
                raise ValueError("training team totals must be finite and nonnegative")
            if frame.groupby(GROUP)[f"team_{self.target}"].nunique().gt(1).any():
                raise ValueError("team total disagrees within a team-week")

    @staticmethod
    def _groups(frame: pd.DataFrame) -> tuple[np.ndarray, int]:
        index = pd.MultiIndex.from_frame(frame[GROUP])
        group, unique = pd.factorize(index, sort=False)
        return group.astype(int), len(unique)

    def _prepare(self, frame: pd.DataFrame, *, fit: bool):
        prior = pd.to_numeric(frame[self.prior], errors="raise").fillna(0).to_numpy(float)
        group, n_groups = self._groups(frame)
        prior_mass = np.bincount(group, weights=prior, minlength=n_groups)
        other_prior = np.maximum(1.0 - prior_mass, 0.0)
        log_player = np.log(prior + self.epsilon)
        log_other = np.log(other_prior + self.epsilon)
        numeric = frame[self.numeric_cols].apply(pd.to_numeric, errors="raise").to_numpy(float)
        # The log offset can be corrected by a train-only fitted coefficient.
        numeric = np.column_stack([numeric, log_player])
        if fit:
            self._median = pd.DataFrame(numeric).median().fillna(0.0).to_numpy(float)
            self._slot_categories = sorted(frame.slot.unique())
        numeric = np.where(np.isnan(numeric), self._median, numeric)
        if fit:
            self._mean = numeric.mean(axis=0)
            self._scale = numeric.std(axis=0)
            self._scale[self._scale == 0] = 1.0
            self._other_mean = float(log_other.mean())
            self._other_scale = float(log_other.std()) or 1.0
        z = (numeric - self._mean) / self._scale
        slots = np.zeros((len(frame), max(len(self._slot_categories) - 1, 0)), dtype=float)
        slot_index = {value: i - 1 for i, value in enumerate(self._slot_categories) if i > 0}
        for row, slot in enumerate(frame.slot):
            if slot in slot_index:
                slots[row, slot_index[slot]] = 1.0
        player_x = np.column_stack([np.ones(len(frame)), slots, z])
        other_x = np.column_stack([np.ones(n_groups), (log_other - self._other_mean) / self._other_scale])
        return group, player_x, other_x, log_player, log_other

    @staticmethod
    def _probabilities(group, player_utility, other_utility):
        n_groups = len(other_utility)
        maximum = other_utility.copy()
        np.maximum.at(maximum, group, player_utility)
        p_exp = np.exp(player_utility - maximum[group])
        o_exp = np.exp(other_utility - maximum)
        denom = np.bincount(group, weights=p_exp, minlength=n_groups) + o_exp
        log_denom = np.log(denom)
        log_player = player_utility - maximum[group] - log_denom[group]
        log_other = other_utility - maximum - log_denom
        return p_exp / denom[group], o_exp / denom, log_player, log_other

    def _predict_components(self, prepared, coef):
        group, player_x, other_x, log_player, log_other = prepared
        split = player_x.shape[1]
        player_utility = log_player + player_x @ coef[:split]
        other_utility = log_other + other_x @ coef[split:]
        return self._probabilities(group, player_utility, other_utility)

    def fit(self, frame: pd.DataFrame):
        self._coef = None
        self._validate(frame, training=True)
        # A zero-total team-week has no share composition to learn from.
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
        n_player_coef = prepared[1].shape[1]
        n_params = n_player_coef + prepared[2].shape[1]
        def objective(coef):
            player_p, other_p, log_player, log_other = self._predict_components(prepared, coef)
            loss = -(np.dot(y, log_player) + np.dot(other_y, log_other)) / n_groups
            grad_player = prepared[1].T @ (player_p - y) / n_groups
            grad_other = prepared[2].T @ (other_p - other_y) / n_groups
            regularized = coef.copy()
            regularized[0] = 0.0  # preserve group-level intercept flexibility
            regularized[n_player_coef] = 0.0
            return (loss + self.penalty * np.dot(regularized, regularized) / 2,
                    np.r_[grad_player, grad_other] + self.penalty * regularized)
        result = minimize(objective, np.zeros(n_params), method="L-BFGS-B", jac=True,
                          options={"maxiter": self.maxiter, "ftol": 1e-10})
        self.fit_diagnostics_ = {"converged": bool(result.success), "message": str(result.message),
                                 "iterations": int(result.nit), "objective": float(result.fun),
                                 "n_player_rows": len(active), "n_team_weeks": n_groups,
                                 "zero_total_rows_excluded_from_fit": int(len(frame) - len(active)),
                                 "n_parameters": n_params, "epsilon": self.epsilon,
                                 "penalty": self.penalty}
        if not result.success or not np.isfinite(result.x).all():
            raise RuntimeError(f"joint allocation did not converge: {self.fit_diagnostics_}")
        self._coef = result.x
        return self

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._coef is None:
            raise RuntimeError("call fit() before predict()")
        self._validate(frame, training=False)
        prepared = self._prepare(frame, fit=False)
        candidate, other, _, _ = self._predict_components(prepared, self._coef)
        prior, _, _, _ = self._predict_components(prepared, np.zeros_like(self._coef))
        if not (np.isfinite(candidate).all() and np.isfinite(other).all()
                and np.isfinite(prior).all()):
            raise RuntimeError("nonfinite joint allocation")
        group = prepared[0]
        if not np.allclose(np.bincount(group, weights=candidate, minlength=len(other)) + other, 1.0, atol=1e-10):
            raise AssertionError("joint allocation violates the other-mass conservation constraint")
        return candidate, prior, other
