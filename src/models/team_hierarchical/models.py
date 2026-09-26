"""Plan B's mixed-effects baseline: roster fixed effects and player intercepts.

Known-player random effects are added explicitly because MixedLM.predict only
returns fixed effects. This baseline is not a jointly constrained team model.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.linalg import qr
import statsmodels.formula.api as smf


class MixedEffectsFitError(RuntimeError):
    """No usable converged fit; callers must not score a hidden substitute."""


class MixedEffectsShareModel:
    def __init__(self, target_col: str = "y", reml: bool = True,
                 failure_policy: str = "raise", maxiter: int = 200):
        if failure_policy not in {"raise", "mean"}:
            raise ValueError("failure_policy must be 'raise' or explicitly 'mean'")
        if not isinstance(maxiter, int) or maxiter < 1:
            raise ValueError("maxiter must be positive")
        self.target_col, self.reml = target_col, reml
        self.failure_policy, self.maxiter = failure_policy, maxiter
        self.result_ = None
        self.converged_ = None
        self._slot_categories = None
        self._fixed_effect_cols = []
        self._fallback_mean_ = None
        self._random_effects = {}
        self.fit_diagnostics_ = {}
        self.prediction_diagnostics_ = {}

    @staticmethod
    def _check_identity(X):
        if X.columns.duplicated().any():
            raise ValueError("X has duplicate columns; feature_columns already includes 'slot'")
        if not {"player_id", "slot"}.issubset(X):
            raise ValueError("X must include player_id and slot")
        for col in ("player_id", "slot"):
            if X[col].isna().any() or not X[col].map(lambda v: isinstance(v, str) and bool(v.strip())).all():
                raise ValueError(f"missing or invalid {col}")

    def _formula(self):
        terms = [f"Q({col!r})" for col in self._fixed_effect_cols if col != "slot"]
        return f"Q({self.target_col!r}) ~ " + " + ".join(["C(slot)"] + terms)

    def _failed(self, reason):
        self.result_, self.converged_ = None, False
        self._random_effects = {}
        self.fit_diagnostics_.update(converged=False, failure_reason=reason,
                                     fallback=self.failure_policy == "mean")
        if self.failure_policy == "raise":
            raise MixedEffectsFitError(reason)
        return self

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "MixedEffectsShareModel":
        # An invalid refit must not leave the previous fit usable by accident.
        self.result_, self.converged_, self._slot_categories = None, False, None
        self._random_effects = {}
        self.fit_diagnostics_ = {}
        self.prediction_diagnostics_ = {}
        self._check_identity(X)
        y = np.asarray(y, dtype=float)
        if X.empty or y.shape != (len(X),) or not np.isfinite(y).all():
            raise ValueError("training labels must be finite, nonempty, and align one-to-one with X")
        if self.target_col in X:
            raise ValueError("target_col must not also be an input feature")
        self.result_, self.converged_ = None, False
        self._random_effects = {}
        self._input_cols = list(X)
        self._slot_categories = pd.Index(sorted(X.slot.unique()))
        self._fallback_mean_ = float(y.mean())
        data = X.reset_index(drop=True).copy()
        numeric_cols = [col for col in X if col not in {"slot", "player_id"}]
        numeric = data[numeric_cols].apply(pd.to_numeric, errors="raise").astype(float)
        if np.isinf(numeric.to_numpy()).any():
            raise ValueError("numeric features contain infinity")
        self._medians = numeric.median().fillna(0.0)
        self._fixed_effect_cols = ["slot"]
        self.fit_diagnostics_ = {"n_train": len(X), "n_players": X.player_id.nunique(),
                                 "missing_training_cells": int(numeric.isna().sum().sum()),
                                 "all_missing_columns": numeric.columns[numeric.isna().all()].tolist(),
                                 "dropped_redundant_columns": [], "attempts": []}
        numeric = numeric.fillna(self._medians)
        # C(slot) already spans numeric slot_rank. Residualize numeric columns
        # on slot means and use rank-revealing QR to discard redundant directions.
        # This uses only training data. Missing values use training medians.
        if numeric_cols:
            scales = numeric.std(ddof=0).replace(0, 1)
            normalized = numeric / scales
            residual = normalized - normalized.groupby(data.slot, observed=True).transform("mean")
            _, triangular, pivots = qr(residual.to_numpy(), mode="economic", pivoting=True)
            tolerance = np.finfo(float).eps * max(residual.shape) * max(1.0, float(np.abs(triangular).max(initial=0)))
            rank = int((np.abs(np.diag(triangular)) > tolerance).sum())
            keep = set(pivots[:rank])
            self._fixed_effect_cols += [col for i, col in enumerate(numeric_cols) if i in keep]
            self.fit_diagnostics_["dropped_redundant_columns"] = [col for i, col in enumerate(numeric_cols) if i not in keep]
        data[numeric_cols] = numeric
        data[self.target_col] = y
        data["slot"] = pd.Categorical(data.slot, categories=self._slot_categories)
        n_fixed = len(self._slot_categories) + len(self._fixed_effect_cols) - 1
        if len(X) <= n_fixed or X.player_id.nunique() < 2:
            return self._failed("insufficient rows or player groups for mixed-effects fitting")

        model = smf.mixedlm(self._formula(), data=data, groups=data.player_id, missing="raise")
        # Retry only numerical convergence, never selecting on holdout accuracy.
        for method in ("lbfgs", "powell"):
            attempt = {"method": method, "warnings": []}
            captured = []
            try:
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter("always")
                    fitted = model.fit(reml=self.reml, method=method, maxiter=self.maxiter)
                attempt["warnings"] = [str(w.message) for w in captured]
                if not fitted.converged or not np.isfinite(np.asarray(fitted.params)).all():
                    raise ValueError("optimizer did not converge to finite parameters")
                # Converged results can still have singular random-effect
                # covariance. Extract now, rather than failing during serving.
                effects = {key: float(value.iloc[0]) for key, value in fitted.random_effects.items()}
                if not np.isfinite(list(effects.values())).all():
                    raise ValueError("nonfinite player random effects")
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
                attempt["warnings"] = [str(w.message) for w in captured]
                attempt["error"] = f"{type(exc).__name__}: {exc}"
                self.fit_diagnostics_["attempts"].append(attempt)
                continue
            self.fit_diagnostics_["attempts"].append(attempt)
            self.result_, self.converged_, self._random_effects = fitted, True, effects
            self.fit_diagnostics_.update(converged=True, fallback=False, used_training_rows=len(data),
                                         fixed_effect_columns=list(self._fixed_effect_cols))
            return self
        return self._failed("all mixed-effects optimizer attempts failed; inspect fit_diagnostics_['attempts']")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._slot_categories is None:
            raise RuntimeError("call fit() before predict()")
        if self.result_ is None and self.failure_policy != "mean":
            raise MixedEffectsFitError("the previous fit failed")
        self._check_identity(X)
        missing = set(self._input_cols) - set(X)
        if missing:
            raise ValueError(f"prediction features missing: {sorted(missing)}")
        data = X[self._input_cols].reset_index(drop=True).copy()
        numeric_cols = list(self._medians.index)
        numeric = data[numeric_cols].apply(pd.to_numeric, errors="raise").astype(float)
        if np.isinf(numeric.to_numpy()).any():
            raise ValueError("prediction features contain infinity")
        data[numeric_cols] = numeric.fillna(self._medians)
        unseen = ~data.slot.isin(self._slot_categories)
        self.prediction_diagnostics_ = {"n_rows": len(X), "unseen_players": int((~data.player_id.isin(self._random_effects)).sum()),
                                        "unseen_slots": int(unseen.sum()), "mean_fallback_rows": 0,
                                        "missing_feature_cells": int(numeric.isna().sum().sum())}
        if self.result_ is None:
            if self.failure_policy != "mean":
                raise MixedEffectsFitError("the previous fit failed")
            self.prediction_diagnostics_["mean_fallback_rows"] = len(X)
            return np.full(len(X), self._fallback_mean_)
        if unseen.any():
            fallback = data.loc[unseen, "slot"].str.extract(r"^([A-Za-z]+)")[0] + "1"
            data.loc[unseen, "slot"] = fallback.where(fallback.isin(self._slot_categories), np.nan).to_numpy()
        valid_slot = data.slot.notna()
        self.prediction_diagnostics_["mean_fallback_rows"] = int((~valid_slot).sum())
        data["slot"] = pd.Categorical(data.slot, categories=self._slot_categories)
        fixed = np.full(len(data), self._fallback_mean_, dtype=float)
        if valid_slot.any():
            fixed[valid_slot] = np.asarray(self.result_.predict(exog=data.loc[valid_slot]), dtype=float)
        prediction = fixed + data.player_id.map(self._random_effects).fillna(0).to_numpy(float)
        if not np.isfinite(prediction).all():
            raise MixedEffectsFitError("nonfinite prediction from a converged mixed-effects model")
        return prediction
