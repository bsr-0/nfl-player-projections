"""
Horizon-specific models: 4-week LSTM+ARIMA hybrid, 18-week deep residual feedforward.

- 4-week: LSTM (60%) + ARIMA (40%); injury/workload/EWMA features.
- 18-week: Residual feedforward (2 stages, ~8 effective layers); 70% deep + 30% traditional.

Uses PyTorch with MPS (Apple Silicon GPU) acceleration when available.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import joblib

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import MODELS_DIR, MODEL_CONFIG

# PyTorch for LSTM and deep net (with MPS/GPU support)
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    HAS_TORCH = True
except (ImportError, OSError):
    HAS_TORCH = False

# Legacy alias so train.py imports still work
HAS_TF = HAS_TORCH

# Optional statsmodels for ARIMA
try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_ARIMA = True
except ImportError:
    HAS_ARIMA = False


def _get_device(model_type: str = "feedforward") -> "torch.device":
    """Select best available device: CUDA > MPS (Apple GPU, PyTorch >=2.4) > CPU.

    PyTorch MPS backend has known stability issues with LSTM and very deep
    networks in versions < 2.4. On those versions we fall back to CPU, which
    still benefits from Apple Accelerate / multi-core BLAS on Apple Silicon.
    """
    _ver = tuple(int(x) for x in torch.__version__.split(".")[:2])
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and _ver >= (2, 4):
        return torch.device("mps")
    return torch.device("cpu")


def _season_split(
    n: int,
    seasons: Optional[np.ndarray] = None,
    n_val_seasons: int = 1,
    gap_seasons: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx) splitting by complete seasons.

    Holds out the last ``n_val_seasons`` as validation and optionally
    excludes ``gap_seasons`` seasons immediately before them (purge gap)
    to reduce temporal feature leakage.

    Falls back to a simple last-20% index split when season data is
    absent, insufficient, or produces empty partitions.
    """
    if seasons is None or len(seasons) != n:
        split = int(n * 0.8)
        return np.arange(split), np.arange(split, n)

    unique = sorted(set(seasons))
    if len(unique) < n_val_seasons + 1:
        # Not enough seasons for a meaningful temporal split
        split = int(n * 0.8)
        return np.arange(split), np.arange(split, n)

    val_set = set(unique[-n_val_seasons:])

    # Purge gap: exclude seasons immediately before validation
    gap_end = len(unique) - n_val_seasons
    gap_start = max(0, gap_end - gap_seasons)
    purge_set = set(unique[gap_start:gap_end])

    train_idx = np.where(
        ~np.isin(seasons, list(val_set | purge_set))
    )[0]
    val_idx = np.where(np.isin(seasons, list(val_set)))[0]

    # Fallback if either partition is too small
    if len(train_idx) < 50 or len(val_idx) < 20:
        split = int(n * 0.8)
        return np.arange(split), np.arange(split, n)

    return train_idx, val_idx


# -----------------------------------------------------------------------------
# 4-WEEK: LSTM + ARIMA HYBRID (60% LSTM, 40% ARIMA)
# -----------------------------------------------------------------------------

if HAS_TORCH:
    class _LSTMNet(nn.Module):
        """PyTorch LSTM network: 3 LSTM layers + dense head."""
        def __init__(self, n_features: int, lstm_units: int = 256, dropout: float = 0.25):
            super().__init__()
            u1 = min(256, max(128, lstm_units))
            u2, u3 = 128, 64
            self.lstm1 = nn.LSTM(n_features, u1, batch_first=True)
            self.drop1 = nn.Dropout(dropout)
            self.lstm2 = nn.LSTM(u1, u2, batch_first=True)
            self.drop2 = nn.Dropout(dropout)
            self.lstm3 = nn.LSTM(u2, u3, batch_first=True)
            self.drop3 = nn.Dropout(dropout)
            self.fc1 = nn.Linear(u3, 32)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(32, 1)

        def forward(self, x):
            x, _ = self.lstm1(x)
            x = self.drop1(x)
            x, _ = self.lstm2(x)
            x = self.drop2(x)
            x, _ = self.lstm3(x)
            x = self.drop3(x[:, -1, :])  # last timestep
            x = self.relu(self.fc1(x))
            return self.fc2(x).squeeze(-1)


class LSTM4WeekModel:
    """LSTM for 4-week horizon; 3 LSTM layers, 128-256 units per layer, sequence length 8-12."""
    def __init__(self,
                 sequence_length: int = None,
                 lstm_units: int = None,
                 dropout: float = None,
                 learning_rate: float = None):
        if not HAS_TORCH:
            raise ImportError("PyTorch required for 4-week LSTM. pip install torch")
        self.sequence_length = sequence_length or MODEL_CONFIG.get("lstm_sequence_length", 10)
        self.lstm_units = lstm_units or MODEL_CONFIG.get("lstm_units", 256)
        self.dropout = dropout if dropout is not None else MODEL_CONFIG.get("lstm_dropout", 0.25)
        self.learning_rate = learning_rate or MODEL_CONFIG.get("lstm_learning_rate", 0.001)
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.is_fitted = False
        self.device = _get_device("lstm")

    @staticmethod
    def tune_hyperparameters(X: np.ndarray, y: np.ndarray, player_ids: np.ndarray,
                             feature_names: List[str], n_trials: int = 20,
                             seasons: Optional[np.ndarray] = None) -> Dict:
        """Tune LSTM hyperparameters with Optuna using season-aware splits."""
        try:
            import optuna
            from optuna.samplers import TPESampler
        except ImportError:
            print("  Optuna not available, using default LSTM hyperparameters")
            return {}

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        device = _get_device("lstm")

        def objective(trial):
            seq_len = trial.suggest_int("sequence_length", 6, 14)
            units = trial.suggest_categorical("lstm_units", [128, 192, 256])
            dropout = trial.suggest_float("dropout", 0.15, 0.40)
            lr = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)

            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)

            # Build sequences with season tracking
            X_seq, y_seq, s_seq = [], [], []
            for pid in np.unique(player_ids):
                mask = player_ids == pid
                Xi, yi = Xs[mask], y[mask]
                si = seasons[mask] if seasons is not None else None
                for i in range(len(Xi) - seq_len):
                    X_seq.append(Xi[i:i + seq_len])
                    y_seq.append(yi[i + seq_len])
                    if si is not None:
                        s_seq.append(si[i + seq_len])
            if len(X_seq) < 50:
                return float("inf")
            X_seq, y_seq = np.array(X_seq), np.array(y_seq)
            seq_seasons = np.array(s_seq) if s_seq else None

            # Season-aware train/val split for tuning
            gap = int(MODEL_CONFIG.get("cv_gap_seasons", 1))
            train_idx, val_idx = _season_split(
                len(X_seq), seq_seasons, n_val_seasons=1, gap_seasons=gap
            )
            X_tr = torch.tensor(X_seq[train_idx], dtype=torch.float32)
            y_tr = torch.tensor(y_seq[train_idx], dtype=torch.float32)
            X_va = torch.tensor(X_seq[val_idx], dtype=torch.float32).to(device)
            y_va = torch.tensor(y_seq[val_idx], dtype=torch.float32).to(device)

            model = _LSTMNet(X_seq.shape[2], units, dropout).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.HuberLoss(delta=1.0)

            best_val = float("inf")
            patience, no_improve = 5, 0
            loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=32, shuffle=True)

            for _ in range(30):  # Reduced epochs for tuning speed
                model.train()
                for xb, yb in loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    criterion(model(xb), yb).backward()
                    optimizer.step()
                model.eval()
                with torch.no_grad():
                    val_loss = criterion(model(X_va), y_va).item()
                if val_loss < best_val:
                    best_val = val_loss
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        break
            return best_val

        study = optuna.create_study(direction="minimize",
                                    sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        print(f"  LSTM best params: {study.best_params} (val loss: {study.best_value:.4f})")
        return study.best_params

    def _build(self, n_features: int):
        return _LSTMNet(n_features, self.lstm_units, self.dropout).to(self.device)

    def _sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
        player_ids: np.ndarray,
        seasons: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Build LSTM input sequences per player.

        Returns (X_seq, y_seq, seq_seasons) where ``seq_seasons[j]`` is
        the season of the target row for sequence *j*.  When *seasons* is
        ``None``, ``seq_seasons`` is returned as ``None``.
        """
        X_seq, y_seq, s_seq = [], [], []
        for pid in np.unique(player_ids):
            mask = player_ids == pid
            Xi = X[mask]
            yi = y[mask]
            si = seasons[mask] if seasons is not None else None
            for i in range(len(Xi) - self.sequence_length):
                X_seq.append(Xi[i : i + self.sequence_length])
                y_seq.append(yi[i + self.sequence_length])
                if si is not None:
                    s_seq.append(si[i + self.sequence_length])
        seq_seasons = np.array(s_seq) if s_seq else None
        return np.array(X_seq), np.array(y_seq), seq_seasons

    def fit(self, X: np.ndarray, y: np.ndarray, player_ids: np.ndarray,
            feature_names: List[str],
            epochs: int = None,
            batch_size: int = None,
            seasons: Optional[np.ndarray] = None) -> "LSTM4WeekModel":
        epochs = epochs or MODEL_CONFIG.get("lstm_epochs", 80)
        batch_size = batch_size or MODEL_CONFIG.get("lstm_batch_size", 32)
        from sklearn.preprocessing import StandardScaler
        self.feature_names = list(feature_names)
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)
        X_seq, y_seq, seq_seasons = self._sequences(Xs, y, player_ids, seasons=seasons)
        if len(X_seq) < 50:
            self.is_fitted = False
            return self
        n_features = X_seq.shape[2]
        self.model = self._build(n_features)

        # Season-aware train/val split (falls back to last 20% if no season data)
        gap = int(MODEL_CONFIG.get("cv_gap_seasons", 1))
        train_idx, val_idx = _season_split(
            len(X_seq), seq_seasons, n_val_seasons=1, gap_seasons=gap
        )
        X_train_t = torch.tensor(X_seq[train_idx], dtype=torch.float32)
        y_train_t = torch.tensor(y_seq[train_idx], dtype=torch.float32)
        X_val_t = torch.tensor(X_seq[val_idx], dtype=torch.float32).to(self.device)
        y_val_t = torch.tensor(y_seq[val_idx], dtype=torch.float32).to(self.device)

        dataset = TensorDataset(X_train_t, y_train_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.HuberLoss(delta=1.0)  # Robust to outlier games

        best_val_loss = float("inf")
        patience, patience_limit = 0, 10
        best_state = None

        for epoch in range(epochs):
            self.model.train()
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimizer.step()

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_loss = criterion(self.model(X_val_t), y_val_t).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience += 1
                if patience >= patience_limit:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray, player_ids: np.ndarray) -> np.ndarray:
        """Predict using LSTM sequences."""
        if not self.is_fitted or self.model is None:
            return np.full(X.shape[0], np.nan)
        Xs = self.scaler.transform(X)
        out = np.full(X.shape[0], np.nan)
        self.model.eval()
        with torch.no_grad():
            for pid in np.unique(player_ids):
                mask = player_ids == pid
                Xi = Xs[mask]
                n = mask.sum()
                if n < self.sequence_length:
                    continue
                seqs = np.array([Xi[i:i + self.sequence_length] for i in range(n - self.sequence_length + 1)])
                if len(seqs) == 0:
                    continue
                t = torch.tensor(seqs, dtype=torch.float32).to(self.device)
                p = self.model(t).cpu().numpy()
                indices = np.where(mask)[0]
                for j, pred_val in enumerate(p):
                    target_idx = indices[j + self.sequence_length - 1]
                    out[target_idx] = pred_val
        return out


class ARIMA4WeekModel:
    """
    Per-player ARIMA for 4-week horizon. Fits univariate ARIMA on each player's
    target series; produces a 4-step-ahead forecast (mean over next 4 weeks)
    per player. Fallback: global mean or 4-week rolling mean when ARIMA fails
    or series too short.
    """
    MIN_LENGTH = 6  # minimum observations to fit ARIMA
    FORECAST_STEPS = 4  # 4-week horizon

    def __init__(self, order: Tuple[int, int, int] = None):
        if not HAS_ARIMA:
            raise ImportError("statsmodels required for ARIMA. pip install statsmodels")
        self.order = order or tuple(MODEL_CONFIG.get("arima_order", (2, 1, 2)))
        self.is_fitted = False
        self.fallback_mean = 0.0
        self._player_forecast: Dict[str, float] = {}  # str(player_id) -> 4-week forecast
        self._player_train_series: Dict[str, np.ndarray] = {}  # training series for online update
        self._player_fit_result: Dict[str, Any] = {}  # fitted ARIMA result objects

    def _key(self, pid: Any) -> str:
        return str(pid)

    def fit(self, y_series: np.ndarray, player_ids: np.ndarray) -> "ARIMA4WeekModel":
        self.fallback_mean = float(np.nanmean(y_series))
        if not np.isfinite(self.fallback_mean):
            self.fallback_mean = 0.0
        self._player_forecast = {}
        self._player_train_series = {}
        self._player_fit_result = {}
        for pid in np.unique(player_ids):
            mask = player_ids == pid
            y = np.asarray(y_series[mask], dtype=np.float64)
            y = np.where(np.isfinite(y), y, np.nan)
            y = y[~np.isnan(y)]
            k = self._key(pid)
            if len(y) < self.MIN_LENGTH:
                self._player_forecast[k] = float(np.mean(y[-4:])) if len(y) >= 1 else self.fallback_mean
                self._player_train_series[k] = y.copy()
                continue
            try:
                from statsmodels.tsa.arima.model import ARIMA as _ARIMA
                p, d, q = self.order
                model = _ARIMA(y, order=(p, d, q))
                fit_result = model.fit()
                # Forecast 4 steps ahead and average for the 4-week horizon
                fcast = fit_result.forecast(steps=self.FORECAST_STEPS)
                val = float(np.mean(fcast))
                self._player_forecast[k] = val if np.isfinite(val) else float(np.mean(y[-4:]))
                # Store fit result and training series for online updating at predict time
                self._player_fit_result[k] = fit_result
                self._player_train_series[k] = y.copy()
            except Exception:
                self._player_forecast[k] = float(np.mean(y[-4:])) if len(y) >= 4 else float(np.mean(y))
                self._player_train_series[k] = y.copy()
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray, y_hist, player_ids: np.ndarray,
                n_steps: int = None) -> np.ndarray:
        """Predict with ARIMA, optionally updating with recent target values.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (kept for API consistency; not used by ARIMA).
        y_hist : dict or np.ndarray
            If *dict* mapping ``player_id -> np.ndarray``, the values are
            appended to each player's training series via the ARIMA
            ``append()`` method (Kalman filter update) before forecasting.
            If *np.ndarray* of zeros (legacy callers), fall back to cached
            forecasts.
        player_ids : np.ndarray
            Player identifiers aligned with rows of *X*.
        n_steps : int, optional
            Forecast horizon (default: ``FORECAST_STEPS``).
        """
        # Backward-compat for old pickles missing new attributes
        if not hasattr(self, "_player_train_series"):
            self._player_train_series = {}
        if not hasattr(self, "_player_fit_result"):
            self._player_fit_result = {}

        n_steps = n_steps or self.FORECAST_STEPS
        n = X.shape[0]
        if not self.is_fitted:
            return np.full(n, self.fallback_mean)

        out = np.full(n, self.fallback_mean, dtype=np.float64)

        # Normalize y_hist to a dict mapping pid_key -> recent values array
        y_hist_dict: Dict[str, np.ndarray] = {}
        if isinstance(y_hist, dict):
            for pid, vals in y_hist.items():
                y_hist_dict[self._key(pid)] = np.asarray(vals, dtype=np.float64)

        # Cache per-player updated forecasts so each player re-fits at most once
        _updated_cache: Dict[str, float] = {}

        for i in range(n):
            pid = player_ids.flat[i] if hasattr(player_ids, "flat") else player_ids[i]
            k = self._key(pid)

            # Return from per-call cache if already computed for this player
            if k in _updated_cache:
                out[i] = _updated_cache[k]
                continue

            # Try online update when recent data is available
            recent = y_hist_dict.get(k)
            if recent is not None and k in self._player_fit_result:
                recent_clean = recent[np.isfinite(recent)]
                if len(recent_clean) > 0:
                    try:
                        updated = self._player_fit_result[k].append(recent_clean)
                        fcast = updated.forecast(steps=n_steps)
                        val = float(np.mean(fcast))
                        if np.isfinite(val):
                            _updated_cache[k] = val
                            out[i] = val
                            continue
                    except Exception:
                        pass  # fall through to cached forecast

            # Fallback: cached static forecast
            if k in self._player_forecast:
                val = self._player_forecast[k]
                _updated_cache[k] = val
                out[i] = val

        return out


class Hybrid4WeekModel:
    """LSTM + ARIMA for 4-week horizon with fixed blend weights."""
    def __init__(self, position: str):
        self.position = position
        self.lstm = None
        self.arima = None
        self.lstm_weight = MODEL_CONFIG.get("lstm_weight", 0.6)
        self.arima_weight = MODEL_CONFIG.get("arima_weight", 0.4)
        self.is_fitted = False
        self.fallback_ensemble_pred = None

    @property
    def feature_names(self) -> List[str]:
        """Feature names required for predict (from LSTM if fitted)."""
        if self.lstm and getattr(self.lstm, "feature_names", None):
            return list(self.lstm.feature_names)
        return []

    def fit(self, X: pd.DataFrame, y: pd.Series, player_ids: np.ndarray,
            feature_cols: List[str], epochs: int = 80,
            tune_lstm: bool = True,
            seasons: Optional[np.ndarray] = None) -> "Hybrid4WeekModel":
        X_np = X[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
        y_np = y.values.astype(np.float64)
        try:
            # Optuna-tune LSTM hyperparameters when enabled
            tuned_params = {}
            if tune_lstm:
                n_trials = int(MODEL_CONFIG.get("lstm_optuna_trials", 15) or 0)
                if n_trials > 0:
                    print(f"  Tuning LSTM hyperparameters ({n_trials} trials)...")
                    tuned_params = LSTM4WeekModel.tune_hyperparameters(
                        X_np, y_np, player_ids, feature_cols, n_trials=n_trials,
                        seasons=seasons,
                    )
            self.lstm = LSTM4WeekModel(
                sequence_length=int(tuned_params.get("sequence_length", MODEL_CONFIG.get("lstm_sequence_length", 10))),
                lstm_units=int(tuned_params.get("lstm_units", MODEL_CONFIG.get("lstm_units", 256))),
                dropout=float(tuned_params.get("dropout", MODEL_CONFIG.get("lstm_dropout", 0.25))),
                learning_rate=float(tuned_params.get("learning_rate", MODEL_CONFIG.get("lstm_learning_rate", 0.001))),
            )
            self.lstm.fit(X_np, y_np, player_ids, feature_cols, epochs=epochs,
                          seasons=seasons)
        except Exception:
            self.lstm = None
        try:
            self.arima = ARIMA4WeekModel()
            self.arima.fit(y_np, player_ids)
        except Exception:
            self.arima = None
        self.is_fitted = self.lstm is not None or self.arima is not None

        return self

    def predict(self, X: pd.DataFrame, player_ids: np.ndarray, feature_cols: List[str],
                fallback_pred: np.ndarray, n_weeks: int = 4) -> np.ndarray:
        if not self.is_fitted:
            return fallback_pred
        X_np = X[feature_cols].reindex(columns=feature_cols, fill_value=0).values

        # Build y_hist dict from recent target/utilization features for ARIMA
        y_hist: Dict[str, np.ndarray] = {}
        if hasattr(X, "columns"):
            target_col = None
            for cand in ["utilization_score_lag1", "utilization_score_roll3_mean",
                         "utilization_score_roll4_mean", "fantasy_points_roll3_mean"]:
                if cand in X.columns:
                    target_col = cand
                    break
            if target_col is not None:
                for pid in np.unique(player_ids):
                    pmask = player_ids == pid
                    vals = X.loc[X.index[pmask], target_col].values.astype(np.float64)
                    vals = vals[np.isfinite(vals)]
                    if len(vals) > 0:
                        k = self.arima._key(pid) if self.arima else str(pid)
                        y_hist[k] = vals

        pred_lstm = self.lstm.predict(X_np, player_ids) if self.lstm and self.lstm.is_fitted else np.full(len(X_np), np.nan)
        pred_arima = self.arima.predict(X_np, y_hist, player_ids, n_steps=n_weeks) if self.arima else np.full(len(X_np), np.nan)
        out = np.where(np.isfinite(pred_lstm) & np.isfinite(pred_arima),
                      self.lstm_weight * pred_lstm + self.arima_weight * pred_arima,
                      np.where(np.isfinite(pred_lstm), pred_lstm, np.where(np.isfinite(pred_arima), pred_arima, fallback_pred)))
        return out

    def save(self, path: Path = None):
        path = path or MODELS_DIR / f"hybrid_4w_{self.position.lower()}.joblib"
        # Move LSTM model to CPU before saving for portability
        if self.lstm and self.lstm.model is not None:
            self.lstm.model.cpu()
        joblib.dump({"position": self.position, "lstm_weight": self.lstm_weight, "arima_weight": self.arima_weight,
                     "lstm": self.lstm, "arima": self.arima, "is_fitted": self.is_fitted}, path)
        # Move back to device
        if self.lstm and self.lstm.model is not None:
            self.lstm.model.to(self.lstm.device)

    @classmethod
    def load(cls, position: str, path: Path = None) -> "Hybrid4WeekModel":
        path = path or MODELS_DIR / f"hybrid_4w_{position.lower()}.joblib"
        if not path.exists():
            return cls(position)
        d = joblib.load(path)
        m = cls(d["position"])
        m.lstm = d.get("lstm")
        m.arima = d.get("arima")
        m.lstm_weight = d.get("lstm_weight", 0.6)
        m.arima_weight = d.get("arima_weight", 0.4)
        m.is_fitted = d.get("is_fitted", False)
        # Move LSTM back to device after loading
        if m.lstm and m.lstm.model is not None and HAS_TORCH:
            m.lstm.device = _get_device()
            m.lstm.model.to(m.lstm.device)
        # Backward-compat: ensure ARIMA has new attributes from old pickles
        if m.arima is not None:
            if not hasattr(m.arima, "_player_train_series"):
                m.arima._player_train_series = {}
            if not hasattr(m.arima, "_player_fit_result"):
                m.arima._player_fit_result = {}
        return m


# -----------------------------------------------------------------------------
# 18-WEEK: DEEP FEEDFORWARD (98+ LAYERS) + 30% TRADITIONAL BLEND
# -----------------------------------------------------------------------------

if HAS_TORCH:
    class _ResidualBlock(nn.Module):
        """Single residual block: Linear -> BatchNorm -> ReLU -> Dropout + skip connection."""
        def __init__(self, dim: int, dropout: float = 0.35):
            super().__init__()
            self.block = nn.Sequential(
                nn.Linear(dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        def forward(self, x):
            return x + self.block(x)

    class _DeepFeedforwardNet(nn.Module):
        """Residual feedforward network with decreasing-width stages.

        Architecture: input projection -> [stage1: 256 x N_blocks] ->
        [stage2: 64 x N_blocks] -> output.

        Each stage has a linear projection to change dimensionality, followed by
        N residual blocks (skip connections within same-width layers). This enables
        gradient flow without vanishing gradients.

        Default: 2 stages with 2 residual blocks each (~50K parameters).
        Sized for tabular NFL data (~3K-8K samples per position).
        """
        def __init__(self, n_features: int, hidden_units: List[int] = None, dropout: float = 0.35):
            super().__init__()
            # Default architecture: 2 stages of residual blocks (right-sized for NFL data volume)
            if hidden_units is None:
                stage_configs = [(256, 2), (64, 2)]
            else:
                # Parse legacy hidden_units list into stages by grouping consecutive same-width layers
                stage_configs = []
                if hidden_units:
                    current_width = hidden_units[0]
                    count = 0
                    for u in hidden_units:
                        if u == current_width:
                            count += 1
                        else:
                            stage_configs.append((current_width, max(count // 3, 1)))
                            current_width = u
                            count = 1
                    stage_configs.append((current_width, max(count // 3, 1)))

            stages = []
            in_dim = n_features
            for width, n_blocks in stage_configs:
                # Projection layer to change dimensionality
                stages.append(nn.Sequential(
                    nn.Linear(in_dim, width),
                    nn.BatchNorm1d(width),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ))
                # Residual blocks at this width
                for _ in range(n_blocks):
                    stages.append(_ResidualBlock(width, dropout))
                in_dim = width

            self.hidden = nn.Sequential(*stages)
            self.output = nn.Linear(in_dim, 1)

        def forward(self, x):
            return self.output(self.hidden(x)).squeeze(-1)


class DeepSeasonLongModel:
    """Residual feedforward for 18-week horizon; blend 70% deep + 30% traditional."""
    def __init__(self, position: str, n_features: int = None,
                 dropout: float = None, learning_rate: float = None):
        if not HAS_TORCH:
            raise ImportError("PyTorch required for 18-week deep model. pip install torch")
        self.position = position
        self.n_features = n_features or MODEL_CONFIG.get("deep_n_features", 150)
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.is_fitted = False
        self.dropout = dropout or MODEL_CONFIG.get("deep_dropout", 0.35)
        self.learning_rate = learning_rate or MODEL_CONFIG.get("deep_learning_rate", 0.0005)
        self.device = _get_device()
        self.regression_to_mean_scale = 0.95

    @staticmethod
    def tune_hyperparameters(X: np.ndarray, y: np.ndarray, n_trials: int = 15,
                             seasons: Optional[np.ndarray] = None) -> Dict:
        """Tune deep feedforward hyperparameters with Optuna using season-aware splits."""
        try:
            import optuna
            from optuna.samplers import TPESampler
        except ImportError:
            print("  Optuna not available, using default deep model hyperparameters")
            return {}

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        device = _get_device()

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        # Season-aware train/val split for tuning
        gap = int(MODEL_CONFIG.get("cv_gap_seasons", 1))
        train_idx, val_idx = _season_split(
            len(Xs), seasons, n_val_seasons=1, gap_seasons=gap
        )
        X_tr = torch.tensor(Xs[train_idx], dtype=torch.float32)
        y_tr = torch.tensor(y[train_idx], dtype=torch.float32)
        X_va = torch.tensor(Xs[val_idx], dtype=torch.float32).to(device)
        y_va = torch.tensor(y[val_idx], dtype=torch.float32).to(device)

        def objective(trial):
            dropout = trial.suggest_float("dropout", 0.20, 0.50)
            lr = trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True)
            n_blocks = trial.suggest_int("n_blocks_per_stage", 1, 3)
            stage_widths = [(256, n_blocks), (64, n_blocks)]

            hidden_units_flat = []
            for w, nb in stage_widths:
                hidden_units_flat.extend([w] * (nb * 3))  # Approx legacy format

            model = _DeepFeedforwardNet(X.shape[1], hidden_units=hidden_units_flat, dropout=dropout).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.HuberLoss(delta=1.0)

            loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=64, shuffle=True)
            best_val = float("inf")
            patience, no_improve = 5, 0

            for _ in range(25):
                model.train()
                for xb, yb in loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    criterion(model(xb), yb).backward()
                    optimizer.step()
                model.eval()
                with torch.no_grad():
                    val_loss = criterion(model(X_va), y_va).item()
                if val_loss < best_val:
                    best_val = val_loss
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        break
            return best_val

        study = optuna.create_study(direction="minimize",
                                    sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        print(f"  Deep model best params: {study.best_params} (val loss: {study.best_value:.4f})")
        return study.best_params

    def _build(self):
        hidden_units = MODEL_CONFIG.get("deep_hidden_units", None)
        return _DeepFeedforwardNet(self.n_features, hidden_units=hidden_units, dropout=self.dropout).to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str],
            epochs: int = None, batch_size: int = None,
            seasons: Optional[np.ndarray] = None) -> "DeepSeasonLongModel":
        epochs = epochs or MODEL_CONFIG.get("deep_epochs", 100)
        batch_size = batch_size or MODEL_CONFIG.get("deep_batch_size", 64)
        from sklearn.preprocessing import StandardScaler
        self.feature_names = list(feature_names)
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)
        self.n_features = Xs.shape[1]
        self.model = self._build()

        # Season-aware train/val split (falls back to last 20% if no season data)
        gap = int(MODEL_CONFIG.get("cv_gap_seasons", 1))
        train_idx, val_idx = _season_split(
            len(Xs), seasons, n_val_seasons=1, gap_seasons=gap
        )
        X_train_t = torch.tensor(Xs[train_idx], dtype=torch.float32)
        y_train_t = torch.tensor(y[train_idx], dtype=torch.float32)
        X_val_t = torch.tensor(Xs[val_idx], dtype=torch.float32).to(self.device)
        y_val_t = torch.tensor(y[val_idx], dtype=torch.float32).to(self.device)

        dataset = TensorDataset(X_train_t, y_train_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.HuberLoss(delta=1.0)  # Robust to outlier games

        best_val_loss = float("inf")
        patience, patience_limit = 0, 15
        best_state = None

        for epoch in range(epochs):
            self.model.train()
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val_loss = criterion(self.model(X_val_t), y_val_t).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience += 1
                if patience >= patience_limit:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray, traditional_pred: np.ndarray, blend_traditional: float = None) -> np.ndarray:
        blend_traditional = blend_traditional if blend_traditional is not None else getattr(self, '_learned_blend_traditional', MODEL_CONFIG.get("deep_blend_traditional", 0.3))
        if not self.is_fitted or self.model is None:
            return traditional_pred
        X_in = X[:, :self.n_features] if X.shape[1] >= self.n_features else np.hstack([X, np.zeros((X.shape[0], self.n_features - X.shape[1]))])
        Xs = self.scaler.transform(X_in)
        self.model.eval()
        with torch.no_grad():
            t = torch.tensor(Xs, dtype=torch.float32).to(self.device)
            deep_pred = self.model(t).cpu().numpy()
        return (1 - blend_traditional) * deep_pred + blend_traditional * traditional_pred

    def learn_blend_weight(self, X: np.ndarray, y: np.ndarray, traditional_pred: np.ndarray):
        """Learn the optimal deep/traditional blend weight on validation data."""
        if not self.is_fitted or self.model is None:
            return
        try:
            X_in = X[:, :self.n_features] if X.shape[1] >= self.n_features else np.hstack([X, np.zeros((X.shape[0], self.n_features - X.shape[1]))])
            Xs = self.scaler.transform(X_in)
            self.model.eval()
            with torch.no_grad():
                t = torch.tensor(Xs, dtype=torch.float32).to(self.device)
                deep_pred = self.model(t).cpu().numpy()
            valid = np.isfinite(deep_pred) & np.isfinite(traditional_pred) & np.isfinite(y)
            if valid.sum() < 10:
                return
            from scipy.optimize import minimize_scalar
            def loss(w):
                blend = (1 - w) * deep_pred[valid] + w * traditional_pred[valid]
                return np.mean((blend - y[valid]) ** 2)
            result = minimize_scalar(loss, bounds=(0.05, 0.95), method='bounded')
            self._learned_blend_traditional = float(result.x)
            print(f"    Learned 18w blend: deep={1 - self._learned_blend_traditional:.2f}, traditional={self._learned_blend_traditional:.2f}")
        except Exception:
            pass

    def save(self, path: Path = None):
        path = path or MODELS_DIR / f"deep_18w_{self.position.lower()}"
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self.model is not None:
            self.model.cpu()
            torch.save(self.model.state_dict(), path / "model.pt")
            self.model.to(self.device)
        joblib.dump({"scaler": self.scaler, "feature_names": self.feature_names, "n_features": self.n_features,
                     "is_fitted": self.is_fitted, "position": self.position,
                     "regression_to_mean_scale": self.regression_to_mean_scale,
                     "_learned_blend_traditional": getattr(self, "_learned_blend_traditional", None)}, path / "config.joblib")

    @classmethod
    def load(cls, position: str, path: Path = None) -> "DeepSeasonLongModel":
        path = path or MODELS_DIR / f"deep_18w_{position.lower()}"
        path = Path(path)
        if not path.exists() or not (path / "config.joblib").exists():
            return cls(position)
        cfg = joblib.load(path / "config.joblib")
        m = cls(cfg["position"], cfg.get("n_features", 150))
        m.scaler = cfg.get("scaler")
        m.feature_names = cfg.get("feature_names", [])
        m.n_features = cfg.get("n_features", 150)
        m.is_fitted = cfg.get("is_fitted", False)
        m.regression_to_mean_scale = cfg.get("regression_to_mean_scale", 0.95)
        learned_blend = cfg.get("_learned_blend_traditional")
        if learned_blend is not None:
            m._learned_blend_traditional = learned_blend
        if (path / "model.pt").exists() and HAS_TORCH:
            m.model = _DeepFeedforwardNet(m.n_features).to(m.device)
            m.model.load_state_dict(torch.load(path / "model.pt", map_location=m.device))
            m.model.eval()
        return m
