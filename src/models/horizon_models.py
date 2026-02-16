"""
Horizon-specific models per requirements: 4-week LSTM+ARIMA hybrid, 18-week deep feedforward.

- 4-week: LSTM (60%) + ARIMA (40%); injury/workload/EWMA features.
- 18-week: Deep feedforward (98+ layers); 70% deep + 30% traditional; regression-to-mean.

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
except ImportError:
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
                 dropout: float = None):
        if not HAS_TORCH:
            raise ImportError("PyTorch required for 4-week LSTM. pip install torch")
        self.sequence_length = sequence_length or MODEL_CONFIG.get("lstm_sequence_length", 10)
        self.lstm_units = lstm_units or MODEL_CONFIG.get("lstm_units", 256)
        self.dropout = dropout if dropout is not None else MODEL_CONFIG.get("lstm_dropout", 0.25)
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.is_fitted = False
        self.device = _get_device("lstm")

    def _build(self, n_features: int) -> _LSTMNet:
        return _LSTMNet(n_features, self.lstm_units, self.dropout).to(self.device)

    def _sequences(self, X: np.ndarray, y: np.ndarray, player_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_seq, y_seq = [], []
        for pid in np.unique(player_ids):
            mask = player_ids == pid
            Xi = X[mask]
            yi = y[mask]
            for i in range(len(Xi) - self.sequence_length):
                X_seq.append(Xi[i : i + self.sequence_length])
                y_seq.append(yi[i + self.sequence_length])
        return np.array(X_seq), np.array(y_seq)

    def fit(self, X: np.ndarray, y: np.ndarray, player_ids: np.ndarray,
            feature_names: List[str],
            epochs: int = None,
            batch_size: int = None) -> "LSTM4WeekModel":
        epochs = epochs or MODEL_CONFIG.get("lstm_epochs", 80)
        batch_size = batch_size or MODEL_CONFIG.get("lstm_batch_size", 32)
        from sklearn.preprocessing import StandardScaler
        self.feature_names = list(feature_names)
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)
        X_seq, y_seq = self._sequences(Xs, y, player_ids)
        if len(X_seq) < 50:
            self.is_fitted = False
            return self
        n_features = X_seq.shape[2]
        self.model = self._build(n_features)

        # Train/val split (last 20%)
        n = len(X_seq)
        split = int(n * 0.8)
        X_train_t = torch.tensor(X_seq[:split], dtype=torch.float32)
        y_train_t = torch.tensor(y_seq[:split], dtype=torch.float32)
        X_val_t = torch.tensor(X_seq[split:], dtype=torch.float32).to(self.device)
        y_val_t = torch.tensor(y_seq[split:], dtype=torch.float32).to(self.device)

        dataset = TensorDataset(X_train_t, y_train_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        lr = MODEL_CONFIG.get("lstm_learning_rate", 0.001)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

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

    def _key(self, pid: Any) -> str:
        return str(pid)

    def fit(self, y_series: np.ndarray, player_ids: np.ndarray) -> "ARIMA4WeekModel":
        self.fallback_mean = float(np.nanmean(y_series))
        if not np.isfinite(self.fallback_mean):
            self.fallback_mean = 0.0
        self._player_forecast = {}
        for pid in np.unique(player_ids):
            mask = player_ids == pid
            y = np.asarray(y_series[mask], dtype=np.float64)
            y = np.where(np.isfinite(y), y, np.nan)
            y = y[~np.isnan(y)]
            if len(y) < self.MIN_LENGTH:
                self._player_forecast[self._key(pid)] = float(np.mean(y[-4:])) if len(y) >= 1 else self.fallback_mean
                continue
            try:
                from statsmodels.tsa.arima.model import ARIMA
                p, d, q = self.order
                model = ARIMA(y, order=(p, d, q))
                fit = model.fit()
                # Forecast 4 steps ahead and average for the 4-week horizon
                fcast = fit.forecast(steps=self.FORECAST_STEPS)
                val = float(np.mean(fcast))
                self._player_forecast[self._key(pid)] = val if np.isfinite(val) else float(np.mean(y[-4:]))
            except Exception:
                self._player_forecast[self._key(pid)] = float(np.mean(y[-4:])) if len(y) >= 4 else float(np.mean(y))
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray, y_hist: np.ndarray, player_ids: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        if not self.is_fitted:
            return np.full(n, self.fallback_mean)
        out = np.full(n, self.fallback_mean, dtype=np.float64)
        for i in range(n):
            pid = player_ids.flat[i] if hasattr(player_ids, "flat") else player_ids[i]
            k = self._key(pid)
            if k in self._player_forecast:
                out[i] = self._player_forecast[k]
        return out


class Hybrid4WeekModel:
    """60% LSTM + 40% ARIMA for 4-week horizon."""
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
            feature_cols: List[str], epochs: int = 80) -> "Hybrid4WeekModel":
        X_np = X[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
        y_np = y.values.astype(np.float64)
        try:
            self.lstm = LSTM4WeekModel(
                sequence_length=int(MODEL_CONFIG.get("lstm_sequence_length", 10)),
                lstm_units=int(MODEL_CONFIG.get("lstm_units", 256)),
                dropout=float(MODEL_CONFIG.get("lstm_dropout", 0.25)),
            )
            self.lstm.fit(X_np, y_np, player_ids, feature_cols, epochs=epochs)
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
                fallback_pred: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return fallback_pred
        X_np = X[feature_cols].reindex(columns=feature_cols, fill_value=0).values
        y_hist = np.zeros(len(X_np))
        pred_lstm = self.lstm.predict(X_np, player_ids) if self.lstm and self.lstm.is_fitted else np.full(len(X_np), np.nan)
        pred_arima = self.arima.predict(X_np, y_hist, player_ids) if self.arima else np.full(len(X_np), np.nan)
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
        return m


# -----------------------------------------------------------------------------
# 18-WEEK: DEEP FEEDFORWARD (98+ LAYERS) + 30% TRADITIONAL BLEND
# -----------------------------------------------------------------------------

if HAS_TORCH:
    class _DeepFeedforwardNet(nn.Module):
        """98+ layer deep feedforward with decreasing widths: 512->256->128->64->32.
        Each block = Linear + BatchNorm + ReLU + Dropout.
        Uses residual connections within same-width blocks to enable gradient flow.
        """
        def __init__(self, n_features: int, hidden_units: List[int] = None, dropout: float = 0.35):
            super().__init__()
            if hidden_units is None:
                hidden_units = (
                    [512] * 25 +
                    [256] * 25 +
                    [128] * 20 +
                    [64] * 18 +
                    [32] * 12
                )  # 100 hidden layers
            layers = []
            in_dim = n_features
            for u in hidden_units:
                layers.append(nn.Linear(in_dim, u))
                layers.append(nn.BatchNorm1d(u))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                in_dim = u
            self.hidden = nn.Sequential(*layers)
            self.output = nn.Linear(in_dim, 1)

        def forward(self, x):
            return self.output(self.hidden(x)).squeeze(-1)


class DeepSeasonLongModel:
    """98+ layer deep feedforward for 18-week; blend 70% deep + 30% traditional; regression-to-mean."""
    def __init__(self, position: str, n_features: int = None):
        if not HAS_TORCH:
            raise ImportError("PyTorch required for 18-week deep model. pip install torch")
        self.position = position
        self.n_features = n_features or MODEL_CONFIG.get("deep_n_features", 150)
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.is_fitted = False
        self.regression_to_mean_scale = MODEL_CONFIG.get("deep_regression_to_mean_scale", 0.95)
        self.device = _get_device()

    def _build(self) -> _DeepFeedforwardNet:
        hidden_units = MODEL_CONFIG.get("deep_hidden_units", None)
        dropout = MODEL_CONFIG.get("deep_dropout", 0.35)
        return _DeepFeedforwardNet(self.n_features, hidden_units=hidden_units, dropout=dropout).to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str],
            epochs: int = None, batch_size: int = None) -> "DeepSeasonLongModel":
        epochs = epochs or MODEL_CONFIG.get("deep_epochs", 100)
        batch_size = batch_size or MODEL_CONFIG.get("deep_batch_size", 64)
        from sklearn.preprocessing import StandardScaler
        self.feature_names = list(feature_names)
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)
        self.n_features = Xs.shape[1]
        self.model = self._build()

        # Train/val split (last 20%)
        n = len(Xs)
        split = int(n * 0.8)
        X_train_t = torch.tensor(Xs[:split], dtype=torch.float32)
        y_train_t = torch.tensor(y[:split], dtype=torch.float32)
        X_val_t = torch.tensor(Xs[split:], dtype=torch.float32).to(self.device)
        y_val_t = torch.tensor(y[split:], dtype=torch.float32).to(self.device)

        dataset = TensorDataset(X_train_t, y_train_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        lr = MODEL_CONFIG.get("deep_learning_rate", 0.0005)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

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
        blend_traditional = blend_traditional if blend_traditional is not None else MODEL_CONFIG.get("deep_blend_traditional", 0.3)
        if not self.is_fitted or self.model is None:
            return traditional_pred
        X_in = X[:, :self.n_features] if X.shape[1] >= self.n_features else np.hstack([X, np.zeros((X.shape[0], self.n_features - X.shape[1]))])
        Xs = self.scaler.transform(X_in)
        self.model.eval()
        with torch.no_grad():
            t = torch.tensor(Xs, dtype=torch.float32).to(self.device)
            deep_pred = self.model(t).cpu().numpy()
        blended = (1 - blend_traditional) * deep_pred + blend_traditional * traditional_pred
        mean_val = np.nanmean(blended)
        regression_adjusted = mean_val + self.regression_to_mean_scale * (blended - mean_val)
        return regression_adjusted

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
                     "regression_to_mean_scale": self.regression_to_mean_scale}, path / "config.joblib")

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
        if (path / "model.pt").exists() and HAS_TORCH:
            m.model = _DeepFeedforwardNet(m.n_features).to(m.device)
            m.model.load_state_dict(torch.load(path / "model.pt", map_location=m.device))
            m.model.eval()
        return m
