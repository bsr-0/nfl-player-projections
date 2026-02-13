"""
Horizon-specific models per requirements: 4-week LSTM+ARIMA hybrid, 18-week deep feedforward.

- 4-week: LSTM (60%) + ARIMA (40%); injury/workload/EWMA features.
- 18-week: Deep feedforward (98+ layers); 70% deep + 30% traditional; regression-to-mean.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import joblib

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import MODELS_DIR, MODEL_CONFIG

# Optional TensorFlow for LSTM and deep net
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    HAS_TF = True
except ImportError:
    HAS_TF = False

# Optional statsmodels for ARIMA
try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_ARIMA = True
except ImportError:
    HAS_ARIMA = False


# -----------------------------------------------------------------------------
# 4-WEEK: LSTM + ARIMA HYBRID (60% LSTM, 40% ARIMA)
# -----------------------------------------------------------------------------
class LSTM4WeekModel:
    """LSTM for 4-week horizon; 3-4 LSTM layers, 128-256 units per layer (requirements), sequence length 8-12."""
    def __init__(self, sequence_length: int = 10, lstm_units: int = 256, dropout: float = 0.25):
        if not HAS_TF:
            raise ImportError("TensorFlow required for 4-week LSTM. pip install tensorflow")
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units  # first layer size (128-256)
        self.dropout = dropout
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.is_fitted = False

    def _build(self, n_features: int):
        from tensorflow.keras.layers import LSTM
        # Requirements: 3-4 LSTM layers, 128-256 units; use 256, 128, 64
        u1 = min(256, max(128, self.lstm_units))
        u2, u3 = 128, 64
        m = Sequential([
            Input(shape=(self.sequence_length, n_features)),
            LSTM(u1, activation="tanh", return_sequences=True),
            Dropout(self.dropout),
            LSTM(u2, activation="tanh", return_sequences=True),
            Dropout(self.dropout),
            LSTM(u3, activation="tanh"),
            Dropout(self.dropout),
            Dense(32, activation="relu"),
            Dense(1),
        ])
        m.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
        return m

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
            feature_names: List[str], epochs: int = 80, batch_size: int = 32) -> "LSTM4WeekModel":
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
        self.model.fit(
            X_seq, y_seq,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
            verbose=0,
        )
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray, player_ids: np.ndarray) -> np.ndarray:
        """Predict using LSTM sequences. Maps each player's last valid sequence
        prediction to their last row so the output aligns with the input index."""
        if not self.is_fitted or self.model is None:
            return np.full(X.shape[0], np.nan)
        Xs = self.scaler.transform(X)
        out = np.full(X.shape[0], np.nan)
        for pid in np.unique(player_ids):
            mask = player_ids == pid
            Xi = Xs[mask]
            n = mask.sum()
            if n < self.sequence_length:
                continue
            # Build all valid sequences for this player
            seqs = np.array([Xi[i:i + self.sequence_length] for i in range(n - self.sequence_length + 1)])
            if len(seqs) == 0:
                continue
            p = self.model.predict(seqs, verbose=0).flatten()
            # Map predictions: each seq[i] predicts for row i + sequence_length - 1
            indices = np.where(mask)[0]
            for j, pred_val in enumerate(p):
                target_idx = indices[j + self.sequence_length - 1]
                out[target_idx] = pred_val
        return out


class ARIMA4WeekModel:
    """
    Per-player ARIMA for 4-week horizon. Fits univariate ARIMA on each player's
    target series; stores one-step-ahead forecast per player. Fallback: global mean
    or 4-week rolling mean when ARIMA fails or series too short.
    """
    MIN_LENGTH = 6  # minimum observations to fit ARIMA

    def __init__(self, order: Tuple[int, int, int] = (2, 1, 2)):
        if not HAS_ARIMA:
            raise ImportError("statsmodels required for ARIMA. pip install statsmodels")
        self.order = order
        self.is_fitted = False
        self.fallback_mean = 0.0
        self._player_forecast: Dict[str, float] = {}  # str(player_id) -> one-step-ahead forecast

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
                fcast = fit.forecast(steps=1)
                val = float(fcast[0])
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
        self.lstm_weight = 0.6
        self.arima_weight = 0.4
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
            self.lstm = LSTM4WeekModel(sequence_length=10, lstm_units=256, dropout=0.25)
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
        joblib.dump({"position": self.position, "lstm_weight": self.lstm_weight, "arima_weight": self.arima_weight,
                     "lstm": self.lstm, "arima": self.arima, "is_fitted": self.is_fitted}, path)

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
        return m


# -----------------------------------------------------------------------------
# 18-WEEK: DEEP FEEDFORWARD (98+ LAYERS) + 30% TRADITIONAL BLEND
# -----------------------------------------------------------------------------
def _deep_feedforward_layers(input_dim: int, hidden_units: List[int] = None) -> List:
    """Build 98+ layer deep feedforward with decreasing nodes per requirements.

    Architecture: ~100 hidden layers (each Dense+BatchNorm+Dropout = 3 Keras layers)
    organised as blocks of decreasing width: 512→256→128→64→32.
    Total Dense layers ≈ 100, satisfying the 98+ layer requirement.
    """
    if not HAS_TF:
        raise ImportError("TensorFlow required for deep feedforward layers.")
    from tensorflow.keras.layers import Dense as _Dense, BatchNormalization as _BN, Dropout as _DO
    if hidden_units is None:
        # 100 Dense layers with gradually decreasing widths (512→256→128→64→32)
        hidden_units = (
            [512] * 25 +   # 25 layers at 512
            [256] * 25 +   # 25 layers at 256
            [128] * 20 +   # 20 layers at 128
            [64] * 18 +    # 18 layers at 64
            [32] * 12      # 12 layers at 32
        )  # total = 100 Dense hidden layers
    layers = []
    for u in hidden_units:
        layers.append(_Dense(u, activation="relu"))
        layers.append(_BN())
        layers.append(_DO(0.35))
    layers.append(_Dense(1))
    return layers


class DeepSeasonLongModel:
    """98+ layer deep feedforward for 18-week; blend 70% deep + 30% traditional; regression-to-mean."""
    def __init__(self, position: str, n_features: int = 150):
        if not HAS_TF:
            raise ImportError("TensorFlow required for 18-week deep model. pip install tensorflow")
        self.position = position
        self.n_features = n_features
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.is_fitted = False
        self.regression_to_mean_scale = 0.95

    def _build(self):
        layers = _deep_feedforward_layers(self.n_features)
        m = Sequential([Input(shape=(self.n_features,))] + layers)
        m.compile(optimizer=Adam(learning_rate=0.0005), loss="mse", metrics=["mae"])
        return m

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str],
            epochs: int = 100, batch_size: int = 64) -> "DeepSeasonLongModel":
        from sklearn.preprocessing import StandardScaler
        self.feature_names = list(feature_names)
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)
        self.n_features = Xs.shape[1]
        self.model = self._build()
        self.model.fit(
            Xs, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[EarlyStopping(patience=15, restore_best_weights=True)],
            verbose=0,
        )
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray, traditional_pred: np.ndarray, blend_traditional: float = 0.3) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            return traditional_pred
        Xs = self.scaler.transform(X[:, :self.n_features] if X.shape[1] >= self.n_features else np.hstack([X, np.zeros((X.shape[0], self.n_features - X.shape[1]))]))
        deep_pred = self.model.predict(Xs, verbose=0).flatten()
        blended = (1 - blend_traditional) * deep_pred + blend_traditional * traditional_pred
        mean_val = np.nanmean(blended)
        regression_adjusted = mean_val + self.regression_to_mean_scale * (blended - mean_val)
        return regression_adjusted

    def save(self, path: Path = None):
        path = path or MODELS_DIR / f"deep_18w_{self.position.lower()}"
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self.model is not None:
            self.model.save(path / "model.keras")
        joblib.dump({"scaler": self.scaler, "feature_names": self.feature_names, "n_features": self.n_features,
                     "is_fitted": self.is_fitted, "position": self.position}, path / "config.joblib")

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
        if (path / "model.keras").exists():
            m.model = load_model(path / "model.keras")
        return m
