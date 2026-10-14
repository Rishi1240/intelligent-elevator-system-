"""
Multi-Output LSTM Neural Network for Elevator Congestion and Per-Floor Demand Prediction.

Architecture: Two-head Keras Functional model
  Shared encoder: LSTM(64) -> Dropout(0.2) -> LSTM(32) -> Dropout(0.2) -> Dense(16, relu)
  Head 1 (congestion): Dense(1, sigmoid)   – predicts overall congestion at t+5 min
  Head 2 (floor demand): Dense(10, relu)   – predicts demand at each of 10 floors at t+5 min

Backward-compatible: the predict() method returns a single float for congestion (as before).
New predict_full() method returns both congestion and per-floor demand.

IMPORTANT: This model predicts SYNTHETIC TRAINING DATA patterns.
Do NOT represent its output as measured real-building performance.
"""

import os
import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model, load_model as keras_load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    TF_AVAILABLE = True
except ImportError:
    tf = None
    Sequential = None
    Model = None
    keras_load_model = None
    LSTM = None
    Dense = None
    Dropout = None
    Input = None
    EarlyStopping = None
    ModelCheckpoint = None
    TF_AVAILABLE = False


class LSTMModel:
    """
    Multi-output LSTM that predicts:
      1. Overall building congestion 5 minutes ahead  (congestion_t5)
      2. Per-floor passenger demand 5 minutes ahead   (demand_f1_t5 .. demand_f10_t5)

    Falls back to a simple heuristic when TensorFlow is unavailable or weights
    have not been loaded.
    """

    def __init__(
        self,
        input_features: int = 20,
        sequence_length: int = 10,
        hidden_units: int = 64,
        num_floors: int = 10,
    ):
        self.input_features  = input_features
        self.sequence_length = sequence_length
        self.hidden_units    = hidden_units
        self.num_floors      = num_floors
        self.model           = None

        # Diagnostic flags exposed to callers
        self.is_loaded       = False      # True when real weights are in memory
        self.model_mode      = "fallback" # "loaded" | "fallback"

    # ── model construction ───────────────────────────────────────────────────

    def build_model(self):
        """Builds and compiles the Keras Functional multi-output LSTM model."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow/Keras is required to build the LSTM model.")

        inp = Input(shape=(self.sequence_length, self.input_features), name="sequence_input")

        # Shared encoder
        x = LSTM(self.hidden_units, return_sequences=True, name="lstm_1")(inp)
        x = Dropout(0.2, name="drop_1")(x)
        x = LSTM(32, return_sequences=False, name="lstm_2")(x)
        x = Dropout(0.2, name="drop_2")(x)
        shared = Dense(16, activation="relu", name="shared_dense")(x)

        # Head 1: congestion prediction (t+5)
        congestion_out = Dense(1, activation="sigmoid", name="congestion_out")(shared)

        # Head 2: per-floor demand prediction (t+5)
        demand_out = Dense(self.num_floors, activation="relu", name="demand_out")(shared)

        self.model = Model(inputs=inp, outputs=[congestion_out, demand_out])
        self.model.compile(
            optimizer="adam",
            loss={
                "congestion_out": "mse",
                "demand_out":     "mse",
            },
            loss_weights={
                "congestion_out": 1.0,
                "demand_out":     0.5,   # demand prediction slightly lower weight
            },
            metrics={
                "congestion_out": ["mae"],
                "demand_out":     ["mae"],
            },
        )
        return self.model

    # ── training ─────────────────────────────────────────────────────────────

    def train(
        self,
        X_train,
        y_cong_train,
        y_demand_train,
        epochs: int = 30,
        batch_size: int = 64,
        validation_data=None,
        checkpoint_path: str = "saved_model/lstm_weights.keras",
    ):
        """
        Trains the multi-output LSTM.

        Parameters
        ----------
        X_train       : np.ndarray, shape (N, seq_len, features)
        y_cong_train  : np.ndarray, shape (N,)   – congestion at t+5
        y_demand_train: np.ndarray, shape (N, 10) – per-floor demand at t+5
        validation_data: tuple ((X_val, y_cong_val, y_demand_val)) or None
        """
        if self.model is None:
            if hasattr(X_train, "shape") and len(X_train.shape) == 3:
                self.sequence_length = X_train.shape[1]
                self.input_features  = X_train.shape[2]
            self.build_model()

        callbacks: list = []
        val_data = None

        if validation_data is not None:
            X_v, y_cong_v, y_demand_v = validation_data
            val_data = (X_v, {"congestion_out": y_cong_v, "demand_out": y_demand_v})
            monitor = "val_loss"
        else:
            monitor = "loss"

        if EarlyStopping is not None:
            callbacks.append(
                EarlyStopping(monitor=monitor, patience=6, restore_best_weights=True)
            )

        if checkpoint_path and ModelCheckpoint is not None:
            os.makedirs(
                os.path.dirname(checkpoint_path) if os.path.dirname(checkpoint_path) else ".",
                exist_ok=True,
            )
            callbacks.append(
                ModelCheckpoint(
                    filepath=checkpoint_path,
                    monitor=monitor,
                    save_best_only=True,
                    verbose=1,
                )
            )

        history = self.model.fit(
            X_train,
            {"congestion_out": y_cong_train, "demand_out": y_demand_train},
            epochs=epochs,
            batch_size=batch_size,
            validation_data=val_data,
            callbacks=callbacks,
            verbose=1,
        )
        self.is_loaded  = True
        self.model_mode = "loaded"
        return history

    # ── inference ────────────────────────────────────────────────────────────

    def predict(self, sequence) -> float:
        """
        Backward-compatible single-value congestion prediction.
        Returns a float in [0.0, 1.0].
        Uses the real LSTM if weights are loaded; otherwise returns heuristic fallback.
        """
        cong, _ = self._predict_internal(sequence)
        return cong

    def predict_full(self, sequence) -> dict:
        """
        Returns both congestion and per-floor demand predictions.

        Returns
        -------
        {
            "congestion_t5": float in [0,1],
            "floor_demand_t5": list[float] of length num_floors,
            "model_mode": "loaded" | "fallback"
        }
        """
        cong, demand = self._predict_internal(sequence)
        return {
            "congestion_t5":  cong,
            "floor_demand_t5": demand,
            "model_mode":      self.model_mode,
        }

    def _predict_internal(self, sequence):
        """Core inference: returns (congestion_float, demand_list)."""
        seq_arr = np.array(sequence, dtype=np.float32)
        if seq_arr.ndim == 2:
            seq_arr = np.expand_dims(seq_arr, axis=0)
        elif seq_arr.ndim != 3:
            raise ValueError(f"Expected 2D or 3D sequence, got shape {seq_arr.shape}")

        if self.model is not None:
            preds = self.model.predict(seq_arr, verbose=0)
            # preds is a list of two arrays: [congestion (1,1), demand (1,num_floors)]
            cong   = float(np.clip(preds[0][0][0], 0.0, 1.0))
            demand = [float(max(0.0, v)) for v in preds[1][0].tolist()]
            return cong, demand

        # ── Fallback heuristic (no trained weights) ──────────────────────────
        last_ts = seq_arr[0, -1]
        approx_cong = 0.25
        if len(last_ts) >= 14:
            total_req   = float(last_ts[13]) if len(last_ts) > 13 else 5.0
            approx_cong = min(1.0, max(0.05, total_req / 40.0))
        # Distribute fallback demand proportional to current floor counts
        demand_raw = [float(max(0.0, last_ts[3 + i])) for i in range(min(10, len(last_ts) - 3))]
        while len(demand_raw) < 10:
            demand_raw.append(0.0)
        return float(round(approx_cong, 4)), demand_raw

    # ── evaluation ───────────────────────────────────────────────────────────

    def evaluate(self, X_test, y_cong_test, y_demand_test):
        """Evaluates model on test set. Returns metrics dict."""
        if self.model is None:
            raise RuntimeError("Model has not been built or loaded.")
        results = self.model.evaluate(
            X_test,
            {"congestion_out": y_cong_test, "demand_out": y_demand_test},
            verbose=0,
            return_dict=True,
        )
        return results

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, path: str):
        """Saves full model (architecture + weights) to path."""
        if self.model is None:
            raise RuntimeError("No model to save.")
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        self.model.save(path)

    def load(self, path: str) -> bool:
        """
        Loads the full model from path.
        Returns True on success, False if TF unavailable.
        Raises FileNotFoundError if the path does not exist.
        """
        if not TF_AVAILABLE:
            print("Warning: TensorFlow not found. Using fallback predictor.")
            self.model_mode = "fallback"
            return False

        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")

        try:
            self.model     = keras_load_model(path)
            self.is_loaded  = True
            self.model_mode = "loaded"
            print(f"[LSTMModel] Loaded multi-output model from {path}")
            return True
        except Exception as exc:
            print(f"[LSTMModel] load() failed: {exc}. Using fallback predictor.")
            self.model_mode = "fallback"
            return False
