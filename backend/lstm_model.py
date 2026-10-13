"""
LSTM Neural Network for Elevator Congestion Prediction.
Implements Keras sequential architecture:
LSTM(64) -> Dropout(0.2) -> LSTM(32) -> Dropout(0.2) -> Dense(16, relu) -> Dense(1, sigmoid)
"""

import os
import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model as keras_load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    TF_AVAILABLE = True
except ImportError:
    tf = None
    Sequential = None
    keras_load_model = None
    LSTM = None
    Dense = None
    Dropout = None
    Input = None
    EarlyStopping = None
    ModelCheckpoint = None
    TF_AVAILABLE = False



class LSTMModel:
    def __init__(self, input_features=20, sequence_length=10, hidden_units=64):
        self.input_features = input_features
        self.sequence_length = sequence_length
        self.hidden_units = hidden_units
        self.model = None

    def build_model(self):
        """Builds and compiles the Keras Sequential LSTM model."""
        if not TF_AVAILABLE or Sequential is None or Input is None or LSTM is None or Dropout is None or Dense is None:
            raise ImportError("TensorFlow/Keras is required to build and train the LSTM model.")

        model = Sequential([
            Input(shape=(self.sequence_length, self.input_features)),
            LSTM(self.hidden_units, return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid")
        ])

        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        self.model = model
        return model

    def train(self, X_train, y_train, epochs=50, batch_size=32, validation_data=None, checkpoint_path="saved_model/lstm_weights.h5"):
        """
        Trains the LSTM model with EarlyStopping and ModelCheckpoint.
        """
        if self.model is None:
            # Dynamically infer feature dimension from input shape if available
            if hasattr(X_train, 'shape') and len(X_train.shape) == 3:
                self.sequence_length = X_train.shape[1]
                self.input_features = X_train.shape[2]
            self.build_model()

        callbacks: list = []
        if EarlyStopping is not None:
            callbacks.append(
                EarlyStopping(monitor="val_loss" if validation_data is not None else "loss", patience=5, restore_best_weights=True)
            )


        if checkpoint_path and ModelCheckpoint is not None:
            os.makedirs(os.path.dirname(checkpoint_path) if os.path.dirname(checkpoint_path) else ".", exist_ok=True)
            checkpoint = ModelCheckpoint(
                filepath=checkpoint_path,
                monitor="val_loss" if validation_data is not None else "loss",
                save_best_only=True,
                verbose=1
            )
            callbacks.append(checkpoint)

        if self.model is None:
            raise RuntimeError("Model is not initialized.")

        history = self.model.fit(

            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        return history

    def predict(self, sequence):
        """
        Predicts congestion level (float between 0.0 and 1.0).
        Accepts:
        - 2D sequence of shape (sequence_length, features)
        - 3D sequence of shape (1, sequence_length, features)
        """
        seq_arr = np.array(sequence, dtype=np.float32)

        if seq_arr.ndim == 2:
            seq_arr = np.expand_dims(seq_arr, axis=0)
        elif seq_arr.ndim != 3:
            raise ValueError(f"Expected 2D or 3D sequence array, got shape {seq_arr.shape}")

        if self.model is not None:
            pred = self.model.predict(seq_arr, verbose=0)
            return float(np.clip(pred[0][0], 0.0, 1.0))

        # Fallback heuristic if weights are not yet loaded
        # Heuristic estimates congestion from total_requests or recent occupancies in sequence
        last_timestep = seq_arr[0, -1]
        approx_congestion = 0.25
        if len(last_timestep) >= 14:
            # Approximate from total requests and occupancy
            total_req = last_timestep[13] if len(last_timestep) > 13 else 5
            approx_congestion = min(1.0, max(0.05, total_req / 40.0))
        return float(round(approx_congestion, 4))

    def evaluate(self, X_test, y_test):
        """Evaluates model performance on test set."""
        if self.model is None:
            raise RuntimeError("Model has not been built or loaded.")
        loss, mae = self.model.evaluate(X_test, y_test, verbose=0)
        return {"loss_mse": float(loss), "mae": float(mae)}

    def save(self, path):
        """Saves model weights or full model."""
        if self.model is None:
            raise RuntimeError("No model to save.")
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        if path.endswith(".h5") or path.endswith(".keras"):
            self.model.save(path)
        else:
            self.model.save_weights(path)

    def load(self, path):
        """Loads model or weights from path."""
        if not TF_AVAILABLE:
            print(f"Warning: TensorFlow not found. Using fallback predictor.")
            return False

        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")

        try:
            if keras_load_model is not None:
                self.model = keras_load_model(path)
        except Exception:
            # Try building architecture then loading weights
            self.build_model()
            if self.model is not None:
                self.model.load_weights(path)
        return True

