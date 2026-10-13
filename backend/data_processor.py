"""
Data Preprocessing Module for Elevator Congestion Prediction.
Handles CSV loading, feature scaling, sequence generation for LSTM, and scaler persistence.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, feature_columns=None):
        if feature_columns is None:
            self.feature_columns = [
                "hour", "minute", "day_of_week",
                "floor_req_f1", "floor_req_f2", "floor_req_f3", "floor_req_f4", "floor_req_f5",
                "floor_req_f6", "floor_req_f7", "floor_req_f8", "floor_req_f9", "floor_req_f10",
                "total_requests",
                "elev1_floor", "elev2_floor", "elev3_floor",
                "elev1_occ", "elev2_occ", "elev3_occ"
            ]
        else:
            self.feature_columns = feature_columns
        self.target_column = "congestion_level"
        self.scaler = StandardScaler()
        self.is_fitted = False

    def get_feature_columns(self):
        """Returns list of feature column names."""
        return list(self.feature_columns)

    def load_csv(self, path):
        """Loads dataset CSV using pandas."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found at {path}")
        df = pd.read_csv(path)
        return df

    def preprocess(self, df):
        """
        Fits StandardScaler on feature columns and returns scaled X and target y.
        """
        # Ensure required columns exist
        available_features = [col for col in self.feature_columns if col in df.columns]
        if len(available_features) != len(self.feature_columns):
            # Fall back to intersection if slight schema variation
            self.feature_columns = available_features

        X_raw = df[self.feature_columns].values
        y = df[self.target_column].values if self.target_column in df.columns else None

        X_scaled = self.scaler.fit_transform(X_raw)
        self.is_fitted = True
        return X_scaled, y

    def create_sequences(self, X, y=None, seq_len=10):
        """
        Creates sliding window sequences for LSTM input.
        Returns 3D array: shape (num_samples, seq_len, num_features).
        """
        num_samples = len(X) - seq_len + 1
        if num_samples <= 0:
            raise ValueError(f"Data length ({len(X)}) is smaller than seq_len ({seq_len})")

        num_features = X.shape[1]
        X_seq = np.zeros((num_samples, seq_len, num_features), dtype=np.float32)
        y_seq = np.zeros(num_samples, dtype=np.float32) if y is not None else None

        for i in range(num_samples):
            X_seq[i] = X[i : i + seq_len]
            if y is not None and y_seq is not None:
                # Target is congestion_level at the end of the sequence window
                y_seq[i] = y[i + seq_len - 1]

        if y is not None:
            return X_seq, y_seq
        return X_seq

    def train_test_split_data(self, X_seq, y_seq, split=0.8):
        """Splits sequential data into train and test sets (time-series order preserved)."""
        train_size = int(len(X_seq) * split)
        X_train, X_test = X_seq[:train_size], X_seq[train_size:]
        y_train, y_test = y_seq[:train_size], y_seq[train_size:]
        return X_train, X_test, y_train, y_test

    def scale_single_sequence(self, seq):
        """
        Scales a single sequence of shape (seq_len, features) or (1, seq_len, features)
        using the fitted scaler.
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler is not fitted yet. Fit or load scaler first.")

        seq_arr = np.array(seq)
        if seq_arr.ndim == 3:
            orig_shape = seq_arr.shape
            reshaped = seq_arr.reshape(-1, orig_shape[-1])
            scaled = np.asarray(self.scaler.transform(reshaped))
            return scaled.reshape(orig_shape)
        elif seq_arr.ndim == 2:
            return self.scaler.transform(seq_arr)
        else:
            raise ValueError(f"Expected sequence of 2D or 3D, got ndim={seq_arr.ndim}")


    def save_scaler(self, path):
        """Saves fitted StandardScaler to disk using joblib."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        joblib.dump(self.scaler, path)

    def load_scaler(self, path):
        """Loads fitted StandardScaler from disk using joblib."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Scaler file not found at {path}")
        self.scaler = joblib.load(path)
        self.is_fitted = True
