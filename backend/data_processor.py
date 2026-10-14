"""
Data Preprocessing Module for Multi-Output Elevator LSTM.
Handles:
  - CSV loading of training_data_future_5min.csv
  - Feature scaling (StandardScaler fitted on TRAIN split only)
  - Sliding-window sequence generation for LSTM (seq_len=10 default)
  - Separate label arrays for congestion_t5 and demand_f*_t5
  - Scaler persistence

Feature columns (20, used as LSTM input, same as original):
  hour, minute, day_of_week,
  floor_req_f1..f10, total_requests,
  elev1_floor, elev2_floor, elev3_floor,
  elev1_occ, elev2_occ, elev3_occ

Target columns (NOT fed to LSTM as input):
  congestion_t5                       – overall building congestion at t+5 min
  demand_f1_t5 .. demand_f10_t5       – per-floor demand at t+5 min
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ── canonical column names ───────────────────────────────────────────────────
FEATURE_COLUMNS = [
    "hour", "minute", "day_of_week",
    "floor_req_f1", "floor_req_f2", "floor_req_f3", "floor_req_f4", "floor_req_f5",
    "floor_req_f6", "floor_req_f7", "floor_req_f8", "floor_req_f9", "floor_req_f10",
    "total_requests",
    "elev1_floor", "elev2_floor", "elev3_floor",
    "elev1_occ",   "elev2_occ",   "elev3_occ",
]

CONGESTION_TARGET = "congestion_t5"   # 5-min-ahead congestion

DEMAND_TARGET_COLS = [
    "demand_f1_t5",  "demand_f2_t5",  "demand_f3_t5",  "demand_f4_t5",
    "demand_f5_t5",  "demand_f6_t5",  "demand_f7_t5",  "demand_f8_t5",
    "demand_f9_t5",  "demand_f10_t5",
]

# Legacy target name kept for backward compatibility
LEGACY_TARGET = "congestion_level"


class DataProcessor:
    def __init__(self, feature_columns=None):
        self.feature_columns = list(feature_columns or FEATURE_COLUMNS)
        self.scaler    = StandardScaler()
        self.is_fitted = False

    # ── accessors ────────────────────────────────────────────────────────────

    def get_feature_columns(self) -> list:
        return list(self.feature_columns)

    # ── I/O ──────────────────────────────────────────────────────────────────

    def load_csv(self, path: str) -> pd.DataFrame:
        """Loads dataset CSV using pandas."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found at {path}")
        df = pd.read_csv(path)
        return df

    # ── preprocessing ────────────────────────────────────────────────────────

    def preprocess(self, df: pd.DataFrame):
        """
        Fits StandardScaler on FEATURE_COLUMNS and returns scaled X and targets y.

        Supports both the new multi-target schema (training_data_future_5min.csv)
        and the legacy single-target schema (training_data.csv).

        Returns
        -------
        X_scaled : np.ndarray, shape (N, num_features) – scaled input features
        y_cong   : np.ndarray, shape (N,)              – congestion target
        y_demand : np.ndarray or None                  – shape (N, 10) if available
        """
        # Use intersection in case schema differs slightly
        avail = [c for c in self.feature_columns if c in df.columns]
        if len(avail) != len(self.feature_columns):
            self.feature_columns = avail

        X_raw = df[self.feature_columns].values.astype(np.float32)
        X_scaled = self.scaler.fit_transform(X_raw).astype(np.float32)
        self.is_fitted = True

        # ── congestion target ────────────────────────────────────────────────
        if CONGESTION_TARGET in df.columns:
            y_cong = df[CONGESTION_TARGET].values.astype(np.float32)
        elif LEGACY_TARGET in df.columns:
            # old dataset – use present-time congestion as fallback target
            y_cong = df[LEGACY_TARGET].values.astype(np.float32)
        else:
            y_cong = None

        # ── per-floor demand target ──────────────────────────────────────────
        avail_demand = [c for c in DEMAND_TARGET_COLS if c in df.columns]
        if len(avail_demand) == len(DEMAND_TARGET_COLS):
            y_demand = df[DEMAND_TARGET_COLS].values.astype(np.float32)
        else:
            y_demand = None

        return X_scaled, y_cong, y_demand

    # ── sequence generation ──────────────────────────────────────────────────

    def create_sequences(self, X, y_cong=None, y_demand=None, seq_len: int = 10):
        """
        Creates sliding-window sequences for LSTM input.

        Parameters
        ----------
        X        : (N, features) scaled input array
        y_cong   : (N,)   congestion targets or None
        y_demand : (N,10) demand targets or None
        seq_len  : sliding window length (default 10)

        Returns
        -------
        X_seq       : (M, seq_len, features)
        y_cong_seq  : (M,)   or None
        y_demand_seq: (M,10) or None

        where M = N - seq_len + 1
        Note: target at index i is aligned to the LAST timestep of the window.
        """
        M = len(X) - seq_len + 1
        if M <= 0:
            raise ValueError(f"Data length ({len(X)}) < seq_len ({seq_len})")

        num_feat = X.shape[1]
        X_seq = np.zeros((M, seq_len, num_feat), dtype=np.float32)
        for i in range(M):
            X_seq[i] = X[i: i + seq_len]

        y_cong_seq   = None
        y_demand_seq = None

        if y_cong is not None:
            y_cong_seq = np.zeros(M, dtype=np.float32)
            for i in range(M):
                y_cong_seq[i] = y_cong[i + seq_len - 1]

        if y_demand is not None:
            y_demand_seq = np.zeros((M, y_demand.shape[1]), dtype=np.float32)
            for i in range(M):
                y_demand_seq[i] = y_demand[i + seq_len - 1]

        return X_seq, y_cong_seq, y_demand_seq

    # ── train/val/test split ─────────────────────────────────────────────────

    def chronological_split(self, X_seq, y_cong_seq, y_demand_seq,
                             train_frac: float = 0.70,
                             val_frac:   float = 0.15):
        """
        Chronological 70/15/15 split (preserves temporal ordering).
        Returns six arrays: X_train, X_val, X_test, y_c_train, y_c_val, y_c_test
                             and optionally demand arrays.
        """
        N = len(X_seq)
        t1 = int(N * train_frac)
        t2 = int(N * (train_frac + val_frac))

        X_train, X_val, X_test = X_seq[:t1], X_seq[t1:t2], X_seq[t2:]

        def _split(arr):
            if arr is None:
                return None, None, None
            return arr[:t1], arr[t1:t2], arr[t2:]

        yc_tr, yc_v, yc_te = _split(y_cong_seq)
        yd_tr, yd_v, yd_te = _split(y_demand_seq)

        return (X_train, X_val, X_test,
                yc_tr,  yc_v,  yc_te,
                yd_tr,  yd_v,  yd_te)

    # ── backward-compatible helper kept for legacy callers ──────────────────

    def train_test_split_data(self, X_seq, y_seq, split: float = 0.8):
        """Legacy 80/20 split for old training scripts or tests."""
        t = int(len(X_seq) * split)
        return X_seq[:t], X_seq[t:], y_seq[:t], y_seq[t:]

    # ── live inference scaling ───────────────────────────────────────────────

    def scale_single_sequence(self, seq):
        """
        Scales a single sequence of shape (seq_len, features) or (1, seq_len, features)
        using the fitted scaler.
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler is not fitted yet. Fit or load scaler first.")

        seq_arr = np.array(seq, dtype=np.float32)
        if seq_arr.ndim == 3:
            orig = seq_arr.shape
            scaled = self.scaler.transform(seq_arr.reshape(-1, orig[-1]))
            return scaled.reshape(orig)
        elif seq_arr.ndim == 2:
            return self.scaler.transform(seq_arr).astype(np.float32)
        else:
            raise ValueError(f"Expected 2D or 3D sequence, got ndim={seq_arr.ndim}")

    # ── scaler persistence ───────────────────────────────────────────────────

    def save_scaler(self, path: str):
        """Saves fitted StandardScaler to disk using joblib."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        joblib.dump(self.scaler, path)

    def load_scaler(self, path: str):
        """Loads fitted StandardScaler from disk using joblib."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Scaler file not found at {path}")
        self.scaler    = joblib.load(path)
        self.is_fitted = True
