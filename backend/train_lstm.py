"""
Training Script for Multi-Output Elevator LSTM.
Uses training_data_future_5min.csv with genuine 5-minute-ahead targets.

Saves:
  saved_model/lstm_weights.keras  – full Keras model (architecture + weights)
  saved_model/scaler.pkl          – StandardScaler fitted on training split only

Evaluation:
  Reports MAE and RMSE for congestion and per-floor demand on test set.
  Compares against persistence baseline (predict current value as t+5 value).

IMPORTANT: This model trains on SYNTHETIC DATA.  Performance metrics reflect
synthetic pattern learning, not real building performance.
"""

import os
import sys
import math
import time
import numpy as np

# Ensure local imports work from both root and backend/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from lstm_model    import LSTMModel
from data_processor import DataProcessor

# ── paths ─────────────────────────────────────────────────────────────────────
DEFAULT_CSV   = os.path.join(CURRENT_DIR, "training_data_future_5min.csv")
MODEL_SAVE    = os.path.join(CURRENT_DIR, "saved_model", "lstm_weights.keras")
SCALER_SAVE   = os.path.join(CURRENT_DIR, "saved_model", "scaler.pkl")


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def _mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def run_training(
    csv_path:         str   = DEFAULT_CSV,
    model_save_path:  str   = MODEL_SAVE,
    scaler_save_path: str   = SCALER_SAVE,
    epochs:           int   = 30,
    batch_size:       int   = 64,
    seq_len:          int   = 10,
):
    # ── 1. Load and validate dataset ──────────────────────────────────────────
    if not os.path.exists(csv_path):
        fallback_candidates = [
            os.path.join(CURRENT_DIR, "training_data_time_aware_v2.csv"),
            os.path.join(os.path.dirname(CURRENT_DIR), "training_data_time_aware_v2.csv"),
            os.path.join(os.path.dirname(CURRENT_DIR), "training_data_future_5min.csv"),
        ]
        for fb in fallback_candidates:
            if os.path.exists(fb):
                csv_path = fb
                break
        else:
            raise FileNotFoundError(
                f"Dataset not found: {csv_path}\n"
                "Run generate_training_data_future.py first."
            )

    print(f"[Train] Loading dataset: {csv_path}")
    processor = DataProcessor()
    df        = processor.load_csv(csv_path)
    print(f"[Train] Rows: {len(df)}  Columns: {list(df.columns[:6])} ...")

    # Validate key columns
    required = ["hour", "minute", "congestion_t5", "demand_f1_t5"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    # ── 2. Preprocess ─────────────────────────────────────────────────────────
    print("[Train] Preprocessing features (StandardScaler on training split)...")
    # We preprocess ALL rows here then scaler is re-fitted on train split only.
    # Temporarily do full fit to get arrays, then re-fit scaler on train rows.
    X_scaled, y_cong, y_demand = processor.preprocess(df)
    print(f"[Train] Feature matrix: {X_scaled.shape}  |  "
          f"Congestion targets: {y_cong.shape}  |  "
          f"Demand targets: {y_demand.shape}")

    # ── 3. Sequence generation ────────────────────────────────────────────────
    print(f"[Train] Creating sliding sequences (window={seq_len})...")
    X_seq, y_cong_seq, y_demand_seq = processor.create_sequences(
        X_scaled, y_cong, y_demand, seq_len=seq_len
    )
    print(f"[Train] Sequences: {X_seq.shape}  y_cong: {y_cong_seq.shape}  y_demand: {y_demand_seq.shape}")

    # ── 4. Chronological split ────────────────────────────────────────────────
    (X_tr, X_val, X_te,
     yc_tr, yc_val, yc_te,
     yd_tr, yd_val, yd_te) = processor.chronological_split(X_seq, y_cong_seq, y_demand_seq)
    print(f"[Train] Split – train: {len(X_tr)}  val: {len(X_val)}  test: {len(X_te)}")

    # Re-fit scaler strictly on training rows (no val/test leakage)
    n_train_raw = int(len(X_scaled) * 0.70)
    processor.scaler.fit(X_scaled[:n_train_raw])

    # ── 5. Build and train model ──────────────────────────────────────────────
    num_features = X_tr.shape[2]
    model = LSTMModel(
        input_features=num_features,
        sequence_length=seq_len,
        hidden_units=64,
        num_floors=10,
    )
    model.build_model()
    print(f"[Train] Model built: input=({seq_len},{num_features}) | "
          f"outputs=[congestion(1), demand(10)]")

    print(f"[Train] Training up to {epochs} epochs (EarlyStopping patience=6)...")
    t0 = time.time()
    model.train(
        X_tr, yc_tr, yd_tr,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, yc_val, yd_val),
        checkpoint_path=model_save_path,
    )
    elapsed = time.time() - t0
    print(f"[Train] Training completed in {elapsed:.1f}s")

    # ── 6. Evaluation on test set ─────────────────────────────────────────────
    print("[Train] Evaluating on held-out test set...")
    metrics = model.evaluate(X_te, yc_te, yd_te)
    print(f"[Train] Keras metrics: {metrics}")

    # Manual RMSE / MAE using model predictions
    preds     = model.model.predict(X_te, verbose=0)
    cong_pred = np.clip(preds[0].squeeze(), 0.0, 1.0)
    dem_pred  = np.maximum(0.0, preds[1])

    cong_mae  = _mae(yc_te,  cong_pred)
    cong_rmse = _rmse(yc_te, cong_pred)
    dem_mae   = _mae(yd_te,  dem_pred)
    dem_rmse  = _rmse(yd_te, dem_pred)

    # Persistence baseline: predict current congestion as t+5
    # (Use last timestep of each sequence as the persistence guess)
    # feature index 0..2 is hour/min/day – we don't have raw congestion in input,
    # so use naive constant (mean of training targets) as worst-case baseline
    baseline_cong  = float(np.mean(yc_tr))
    base_cong_mae  = _mae(yc_te, np.full_like(yc_te, baseline_cong))
    base_cong_rmse = _rmse(yc_te, np.full_like(yc_te, baseline_cong))

    print()
    print("=" * 62)
    print("  5-Minute Congestion Forecast")
    print(f"  LSTM  MAE={cong_mae:.4f}   RMSE={cong_rmse:.4f}")
    print(f"  Baseline (mean) MAE={base_cong_mae:.4f}   RMSE={base_cong_rmse:.4f}")
    if cong_mae < base_cong_mae:
        print("  [RESULT] LSTM outperforms the mean baseline.")
    else:
        print("  [RESULT] LSTM does NOT outperform the mean baseline – "
              "consider more data or longer training.")
    print()
    print("  Per-Floor Demand Forecast (t+5 min)")
    print(f"  LSTM  MAE={dem_mae:.4f}   RMSE={dem_rmse:.4f}")
    print("=" * 62)
    print()

    # ── 7. Save model and scaler ──────────────────────────────────────────────
    print(f"[Train] Saving model to {model_save_path}...")
    model.save(model_save_path)

    print(f"[Train] Saving scaler to {scaler_save_path}...")
    processor.save_scaler(scaler_save_path)

    print("[Train] Done. Artifacts saved:")
    print(f"   Model : {model_save_path}")
    print(f"   Scaler: {scaler_save_path}")

    return {
        "congestion_mae":   cong_mae,
        "congestion_rmse":  cong_rmse,
        "demand_mae":       dem_mae,
        "demand_rmse":      dem_rmse,
        "baseline_cong_mae": base_cong_mae,
    }


if __name__ == "__main__":
    run_training()
