"""
Training Script for Elevator Traffic Congestion LSTM Model.
Loads CSV dataset, generates sequences, trains LSTM neural net, evaluates performance,
and persists trained weights and scaler.
"""

import os
import sys

# Ensure local imports work whether executed from root or backend/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from lstm_model import LSTMModel
from data_processor import DataProcessor

def run_training(csv_path="training_data.csv",
                 model_save_path="saved_model/lstm_weights.h5",
                 scaler_save_path="saved_model/scaler.pkl",
                 epochs=20,
                 batch_size=64,
                 seq_len=10):
    
    # Check possible paths for training_data.csv
    if not os.path.exists(csv_path):
        alt_path = os.path.join(CURRENT_DIR, csv_path)
        if os.path.exists(alt_path):
            csv_path = alt_path
        else:
            parent_path = os.path.join(CURRENT_DIR, "..", csv_path)
            if os.path.exists(parent_path):
                csv_path = parent_path

    print(f"Loading training data from {csv_path}...")
    processor = DataProcessor()
    df = processor.load_csv(csv_path)
    print(f"Loaded {len(df)} records. Preprocessing features...")

    X_scaled, y = processor.preprocess(df)
    print(f"Creating sliding sequences with window length {seq_len}...")
    X_seq, y_seq = processor.create_sequences(X_scaled, y, seq_len=seq_len)
    print(f"Generated {len(X_seq)} sequences of shape {X_seq.shape[1:]}.")

    print("Splitting into train (80%) and test (20%) sets...")
    X_train, X_test, y_train, y_test = processor.train_test_split_data(X_seq, y_seq, split=0.8)

    print("Initializing LSTM model...")
    num_features = X_train.shape[2]
    model = LSTMModel(input_features=num_features, sequence_length=seq_len, hidden_units=64)
    model.build_model()

    print(f"Training LSTM model for up to {epochs} epochs (batch_size={batch_size})...")
    model.train(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        checkpoint_path=model_save_path
    )

    print("Evaluating trained model on test set...")
    metrics = model.evaluate(X_test, y_test)
    mse = metrics["loss_mse"]
    mae = metrics["mae"]

    print(f"Test Loss (MSE): {mse:.4f}")
    print(f"Test MAE: {mae:.4f}")

    # Ensure model and scaler are saved
    print(f"Saving model to {model_save_path}...")
    model.save(model_save_path)

    print(f"Saving feature scaler to {scaler_save_path}...")
    processor.save_scaler(scaler_save_path)

    print(f"Training complete. MAE: {mae:.2f}")

if __name__ == "__main__":
    # If run from inside backend/, adjust relative paths
    csv_file = "training_data.csv"
    if not os.path.exists(csv_file) and os.path.exists(os.path.join(CURRENT_DIR, "training_data.csv")):
        csv_file = os.path.join(CURRENT_DIR, "training_data.csv")
    elif not os.path.exists(csv_file) and os.path.exists(os.path.join(CURRENT_DIR, "..", "training_data.csv")):
        csv_file = os.path.join(CURRENT_DIR, "..", "training_data.csv")

    model_target = os.path.join(CURRENT_DIR, "saved_model", "lstm_weights.h5")
    scaler_target = os.path.join(CURRENT_DIR, "saved_model", "scaler.pkl")

    run_training(csv_path=csv_file, model_save_path=model_target, scaler_save_path=scaler_target)
