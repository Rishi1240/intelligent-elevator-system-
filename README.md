# AI Elevator Optimization System (Q-Learning + LSTM)

An intelligent multi-car elevator dispatching and traffic forecasting system combining **Tabular Q-Learning** (reinforcement learning with state bucketing) and **Deep LSTM Neural Networks** (time-series passenger demand and congestion forecasting).

---

## 🏛️ System Architecture

1. **LSTM Congestion Predictor (`backend/lstm_model.py`)**:
   - Sequential Deep Recurrent Network: `LSTM(64) -> Dropout(0.2) -> LSTM(32) -> Dropout(0.2) -> Dense(16) -> Dense(1, sigmoid)`
   - Predicts building traffic congestion level $\in [0.0, 1.0]$ across temporal sliding windows (10 steps $\times$ 20 features).
   - Accurately captures morning lobby surges, lunch rushes, evening exit waves, idle nights, and random traffic surges.

2. **Tabular Q-Learning Dispatch Agent (`backend/q_learning_agent.py`)**:
   - Pure Q-table architecture without black-box neural networks for RL.
   - Discrete state compression via bucketing: `(elev1_pos, elev2_pos, elev3_pos, pending_requests_bucket, congestion_bucket)`.
   - Discrete action space: 30 actions `(elevator_id: 0..2, target_floor: 1..10)`.
   - Bellman update with $\epsilon$-greedy exploration decay and multi-objective reward formulation:
     $$R = -(\alpha \cdot \text{wait} + \beta \cdot \text{movement} + \gamma \cdot \text{unused\_capacity}) + \text{service bonuses}$$

3. **Building Simulator Environment (`backend/elevator_env.py`)**:
   - 10 floors, 3 elevators, capacity 8 passengers per elevator.
   - Simulates physical movement, passenger boarding/alighting, floor demand tracking, and wait time dynamics.

4. **Flask REST API (`backend/app.py`)**:
   - Connects simulator, LSTM, and Q-learning agent on port `5000`.
   - Endpoints: `/api/start`, `/api/reset`, `/api/state`, `/api/request`, `/api/step`, `/api/stats`, `/api/predict`.

5. **React Dashboard (`frontend/src/App.jsx`)**:
   - Real-time dark UI with moving car indicators across all 3 shafts.
   - Dynamic Congestion Meter (Green <30%, Yellow 30-69%, Red $\ge$ 70%).
   - Passenger dispatch panel with presets.
   - Live telemetry and event logging.

---

## 🚀 Quickstart Guide

### 1. Dataset Generation
The 50,000-row training dataset is generated at `training_data.csv` and `backend/training_data.csv`:
```bash
python generate_training_data.py
```

### 2. LSTM Model Training
To train the LSTM neural network on the dataset and save `lstm_weights.h5` and `scaler.pkl`:
```bash
cd backend
pip install -r requirements.txt
python train_lstm.py
```

### 3. Start Flask Backend
```bash
cd backend
python app.py
```
Backend runs on `http://localhost:5000`.

### 4. Start React Frontend
```bash
cd frontend
npm install
npm run dev
```
Frontend runs on `http://localhost:3000`.
