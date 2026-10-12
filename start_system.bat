@echo off
echo ===================================================
echo     AI ELEVATOR OPTIMIZATION SYSTEM LAUNCHER
echo ===================================================
echo.
echo 1. Launch Interactive Python Simulator (CLI)
echo 2. Launch Flask Backend API Server (Port 5000)
echo 3. Generate 50,000 Rows Training Dataset CSV
echo 4. Train LSTM Neural Network
echo 5. Launch React Frontend Web Dashboard
echo.
set /p choice="Select an option (1-5): "

if "%choice%"=="1" (
    python complete_elevator_system.py
)
if "%choice%"=="2" (
    python backend\app.py
)
if "%choice%"=="3" (
    python generate_training_data.py
)
if "%choice%"=="4" (
    cd backend && python train_lstm.py
)
if "%choice%"=="5" (
    cd frontend && npm install && npm run dev
)
pause
