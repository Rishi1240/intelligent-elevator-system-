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

if exist .venv\Scripts\python.exe (
    set PY_CMD=.venv\Scripts\python.exe
) else (
    set PY_CMD=python
)

if "%choice%"=="1" (
    %PY_CMD% complete_elevator_system.py
)
if "%choice%"=="2" (
    %PY_CMD% backend\app.py
)
if "%choice%"=="3" (
    %PY_CMD% generate_training_data_future.py
)
if "%choice%"=="4" (
    %PY_CMD% backend\train_lstm.py
)
if "%choice%"=="5" (
    cd frontend && npm install && npm run dev
)
pause
