@echo off
setlocal
cd /d "%~dp0.."

if exist ".venv\Scripts\python.exe" (
    set "MINDGATE_PYTHON=.venv\Scripts\python.exe"
) else if exist "venv\Scripts\python.exe" (
    set "MINDGATE_PYTHON=venv\Scripts\python.exe"
) else (
    echo MindGate Python environment was not found.
    echo Follow the installation steps in README.md, then run this launcher again.
    exit /b 1
)

"%MINDGATE_PYTHON%" -m uvicorn ui.app_server:app --host 127.0.0.1 --port 8000
