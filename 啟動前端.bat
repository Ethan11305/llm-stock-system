@echo off
title LLM Frontend

cd /d "%~dp0"

if exist "venv\Scripts\python.exe" (
    set PYTHON=venv\Scripts\python.exe
    set PIP=venv\Scripts\pip.exe
) else (
    set PYTHON=python
    set PIP=pip
)

echo Installing dependencies...
%PIP% install -q streamlit requests yfinance plotly

echo.
echo Frontend: http://localhost:8501
echo Make sure the backend is running on port 8000.
echo Press Ctrl+C to stop.
echo.

%PYTHON% -m streamlit run frontend/app.py --server.port 8501

pause
