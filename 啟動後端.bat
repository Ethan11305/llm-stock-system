@echo off
title LLM Backend

cd /d "%~dp0"

if exist "venv\Scripts\python.exe" (
    set PYTHON=venv\Scripts\python.exe
    set PIP=venv\Scripts\pip.exe
) else (
    set PYTHON=python
    set PIP=pip
)

echo Installing package...
%PIP% install -q -e .

echo.
echo Backend: http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo Health:   http://localhost:8000/api/health
echo Press Ctrl+C to stop.
echo.

%PYTHON% -m uvicorn llm_stock_system.api.app:create_app --factory --reload

pause
