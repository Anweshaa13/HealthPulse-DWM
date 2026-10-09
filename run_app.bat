@echo off
title HealthPulse DWM Healthcare Analytics & Predictive Mining System
echo ======================================================================
echo Launching HealthPulse Healthcare Data Warehouse and Mining System...
echo ======================================================================
echo.

:: Check if virtual environment or python exists
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not found in PATH. Please install Python 3.9+ or add it to PATH.
    pause
    exit /b
)

:: Run streamlit app
echo Starting Streamlit Web Application on http://localhost:8501 ...
python -m streamlit run app.py

pause
