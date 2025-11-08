@echo off
REM SecureMindAI - Quick Start Script for Windows
REM CyborgDB Hackathon 2025

echo ========================================
echo SecureMindAI Fraud Detection System
echo CyborgDB Hackathon 2025
echo ========================================
echo.

echo Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)
echo.

echo Step 1: Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo WARNING: Some dependencies may have failed to install
    echo The system will use built-in alternatives
)
echo.

echo Step 2: Generating transaction dataset...
cd data
python generate_transactions.py
cd ..
echo.

echo Step 3: Training fraud detection model...
python fraud_model.py
echo.

echo Step 4: Running system tests...
python test_system.py
echo.

echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo To launch the dashboard, run:
echo   streamlit run app.py
echo.
echo Or press any key to launch now...
pause

streamlit run app.py
