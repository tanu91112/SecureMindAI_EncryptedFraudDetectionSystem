#!/bin/bash
# SecureMindAI - Quick Start Script for Linux/Mac
# CyborgDB Hackathon 2025

echo "========================================"
echo "SecureMindAI Fraud Detection System"
echo "CyborgDB Hackathon 2025"
echo "========================================"
echo ""

echo "Checking Python installation..."
python3 --version
if [ $? -ne 0 ]; then
    echo "ERROR: Python is not installed or not in PATH"
    exit 1
fi
echo ""

echo "Step 1: Installing dependencies..."
pip3 install -r requirements.txt
echo ""

echo "Step 2: Generating transaction dataset..."
cd data
python3 generate_transactions.py
cd ..
echo ""

echo "Step 3: Training fraud detection model..."
python3 fraud_model.py
echo ""

echo "Step 4: Running system tests..."
python3 test_system.py
echo ""

echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "To launch the dashboard, run:"
echo "  streamlit run app.py"
echo ""
read -p "Press Enter to launch now..."

streamlit run app.py
