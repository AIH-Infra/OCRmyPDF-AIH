#!/bin/bash

# OCRmyPDF-AIH Start Script (macOS/Linux)

cd "$(dirname "$0")"

# Ensure Homebrew paths are available (macOS)
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

echo ""
echo "=========================================="
echo "  OCRmyPDF-AIH Starting..."
echo "  Batch PDF OCR Pipeline"
echo "=========================================="
echo ""

# Release port 6106
if command -v lsof &> /dev/null; then
    PID=$(lsof -ti:6106 2>/dev/null)
    if [ -n "$PID" ]; then
        kill -9 $PID 2>/dev/null
    fi
fi

# Check venv
if [ ! -f ".venv/bin/activate" ]; then
    echo "[Error] Virtual environment not found"
    echo "Please run ./install.sh first"
    exit 1
fi

source .venv/bin/activate

echo "URL: http://localhost:6106"
echo "Press Ctrl+C to stop"
echo ""

python -m streamlit run misc/_webservice_pro.py --server.port 6106
