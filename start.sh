#!/bin/bash

# AIH-OCRmyPDF Start Script (macOS/Linux)

cd "$(dirname "$0")"

# Ensure Homebrew paths are available (macOS)
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

echo ""
echo "=========================================="
echo "  AIH-OCRmyPDF Starting..."
echo "  Batch PDF OCR Pipeline"
echo "=========================================="
echo ""

if [ ! -f ".venv/bin/activate" ]; then
    echo "[Error] Virtual environment not found"
    echo "Please run ./install.sh first"
    exit 1
fi

source .venv/bin/activate

export PYTHONWARNINGS=ignore

echo "Web UI will choose an available port starting from 8501."
echo "Your default browser should open automatically."
echo "If 8501 is unavailable or reserved, the app will try the next port."
echo "Press Ctrl+C to stop"
echo ""

python -m ocrmypdf.webui_launcher
